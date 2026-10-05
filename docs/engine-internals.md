# Engine internals

How Prosimos runs a case, and what that means for messaging. For developers changing the engine
(`prosimos/simulation_engine.py`, `prosimos/control_flow_manager.py`) or the messaging code. The
user-facing behaviour is described in [messaging.md](messaging.md) and
[orchestrator.md](orchestrator.md).

## How a case moves

A case is a `ProcessState`: a token count per sequence flow, plus a bitmask of the flows that hold a
token (`state_mask`). `BPMNGraph.update_process_state` moves tokens forward from an element that has
just completed:

- **passed straight through**, in the same instant: gateways, intermediate throw events and end
  events (the `to_execute` loop);
- **queued**: tasks and intermediate catch events become an `EnabledEvent` in the engine's
  time-ordered queue (`_find_next`).

An element with a token on its incoming flow but not yet enabled (e.g. a parallel join missing a
branch) simply holds the token. When a join fires, `_check_and_update_enabling_time` gives the next
element the latest of the branches' times.

`update_process_state` returns two values, the enabled tasks and the times elements were reached, but
when the element it is called for isn't enabled it returns a bare empty list, and the callers, which
unpack two values, crash (`ValueError: not enough values to unpack`). It was behind the race-timer
crash (a timer firing for a case that had left the race), now prevented before the call. Calling it
only for enabled elements avoids it.

In a process with an arrival schedule, all cases are created before the first event
(`generate_all_arrival_events`): case ids are `0 .. total_cases - 1`, case attributes are drawn up
front, and each case is moved from its start event at once, so its first task is queued for its
arrival time.

In a process started by messages there are no planned cases. `SimBPMEnv.create_case(now)` creates
each case when a start message is claimed: it takes the next consecutive case id (the log's
`trace_list` is indexed by case id), draws the case attributes and adds them to both
`CasePrioritisation.all_case_attributes` and `bpmn_graph.all_attributes`, and then does what a planned
arrival does (`_update_initial_event_info`): `last_datetime`, a new `ProcessState`, a `Trace`, and the
start event's enabled tasks queued at `now`. Until the first claim the engine has nothing queued, so
`next_event_time()` is `None`.

## Effects are computed ahead of their time

The engine executes a task in one go when it takes it from the queue at its enabled time: it books
the resource, computes the end time, and calls `update_process_state` with that end time. Everything
passed straight through after the task (gateways, throw and end events) is therefore passed during
the step at the task's *enabled* time, while its own time is the task's *end* time:

```
step at 09:16:11 → logs Task A (ends 09:19:21) and Throw at 09:19:21
```

Elements right after a start event are passed even earlier, while arrivals are generated. The log
times are right, but anything that must happen at the real time, such as publishing, can't be done
at that moment.

**Holding.** The graph records each pass of a publishing event (`BPMNGraph.watched_elements`,
`passed_watched_elements`), and `SimBPMEnv` holds it with its due time in a heap (`_held`, ordered by
time, case id, pass order). `next_event_time()` is the earlier of the queue head and the earliest
held item; `step()` releases all items due at that time instead of running an event, and at equal
times held items go first. An engine that still holds items isn't finished, even with an empty
queue. The message is built when it is released, so attribute values are those at the real time.

**Batching.** The same applies to a batched task: a case joins its waiting list
(`BPMNGraph.batch_waiting_processes`) during the step of the task before it, with the time it will
reach the batched task. The list therefore holds cases that reach the task only later, and it is not
in time order. Batching used to treat it as if it were, which went wrong once task durations varied:

- the firing rule and the batch size counted cases that hadn't arrived yet, and `.seconds` turned
  their negative waiting times into large positive ones, so the two disagreed (the "batch size ... 0"
  warning);
- after a batch fired, the first cases in the list were removed, not the ones the batch had taken, so
  one case could run twice (a crash) while another was lost.

This is fixed: the firing rule and the batch size see only the cases that have reached the task, in
the order they reached it (`_reached_batch`), and exactly the cases a batch takes leave the list.
`testing_scripts/test_batching_random_durations.py` covers it.

Three older batching problems, which also happen with fixed durations, were fixed with it:

- a single waiting case between the `ready_wt` boundaries no longer prints a false "batch size ... 0"
  warning: the firing rule now agrees with the batch size that it waits for a second case;
- the last cases of a run are no longer lost when the gap after the last of them is below the low
  boundary: the end of the run now gives them a firing time;
- a lone last case under `large_wt` waits up to the upper boundary instead of firing at the lower one,
  which also makes `test_range_large_wt_rule_correct_log_distances` stable.

Still open, both also on `main`:

- with a `size >= 2` firing rule the batch never fires, so the batched task never runs and those cases
  are stuck;
- with `daily_hour`, a case left alone at the end of a run can be lost.

Some batching tests rewrite `batch-example-with-batch.json` in place (firing rules, arrival
distribution), so the new tests reset the sections they rely on to the committed values.

## Waiting at catch events

A case is moved onto a catch event during the step of the task before it, but the catch event is
queued at the time the case really reaches it. Parking therefore happens where that queued event is
executed (`execute_enabled_event`), not where it is created (`_find_next`): parking there would let
the case wait, and claim a message, before it has arrived.

- **Parking**: the catch event is not completed; its token stays on the incoming flow, and the
  parked `EnabledEvent` is kept in `SimBPMEnv._parked_events`, keyed by (case, event).
- **Resuming** (`deliver`): a new `EnabledEvent` for the catch event is queued at the message's time,
  with `parked_event` pointing to the parked one; when it is executed, the event completes at once
  and the case continues.
- **Stale records**: a case can lose its waiting token without a message, e.g. through a terminate
  end event on another branch. `parked_events()` drops such records, so the case is neither resumed
  nor reported as stalled.
- **Joins after a waiting branch**: an AND join fires at the later of the branches. An inclusive (OR)
  join waits for a parked branch, because its look-ahead (`or_join_pred`) checks the case's tokens
  (`state_mask`), not the event queue, and a parked token stays on its flow.

Before the messaging work, every catch event, whatever its kind, completed after a delay drawn from
`event_distribution` (`execute_event`); catch events not listed under `consume` still do.

**Races.** An event-based gateway with a branch listed under `consume` is a race gateway
(`BPMNGraph.race_gateways`, set by `SimBPMEnv`). There `update_process_state` puts a token on every
branch instead of choosing one, so every branch's catch event is queued at the time the case reaches
the gateway. The branches of one case share a `Race` (`SimBPMEnv._races`, keyed by case and gateway):

- When a timer branch comes off the queue, its delay is drawn, and it is queued again for when it fires,
  as a new `EnabledEvent` with `race` set and `armed_event` pointing to the armed one (so the log row
  spans the wait). Each message branch parks as usual, with `race` set on its parked event.
- The first branch to happen wins (`_win_race`): a timer that comes off the queue with its race not yet
  won, or a message branch when its case is resumed (`_resume`). The other branches' tokens are
  taken off the flows into them, the other parked branches' waiting records are dropped, and a
  canceled timer is skipped when it comes off the queue (its race is already won).
- When a delivered message is accepted by a case waiting in a race, `deliver` looks at all the parked
  branches of that race that accept it (`_first_listed_branch`): the one whose `consume` entry comes
  first in the JSON takes it (`_consume_order`, the entries' positions), with one warning per gateway
  if there are several. Without this, the parked events' order (by time, then case and event id) would
  decide, since a case's branches are all parked at the same time.
- A timer that comes off the queue after its case has left the race without it (no token on the flow
  into it anymore, e.g. after a terminate end event on another branch) is skipped too, and its race is
  dropped; the parked message branches are cleaned up as any stale waiting record.
- A case still waiting in a race when the run ends has one parked event per message branch;
  `stalled_waits` reports it once, at the gateway, with the message types of all of them.

The timer is queued for its firing time, rather than completed when it comes off the queue as other
catch events are, so that a message coming before that time can still cancel it.

**Ties.** A message at exactly the timer's time wins. Across engines, an engine can't know whether a
message for that instant is still coming (another engine may step after it at the same time, ties
being broken by process name), so a race timer is queued one microsecond after its time
(`_race_step`): every message of that instant is offered first. The real firing time travels with it
(`duration_sec` holds the delay, counted from `armed_event`), so the timer's log row ends at the real
time. After a race timer wins, the case continues one microsecond after the timer's time: only the
timer's own log row shows its real time, and what follows it starts a microsecond later. Continuing at
the real time instead would make the engine go back in time, which it must never do.

## Implicit merges differ from the BPMN standard

BPMN lets a modeler draw several arrows into an element without a gateway. The standard calls this
uncontrolled flow: the element fires once per arriving token, without waiting for the others (an AND
merge must be drawn as an explicit parallel gateway). Prosimos follows this for tasks and
intermediate throw events (the parser adds a hidden `xor_join_<id>`), but not for:

- **intermediate catch events**: no hidden gateway, and `is_enabled` requires a token on every
  incoming arrow, so they behave as an AND merge. With an AND split into Task A and Task B and both
  arrows into one timer catch event before Task C, the catch event fires once, after the later
  branch, and Task C runs once per case; the standard gives two firings;
- **end events**: the parser adds a hidden `or_join_<id>`, which waits for every branch that can
  still arrive, so an end event after an AND split is passed once per case instead of twice.

This is left unchanged, since it affects existing models. Message end events and message catch
events that publish or wait must have a single incoming arrow instead (checked at load).

## Model restrictions

- **One start event** per process (`BPMNGraph.starting_event`; the last one parsed wins). A model
  whose start event is started by messages is rejected if it has more than one.
- **One end event** per model (`BPMNGraph.validate_model`).
- **Event-based gateways** without a branch waiting for a message are decided when the case reaches
  them: `get_event_gateway_choice` draws a duration for each following event and takes the shortest,
  breaking ties with `random.choice`. Those with such a branch are races (see "Waiting at catch
  events").

## Random numbers

All of Prosimos's random draws come from the two global generators:

| Source                                                                       | Used for                                                                                                                                    |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Python `random` (`random.shuffle`, `choices`, `choice`, `randint`, `random`) | order of parallel branches, discrete attributes, batch sizes, resource choice, multitasking, ties at event-based gateways                   |
| NumPy global (`np.random.rand`, `numpy.random.choice/uniform`)               | histograms, gateway probabilities, fuzzy calendars                                                                                          |
| SciPy `.rvs()` inside pix-framework `DurationDistribution.generate_sample`   | every duration, arrival gap, continuous attribute; pix-framework calls `.rvs()` without `random_state`, so it uses NumPy's global generator |

A single-process run (`--seed`) seeds these global generators directly.

**One stream per engine.** In a multi-process run every `ProsimosEngine` keeps its own Python and
NumPy generator states. At the start of each of its methods, building it included, it saves the
caller's states and loads its own; at the end it saves its own and puts the caller's back (together
with its warning list, see [orchestrator.md](orchestrator.md)). Nothing in Prosimos or pix-framework
changes: they keep drawing from "the global generators", which during the call are the engine's.
So one engine's draws don't depend on which other engines run beside it, and a run leaves the
caller's generators untouched. On a small example this cost about 36 µs per call.

An engine's states are seeded from the simulation seed and its process name, with a stable hash
(SHA-256 of `"<seed>/<name>"`, not Python's `hash()`, which differs between runs), so two engines of
the same model under different names draw different values. Without a simulation seed, an engine is
seeded from draws of the caller's global generators, so a run is unrepeatable unless the caller
seeded those.
