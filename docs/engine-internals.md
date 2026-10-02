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

All cases are created before the first event (`generate_all_arrival_events`): case ids are
`0 .. total_cases - 1`, case attributes are drawn up front, and each case is moved from its start
event at once, so its first task is queued for its arrival time.

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

## Waiting at catch events

A case is moved onto a catch event during the step of the task before it, but the catch event is
queued at the time the case really reaches it. Parking therefore happens where that queued event is
executed (`execute_enabled_event`), not where it is created (`_find_next`): parking there would let
the case wait, and claim a message, before it has arrived.

- **Parking**: the catch event is not completed; its token stays on the incoming flow, and the
  parked `EnabledEvent` is kept in `SimBPMEnv._waiting`, keyed by (case, event).
- **Resuming** (`deliver`): a new `EnabledEvent` for the catch event is queued at the message's time,
  with `parked_event` pointing to the parked one; when it is executed, the event completes at once
  and the case continues.
- **Stale records**: a case can lose its waiting token without a message, e.g. through a terminate
  end event on another branch. `waiting_cases()` drops such records, so the case is neither resumed
  nor reported as stalled.
- **Joins after a waiting branch**: an AND join fires at the later of the branches. An inclusive (OR)
  join waits for a parked branch, because its look-ahead (`or_join_pred`) checks the case's tokens
  (`state_mask`), not the event queue, and a parked token stays on its flow.

Before the messaging work, every catch event, whatever its kind, completed after a delay drawn from
`event_distribution` (`execute_event`); catch events not listed under `consume` still do.

## Implicit merges differ from the BPMN standard

BPMN lets a modeller draw several arrows into an element without a gateway. The standard calls this
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

- **One start event** per process (`BPMNGraph.starting_event`; the last one parsed wins).
- **One end event** per model (`BPMNGraph.validate_model`).
- **Event-based gateways** are decided when the case reaches them: `get_event_gateway_choice` draws
  a duration for each following event and takes the shortest, breaking ties with `secrets.choice`.
  Nothing waits there.

## Random numbers

Random draws come from four sources:

| Source                                                                           | Used for                                                                                    | Can a per-engine generator control it?                                                                    |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Python `random` (`random.shuffle`, `choices`, `random.randint`, `random.random`) | order of parallel branches, discrete attributes, batch sizes, resource choice, multitasking | only if replaced by a per-engine `random.Random` at ~10 call sites                                        |
| NumPy global (`np.random.rand`, `numpy.random.choice/uniform`)                   | histograms, gateway probabilities, fuzzy calendars                                          | only if replaced at ~7 call sites                                                                         |
| SciPy `.rvs()` inside pix-framework `DurationDistribution.generate_sample`       | every duration, arrival gap, continuous attribute                                           | **no**: pix-framework calls `.rvs()` without `random_state`, so it always uses NumPy's global generator |
| `secrets.choice` in `get_event_gateway_choice`                                   | ties at event-based gateways                                                                | **no**: never seedable; should become an ordinary random call                                             |

`--seed` (and the seed of a multi-process configuration) seeds the global Python and NumPy
generators, which makes runs repeatable apart from the rare `secrets.choice` ties. In a
multi-process run all engines share these generators, so one engine's draws depend on the others.

Giving each engine its own stream needs no library changes: before each call to an engine, load that
engine's saved Python and NumPy generator states, and save them after. Measured on a small example,
an engine's draws were then identical with or without another engine beside it, at about 36 µs per
call, roughly doubling the orchestrator's time per step (~25 µs). This isn't implemented yet.
