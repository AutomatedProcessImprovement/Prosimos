# Process engine notes: waiting cases, new cases, randomness

Groundwork for future work (a case waits at a message event; a message starts a new case). Based on
reading `simulation_engine.py`, `control_flow_manager.py`, `simulation_setup.py` and
`prioritisation.py` at commit `44274d6`, plus small runs where marked *(run)*.

**How a case moves today.** A case is a `ProcessState`: a token count per sequence flow.
`update_process_state` moves tokens forward: gateways and end events are passed straight through, and
every task or intermediate catch event reached becomes an `EnabledEvent` in the engine's time-ordered
queue. An element with a token on its incoming flow but not yet enabled (e.g. a parallel join
missing a branch) simply holds the token; it is not queued.

## 1. Where a case could be parked and resumed

- **Intermediate catch events** are the natural place. The parser already reads
  `messageEventDefinition`, but every catch event (timer, message, signal) is simulated the same way:
  queued at once and completed after a delay drawn from `event_distribution` (`execute_event`).
  The case reaches a catch event during the step of the task before it (see "Effects are computed
  ahead of their time" below), but the catch event is queued at the time the case reaches it and only stepped
  then *(run)*. So parking belongs where that queued event is executed, not in `_find_next`, where
  it is created: parking in `_find_next` would make the case wait before it has reached the event. Parking =
  on that step, leave the case waiting at the event and record `(case, event, time reached)` instead
  of completing it. Resuming = queue an `EnabledEvent` for that event at the message's time and let
  the existing path continue. Implemented this way, see "Waiting" in
  [messaging-model.md](messaging-model.md).
- **Event-based gateways** don't wait either: `get_event_gateway_choice` draws a duration for each
  following event the moment the case reaches the gateway and takes the shortest. A real race between a timer and
  a message needs a new mechanism (wait, first one wins, cancel the others).
- **Sending.** Intermediate throw events (message and none) are parsed and passed straight
  through, like gateways; signal, escalation, compensation and link throws are rejected at load
  (see [messaging-model.md](messaging-model.md)). Before that, a model with a throw event crashed
  with `KeyError` as soon as a case reached it *(run)*. Message end events are parsed, as ordinary
  end events. Send/receive tasks and boundary events are still not parsed.
- **Effects are computed ahead of their time.** A task is executed in one go when it is taken from
  the queue at its enabled time: the resource is booked, the end time computed, and
  `update_process_state` called with that end time. Everything passed straight through after it
  (gateways, throw and end events) is therefore passed during the step at the task's *enabled* time,
  while its own time is the task's *end* time *(run)*:
  `step at 09:16:11 → logs Task A (ends 09:19:21) and Throw at 09:19:21`. The log times are right,
  but a message published there would be early. Solved by holding: the engine holds such zero-time
  effects and releases each at its own time (`next_event_time()` is the earlier of the queue head
  and the earliest held effect; at equal times held effects go first; an engine with held effects
  isn't finished). See "Publishing" in [messaging-model.md](messaging-model.md).
- **Elements right after the start event are passed too early.** All arrivals are generated up
  front (section 2), and `_update_initial_event_info` runs `update_process_state` from the start
  event straight away. So a throw or end event directly after the start (with only gateways in
  between) is passed when arrivals are generated, before the simulation reaches that case's arrival
  time *(run)*. The log is unaffected apart from row order (the times are right), but publishing a
  message there would hand it to the orchestrator too early. Holding (above) covers this too: the
  message is held until the case's arrival time.

### Implicit merges differ from the BPMN standard

BPMN lets a modeller draw several arrows into an element without a gateway. The standard calls
this uncontrolled flow: the element fires once per arriving token, without waiting for the others
(an AND merge must be drawn as an explicit parallel gateway). Prosimos follows this for tasks and
intermediate throw events (the parser adds a hidden `xor_join_<id>`), but not for:

- **Intermediate catch events:** no hidden gateway, and `is_enabled` requires a token on every
  incoming arrow, so they behave as an AND merge. AND split -> Task A, Task B -> both into one timer
  catch event -> Task C: the catch event fired once, after the later branch, and Task C ran once per
  case; the standard gives two firings *(run)*.
- **End events:** the parser adds a hidden `or_join_<id>`, which waits for every branch that can
  still arrive. AND split -> Task A, Task B -> both into one message end event: the end event was
  passed once per case, after the later branch; the standard passes it twice *(run)*.

Not changed, since it affects models outside the messaging work. For messages the ambiguity is
avoided instead: a message end event listed under `publish` or a message catch event listed under
`consume` must have a single incoming arrow, checked when the model is loaded.

## 2. What creating a case mid-run needs

All cases are created by `generate_all_arrival_events` before the first event, using
`total_num_cases`. A mid-run `create_case(time, attributes)` must do what `_update_initial_event_info`
does, plus two things done up front today:

- **Case id:** the next integer. Ids must stay consecutive, because `log_info.trace_list` is indexed
  by position (`trace_list[p_case]`).
- **Per-case setup:** `last_datetime[element][case]` for every element, a new `ProcessState` in
  `all_process_states`, a `Trace`, and queueing the start event's enabled tasks.
- **Case attributes:** drawn for all `total_num_cases` cases when the engine is built
  (`CasePrioritisation`), then copied into `bpmn_graph.all_attributes`. A new case needs its own draw
  (or values from the message) added to both.
- **Start events:** only one per process is supported (`BPMNGraph.starting_event`, last one parsed
  wins), so a process can't have both a normal and a message start yet.
- **`total_cases`:** only counts generated arrivals. A process started only by messages would have 0
  and report "finished" immediately, the same problem as in (1).

Global attributes are updated on a case's first event (`execute_full_process`) and batching and
execution statistics are keyed by case id as they go, so those need no change.

## 3. Parallel branches where only one waits

Works with the existing token logic *(run: split → Task A and a 10-hour message event → join → Task B)*.
Task A's token waits on the join's incoming flow; the join fires only when the other branch arrives,
and Task B is enabled at the later branch's time (19:00, not 10:00), because
`_check_and_update_enabling_time` keeps the latest time per element and case. All branches share one
`ProcessState`, so resuming must use `all_process_states[case]`. Still to check: a *terminate* end
event on the other branch clears every token (the parked one included) and any later message for that
case must be dropped; inclusive (OR) joins use look-ahead (`or_join_pred`) and need a test.

## 4. Can every random draw use a per-engine generator?

Not directly. Draws, by source:

| Source                                                                           | Used for                                                                                    | Seedable by an engine's own generator?                                                                          |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Python `random` (`random.shuffle`, `choices`, `random.randint`, `random.random`) | order of parallel branches, discrete attributes, batch sizes, resource choice, multitasking | only if replaced by a per-engine `random.Random` at ~10 call sites                                              |
| NumPy global (`np.random.rand`, `numpy.random.choice/uniform`)                   | histograms, gateway probabilities, fuzzy calendars                                          | only if replaced at ~7 call sites                                                                               |
| SciPy `.rvs()` inside pix-framework `DurationDistribution.generate_sample`       | every duration, arrival gap, continuous attribute                                           | **no**: pix-framework calls `.rvs()` without `random_state`, so it always uses NumPy's global generator *(run)* |
| `secrets.choice` in `get_event_gateway_choice`                                   | ties at event-based gateways                                                                | **no**: never seedable; should become an ordinary random call                                                   |

A cheap alternative needs no library changes: before each engine call (building, `next_event_time`,
`step`, `deliver`), load that engine's saved Python and NumPy generator states; save them after.
*(run)* Engine A's draws are then identical with or without engine B running beside it, at about 36 µs
per call, roughly doubling the orchestrator's per-step time (~25 µs).

## 5. Rough size

| Item                                                                                     | Size                   |
|------------------------------------------------------------------------------------------|------------------------|
| Park/resume at an intermediate message catch event, incl. "blocked" reporting            | medium, 2–4 days       |
| Real race at event-based gateways (message vs timer)                                     | large, 3–5 days        |
| Sending: parse throw/end message events, produce `pending_publish()` items               | small–medium, 1–3 days |
| `create_case` mid-run, incl. attributes, ids, message start event                        | medium, 2–3 days       |
| Parallel branches: tests for join, terminate end, OR join                                | small, ~1 day          |
| Per-engine randomness by saving/restoring generator states (+ replace `secrets`)         | small, ~1 day          |
| Per-engine randomness with explicit generators everywhere (needs a pix-framework change) | large, 3–5 days        |
