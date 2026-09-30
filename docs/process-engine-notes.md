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
  Parking = leave the token on the event's incoming flow and record `(case, event, enabled time)`
  instead of queueing it (in `_find_next`, where `is_event=True` tasks are created). Resuming = queue
  an `EnabledEvent` for that event at the message's time and let the existing path continue.
- **Event-based gateways** don't wait either: `get_event_gateway_choice` draws a duration for each
  following event the moment the case arrives and takes the shortest. A real race between a timer and
  a message needs a new mechanism (wait, first one wins, cancel the others).
- **Sending is not supported yet.** Intermediate throw events, send/receive tasks and boundary events
  are not parsed; a model with an intermediate throw event crashes with `KeyError` as soon as a case
  reaches it *(run)*. Message end events are parsed, as ordinary end events.
- A parked case has nothing in the queue, so the engine would report "finished" (`next_event_time()`
  = `None`) while cases still wait. This is what `blocked_on()` has to express.

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
`step`, future `inject`), load that engine's saved Python and NumPy generator states; save them after.
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
