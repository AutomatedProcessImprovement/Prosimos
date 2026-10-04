# Multi-process simulation

The orchestrator simulates several processes side by side on one shared clock. Each process is
simulated by its own engine, usually a Prosimos simulation of a BPMN model. The orchestrator
repeatedly steps the engine whose next event is earliest, passes the messages engines publish to the
processes that consume them, and writes one merged event log.

How a process model publishes and waits for messages is described in [messaging.md](messaging.md);
a complete example is in [running-example.md](running-example.md).

## Quick start

A configuration file lists the processes, the shared start time and, optionally, a seed and the
consumer groups. Here the warehouses are started by Sales's orders, so they have no `total_cases`:

```json
{
  "processes": [
    {"name": "Sales", "bpmn_path": "sales.bpmn", "json_path": "sales.json", "total_cases": 100},
    {"name": "TartuWarehouse", "bpmn_path": "tartu.bpmn", "json_path": "tartu.json"},
    {"name": "TallinnWarehouse", "bpmn_path": "tallinn.bpmn", "json_path": "tallinn.json"}
  ],
  "seed": 42,
  "start_time": "2024-01-01T09:00:00+00:00",
  "consumer_groups": {
    "Sales": ["Sales"],
    "Warehouses": ["TartuWarehouse", "TallinnWarehouse"]
  }
}
```

Run it from the command line:

```
poetry run prosimos start-orchestration --config config.json --log_out_path merged_log.csv --report_out_path report.json
```

`--log_out_path` (the merged event log) and `--report_out_path` (the run report as JSON) are optional,
and `--seed` overrides the configuration's seed. The command prints a short summary, e.g.

```
Messages: 37 published, 54 claims, 9 discards, 0 unclaimed
Stalled cases:
  Sales: 3 (Sales-2, Sales-6, Sales-7)
Warnings: none
```

or from Python:

```python
from prosimos.orchestrator import SimulationConfig, run_orchestrator

config = SimulationConfig.from_json("config.json")
report = run_orchestrator(config, log_out_path="merged_log.csv")

print(report.stalled)    # cases still waiting for a message at the end
print(report.warnings)   # e.g. messages nobody could use
```

Extra engines (see "Configuration") can only be passed from Python.

## Configuration

`run_orchestrator` takes a `SimulationConfig`, built in code or loaded with
`SimulationConfig.from_json(path)`:

| Field             | Meaning                                                                                                                                                                                                                                                                 |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `processes`       | one entry per process: a unique `name`, `bpmn_path`, `json_path`, and `total_cases` <br/>unless the process is started by messages ([messaging.md](messaging.md))                                                                                                       |
| `start_time`      | the simulation's start, shared by all processes; a time without a time zone is taken as UTC                                                                                                                                                                             |
| `seed`            | optional; the same seed gives the same run. Each Prosimos engine gets its own random generators, seeded from the seed and its process name, so adding or changing one process doesn't change the others' draws. Without a seed, every run draws different random values |
| `consumer_groups` | optional; group name → processes in it. Without it, every process is its own group (see "Message routing")                                                                                                                                                              |

File paths are relative to the configuration file's folder. A configuration is rejected if process
names repeat, a process is in no group or in more than one, or a group names a process that doesn't
exist.

**Extra engines.** Engines that aren't built from a BPMN and JSON file, such as the scripted engines
in the tests, can run alongside the configured ones. List their names in the configuration's
`extra_processes` (in code only, not in the JSON file), so that consumer groups can name them, and
pass the engines to `run_orchestrator(config, log_out_path, extra_engines={name: engine, ...})`. The
engines given must be exactly the declared names.

## Output

**Merged log.** With `log_out_path`, the events of every configured process are written to one CSV,
sorted by start time. The first column is the process name; the other columns are the union of the
processes' log columns (`case_id`, `activity`, `enable_time`, `start_time`, `end_time`, `resource`,
plus attribute and batch columns where a model has them). A process leaves blank the columns it
doesn't produce. Extra engines get no log writer, so their events aren't in the merged log.

**Run report.** `run_orchestrator` (and `run_engines`) return a `RunReport`; `RunReport.to_dict()`
gives it as plain lists and dicts with times as ISO strings, which is what `--report_out_path` saves:

| Field              | Content                                                                                               |
|--------------------|-------------------------------------------------------------------------------------------------------|
| `executed`         | (time, process) for every step                                                                        |
| `published`        | every message, as stamped by the orchestrator                                                         |
| `copies`           | (message id, group) for every copy put in the pool                                                    |
| `claims`           | (message id, process, time) for every claim                                                           |
| `discards`         | (message id, process, time) for every discard                                                         |
| `unclaimed`        | (group, message) for the copies still in the pool at the end                                          |
| `warnings`         | the orchestrator's warnings: a message type nobody subscribes to; a message every recipient discarded |
| `stalled`          | (process, `StalledCase`) for every case still waiting for a message at the end                        |
| `engine_warnings`  | (process, warning) for the warnings raised inside each engine                                         |
| `discarded_counts` | discards per (message type, process)                                                                  |

A `StalledCase` has the case id (e.g. `Sales-1`), the catch event it waits at, the message types it
waits for, and since when. The "discarded by every recipient" warning appears even when the discard
is expected, for example a shipment for an order that was cancelled.

## Message routing

A `Message` has a `type` and `attributes`, set by the publishing process. The orchestrator stamps
it with an `id` (`m1`, `m2`, ...), its `source` process and the `time` of the step that published it.

- **Consumer groups.** Each process belongs to exactly one group. Every group whose members
  subscribe to a message type gets its own copy of each such message, and only one member of the
  group can claim it. Two warehouses in one group compete for an order; a separate Billing group
  gets its own copy.
- **Routing table**, built once at the start: for each message type, the groups with at least one
  member subscribed to it.
- **Pool.** Copies wait in a pool until a member claims them, or until every member subscribed to
  the type has discarded them. A message type nobody subscribes to gets no copy and a warning.
- **Offers.** After each step, the orchestrator offers the new copies, then the pooled copies of the
  stepping engine's group, to the group's members, in random order, at the step's time, until one
  claims it. A member is offered copies only of the types it subscribes to, and never a copy it
  discarded before. An engine is offered new copies even when it has no next event of its own.
- **Randomness.** The member order comes from the orchestrator's own random generator, seeded from
  the simulation seed, so choosing a member never changes the engines' own random draws.
- **Order of steps.** The engine with the earliest next event steps first; ties are broken by
  process name. Engines are built in name order. Together with the seed, this makes runs repeatable.

## Engine interface

Everything the orchestrator may ask of an engine; nothing else crosses that boundary. It follows the
Orchestrator Protocol (kept outside this repository). In code it is the `SimulationEngine` class in
`prosimos/orchestrator.py`, and `ProsimosEngine` implements it for a Prosimos simulation. To add
another kind of engine, implement these five methods.

Each engine is a black box: no process may read or change another process's cases, queues,
resources, attributes or logs, and the orchestrator doesn't look inside engines either. Anything
that passes between processes is a message: it leaves an engine as the result of `step()` and enters
another through `deliver()`. Engines never call the orchestrator.

| Method                  | Returns                             | Called                            |
|-------------------------|-------------------------------------|-----------------------------------|
| `subscriptions()`       | list of message types               | once, at the start                |
| `next_event_time()`     | date and time, or `None`            | before every step                 |
| `step()`                | list of published messages          | to perform one event              |
| `deliver(message, now)` | `CLAIMED`, `DISCARDED` or `PENDING` | to offer one message to an engine |
| `finish()`              | an `EngineReport`                   | once, after the loop stops        |

The first four are the messaging calls; `finish()` is a lifecycle call that publishes and delivers
nothing.

**`subscriptions()`**: the message types this process consumes. For a Prosimos engine, the types
under `consume` in its model ([messaging.md](messaging.md)).

**`next_event_time()`**: the date and time of what the engine will do on its next `step()`, or `None`
when it has nothing to do right now. `None` doesn't mean finished: cases waiting for a message don't
count, and a delivered message can give the engine work again. Work waiting in a batch to fire does
count. Times are absolute dates, so answers from different engines can be compared; all engines get
the same start time. An engine handles its events strictly in time order, so it never goes back in
time. The first call prepares a Prosimos engine (it generates the arrival times of all its cases);
after that, asking twice gives the same answer.

**`step()`**: performs exactly one event and returns the messages it published (often none). This is
the only way to publish. The orchestrator calls it only after `next_event_time()` returned a time.
A Prosimos engine may instead release messages it held until they were due; `next_event_time()`
announces that step like any other.

**`deliver(message, now)`**: offers one message at time `now` and returns a `Verdict`:

- **`CLAIMED`** is a commitment: the message is now bound to one case, taking effect at `now`, and
  this copy is never offered to anyone again.
- **`DISCARDED`** is permanent: the engine will never want this message, even after its state
  changes, so it is never offered to this engine again.
- **`PENDING`**: not now; it may be offered again later. An engine that isn't sure must answer
  `PENDING`.

It is the same per-message acknowledgement that message brokers such as RabbitMQ use (ack, reject,
requeue).

**`finish()`**: called once on every engine after the loop stops, i.e. when no engine has a next
event. It returns an `EngineReport` with the engine's stalled cases and the warnings it raised during
the run (including while it was built). The orchestrator adds both to the run report, tagged with
the process name.

A Prosimos engine collects its own warnings: Prosimos writes warnings to one list shared by all
engines (`warning_logger`), so `ProsimosEngine` hands Prosimos its own list for the duration of each
of its methods, and two engines' warnings never mix. Prosimos's end-of-run usage statistics
(`find_issues`, e.g. "element used in less than 1% of cases") aren't included.

## Limitations

- **Outside the interface.** Building an engine (from a BPMN file, a JSON file, a number of cases and
  the start time) and handing it a log writer happen outside the five methods.
