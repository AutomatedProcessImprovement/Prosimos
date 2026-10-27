# Multi-process simulation

The orchestrator simulates several processes side by side on one shared clock. Each process is
simulated by its own engine, usually a Prosimos simulation of a BPMN model. The orchestrator
repeatedly steps the engine whose next event is earliest, passes the messages engines publish to the
processes that consume them, and writes one merged event log.

How a process model publishes and waits for messages is described in [messaging.md](messaging.md).
There are two examples: the [running example](running-example.md), with orders, invoices and shipments, and the
[Order Management example](order-management.md), after a real OCEL 2.0 event log.

## Quick start

A configuration file lists the processes, the shared start time and, optionally, a seed and the
consumer groups. This is the [running example](running-example.md)'s,
`testing_scripts/assets/running_example/simulation.json`: Sales's orders are billed by Billing and
packed by one of two warehouses, which are started by those orders, so they have no `arrivals`:

```json
{
  "processes": [
    {"name": "Sales", "bpmn_path": "sales.bpmn", "json_path": "sales.json", "arrivals": 20, "object_type": "order"},
    {"name": "Billing", "bpmn_path": "billing.bpmn", "json_path": "billing.json", "object_type": "invoice"},
    {"name": "TartuWarehouse", "bpmn_path": "tartu_warehouse.bpmn", "json_path": "tartu_warehouse.json",
     "object_type": "shipment"},
    {"name": "TallinnWarehouse", "bpmn_path": "tallinn_warehouse.bpmn", "json_path": "tallinn_warehouse.json",
     "object_type": "shipment"}
  ],
  "seed": 1,
  "start_time": "2024-01-01T09:00:00+00:00",
  "consumer_groups": {
    "Sales": ["Sales"],
    "Billing": ["Billing"],
    "Warehouses": ["TartuWarehouse", "TallinnWarehouse"]
  }
}
```

Run it from the repository's root folder:

```
poetry run prosimos start-orchestration --config testing_scripts/assets/running_example/simulation.json --log_out_path merged_log.csv --report_out_path report.json --ocel_out_path running_example.json
```

`--log_out_path` (the merged event log), `--report_out_path` (the run report as JSON) and
`--ocel_out_path` (the run as an OCEL 2.0 JSON file, see "Output") are optional, and `--seed` overrides
the configuration's seed. The command prints a short summary:

```
Messages: 37 published, 54 claims, 9 discards, 0 unclaimed
Stalled cases:
  Sales: 3 (order-2, order-6, order-7)
Warnings: none
```

or from Python:

```python
from prosimos.orchestrator import SimulationConfig, run_orchestrator

config = SimulationConfig.from_json("testing_scripts/assets/running_example/simulation.json")
report = run_orchestrator(config, log_out_path="merged_log.csv", ocel_out_path="running_example.json")

print(report.stalled)    # cases still waiting for a message at the end
print(report.warnings)   # e.g. messages nobody could use
```

Extra engines (see "Configuration") can only be passed from Python.

## Configuration

`run_orchestrator` takes a `SimulationConfig`, built in code or loaded with
`SimulationConfig.from_json(path)`:

| Field             | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `processes`       | one entry per process: a unique `name`, `bpmn_path`, `json_path`, and `arrivals`, how many cases arrive, <br/>unless the process is started by messages ([messaging.md](messaging.md)); optionally `object_type`, the type of the objects its cases are, one kind of thing, named in the singular (without one, the process is left out of the OCEL output; several processes may share one, e.g. two warehouses that both make shipments), `qualifier`, `qualifier_by_activity` and `carry_links` for its OCEL links and `ocel_attributes`, the attributes written to its objects (see "Output"), and `catalogues` and `references`, its objects with no events of their own and which of its attributes hold their ids ([catalogue.md](catalogue.md)) |
| `start_time`      | the simulation's start, shared by all processes; a time without a time zone is taken as UTC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `seed`            | optional; the same seed gives the same run. Each Prosimos engine gets its own random generators, seeded from the seed and its process name, so adding or changing one process doesn't change the others' draws. Without a seed, every run draws different random values                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `consumer_groups` | optional; group name → processes in it. Without it, every process is its own group (see "Message routing")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

File paths are relative to the configuration file's folder. A configuration is rejected if process
names repeat, a process is in no group or in more than one, a group names a process that doesn't exist,
or the catalogues and references break one of the rules listed in [catalogue.md](catalogue.md).
It is also rejected for a key it doesn't know, at the top level, in a process entry, a catalogue (a listed
object, a pool or a growing catalogue), a reference or its `pick`, since a misspelled key would otherwise
be ignored and change the run without any message:
`processes[0].references[0]: unknown key 'link_event'; allowed: catalogue, count, link_events, o2o, object_attribute, pick, reverse, tag_along, with`.

**A process's JSON file** (`json_path`) holds its Prosimos simulation settings, with the attributes drawn for each
new object under `object_attributes`, in Prosimos's format for its case attributes, and the `messages` section
([messaging.md](messaging.md)). The orchestrator hands the settings to Prosimos translated (`object_attributes` as
Prosimos's `case_attributes`, `arrivals` as its `total_cases`); a top-level key that isn't one of Prosimos's sections,
`object_attributes` or `messages` is rejected:
`sales.json: top level: unknown key 'object_atributes'; allowed: arrival_time_calendar, ...`. Single-process
Prosimos (`start-simulation`) uses its own names, `--total_cases` and `case_attributes`.

**Extra engines.** Engines that aren't built from a BPMN and JSON file, such as the scripted engines
in the tests, can run alongside the configured ones. List their names in the configuration's
`extra_processes` (in code only, not in the JSON file), so that consumer groups can name them, and
pass the engines to `run_orchestrator(config, log_out_path, extra_engines={name: engine, ...})`. The
engines given must be exactly the declared names.

## Output

**Merged log.** With `log_out_path`, the events of every configured process are written to one CSV,
sorted by start time. The first column is the process name; the other columns are the union of the
processes' log columns (`case_id`, the case's number in its process, e.g. 7 for `order-7`, `activity`,
`enable_time`, `start_time`, `end_time`, `resource`, plus attribute and batch columns where a model has them). A process leaves blank the columns it
doesn't produce. Extra engines get no log writer, so their events aren't in the merged log.

**OCEL 2.0 output.** With `ocel_out_path` (`--ocel_out_path` on the command line), the run is also
written as an OCEL 2.0 JSON file, the format of the OCEL 2.0 sample logs, readable with
`pm4py.read_ocel2_json`:

- **Objects:** one per case that some event refers to (see below), typed by its process's `object_type`, with the
  id `<object type>-<number>`, numbered from 0 per object type across all its processes, in the order the objects
  were created (objects created at the
  same moment go by the order of their processes in the configuration, and those of one process by its own case
  number), e.g. `order-7`, the eighth order. In messages and the run report, a case is named by its object type and
  its number in its own process (`object_id`, see [messaging.md](messaging.md); `Sales-7` for a process without an
  object type, which writes no objects). A process numbers its cases in the order it creates them, so for an object
  type only one process has, that is also the id in the OCEL output. Processes sharing an object type each number
  from 0, so the Tartu and the Tallinn warehouse both have a `shipment-0`; the OCEL output names them apart, e.g.
  `shipment-0` and `shipment-1`, and every link uses those ids. Its attributes are those its process lists in
  `ocel_attributes`, without it none, so that attributes only routing messages (an order id copied into an item, a
  list of item ids) stay out: the values each had when the object was created, at that time, and every value written
  into it later at a catch event, at the time of the claim (an appended list as JSON text). Of several values of one
  attribute at the same moment only the last is written: a package that takes 15 items at once gets one weight, the
  total, while one that takes 7, then 3, then 5 items at three moments gets three. A sum's or count's starting 0 is
  left out until a message writes it: a package created before its first item has no weight until it takes one. Only
  the OCEL output does this; the CSV log and the run report are unchanged. `ocel_attributes` may name only
  attributes the objects have: drawn in `object_attributes`, picked or brought (`with`) by a reference, or copied,
  appended or aggregated from messages; any other name is a load error (`Invalid ocel_attributes:
  processes[1].ocel_attributes names ['colour'], which the objects of process Warehouse never have (...)`). Its
  relationships are its links to other objects, from the `o2o` keys of message entries
  ([messaging.md](messaging.md)): e.g. an order comprises each item its ItemOrdered messages started. Without `o2o`
  keys, objects have no relationships. The catalogue's objects are written too, with all their attributes (see
  [catalogue.md](catalogue.md)).
- **Events:** one per task, i.e. per row of the merged log, with the activity as event type, the task's
  completion as time, `resource` as attribute, a link to its case's object, and links to the objects
  on the other end of its messages (see below) and to the catalogue objects its references name. Event ids (`e1`, `e2`, ...) follow time order.
- `objectTypes` and `eventTypes` list each type with its attributes; an attribute's type is inferred
  from its values.
- A process without an `object_type` writes no objects and no events. Extra engines write neither.

A case that has no task in the log yet, e.g. an order still collecting its items when the run ends, has no
events of its own, but it can still be linked from other events through its messages: the still-collecting
order is linked from the tasks that published the items it took (see below), so its object is written. A case
that no event refers to, e.g. a packing round still waiting for its first item when the run ends, is left out,
with the links from and to its object, as `pm4py.read_ocel2_json` would leave it out on reading; the run
report still lists it as stalled. Catalogue objects have no events by design and are always written.

**Objects without events: the catalogue.** Some objects never have events of their own: customers,
products, suppliers. A process's `catalogues` declare them (each a list, a generated pool, or a growing
catalogue), and each of its `references` says that one of its attributes' value is the id of an object of one
of its own catalogues, which OCEL links to write for it (`o2o`, `reverse`, `link_events`, `tag_along`), and,
with `pick`, chooses the value when a case is created. A process never refers to another's catalogue: what it
knows of another's objects comes in messages.
The catalogue's objects are written with the catalogue name as type. All keys, with examples, the Order
Management configuration as a worked example, and every error and warning are in
[catalogue.md](catalogue.md).

**One event linked to many objects.** Messages only pass at events, which aren't in the log, so a
message's links go to the task next to its event in the same case, on the token's path, and link the
object of the case on the other end of the message. Each end writes its links only if its entry has a
`qualifier` (see below), so a message may link from both ends, one, or neither:

- **Publishing** (throw or end event): the last task before it. In Order Management, place order →
  ItemOrdered ×3, so place order is linked to the 3 items the messages started.
- **Claiming** (start or catch event): the next task after it. Collect ItemPicked ×6 → create package,
  so create package is linked to the 6 items that sent them. At an event-based gateway, the claiming
  event is the winning branch's catch event, so the links go to the next task on that branch; when the
  timer wins, there is nothing to link. With `capacity`, each resumed case links its own next task.

The task is found by following the flows from the event through any other events and gateways, and if
the case ran that task more than once, by time: the last one completed at or before the publish, or
the first one enabled at or after the claim. Without such a task, e.g. a catch event followed only by
the end event, or a case still waiting further on when the run ends, the links are dropped, with one
warning per element in `engine_warnings`. An end without a qualifier, or of a process without an object type,
looks for no tasks and warns nothing; links to a process without an object type aren't written.

**Gateways between the event and its tasks.** The links go to the tasks the case actually ran next to
the event, so a gateway on the way links what it let through in that case:

| Gateway                | After a claiming event (split)                                      | Before a publishing event (join)                                       |
|------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------|
| exclusive, event-based | the one task on the branch taken (or on the winning event's branch) | the last task before the event, on the branch that came in             |
| parallel               | every task on its branches, also one a timer on its branch delays   | the last task of every branch, though they finished at different times |
| inclusive              | every task on the branches taken in that case, not on every branch  | the last task of every branch taken in that case                       |

After a claim, they are the first run of each task next to the event enabled at or after the claim;
before a publish, the last run of each task next to the event completed at or before the publish. Only
the case's runs in the same round count: after a claim, runs enabled before its next claim at that
event once it has moved on (with `collect`, the case claims several messages there before it moves
on); before a publish, runs completed since its previous publish there. So in a loop, a message
doesn't link the tasks of another round, and a branch that wasn't taken in that round links nothing.
A loop that returns to a point after the catch event makes no new claim to separate its passes, so a
branch taken only in a later pass is still linked; likewise, a loop that ends before the throw event
leaves no earlier publish there, so a branch taken only in an earlier pass is still linked.

This is an over-approximation: every object the message links is linked to every one of those tasks.
When the branches handle different objects, e.g. one packs the items and the other bills the order,
each object is still linked to all the parallel tasks. Linking each object only to the branch task that
handles its object type is future work.

Three settings shape the links:

| Setting                                | Meaning                                                                                                                                                                                       |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `qualifier` (process)                  | qualifier of each event's link to its own case's object; by default the object type                                                                                                           |
| `qualifier_by_activity` (process)      | task id → qualifier, overriding `qualifier` for that task's events, e.g. `{"Create_Package": "creates", "Send_Package": "shipped package"}`; a key that isn't a task of the model is rejected |
| `carry_links` (process)                | `true`: every later event of a case also links the objects its earlier events linked through messages, e.g. send package and package delivered link the package's items. Off by default       |
| `qualifier` (publish or consume entry) | qualifier of the links this end of the message makes, in the JSON file ([messaging.md](messaging.md)); without one, this end makes no links                                                   |

So with a `"qualifier": "item"` on the ItemOrdered publish entry, place order's links to the items are
qualified `item`; the Warehouse's consume entry has none, so the item's first task doesn't link its order. The
link of every event to its own case's object is always written.

**Run report.** `run_orchestrator` (and `run_engines`) return a `RunReport`; `RunReport.to_dict()`
gives it as plain lists and dicts with times as ISO strings, which is what `--report_out_path` saves:

| Field              | Content                                                                                                            |
|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `executed`         | (time, process) for every step                                                                                     |
| `published`        | every message, as stamped by the orchestrator                                                                      |
| `copies`           | (message id, group) for every copy put in the pool                                                                 |
| `claims`           | (message id, process, time) for every claim                                                                        |
| `discards`         | (message id, process, time) for every discard                                                                      |
| `unclaimed`        | (group, message) for the copies still in the pool at the end                                                       |
| `warnings`         | the orchestrator's warnings: a message nobody but its publisher subscribes to; a message every recipient discarded |
| `stalled`          | (process, `StalledCase`) for every case still waiting for a message at the end                                     |
| `engine_warnings`  | (process, warning) for the warnings raised inside each engine                                                      |
| `discarded_counts` | discards per (message type, process)                                                                               |
| `message_records`  | per message, in publishing order: who published it and who took it (see below)                                     |

A `StalledCase` has `object_id` (e.g. `order-1`), `event_id` (the catch event it waits at, or the
event-based gateway for a case waiting in a race, reported once with the types of all its message
branches), `message_types` (the types it waits for: any one of them resumes it), `waiting_since`, and
`collected` of `needed` (how many messages it had collected there of the ones it needed; more than one
with `collect`, see [messaging.md](messaging.md)). The "discarded by every recipient" warning appears even when the discard
is expected, for example a shipment for an order that was canceled.

**Message records.** Each engine keeps its own records of which of its cases and elements published
and took each message, and hands them over in `finish()`, at the end of the run; the orchestrator joins
them on the message id only then. Nothing it decides during the run uses them, so engines stay black
boxes. A `MessageRecord` has the message's `message_id` and `type`, its `publisher`, and its
`claimers`, each a `CaseElement` (`process`, `object_id`, `element_id`, and for the OCEL links its
end's `qualifier`, `task_rows`, the rows of its process's log the links go to, empty if there is
no such task, and `o2o`, its entry's object link, if any):

- **publisher**: the case and the throw or end event that published it;
- **claimers**: at a start event, the case the message started; at a catch event or race branch, the
  waiting case and that catch event; with `capacity`, every case the message resumed; with `collect`,
  the collecting case, once for each message it claimed. A message that was discarded or never claimed
  has no claimers.

An engine that keeps no records, such as a scripted test engine, leaves `publisher` empty for its
messages and appears in no `claimers`.

## Message routing

A `Message` has a `type` and `attributes`, set by the publishing process. The orchestrator stamps
it with an `id` (`m1`, `m2`, ...), its `source` process and the `time` of the step that published it.

- **Consumer groups.** Each process belongs to exactly one group. Every group whose members
  subscribe to a message type gets its own copy of each such message, and only one member of the
  group can claim it. Two warehouses in one group compete for an order; a separate Billing group
  gets its own copy.
- **Never back to its publisher.** A message is never offered to the process that published it, i.e.
  the process named in its `source`; the other members of its group are offered it as usual. In BPMN,
  message flows connect different participants: a process handing things to itself, e.g. a Workshop
  freeing parts for its later cases, needs a second process, e.g. a parts Stock. Two warehouses that
  forward orders to each other publish and consume the same type, and each gets the other's orders.
- **Routing table**, built once at the start: for each message type, the groups with at least one
  member subscribed to it. A message gets a copy for each of these groups with a subscribed member
  other than its publisher, so a group whose only subscribed member is the publisher gets none.
- **Pool.** Copies wait in a pool until a member claims them, or until every member subscribed to
  the type, other than the publisher, has discarded them. A message that gets no copy at all comes with
  a warning: `Newsletter m1 from P has no subscribers` when nobody subscribes to its type, and
  `PartFreed m1 from Workshop: no subscriber other than its publisher` when only the publisher does.
  Unclaimed copies and stalled cases at the end are the other signs of such a model; none of this is a
  load error.
- **Offers.** After each step, the orchestrator offers the new copies and the pooled copies of the
  stepping engine's group, oldest first, to the group's members, in random order, at the step's time,
  until one claims it. A member is offered copies only of the types it subscribes to, and never a copy it
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
event. It returns an `EngineReport` with the engine's stalled cases, the warnings it raised during the
run (including while it was built), its message records (`published` and `claimed`, each a list of
(message id, object id, element id, qualifier, task rows, o2o)), its cases as objects, and the element id of
each row of its log (`logged_elements`). The orchestrator adds the stalled cases and warnings to the run
report, tagged with the process name, joins the records into `message_records`, and uses the objects
and rows for the OCEL output.

A Prosimos engine learns each message's id without any change to the interface: the orchestrator
stamps the id on the very `Message` objects `step()` returned, and `deliver()` receives the stamped
copy.

A Prosimos engine collects its own warnings: Prosimos writes warnings to one list shared by all
engines (`warning_logger`), so `ProsimosEngine` hands Prosimos its own list for the duration of each
of its methods, and two engines' warnings never mix. Prosimos's end-of-run usage statistics
(`find_issues`, e.g. "element used in less than 1% of cases") aren't included.

## Limitations

- **Outside the interface.** Building an engine (from a BPMN file, a JSON file, a number of cases and
  the start time) and handing it a log writer happen outside the five methods.
- **OCEL links across parallel branches.** A message's objects are linked to every task a parallel or
  inclusive gateway lets through next to its event, even when each branch handles only some of the
  objects (see "Gateways between the event and its tasks"). Routing each object to the branch task that
  handles its object type is future work.
- **Catalogue.** See [catalogue.md](catalogue.md) for its limitations.
