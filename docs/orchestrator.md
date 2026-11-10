# Multi-process simulation

The orchestrator simulates several processes side by side on one shared clock. Each process is
simulated by its own engine, usually a Prosimos simulation of a BPMN model. The orchestrator
repeatedly steps the engine whose next event is earliest, passes the messages engines publish to the
processes that consume them, and writes one merged event log.

How a process model publishes and waits for messages is described in [messaging.md](messaging.md).
There are three examples: the [Order Routing example](order-routing.md), with orders, invoices and shipments, the
[Order Management example](order-management.md), after a real OCEL 2.0 event log, and the
[Stock Reservation example](stock-reservation.md), with quotes.

## Quick start

A configuration file lists the organizations, each with its processes, the shared start time and,
optionally, a seed and the consumer groups. This is the [Order Routing example](order-routing.md)'s,
`testing_scripts/assets/order_routing/simulation.json`: Sales's orders are billed by Billing and
packed by one of two warehouses, which are started by those orders, so they have no `arrivals`:

```json
{
  "organizations": [
    {"name": "Shop",
     "processes": [
       {"name": "Sales", "bpmn_path": "sales.bpmn", "json_path": "sales.json", "arrivals": 20, "object_type": "order"},
       {"name": "Billing", "bpmn_path": "billing.bpmn", "json_path": "billing.json", "object_type": "invoice"},
       {"name": "TartuWarehouse", "bpmn_path": "tartu_warehouse.bpmn", "json_path": "tartu_warehouse.json",
        "object_type": "shipment"}
     ]},
    {"name": "Tallinn partner",
     "processes": [
       {"name": "TallinnWarehouse", "bpmn_path": "tallinn_warehouse.bpmn", "json_path": "tallinn_warehouse.json",
        "object_type": "shipment"}
     ]}
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
poetry run prosimos start-orchestration --config testing_scripts/assets/order_routing/simulation.json --log_out_path merged_log.csv --report_out_path report.json --ocel_out_path order_routing.json
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
from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation

config = SimulationConfig.from_json("testing_scripts/assets/order_routing/simulation.json")
report = run_multiprocess_simulation(config, log_out_path="merged_log.csv", ocel_out_path="order_routing.json")

print(report.stalled)    # cases still waiting for a message at the end
print(report.warnings)   # e.g. messages nobody could use
```

Extra engines (see "Configuration") can only be passed from Python.

## Configuration

`run_multiprocess_simulation` takes a `SimulationConfig`, built in code or loaded with
`SimulationConfig.from_json(path)`:

| Field             | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `organizations`   | one entry per organization, a business: a unique `name`, its `processes` and, optionally, its `catalogues`, objects with no events of their own that its processes share, and its `catalogue_changes`, which change their attributes on the clock ([catalogue.md](catalogue.md)). Every process belongs to exactly one organization; a business with one process is an organization with one process (see "Organizations" below)                                                                                                                                                                                                                                                                                                      |
| `processes`       | in an organization, one entry per process: a unique `name`, `bpmn_path`, `json_path`, and `arrivals`, how many cases arrive, <br/>unless the process is started by messages ([messaging.md](messaging.md)); optionally `object_type`, the type of the objects its cases are, one kind of thing, named in the singular (without one, the process is left out of the OCEL output; several processes may share one, e.g. two warehouses that both make shipments), `qualifier` and `qualifier_by_activity` for its OCEL links and `ocel_attributes`, the attributes written to its objects (see "Output"), and `references`, which of its attributes hold the ids of its organization's catalogue objects ([catalogue.md](catalogue.md)) |
| `start_time`      | the simulation's start, shared by all processes; a time without a time zone is taken as UTC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `seed`            | optional; the same seed gives the same run. Each Prosimos engine gets its own random generators, seeded from the seed and its process name, so adding or changing one process doesn't change the others' draws. Without a seed, every run draws different random values                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `consumer_groups` | optional; group name → processes in it. Without it, every process is its own group (see "Message routing")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `load_balancing`  | optional; group name → its policy, the order its members are offered a message: `{"weighted": {member: weight}}` or `{"preferred": [members]}`. A group without one is offered in random order (see "Message routing")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

File paths are relative to the configuration file's folder. A configuration is rejected if organization or process
names repeat, a process is named `orchestrator` (see "Quotes"), a process is in no group or in more than one, a
group names a process that doesn't exist, or the catalogues and references break one of the rules listed in
[catalogue.md](catalogue.md). In `load_balancing`, each entry names a consumer group with more than one member and
has exactly one policy; a weighted policy gives every member, and only members, a weight that is a number of at
least 0, not all 0, and a preferred one lists each member exactly once.
It is also rejected for a key it doesn't know, at the top level (a top-level `processes`, for example: processes
belong to an organization), in an organization entry, a process entry, a catalogue (a listed object, a pool or a
growing catalogue), a reference or its `pick`, since a misspelled key would otherwise be ignored and change the
run without any message, and every message names where the key is:
`organizations[0].processes[0].references[0]: unknown key 'link_event'; allowed: catalogue, link_events, o2o, object_attribute, pick, quantity, reverse, tag_along, with`.

**Organizations.** An organization is a box around a business's processes and its data: inside, its processes
share its catalogues; between organizations, only messages pass. Messages, routing and consumer groups don't
depend on organizations: a group may name processes of any organization, e.g. a partner's warehouse competing
with a business's own (the Order Routing example's Warehouses group holds Shop's TartuWarehouse and Tallinn partner's
TallinnWarehouse). Engines never learn their organization: catalogues and picks are handled before the engines
are built, and each process still draws its picks from its own generator. The organization isn't written in
the output yet. Ready-made engines passed from Python (`extra_processes`) belong to no organization and read no
catalogues.

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
pass the engines to `run_multiprocess_simulation(config, log_out_path, extra_engines={name: engine,
...})`. The engines given must be exactly the declared names.

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
  left out until a message writes it: a package created before its first item has no weight until it takes one. A
  value that goes back to no value, e.g. a message copying `None` into it, is written as `"value": null` at that
  moment, a clearing, so the file doesn't keep showing the old value; an attribute that never had a value written,
  or is already cleared, writes nothing (`pm4py.read_ocel2_json` reads the clearing as a change to no value). Only
  the OCEL output does this; the CSV log and the run report are unchanged. `ocel_attributes` may name only
  attributes the objects have: drawn in `object_attributes`, picked or brought (`with`) by a reference, or copied,
  appended or aggregated from messages; any other name is a load error (`Invalid ocel_attributes:
  organizations[0].processes[1].ocel_attributes names ['colour'], which the objects of process Warehouse never have
  (...)`). Its
  relationships are its links to other objects, from the `o2o` keys of message entries
  ([messaging.md](messaging.md)): e.g. an order comprises each item its ItemOrdered messages started. Without `o2o`
  keys, objects have no relationships. The catalogue's objects are written too, with all their attributes (see
  [catalogue.md](catalogue.md)).
- **Events:** one per task, i.e. per row of the merged log, with the activity as event type, the task's
  completion as time, `resource`, `start_time` and `enable_time` as attributes, a link to its case's object and
  to the catalogue objects its references name; and one per case, element and moment a message end with a
  qualifier sent or took messages, linked to the objects on the other end (see below). Event ids (`e1`, `e2`, ...)
  follow time order.
- **Event times, an extension of OCEL.** OCEL events are atomic: one time each, in OCEL 2.0 and in the 2.1
  pre-release. A simulated task takes time and may wait to start, and both matter for simulation (waiting, working
  time, resource use), so every event also carries the task's `start_time` and `enable_time` from its log row,
  written like `time` (e.g. `2023-04-03T09:00:00.000000Z`) and declared with type `time` in every event type. OCEL
  allows extra event attributes, so the file stays OCEL 2.0, and `pm4py.read_ocel2_json` reads both as times. The
  event's own `time` stays the completion. The values are the log's: Prosimos computes a task's start from seconds
  since the start of the run, so a task that starts the moment it is enabled may show an `enable_time` 1 µs after
  its `start_time` (90 of about 22,000 tasks of the Order Management example with seed 1). A message event has
  no `resource`, as nobody performs it: its `time` and `start_time` are the moment it sent or took its messages,
  its `enable_time` when the case reached the event (at a catch event, when the case began waiting there), and
  only task event types declare `resource`.
- `objectTypes` and `eventTypes` list each type with its attributes; an attribute's type is inferred
  from its values.
- A process without an `object_type` writes no objects and no events. Extra engines write neither.

A case that has no task in the log yet, e.g. an order still collecting its items when the run ends, may still
have events of its own, its message events, and be linked from other cases' message events: the
still-collecting order is linked from the events that published the items it took (see below), so its object is
written. A case
that no event refers to, e.g. a packing round still waiting for its first item when the run ends, is left out,
with the links from and to its object, as `pm4py.read_ocel2_json` would leave it out on reading; the run
report still lists it as stalled. Catalogue objects have no events by design and are always written.

**Objects without events: the catalogue.** Some objects never have events of their own: customers,
products, suppliers. An organization's `catalogues` declare them (each a list, a generated pool, or a growing
catalogue), shared by its processes, and each of a process's `references` says that one of its attributes' value
is the id of an object of one of its organization's catalogues, which OCEL links to write for it (`o2o`,
`reverse`, `link_events`, `tag_along`), and, with `pick`, chooses the value when a case is created. A process
never refers to another organization's catalogue: what it knows of another organization's objects comes in
messages. Each catalogue object is written once, whichever processes link it.
The catalogue's objects are written with the catalogue name as type. All keys, with examples, the Order
Management configuration as a worked example, and every error and warning are in
[catalogue.md](catalogue.md).

**One event linked to many objects.** Messages pass at events, which aren't in the log of tasks. A message end
(a throw, catch, start or end event) whose entry has a `qualifier` (see below) writes an OCEL event of its own
per case, element and moment it sends or takes messages, named after the event's BPMN name (its id without one),
and that event links the objects of the cases at the other end of every message it sent or took then. So a
message may link from both ends, one, or neither:

- **Publishing** (throw or end event): one event for the messages a case sends at once. In Order Management,
  place order sends ItemOrdered ×3, so it writes one event, linked to the 3 items the messages started. With
  `fan_in`, each member writes its own event for its part: each item's Order picked is linked to the package its
  OrderPicked started.
- **Claiming** (start or catch event): one event per moment the case takes messages. Items received collecting
  ItemPicked ×6 writes one event if the 6 come at once, or one per arrival, each linked to the items whose messages
  it took then, all enabled when the package began waiting. A message taken on a race branch that later lost the
  race is still written. With `capacity`, each resumed case writes its own event.

An end without a qualifier, or of a process without an object type, writes no event and no links; links to a
process without an object type aren't written. The CSV log stays a log of tasks: message events are only in the
OCEL file.

Three settings shape the links:

| Setting                                | Meaning                                                                                                                                                                                                                                                                                                                                                             |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `qualifier` (process)                  | qualifier of each event's link to its own case's object; by default the object type                                                                                                                                                                                                                                                                                 |
| `qualifier_by_activity` (process)      | task or message event id → qualifier, overriding `qualifier` for its events, e.g. `{"Create_Package": "creates", "Send_Package": "shipped package"}`; a key that isn't a task or message event of the model is rejected, as is a message event whose entry has no `qualifier`, which writes no event                                                                |
| `carry` (publish or consume entry)     | `true`: every later event of the same case, after the message's event, also links the objects that message linked, e.g. create package, send package and package delivered link the package's items; a list of activities (names or ids of tasks or message events, as in `link_events`): only those later events. Needs a `qualifier` on the entry; off by default |
| `qualifier` (publish or consume entry) | qualifier of the links this end of the message makes, in the JSON file ([messaging.md](messaging.md)); without one, this end makes no links                                                                                                                                                                                                                         |

So with a `"qualifier": "item"` on the ItemOrdered publish entry, each place order event's link to its item is
qualified `item`; the Warehouse's consume entry has none, so the item's message start writes no event and doesn't
link its order. The link of every event to its own case's object is always written.

**Run report.** `run_multiprocess_simulation` (and `orchestrate`) return a `RunReport`;
`RunReport.to_dict()` gives it as plain lists and dicts with times as ISO strings, which is what
`--report_out_path` saves:

| Field              | Content                                                                                                                                                                                                                  |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `executed`         | (time, process) for every step                                                                                                                                                                                           |
| `published`        | every message, as stamped by the orchestrator                                                                                                                                                                            |
| `copies`           | (message id, group) for every copy put in the pool                                                                                                                                                                       |
| `claims`           | (message id, process, time) for every claim                                                                                                                                                                              |
| `discards`         | (message id, process, time) for every discard                                                                                                                                                                            |
| `unclaimed`        | (group, message) for the copies still in the pool at the end                                                                                                                                                             |
| `warnings`         | the orchestrator's warnings: a message nobody but its publisher subscribes to; a message every recipient discarded                                                                                                       |
| `stalled`          | (process, `StalledCase`) for every case still waiting for a message at the end                                                                                                                                           |
| `engine_warnings`  | (process, warning) for the warnings raised inside each engine                                                                                                                                                            |
| `discarded_counts` | discards per (message type, process)                                                                                                                                                                                     |
| `message_records`  | per message, in publishing order: who published it and who took it (see below)                                                                                                                                           |
| `quotes`           | per quote type, how often each outcome came: `accepted`, `declined`, `canceled`, `cancel_too_late`, `modified_accepted`, `modified_declined`, `declined_as_second`                                                       |
| `broken_promises`  | per quote type: `count` of the portions that moved on after their quote's `ready_by`, `average_late`, `most_late`; `undelivered`, the portions past their `ready_by` not moved on by the end, `most_overdue` (see below) |

A `StalledCase` has `object_id` (e.g. `order-1`), `event_id` (the catch event it waits at, or the
event-based gateway for a case waiting in a race, reported once with the types of all its message
branches), `message_types` (the types it waits for: any one of them resumes it), `waiting_since`, and
`collected` of `needed` (how many messages it had collected there of the ones it needed; more than one
with `collect`, see [messaging.md](messaging.md)). The "discarded by every recipient" warning appears even when the discard
is expected, for example a shipment for an order that was canceled.

**Quotes in the report.** `quotes` and `broken_promises` are only in the report of a run with quotes, so a run
without keeps its report. Each counts per quote type: a quote counts as `accepted` or `declined`, a modify as
`modified_accepted` or `modified_declined`, a quote declined at once because one of the same type is open as
`declined_as_second`, and a cancel as `canceled` or, once some portion has moved on, `cancel_too_late`.
A promise is kept or broken per member's portion: a portion that moves on after the `ready_by` of its quote's
last accepted reply is late, by that much; a quote a later one replaced keeps its promise. A portion of an open or
locked quote still not moved on when the run ends, past its `ready_by`, is undelivered; `most_overdue` is how long
past it the run ended, a lower bound. A portion promised after the end of the run, or of a canceled quote, isn't
counted. The engines say when each portion moved on in `finish()`; durations are seconds in `to_dict()`.

**Message records.** Each engine keeps its own records of which of its cases and elements published
and took each message, and hands them over in `finish()`, at the end of the run; the orchestrator joins
them on the message id only then. Nothing it decides during the run uses them, so engines stay black
boxes. A `MessageRecord` has the message's `message_id` and `type`, its `publishers`, and its
`claimers`, each a `CaseElement` (`process`, `object_id`, `element_id`, and for the OCEL output its
end's `qualifier`; with one, its event's `activity`, `at`, the moment it sent or took the message, and `enabled`,
when its case reached the element, else `None`; its entry's `o2o` object link and `carry`, if any):

- **publishers**: the case and the throw or end event that published it; with `fan_in`, every case that
  handed in a part, in the order the parts came in, each with its own publish record;
- **claimers**: at a start event, the case the message started; at a catch event or race branch, the
  waiting case and that catch event; with `capacity`, every case the message resumed; with `collect`,
  the collecting case, once for each message it claimed. A message that was discarded or never claimed
  has no claimers.
- **quotes**: a reply's publisher is the orchestrator (`process` `orchestrator`, with no object or element) and
  its claimer the case that caught it; a quote's or modify's claimers are the items that took its lines, at the
  moment their portion moved on, so the OCEL links each item to the order as for an ordinary message. A case that
  ends with an open quote publishes its cancel where it ended, at the end event (or the terminate end event, or
  the timer's branch it was on), not at the cancel's own throw event, which it never passed.

An engine that keeps no records, such as a scripted test engine, leaves `publishers` empty for its
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
- **Offers.** After each step, the orchestrator offers the new copies and the pooled copies of the stepping
  engine's group, oldest first, to the group's members, in the order of the group's policy (see "Load
  balancing"), at the step's time, until one claims it. A member is offered copies only of the types it
  subscribes to, and never a copy it discarded before. An engine is offered new copies even when it has no next
  event of its own.
- **Randomness.** The member order comes from the orchestrator's own random generator, seeded from
  the simulation seed, so choosing a member never changes the engines' own random draws.

**Load balancing.** A group's policy in `load_balancing` decides only the order in which its members are offered
a copy; the round itself is the same for every policy (only subscribed members, never the publisher or a member
that discarded the copy, one by one, the first claim wins, a PENDING answer passes it to the next):

| Policy           | The order of the members asked                                             | Example                                                                                    |
|------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| random (default) | a fresh shuffle                                                            | Tartu or Tallinn first, 50/50                                                              |
| `weighted`       | the first member drawn by weight, then the next by weight among those left | `{"weighted": {"TartuWarehouse": 8, "TallinnWarehouse": 2}}`: Tartu first in 80% of offers |
| `preferred`      | a fixed ranking, the fallback chain                                        | `{"preferred": ["TartuWarehouse", "RigaWarehouse", "TallinnWarehouse"]}`                   |

- **A weight is a preference, not a quota:** it says how often a member is asked first. Weights are relative
  (8 and 2 mean 80% and 20%), and members with weight 0 are asked last, in random order. A member that discards
  a copy or answers PENDING passes it on, so its share of the messages handled can end up lower: with 8/2 and
  Tartu unable to take every other order yet, Tartu handles about 40% of them.
- **Preferred is a ranking, not just a favourite:** with two members it equals weights 100/0, with three or more
  it also fixes the fallback order. Members not asked in a round are left out of it.
- **Decided at every offer round:** a copy waiting in the pool gets a fresh order each time it is offered: a new
  draw (weighted), the same ranking (preferred) or a new shuffle (random). Every draw comes from the
  orchestrator's generator, so the same seed gives the same run, and a configuration without `load_balancing`
  makes exactly the draws it made before policies existed.
- **Order of steps.** The engine with the earliest next event steps first; ties are broken by
  process name. Engines are built in name order. Together with the seed, this makes runs repeatable.

## Quotes

A process can ask a quote instead of throwing a message and hoping someone claims it: "can you deliver an iPhone
and two iPads within 3 days?" The quote goes to the members of the one group that subscribes to its type; each
answers when it could have each line ready, and the orchestrator either books the items and replies accepted,
with the promised date, or books nothing and replies declined. Engines stay black boxes: the requester sees only
the reply and its date, and a member never sees the limit. How a model asks a quote, and how a Prosimos engine
answers and holds its items, is in [messaging.md](messaging.md), "Quotes"; the
[Stock Reservation example](stock-reservation.md) runs it end to end.

**One step.** A message from `step()` that carries a quote (`message.quote`) is stamped as usual, and its lines
get ids of their own (`l1`, `l2`, ...), but it isn't put in the pool. In the same step, the orchestrator:

1. asks each member of the group that subscribes to the quote type, other than the requester, for its slots:
   `estimate()` answers, per line, when its first, second, ... matching item not promised yet would be ready;
2. plans the lines by the slot rule: each line takes the earliest unused slot across the members, and identical
   lines share one slot list, so they take its first, second, ... slot; a line with no slot makes the plan
   impossible. The plan's `ready_by` is when its latest line would be ready, and never before the quote's own time;
3. compares it with the limit, the quote's time plus its `within`, a date fixed once, when the quote is made;
   without `within` there is no limit (a backorder);
4. within the limit, books each line at its member (`reserve()`) and publishes the accepted reply; otherwise
   books nothing and publishes the declined one.

Asking, comparing, and booking happen at one moment, so no answer is stale. `load_balancing` plays no part: it
orders the offers of ordinary messages only.

| Reply                          | Attributes                                                                                  |
|--------------------------------|---------------------------------------------------------------------------------------------|
| accepted, e.g. `QuoteAccepted` | `request`, the requester's `object_id`; `ready_by`; `portions`, how many members hold lines |
| declined, e.g. `QuoteDeclined` | `request`; `earliest`, when every line could be ready, or `None` when some line has no slot |

Replies are ordinary messages, published by the orchestrator: their source is `orchestrator`, and they are routed
like any message, in the same step; the requester catches them with a condition on `request`. A process named
`orchestrator` is therefore a load error (`A process can't be named orchestrator: the replies to quotes come from
the orchestrator under that name`). A quote without lines is accepted at once, ready by its own time, with
`portions` 0 and a warning (`ItemQuote m1 from Sales has no lines`).

**The book.** The orchestrator keeps each quote by its requesting process, the requester's object, and the quote
type: its state, its limit date, and which member holds each line, booked by which message. It never learns that
a quote was delivered; it asks the members holding lines whether their portion has moved on (`moved()`, with the
message that reserved the lines there) when a second quote, a modify, or a cancel comes. Some portion moved on,
and the quote is locked; every portion, and it is complete.

- **A second quote** of the same type from a case whose quote is open, nothing of it moved on, is declined at
  once, with a warning (`order-7 already has an open ItemQuote; use modify`). After a decline, or once something
  of the first has moved on, as for an order that loops and quotes again, the second is answered as usual and
  replaces the first.
- **A modify** (a message of the quote's modify type, with its new list) is declined once some portion has moved
  on. Otherwise the lines in both lists, matched by their attributes, stay where they are, at the same member and
  in the same place of its queue; the added lines get slots by the slot rule; and `ready_by` is the latest of the
  kept and added lines, checked against the original limit date. Within it, the dropped lines are released, the
  added ones reserved, and the reply is accepted, with the new `ready_by` and `portions`; otherwise nothing
  changes and it is declined, with `earliest`. After a declined quote, a modify is a fresh quote of its list
  against the original limit date; without a quote before it, it is declined, with a warning (`order-7 has no
  ItemQuote to modify`).
- **A cancel** (a message of the cancel type) releases every line at its member (`release()`), unless some
  portion has moved on: then it is too late, and nothing is released. A cancel of a declined, complete, or
  unknown quote does nothing. A cancel gets no reply. A case that ends with an open quote cancels it
  automatically ([messaging.md](messaging.md), "Quotes").

So a quote is locked at its first shipment: once the first member's portion moves on, it can be neither canceled
nor modified, and the rest still ships. After a release, the orchestrator offers the group's pooled messages
again, so a freed item can take an ordinary message already waiting.

**One group.** A quote type must reach exactly one group, whose members answer its quotes. Only the requester's
settings say which types are quote types, so the orchestrator checks the first quote of each type: one whose type
more than one group subscribes to stops the run (`ItemQuote m1 from Sales asks a quote of type ItemQuote, which
the groups ['Warehouses', 'Outlets'] subscribe to: a quote type must reach exactly one group`).

## Engine interface

Everything the orchestrator may ask of an engine; nothing else crosses that boundary. It follows the
Orchestrator Protocol (kept outside this repository). In code it is the `SimulationEngine` class in
`prosimos/orchestration/protocol.py`, and `ProsimosEngine` implements it for a Prosimos simulation. To
add another kind of engine, implement these nine methods.

Each engine is a black box: no process may read or change another process's cases, queues,
resources, attributes or logs, and the orchestrator doesn't look inside engines either. Anything
that passes between processes is a message: it leaves an engine as the result of `step()` and enters
another through `deliver()`. Engines never call the orchestrator.

| Method                            | Returns                                                                     | Called                                  |
|-----------------------------------|-----------------------------------------------------------------------------|-----------------------------------------|
| `subscriptions()`                 | list of message types                                                       | once, at the start                      |
| `next_event_time()`               | date and time, or `None`                                                    | before every step                       |
| `step()`                          | list of published messages                                                  | to perform one event                    |
| `deliver(message, now)`           | `CLAIMED`, `DISCARDED` or `PENDING`                                         | to offer one message to an engine       |
| `estimate(message, now)`          | its slots: per line id, when the engine could have it ready                 | to ask a member about a quote           |
| `moved(message, now)`             | `True` or `False`: whether some line of the quote has already moved on here | to ask a member holding a quote's lines |
| `reserve(message, line_ids, now)` | nothing                                                                     | to book a quote's lines at a member     |
| `release(message, line_ids, now)` | nothing                                                                     | to free lines a member booked           |
| `finish()`                        | an `EngineReport`                                                           | once, after the loop stops              |

The first four are the messaging calls; `estimate()`, `moved()`, `reserve()`, and `release()` are the quote
calls, for a process that asks the members of a group whether they could deliver. Like `deliver()`, they take
the stamped message, with its id and source, and the quote it asks is `message.quote`. `finish()` is a lifecycle
call that publishes and delivers nothing.

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

**`estimate(message, now)`**: per line id of `message.quote`, the times this member's matching items not yet
promised would be ready, earliest first (its `Slots`); a line it can't serve gets no times. It changes nothing.
How a Prosimos engine estimates is in [messaging.md](messaging.md), "Quotes".

**`moved(message, now)`**: whether some line that `message`, a quote or modify, reserved at this member has
already moved on by `now`; it changes nothing. A Prosimos engine's portion moves on all at once.

**`reserve(message, line_ids, now)`** books those lines of the quote at this member, for its requester;
**`release(message, line_ids, now)`** frees lines it booked. A Prosimos engine opens booked lines in a queue that
its items take in order, and returns a freed line's item to the shelf ([messaging.md](messaging.md), "Quotes").

**`finish()`**: called once on every engine after the loop stops, i.e. when no engine has a next
event. It returns an `EngineReport` with the engine's stalled cases, the warnings it raised during the
run (including while it was built), its message records (`published` and `claimed`, each a list of `MessageSide`:
the message id, object id, element id, and the end's `qualifier`, `activity`, `at`, `enabled`, `o2o`, and `carry`),
its cases as objects, the element id of each row of its log (`logged_elements`), and when each quote's portion
held here moved on (`moved`, by the id of the message that reserved it). The orchestrator adds the stalled cases
and warnings to the run report, tagged with the process name, joins the records into `message_records`, uses the
objects and rows for the OCEL output, and the moves for the broken promises.

A Prosimos engine learns each message's id without any change to the interface: the orchestrator
stamps the id on the very `Message` objects `step()` returned, and `deliver()` receives the stamped
copy.

A Prosimos engine collects its own warnings: Prosimos writes warnings to one list shared by all
engines (`warning_logger`), so `ProsimosEngine` hands Prosimos its own list for the duration of each
of its methods, and two engines' warnings never mix. Prosimos's end-of-run usage statistics
(`find_issues`, e.g. "element used in less than 1% of cases") aren't included.

## Limitations

- **Outside the interface.** Building an engine (from a BPMN file, a JSON file, a number of cases and
  the start time) and handing it a log writer happen outside the nine methods.
- **Quotes:** a quote is locked at its first shipment, so a modify or cancel after it is too late even for
  the portions that haven't moved yet; a quote type reaches one group only; a requester has one open quote per
  quote type. The estimate's approximations are in [messaging.md](messaging.md), "Limitations".
- **Catalogue.** See [catalogue.md](catalogue.md) for its limitations.
