# Messages in a process model

In a multi-process simulation ([orchestrator.md](orchestrator.md)), Prosimos processes exchange
messages: an order placed in Sales is announced to the warehouses, and a Sales case waits until its
shipment arrives. A Prosimos process model is a BPMN file plus a JSON settings file, and messages
use both:

- the **BPMN model** says *where* a case publishes a message or waits for one, with standard BPMN
  message events;
- the **JSON settings** say *what* is published or accepted there, in a `messages` section.

A model without a `messages` section behaves exactly as before: it publishes nothing and
subscribes to nothing.

## BPMN: where

| BPMN element                                        | Meaning                                   |
|-----------------------------------------------------|-------------------------------------------|
| intermediate message throw event, message end event | the case publishes a message here         |
| intermediate message catch event                    | the case waits here for a message         |
| message start event                                 | a message starts a new case               |

An event is a message event when it has a `messageEventDefinition`.

Intermediate throw events take no time: a case passes them instantly, whether they are message
events or have no definition at all (a milestone). They appear in the log only when intermediate
events are logged (`--is_event_added_to_log`). Signal, escalation, compensation and link throw
events are rejected when the model is loaded (`throw event <id> of kind <kind> is not supported`):
they stand for behaviour the simulator doesn't have, so silently passing them would change the
model.

## JSON: what

The Sales process of the [running example](running-example.md) places an order, announces it, and
waits for the shipment of that order (model: `testing_scripts/assets/running_example/`):

```
Start ──> Place order ──> (Throw_OrderPlaced) ──> (Catch_Shipment) ──> Close order ──> End
                              publishes               waits for
                              OrderPlaced             Shipment of this order
```

```json
"messages": {
  "publish": [
    {"event_id": "Throw_OrderPlaced", "type": "OrderPlaced",
     "attributes": ["case_id", "city"]}
  ],
  "consume": [
    {"event_id": "Catch_Shipment", "type": "Shipment",
     "condition": [[{"attribute": "order_id", "comparison": "=",
                     "case_attribute": "case_id"}]]}
  ]
}
```

**publish**: each entry names a publishing event, the message `type`, and the `attributes` the
message carries (optional, default none). Their values are copied from the case when it passes the
event. `case_id` is a reserved name for the case's identifier: the process name followed by the case
number, e.g. `Sales-7`, so orders get unique ids without a new case attribute.

**consume**: each entry names a catch event (or the start event, see "Processes started by
messages"), the message `type` it accepts, and an optional `condition`; without one, any message of
that type is accepted. The process subscribes to every type listed under `consume`.

**copy** (optional, on any `consume` entry) maps case attributes to message attributes: when the
entry accepts a message, the listed message attributes are written into the case.
`"copy": {"order_id": "case_id"}` sets the case's `order_id` to the message's `case_id`; `source`
can be copied too. A copied name counts as declared, so the process can publish it later (e.g.
`Shipment{order_id}`). The target can't be `case_id`, which is reserved.

- **At a start event**, the values go into the new case, on top of its drawn case attributes, before
  it starts: the Tartu warehouse's case gets `order_id = Sales-3` and later publishes
  `Shipment{order_id: Sales-3}`.
- **At a catch event**, they go into the waiting case that claims the message, at the time of the
  claim, replacing earlier values. Sales waiting for `Shipment{order_id, tracking_no}` with
  `"copy": {"tracking_no": "tracking_no"}` can then publish `OrderClosed{case_id, tracking_no}`.

Copied values are ordinary case attribute values from then on: later gateways, conditions,
published messages and the log see them. Every copy target gets a column in the event log, even when
it isn't a declared attribute (like the warehouse's `order_id`); the cell stays empty until a value
is copied. A message without one of the listed attributes
leaves that case attribute unchanged, with one warning per event and attribute.

**capacity** (optional, on a catch event's entry) says how many waiting cases one message resumes:
a fixed `{"value": 2}`, or `{"attribute": "capacity"}` to read it from the message. Without it, a
message resumes one case. For example, orders waiting at the dock and a truck taking up to its
capacity of them:

```json
{"event_id": "Catch_Truck", "type": "Truck",
 "condition": [[{"attribute": "dock", "comparison": "=", "value": "Tartu"}]],
 "capacity": {"attribute": "capacity"},
 "copy": {"truck_id": "case_id"}}
```

**collect** (optional, on a catch event's entry) says how many matching messages a case must claim
there before it continues: a fixed `{"value": 3}`, or `{"case_attribute": "items"}` to read it from the
case when it reaches the event. Without it, one message is enough. For example, an order waiting until
an `ItemReady` has come for each of its items:

```json
{"event_id": "Catch_Items", "type": "ItemReady",
 "condition": [[{"attribute": "order_id", "comparison": "=", "case_attribute": "case_id"}]],
 "collect": {"case_attribute": "items"}}
```

**count** (optional, on a `publish` entry) publishes several messages at once, one per object: a
fixed `{"value": 3}`, or `{"case_attribute": "items"}` to read it from the case when it passes the
event. Without it, the entry publishes one message. `index` is a reserved name for the message's
number, 1..N, and can only be listed on an entry with `count`; together with `case_id` it identifies
the object (`Sales-7` item 2), e.g. for the receiver to copy into its new case. For example, an order
announcing each of its items, so that Picking starts one case per item:

```json
{"event_id": "Throw_Items", "type": "ItemOrdered",
 "count": {"case_attribute": "items"},
 "attributes": ["case_id", "index"]}
```

This is how an object-centric event log records it: one event (e.g. Place order) linked to N new
objects. A loop publishing one message per round would put N extra events and a counter task into
the log, and would be discovered as a loop rather than as one event creating N objects. The fan-out
stays on the sending side: a start event still starts exactly one case per message.

One event may appear in several entries: under `publish`, passing it publishes one message per
entry (or `count` messages); under `consume`, a case waiting there accepts any of the entries' types. To wait for all of
them, use one catch event per message.

### Conditions

A condition uses the same format as Prosimos's branch rules: a list of alternatives, each a list of
terms. A message is accepted when all the terms of at least one alternative hold. Each term reads
"the message's `attribute` `comparison` X", where X is either a fixed `value` or an attribute of the
waiting case:

| Term                                                                        | Reads as                                              |
|-----------------------------------------------------------------------------|-------------------------------------------------------|
| `{"attribute": "source", "comparison": "=", "value": "TartuWarehouse"}`     | the message was published by TartuWarehouse           |
| `{"attribute": "weight", "comparison": "in", "value": [5, 10]}`              | the message's `weight` is between 5 and 10            |
| `{"attribute": "order_id", "comparison": "=", "case_attribute": "case_id"}` | the message's `order_id` is this case's id (correlation) |
| `{"attribute": "price", "comparison": "<=", "case_attribute": "budget"}`     | the message's `price` is at most this case's `budget` |

`attribute` is a message attribute, or `source` for the process that published the message.
Comparisons are those of branch rules: `=`, `!=`, `<`, `<=`, `>`, `>=` and `in` (with a fixed
`[low, high]`).

```json
"condition": [
  [ {"attribute": "order_id", "comparison": "=", "case_attribute": "case_id"},
    {"attribute": "status", "comparison": "!=", "value": "cancelled"} ],
  [ {"attribute": "source", "comparison": "=", "value": "HeadOffice"} ]
]
```

reads: (the shipment is for this case AND isn't cancelled) OR (it comes from HeadOffice).

## Validation

The section is checked when the model is loaded. An invalid section stops loading with an
`InvalidSimScenarioException` that names the entry, e.g.
`Invalid 'messages' section: messages.consume[0]: event_id 'Timer_Cancel' is an intermediateCatchEvent, expected an intermediate message catch event`.

- The section holds only `publish` and `consume`, each a list.
- Every `event_id` exists in the BPMN model.
- A `publish` event is an intermediate message throw event or a message end event.
- A `consume` event is an intermediate message catch event or a message start event.
- `type` is a non-empty string.
- `attributes` is a list of names, each `case_id`, `index` (only with `count`) or a declared case,
  global or event attribute.
- `count` is either `{"value": <whole number of at least 0>}` or `{"case_attribute": <declared
  attribute>}`, and only on a `publish` entry: only throw and end events publish, and a start event
  starts exactly one case per message.
- A message end event under `publish`, or a catch event under `consume`, has exactly one incoming
  arrow; otherwise: "draw an explicit gateway before <event id>". In BPMN, several arrows into one
  element mean "fire once per arriving token", but Prosimos joins several arrows into an end event
  as an OR join and into a catch event as an AND merge (see
  [engine-internals.md](engine-internals.md)). With an explicit gateway, the meaning is clear.
- After an event-based gateway with a branch that waits for a message (a race, see "Races at
  event-based gateways"), every branch is a message catch event or a timer, and no branch has a fixed
  `collect` of 0.
- `copy` maps case attribute names to message attribute names, and doesn't set `case_id`.
- `capacity` is either `{"value": <whole number of at least 1>}` or `{"attribute": <name>}`, and only
  on a catch event: a start message starts exactly one case.
- `collect` is either `{"value": <whole number of at least 0>}` or `{"case_attribute": <name>}`.
- `collect` is only on a catch event with exactly one `consume` entry: a case waiting there keeps one
  count, so it must be clear which messages it counts. To accept several variants of one type, use one
  entry with alternatives in its condition; to wait for several kinds, use one catch event per type.
- `collect` isn't on a start event: a start event turns one message into a new case, and before that
  case exists nothing holds the earlier messages or tells which ones belong together. Start on the
  first message instead, then collect the rest at a catch event right after the start (e.g. a picking
  batch: start on 1 order, then `"collect": {"value": 4}`). The rest is a fixed number, or a case
  attribute holding the remaining count, since a count can't do arithmetic.
- `collect` and `capacity` aren't on the same entry (see "Many messages for many cases").
- A process started by messages (a `consume` entry on its start event) has exactly one start event,
  and the start event's condition uses only fixed values and `source`, not `case_attribute`: there
  is no case yet to compare with.
- `condition` is a non-empty list of non-empty lists of terms.
- A term has `attribute`, a known `comparison`, and exactly one of `value` and `case_attribute`;
  `in` takes a fixed `value` `[low, high]`.

## Publishing

When a case passes an event listed under `publish`, the engine builds the message and returns it
from `step()` at the time the case passes the event. Several messages due at the same time (two
throw events in a row, one on each of two parallel branches, several cases at once) come out of one
step, ordered by case id, then by the order in which the case passed the events.

Attribute values are taken at that time. A declared attribute without a value yet for this case
(e.g. an event attribute of a task the case hasn't done) is sent as `None`, with one warning per
event and attribute in the engine's warnings.

An entry with `count` reads the number at that time too, and publishes that many messages at once, in
order of `index` 1..N. A count of 0 publishes nothing; a value that isn't a whole number of at least 0
(or no value) publishes one message, with one warning per event and attribute.

Publishing doesn't change the simulation itself: the log is the same with or without the `publish`
entries.

## Waiting

A case reaching a catch event listed under `consume` waits there until a matching message is
delivered, instead of getting a delay. It needs no `event_distribution` entry; one given anyway is
ignored, with a warning when the model is loaded (`duration of Catch_Shipment is ignored: it waits
for a message`). Catch events not listed under `consume` keep their delay drawn from
`event_distribution`. While a case waits, the process's other cases carry on.

Each message delivered to the process gets one of three answers:

- **`CLAIMED`**: it matches a waiting case's condition. That case continues from the catch event at
  the message's time. If several waiting cases match, the one waiting longest takes it; ties go to
  the lower case id. With a `capacity` above 1, the message then also resumes the next matching
  cases waiting at the same entry, longest-waiting first, up to the capacity, all at the message's
  time, and `copy` is applied to each. It doesn't wait to fill up: a truck that finds one order
  takes one, and a truck that finds none stays `PENDING` and takes the first order that starts
  waiting. A capacity read from the message that is missing or not a whole number of at least 1
  counts as 1, with one warning per event and attribute. Only if no waiting case takes it, a
  matching start event does: it starts a new case (see "Processes started by messages").
- **`DISCARDED`**: no case can ever accept it: it fails every condition on the message alone (fixed
  values, `source`), or the case it names through `case_id` doesn't exist in this process or has
  already finished. A case that just hasn't started yet still counts, and so, in a process started
  by messages, does a case number beyond the cases created so far (e.g. `Warehouse-5` when there are
  5): a start message may still create it. A start event whose condition fails never accepts it
  either.
- **`PENDING`**: otherwise, e.g. its case hasn't reached the catch event yet, or doesn't exist yet in
  a process started by messages (if it never appears, the message ends the run unclaimed). The orchestrator offers
  it again later. A condition on a case attribute other than `case_id` never causes a discard,
  because the attribute can still change.

A case reaches a catch event, and can claim a message, only at the time it really gets there, not
while the task before it is still running. If its message was already waiting in the orchestrator's
pool, it is claimed at once and the wait lasts zero time.

When intermediate events are logged, a catch event a case waited at appears as lasting from the
case's arrival at the event until the message.

**Collecting.** With `collect`, the number is read when the case reaches the catch event. Each matching
message is CLAIMED and bound to the case, and `copy` is applied at every claim (a later message
overwrites an earlier value); the case stays parked until it has the number it needs, then continues
at the time of the last one. Messages that came earlier wait in the orchestrator's pool and are
claimed as soon as the case parks, possibly several in one step. When several waiting cases match,
the one waiting longest that still needs messages gets it. A number of 0 means nothing to collect:
the case passes straight on. A case attribute that isn't a whole number of at least 0 counts as 1,
with one warning per event and attribute.

Cases still waiting when the run ends are reported as stalled cases ([orchestrator.md](orchestrator.md)),
with how many messages they had collected, e.g. `collected 2 of 3`.

## Many messages for many cases

`capacity` lets one message complete several cases (one → many), and `collect` lets several
messages complete one case (many → one). Both on one entry would mean several messages jointly
completing several cases in a single step. That is a design choice, not a missing feature: a model
expresses it as **two simple steps through an intermediate case**. The intermediate case collects the
messages (many → one), then publishes one message per target (one → many). For example, a truck
collects packages, then sends one Shipment per package.

The intermediate case is usually a real object (a truck, a package, a pallet). Making it explicit
puts it in the log with its own links, and keeps each step readable and discoverable.

## Races at event-based gateways

An event-based gateway whose branches include a catch event listed under `consume` is a race, e.g.
"the shipment or a 6-hour deadline, whichever comes first":

```
                 +--> (catch Shipment) --> Close order --+
Order placed --> <event-based gateway>                    <XOR> --> end
                 +--> (timer 6 h)      --> Cancel order --+
```

A race needs no new keys: the message branch is listed under `consume` like any catch event, and the timer
gets its delay from `event_distribution` (model: `testing_scripts/assets/messaging/sales_with_deadline`):

```json
"messages": {
  "consume": [
    {"event_id": "Catch_Shipment", "type": "Shipment",
     "condition": [[{"attribute": "order_id", "comparison": "=", "case_attribute": "case_id"}]]}
  ]
},
"event_distribution": [
  {"event_id": "Deadline", "distribution_name": "fix", "distribution_params": [{"value": 21600}]}
]
```

When a case reaches the gateway, every branch is armed at once: each timer is set to fire after its
delay (from `event_distribution`, as for any timer), and the case waits at the message branch as at
any catch event. The first to happen wins and the others are canceled:

- **The message comes first:** the case continues on the message branch at the message's time, and
  the timers never fire.
- **A timer fires first:** the case continues on that timer's branch, one microsecond after the
  timer's time, and stops waiting for the message, so a message for it that comes later finds a case
  that is no longer waiting (and is discarded once the case has finished). Only the timer's own log
  row shows its real time; what follows it (e.g. Cancel order) starts one microsecond later.
- **A message at exactly the timer's time** wins, even when it comes from another process that steps
  after this one at that time: that is what the microsecond is for (see
  [engine-internals.md](engine-internals.md)).

The other branches' tokens are removed, so the case continues on one branch only.

A race can have several message branches, with or without timers, e.g. "the quote is accepted or
rejected":

```
                 +--> (catch QuoteAccepted) --> Confirm order --+
Send quote --> <event-based gateway>                              <XOR> --> end
                 +--> (catch QuoteRejected) --> Archive quote --+
```

- **The first claim wins**, and the case stops waiting at the other branches, so a later message for
  them finds a case that is no longer waiting (and is discarded once the case has finished).
- **Two messages for different branches at the same instant:** whichever the orchestrator offers
  first wins; with the same seed, that order is always the same.
- **One message matching two branches** (the same type, with overlapping conditions, e.g. `Approval`
  with `decision = yes` on one branch and no condition on the other): the branch whose `consume` entry
  comes first in the JSON takes it, with one warning per gateway the first time it happens.

`capacity` and `collect` work on a race's message branches as on any catch event:

- **Capacity:** a truck that picks up several waiting orders wins the race of each one it takes (their
  timers never fire); orders that didn't fit keep waiting in their own races.
- **Collect: the complete set decides.** The race stays open until the case has claimed all the
  messages it needs; only then does the message branch win. If a timer fires first (e.g. after 1 of 3
  items), the timer wins.
- A branch with `collect` of 0 has nothing to wait for, so it wins its race immediately when the case
  reaches the gateway. A fixed `{"value": 0}` on a race branch is therefore rejected at load ("a race
  branch with a fixed collect of 0 always wins its race; remove the race or the branch"); a `collect`
  read from a case attribute can still be 0 for some cases. If several branches have nothing to
  collect, the one whose `consume` entry comes first in the JSON wins, with one warning per gateway
  the first time it happens.

A case still waiting when the run ends is reported once, as stalled at the gateway, with the message
types of all its branches. An event-based gateway without a branch waiting for a message works as
before: it is decided as soon as a case reaches it, by drawing a duration for each branch and taking
the shortest.

## Processes started by messages

A `consume` entry on the model's start event (a message start event) makes messages start the
process's cases, instead of an arrival schedule. The Tartu warehouse, for example, starts a case for
every order from Tartu or Tapa and keeps the order's id (model: `testing_scripts/assets/messaging/tartu_warehouse.*`):

```json
"consume": [
  {"event_id": "Start_Order", "type": "OrderPlaced",
   "condition": [[{"attribute": "city", "comparison": "=", "value": "Tartu"}],
                 [{"attribute": "city", "comparison": "=", "value": "Tapa"}]],
   "copy": {"order_id": "case_id"}}
]
```

A process is started either by its arrival schedule or by messages, never both:

- the model has exactly one start event;
- in the simulation configuration, the process has no `total_cases` (giving one is an error, so
  nobody thinks it limits anything), while every other process needs one;
- its JSON settings need no `arrival_time_distribution` or `arrival_time_calendar`; every other
  process needs an `arrival_time_distribution`.

Each message the start event accepts starts exactly one new case, at the time the message is
claimed, with its case attributes drawn as usual. A message that a waiting case also accepts goes to
the waiting case, not to the start event. A message the start condition rejects is discarded. Until
its first message, such a process has nothing to do, so the orchestrator never steps it, but it is
still offered every message it subscribes to.

## One verdict per item

An order can decide the fate of its items, e.g. ship them all or send them back, without remembering
them: each item is its own case, waiting for its order's verdict on it, and the order sends one
verdict per item with `count`. Each verdict is addressed to exactly one item, so none is shared or
used up by another item. Only the features above are needed (models:
`testing_scripts/assets/messaging/orders_with_verdicts` and `items_awaiting_verdict`):

```
Order (Sales):   Place order -> (throw ItemOrdered x items, index)
                 -> race: (collect ItemReady, items) -> (throw Shipped x items, index)
                        | (timer 2 days)           -> (throw Cancelled x items, index)
Item (Picking):  start ItemOrdered (copy order_id, index) -> Pick -> (throw ItemReady{order_id})
                 -> race: (catch Shipped,   case_id = order_id and index = index) -> done
                        | (catch Cancelled, case_id = order_id and index = index) -> Return to stock
```

- **The order** publishes `ItemOrdered` with `"count": {"case_attribute": "items"}` and attributes
  `case_id` and `index`; after its race, `Shipped` or `Cancelled` the same way, so verdict *i* carries
  the same `case_id` and `index` as item *i*'s `ItemOrdered`.
- **The item** starts on `ItemOrdered` and copies both into the case
  (`"copy": {"order_id": "case_id", "index": "index"}`). Its two catch events accept only the verdict for
  this order and this item:

  ```json
  "condition": [[{"attribute": "case_id", "comparison": "=", "case_attribute": "order_id"},
                 {"attribute": "index", "comparison": "=", "case_attribute": "index"}]]
  ```

- **Late items:** an item still being picked when its order is canceled isn't waiting yet, so its
  `Cancelled` is pending: it stays in the pool and is claimed as soon as the item reaches its race. The
  items already waiting are released at once.
- An `ItemReady` sent after the order was canceled finds no waiting order and is discarded once the
  order has finished, with the usual "discarded by every recipient" warning; that is expected.

## Limitations

- **Races:**
  - the branches after the gateway can only be message catch events and timers (no signal,
    conditional or other catch events, and no receive tasks);
  - with `collect`, the messages a case claimed before a timer won stay bound to it: they count as
    claims and aren't offered to anyone else (to send items back, see "One verdict per item"), and
    later ones for it are discarded;
  - a message for a branch the case no longer waits at is pending, not discarded, while the case is
    still running and the condition is on `case_id`, since the case can still change; it is discarded
    only when offered after the case has finished;
  - a message matching several branches of one race goes to one of them, the one listed first, not to
    all.
- **One end event per model.** Prosimos supports only one end event in a model ("Temporarily not
  supporting multiple end events"), so a model can't have a message end event next to another end
  event. Instead, publish with an intermediate message throw event on that branch, then merge the
  branches with an explicit XOR gateway into the single end event.
- **`index` is reserved for `count`.** A publish entry can list `index` only with `count`, where it is the
  message's number, so a case can't pass on a number it was given under that name (e.g. an item publishing
  the `index` it copied from `ItemOrdered`). Copy it into a case attribute with another name, e.g.
  `"copy": {"item_index": "index"}`, and publish that.
- **`case_attribute` names** in conditions aren't checked against the declared attributes; a term
  naming an attribute the case doesn't have is simply false.
- **Capacity is per `consume` entry.** One message resumes extra cases only at the entry that resumed
  the first case, so a truck accepted at two different catch events doesn't load cases from both. To
  let one truck serve several kinds of waiting cases, use one catch event whose condition accepts all
  of them.
