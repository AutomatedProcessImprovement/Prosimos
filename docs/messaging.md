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

**consume**: each entry names a catch event, the message `type` it accepts, and an optional
`condition`; without one, any message of that type is accepted. The process subscribes to every
type listed under `consume`.

One event may appear in several entries: under `publish`, passing it publishes one message per
entry; under `consume`, a case waiting there accepts any of the entries' types. To wait for all of
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
- A `consume` event is an intermediate message catch event.
- `type` is a non-empty string.
- `attributes` is a list of names, each `case_id` or a declared case, global or event attribute.
- A message end event under `publish`, or a catch event under `consume`, has exactly one incoming
  arrow; otherwise: "draw an explicit gateway before <event id>". In BPMN, several arrows into one
  element mean "fire once per arriving token", but Prosimos joins several arrows into an end event
  as an OR join and into a catch event as an AND merge (see
  [engine-internals.md](engine-internals.md)). With an explicit gateway, the meaning is clear.
- A catch event under `consume` doesn't directly follow an event-based gateway (see "Limitations").
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
  the message's time; one message resumes one case. If several waiting cases match, the one waiting
  longest takes it; ties go to the lower case id.
- **`DISCARDED`**: no case can ever accept it: it fails every condition on the message alone (fixed
  values, `source`), or the case it names through `case_id` doesn't exist in this process or has
  already finished. A case that just hasn't started yet still counts.
- **`PENDING`**: otherwise, e.g. its case hasn't reached the catch event yet. The orchestrator offers
  it again later. A condition on a case attribute other than `case_id` never causes a discard,
  because the attribute can still change.

A case reaches a catch event, and can claim a message, only at the time it really gets there, not
while the task before it is still running. If its message was already waiting in the orchestrator's
pool, it is claimed at once and the wait lasts zero time.

When intermediate events are logged, a catch event a case waited at appears as lasting from the
case's arrival at the event until the message.

Cases still waiting when the run ends are reported as stalled cases ([orchestrator.md](orchestrator.md)).

## Limitations

- **Message start events** (a message starting a new case) aren't supported yet.
- **Event-based gateways.** A catch event listed under `consume` can't directly follow an
  event-based gateway, so a race such as "the shipment or a timeout, whichever comes first" can't be
  modelled yet; such a model is rejected when it is loaded. Prosimos decides an event-based gateway
  as soon as a case reaches it, by drawing a duration for each event after it, and a waiting catch
  event has no duration.
- **One end event per model.** Prosimos supports only one end event in a model ("Temporarily not
  supporting multiple end events"), so a model can't have a message end event next to another end
  event. Instead, publish with an intermediate message throw event on that branch, then merge the
  branches with an explicit XOR gateway into the single end event.
- **`case_attribute` names** in conditions aren't checked against the declared attributes; a term
  naming an attribute the case doesn't have is simply false.
