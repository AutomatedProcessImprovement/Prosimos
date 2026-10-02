# How messages appear in a process model

A Prosimos process model is a BPMN file plus a JSON settings file. Messages use both:

- the **BPMN model** says *where* a case publishes a message or waits for one, with standard BPMN
  message events;
- the **JSON settings** say *what* is published or accepted there, in a `messages` section.

This is internal to the Prosimos engine. The orchestrator only sees what the engine interface
exposes (`subscriptions()`, the messages `step()` returns, the verdicts `deliver()` gives; see
[orchestrator.md](orchestrator.md)), so none of this is part of the protocol document.

## BPMN: where

| BPMN element                                        | Meaning                                   |
|-----------------------------------------------------|-------------------------------------------|
| intermediate message throw event, message end event | the case publishes a message here         |
| intermediate message catch event                    | the case waits here for a message         |
| message start event                                 | a message starts a new case (future work) |

An event is a message event when it has a `messageEventDefinition`. A throw event without one, or
a catch event with a timer definition, can't be used for messages.

A case passes an intermediate throw event instantly, like a gateway, whether it is a message
event or has no definition at all (a milestone). It appears in the log only when intermediate
events are logged (`is_event_added_to_log`). Signal, escalation, compensation and link throw events
are rejected when the model is loaded (`throw event <id> of kind <kind> is not supported`): they
mean something the simulator doesn't do, so passing them silently would change the model.

## JSON: what

The Sales process of the [running example](running-example.md): a case places an order, announces
it, and waits for the shipment of that order.

```
Start ──> Place order ──> (Throw_OrderPlaced) ──> (Catch_Shipment) ──> End
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
message carries (optional, default none). Their values are copied from the case's current
values when the case reaches the event. `case_id` is a reserved name for the engine's own case
identifier, e.g. `Sales-7`, so orders get unique ids without adding a new case attribute.

**consume**: each entry names a catch event, the message `type` it accepts, and an optional
`condition`. Without a condition, any message of that type is accepted. The condition uses the
same format as Prosimos's branch rules: a list of alternatives, each a list of terms. A message
is accepted when all the terms of at least one alternative hold. A term
compares a message attribute, or `source` (the process that published the message), with either:

- a fixed `value`: `{"attribute": "source", "comparison": "=", "value": "TartuWarehouse"}`
- a `case_attribute` of the waiting case (correlation):
  `{"attribute": "order_id", "comparison": "=", "case_attribute": "case_id"}`

In the example, a Sales case accepts only the `Shipment` whose `order_id` is its own case id.

`subscriptions()` returns the distinct types under `consume`, here `["Shipment"]`
(`MessagingModel.subscriptions()`; `ProsimosEngine` starts using it once it can consume, see below).

## Validation

The section is checked when the model is loaded (`SimDiffSetup`, code in
`prosimos/messaging_parser.py`). An invalid section stops loading with an
`InvalidSimScenarioException` that names the entry, e.g.
`Invalid 'messages' section: messages.consume[0]: event_id 'Timer_Cancel' is an intermediateCatchEvent, expected an intermediate message catch event`.

| Rule                                                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| the section holds only `publish` and `consume`, each a list                                                                                                                                |
| every `event_id` exists in the BPMN model                                                                                                                                                  |
| a `publish` event is an intermediate message throw event or a message end event                                                                                                            |
| a `consume` event is an intermediate message catch event (message start events: future work)                                                                                               |
| `type` is a non-empty string                                                                                                                                                               |
| `attributes` is a list of names, each `case_id` or a declared case, global or event attribute                                                                                              |
| a message end event under `publish`, or a catch event under `consume`, has exactly one incoming arrow ("draw an explicit gateway before <event id>")                                       |
| a catch event under `consume` doesn't follow an event-based gateway (see "Waiting"; test in `testing_scripts/test_waiting.py`)                                                             |
| `condition` is a non-empty list of non-empty lists of terms                                                                                                                                |
| a term has `attribute`, a `comparison` branch rules know (`=`, `!=`, `<`, `<=`, `>`, `>=`, `in`) and exactly one of `value` and `case_attribute`; `in` takes a fixed `value` `[low, high]` |

Each rule has a test in `testing_scripts/test_messaging_parser.py`, unless noted.

Why one incoming arrow: in BPMN, several arrows into one element mean "fire once per arriving
token", but Prosimos joins several arrows into an end event as an OR join (once per case) and into
a catch event as an AND merge (waits for all); see "Implicit merges" in
[process-engine-notes.md](process-engine-notes.md). Requiring an explicit gateway means the
question never comes up for messages, without changing existing models. Throw events need no such
rule: they get a hidden XOR join, which matches the standard.

A model without a `messages` section behaves exactly as before: it publishes nothing and
subscribes to nothing.

## Publishing

When a case passes an event listed under `publish`, the engine returns a message from `step()` at
the time the case really passes the event, not earlier:

- **Why holding is needed.** Prosimos executes a task in one go when it leaves the queue at its
  enabled time, and moves the case on right away, so the case passes the throw event after a task
  during the step at the task's enabled time, while the event's real time is the task's end. Events
  right after the start are passed even earlier, while arrivals are generated. So passing a
  publishing event is *held* with its due time (`SimBPMEnv._held`), and the message is built and
  returned when it is due. The control flow is untouched, so logs are identical with or without a
  `messages` section.
- **`next_event_time()`** is the earlier of the next queued event and the earliest held item.
- **`step()`** releases all held items due at that time instead of running an event. If a held item
  and the next event are due at the same time, the held item goes first. An engine that still holds
  items isn't finished, even when its queue is empty.
- **Order.** Messages returned together are ordered by time, then case id, then the order in which
  the case passed the events (two throw events in a row, one on each of two parallel branches).
- **Values.** The attributes are copied from the case's values when the message is released, i.e.
  at its real time. `case_id` is the process name from the simulation configuration plus the case
  number, e.g. `Sales-7`. A declared attribute that has no value yet for this case (e.g. an event
  attribute of a task the case hasn't done) is sent as `None`, with one warning per event and
  attribute, among the engine's warnings (reported by `finish()`, see
  [orchestrator.md](orchestrator.md)).

Holding is used only for publishing: catch events are already queued at the time the case
reaches them, so waiting needs no holding (see "Waiting"). Tested with the Sales model in isolation
(`testing_scripts/test_publishing.py`, models in `testing_scripts/assets/messaging/`).

## Waiting

A case reaching a catch event listed under `consume` waits there for a message instead of getting
a delay. `subscriptions()` returns the types under `consume`.

- **Where it parks.** The case is moved onto the catch event during the step of the task before
  it, but the catch event is queued at the time the case really reaches it. The case parks when that
  queued event is executed, so it never waits (or claims a message) before it is there. Its token
  stays on the event's incoming flow, and the engine records (case, event, time it reached it).
- **`deliver(message, now)`** checks the message against the waiting cases:
  - **`CLAIMED`**: it matches a waiting case's condition. The catch event is queued for that case
    at `now` and the case continues from there; one message resumes one case. If several waiting
    cases match, the one waiting longest claims it; ties by case id.
  - **`DISCARDED`**: no case can ever accept it: it fails every condition on the message alone
    (fixed values, `source`), or the case it correlates with through `case_id` doesn't exist or has
    finished.
  - **`PENDING`**: otherwise, e.g. its case hasn't reached the catch event yet. Other case
    attributes can still change, so a condition on them alone never makes a discard.
- **Conditions** use the branch-rule comparisons. `case_attribute: "case_id"` resolves to the same
  `<process name>-<n>` as when publishing (both use `SimBPMEnv.case_id`).
- **No duration.** A waiting catch event needs no `event_distribution` entry; one given anyway is
  ignored, with a warning when the model is loaded (`duration of Catch_Shipment is ignored: it waits
  for a message`). Catch events not listed under `consume` keep their drawn delay.
- **After an event-based gateway** a catch event can't wait yet, so such a model is rejected when
  it is loaded (`... follows an event-based gateway, where waiting for a message isn't supported
  yet`). Prosimos decides an event-based gateway the moment a case reaches it: it draws a duration
  for every event after the gateway and takes the shortest (`get_event_gateway_choice`). A waiting
  catch event has no duration, so without this rule the run would crash (no `event_distribution`
  entry) or treat the message like a timer (an entry given anyway). A real race, where the case
  waits and the first of a message or a timer wins, is future work. This restriction was added while
  implementing waiting and is still to be agreed.
- **An engine whose cases all wait** returns `next_event_time() = None`: it has nothing to do now,
  but it isn't finished. The orchestrator still offers it every new message, and a claim queues an
  event, so it steps again.

Tested in `testing_scripts/test_waiting.py` (models in `testing_scripts/assets/messaging/`).

## Not done yet

- A catch event racing other events after an event-based gateway (see above).
- Names under `case_attribute` in conditions aren't checked against the declared attributes, and
  one event may appear in more than one entry.
