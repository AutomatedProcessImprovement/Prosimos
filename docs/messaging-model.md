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

| Rule                                                                                                 |
|------------------------------------------------------------------------------------------------------|
| the section holds only `publish` and `consume`, each a list                                          |
| every `event_id` exists in the BPMN model                                                            |
| a `publish` event is an intermediate message throw event or a message end event                      |
| a `consume` event is an intermediate message catch event (message start events: future work)   |
| `type` is a non-empty string                                                                         |
| `attributes` is a list of names                                                                      |
| `condition` is a non-empty list of non-empty lists of terms                                          |
| a term has `attribute`, a `comparison` branch rules know (`=`, `!=`, `<`, `<=`, `>`, `>=`, `in`) and exactly one of `value` and `case_attribute`; `in` takes a fixed `value` `[low, high]` |

Each rule has a test in `testing_scripts/test_messaging_parser.py`.

A model without a `messages` section behaves exactly as before: it publishes nothing and
subscribes to nothing.

## Not done yet

- Only parsing and validation are done: cases don't publish or wait for messages yet, and
  conditions aren't evaluated yet. `ProsimosEngine` still subscribes to nothing: subscribing
  now would get it offered messages it can only discard, and a discard is permanent.
- The simulator's BPMN reader doesn't know intermediate throw events yet: it keeps them only as
  nodes of unknown type. They need handling before a case can pass through one.
- Names under `attributes` and `case_attribute` aren't checked against the model's case
  attributes, and one event may appear in more than one entry.
