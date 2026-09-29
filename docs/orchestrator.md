# Multi-process orchestrator

`run_orchestrator` in `prosimos/orchestrator.py` simulates several processes side by side on one
shared clock. Each process is simulated by its own engine; the orchestrator repeatedly steps the
engine whose next event is earliest (ties broken by process name) and writes one merged log sorted
by start time.

## Engine interface

This section lists everything the orchestrator may ask of an engine. **Nothing else may cross that
boundary.** It follows the Orchestrator Protocol (kept outside this repo). In code, it is the
`SimulationEngine` class in `prosimos/orchestrator.py`; `ProsimosEngine` implements it for a Prosimos
simulation (`SimBPMEnv`), and the orchestrator only talks to engines through it.

### The rule

Each engine is a black box. A process may not read or change another process's data (its cases,
queues, resources, attributes, logs) by any route other than the four methods below. That includes
the orchestrator: it may not look inside an engine either. Anything that needs to pass between
processes is a message: it leaves an engine as the result of `step()` and enters another through
`deliver()`. Engines never call the orchestrator; they only answer its calls.

### The four methods

| Method               | Returns                        | When                          |
|----------------------|--------------------------------|-------------------------------|
| `subscriptions()`    | list of message types          | once, at setup                |
| `next_event_time()`  | date and time, or `None`       | before every step             |
| `step()`             | list of published messages     | to perform one event          |
| `deliver(msgs, now)` | `(claimed ids, discarded ids)` | to hand messages to an engine |

`ProsimosEngine` implements all four, but doesn't exchange messages yet: it subscribes to nothing,
`step()` always returns an empty list, and `deliver()` claims and discards nothing. The orchestrator
doesn't route messages yet either.

#### `subscriptions()`

The message types this process consumes, read from its model configuration. The orchestrator asks
once, when the engine is set up.

#### `next_event_time()`

Returns the date and time of the event the engine will perform on its next `step()`, or `None` when
there is nothing to do right now. Cases waiting for a message don't count, so `None` doesn't mean the
engine is finished: a delivered message can give it work again. Work parked in a batch waiting to
fire does count: the answer is that batch's time. The first call also prepares the engine (it
generates the arrival times of all its cases); after that, asking changes nothing and asking twice
gives the same answer.

Times are absolute dates and times, not "seconds since start", so answers from different engines can
be compared directly. All engines are given the same start time when they are built.

Each engine handles its events strictly in time order (events due at the same moment in the order
they were added), so this is always its earliest pending event and an engine never goes back in
time. Case priority rules therefore no longer decide which case is handled first; they only order
the cases inside a batch.

#### `step()`

Performs exactly one event and returns the messages that event published, an empty list if none.
This is the only way to publish messages. The orchestrator calls it only after `next_event_time()`
returned a time for this engine. The sending engine never addresses another engine directly; the
orchestrator routes its messages.

#### `deliver(msgs, now)`

Offers the engine its pending messages at time `now` and returns two lists of message ids:
**claimed** (used by a case, taking effect at `now`) and **discarded** (messages it will never use).
Every other message stays pending and is offered again later. This is the only way anything from
another process enters an engine.

### What else crosses the boundary today

To be honest about where the current code stands, beyond the four methods:

1. **Building an engine.** The orchestrator builds each engine (`ProsimosEngine(...)`) from a BPMN
   file, a JSON file, a number of cases and the shared start time. This happens once, before any of
   the four methods.
2. **The event log.** When it builds an engine, the orchestrator gives it a writer to send its log
   rows to. The engine hands its rows over after every step, so the orchestrator never reaches into
   the engine to collect them.
3. **Random numbers.** All engines draw from one shared random number generator, so a process's
   results depend on which other processes run beside it. This isn't a route to another process's
   data, but it means processes aren't fully independent yet. The protocol calls for one generator
   per engine, derived from the seed and the process name; that isn't implemented yet.

### Questions to agree on

1. Is building an engine and collecting its log part of this interface, or a separate one? The
   protocol doesn't cover either.
