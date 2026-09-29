# Multi-process orchestrator

`run_orchestrator` in `prosimos/orchestrator.py` simulates several processes side by side on one
shared clock. Each process is simulated by its own engine; the orchestrator repeatedly steps the
engine whose next event is earliest (ties broken by process name) and writes one merged log sorted
by start time.

## Engine interface

This section lists everything the orchestrator may ask of an engine. **Nothing else may cross that
boundary.** In code it is the `SimulationEngine` class in `prosimos/orchestrator.py`;
`ProsimosEngine` implements it for a Prosimos simulation (`SimBPMEnv`), and the orchestrator only
talks to engines through it.

### The rule

Each engine is a black box. A process may not read or change another process's data (its cases,
queues, resources, attributes, logs) by any route other than the five methods below. That includes
the orchestrator: it may not look inside an engine either. Anything that needs to pass between
processes goes out through `pending_publish()` and comes in through `inject(objects)`.

### The five methods

| Method              | Question it answers                          | Status          |
|---------------------|----------------------------------------------|-----------------|
| `next_event_time()` | When is your next event due?                 | **Implemented** |
| `step()`            | Perform exactly one event.                   | **Implemented** |
| `pending_publish()` | Did that event produce anything to send out? | Planned         |
| `blocked_on()`      | Is your next event waiting, and for what?    | Planned         |
| `inject(objects)`   | Here are the things you were waiting for.    | Planned         |

#### `next_event_time()`, implemented

Returns the date and time of the event the engine will perform on its next `step()`, or `None` when
the engine has nothing left to do. The first call also prepares the engine (it generates the
arrival times of all its cases); after that, asking changes nothing and asking twice gives the same
answer. `None` means genuinely finished: if work is parked in a batch waiting to fire, the answer is
that batch's time, not `None`.

Times are absolute dates and times, not "seconds since start", so answers from different engines can
be compared directly. All engines are given the same start time when they are built.

Each engine handles its events strictly in time order (events due at the same moment in the order
they were added), so this is always its earliest pending event and an engine never goes back in
time. Case priority rules therefore no longer decide which case is handled first; they only order
the cases inside a batch.

#### `step()`, implemented

Performs exactly one event and nothing more, and returns nothing. The orchestrator calls it only
after `next_event_time()` returned a time for this engine.

#### `pending_publish()`, planned

After a `step()`, returns whatever that event produced for other processes (for example an order the
warehouse must pack), or nothing. The orchestrator delivers it; the sending engine never addresses
another engine directly.

#### `blocked_on()`, planned

Says whether the engine's next event can't happen until something arrives from outside, and what it
is waiting for. A blocked engine is not stepped; `next_event_time()` alone can't express "I'm waiting".

#### `inject(objects)`, planned

Hands an engine the things it was waiting for. This is the only way anything from another process
enters an engine.

### What else crosses the boundary today

To be honest about where the current code stands, beyond the two implemented methods:

1. **Building an engine.** The orchestrator builds each engine (`ProsimosEngine(...)`) from a BPMN
   file, a JSON file, a number of cases and the shared start time. This happens once, before any of
   the five methods.
2. **The event log.** When it builds an engine, the orchestrator gives it a writer to send its log
   rows to. The engine hands its rows over after every step, so the orchestrator never reaches into
   the engine to collect them.
3. **Random numbers.** All engines draw from one shared random number generator, so a process's
   results depend on which other processes run beside it. This isn't a route to another process's
   data, but it means processes aren't fully independent yet. Fixing it needs one generator per
   engine, which is deliberately not in this sprint.

### Questions to agree on

1. Is building an engine and collecting its log part of this interface, or a separate one?
2. What is an "object": a case, a message, something else? What does it carry (at least a time, a
   type and the attributes the receiver needs)?
3. When an object arrives, can it start a new case in the receiving process, continue a waiting one,
   or both?
