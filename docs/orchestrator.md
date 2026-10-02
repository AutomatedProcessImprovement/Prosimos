# Multi-process orchestrator

`run_orchestrator` in `prosimos/orchestrator.py` simulates several processes side by side on one
shared clock. Each process is simulated by its own engine; the orchestrator repeatedly steps the
engine whose next event is earliest (ties broken by process name), routes the messages engines
publish to the processes that consume them, and writes one merged log sorted by start time. The loop
itself is `run_engines`, which works on any engines implementing the interface below; the tests run
it with scripted fake engines (see `running-example.md`).

## Simulation configuration

`run_orchestrator` takes a `SimulationConfig`: the processes (a name, BPMN file, JSON parameters and
number of cases each), an optional seed, the shared start time and the consumer groups. It can be
built in code or loaded with `SimulationConfig.from_json(path)`:

```json
{
  "processes": [
    {"name": "Sales", "bpmn_path": "sales.bpmn", "json_path": "sales.json", "total_cases": 100},
    {"name": "TartuWarehouse", "bpmn_path": "tartu.bpmn", "json_path": "tartu.json", "total_cases": 0},
    {"name": "TallinnWarehouse", "bpmn_path": "tallinn.bpmn", "json_path": "tallinn.json", "total_cases": 0}
  ],
  "seed": 42,
  "start_time": "2024-01-01T09:00:00+00:00",
  "consumer_groups": {
    "Sales": ["Sales"],
    "Warehouses": ["TartuWarehouse", "TallinnWarehouse"]
  }
}
```

File paths are relative to the configuration file's folder. `seed` is optional, and a start time
without a time zone is taken as UTC. Without `consumer_groups`, every process is its own group. A
configuration is rejected if process names repeat, a process is in no group or in more than one, or
a group names a process that doesn't exist.

A `Message` has a `type` and `attributes`, set by the publishing process, and an `id`, `source` and
`time`, set by the orchestrator when it stamps the message.

## Engine interface

This section lists everything the orchestrator may ask of an engine. **Nothing else may cross that
boundary.** It follows the Orchestrator Protocol (kept outside this repo). In code, it is the
`SimulationEngine` class in `prosimos/orchestrator.py`; `ProsimosEngine` implements it for a Prosimos
simulation (`SimBPMEnv`), and the orchestrator only talks to engines through it.

### The rule

Each engine is a black box. A process may not read or change another process's data (its cases,
queues, resources, attributes, logs) by any route other than the five methods below. That includes
the orchestrator: it may not look inside an engine either. Anything that needs to pass between
processes is a message: it leaves an engine as the result of `step()` and enters another through
`deliver()`. Engines never call the orchestrator; they only answer its calls.

### The five methods

| Method                  | Returns                             | When                              |
|-------------------------|-------------------------------------|-----------------------------------|
| `subscriptions()`       | list of message types               | once, at setup                    |
| `next_event_time()`     | date and time, or `None`            | before every step                 |
| `step()`                | list of published messages          | to perform one event              |
| `deliver(message, now)` | `CLAIMED`, `DISCARDED` or `PENDING` | to offer one message to an engine |
| `finish()`              | an `EngineReport`                   | once, after the loop stops        |

The first four are the messaging calls. `finish()` is a lifecycle call: it publishes and delivers
nothing, so it doesn't change how processes interact.

`ProsimosEngine` implements all five. It publishes the messages its model lists under `publish`,
subscribes to the types under `consume`, and lets cases wait at those catch events until `deliver()`
resumes them ([messaging-model.md](messaging-model.md)).

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
A Prosimos engine may instead release messages it held until they were due, without performing an
event (see "Publishing" in [messaging-model.md](messaging-model.md)); `next_event_time()` announces
that step like any other.
This is the only way to publish messages. The orchestrator calls it only after `next_event_time()`
returned a time for this engine. The sending engine never addresses another engine directly; the
orchestrator routes its messages.

#### `deliver(message, now)`

Offers the engine one pending message at time `now` and returns a `Verdict`:

- **`CLAIMED`** is a commitment: the message is already bound to one case, taking effect at `now`,
  and the orchestrator never offers this copy to anyone again (other groups keep their own copies).
- **`DISCARDED`** is permanent: the engine will never want this message, even after its state
  changes, so the orchestrator never offers it to this engine again.
- **`PENDING`**: not now; the message may be offered again later. An engine that isn't sure must
  answer `PENDING`.

This is the only way anything from another process enters an engine.

#### `finish()`

Called once on every engine, after the loop stops (no engine has a next event). It returns an
`EngineReport`:

- **`stalled`**: the cases still waiting for a message, each as a `StalledCase` with its case id
  (e.g. `Sales-1`), the catch event, the message types it waits for (a list, usually of one: an event
  listed several times under `consume` accepts any of its types), and since when it waits.
- **`warnings`**: the warnings the engine raised during the run, including while it was built.

The orchestrator adds both to the `RunReport`, tagged with the engine's process name.

Prosimos writes its warnings to one list shared by all engines (`warning_logger`, used from 5
modules), without process names. `ProsimosEngine` therefore hands Prosimos its own list at the start
of each of its methods, including building it, and puts the shared one back at the end, so two
engines' warnings never mix and Prosimos itself is unchanged. The swap lives in the engine rather
than the orchestrator, so the orchestrator stays independent of Prosimos and engines of other kinds
need nothing. It only collects warnings raised during the run: Prosimos's own end-of-run usage
statistics (`find_issues`, e.g. "element used in less than 1% of cases") are left out.

The protocol originally defined this as `deliver(msgs, now)`, returning lists of claimed and
discarded ids. The orchestrator always offers one copy at a time, because a claim by one member must
stop the offer to the others, so a single message and a single answer carry the same information,
and an engine can no longer give contradictory answers. It is the same per-message acknowledgement
that brokers such as RabbitMQ use (ack, reject, requeue).

### What else crosses the boundary today

To be honest about where the current code stands, beyond the five methods:

1. **Building an engine.** The orchestrator builds each engine (`ProsimosEngine(...)`) from the
   configuration: a BPMN file, a JSON file, a number of cases and the shared start time. This
   happens once, before any of the five methods.
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

## Message routing

`run_engines` implements the orchestrator loop from the protocol:

- **Routing table**, built once at setup: for each message type, the consumer groups with at least
  one member subscribed to it.
- **Stamping**: every message returned by `step()` gets an id (`m1`, `m2`, ...), its source process
  and the step's time.
- **Pool**: one copy of each message per subscribed group. A copy remembers its group and which
  members have discarded it. A message type nobody subscribes to gets no copy and a warning.
- **Offers**: after each step, the orchestrator offers the new copies, then the pooled copies of the
  stepping engine's group, to the group's members in random order at the step's time, until one
  claims it. A member that discarded a copy is never offered it again. A copy leaves the pool when
  a member claims it, or when every member subscribed to its type has discarded it.
- **Randomness**: the member order comes from the orchestrator's own generator, seeded from the
  simulation seed, never from the global `random` module the engines draw from, so choosing a
  warehouse can't shift the engines' random draws.
- **Report**: `run_engines` and `run_orchestrator` return a `RunReport` with every step, published
  message, copy, claim and discard, the copies still in the pool at the end (unclaimed messages),
  discard counts per message type and process, the orchestrator's warnings, and, from each engine's
  `finish()`, its stalled cases (`stalled`) and warnings (`engine_warnings`), as (process, ...) pairs.

The orchestrator's own warnings (`warnings`) are collected in the report as they occur, not printed:
a message type nobody subscribes to, and a message whose every recipient discarded it without anyone claiming a copy. The second applies
even when the discard is expected, for example a shipment for an order that was canceled.

Copies are offered only to the members that subscribe to the message's type. In the protocol's
example every member of a shared group subscribes to the same types, so this only matters for groups
whose members consume different messages.
