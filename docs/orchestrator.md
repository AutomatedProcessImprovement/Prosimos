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

One known limitation: this is the time of the event the engine will do *next*, which is not always
its *earliest* pending event. Timers and case priority rules can jump ahead of earlier tasks. This
does no harm while processes don't exchange anything, but it must be solved before messages exist.
See "Known limitation: event ordering" below.

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
4. Is it acceptable that `next_event_time()` isn't always the earliest pending event, until
   messages are designed?

## Known limitation: event ordering

Status: known, deliberately left as is. Revisit before processes can send each other messages.

The orchestrator repeatedly asks every engine "when is your next event?" (`next_event_time()`) and
steps the engine with the earliest answer. This assumes each engine works through its events in time
order. It doesn't always:

- **Timers and other intermediate events jump ahead of every task**, whatever their time.
- **Case prioritisation rules make important cases' tasks jump ahead** of less important cases'
  tasks, whatever their time.

So an engine can say "my next event is at 14:00", handle that, and then handle something due at
9:00. `next_event_time()` returns the time of the item at the head of the engine's to-do list, not
the earliest time on it.

### Why it's harmless today

Each engine still produces exactly the same results as it would when run on its own. The dates
and times in the log are correct; only the order in which the engine works them out jumps around.
With no messages between processes, engines can't affect each other, so this doesn't matter. The
merged log is sorted by start time before it's written, so its rows come out in order too.

### Why it will matter once messages exist

Example: a shop process sends a message to a warehouse process for every order, starting a
packing case there. The warehouse also has other packing jobs at 10:00, 11:00 and 12:00.

1. The shop has a VIP order at 14:00 and a regular order at 9:00. The VIP order is at the head of
   its list, so it reports "14:00". The warehouse reports "10:00".
2. The orchestrator runs the warehouse's 10:00, 11:00 and 12:00 jobs, since they're earlier than
   14:00. The packer is booked through them.
3. The shop handles the VIP order, then the regular 9:00 order, which sends "new packing case at
   9:15".
4. The warehouse can't fit the 9:15 case into the morning. Each worker only keeps a single "free
   again at" time and a task starts at the later of that and its own ready time
   (`r_avail_at = max(c_event.enabled_at, r_avail_at)` in `SimBPMEnv.execute_task`), so the order
   is packed after the 12:00 job even though the packer was idle at 9:15.

The regular order waits hours for no reason, and the result depends on how the orchestrator
happened to interleave the engines. The warehouse only gives correct results if its work arrives
in time order.

If the warehouse has no earlier work of its own when the late message arrives, nothing breaks: it
simply handles the 9:15 case before the 14:15 one.

### Where the behaviour comes from

The engine's to-do list (`EventQueue` in `prosimos/simulation_queues_ds.py`) is ordered by
`(priority, enabled time)`, priority first. This came with the case prioritisation feature
(issue #43, January 2023). A task claims a worker the moment it's taken off the list, so taking
important cases off first is how they get workers first.

Cases with no matching priority rule get the lowest priority (`sys.maxsize`). Intermediate events
were meant to be unaffected by prioritisation, which was done by giving them the highest
priority, 0 (commit `e103f26`: "Event is executed out of the scope of prioritisation and have the
highest priority (0)"; see `SimBPMEnv.calc_priority_and_append_to_queue`). The side effect is
that they overtake every task.

The shortcut is safe inside one engine because Prosimos creates every case at the start of the
run, so a single engine knows everything that will ever happen in it. Message-triggered cases
break that assumption: they only appear when a message arrives.

### Options considered

- **Report the earliest queued time from `next_event_time()` instead of the head's time.**
  Doesn't help: `step()` would still run the head, so the orchestrator would choose an engine
  based on one event and run a different, later one.
- **Remove only the timer shortcut.** Timers go back to their place in time, but priority rules
  still reorder time, and single-run results for models with timers and priorities would change.
- **Pass priorities on with messages.** Good for keeping a VIP a VIP end to end, but it doesn't fix
  the timing problem: the late order in the example isn't unimportant, it just arrives too late.
  Better to pass the case's attributes (e.g. `client_type`) and let each process apply its own
  priority rules.
- **Handle the to-do list in time order, and use priority only to decide which waiting task gets
  a free worker.** This is how most simulators handle priorities and would fix the problem. It's a
  real change to the engine, and single-run results with priority rules would change, so it needs
  checking against the existing prioritisation tests.

Suggested direction when messages are designed: pass case attributes with messages, and move the
engine to time-ordered handling with priority only deciding who gets a free worker.

### Related, not verified

Because a worker can only be booked forward, the priority shortcut probably has a similar effect
inside a single engine: if an important case's 14:00 task takes a shared worker first, a regular
9:00 task can't use the idle morning. This comes from reading the code; it hasn't been confirmed
with a run.
