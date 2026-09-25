# Orchestrator: event ordering limitation

Status: known, deliberately left as is. Revisit before processes can send each other messages.

## Summary

The orchestrator (`prosimos/orchestrator.py`) runs several process simulations side by side. It
repeatedly asks every engine "when is your next event?" (`SimBPMEnv.next_event_time()`) and steps
the engine with the earliest answer. This assumes each engine works through its events in time
order. It doesn't always:

- **Timers and other intermediate events jump ahead of every task**, whatever their time.
- **Case prioritisation rules make important cases' tasks jump ahead** of less important cases'
  tasks, whatever their time.

So an engine can say "my next event is at 14:00", handle that, and then handle something due at
9:00. `next_event_time()` returns the time of the item at the head of the engine's to-do list, not
the earliest time on it.

## Why it's harmless today

Each engine still produces exactly the same results as it would when run on its own. The dates
and times in the log are correct; only the order in which the engine works them out jumps around.
With no messages between processes, engines can't affect each other, so this doesn't matter. The
merged log is sorted by start time before it's written, so its rows come out in order too.

## Why it will matter once messages exist

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

## Where the behaviour comes from

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

## Options considered

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

## Related, not verified

Because a worker can only be booked forward, the priority shortcut probably has a similar effect
inside a single engine: if an important case's 14:00 task takes a shared worker first, a regular
9:00 task can't use the idle morning. This comes from reading the code; it hasn't been confirmed
with a run.
