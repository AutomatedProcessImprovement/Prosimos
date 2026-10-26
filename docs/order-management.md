# Order Management example

An example shaped like the OCEL 2.0 **Order Management** log (2,000 orders, 7,659 items, 1,128
packages), to show the multi-process features working together on a realistic model and to compare counts
and shares with the real log. Its task durations follow the rules of the program that generated the log
([Durations](#durations)). The models are in `testing_scripts/assets/order_management/`; the tests are in
`testing_scripts/test_order_management.py`.

It uses most messaging features ([messaging.md](messaging.md)): one message per object with `count` and
`index`, processes started by messages, `copy`, `append` and `aggregate`, and `collect` with `same`; and the
catalogue ([catalogue.md](catalogue.md)) for its customers and products.

## The processes

![Process models of the Order Management example](order-management-processes.svg)

```
Sales (order)        arrivals: weekdays 6:00-22:00, about one every 4.3 h, 2,000 orders
                     object attributes: customer (picked from c1..c15), items (1..16, as in the log)
                     place order -> (throw ItemOrdered x items: object_id, index, customer)
                     -> confirm order (Sales, about a day)
                     -> XOR: 80% pay order (Payments, about 7 days) -> end
                           | 20% timer 20 days -> payment reminder -> back to the XOR
Warehouse (item)     start ItemOrdered (copy order_id, item_index from index, customer)
                     object attributes: product (picked uniformly from the log's 20, as in the generator), and its
                        weight and price
                     -> XOR: 20% item out of stock -> reorder item -> timer: item restocked | 80% straight on
                     -> pick item (Warehousing; about two days, as are out of stock and reorder)
                     -> message end ItemPicked{order_id, item_index, customer, object_id, weight}
Packaging (package)  arrivals: weekdays 7:00-17:00, about one packing round every 2.6 h, 1,440 rounds
                     -> catch ItemPicked: the oldest waiting one and every other of its customer (collect "all",
                        same customer); copy customer, append object_id to item_ids, weight = sum of their weights
                     -> create package (Packer) -> send package (Shipment, about a day)
                     -> XOR: 80% package delivered -> end | 20% failed delivery
                        -> XOR: 70% package delivered -> end | 30% failed delivery again
```

The customer is picked for the order and travels with the messages: order → `ItemOrdered` → item →
`ItemPicked` → package. Packages are made as in the log: a packing round starts in working hours, takes the
oldest waiting item and then all of that customer's waiting items, and creates the package. If no item is
waiting, the round waits for the next one, and takes it with the others of its customer offered at that
moment. How long an item waits for its package therefore depends on how often rounds come, not on a drawn
package size.

Rounds keep coming after the last item, and would wait forever, so their number is limited: 1,440 rounds last
until the end of May 2024 (with seed 1 the last starts on 2024-05-29), as packing does in the log. The orders
end on 2024-03-19 and restocked items are picked up to four weeks later (with seed 1 the last on 2024-04-18),
so every item is packed. The rounds left over wait for an item until the run ends: the run report lists them as
stalled, and the OCEL file leaves them out (199 with seed 1). The number covers the run; it isn't tuned to the
log's package count, since the rounds only approximate the generator's packing ([Packing](#packing)).

Not modeled, and where each goes:

- **Confirmation by the customer's primary sales rep** (88% of confirmations in the log; any Sales employee
  confirms here): under discussion.
- **The forwarder of a package** (the shipper sends it here): under discussion.

## Who talks to whom

Every arrow goes through the orchestrator; processes never address each other. Each process is its own
consumer group, so every message goes to the one process that subscribes to its type.

```mermaid
flowchart LR
    Sales([Sales<br/>one case per order])
    Warehouse([Warehouse<br/>one case per item])
    Packaging([Packaging<br/>one case per packing round])

    Sales -->|"ItemOrdered<br/>(one per item: count)"| Warehouse
    Warehouse -->|"ItemPicked<br/>(a packing round takes all of one customer's)"| Packaging
```

## Messages

| Process       | Publishes                                                                                 | Consumes                                                                                                                                                                   |
|---------------|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Sales**     | `ItemOrdered{object_id, index, customer}`, `count` = `items`                              | nothing                                                                                                                                                                    |
| **Warehouse** | `ItemPicked{order_id, item_index, customer, object_id, weight}`, at its message end event | `ItemOrdered` at its start event; copies `order_id` from `object_id`, `item_index` from `index`, `customer` from `customer`                                                |
| **Packaging** | nothing                                                                                   | `ItemPicked` at "items picked": `"collect": "all"` with `"same": ["customer"]`, copying `customer`, appending `object_id` to `item_ids` and summing `weight` into `weight` |

## How the log maps to the model

| In the log                     | Process   | BPMN element               | Feature                                                                                                          |
|--------------------------------|-----------|----------------------------|------------------------------------------------------------------------------------------------------------------|
| object type **orders**         | Sales     | one case per order         | arrivals on a weekday calendar; object attribute `items`; `customer` picked from the catalogue                   |
| object type **items**          | Warehouse | one case per item          | a message start event on `ItemOrdered`, published once per item with `count`; `index` → `item_index`             |
| object type **packages**       | Packaging | one case per packing round | arrivals in working hours; `collect` `"all"` with `same` customer; `weight` the sum of its items'                |
| object type **customers**      | all       | attribute `customer`       | 15 in the catalogue; picked for the order, copied to items and packages; what a package's items share            |
| object type **products**       | Warehouse | attribute `product`        | the log's 20 in the catalogue; one picked uniformly for each item, and its weight and price with it              |
| object type **employees**      | all       | resources                  | Sales, Warehousing, `Packer` (Packaging) and Shipment, large enough that no task waits for a person              |
| activity **place order**       | Sales     | task `Place_Order`         | done by the `Shop` resource; `ItemOrdered` is published right after it                                           |
| activity **confirm order**     | Sales     | task `Confirm_Order`       | Sales                                                                                                            |
| activity **payment reminder**  | Sales     | task `Payment_Reminder`    | the 20% branch of an XOR, after a 20-day timer, then back to the XOR                                             |
| activity **pay order**         | Sales     | task `Pay_Order`           | the 80% branch of that XOR; done by the `Payments` resource, its duration the delay before paying                |
| activity **item out of stock** | Warehouse | task `Item_Out_Of_Stock`   | the 20% branch of an XOR                                                                                         |
| activity **reorder item**      | Warehouse | task `Reorder_Item`        | after item out of stock                                                                                          |
| activity **pick item**         | Warehouse | task `Pick_Item`           | Warehousing; `ItemPicked` is published at the message end event after it                                         |
| not in the log                 | Warehouse | timer `Item_Restocked`     | the supplier's delay between reorder item and pick item                                                          |
| activity **create package**    | Packaging | task `Create_Package`      | after the packing round has taken its items                                                                      |
| activity **send package**      | Packaging | task `Send_Package`        | Shipment                                                                                                         |
| activity **failed delivery**   | Packaging | task `Failed_Delivery`     | the 20% branch after send package, then the 30% branch after each failure                                        |
| activity **package delivered** | Packaging | task `Package_Delivered`   | the other branches: 80% after send package, 70% after a failure                                                  |

## Numbers from the log

| Parameter         | Real log                                                                               | In the model                                                                      |
|-------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Orders            | 2,000; one every 4.26 h on average; 95% on weekdays, mostly 6-22                       | exponential gaps of 2.3 h mean within a weekday 6:00-22:00 calendar               |
| Customers         | 15, 103-152 orders each                                                                | `customer`: picked uniformly from the catalogue's c1..c15                         |
| Items per order   | 1-16 (1: 159, 2: 360, 3: 489, 4: 367, 5: 275, 6: 164, 7: 100, 8+: 86)                  | `items`: these frequencies, 8+ as in the log                                      |
| Out of stock      | 1,544 of 7,659 items (20%)                                                             | XOR 20 / 80                                                                       |
| Confirm to pay    | median 7 days, quartiles 2.34 / 17.86, max 91; 23.5% after 20 days                     | the generator's loop: XOR 80 / 20 after confirm and after every reminder          |
| Payment reminder  | exactly 20 days after confirm or the previous reminder; 0-4 per order                  | timer, fixed 20 days on the clock                                                 |
| Packing rounds    | 1,128 packages; gaps between create package: median 2.6 h; mostly 7-17                 | exponential gaps of 2.6 h mean within a weekday 7:00-17:00 calendar, 1,440 rounds |
| Items per package | 1-20, median 6; always one customer: the oldest waiting item and all of its customer's | the same rule: `"collect": "all"` with `"same": ["customer"]`                     |
| Products          | 20, 343-433 items each; weight 0.166-1.48 (an item weighs what its product weighs)     | picked uniformly, as in the generator; `with` gives each item its weight          |
| Package weight    | the sum of its items' weights                                                          | `"aggregate": {"weight": {"sum": "weight"}}`                                      |
| Failed delivery   | 311 failures in 1,439 attempts; 0 / 1 / 2 / 3+ per package: 81.0 / 13.2 / 4.0 / 1.8%   | the generator's odds: 20% at the first attempt, 30% after a failure               |
| Employees         | Sales 5, Warehousing 7, Shipment 6, each task's picked at random                       | pools of 100 (Sales), 300 + 20 (Warehousing, `Packer`), 100 (Shipment)            |
| Task durations    | the generator's delays, in its working time                                            | see [Durations](#durations)                                                       |

Place order, pay order and payment reminder have no employee in the log; here they are done by `Shop`
(always available) and `Payments` resources, a pool of 100 on the staff's calendar (see
[Durations](#durations)).

## The log's generator

The Order Management log is artificial: it was simulated with CPN Tools, from a colored Petri net by Knopp and
van der Aalst, published with the log ([Zenodo record 8428112](https://zenodo.org/records/8428112)). The
model's files (`order-management.cpn`, `sim-utils.sml` and the event writers) are in `order-management.zip`
on the record, not in this repository. Its rules:

- **No staff capacity.** Employees are labels, picked at random for each event from those with the right role;
  nobody is ever busy, so nothing waits for a person.
- **Every wait is a delay before an instant event.** The step that leads to an event draws a random delay,
  `expb(mean, cap)`: exponential with that mean, drawn again while above the cap, and running only in working
  time, a speed per hour of the day (see [Durations](#durations)). When it ends, the event happens at once.
- **Packing triggers on a backlog.** When more than 50 picked items wait, the first of them and every waiting
  item of its customer are packed; otherwise, every 5 days, the same happens for whatever waits.
- **Prices rise with inflation.** About every 60 days, inflation rises by 1-6 points, up to 15%, and every
  product's price with it.
- **Orders arrive at an even pace**, one every week / 50 of arrival time, whose speed changes by hour, weekday
  and month ([Arrivals](#arrivals)).
- **An order's items and price.** An order gets `poisson(2 × (1 − inflation/100)) + 1` items, each of one of the
  20 products, all equally likely; the net can keep adding items until it prices the order, which gives the
  log's 1 to 16 (mean 3.83, where the Poisson part alone averages at most 3). The order's price is the sum of
  its items' prices plus 5.00. Place order and create package happen at once, at the end of the step before;
  here too: they take no time, and create package is done by a pool that is always available.
- **Orders are paid in a loop.** After confirm order, 80% pay after `expb(7 days, 20)` and 20% get a reminder
  after exactly 20 days on the clock; after every reminder, the same choice again. Here the same loop: an XOR
  80 / 20, pay order taking the generator's delay as its duration, and a 20-day timer before the reminder.

Where the model differs from the generator, and why:

| Generator                                                                                                                              | Here                                                                                                                   | Why                                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| working time is a speed per hour (0.1 to 1), 0.9 on Mondays and Fridays, 0.7 in July and 0.5 in August, in German local time           | a weekly calendar working at full speed for as many hours, shorter on Mondays and Fridays, 1.5 hours earlier in UTC    | a Prosimos calendar has no speeds, months or summer time ([Durations](#durations))    |
| a reordered item is picked when the supplier's delay, `expb(10 days, 40)`, ends                                                        | a restock timer of 8 days on the clock, then pick item's 2 days                                                        | pick item is one task on both paths; the same mean, an 11% higher median (accepted)   |
| packing when more than 50 items wait, or every 5 days                                                                                  | packing rounds in working hours, about one every 2.6 h, each taking the oldest waiting item and its customer's         | future work ([Packing](#packing))                                                     |
| delivery: 80% at the first attempt, 70% after a failure                                                                                | the same odds, with a second gateway after failed delivery                                                             | a gateway has one set of odds                                                         |
| confirm order by the customer's primary sales rep (86.4%), else 70% by the secondary, else by any                                      | any Sales person                                                                                                       | under discussion                                                                      |
| send package by two people: a forwarder (90% a Warehousing employee, else 70% the package's shipper, else any shipper) and the shipper | one Shipment person                                                                                                    | a task has one resource, and resources aren't shared between processes                |
| prices rise with inflation (6 prices per product)                                                                                      | each product keeps its first price                                                                                     | a catalogue's attributes are fixed                                                    |
| items per order: `poisson(2 × (1 − inflation/100)) + 1`, and more while the net keeps adding items                                     | drawn with the log's frequencies of 1 to 16                                                                            | an object attribute's distribution doesn't change during the run                      |
| an order's price: its items' prices plus 5.00                                                                                          | orders have no price                                                                                                   | future work: the order choosing its products                                          |
| arrivals: one every week / 50 of arrival time, slower at night, at weekends and in April                                               | exponential gaps within a weekday 6:00-22:00 calendar                                                                  | an arrival calendar is on or off: future work ([Arrivals](#arrivals))                 |
| employees are labels; nobody is ever busy                                                                                              | pools of 100 (Sales), 300 + 20 (Warehousing, `Packer`) and 100 (Shipment): at most 116 Warehousing people busy at once | a Prosimos resource is held while it works; enough of them, and no task waits for one |

### Packing

The generator packs on a backlog. When more than 50 picked items wait, it takes the oldest of them and every
waiting item of the same customer, with no limit on their number, and creates the package. A timer also packs
every 5 days, at 23:00, the oldest waiting item and its customer's, however few wait. The log's largest
package, 20 items, is a side effect of the backlog of 51: it is rare for one of the 15 customers to have more
than 20 of them.

Copying this needs two start events, and a message start event that fires only when more than a number of
messages wait: future work. Until then, packing rounds start at a fixed rate in working hours, each taking the
oldest waiting item and all of its customer's ([The processes](#the-processes)). How often they come doesn't
depend on how many items wait, and that shows in the comparison (seed 1; the log's numbers first):

- **Packages: 1,246 against 1,128.** Up to the last order, 1,188 against 1,107. After it, the backlog stays
  small: the log has 21 packages, 14 of them by the 5-day timer, of 5.3 items on average; here the rounds keep
  coming at the same rate, and the 58 packages after the last order hold 136 items, 36 of them a single item.
- **Items per package: 6.79 / 6 / 20 against 6.21 / 5 / 26** (mean / median / max), from the same small
  packages, and from rounds that come when one customer has more than 20 items waiting.
- **Pick item to create package: median 46.0 h against 45.7 h**, third quartile 79.6 h against 74.7 h: how
  long an item waits depends on the backlog in the generator, on the rounds' rate here.
- **Create package → send package ends the same day for 45.5% against 60.0%.** The log creates 85 packages
  (7.5%) at 23:00, by the 5-day timer, and those are all sent the next day; the rounds create packages in
  working hours only.

These numbers are reported next to the log's as information, not as targets.

### Arrivals

The generator's orders arrive at an even pace: one every week / 50 of arrival time, with no randomness. Arrival
time has its own speed per hour, in German local time like the working time: full from 9 to 19, half from 8 to
9 and from 19 to 23, 0.1 at night; at weekends a tenth of that; in April 0.3 of it. As for the delays, its clock
counts the first, partial hour of each gap as a whole one, so the orders come 39.5 a week rather than 50, and
the 2,000 end on 2024-03-23 (at 50 a week they would end in January 2024). Replayed with these rules, the
generator gives every one of the log's 2,000 order times, to the second.

Here, orders arrive with exponential gaps of 2.3 h mean within a weekday 6:00-22:00 calendar (UTC). An arrival
calendar is on or off, so it can't express the speeds, and the engine doesn't use it as working time: it draws
each gap on the clock, and an arrival that falls outside the calendar moves to the next opening, from which the
next gap counts. So every closed night ends with one order at the opening. Copying the generator needs arrival
times computed from a speed profile and passed to the engine: future work. The effects (seed 1; the log's
numbers first):

- **No orders at weekends** (4.9% in the log) and almost none at night: 6.5% against 0.1% between 22:00 and
  6:00 UTC.
- **Many orders at the calendar's opening: 2.5% against 16.1% between 6:00 and 7:00 UTC**; 245 of them (12%)
  are placed at the opening itself, 6:00, 30 of those a microsecond before it, from rounding.
- **Confirm order 12% and pick item 18% sooner** (medians 17.7 h against 15.5 h, and 32.6 h against 26.7 h).
  An order placed at the opening, as the staff start work, crosses fewer nights before it is confirmed and its
  items are picked. From the log's own order times, the same durations and calendar give 18.0 h and 28.4 h
  (+2% and -13%). Pick item's median, like the Shipment's, falls in the night between the first and the second
  day: the generator itself, run again from the log's own order times with ten seeds, gives 28.7 to 33.4 h
  (confirm order 17.9 to 19.1 h), and the log's 32.6 h is near the top of that range.

## Durations

Each task here takes, as its duration, the generator's delay for the step that leads to it, and every staff
pool is large enough that no task waits for a person.

The generator divides its delays by its own ratio of clock time to working time, so "1 day" lasts about a day
on the clock on average. Durations here run in the staff's calendar, so each mean and cap is converted:
working duration = generator value × (the calendar's 41.28 working hours a week ÷ 168) = value × 0.2457.

| Task                                                   | Generator delay (mean, cap) | Duration here (mean, cap), working time                |
|--------------------------------------------------------|-----------------------------|--------------------------------------------------------|
| `Confirm_Order`                                        | 1 day, 10 days              | 21,230 s (5.9 h), 212,297 s (59.0 h)                   |
| `Item_Out_Of_Stock`, `Reorder_Item`, `Pick_Item`       | 2 days, 10 days             | 42,459 s (11.8 h), 212,297 s (59.0 h)                  |
| `Send_Package`, `Failed_Delivery`, `Package_Delivered` | 1 day, 2 days               | 21,230 s (5.9 h), 42,459 s (11.8 h)                    |
| `Pay_Order`                                            | 7 days, 20 days             | 148,608 s (41.3 h), 424,594 s (117.9 h)                |
| timer `Item_Restocked`, then `Pick_Item`               | 10 days, 40 days            | timer: 8 days, 30 days, on the clock; then pick item's |

A reordered item is picked when the supplier's delay ends, with no delay of its own; here it goes through pick
item like any other item. So the restocking timer takes 8 days and pick item's 2 days make up the generator's 10,
with caps 30 and 10 for its 40. Timers run on the clock, so the timer isn't converted. This is an
approximation: the sum of the two has the generator's mean, but not the shape of one exponential, and a higher
median. The reminder's 20 days are a timer on the clock, as in the generator, which doesn't stretch them by its
working time. Place order and create package take no time, as in the generator, by the `Shop` and `Packer`
pools, which are always available: an order is placed when it arrives, and a package is created when its round
has taken its items. The arrivals keep their own rule ([Arrivals](#arrivals)).

The generator's working time is a speed per hour, in German local time: full from 9 to 12 and 14 to 17, half
from 8 to 9, 12 to 14 and 17 to 18, 0.3 from 18 to 19, 0.1 from 19 to 22, none at night or at weekends;
Monday and Friday at 0.9; July at 0.7 and August at 0.5. That is 8.6 working hours on Tuesday to Thursday and
7.74 on Monday and Friday. A calendar has no speeds, so the staff's calendar (`Generator`) works at full speed
for those hours, in two blocks around the generator's midday dip: 8:30-12:30 and 13:30-18:06 local time, each
half-speed hour becoming half an hour of work next to the full-speed hours, and the evening's 0.6 hour coming
straight after 17:30; on Monday and Friday each block is 10% shorter around its middle. The log and the model
are in UTC, and German time is UTC+2 in summer time and UTC+1 in winter time; 57% of the orders fall in summer
time, so the calendar is 1.5 hours earlier, half an hour off in either:

| Days                    | Working hours (UTC)                  | Hours |
|-------------------------|--------------------------------------|-------|
| Tuesday to Thursday     | 07:00-11:00, 12:00-16:36             | 8.6   |
| Monday, Friday          | 07:12-10:48, 12:13:48-16:22:12       | 7.74  |

Not expressible in a weekly calendar: the partial speeds, the slower July and August, and summer and winter
time. The comparison script counts working hours in this calendar, for both files.

## Running it

```
poetry run prosimos start-orchestration --config testing_scripts/assets/order_management/simulation.json \
    --log_out_path order_management_log.csv --ocel_out_path order_management_log.json
poetry run python testing_scripts/order_management_compare.py <path to order-management.json> order_management_log.json
```

The configuration names the object types in the singular, one kind of thing each: `order` (Sales), `item`
(Warehouse) and `package` (Packaging), and the catalogue adds `customer` and `product`; objects are named
`<object type>-<number>`, e.g. `order-7`. The comparison script reads them under the real log's plural names
(`order` as `orders`). Each event
is linked to the objects on the other end of its messages ([orchestrator.md](orchestrator.md), "One event linked
to many objects"), as the Order Management log links them. Like the log, only one end of the messages between
the items and the others links: `ItemOrdered` from the order's place order, `ItemPicked` from the package's
create package; the Warehouse's entries have no qualifier, so the item's events link neither its order nor its
package. With `"tag_along": true` on the product reference, every event that links an item also links its
product (`product`), as in the log; customers don't tag along, so pay order links no customer:

| Event                                            | Linked to                                                                                                              |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| place order                                      | its order (`order`), the items its ItemOrdered messages started (`item`), their products and its customer (`customer`) |
| confirm order                                    | its order, its customer and, with `carry_links`, its items and their products                                          |
| payment reminder, pay order                      | its order and, with `carry_links`, its items and their products                                                        |
| every event of an item                           | its item (`item`) and its product (`product`), from the reference that picks it; not its order or its package          |
| create package                                   | its package (`creates`), the items it collected (`item`) and their products                                            |
| send package, failed delivery, package delivered | its package (`shipped package`) and, with `carry_links`, its items and their products                                  |

Objects are linked to each other too: each order **comprises** its items (`"o2o": "comprises"` on Sales's
ItemOrdered entry), each package **contains** its items (`"o2o": "contains"` on Packaging's ItemPicked entry),
and each item **is a** product (`"o2o": "is a"` on the product reference).

Customers have no events of their own, so they come from a catalogue of Sales: 15 customers, c1 to c15, without
attributes, as in the log ([catalogue.md](catalogue.md) uses this configuration as its worked example). A reference
of Sales on its `customer` attribute picks it when an order is created (`"pick": "uniform"`; `sales.json` doesn't
draw it), links each customer to the orders it **places** (`"reverse": true`, from the customer to the order), and
links place order and confirm order to their customer; pay order isn't linked. Since `ItemOrdered` carries the
picked `customer`, `sales.json` only runs inside this configuration, not on its own. Products come from a catalogue
of the Warehouse: the log's 20, each with its `weight` and first `price`. A reference of the Warehouse picks one
for each item, uniformly, as the generator does, links the item to it (**is a**),
links every event of the item to it and, with `"tag_along": true`, every other event that links the item, and, with
`"with": {"weight": "weight", "price": "price"}`, gives the item its product's weight and price when it is created. They are the
item's own attributes from then on: written on its OCEL object (the Warehouse's `ocel_attributes`), and `weight`
travels on `ItemPicked` to be summed into the package's.

Only the attributes the log has are written to the objects (`ocel_attributes`): an item's `weight` and `price`
and a package's `weight`. The others route messages (`order_id`, `item_index`, `item_ids`, `items`) or are links
already (`customer`, `product`), and stay in the CSV log only. A packing round takes all its items at the same
moment, so its package gets one weight, the sum of its items', as in the log, without the running totals in
between or the starting 0.

Sales and Packaging have `carry_links`; Warehouse doesn't, so a reorder is linked only to its item and its
product. A packing round that never took an item has no events and no links, so the OCEL file leaves its package
out; the run report still lists it as stalled. The full 2,000-order run takes about 21 seconds (Apple M4 Pro).

The comparison script reads the real log and the simulated OCEL file the same way, from their objects,
events and links. It counts object-to-object links from the files themselves, since pm4py leaves out objects
that no event refers to.

## Comparison with the log

2,000 orders, seed 1; real log against the simulated OCEL file, both read by the comparison script.

| Metric                                                              |                       Real log |                      Simulated |
|---------------------------------------------------------------------|-------------------------------:|-------------------------------:|
| orders                                                              |                          2,000 |                          2,000 |
| items                                                               |                          7,659 |                          7,735 |
| packages                                                            |                          1,128 |                          1,246 |
| customers / products                                                |                        15 / 20 |                        15 / 20 |
| picked items left unpacked                                          |                              0 |                              0 |
| items per order: mean / median / max                                |                  3.83 / 3 / 16 |                  3.87 / 3 / 16 |
| items per order: 1 / 2 / 3 / 4                                      |      8.0 / 18.0 / 24.4 / 18.4% |      8.1 / 17.4 / 24.9 / 17.8% |
| items per order: 5 / 6 / 7 / 8+                                     |        13.8 / 8.2 / 5.0 / 4.3% |        13.8 / 7.8 / 5.5 / 5.0% |
| reminders per order: mean / max                                     |                       0.28 / 4 |                       0.24 / 4 |
| reminders per order: 0 / 1 / 2 / 3 / 4+                             | 77.8 / 17.2 / 3.9 / 0.9 / 0.1% | 80.5 / 16.0 / 2.9 / 0.5 / 0.1% |
| orders paid after 20 days                                           |                          23.5% |                          20.0% |
| items out of stock                                                  |                          20.2% |                          20.0% |
| items per package: mean / median / max                              |                  6.79 / 6 / 20 |                  6.21 / 5 / 26 |
| customers per package                                               |                       always 1 |                       always 1 |
| packages per customer: mean / median / max                          |                 75.2 / 76 / 80 |                 83.1 / 83 / 89 |
| failed deliveries per package: mean / max                           |                       0.28 / 7 |                       0.24 / 5 |
| failed deliveries per package: 0 / 1 / 2 / 3+                       |       81.0 / 13.2 / 4.0 / 1.8% |       82.7 / 12.0 / 3.7 / 1.5% |
| items per event: place order / create package / send package (mean) |             3.83 / 6.79 / 6.79 |             3.87 / 6.21 / 6.21 |
| links: customers places orders                                      |                          2,000 |                          2,000 |
| links: orders comprises items                                       |                          7,659 |                          7,735 |
| links: packages contains items                                      |                          7,659 |                          7,735 |
| links: items is a products                                          |                          7,659 |                          7,735 |
| links to employees (sales reps, packer, forwarder, shipper)         |                          3,414 |                              - |
| price values per product: min / median / max                        |                      6 / 6 / 6 |                      1 / 1 / 1 |
| place order → confirm order (h): median / in working hours          |                     17.7 / 4.2 |                     15.5 / 3.9 |
| confirm order → pay order (h)                                       |                   119.2 / 28.8 |                   114.2 / 25.7 |
| payment reminder → pay order (h)                                    |                    98.1 / 27.0 |                    96.2 / 25.6 |
| place order → item out of stock (h)                                 |                     28.0 / 8.4 |                     26.6 / 8.0 |
| item out of stock → reorder item (h)                                |                     25.7 / 8.3 |                     25.0 / 8.1 |
| place order → pick item (h)                                         |                     32.6 / 8.6 |                     26.7 / 8.2 |
| reorder item → pick item (h)                                        |                   167.0 / 40.7 |                   186.2 / 45.2 |
| create package → send package: the same day                         |                          45.5% |                          60.0% |
| send package → failed delivery: the same day                        |                          51.9% |                          54.4% |
| failed delivery → failed delivery: the same day                     |                          47.4% |                          57.3% |
| send package → package delivered: the same day                      |                          48.4% |                          57.1% |
| failed delivery → package delivered: the same day                   |                          43.5% |                          51.6% |
| out of order (FIFO displacement): pick item / send package (places) |                         21 / 1 |                         21 / 1 |
| pick item to create package: quartiles (h)                          |             22.0 / 46.0 / 79.6 |             22.0 / 45.7 / 74.7 |
| pick item to create package: mean (h)                               |                           56.4 |                           51.3 |
| items picked before their order is confirmed                        |                          25.4% |                          25.5% |
| orders paid after their first delivery                              |                          67.3% |                          65.8% |

The gaps are medians from the event that enables a task to the task, on the clock and counting only the
calendar's working hours; the Shipments are compared by the share that ends the day it starts (UTC), since
their medians flip (below). An object's events at the same moment count in the order their activities enable
each other: a package can be created the moment its item is picked, in the log as here. The FIFO displacement ranks the in-stock items by when they were ordered and by when
they were picked (the packages by when they were created and sent), and is the median of how many places each
moves.

The counts and shares that come from the model's structure and parameters (items per order, out of stock,
customers per package, failed deliveries, the links between objects) match closely; failed deliveries per
package follow from the generator's odds of 20% at the first attempt and 30% after a failure. With the
generator's durations, the gaps to item out of stock and reorder item, how far items are picked out of order
(21 places in both), and the share of items picked before their order is confirmed match the log too; the tasks wait for nobody, as in the
log. What still differs comes from the generator's rules, or from parts of the model left as they are:

- **Confirm order and pick item come about 12% and 18% sooner**, from the arrivals: no order comes at a
  weekend and 12% come at the calendar's opening, 6:00 UTC ([Arrivals](#arrivals)). From the log's own order times, the
  same durations and calendar give 18.0 h and 28.4 h, and the generator's own pick item median moves between
  28.7 and 33.4 h from seed to seed.
- **The Shipment's steps end the same day a little more often.** Their delays average a day on the clock, so
  about half end the day they start and half the next, with the night in between: send → package delivered
  has quartiles 3.1 / 14.4 / 22.9 h in the log, 1.6 / 5.7 / 22.0 h here. The median falls in that night, and a
  few points more on either side move it from the afternoon to the next morning: the generator itself, run
  again from the log's own send events with ten seeds, gives medians from 7.7 to 14.5 h (send → failed
  delivery 6.1 to 15.9 h), but same-day shares within a few points (48.5 to 52.2% for send → package
  delivered). Ours are up to 10 points higher than the log's, from two generator rules a weekly calendar can't
  follow, which push its steps into the next day: the slower July and August, and its clock counting a
  delay's first, partial hour as a whole one (0 to 60 minutes more per delay). Create package → send package
  (60% against 46%) also depends on the packing ([Packing](#packing)).
- **Reorder item → pick item is 12% longer** (2 to 12% with seeds 1 to 6). The restock timer plus pick item has the generator's mean but a
  higher median (from the log's own reorders, 172.5 h against the generator's 167.7 h), and the reorders come
  at other times than the log's.
- **Products have 6 prices each in the log**, the generator raising them for inflation about every 60 days;
  here each product keeps its first price.

The payments follow the generator's loop: 19.2% of the 2,476 choices after a confirmation or a reminder lead to
a reminder, a standard deviation below the 20% of its rule, and the confirm → pay order and
reminder → pay order gaps are within 5% of the log's. The log's own draws give 22.1% of 2,566 choices, 2.6
standard deviations above its rule, which happens about once in a hundred runs; so it has more orders with
reminders (22.2% against 19.6%). Orders paid after 20 days are those with a reminder, and those whose capped
delay still ran past 20 days on the clock: 1.35% in the log, 0.45% here, since the generator's slower July and
August and its rounding of a delay's first, partial hour stretch some delays, which a weekly calendar can't.
The packages, their sizes and how long items wait for them follow from the
packing rounds ([Packing](#packing)).

Numbers that hang on the order of the random draws move from seed to seed; with seeds 1 to 6 the model gives
1,218 to 1,338 packages, 15.5 to 16.7 h from place order to confirm order, 114 to 121 h from confirm order to
pay order, 171 to 187 h from reorder item to pick item, 79.3 to 80.9% of orders without a reminder, and 46 to
57% of failed deliveries followed by another the same day. Small changes to the model, such as one that shifts
every order by a minute, move them within these ranges. The log links packages and confirmations to employees; here employees
are resources, not objects.

## Gaps found

Found while building the example; no feature was added for them.

- **`index` can't be passed on.** `index` is reserved for `count` in publish entries, so an item can't
  publish its own number under that name. The Warehouse keeps it as `item_index` instead.
- **Resources aren't shared between processes.** Picking (Warehouse) and packing (Packaging) are done by the
  same 7 Warehousing employees in the log, but each Prosimos process has its own resource pool (`Warehousing`
  and `Packer` here). With pools large enough that no task waits, it doesn't change the times. Under
  discussion.
- **A process started by arrivals has no end time.** The packing rounds are limited by their number, 1,440,
  chosen so that they last until the end of May 2024, as packing in the log, and every item is packed; the
  rounds left over stall at the end (199 with seed 1).
- **A calendar has no speeds.** The generator works at partial speed in some hours, months and weekdays; a
  calendar only works or doesn't ([Durations](#durations)).
