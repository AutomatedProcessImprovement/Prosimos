# Order Management example

An example shaped like the OCEL 2.0 **Order Management** log (2,000 orders, 7,659 items, 1,128
packages), to show the multi-process features working together on a realistic model and to compare counts
and shares with the real log. Its task durations follow the rules of the program that generated the log
([Durations](#durations)). The models are in `testing_scripts/assets/order_management/`; the tests are in
`testing_scripts/test_order_management.py`.

It uses most messaging features ([messaging.md](messaging.md)): one message per object with `count` and
`index`, processes started by messages, `copy`, `append` and `aggregate`, conditions on case attributes, a race
with a timer that repeats, and `collect` with `same`; and the catalogue ([catalogue.md](catalogue.md)) for its
customers and products.

## The processes

![Process models of the Order Management example](order-management-processes.svg)

```
Sales (order)        arrivals: weekdays 6:00-22:00, about one every 4.3 h, 2,000 orders
                     object attributes: customer (picked from c1..c15), items (1..16, as in the log)
                     place order -> (throw ItemOrdered x items: object_id, index, customer)
                     -> confirm order (Sales, about a day) -> (throw OrderConfirmed: object_id)
                     -> race: catch Payment (order_id = object_id) -> pay order -> end
                            | timer 20 days -> payment reminder -> back to the race
Customer             start OrderConfirmed (copy order_id) -> timer (confirm-to-pay delay)
                     -> message end Payment{order_id}
Warehouse (item)     start ItemOrdered (copy order_id, item_index from index, customer)
                     object attributes: product (picked from the log's 20, as often as in the log), and its
                        weight and price
                     -> XOR: 20% item out of stock -> reorder item -> timer: item restocked | 80% straight on
                     -> pick item (Warehousing; about two days, as are out of stock and reorder)
                     -> message end ItemPicked{order_id, item_index, customer, object_id, weight}
Packaging (package)  arrivals: weekdays 7:00-17:00, about one packing round every 2.6 h, 1,200 rounds
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

Rounds keep coming after the last item, and would wait forever, so their number is limited: 1,200 rounds last
until a few days after the last order (the orders end on 2024-03-19, the last round starts on 2024-03-22).
Restocked items are picked up to four weeks later (with seed 1 the last on 2024-04-18), so every round takes
items and 66 items are left unpacked: their `ItemPicked` messages are still unclaimed when the run ends. In the
log, packing goes on until the end of May 2024. How the rounds should end is future work.

Not modeled, and where each goes:

- **Confirmation by the customer's primary sales rep** (88% of confirmations in the log; any Sales employee
  confirms here): under discussion.
- **The forwarder of a package** (the shipper sends it here): under discussion.

The Customer process only models how long a customer takes to pay; its rows aren't compared with the log.

## Who talks to whom

Every arrow goes through the orchestrator; processes never address each other. Each process is its own
consumer group, so every message goes to the one process that subscribes to its type.

```mermaid
flowchart LR
    Sales([Sales<br/>one case per order])
    Customer([Customer<br/>one case per confirmed order])
    Warehouse([Warehouse<br/>one case per item])
    Packaging([Packaging<br/>one case per packing round])

    Sales -->|"ItemOrdered<br/>(one per item: count)"| Warehouse
    Sales -->|"OrderConfirmed<br/>(starts a case)"| Customer
    Customer -->|"Payment<br/>(same order)"| Sales
    Warehouse -->|"ItemPicked<br/>(a packing round takes all of one customer's)"| Packaging
```

## Messages

| Process       | Publishes                                                                                 | Consumes                                                                                                                                                                   |
|---------------|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Sales**     | `ItemOrdered{object_id, index, customer}`, `count` = `items`; `OrderConfirmed{object_id}` | `Payment` where `order_id == object_id`, in a race against the 20-day reminder timer                                                                                       |
| **Customer**  | `Payment{order_id}`, at its message end event                                             | `OrderConfirmed` at its start event; copies `order_id` from `object_id`                                                                                                    |
| **Warehouse** | `ItemPicked{order_id, item_index, customer, object_id, weight}`, at its message end event | `ItemOrdered` at its start event; copies `order_id` from `object_id`, `item_index` from `index`, `customer` from `customer`                                                |
| **Packaging** | nothing                                                                                   | `ItemPicked` at "items picked": `"collect": "all"` with `"same": ["customer"]`, copying `customer`, appending `object_id` to `item_ids` and summing `weight` into `weight` |

## How the log maps to the model

| In the log                     | Process   | BPMN element               | Feature                                                                                                          |
|--------------------------------|-----------|----------------------------|------------------------------------------------------------------------------------------------------------------|
| object type **orders**         | Sales     | one case per order         | arrivals on a weekday calendar; case attribute `items`; `customer` picked from the catalogue                     |
| object type **items**          | Warehouse | one case per item          | a message start event on `ItemOrdered`, published once per item with `count`; `index` → `item_index`             |
| object type **packages**       | Packaging | one case per packing round | arrivals in working hours; `collect` `"all"` with `same` customer; `weight` the sum of its items'                |
| object type **customers**      | all       | case attribute `customer`  | 15 in the catalogue; picked for the order, copied to items and packages; what a package's items share            |
| object type **products**       | Warehouse | case attribute `product`   | the log's 20 in the catalogue; one picked for each item as often as in the log, and its weight and price with it |
| object type **employees**      | all       | resources                  | Sales, Warehousing, `Packer` (Packaging) and Shipment, large enough that no task waits for a person              |
| activity **place order**       | Sales     | task `Place_Order`         | done by the `Shop` resource; `ItemOrdered` is published right after it                                           |
| activity **confirm order**     | Sales     | task `Confirm_Order`       | Sales; `OrderConfirmed` is published right after it                                                              |
| activity **payment reminder**  | Sales     | task `Payment_Reminder`    | the timer branch of the race (20 days), then back to the race                                                    |
| activity **pay order**         | Sales     | task `Pay_Order`           | after the message branch of the race (`Payment`); done by the `Payments` resource                                |
| activity **item out of stock** | Warehouse | task `Item_Out_Of_Stock`   | the 20% branch of an XOR                                                                                         |
| activity **reorder item**      | Warehouse | task `Reorder_Item`        | after item out of stock                                                                                          |
| activity **pick item**         | Warehouse | task `Pick_Item`           | Warehousing; `ItemPicked` is published at the message end event after it                                         |
| not in the log                 | Warehouse | timer `Item_Restocked`     | the supplier's delay between reorder item and pick item                                                          |
| activity **create package**    | Packaging | task `Create_Package`      | after the packing round has taken its items                                                                      |
| activity **send package**      | Packaging | task `Send_Package`        | Shipment                                                                                                         |
| activity **failed delivery**   | Packaging | task `Failed_Delivery`     | the 20% branch after send package, then the 30% branch after each failure                                        |
| activity **package delivered** | Packaging | task `Package_Delivered`   | the other branches: 80% after send package, 70% after a failure                                                  |
| not in the log                 | Customer  | timer `Pay_Delay`          | how long the customer takes to pay; publishes `Payment`                                                          |

## Numbers from the log

| Parameter         | Real log                                                                               | In the model                                                                      |
|-------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Orders            | 2,000; one every 4.26 h on average; 95% on weekdays, mostly 6-22                       | exponential gaps of 2.3 h mean within a weekday 6:00-22:00 calendar               |
| Customers         | 15, 103-152 orders each                                                                | `customer`: picked uniformly from the catalogue's c1..c15                         |
| Items per order   | 1-16 (1: 159, 2: 360, 3: 489, 4: 367, 5: 275, 6: 164, 7: 100, 8+: 86)                  | `items`: these frequencies, 8+ as in the log                                      |
| Out of stock      | 1,544 of 7,659 items (20%)                                                             | XOR 20 / 80                                                                       |
| Confirm to pay    | median 7 days, quartiles 2.34 / 17.86, max 91; 23.5% after 20 days                     | Customer timer: log-normal with that median and quartiles, at most 91 days        |
| Payment reminder  | exactly 20 days after confirm or the previous reminder; 0-4 per order                  | race timer, fixed 20 days                                                         |
| Packing rounds    | 1,128 packages; gaps between create package: median 2.6 h; mostly 7-17                 | exponential gaps of 2.6 h mean within a weekday 7:00-17:00 calendar, 1,200 rounds |
| Items per package | 1-20, median 6; always one customer: the oldest waiting item and all of its customer's | the same rule: `"collect": "all"` with `"same": ["customer"]`                     |
| Products          | 20, 343-433 items each; weight 0.166-1.48 (an item weighs what its product weighs)     | picked with these frequencies; `with` gives each item its product's weight        |
| Package weight    | the sum of its items' weights                                                          | `"aggregate": {"weight": {"sum": "weight"}}`                                      |
| Failed delivery   | 311 failures in 1,439 attempts; 0 / 1 / 2 / 3+ per package: 81.0 / 13.2 / 4.0 / 1.8%   | the generator's odds: 20% at the first attempt, 30% after a failure               |
| Employees         | Sales 5, Warehousing 7, Shipment 6, each task's picked at random                       | pools of 100 (Sales), 300 + 20 (Warehousing, `Packer`), 100 (Shipment)            |
| Task durations    | the generator's delays, in its working time                                            | see [Durations](#durations)                                                       |

Place order, pay order and payment reminder have no employee in the log; here they are done by `Shop`
(always available) and `Payments` (weekdays 6-22) resources.

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
- **Orders arrive 50 a week** on average, a fixed gap in the arrivals' own working time, which is slower at
  night, at weekends and in April.

Where the model differs from the generator, and why:

| Generator                                                                                                                              | Here                                                                                                                   | Why                                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| working time is a speed per hour (0.1 to 1), 0.9 on Mondays and Fridays, 0.7 in July and 0.5 in August, in German local time           | a weekly calendar working at full speed for as many hours, shorter on Mondays and Fridays, 1.5 hours earlier in UTC    | a Prosimos calendar has no speeds, months or summer time ([Durations](#durations))    |
| a reordered item is picked when the supplier's delay, `expb(10 days, 40)`, ends                                                        | a restock timer of 8 days on the clock, then pick item's 2 days                                                        | pick item is one task on both paths; the same mean, an 11% higher median (accepted)   |
| packing when more than 50 items wait, or every 5 days                                                                                  | packing rounds in working hours, about one every 2.6 h, each taking the oldest waiting item and its customer's         | future work                                                                           |
| payment: 80% pay after `expb(7 days, 20)`; 20% get a reminder after exactly 20 days, and the same again                                | the Customer process pays after a log-normal delay fitted to the log; Sales reminds every 20 days until paid           | future work                                                                           |
| delivery: 80% at the first attempt, 70% after a failure                                                                                | the same odds, with a second gateway after failed delivery                                                             | a gateway has one set of odds                                                         |
| confirm order by the customer's primary sales rep (86.4%), else 70% by the secondary, else by any                                      | any Sales person                                                                                                       | under discussion                                                                      |
| send package by two people: a forwarder (90% a Warehousing employee, else 70% the package's shipper, else any shipper) and the shipper | one Shipment person                                                                                                    | a task has one resource, and resources aren't shared between processes                |
| prices rise with inflation (6 prices per product)                                                                                      | each product keeps its first price                                                                                     | a catalogue's attributes are fixed                                                    |
| arrivals: 50 a week, slower at night, at weekends and in April                                                                         | exponential gaps within a weekday 6:00-22:00 calendar                                                                  | future work                                                                           |
| employees are labels; nobody is ever busy                                                                                              | pools of 100 (Sales), 300 + 20 (Warehousing, `Packer`) and 100 (Shipment): at most 116 Warehousing people busy at once | a Prosimos resource is held while it works; enough of them, and no task waits for one |

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
| timer `Item_Restocked`, then `Pick_Item`               | 10 days, 40 days            | timer: 8 days, 30 days, on the clock; then pick item's |

A reordered item is picked when the supplier's delay ends, with no delay of its own; here it goes through pick
item like any other item. So the restocking timer takes 8 days and pick item's 2 days make up the generator's 10,
with caps 30 and 10 for its 40. Timers run on the clock, so the timer isn't converted. This is an
approximation: the sum of the two has the generator's mean, but not the shape of one exponential, and a higher
median. Create package keeps its own duration (packing rounds are future work), as do the Customer's payment
and the arrivals.

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
(`order` as `orders`). Customer has no object type, so the OCEL file has no objects or events for it. Each event
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

Customers have no events of their own, so they come from a catalogue of Sales: 15 customers, c1 to c15, each
with a `segment` ([catalogue.md](catalogue.md) uses this configuration as its worked example). A reference of
Sales on its `customer` attribute picks it when an order is created (`"pick": "uniform"`; `sales.json` doesn't
draw it), links each customer to the orders it **places** (`"reverse": true`, from the customer to the order),
and links place order and confirm order to their customer; pay order isn't linked. Since `ItemOrdered` carries
the picked `customer`, `sales.json` only runs inside this configuration, not on its own. Products come from a
catalogue of the Warehouse: the log's 20, each with its `weight` and first `price`. A reference of the Warehouse
picks one for each item, with the number of items of each product in the log as weights, links the item to it
(**is a**), links every event of the item to it and, with `"tag_along": true`, every other event that links the
item, and, with `"with": ["weight", "price"]`, gives the item its product's weight and price when it is created.
They are the item's own attributes from then on: written on its OCEL object (the Warehouse's `ocel_attributes`),
and `weight` travels on `ItemPicked` to be summed into the package's.

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
| packages                                                            |                          1,128 |                          1,200 |
| customers / products                                                |                        15 / 20 |                        15 / 20 |
| items per order: mean / median / max                                |                  3.83 / 3 / 16 |                  3.87 / 3 / 16 |
| items per order: 1 / 2 / 3 / 4                                      |      8.0 / 18.0 / 24.4 / 18.4% |      8.1 / 17.4 / 24.9 / 17.8% |
| items per order: 5 / 6 / 7 / 8+                                     |        13.8 / 8.2 / 5.0 / 4.3% |        13.8 / 7.8 / 5.5 / 5.0% |
| reminders per order: mean / max                                     |                       0.28 / 4 |                       0.32 / 4 |
| reminders per order: 0 / 1 / 2 / 3 / 4+                             | 77.8 / 17.2 / 3.9 / 0.9 / 0.1% | 80.0 / 12.4 / 4.2 / 2.4 / 0.9% |
| orders paid after 20 days                                           |                          23.5% |                          20.3% |
| items out of stock                                                  |                          20.2% |                          20.0% |
| items per package: mean / median / max                              |                  6.79 / 6 / 20 |                  6.39 / 6 / 24 |
| customers per package                                               |                       always 1 |                       always 1 |
| packages per customer: mean / median / max                          |                 75.2 / 76 / 80 |                   80 / 79 / 85 |
| failed deliveries per package: mean / max                           |                       0.28 / 7 |                       0.29 / 6 |
| failed deliveries per package: 0 / 1 / 2 / 3+                       |       81.0 / 13.2 / 4.0 / 1.8% |       79.9 / 14.1 / 4.3 / 1.7% |
| items per event: place order / create package / send package (mean) |             3.83 / 6.79 / 6.79 |             3.87 / 6.39 / 6.39 |
| links: customers places orders                                      |                          2,000 |                          2,000 |
| links: orders comprises items                                       |                          7,659 |                          7,735 |
| links: packages contains items                                      |                          7,659 |                          7,669 |
| links: items is a products                                          |                          7,659 |                          7,735 |
| links to employees (sales reps, packer, forwarder, shipper)         |                          3,414 |                              - |
| price values per product: min / median / max                        |                      6 / 6 / 6 |                      1 / 1 / 1 |
| place order → confirm order (h): median / in working hours          |                     17.7 / 4.2 |                     15.7 / 3.8 |
| place order → item out of stock (h)                                 |                     28.0 / 8.4 |                     26.0 / 8.1 |
| item out of stock → reorder item (h)                                |                     25.7 / 8.3 |                     25.2 / 8.1 |
| place order → pick item (h)                                         |                     32.5 / 8.6 |                     26.2 / 7.9 |
| reorder item → pick item (h)                                        |                   166.9 / 40.7 |                   185.3 / 45.0 |
| create package → send package: the same day                         |                          45.5% |                          61.7% |
| send package → failed delivery: the same day                        |                          51.9% |                          50.6% |
| failed delivery → failed delivery: the same day                     |                          47.4% |                          55.3% |
| send package → package delivered: the same day                      |                          48.4% |                          54.0% |
| failed delivery → package delivered: the same day                   |                          43.5% |                          53.1% |
| out of order (FIFO displacement): pick item / send package (places) |                         21 / 1 |                         21 / 1 |
| pick item to create package: quartiles (h)                          |             22.0 / 46.0 / 79.6 |             23.7 / 47.9 / 90.5 |
| pick item to create package: mean (h)                               |                           56.4 |                           56.9 |
| items picked before their order is confirmed                        |                          25.4% |                          26.6% |
| orders paid after their first delivery                              |                          67.3% |                          65.9% |

The gaps are medians from the event that enables a task to the task, on the clock and counting only the
calendar's working hours; the Shipments are compared by the share that ends the day it starts (UTC), since
their medians flip (below). The FIFO displacement ranks the in-stock items by when they were ordered and by when
they were picked (the packages by when they were created and sent), and is the median of how many places each
moves.

The counts and shares that come from the model's structure and parameters (items per order, out of stock,
customers per package, failed deliveries, the links between objects) match closely; failed deliveries per
package follow from the generator's odds of 20% at the first attempt and 30% after a failure. With the
generator's durations, the gaps to item out of stock and reorder item, how far items are picked out of order
(21 places in both), and the share of items picked before their order is confirmed match the log too; the tasks wait for nobody, as in the
log. What still differs comes from the generator's rules, or from parts of the model left as they are:

- **Confirm order and pick item come about 11% and 19% sooner.** The arrivals aren't the log's: no order comes
  at a weekend (4.7% in the log) and 18% come between 6:00 and 7:00 UTC (3%), when the calendar is about to
  open, so their gaps cross fewer nights. From the log's own order times, the same durations and calendar give
  18.0 h and 28.4 h.
- **The Shipment's steps end the same day a little more often.** Their delays average a day on the clock, so
  about half end the day they start and half the next, with the night in between: send → package delivered
  has quartiles 3.1 / 14.4 / 22.9 h in the log, 1.7 / 5.8 / 21.8 h here. The median falls in that night, and a
  few points more on either side move it from the afternoon to the next morning: the generator itself, run
  again from the log's own send events with ten seeds, gives medians from 7.7 to 14.5 h (send → failed
  delivery 6.1 to 15.9 h), but same-day shares within a few points (48.5 to 52.2% for send → package
  delivered). Ours are up to 10 points higher than the log's (send → failed delivery 1 point lower), from two
  generator rules a weekly calendar can't
  follow, which push its steps into the next day: the slower July and August, and its clock counting a
  delay's first, partial hour as a whole one (0 to 60 minutes more per delay). Create package → send package
  (62% against 46%) also starts from the packing rounds: the log creates 7% of its packages at 23:00, by the
  every-5-days trigger, and those are all sent the next day.
- **Reorder item → pick item is 11% longer.** The restock timer plus pick item has the generator's mean but a
  higher median (from the log's own reorders, 172.5 h against the generator's 167.7 h), and the reorders come
  at other times than the log's.
- **Products have 6 prices each in the log**, the generator raising them for inflation about every 60 days;
  here each product keeps its first price.

The fitted confirm-to-pay distribution pays a little earlier than the log, but has a longer tail, so more
orders get 3 or 4 reminders. With rounds about as frequent as in the log, there are 6% more packages, slightly
smaller on average (6.39 items against 6.79). The log links packages and confirmations to employees; here
employees are resources, not objects.

## Gaps found

Found while building the example; no feature was added for them.

- **`index` can't be passed on.** `index` is reserved for `count` in publish entries, so an item can't
  publish its own number under that name. The Warehouse keeps it as `item_index` instead.
- **Resources aren't shared between processes.** Picking (Warehouse) and packing (Packaging) are done by the
  same 7 Warehousing employees in the log, but each Prosimos process has its own resource pool (`Warehousing`
  and `Packer` here). With pools large enough that no task waits, it doesn't change the times. Under
  discussion.
- **A process started by arrivals has no end time.** The packing rounds are limited by their number, 1,200,
  chosen so that they last until just after the orders; restocked items picked after the last round are left
  unpacked (66 with seed 1).
- **A calendar has no speeds.** The generator works at partial speed in some hours, months and weekdays; a
  calendar only works or doesn't ([Durations](#durations)).
- **Reminders are 20 days and one microsecond apart.** A race timer's branch continues one microsecond after
  the timer's time, so that a message at exactly that time wins ([messaging.md](messaging.md)). By design.
