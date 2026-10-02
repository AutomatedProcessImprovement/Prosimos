# Running example: orders, warehouses and trucks

This scenario is used throughout the design documents and the protocol tests.
It exercises every rule of the orchestrator protocol in a single run: announcements, a shared
group, a random tie, correlation, pending messages, discards, warnings and the end-of-run report.

The processes are scripted fake engines in the tests (`testing_scripts/protocol_scenario.py`, checked
by `testing_scripts/test_protocol_scenario.py`). A first version with a real Prosimos Sales model is
described at the end ("First real-engine version"); replacing the other processes is future work.

## Processes and messages

| Process              | Publishes                                            | Consumes                                                                       |
|----------------------|------------------------------------------------------|--------------------------------------------------------------------------------|
| **Sales**            | `OrderPlaced{order_id, city}`, `Newsletter`          | `Shipment`, correlated on `order_id == case.order_id`                          |
| **Billing**          | nothing                                              | every `OrderPlaced`                                                            |
| **TartuWarehouse**   | `Shipment{order_id}` when an order leaves on a truck | `OrderPlaced` where `city in [Tartu, Tapa]`; `Truck` where `dock == Tartu`     |
| **TallinnWarehouse** | `Shipment{order_id}` when an order leaves on a truck | `OrderPlaced` where `city in [Tallinn, Tapa]`; `Truck` where `dock == Tallinn` |
| **Carrier**          | `Truck{dock, capacity: 2}`                           | nothing                                                                        |

```
consumer_groups:
    Sales:      [Sales]
    Billing:    [Billing]
    Carrier:    [Carrier]
    Warehouses: [TartuWarehouse, TallinnWarehouse]
```

Packing takes 1 h in Tartu and 1.5 h in Tallinn. A packed order waits for a truck. A warehouse that
claims a truck loads up to 2 waiting orders, oldest first, and publishes a `Shipment` for each.

## Process models

![Process models of the running example](running-example-processes.svg)

## Who talks to whom

Every arrow goes through the orchestrator; processes never address each other.

```mermaid
flowchart LR
    Sales([Sales])
    Carrier([Carrier])
    Billing([Billing])
    Nobody[[no subscriber]]

    subgraph WH["group Warehouses: one copy, one taker"]
        direction TB
        Tartu([TartuWarehouse<br/>city: Tartu, Tapa])
        Tallinn([TallinnWarehouse<br/>city: Tallinn, Tapa])
    end

    Sales -->|"OrderPlaced<br/>(own copy)"| Billing
    Sales -->|"OrderPlaced<br/>(one copy)"| WH
    Carrier -->|"Truck<br/>(dock decides)"| WH
    WH -->|Shipment| Sales
    Sales -.->|Newsletter| Nobody
```

## Timeline

```mermaid
sequenceDiagram
    participant S as Sales
    participant B as Billing
    participant TA as TartuWarehouse
    participant TL as TallinnWarehouse
    participant C as Carrier

    Note over S,C: 09:00 ord1 Tartu
    S->>B: OrderPlaced ord1 (claimed)
    S->>TA: OrderPlaced ord1 (claimed, packed 10:00)
    Note over S: 09:05 Newsletter, warning: no subscribers
    Note over S,C: 09:10 ord2 Tallinn
    S->>B: OrderPlaced ord2 (claimed)
    S->>TL: OrderPlaced ord2 (claimed, packed 10:40)
    Note over S,C: 09:20 ord3 Tapa, both warehouses accept, random choice fixed by the seed
    Note over S,C: 09:30 ord4 Pärnu, Billing claims, both warehouses discard, ord4 stalls
    Note over S,C: 09:40 ord5 Tartu
    S->>TA: OrderPlaced ord5 (claimed, packed 10:40)
    Note over C: 09:45 Truck T0 Narva, both warehouses discard, warning: discarded by every recipient
    Note over S: 10:00 ord2 cancelled
    C->>TA: 10:30 Truck T1 Tartu (claimed)
    Note over TA: loads ord1 (+ ord3 if Tartu), ord5 not packed yet
    TA-->>S: Shipment ord1 (pending, ord1 not waiting yet)
    C->>TL: 12:00 Truck T2 Tallinn (claimed)
    TL-->>S: Shipment ord2 (discarded, order cancelled)
    Note over S: warning: Shipment ord2 discarded by every recipient
    Note over S: 12:00 ord1 starts waiting, pending Shipment ord1 claimed, effect at 12:00
    C->>TA: 13:00 Truck T3 Tartu (claimed, loads ord5)
    TA-->>S: Shipment ord5 (claimed)
    Note over C: 16:00 Truck T4 Tartu, nothing waiting, stays unclaimed
```

## Expected results

| Check        | Expected                                                                                                                                                           |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Announcement | Billing claims all five `OrderPlaced`; the Warehouses group gets one copy of each                                                                                  |
| Shared group | every order is claimed by at most one warehouse; ord1 and ord5 go to Tartu, ord2 to Tallinn                                                                        |
| Random tie   | ord3 goes to the same warehouse on every run with the same seed                                                                                                    |
| Pending      | `Shipment ord1` is published at 10:30 and takes effect at 12:00                                                                                                    |
| Discard      | ord4 and the Narva truck (T0) are discarded by both warehouses; `Shipment ord2` is discarded by Sales (case finished)                                              |
| Warnings     | exactly three: `Newsletter` (no subscribers), the Narva truck and `Shipment ord2` (each discarded by every recipient); ord4 raises none because Billing claimed it |
| Capacity     | no truck carries more than 2 orders; ord5 misses T1 and leaves on T3                                                                                               |
| End of run   | unclaimed: Truck T4 and `Shipment ord3`; discarded counts match what the engines discarded; ord4 still waiting in Sales                                            |
| Time         | step times never go backwards; the same seed gives the same run                                                                                                    |

`Shipment ord2` is a valid outcome, the order was cancelled after the warehouse packed it, but it is
still a negative one, so the orchestrator warns about it like any other message nobody could use.

Details that depend on ord3's warehouse (which truck carries it, when its shipment is published)
change with the seed. Tests check the rules that always hold for any seed, plus the exact outcome of
one fixed seed.

## First real-engine version

The first run in which a real Prosimos engine publishes and waits inside the protocol: Sales is a
Prosimos model, Billing and the two warehouses are scripted fake engines, given to `run_orchestrator`
as extra engines (`testing_scripts/real_sales_scenario.py`, checked by `testing_scripts/test_real_sales.py`).

```
Sales (real Prosimos, testing_scripts/assets/running_example/):
  start -> Place order -> throw OrderPlaced{case_id, city}
        -> catch Shipment (order_id == case_id) -> Close order -> end

Billing:                claims every OrderPlaced
TartuWarehouse:         claims OrderPlaced for Tartu and Tapa,   ships 1 h after claiming
TallinnWarehouse:       claims OrderPlaced for Tallinn and Tapa, ships 1.5 h after claiming
                        (Shipment{order_id}, where order_id is the order's case_id, e.g. Sales-3)
```

`city` is a case attribute drawn by Prosimos: Tartu 40%, Tallinn 30%, Tapa 20%, Pärnu 10%. Orders
arrive about every 30 minutes; consumer groups are as above (the warehouses share one group).

Compared with the scripted version, this one is simpler: no Carrier and no trucks (a warehouse
ships a fixed time after claiming), no Newsletter, and no canceled order.

What it shows, for seed 1 and 20 orders (7 Tartu, 5 Tallinn, 6 Tapa, 2 Pärnu):

| Check          | Result                                                                                                                                                   |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Shipped orders | every Tartu, Tallinn and Tapa order is closed after its shipment's time, shipped by a warehouse that serves its city; Tapa orders go to either warehouse |
| Pärnu orders   | Billing claims them, both warehouses discard them, so they wait forever: `finish()` reports them as the run's only stalled cases                         |
| Repeatability  | the same seed gives the same report and the same merged log                                                                                              |

The merged log holds only Sales rows, since the fake engines keep no log. Each order's
"Shipment received" wait shows in the log as the gap between Place order and Close order.

