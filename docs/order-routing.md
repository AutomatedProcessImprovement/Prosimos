# Order Routing example: orders, invoices and shipments

One scenario, used throughout this documentation and in the tests, that shows how processes
exchange messages through the orchestrator ([orchestrator.md](orchestrator.md)): orders placed in
Sales are billed and routed to two competing warehouses, one of them a partner's; the one that takes an
order packs and ships it, and Sales closes it. All four processes are real
Prosimos models in `testing_scripts/assets/order_routing/`, run from one configuration file,
`simulation.json` (checked by
`testing_scripts/multiprocess/examples/test_order_routing.py`).

A larger example, with orders, items and packages, is the [Order Management example](order-management.md).

## The scenario

![Process models of the Order Routing example](order-routing-processes.svg)

### Processes and messages

| Process              | Its cases  | Publishes                                      | Consumes                                                          |
|----------------------|------------|------------------------------------------------|-------------------------------------------------------------------|
| **Sales**            | `order`    | `OrderPlaced{object_id, city}`                 | `Shipment` where `order_id = object_id`                           |
| **Billing**          | `invoice`  | nothing                                        | every `OrderPlaced`, at its start event                           |
| **TartuWarehouse**   | `shipment` | `Shipment{order_id}`, once the order is packed | `OrderPlaced` where `city` is Tartu or Tapa, at its start event   |
| **TallinnWarehouse** | `shipment` | `Shipment{order_id}`, once the order is packed | `OrderPlaced` where `city` is Tallinn or Tapa, at its start event |

```
Sales:             start -> Place order -> throw OrderPlaced{object_id, city}
                   -> catch Shipment (order_id = object_id) -> Close order -> end
Billing:           message start OrderPlaced (any), copy order_id <- object_id -> Create invoice -> end
TartuWarehouse:    message start OrderPlaced (city Tartu or Tapa), copy order_id <- object_id
                   -> Pack order (1 h) -> message end Shipment{order_id}
TallinnWarehouse:  the same for Tallinn or Tapa, Pack order 1.5 h

consumer_groups:   Sales: [Sales]   Billing: [Billing]   Warehouses: [TartuWarehouse, TallinnWarehouse]
```

Orders arrive about every 30 minutes (`arrivals: 20`), each with a `city` drawn by Prosimos: Tartu
40%, Tallinn 30%, Tapa 20%, Pärnu 10%. Only Sales has an arrival schedule; the other three are
started by `OrderPlaced` messages, so they have no `arrivals` and no arrival settings. Billing and
the warehouses copy the order's `object_id` into `order_id`, so the merged log shows which order each
of their cases belongs to, and a warehouse's `Shipment` carries it back to Sales.

### Who talks to whom

Every arrow goes through the orchestrator; processes never address each other.

```mermaid
flowchart LR
    Sales([Sales])
    Billing([Billing])

    subgraph WH["group Warehouses: one copy, one taker"]
        direction TB
        Tartu([TartuWarehouse<br/>city: Tartu, Tapa])
        Tallinn([TallinnWarehouse<br/>city: Tallinn, Tapa])
    end

    Sales -->|"OrderPlaced<br/>(own copy)"| Billing
    Sales -->|"OrderPlaced<br/>(one copy)"| WH
    WH -->|Shipment| Sales
```

Billing is a group of its own, so it gets its own copy of every order. The two warehouses share one
group: each order is offered to both, and the first that accepts it takes it. A Tapa order suits
both, so which one packs it is a random choice of the orchestrator, fixed by the seed. A Pärnu order
suits neither: both discard it, and its Sales case waits for a shipment that never comes.

The processes belong to two organizations ([orchestrator.md](orchestrator.md), "Organizations"): Shop
runs Sales, Billing and TartuWarehouse, and Tallinn partner runs TallinnWarehouse. The Warehouses group spans
both, so a partner's warehouse competes with Shop's for the same orders; routing doesn't depend on organizations.

The sample leaves the Warehouses group random. To send every Tapa order to Shop's own warehouse first, and to
Tallinn partner's only when TartuWarehouse discards it, the configuration can add a policy
([orchestrator.md](orchestrator.md), "Message routing"):

```json
"load_balancing": {"Warehouses": {"preferred": ["TartuWarehouse", "TallinnWarehouse"]}}
```

`{"weighted": {"TartuWarehouse": 8, "TallinnWarehouse": 2}}` would instead ask TartuWarehouse first for about
80% of the Tapa orders.

## Running it

From the command line (the [Quick start](orchestrator.md#quick-start) shows the configuration):

```
poetry run prosimos start-orchestration --config testing_scripts/assets/order_routing/simulation.json --log_out_path merged_log.csv --ocel_out_path order_routing.json
```

or from Python:

```python
from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation

config = SimulationConfig.from_json("testing_scripts/assets/order_routing/simulation.json")
report = run_multiprocess_simulation(config, log_out_path="merged_log.csv", ocel_out_path="order_routing.json")
```

## What a run shows

For seed 1 and 20 orders (8 Tartu, 2 Tallinn, 7 Tapa, 3 Pärnu):

| Process          | Cases                                                                                                         |
|------------------|---------------------------------------------------------------------------------------------------------------|
| Sales            | 20 orders; 17 closed, each after its shipment; the 3 Pärnu orders wait forever and are the only stalled cases |
| Billing          | 20 invoices, one per order, Pärnu included                                                                    |
| TartuWarehouse   | 12 cases: the 8 Tartu orders and 4 of the Tapa orders                                                         |
| TallinnWarehouse | 5 cases: the 2 Tallinn orders and the other 3 Tapa orders                                                     |

Every order from Tartu, Tallinn or Tapa has exactly one warehouse case, in a warehouse that serves its
city, and is closed after its shipment. There are no warnings and no unclaimed messages, and the same
seed gives the same run.

### The OCEL output

Each process names the type of the objects its cases are, in the singular: `order` (Sales), `invoice`
(Billing) and `shipment` (both warehouses, which make the same kind of thing). The two warehouses each
number their own shipments from `shipment-0`, as the run report and the messages show them; the OCEL
output numbers them across both, in the order they were created, so its 17 shipments are `shipment-0` to
`shipment-16` ([orchestrator.md](orchestrator.md), "Output"). For seed 1 the file has 57 objects (20
orders, 20 invoices, 17 shipments) and 128 events: 74 tasks, and 54 message events, which a message end with a
qualifier writes per case, element and moment: 37 Order placed (20 at Billing, 17 at the warehouses) and 17
Shipment received.

Each `OrderPlaced` a case starts from links it to its order: an invoice `bills` the order, and its message
start Order placed links it as the `billed order`; a shipment `ships` it, and its Order placed links it as the
`packed order`. Sales's `Shipment` entry has its catch event Shipment received link the shipment that ships the
order, as the `received shipment`. The tasks link only their own case's object. Sales's own `OrderPlaced` entry
has no qualifier, so its throw event writes no event.
