# Catalogue and references

Some objects never have events of their own: customers, products, suppliers. Events and other objects only
refer to them, through a case attribute such as `customer = "c7"`. In a plain event log that value is just
text. In the OCEL output it should be an object, with its own attributes, linked to the orders it placed
and to the events that involved it.

Two keys of a process's entry in the configuration file (see [orchestrator.md](orchestrator.md)) do this. Each
process describes only itself, as a company holds its own master data:

- its **catalogues** declare the objects: who exists, and with which attributes;
- a **reference** says that one of its attributes' value *is* the id of an object of one of its catalogues,
  which OCEL links to write for it, and, with `pick`, chooses that value when a case is created; with `with`,
  the case also starts with attributes of the object it picked.

A first example, with two customers and Sales choosing one for every order:

```json
{"name": "Sales", "bpmn_path": "sales.bpmn", "json_path": "sales.json", "arrivals": 100,
 "catalogues": {
   "customer": [
     {"id": "c1", "attributes": {"segment": "retail"}},
     {"id": "c2", "attributes": {"segment": "business"}}
   ]
 },
 "references": [
   {"object_attribute": "customer", "catalogue": "customer",
    "o2o": "places", "reverse": true, "pick": "uniform"}
 ]}
```

Every Sales case gets `customer` set to `c1` or `c2` when it is created. The OCEL file then has the objects
`c1` and `c2` of type `customer`, each linked by `places` to its orders, and every event of an order linked
to its customer.

## Catalogue or process?

A catalogue is for objects that are only referred to. **Objects that do things belong in a process, not in
a catalogue.** A package is created, collects items and is shipped: those are events, so packages are the
cases of a Packaging process. A customer who only places orders is referred to by the orders and belongs
in the catalogue. If customers had events of their own as customers, e.g. registering or changing their
address, they would be the cases of a process too.

## The three catalogue forms

| Form    | Written as                                    | Ids                                     | Use it when                                                    | Written to OCEL                            |
|---------|-----------------------------------------------|-----------------------------------------|----------------------------------------------------------------|--------------------------------------------|
| List    | `[{"id": ..., "attributes": {...}}, ...]`     | as given                                | the objects are few and known, each with its own values        | every object, referred to or not           |
| Pool    | `{"count": N, "attributes": [...]}`           | `<name>-1` to `<name>-N`                | the set is fixed but too large to type out                     | only the objects some case refers to       |
| Growing | `{"attributes": [...]}`                       | `<name>-1`, `<name>-2`, ... as created  | the population isn't fixed: new customers keep arriving        | the objects cases created                  |

The catalogue's name (`customer` above) is the objects' OCEL type.

### List

```json
"customer": [
  {"id": "c1", "attributes": {"segment": "retail"}},
  {"id": "c2", "attributes": {"segment": "retail"}},
  {"id": "c3", "attributes": {"segment": "business"}}
]
```

- `id` (required): a non-empty string, the object's OCEL id.
- `attributes` (optional): name → value, any JSON values, written as given.

Every object is written, with its attributes timed at the start of the run, whether or not a case refers to
it.

### Pool

```json
"customer": {"count": 1500,
  "attributes": [
    {"name": "segment", "type": "discrete",
     "values": [{"key": "retail", "value": 0.8}, {"key": "business", "value": 0.2}]},
    {"name": "credit_limit", "type": "continuous",
     "values": {"distribution_name": "norm",
                "distribution_params": [{"value": 5000}, {"value": 1000}, {"value": 1000}, {"value": 9000}]}}]}
```

- `count` (required): a whole number, at least 1.
- `attributes` (optional): in exactly the format of a process's `object_attributes`, discrete or continuous,
  drawn by Prosimos's case attribute code.

The ids are `customer-1` to `customer-1500`. Each object's attributes are drawn once, at the start of the
run, from a generator seeded from the seed and the catalogue's name, so the same seed gives the same pool.
Only the objects that some case's referenced attribute names are written, with their attributes timed at
the start of the run: with 2,000 orders picking uniformly from 1,500 customers, about 1,100 of them.

### Growing

```json
"customer": {"attributes": [
  {"name": "segment", "type": "discrete",
   "values": [{"key": "retail", "value": 0.8}, {"key": "business", "value": 0.2}]}]}
```

- `attributes` (optional): as for a pool. There is no list and no `count`.

The catalogue starts empty. One reference creates its objects as cases are created, with `"pick": "new"` or
`{"new_share": ...}` (see below). Ids are numbered in creation order: planned cases are created in case
order when the engine is built, message-started ones when their message arrives. Each new object's
attributes are drawn in creation order, from a generator seeded from the seed and the catalogue's name, so
the same seed gives the same objects. They are timed when the first case took the object's id.

## References

A reference, in a process's `references`, ties one of the process's attributes to one of its catalogues:

| Key                | Required | Example                            | Meaning                                                                                                         |
|--------------------|----------|------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `object_attribute` | yes      | `"customer"`                       | the attribute whose value is a catalogue id                                                                     |
| `catalogue`        | yes      | `"customer"`                       | the process's catalogue the id belongs to                                                                       |
| `o2o`              | no       | `"places"`                         | qualifier of a link between the case's object and the catalogue object; without it, no such link                |
| `reverse`          | no       | `true`                             | turns the `o2o` link around: from the catalogue object to the case's object (default `false`)                   |
| `link_events`      | no       | `["place order", "Confirm_Order"]` | which of the case's events link the catalogue object: task names or task ids; without it every event, `[]` none |
| `pick`             | no       | `"uniform"`                        | the reference chooses the value itself (see "Pick")                                                             |
| `with`             | no       | `["weight", "price"]`              | attributes of the picked object that the case also starts with (see "With")                                     |

### Two modes: with and without pick

**With `pick`, the reference assigns the value.** When a case is created, the reference chooses a catalogue
object and sets the attribute to its id, instead of `object_attributes`. Order Management picks its customer
this way:

```json
{"object_attribute": "customer", "catalogue": "customer", "pick": "uniform"}
```

The picked value is a normal case attribute from then on: it is a column of the log, and messages and
`copy` carry it like any other. The attribute must therefore not be in the process's `object_attributes`, and
none of the process's own consume entries may copy, append or aggregate into it, since that value would replace
the picked one. A process JSON whose messages publish a picked attribute only runs inside its configuration:
run on its own, the attribute isn't declared (Order Management's `sales.json` is an example).

**Without `pick`, the reference interprets a value that is already there**: drawn by the process's
`object_attributes`, or copied from a message. If Sales drew `customer` in its `object_attributes` from c1 to c15,
the same reference without `pick` would make the drawn text an object:

```json
{"object_attribute": "customer", "catalogue": "customer", "o2o": "places", "reverse": true}
```

**A reference names a catalogue of its own process**, never another's: what one process knows reaches another
only in messages, copied under any name the receiver chooses. Catalogue names are unique across the
configuration, since they are object types of the OCEL output, and ids are unique across all catalogues. In
Order Management, the items get their order's customer from `ItemOrdered` as an attribute; the customers
belong to Sales, so the items aren't linked to them.

A value that isn't an id of the catalogue gets no link, and a warning (see "Errors and warnings").

### The links

- **`o2o`**: the object of every case of the process is linked to the catalogue object of each value its
  attribute took, with `o2o` as qualifier. With `"o2o": "placed by"`, order `order-13` is linked to
  customer `c1`; with `"o2o": "places", "reverse": true`, customer `c1` is linked to order `order-13`.
- **`link_events`**: each event of the case named there is linked to the catalogue object of the value the
  attribute had when the event completed, with the attribute's name as qualifier. With
  `"link_events": ["place order", "confirm order"]`, place order and confirm order link `customer`, pay
  order doesn't. `"Confirm_Order"`, the task's id, works as well as its name. Without the key, every
  event of the case is linked; with `[]`, none.
- A process without an `object_type` writes no objects and no events, so its references link nothing;
  their values still count as referred to, so the pool objects they name are written.

## Pick

| Pick                        | Catalogue    | Each case gets                                                            |
|-----------------------------|--------------|---------------------------------------------------------------------------|
| `"uniform"`                 | list or pool | any object, all equally likely                                            |
| `{"weights": {id: w, ...}}` | list         | an object with chances in proportion to the weights                       |
| `"new"`                     | growing      | an object of its own, created for it                                      |
| `{"new_share": s}`          | growing      | a new object for share `s` of the cases; for the rest, one created so far |

The tables below are the picks of the first 6 orders of a Sales process (object type `order`) with seed 1, as a run gives them;
the list examples use the catalogue `c1`, `c2`, `c3`.

**`"uniform"`**: every object equally likely.

| Order    | order-0 | order-1 | order-2 | order-3 | order-4 | order-5 |
|----------|---------|---------|---------|---------|---------|---------|
| customer | c2      | c2      | c2      | c3      | c1      | c1      |

**`{"weights": {"c1": 3, "c2": 1, "c3": 1}}`**: c1 three times as likely as c2 or c3, i.e. 60/20/20%.
Ids left out of `weights` are never picked. Weights are numbers of at least 0, not all 0. A list only: a
pool's ids aren't known by name.

| Order    | order-0 | order-1 | order-2 | order-3 | order-4 | order-5 |
|----------|---------|---------|---------|---------|---------|---------|
| customer | c1      | c1      | c2      | c3      | c1      | c1      |

**`"new"`**: every case creates its own object, 1 case ↔ 1 object.

| Order    | order-0    | order-1    | order-2    | order-3    | order-4    | order-5    |
|----------|------------|------------|------------|------------|------------|------------|
| customer | customer-1 | customer-2 | customer-3 | customer-4 | customer-5 | customer-6 |

**`{"new_share": 0.3}`**: 30% of the cases create a new object, the rest reuse one created so far,
uniformly. The first case always creates one. `new_share` is above 0 and at most 1; 1 is the same as
`"new"`.

| Order    | order-0          | order-1    | order-2          | order-3    | order-4    | order-5    |
|----------|------------------|------------|------------------|------------|------------|------------|
| customer | customer-1 (new) | customer-1 | customer-2 (new) | customer-2 | customer-2 | customer-1 |

Since reuse is uniform among the objects created so far, the earliest objects are candidates longest and
end up with the most cases.

Picks are drawn when the case is created, from a generator of their own, seeded from the seed, the process
and the attribute. The same seed therefore gives the same picks, and adding a pick doesn't change any of
the process's other draws. Engines stay black boxes: the orchestrator hands a pick to an engine when it
builds it, as one more case attribute, and joins the links from the case objects and log rows after the
run.

### With

```json
{"object_attribute": "product", "catalogue": "product",
 "pick": {"weights": {...}}, "with": ["weight", "price"]}
```

```
item item-0 picks "iPad Air"  ->  product = "iPad Air", weight = 0.44, price = 476
```

`with` lists attributes of the catalogue object. When the reference picks an object, the case also gets each
listed attribute, under the same name, as one of its starting values. From then on they are case values like
those drawn in `object_attributes`: columns of the log, attributes of the case's OCEL object, and usable in
conditions, `copy`, published messages and `aggregate`.

`with` needs a `pick` on a list or a pool, whose objects and attributes exist when the case is created, and
every object of the catalogue must have each listed attribute. A listed attribute draws nothing: adding
`with` changes neither the picks nor any other draw.

## Worked example: Order Management

The Order Management example ([order-management.md](order-management.md)) has 15 customers in a catalogue of
Sales, which picks one uniformly for each order, and the real log's 20 products in a catalogue of the
Warehouse, which picks one for each item as often as in the log. This is its full configuration,
`testing_scripts/assets/order_management/simulation.json`:

```json
{
  "processes": [
    {
      "name": "Sales",
      "bpmn_path": "sales.bpmn",
      "json_path": "sales.json",
      "arrivals": 2000,
      "object_type": "order",
      "qualifier": "order",
      "carry_links": true,
      "catalogues": {
        "customer": [
          {"id": "c1", "attributes": {"segment": "retail"}},
          {"id": "c2", "attributes": {"segment": "retail"}},
          {"id": "c3", "attributes": {"segment": "business"}},
          {"id": "c4", "attributes": {"segment": "retail"}},
          {"id": "c5", "attributes": {"segment": "retail"}},
          {"id": "c6", "attributes": {"segment": "business"}},
          {"id": "c7", "attributes": {"segment": "retail"}},
          {"id": "c8", "attributes": {"segment": "retail"}},
          {"id": "c9", "attributes": {"segment": "business"}},
          {"id": "c10", "attributes": {"segment": "retail"}},
          {"id": "c11", "attributes": {"segment": "retail"}},
          {"id": "c12", "attributes": {"segment": "business"}},
          {"id": "c13", "attributes": {"segment": "retail"}},
          {"id": "c14", "attributes": {"segment": "retail"}},
          {"id": "c15", "attributes": {"segment": "business"}}
        ]
      },
      "references": [
        {"object_attribute": "customer", "catalogue": "customer",
         "o2o": "places", "reverse": true, "link_events": ["place order", "confirm order"],
         "pick": "uniform"}
      ]
    },
    {
      "name": "Customer",
      "bpmn_path": "customer.bpmn",
      "json_path": "customer.json"
    },
    {
      "name": "Warehouse",
      "bpmn_path": "warehouse.bpmn",
      "json_path": "warehouse.json",
      "object_type": "item",
      "qualifier": "item",
      "catalogues": {
        "product": [
          {"id": "iPhone X", "attributes": {"weight": 0.172, "price": 699}},
          {"id": "iPhone 8", "attributes": {"weight": 0.21, "price": 529}},
          {"id": "Fire Stick", "attributes": {"weight": 0.2, "price": 39.99}},
          {"id": "iPad Air", "attributes": {"weight": 0.44, "price": 476}},
          {"id": "Fire Stick 4K", "attributes": {"weight": 0.28, "price": 89.99}},
          {"id": "iPad Pro", "attributes": {"weight": 0.483, "price": 1099}},
          {"id": "Echo Plus", "attributes": {"weight": 1.28, "price": 149.99}},
          {"id": "Kindle Paperwhite", "attributes": {"weight": 0.495, "price": 129}},
          {"id": "Echo Show 8", "attributes": {"weight": 0.98, "price": 129.99}},
          {"id": "MacBook Pro", "attributes": {"weight": 1.37, "price": 2500}},
          {"id": "iPad mini", "attributes": {"weight": 0.28, "price": 449}},
          {"id": "Echo Studio", "attributes": {"weight": 1.48, "price": 199.99}},
          {"id": "Echo Dot", "attributes": {"weight": 0.38, "price": 29.99}},
          {"id": "iPhone 11 Pro", "attributes": {"weight": 0.188, "price": 1149}},
          {"id": "MacBook Air", "attributes": {"weight": 1.25, "price": 2200}},
          {"id": "Echo Show 5", "attributes": {"weight": 0.88, "price": 89.99}},
          {"id": "Echo", "attributes": {"weight": 0.78, "price": 99.99}},
          {"id": "iPad", "attributes": {"weight": 0.483, "price": 495}},
          {"id": "Kindle", "attributes": {"weight": 0.483, "price": 79.99}},
          {"id": "iPhone 11", "attributes": {"weight": 0.166, "price": 799}}
        ]
      },
      "references": [
        {"object_attribute": "product", "catalogue": "product", "o2o": "is a",
         "pick": {"weights": {
           "iPhone X": 433, "iPhone 8": 408, "Fire Stick": 405, "iPad Air": 405, "Fire Stick 4K": 400,
           "iPad Pro": 394, "Echo Plus": 394, "Kindle Paperwhite": 393, "Echo Show 8": 383, "MacBook Pro": 382,
           "iPad mini": 382, "Echo Studio": 379, "Echo Dot": 375, "iPhone 11 Pro": 373, "MacBook Air": 369,
           "Echo Show 5": 368, "Echo": 365, "iPad": 362, "Kindle": 346, "iPhone 11": 343}},
         "with": ["weight", "price"]}
      ]
    },
    {
      "name": "Packaging",
      "bpmn_path": "packaging.bpmn",
      "json_path": "packaging.json",
      "arrivals": 1200,
      "object_type": "package",
      "qualifier": "package",
      "qualifier_by_activity": {
        "Create_Package": "creates",
        "Send_Package": "shipped package",
        "Failed_Delivery": "shipped package",
        "Package_Delivered": "shipped package"
      },
      "carry_links": true
    }
  ],
  "seed": 1,
  "start_time": "2023-04-03T06:00:00+00:00"
}
```

Run it with

```
poetry run prosimos start-orchestration --config testing_scripts/assets/order_management/simulation.json --ocel_out_path om.json
```

`sales.json` has no `customer` in its `object_attributes`: the reference picks it. Its `ItemOrdered` message
then carries it to the items, and `ItemPicked` to the packages. Likewise, `warehouse.json` has no `product`:
the Warehouse's reference picks it, with the number of items of each product in the log as weights, and brings
the product's `weight` and `price` into the item. `ItemPicked` carries the item's `weight` to the package.

**What appears in the OCEL file** (excerpts of that run, 2,000 orders):

- **The 15 customers**, type `customer`, with `segment` timed at the start of the run, and `objectTypes`
  lists `{"name": "customer", "attributes": [{"name": "segment", "type": "string"}]}`. Each is linked by
  `places` to its orders, 109 to 154 per customer, 2,000 in all:

  ```json
  {"id": "c3", "type": "customer",
   "attributes": [{"name": "segment", "value": "business", "time": "2023-04-03T06:00:00.000000Z"}],
   "relationships": [{"objectId": "order-25", "qualifier": "places"},
                     {"objectId": "order-29", "qualifier": "places"}, ...]}
  ```

- **The orders** keep `customer` as an attribute:

  ```json
  {"id": "order-0", "type": "order",
   "attributes": [{"name": "items", "value": 5, "time": "2023-04-03T06:00:00.000000Z"},
                  {"name": "customer", "value": "c9", "time": "2023-04-03T06:00:00.000000Z"}], ...}
  ```

- **place order and confirm order** are linked to their order's customer, with the qualifier `customer`
  (4,000 links: 2,000 of each); **pay order** isn't, since it isn't in `link_events`:

  ```json
  {"id": "e1", "type": "place order", "time": "2023-04-03T06:01:00.000000Z",
   "relationships": [{"objectId": "order-0", "qualifier": "order"},
                     {"objectId": "item-0", "qualifier": "item"}, ...,
                     {"objectId": "c9", "qualifier": "customer"}]}
  {"id": "e16", "type": "confirm order", "time": "2023-04-03T09:55:27.683160Z",
   "relationships": [{"objectId": "order-0", "qualifier": "order"}, ...,
                     {"objectId": "c9", "qualifier": "customer"}]}
  {"id": "e523", "type": "pay order", "time": "2023-04-11T20:30:28.616308Z",
   "relationships": [{"objectId": "order-0", "qualifier": "order"}, ...]}
  ```

- **The 20 products**, type `product`, with `weight` and `price` as in the log. Each item **is a** product
  (`"o2o": "is a"`, from the item to the product), keeps `product`, `weight` and `price` as attributes, and
  every one of its events is linked to the product with the qualifier `product` (no `link_events`):

  ```json
  {"id": "item-0", "type": "item",
   "attributes": [{"name": "product", "value": "iPad Air", "time": "2023-04-03T06:01:00.000000Z"},
                  {"name": "weight", "value": 0.44, "time": "2023-04-03T06:01:00.000000Z"},
                  {"name": "price", "value": 476, "time": "2023-04-03T06:01:00.000000Z"}, ...],
   "relationships": [{"objectId": "iPad Air", "qualifier": "is a"}]}
  {"id": "e15", "type": "pick item", "time": "2023-04-03T09:27:32.013771Z",
   "relationships": [{"objectId": "item-0", "qualifier": "item"}, ...,
                     {"objectId": "iPad Air", "qualifier": "product"}]}
  ```

Items and packages carry `customer` too, but only as an attribute: the customers are a catalogue of Sales,
and a reference names a catalogue of its own process. To have every order come from a new customer instead,
replace the list with `"customer": {"attributes": [...]}` and the pick with `"new"`.

## Errors and warnings

**Load errors.** A configuration is checked when it is built (`SimulationConfig(...)` or
`SimulationConfig.from_json`). Two errors stop reading the file at once, prefixed by its path:

| Message                                                                | Cause                                                 |
|------------------------------------------------------------------------|-------------------------------------------------------|
| `processes[0].catalogues.customer[0]: 'id' must be a non-empty string` | a list object without an `id`, or an empty one        |
| `processes[0].references[0] is missing ['object_attribute']`           | a reference without `object_attribute` or `catalogue` |

All the others are collected and raised together, as one `ValueError` starting `Invalid catalogue: `, the
problems separated by `; `. A growing catalogue with only a `"uniform"` pick, for example, breaks two rules:

```
Invalid catalogue: growing catalogue 'customer' needs exactly one reference that creates its objects ("new" or new_share), got 0; processes[0].references[0] picks from growing catalogue 'customer', which only takes "new" or {"new_share": ...}
```

The messages, by the key they are about:

| Key                          | Message                                                                                                                                                        |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`                         | `catalogue ids are repeated: ['x']` (ids are unique across all catalogues)                                                                                     |
| `id`                         | `catalogue ids look like object ids (<object type>-<number>): ['order-1']`                                                                                     |
| catalogue name               | `catalogue 'order' is the object type of process Sales`                                                                                                        |
| `count`                      | `catalogue 'customer': count must be a whole number of at least 1, got 0`                                                                                      |
| `attributes` (pool, growing) | `catalogue 'customer': attributes[0]: InvalidCaseAttributeException: Case attribute segment: probabilities' sum should be equal to 1` (Prosimos's own message) |
| growing catalogue            | `catalogue ids look like the ids growing catalogue 'customer' creates (customer-<number>): ['customer-2']`                                                     |
| growing catalogue            | `growing catalogue 'customer' needs exactly one reference that creates its objects ("new" or new_share), got 0`                                                |
| `catalogue`                  | `processes[0].references[0] names unknown catalogue 'clients'`                                                                                                 |
| `catalogue`                  | `processes[2].references[0] names catalogue 'customer' of process Sales; a reference names a catalogue of its own process`                                     |
| catalogue name               | `catalogue 'customer' is declared by more than one process: Sales, Warehouse`                                                                                  |
| `link_events`                | `processes[0].references[0] link_events names ['confrim order'], which are no tasks of process Sales`                                                          |
| `pick`                       | `processes[0].references[0] pick must be "uniform", {"weights": {id: weight, ...}}, "new" or {"new_share": share}`                                             |
| `pick`                       | `processes[0].references[0] picks customer, which process Sales also draws in its object_attributes`                                                           |
| `pick`                       | `processes[2].references[0] picks customer, which process Warehouse also copies from a message (consume entries at ['Start_Item'])`                            |
| `pick`                       | `processes[3].references[0] picks item_ids, which process Packaging also appends to from a message (consume entries at ['Catch_Items'])`                       |
| `pick`                       | `more than one reference picks customer of process Sales`                                                                                                      |
| `weights`                    | `processes[0].references[0] weights name ['c99'], which are no ids of catalogue 'customer'`                                                                    |
| `weights`                    | `processes[0].references[0] weights must be numbers of at least 0, not all 0, got {'c1': 0, 'c2': 0}`                                                          |
| `weights`                    | `processes[0].references[0] has weights on pool 'customer'; a pool can only be picked from uniformly`                                                          |
| `"new"`, `new_share`         | `processes[0].references[0] picks "new" or new_share from catalogue 'customer', which is no growing catalogue`                                                 |
| `"new"`, `new_share`         | `processes[0].references[0] picks from growing catalogue 'customer', which only takes "new" or {"new_share": ...}`                                             |
| `new_share`                  | `processes[0].references[0] new_share must be a number above 0 and at most 1, got 1.5`                                                                         |
| `with`                       | `processes[2].references[0] with must be a list of attribute names, got 'weight'`                                                                              |
| `with`                       | `processes[2].references[0] has with but no pick; a catalogue object brings its attributes only when the reference picks it`                                   |
| `with`                       | `processes[2].references[0] has with on growing catalogue 'product', whose objects don't exist yet when a case is created`                                     |
| `with`                       | `processes[2].references[0] with names ['volume'], which not every object of catalogue 'product' has`                                                          |
| `with`                       | `processes[2].references[0] brings price, which process Warehouse also draws in its object_attributes`                                                         |
| `with`                       | `processes[2].references[0] brings customer, which process Warehouse also copies from a message (consume entries at ['Start_Item'])`                           |
| `with`                       | `processes[2].references[0] brings product, which process Warehouse also picks`                                                                                |
| `with`                       | `processes[2].references[0] brings weight, which another with also brings to process Warehouse`                                                                |
| `with`                       | `processes[2].references[0] brings object_id, which is reserved for the object's own id`                                                                       |

`link_events` is not checked for extra engines, which have no BPMN model.

**Warning.** One kind, during the run, in the run report's `warnings`, once per attribute and value:

```
case attribute customer has the value 'c99', which is not in catalogue customer; no OCEL link is written for it
```

A value is compared with the ids as text, so the number `7` matches the id `"7"`. The value keeps no link,
but everything else about the case is written as usual.

**Outside the configuration.** A process JSON whose messages publish a picked attribute fails on its own,
when Prosimos reads its messages: `... 'customer' not declared as a case, global or event attribute`. Run
it through its configuration, where the pick declares the attribute.

## Limitations

- **One creator per growing catalogue.** Only one reference creates a growing catalogue's objects, so a
  process can't create them through two attributes (e.g. the buyer and the recipient of an order). Two
  processes never share a catalogue: each keeps its own, and what one knows reaches the other in messages.
