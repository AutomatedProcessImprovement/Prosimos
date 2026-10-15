# Catalogue and references

Some objects never have events of their own: customers, products, suppliers. Events and other objects only
refer to them, through a case attribute such as `customer = "c7"`. In a plain event log that value is just
text. In the OCEL output it should be an object, with its own attributes, linked to the orders it placed
and to the events that involved it.

Two parts of the configuration file (see [orchestrator.md](orchestrator.md)) do this:

- the **catalogue** declares the objects: who exists, and with which attributes;
- a **reference** says that a case attribute's value *is* the id of a catalogue object, which OCEL links to
  write for it, and, with `pick`, chooses that value when a case is created.

A first example, with two customers and Sales choosing one for every order:

```json
"catalogue": {
  "customers": [
    {"id": "c1", "attributes": {"segment": "retail"}},
    {"id": "c2", "attributes": {"segment": "business"}}
  ]
},
"references": [
  {"process": "Sales", "case_attribute": "customer", "catalogue": "customers",
   "o2o": "places", "reverse": true, "pick": "uniform"}
]
```

Every Sales case gets `customer` set to `c1` or `c2` when it is created. The OCEL file then has the objects
`c1` and `c2` of type `customers`, each linked by `places` to its orders, and every event of an order linked
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

The catalogue's name (`customers` above) is the objects' OCEL type.

### List

```json
"customers": [
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
"customers": {"count": 1500,
  "attributes": [
    {"name": "segment", "type": "discrete",
     "values": [{"key": "retail", "value": 0.8}, {"key": "business", "value": 0.2}]},
    {"name": "credit_limit", "type": "continuous",
     "values": {"distribution_name": "norm",
                "distribution_params": [{"value": 5000}, {"value": 1000}, {"value": 1000}, {"value": 9000}]}}]}
```

- `count` (required): a whole number, at least 1.
- `attributes` (optional): in exactly the format of a process's `case_attributes`, discrete or continuous,
  drawn by Prosimos's case attribute code.

The ids are `customers-1` to `customers-1500`. Each object's attributes are drawn once, at the start of the
run, from a generator seeded from the seed and the catalogue's name, so the same seed gives the same pool.
Only the objects that some case's referenced attribute names are written, with their attributes timed at
the start of the run: with 2,000 orders picking uniformly from 1,500 customers, about 1,100 of them.

### Growing

```json
"customers": {"attributes": [
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

A reference ties one case attribute of one process to a catalogue:

| Key              | Required | Example                            | Meaning                                                                                                         |
|------------------|----------|------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `process`        | yes      | `"Sales"`                          | the process whose cases have the attribute                                                                      |
| `case_attribute` | yes      | `"customer"`                       | the attribute whose value is a catalogue id                                                                     |
| `catalogue`      | yes      | `"customers"`                      | the catalogue the id belongs to                                                                                 |
| `o2o`            | no       | `"places"`                         | qualifier of a link between the case's object and the catalogue object; without it, no such link                |
| `reverse`        | no       | `true`                             | turns the `o2o` link around: from the catalogue object to the case's object (default `false`)                   |
| `link_events`    | no       | `["place order", "Confirm_Order"]` | which of the case's events link the catalogue object: task names or task ids; without it every event, `[]` none |
| `pick`           | no       | `"uniform"`                        | the reference chooses the value itself (see "Pick")                                                             |

### Two modes: with and without pick

**With `pick`, the reference assigns the value.** When a case is created, the reference chooses a catalogue
object and sets the attribute to its id, instead of `case_attributes`. Order Management picks its customer
this way:

```json
{"process": "Sales", "case_attribute": "customer", "catalogue": "customers", "pick": "uniform"}
```

The picked value is a normal case attribute from then on: it is a column of the log, and messages and
`copy` carry it like any other. The attribute must therefore not be in the process's `case_attributes`, and
none of the process's own consume entries may copy or append into it, since the copied value would replace
the picked one. A process JSON whose messages publish a picked attribute only runs inside its configuration:
run on its own, the attribute isn't declared (Order Management's `sales.json` is an example).

**Without `pick`, the reference interprets a value that is already there**: drawn by the process's
`case_attributes`, or copied from a message. In Order Management, the items get their order's customer from
`ItemOrdered`. A reference on the Warehouse makes that copied text an object again, and links each item to
its customer:

```json
{"process": "Warehouse", "case_attribute": "customer", "catalogue": "customers",
 "o2o": "ordered by", "link_events": []}
```

A value that isn't an id of the catalogue gets no link, and a warning (see "Errors and warnings").

### The links

- **`o2o`**: the object of every case of the process is linked to the catalogue object of each value its
  attribute took, with `o2o` as qualifier. With `"o2o": "placed by"`, order `Sales-13` is linked to
  customer `c1`; with `"o2o": "places", "reverse": true`, customer `c1` is linked to order `Sales-13`.
- **`link_events`**: each event of the case named there is linked to the catalogue object of the value the
  attribute had when the event completed, with the attribute's name as qualifier. With
  `"link_events": ["place order", "confirm order"]`, place order and confirm order link `customer`, pay
  order doesn't. `"Confirm_Order"`, the task's id, works as well as its name. Without the key, every
  event of the case is linked; with `[]`, none.
- A process with `"object_type": null` writes no objects and no events, so its references link nothing;
  their values still count as referred to, so the pool objects they name are written.

## Pick

| Pick                        | Catalogue    | Each case gets                                                            |
|-----------------------------|--------------|---------------------------------------------------------------------------|
| `"uniform"`                 | list or pool | any object, all equally likely                                            |
| `{"weights": {id: w, ...}}` | list         | an object with chances in proportion to the weights                       |
| `"new"`                     | growing      | an object of its own, created for it                                      |
| `{"new_share": s}`          | growing      | a new object for share `s` of the cases; for the rest, one created so far |

The tables below are the picks of the first 6 orders of a Sales process with seed 1, as a run gives them;
the list examples use the catalogue `c1`, `c2`, `c3`.

**`"uniform"`**: every object equally likely.

| Order    | Sales-0 | Sales-1 | Sales-2 | Sales-3 | Sales-4 | Sales-5 |
|----------|---------|---------|---------|---------|---------|---------|
| customer | c2      | c2      | c2      | c3      | c1      | c1      |

**`{"weights": {"c1": 3, "c2": 1, "c3": 1}}`**: c1 three times as likely as c2 or c3, i.e. 60/20/20%.
Ids left out of `weights` are never picked. Weights are numbers of at least 0, not all 0. A list only: a
pool's ids aren't known by name.

| Order    | Sales-0 | Sales-1 | Sales-2 | Sales-3 | Sales-4 | Sales-5 |
|----------|---------|---------|---------|---------|---------|---------|
| customer | c1      | c1      | c2      | c3      | c1      | c1      |

**`"new"`**: every case creates its own object, 1 case ↔ 1 object.

| Order    | Sales-0     | Sales-1     | Sales-2     | Sales-3     | Sales-4     | Sales-5     |
|----------|-------------|-------------|-------------|-------------|-------------|-------------|
| customer | customers-1 | customers-2 | customers-3 | customers-4 | customers-5 | customers-6 |

**`{"new_share": 0.3}`**: 30% of the cases create a new object, the rest reuse one created so far,
uniformly. The first case always creates one. `new_share` is above 0 and at most 1; 1 is the same as
`"new"`.

| Order    | Sales-0           | Sales-1     | Sales-2           | Sales-3     | Sales-4     | Sales-5     |
|----------|-------------------|-------------|-------------------|-------------|-------------|-------------|
| customer | customers-1 (new) | customers-1 | customers-2 (new) | customers-2 | customers-2 | customers-1 |

Since reuse is uniform among the objects created so far, the earliest objects are candidates longest and
end up with the most cases.

Picks are drawn when the case is created, from a generator of their own, seeded from the seed, the process
and the attribute. The same seed therefore gives the same picks, and adding a pick doesn't change any of
the process's other draws. Engines stay black boxes: the orchestrator hands a pick to an engine when it
builds it, as one more case attribute, and joins the links from the case objects and log rows after the
run.

## Worked example: Order Management

The Order Management example ([order-management.md](order-management.md)) has 15 customers, picked
uniformly by Sales. This is its full configuration,
`testing_scripts/assets/order_management/simulation.json`:

```json
{
  "processes": [
    {
      "name": "Sales",
      "bpmn_path": "sales.bpmn",
      "json_path": "sales.json",
      "total_cases": 2000,
      "object_type": "orders",
      "qualifier": "order",
      "carry_links": true
    },
    {
      "name": "Customer",
      "bpmn_path": "customer.bpmn",
      "json_path": "customer.json",
      "object_type": null
    },
    {
      "name": "Warehouse",
      "bpmn_path": "warehouse.bpmn",
      "json_path": "warehouse.json",
      "object_type": "items",
      "qualifier": "item"
    },
    {
      "name": "Packaging",
      "bpmn_path": "packaging.bpmn",
      "json_path": "packaging.json",
      "object_type": "packages",
      "qualifier": "package",
      "qualifier_by_activity": {
        "Create_Package": "creates",
        "Send_Package": "shipped package"
      },
      "carry_links": true
    }
  ],
  "seed": 1,
  "start_time": "2023-04-03T06:00:00+00:00",
  "catalogue": {
    "customers": [
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
    {"process": "Sales", "case_attribute": "customer", "catalogue": "customers",
     "o2o": "places", "reverse": true, "link_events": ["place order", "confirm order"],
     "pick": "uniform"}
  ]
}
```

Run it with

```
poetry run prosimos start-orchestration --config testing_scripts/assets/order_management/simulation.json --ocel_out_path om.json
```

`sales.json` has no `customer` in its `case_attributes`: the reference picks it. Its `ItemOrdered` message
then carries it to the items, and `ItemPicked` to the packages.

**What appears in the OCEL file** (excerpts of that run, 2,000 orders):

- **The 15 customers**, type `customers`, with `segment` timed at the start of the run, and `objectTypes`
  lists `{"name": "customers", "attributes": [{"name": "segment", "type": "string"}]}`. Each is linked by
  `places` to its orders, 109 to 154 per customer, 2,000 in all:

  ```json
  {"id": "c3", "type": "customers",
   "attributes": [{"name": "segment", "value": "business", "time": "2023-04-03T06:00:00.000000Z"}],
   "relationships": [{"objectId": "Sales-25", "qualifier": "places"},
                     {"objectId": "Sales-29", "qualifier": "places"}, ...]}
  ```

- **The orders** keep `customer` as an attribute, as before:

  ```json
  {"id": "Sales-0", "type": "orders",
   "attributes": [{"name": "items", "value": 5, "time": "2023-04-03T06:00:00.000000Z"},
                  {"name": "customer", "value": "c9", "time": "2023-04-03T06:00:00.000000Z"}], ...}
  ```

- **place order and confirm order** are linked to their order's customer, with the qualifier `customer`
  (4,000 links: 2,000 of each); **pay order** isn't, since it isn't in `link_events`:

  ```json
  {"id": "e1", "type": "place order", "time": "2023-04-03T06:01:00.000000Z",
   "relationships": [{"objectId": "Sales-0", "qualifier": "order"},
                     {"objectId": "Warehouse-0", "qualifier": "item"}, ...,
                     {"objectId": "c9", "qualifier": "customer"}]}
  {"id": "e17", "type": "confirm order", "time": "2023-04-03T09:55:27.683160Z",
   "relationships": [{"objectId": "Sales-0", "qualifier": "order"}, ...,
                     {"objectId": "c9", "qualifier": "customer"}]}
  {"id": "e550", "type": "pay order", "time": "2023-04-11T20:30:28.616308Z",
   "relationships": [{"objectId": "Sales-0", "qualifier": "order"}, ...]}
  ```

Items and packages carry `customer` too, but only as an attribute: there is no reference on the Warehouse
or Packaging. Adding the Warehouse reference shown under "Two modes" would link each item to its customer
as well. To have every order come from a new customer instead, replace the list with
`"customers": {"attributes": [...]}` and the pick with `"new"`.

## Errors and warnings

**Load errors.** A configuration is checked when it is built (`SimulationConfig(...)` or
`SimulationConfig.from_json`). Two errors stop reading the file at once, prefixed by its path:

| Message                                                   | Cause                                                          |
|-----------------------------------------------------------|----------------------------------------------------------------|
| `catalogue.customers[0]: 'id' must be a non-empty string` | a list object without an `id`, or an empty one                 |
| `references[0] is missing ['case_attribute']`             | a reference without `process`, `case_attribute` or `catalogue` |

All the others are collected and raised together, as one `ValueError` starting `Invalid catalogue: `, the
problems separated by `; `. A growing catalogue with only a `"uniform"` pick, for example, breaks two rules:

```
Invalid catalogue: growing catalogue 'customers' needs exactly one reference that creates its objects ("new" or new_share), got 0; references[0] picks from growing catalogue 'customers', which only takes "new" or {"new_share": ...}
```

The messages, by the key they are about:

| Key                          | Message                                                                                                                                                         |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`                         | `catalogue ids are repeated: ['x']` (ids are unique across all catalogues)                                                                                      |
| `id`                         | `catalogue ids look like case ids (<process name>-<number>): ['Sales-1']`                                                                                       |
| catalogue name               | `catalogue 'orders' is the object type of process Sales`                                                                                                        |
| `count`                      | `catalogue 'customers': count must be a whole number of at least 1, got 0`                                                                                      |
| `attributes` (pool, growing) | `catalogue 'customers': attributes[0]: InvalidCaseAttributeException: Case attribute segment: probabilities' sum should be equal to 1` (Prosimos's own message) |
| growing catalogue            | `growing catalogue 'Sales' is named like a process; its ids would look like case ids`                                                                           |
| growing catalogue            | `catalogue ids look like the ids growing catalogue 'customers' creates (customers-<number>): ['customers-2']`                                                   |
| growing catalogue            | `growing catalogue 'customers' needs exactly one reference that creates its objects ("new" or new_share), got 0`                                                |
| `catalogue`                  | `references[0] names unknown catalogue 'clients'`                                                                                                               |
| `process`                    | `references[0] names unknown process 'Billing'`                                                                                                                 |
| `link_events`                | `references[0] link_events names ['confrim order'], which are no tasks of process Sales`                                                                        |
| `pick`                       | `references[0] pick must be "uniform", {"weights": {id: weight, ...}}, "new" or {"new_share": share}`                                                           |
| `pick`                       | `references[0] picks customer, which process Sales also draws in its case_attributes`                                                                           |
| `pick`                       | `references[0] picks customer, which process Warehouse also copies from a message (consume entries at ['Start_Item'])`                                          |
| `pick`                       | `references[0] picks item_ids, which process Packaging also appends to from a message (consume entries at ['Start_Package', 'Catch_More_Items'])`               |
| `pick`                       | `references[0] picks for Scripted, an engine not built from a ProcessSpec, which can't take a pick` (extra engines, see [orchestrator.md](orchestrator.md))     |
| `pick`                       | `more than one reference picks customer of process Sales`                                                                                                       |
| `weights`                    | `references[0] weights name ['c99'], which are no ids of catalogue 'customers'`                                                                                 |
| `weights`                    | `references[0] weights must be numbers of at least 0, not all 0, got {'c1': 0, 'c2': 0}`                                                                        |
| `weights`                    | `references[0] has weights on pool 'customers'; a pool can only be picked from uniformly`                                                                       |
| `"new"`, `new_share`         | `references[0] picks "new" or new_share from catalogue 'customers', which is no growing catalogue`                                                              |
| `"new"`, `new_share`         | `references[0] picks from growing catalogue 'customers', which only takes "new" or {"new_share": ...}`                                                          |
| `new_share`                  | `references[0] new_share must be a number above 0 and at most 1, got 1.5`                                                                                       |

`link_events` is not checked for extra engines, which have no BPMN model.

**Warning.** One kind, during the run, in the run report's `warnings`, once per attribute and value:

```
case attribute customer has the value 'c99', which is not in catalogue customers; no OCEL link is written for it
```

A value is compared with the ids as text, so the number `7` matches the id `"7"`. The value keeps no link,
but everything else about the case is written as usual.

**Outside the configuration.** A process JSON whose messages publish a picked attribute fails on its own,
when Prosimos reads its messages: `... 'customer' not declared as a case, global or event attribute`. Run
it through its configuration, where the pick declares the attribute.

## Limitations

- **One creator per growing catalogue.** Only one reference creates a growing catalogue's objects. Two
  processes creating, and reusing, objects of the same catalogue (e.g. customers who first place an order
  or first return one) is future work.
