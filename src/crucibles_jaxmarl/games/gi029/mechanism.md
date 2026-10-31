# astra-d2-00029: Consumable filtration with transported regeneration waste

This initial parentless proposal uses generic flow processing, delayed externalities and filter-regeneration reasoning. It is not a claim of a new game class or nonisomorphism. No benchmark, other proposal, training result, development result or final-test data was supplied to this author.

## Transition mechanism and actions

Each agent controls a four-unit vessel containing volume and contaminant mass, and a filter with one unit of initially unused capacity. Contaminant removed by filtration is stored implicitly as one minus remaining filter capacity. Each role also has one outgoing, one-transition transport compartment.

Actions are 0 admit one unit from one's private inlet, 1 process up to one unit, 2 forward up to one unit to the partner, 3 remove as much vessel contaminant as the remaining filter can hold, and 4 regenerate the filter. All actions are legal even when ineffective. Incoming material from the partner's previously created compartment arrives before the current action is evaluated. All vessel material is mixed; overflow beyond four units discards volume and contaminant in the same proportion.

Forwarding transports the current mixture. Regeneration restores filter capacity to one and, when the filter contains any stored contaminant, sends one unit of externally supplied carrier water containing that stored contaminant to the partner. This carrier water is an explicit exogenous resource rather than a conserved part of the regenerating vessel. The outgoing compartment arrives on the next raw step. The partner can consequently be harmed by a regeneration action that locally restores useful filtering capacity. The partner can also benefit from clean forwarded material.

Processing and forwarding remove contaminant in the mixture proportion. Filtration removes contaminant without removing volume. Each role selects one action, so processing, forwarding, filtration and regeneration do not occur simultaneously within a role.

## Information and delayed interaction

A role sees its own contamination, filter capacity, inlet quality and outbound mixture. Vessel volumes, throughputs, overflows and the preceding actions of both roles are public. The next inbound volume is visible, but the partner's contaminant load and filter capacity are not directly observed. Incoming quality becomes part of the local mixture after arrival.

The sixteen columns are: time/32; own and partner vessel volume/4; own vessel contaminant fraction; own remaining filter capacity; own outbound volume; own outbound contaminant fraction; next inbound volume; own inlet contaminant fraction; own last reward; own and partner previous overflow/2; own and partner cumulative processed volume/32; own and partner previous action/4. Zero-volume fractions use a positive denominator floor and are zero.

Both agents alter their own future processing opportunities and can alter the other's future mixture through forwarding or filter waste. Incoming material can also induce overflow, so the externality depends on the receiver's state rather than solely on a sender action label.

## Incentives and reward bounds

Processing yields 0.8 times processed volume times one minus twice the contaminant fraction: clean processing is valuable and sufficiently dirty processing is costly. Forwarding earns a small 0.04 reward per unit, admitting costs 0.04, filtering costs 0.03 even if ineffective, and regeneration costs 0.05. Overflow costs 0.10 times the smaller of overflow and one. This permits conflict between locally beneficial disposal and the partner's throughput.

A processed quantity is at most one and contaminant fraction lies in [0, 1]. Even summing mutually exclusive action costs, total negative magnitude is at most 1.02, so the explicit [-1, 1] clip is the authoritative reward bound. Under the intended mutually exclusive action semantics, a processing transition ranges from -0.9 to 0.8; the other actions are more tightly bounded. Clipping is a declared saturation rule, not an unreported normalization. No terminal inventory bonus is added.

## Interface and evidence status

The source intends the required A2/O16/U5 interface, 32 transitions, all-legal masks and preserved terminal state. Reset samples initial dirt and private inlet quality through the supplied random key. Later transitions are deterministic and do not use the step key.

No candidate execution, JAX initialization, compiler invocation, isolated validation, duplicate review, mechanism qualification or learning measurement has occurred in the author's process. Syntax parsing alone is not validation. Model backend revision and token telemetry remain unknown.
