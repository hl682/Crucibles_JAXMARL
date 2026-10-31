# astra-d2-00018: cohort commons with ecological hysteresis

This parentless source is inspired by generic renewable commons, competing
harvesters and delayed ecological conversion. It does not use a supplied
benchmark or parent source. The combination is a causal proposal, not a claim
of nonisomorphism, mechanism qualification or ecological realism.

Two agents occupy one of two patches. Each patch has a mature stock, water,
toxin, two age cohorts of planted material and a persistent damage regime.
Each agent has a private reserve. Harvesting at a patch shares its available
stock equally among simultaneous harvesters, each capped at 0.22. This immediate
competition also increases local toxin, which influences the future regime.

Actions are 0 rest, 1 harvest the current patch, 2 spend up to 0.14 reserve to
plant twice that amount, 3 spend up to 0.10 reserve to remove toxin, and 4 spend
0.03 reserve to move to the other patch. Movement is ineffective if that cost
cannot be paid. Rest restores 0.025 reserve; harvesting restores 0.35 times the
yield. All action effects except movement use the previous location, and every
action remains legal even when its special effect is resource limited.

New seeds enter the young cohort after the planting transition, become old
after another transition, and mature from the old cohort on a later transition.
The amount converted is old_cohort * previous_water times 0.25 in a damaged
patch or 1.0 otherwise. The old cohort is consumed even when conditions yield
little usable stock. Mature stock is capped at one, so excess is discarded.
This is a delayed conversion process rather than an immediate planting reward.

Toxin retains 0.97 of its previous level, gains 0.60 times total harvest and
loses 0.90 times cleanup. A patch becomes damaged above 0.65 toxin but only
recovers below 0.25; between these levels it retains its existing regime.
The new regime affects subsequent maturation. Water replenishes by 0.06, loses
0.50 times maturation and 0.08 times harvest, and exchanges 0.05 times the
water difference with the other patch. Stocks, water, toxin and reserves are
bounded to [0, 1].

Reward is 0.70 * harvest/0.22, minus 0.15 times the sum of normalized planting
and cleaning expenditure, minus 0.03 for a successful move, minus 0.08 for
attempting harvest in a previously damaged patch. It is explicitly clipped to
[-1, 1]. Reserve flows and utility costs are distinct. Agents can benefit from
each other's planting or cleaning, impose shared damage, exhaust a competing
harvester's stock, or relocate before delayed benefits appear. Zero, negative
and uncoordinated outcomes are possible.

Observation columns 0 through 15 are: time/32; own patch; own reserve; local
mature stock; other patch mature stock; local water; other water; local toxin;
other toxin; local young cohort; local old cohort; local damaged flag; other
damaged flag; partner position; own previous yield/0.22; and partner previous
yield/0.22. Partner reserve and cohorts at the unoccupied other patch are not
directly observed. Initial mature stock, water and toxin use only the reset
key; later transitions are deterministic given actions.

The intended interface is A2/O16/U5 and 32 transitions with terminal-state
preservation. The author runs syntax parsing only. No compiler, runtime,
duplicate, eligibility or learning assessment is claimed.
