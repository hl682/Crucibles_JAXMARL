# Estuary Council Cities

Status: AUTHORED_NOT_EXECUTED. All gameplay, scripts and causal predictions below are source-derived author hypotheses, not measured results.

## Civic rules and real geography

Twenty sixteen-person towns occupy a 70 by 48 estuary grid. Five continuous vertical rivers divide the connected land, with timed sluice crossings at each town and one-way ferry landings. Eight visible professions include builders, farmers, stewards, ferrymen, couriers and traders; tools 1/2/3/4 enable building, irrigation, witnessing and ferry use and can be exchanged. Supply units are explicit combined construction/seed kits. Initial builder 0 and farmer 1 of each town carry two and one units.

At the assembly, action 6 contributes a unit. The first contributor opens a four-tick window. A second contributor must have a different actor identity; the ledger records the first actor. A nearby civic-tool holder must witness (7) while two credits and time remain. A grant creates one physical permit at the hall; action 8 picks it up into empty hands. The permit can be handed over, consumed at the gate (9) to open twelve ticks, or consumed at the reservoir (10) for four water units. A permit cannot satisfy both uses. Contributions expire without a valid witness; later independent assemblies are possible.

Action 11 invests supply into a levee at a flood gauge, up to strength two. Flood pressure moves from a town to the next real town east on a later step. The upstream levee increases the redirected component when pressure is present. Locally, a levee reduces exposure while an old open sluice adds two exposure units. This is a bounded discrete pressure model, not a mass-conserving fluid solver. Seeded tide offsets only affect the first upstream town in each row. Flood level at least three damages existing crops by three maturity units. A gate can improve irrigation and movement while worsening exposure during a pulse.

Action 12 moves a tool-4 actor from the harbor across the river to a free west landing. It is one-way; a return requires the town gate or another route. A newly occupied landing cancels the ferry. Roads (17) consume supply underfoot and remove walking stamina cost on road destination cells. A levee is a real blocked tile; construction is canceled if an actor moves onto its site in that step. Market investment (18) consumes supply and raises future crop prices up to two levels.

Fields start at maturity two and moisture two. Irrigation (13) requires a farming tool and consumes reservoir water. Existing wet crops gain maturity every fourth tick; planting (14) consumes supply on an empty field; harvest (15) loads one crop when maturity reaches eight. Sell (16) physically transfers the carried crop at a market. Tool exchange (19) and cargo/message handoff (20) are reciprocal adjacent actions.

## Incentives and intended alternatives

Residents can coordinate a gate permit for voyages, allocate it to irrigation, or withhold witness labor; supply invested in local levees can impose delayed exposure downstream. Roads and market upgrades compete with planting and assembly costs. Ferry labor provides an alternative to temporary gates. These are source-defined alternatives without measured equilibrium or learning claims.

Each nonrest action costs 0.01. Donations, levees, roads and market investments cost an additional 0.02. A sale pays its actor 1 plus 0.25 per old market level, and each resident of that market town 0.04. Each damaged town field costs all its residents 0.08. Natural completion is 40 cumulative crop cargos sold; otherwise H=1024 ends the episode. Public decisions are embodied in local action, supplied inventory and physical permits, with no implemented free-text negotiation or off-map voting.

## Literal diagnostics

Script one contributes from actor 0, then from actor 1 while actor 2 witnesses, then actor 1 picks up the expected permit. Script two moves ferryman 3 through the explicit cross-river action and then west. The paired hypothesis removes only actor 2's witness at the second contribution and uses identical later rest actions. Expected permit and expiry differences are unexecuted hypotheses.


## Native boundary and local information

Every listed actor is an independently selectable row of the same int32 action vector. Human or AI control uses that unchanged row; a frontend, keyboard binding, animation, camera, agent-keyed adapter and human usability have not been implemented or verified by this author. All actors remain active. Rest restores one stamina unit; exertion consumes one, with documented game-specific additions. Movement is cardinal; destinations occupied at the start of a step cannot be entered. Simultaneous claims for an empty destination favor the lowest actor index. Spatial interactions use start-of-step positions and Manhattan distance at most one to the nearest relevant object, unless a specific rule says otherwise. Tool exchange requires both adjacent actors to choose exchange and mutually select each other. Invalid or out-of-range actions become rest deterministically.

Each role receives the own-state prefix plus a 5x5 radius-two patch. The manifest lists every channel and normalization; the renderer's entity masks and map mask are derived from the same radius. Out-of-world patch cells are all -1. There is no omniscient policy feed. Agent IDs, fixed role and home crew are public metadata; current tools, cargo, stamina, messages and world attributes are visible only in the patch. Entity positions are used by movement, proximity, ecology or construction; none are layout-only proxies.

A nonterminal call advances the int32 clock once. Natural completion has priority over horizon truncation. A terminal raw call preserves every state leaf, observation and terminal cause, returns zero reward and all dones true, and does not reset. All boundary/state integers are int32; the supplied uint32 key is used only through JAX randomness and is not copied into state. State keys/shapes remain fixed. Rewards declare conservative [-8,8] bounds and observations [-1,1]; these are interface ranges, not scientific performance gates. Concurrency, bounds, terminal absorption, JIT/vmap, causal effects, throughput and numerical parity remain unverified.

## Source-line evidence

- candidate.py:122 'grant =': distinct contributions and witness release a permit
- candidate.py:139 'n["pending"]': levees redirect delayed pressure to a different town
- candidate.py:133 'flood =': opened sluice changes local flood exposure
- candidate.py:106 'ferry =': free landing and actual river transport
- candidate.py:129 'n["sluice"]': physical carried permit changes gate duration

## Provenance and limits

The source is original for this batch and train split. Only current batch authority, four assigned design/request pairs, population/family direction, human-playability addendum and copied interface contracts/templates were read. No prior candidate, IR, result, training, heldout or memory was read; no remote query was made. Official repository references were allowed but not read or copied. The family inspiration comes solely from the supplied design direction: no commercial assets or upstream game implementation is included. PureJaxRL/Craftax baseline code is not used as a game or learner replacement.

The two literal scripts and the paired single-action hypothesis in playthrough.json are unexecuted diagnostics, not a controller restriction or proof of task success. Source-authored invariants and bounds await the independent sealed validator. No candidate import, Python candidate execution, JAX/NumPy computation, validator, training, replay, browser session, human play, accepted R0/M0, solvability, learning, strategic richness, non-isomorphism or transfer result is claimed. Zero, negative, failed, missing and indeterminate outcomes remain admissible.
