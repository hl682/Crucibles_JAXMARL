# Mountain Orchard Neighbours

Status: AUTHORED_NOT_EXECUTED. All gameplay, scripts and causal predictions below are source-derived author hypotheses, not measured results.

## Orchard ecology and embodied work

Thirty-two four-person households occupy connected terraces in a 64 by 40 map. Each has two compatible tree species, a movable hive, bridge, supply depot, water basin, store, pollinator swarm and roaming herd animal. Actors begin with gardener compost, steward water and mason materials; beekeepers begin empty. Tools 1/2/3/4 enable pruning/shading, hive work, herding/bracing and masonry. Bees and livestock have native positions and dynamics but no controllable agent rows.

Pruning (6) restores a mature fruitless tree to flowering stage four. Spray (7) removes three pest units at its target but deposits persistent residue on every tree within six cells; nearby hives lose health while residue remains within five cells. Carried water (8) and compost (9) change local moisture and growth. Supply pickup (5) gives one mason kit to a tool-4 actor or compost to others. Basin pickup (17) creates a carried water unit. Empty trees can be replanted with compost (22). Rain arrives every 24 ticks and bridge erosion/wind changes every 32; seasonal time is encoded by the native clock, without external weather data.

Wind transport requires a flowering unpoisoned source, a healthy hive within five cells and a live swarm within one cell. Sources send pollen to the opposite species downwind within eight horizontal cells and two vertical cells. The queue has two native delay slots: newly emitted pollen first enters pollen_next, then pollen_queue, then the receiving tree, and residue can suppress receipt. This links neighboring households instead of assigning independent crop games. Swarms physically move one cardinal cell toward a blossom class selected every eight ticks. They wait when another swarm occupies the destination, and lower swarm index resolves simultaneous empty-cell claims, keeping their observed entity attributes distinct. Their native target is chosen near their hive, so transporting that hive changes ecology.

Hives require a tool-2 caregiver (11), a different tool-1 shading actor (12), and nectar to gain two health. One actor cannot provide both simultaneous actions. A beekeeper can lift a hive (15) as a backpack, move it, and set it down (16); this backpack is separate from ordinary cargo. Hive cells block normal entry to avoid overlap. Nectar regenerates near mature trees every sixteen ticks. Health and care costs are tied to the real hive, not an independent reward token.

Bridges need a tool-4 mason spending material (13) and a different tool-3 bracing actor (14) to gain two integrity. Zero integrity closes the actual ravine crossing. Roaming herds damage seedlings only when their actual old position is within one cell; guiding (18) with a tool-3 actor prevents that herd's damage this tick and moves it to a refuge. Animals do not act as extra policies.

Wet trees gain maturity every fourth tick. Pests and trampling can reduce it. At maturity at least six with received pollen, a tree creates one fruit and returns to stage two, consuming its accumulated pollen. Harvest (10) loads a fruit; delivery (19) physically transfers it to a village store. Cargo handoff (21) preserves the carried quality code and tool exchange (20) is reciprocal. The current simplified quality code is the observed pollen at harvest, with a minimum of one; it is not a persistent biochemical property of each fruit.

## Incentives and alternatives

Households can keep shared pollination healthy, relocate a hive, or spray for immediate local pest protection despite harm to others. Water/compost labor competes with bridge maintenance, herd management and delivery. A bridge repair requires complementary staffing; hives can make distant households more or less productive after travel and pollen delays. These mechanisms have not been evaluated for optimal strategies or learning.

Each nonrest action costs 0.01. Spray costs another 0.03; compost, successful repair and planting another 0.02. Pruning and spraying each consume an extra stamina unit. A fruit sale pays its actor 1 plus 0.1 times its carried quality code, and each resident of the receiving store's household 0.04. Paired hive care pays that hive household 0.01 per resident; poisoned hives cost their household residents 0.02 each step. Sixty-four total delivered fruit cargos ends naturally; otherwise H=896 truncates.

## Literal diagnostics

Script one moves gardener 0 south three times to shade hive 0 while beekeeper 1 cares, then rests. Script two sprays the nearby tree and follows the residue with idle steps. The paired intervention repeats the approach, then changes only gardener 0's shade action versus rest while beekeeper 1 cares; identical suffixes test intended shared hive-health/nectar consequences. All are unexecuted source hypotheses.


## Native boundary and local information

Every listed actor is an independently selectable row of the same int32 action vector. Human or AI control uses that unchanged row; a frontend, keyboard binding, animation, camera, agent-keyed adapter and human usability have not been implemented or verified by this author. All actors remain active. Rest restores one stamina unit; exertion consumes one, with documented game-specific additions. Movement is cardinal; destinations occupied at the start of a step cannot be entered. Simultaneous claims for an empty destination favor the lowest actor index. Spatial interactions use start-of-step positions and Manhattan distance at most one to the nearest relevant object, unless a specific rule says otherwise. Tool exchange requires both adjacent actors to choose exchange and mutually select each other. Invalid or out-of-range actions become rest deterministically.

Each role receives the own-state prefix plus a 5x5 radius-two patch. The manifest lists every channel and normalization; the renderer's entity masks and map mask are derived from the same radius. Out-of-world patch cells are all -1. There is no omniscient policy feed. Agent IDs, fixed role and home crew are public metadata; current tools, cargo, stamina, messages and world attributes are visible only in the patch. Entity positions are used by movement, proximity, ecology or construction; none are layout-only proxies.

A nonterminal call advances the int32 clock once. Natural completion has priority over horizon truncation. A terminal raw call preserves every state leaf, observation and terminal cause, returns zero reward and all dones true, and does not reset. All boundary/state integers are int32; the supplied uint32 key is used only through JAX randomness and is not copied into state. State keys/shapes remain fixed. Rewards declare conservative [-8,8] bounds and observations [-1,1]; these are interface ranges, not scientific performance gates. Concurrency, bounds, terminal absorption, JIT/vmap, causal effects, throughput and numerical parity remain unverified.

## Source-line evidence

- candidate.py:116 'cared =': hive care requires complementary simultaneous roles
- candidate.py:168 'incoming =': spatial compatible wind pollen enters delayed queue
- candidate.py:131 'poison =': spray harms nearby trees beyond its source
- candidate.py:118 'repaired =': bridge repair requires a separate brace action
- candidate.py:127 'n["hive_pos"]': carried hive position follows a real actor
- candidate.py:187 'trampled =': roaming animals alter young tree growth

## Provenance and limits

The source is original for this batch and train split. Only current batch authority, four assigned design/request pairs, population/family direction, human-playability addendum and copied interface contracts/templates were read. No prior candidate, IR, result, training, heldout or memory was read; no remote query was made. Official repository references were allowed but not read or copied. The family inspiration comes solely from the supplied design direction: no commercial assets or upstream game implementation is included. PureJaxRL/Craftax baseline code is not used as a game or learner replacement.

The two literal scripts and the paired single-action hypothesis in playthrough.json are unexecuted diagnostics, not a controller restriction or proof of task success. Source-authored invariants and bounds await the independent sealed validator. No candidate import, Python candidate execution, JAX/NumPy computation, validator, training, replay, browser session, human play, accepted R0/M0, solvability, learning, strategic richness, non-isomorphism or transfer result is claimed. Zero, negative, failed, missing and indeterminate outcomes remain admissible.
