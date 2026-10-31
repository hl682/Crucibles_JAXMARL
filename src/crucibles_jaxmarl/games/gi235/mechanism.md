# Pasture Weave Commons

Status: **AUTHORED_NOT_EXECUTED**. Candidate identity: game-v3-00235. Source split: train. No final test or previous candidate outcome was an input.

## Native contract and human controls

A=64, O=910, U=27, H=896. Every one of the 64 actor rows is independently controllable through one scalar int32 action. Ambient entities are not counted as agents. A human/AI/group assignment chooses the same native joint-action vector; no alternative UI dynamics are supplied. The source is the complete native kernel, and scene.json is complete declarative presentation data. JaxMARL adapter integration, actual keyboard/click controls, replay, animation, JIT/vmap, throughput, training and transfer remain unexecuted and unaccepted.

Movement is a cardinal one-cell proposal with boundary clipping. Multiple people may occupy a tile; this is an explicit broad-walkway abstraction. Visible objects and terrain govern interaction conditions. Masks describe capability/inventory permissions, not a promise that a distant or blocked operation succeeds. Out-of-range or mask-invalid codes become no-op deterministically. Spatially inapplicable legal attempts cost the stated work cost and change no targeted object. Joint actions are captured together; movement precedes ordered work, and work resolves actor IDs in ascending order. This tie rule is fixed before any execution and can affect scarce-resource competition.

Sixteen four-person households share a 64 by 64 landscape divided by paired fence lines. Twenty-four real gate crossings connect plots. Gate toggles affect both farmers and animals; leaving gates open wears integrity by one every 96 steps, and integrity zero allows uncontrolled passage until repaired. Wood comes from orchard branches, which regrow on a 32-step clock. A herder can push the nearest active animal one cell in the farmer's actual facing direction; fences and gates govern whether the push succeeds.

Thirty-two autonomous livestock are separate ecology entities. Every second step an unconfined animal proposes one cardinal step toward the pasture maximizing four times grass minus L1 distance, with fixed index tie breaking. A blocked proposal stays in place, so steering through gates and rescue labor matter. Every fourth step livestock consume one available grass at their current plot and modify that household's resource and payoff. Resting plots regrow two rather than one grass every eighth non-drought step. Hay plots spend one rather than two grass to create a carried fodder. Animals do not respect a household's intended rest designation, creating a gate/commons tradeoff. Every third 128-step weather phase is drought; grass/wells then stop replenishing. Trough water only benefits animals within distance 3; strong shelters within distance 4 protect from drought exposure. Hunger, illness and unprotected drought can remove health every 16 ticks. Zero-health animals become inactive.

Four shared facilities combine a water well, shearing rack and four-place quarantine pen. Actual nearby animals are moved into a pen and confined; an eight-count timer is decremented during the same step's ecology phase, so admission's first end-state timer is seven. When the countdown reaches zero, a clean facility clears infection; release still needs a farmer action. Dirty equipment can infect a subsequently shorn animal and prevents pen curing until a water steward cleans it. Shearing requires wool >=3, rack distance <=2, shearer distance <=1 and a distinct farmer choosing Hold within distance 1 of that animal. The same holding dependency supports a fodder-funded rescue of a hungry animal. Feeding, tool/fodder exchange, water transport, branch gathering, orchard tending and barn reserves are actual state transfers. Orchards grow every 16 ticks with moisture, and successful fruit harvest resets only that orchard's growth.

## Action legend

- 0: Wait.
- 1: Move north and face north.
- 2: Move east and face east.
- 3: Move south and face south.
- 4: Move west and face west.
- 5: Herd nearby livestock one cell forward with herding tool.
- 6: Toggle nearby livestock gate.
- 7: Repair nearby gate using hammer and wood.
- 8: Gather wood from nearby orchard.
- 9: Harvest ripe fruit from nearby orchard.
- 10: Cut nearby grass into fodder with herding tool.
- 11: Draw water at a shared well with bucket.
- 12: Pour carried water into nearby trough.
- 13: Admit nearby livestock to a shared quarantine pen.
- 14: Release nearby animal whose quarantine timer expired.
- 15: Shear nearby woolly livestock at a rack with a holding partner.
- 16: Hold nearby livestock for another farmer.
- 17: Feed nearby livestock from carried fodder.
- 18: Build nearby weather shelter using hammer and wood.
- 19: Rotate own nearby pasture between grazing, rest and hay.
- 20: Exchange tools with nearest adjacent farmer.
- 21: Give one fodder to nearest adjacent farmer.
- 22: Tend nearby orchard using carried water.
- 23: Store one carried fodder at nearby barn.
- 24: Withdraw one stored fodder from nearby barn.
- 25: Rescue hungry nearby livestock with fodder and a holding partner.
- 26: Clean nearby shared rack and quarantine facility with water.

## Observation and role-view contract

Farmers and all ecology/facility groups use L1 radius 6. Four nearest visible farmer rows plus self are exposed. Clock/weather and each household's own accumulated wool and fruit ledger are known to its members. Other households' changing grass, illness, reserves and gates remain local. Animals are separate ecology rows and cannot be counted as controllable actors.

The first scalar of the self block is actor ID/(A-1). Each nearest-actor slot begins with visibility then actor ID/(A-1); absent slots are all zero. Actor slots use the four smallest L1 distances with ID tie breaking; self is always visible through its self block. All other entity blocks use original entity order, visibility flag, normalized absolute x/y, then the listed attributes; hidden rows are zero. Terrain layout is fixed public knowledge, while changing object states use declared local visibility. The renderer must filter entity and event displays by role visibility; full-state diagnostics are a separate labeled view. Public scene metadata does not add policy input.

- [0, 13): Self: actor ID/63, x/63, y/63, household/15, tool/3, wood/4, fodder/4, water/3, facing/4, t/896, weather phase/2, own wool/40, own fruit/40 (self).
- [13, 53): Four nearest farmer slots, 10 each: visible, ID/63, x/63, y/63, household/15, tool/3, wood/4, fodder/4, water/3, facing/4 (local).
- [53, 149): 16 pasture rows, 6 each: visible, x/63, y/63, grass/12, rotation/2, current grazing/32 (local).
- [149, 245): 16 orchard rows, 6 each: visible, x/63, y/63, growth/8, branches/8, stored moisture/4 (local).
- [245, 357): 16 barn rows, 7 each: visible, x/63, y/63, shelter/3, fodder/24, wool products/40, fruit products/40 (local).
- [357, 421): 16 trough rows, 4 each: visible, x/63, y/63, water/8 (local).
- [421, 541): 24 gate rows, 5 each: visible, x/63, y/63, open, integrity/3 (local).
- [541, 565): 4 facility rows, 6 each: visible, x/63, y/63, well water/12, infected equipment, occupied pen places/4 (local).
- [565, 885): 32 livestock rows, 10 each: visible, x/63, y/63, owner/15, wool/6, hunger/8, health/3, sick, (pen+1)/4, timer/8 (local).
- [885, 910): Row-major 5 by 5 edge-clamped local fence patch, 0 floor or 1 fence; gate entities determine actual crossing (local).

## Rewards, bounds and persistent consequences

Every legal work attempt costs 0.01. Shearing pays its actor 0.75 and shares another 0.25 equally among the actor's four household members; ripe fruit pays its actor 0.4 and shares 0.2. Partner-assisted rescue pays its actor 0.15. Shearing at infected equipment costs that actor 0.25. Each actual consumed grass costs every member of the grazed plot's household 0.04, and each animal injury costs every member of its owning household 0.1. Thus opening a gate can export grazing losses while livestock injury remains attributable to ownership. Food storage, draws, rotation and repeated toggles have no positive shaping reward. At most 32 animals graze, each owner has two animals, and each farmer performs one action, so [-4,4] conservatively contains every per-actor step return.

Rewards are float32[A]. Observations are explicitly float32[A,O]; normalized bounds are [0.0,1]. Declared reward bounds are [-4, 4], analytical containment bounds rather than empirical results or performance gates. Actor/group identity, positions, tool changes and real object levels bind to native state in manifest.json and scene.json; no scripted display transition substitutes for a kernel transition.

## Termination and declared source invariants

Natural failure occurs when all 32 livestock health values are zero. Otherwise H=896 truncates the persistent village. Ordinary shearing, harvest and pasture rotation never reset the world. A live call advances t exactly once. Natural termination has priority at H. A terminal call is absorbing: all state and observation leaves, clock and terminal cause are preserved; reward is zero and every done row is true. There is no raw auto-reset.

All 64 farmer rows remain active until joint termination. Livestock active flags equal health>0. Grass stays 0..12, rotation 0..2, wood/fodder 0..4, carried water 0..3, orchard stage/branches 0..8, orchard water 0..4, shelters/gate integrity 0..3, trough water 0..8, well water 0..12, barn fodder 0..24 and product stocks 0..40. Sheep wool stays 0..6, hunger 0..8, health 0..3 and timer 0..8. Each sheep is in at most one pen and each pen admits at most four. Pen occupancy includes inactive animals; no undisclosed cleanup or resurrection occurs. Farmers and livestock may share cells. Grazing, illness and shared equipment persist across work and movement.

The manifest declares the complete fixed-key state tree. Numeric/bool state leaves use int32/float32/bool. PRNG keys remain external trajectory provenance and are not duplicated in state. The source's reset randomness is only sixteen reset grass levels, each an integer in 6..10 from the supplied key; initial livestock owner, infection, wool and positions are fixed; raw_step consumes no additional random draws. No signed-int or PRNG distribution substitution is made. Human-readable invariants above have not been numerically checked; the optional machine invariant list is empty to avoid inventing an unsupported invariant-check dialect.

## Prospective strategies and fixed diagnostics

A household can open gates to reach a shared rack or a new pasture, while accepting that its neighbors' livestock can consume recovering grass. It can reserve fodder in a barn and keep a plot resting, or rotate into hay to gather more efficiently at the expense of rest regrowth. Another choice is to supply water and hold an animal for a neighbor's shearer instead of claiming production personally. The first literal script combines an initial assisted shear and a water steward's fixed well-to-trough sequence; the second rotates, cuts and stores a fodder reserve. The pair removes only actor 3's initial Hold while actor 1 performs the same shearing action, targeting another actor's product and the animal's wool.

playthrough.json contains two complete literal joint-action arrays at seed 23517, plus one pair whose entire continuation differs in exactly one actor action at one step. All are UNEXECUTED_AUTHOR_HYPOTHESIS. Their finite lengths are diagnostic scopes, not claims of finishing the game. Conditions, delays and intended state changes are source reasoning; legality during rollout, effects, rewards, solvability and human usability remain unverified. There is no adaptive policy, seed search, measured effect size or expected-success threshold.

## Literal source evidence

- Gate state governs native passage: candidate.py:15 — `gate = jnp.any((self.distance(positions, s['gate_pos']) <= 1) & (s['gate_open'] | (s['gate_health'] == 0))[None, :], axis=1)`
- Complete farmer and ecology state: candidate.py:63 — `s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'tiles': tiles, 'pos': pos, 'active': jnp.ones((64,), jnp.bool_), 'tool': ids % 4, 'face': jnp.full((64,), 2, jnp.int32), 'wood': jnp.zeros((64,), jnp.int32), 'fodder': jnp.zeros((64,), jnp.int32), 'water': jnp.zeros((64,), jnp.int32), 'seen': jnp.zeros((64, 4), jnp.int32), 'agent_vis': jnp.zeros((64, 64), jnp.bool_), 'pasture_pos': center, 'grass': jax.random.randint(key, (16,), 6, 11).astype(jnp.int32), 'rotation': jnp.zeros((16,), jnp.int32), 'grazed': jnp.zeros((16,), jnp.int32), 'pasture_active': jnp.ones((16,), jnp.bool_), 'pasture_vis': jnp.zeros((64, 16), jnp.bool_), 'orchard_pos': center + jnp.array([-3, -3], jnp.int32), 'orchard_growth': jnp.full((16,), 4, jnp.int32), 'branches': jnp.full((16,), 8, jnp.int32), 'orchard_water': jnp.full((16,), 2, jnp.int32), 'orchard_active': jnp.ones((16,), jnp.bool_), 'orchard_vis': jnp.zeros((64, 16), jnp.bool_), 'barn_pos': center + jnp.array([0, -3], jnp.int32), 'shelter': jnp.zeros((16,), jnp.int32), 'fodder_stock': jnp.zeros((16,), jnp.int32), 'wool_stock': jnp.zeros((16,), jnp.int32), 'fruit_stock': jnp.zeros((16,), jnp.int32), 'barn_active': jnp.ones((16,), jnp.bool_), 'barn_vis': jnp.zeros((64, 16), jnp.bool_), 'trough_pos': center + jnp.array([3, 2], jnp.int32), 'trough_water': jnp.full((16,), 4, jnp.int32), 'trough_active': jnp.ones((16,), jnp.bool_), 'trough_vis': jnp.zeros((64, 16), jnp.bool_), 'gate_pos': gates, 'gate_open': jnp.zeros((24,), jnp.bool_), 'gate_health': jnp.full((24,), 3, jnp.int32), 'gate_active': jnp.ones((24,), jnp.bool_), 'gate_vis': jnp.zeros((64, 24), jnp.bool_), 'facility_pos': jnp.array([[10, 10], [42, 10], [10, 42], [42, 42]], jnp.int32), 'facility_water': jnp.full((4,), 12, jnp.int32), 'facility_dirty': jnp.zeros((4,), jnp.bool_), 'facility_load': jnp.zeros((4,), jnp.int32), 'facility_active': jnp.ones((4,), jnp.bool_), 'facility_vis': jnp.zeros((64, 4), jnp.bool_), 'sheep_pos': jnp.repeat(center, 2, axis=0) + jnp.tile(jnp.array([[1, 1], [2, 1]], jnp.int32), (16, 1)), 'sheep_owner': jnp.repeat(c, 2), 'sheep_wool': jnp.full((32,), 3, jnp.int32), 'sheep_hunger': jnp.zeros((32,), jnp.int32), 'sheep_health': jnp.full((32,), 3, jnp.int32), 'sheep_sick': jnp.arange(32, dtype=jnp.int32) == 31, 'sheep_pen': jnp.full((32,), -1, jnp.int32), 'sheep_timer': jnp.zeros((32,), jnp.int32), 'sheep_active': jnp.ones((32,), jnp.bool_), 'sheep_vis': jnp.zeros((64, 32), jnp.bool_), 'event_shear': jnp.zeros((64,), jnp.bool_), 'event_feed': jnp.zeros((64,), jnp.bool_), 'event_quarantine': jnp.zeros((64,), jnp.bool_), 'event_fruit': jnp.zeros((64,), jnp.bool_), 'event_rescue': jnp.zeros((64,), jnp.bool_)}`
- Real pasture rotation cost: candidate.py:144 — `cut_cost = jnp.where(z['rotation'][pasture] == 2, 1, 2)`
- Shared finite quarantine places: candidate.py:162 — `confine = (a[i] == 13) & near & (fd[facility] <= 3) & (z['sheep_pen'][sheep] < 0) & (pen_load < 4)`
- Distinct partner holding condition: candidate.py:121 — `helpers = jnp.any((jnp.sum(jnp.abs(z['pos'] - z['sheep_pos'][sheep]), axis=1) <= 1) & (a == 16) & (jnp.arange(64, dtype=jnp.int32) != i))`
- Partner-assisted shearing: candidate.py:171 — `shear = (a[i] == 15) & (sd[sheep] <= 1) & z['sheep_active'][sheep] & (rackdist[rack] <= 2) & (z['sheep_wool'][sheep] >= 3) & helpers & (z['wool_stock'][crew] < 40)`
- Shared equipment infection: candidate.py:172 — `contaminated = shear & z['facility_dirty'][rack]`
- Autonomous native movement: candidate.py:210 — `score = 4 * n['grass'][None, :] - self.distance(n['sheep_pos'], n['pasture_pos'])`
- Grazing changes another household: candidate.py:232 — `z['grazed'] = z['grazed'].at[region].add(eat.astype(jnp.int32))`
- Attributable grazing and injury costs: candidate.py:242 — `rew = rew - (jnp.arange(64, dtype=jnp.int32) // 4 == region).astype(jnp.float32) * 0.04 * eat`
- Preserved terminal state: candidate.py:250 — `n = jax.tree.map(lambda new, old: jnp.where(s['done'], old, new), n, s)`

## Provenance and limits

Only the 21 actual authorized input files in author_input_freeze.json were read. The controlling population/family237 input sets A and H. Older copied population notes were not read. No old source, result, IR, training/selection/heldout file, memory file, network, local model, remote model API or other author feedback was used. The inherited genre direction comes from this batch's independent_family_principles.json and design brief. No supplied upstream reference snapshot or upstream code was read or copied: Craftax mechanics and Pgx board APIs are not claimed as implementations, and PureJaxRL/Craftax PPO baselines are neither new game libraries nor replacements for the frozen learner.

Author-side checks are standard-library AST parsing, literal schema/shape declarations, JSON, byte lengths and SHA256 only. Supplied validator modules were read as input text and were not executed. No import of candidate.py, JAX, NumPy, reset, raw_step, observe or legal_actions occurred. This package establishes authored source/data identity only, not accepted R0/M0, numerical parity, learnability, strategic non-isomorphism, animation acceptance, scientific benefit or transfer.
