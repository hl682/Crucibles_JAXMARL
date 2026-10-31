# Terraced Watershed Republics

Status: **AUTHORED_NOT_EXECUTED**. Candidate identity: game-v3-00234. Source split: train. No final test or previous candidate outcome was an input.

## Native contract and human controls

A=384, O=1084, U=23, H=1024. Every one of the 384 actor rows is independently controllable through one scalar int32 action. Ambient entities are not counted as agents. A human/AI/group assignment chooses the same native joint-action vector; no alternative UI dynamics are supplied. The source is the complete native kernel, and scene.json is complete declarative presentation data. JaxMARL adapter integration, actual keyboard/click controls, replay, animation, JIT/vmap, throughput, training and transfer remain unexecuted and unaccepted.

Movement is a cardinal one-cell proposal with boundary clipping. Multiple people may occupy a tile; this is an explicit broad-walkway abstraction. Visible objects and terrain govern interaction conditions. Masks describe capability/inventory permissions, not a promise that a distant or blocked operation succeeds. Out-of-range or mask-invalid codes become no-op deterministically. Spatially inapplicable legal attempts cost the stated work cost and change no targeted object. Joint actions are captured together; movement precedes ordered work, and work resolves actor IDs in ascending order. This tie rule is fixed before any execution and can affect scarce-resource competition.

Twenty-four sixteen-person settlements occupy a 6 by 4 array of 16-cell territories. Each actor's current tool selects worker, courier, engineer, grower or guard capabilities; tools can be exchanged. Residents walk real grid coordinates. Crossing into another region requires that region's grant to the resident's home crew; motion inside a currently occupied region remains possible after revocation. On flooded central road cells, walls below level 2 block movement. A guard on home territory grants the nearest foreign crew within distance 2 or revokes all foreign grants. Thus border escorts, a nearby guard and material/provision couriers can change another crew's future movement opportunities.

Every sluice starts closed with a capacity-1 canal. Top-row water starts at 6, other reservoirs at 3. Top territories receive 2 rain units every 8 ticks; all locations lose one to drought every 48 ticks. Storage is 6+2*reservoir level, bounded at 12. Incoming water beyond storage plus retaining-wall level produces flooding; storage then clips to its capacity. After resident work, an open gate removes min(water, canal capacity) into a three-slot pipeline. The parcel enters slot 0, shifts twice, then arrives at the following step's environmental update. Each parcel retains its own target when a later diversion changes. Real parcel positions interpolate source and destination at one-quarter, one-half and three-quarters of the physical channel; bottom outlets lead to the map edge. Downstream and east channels are different routes, not interchangeable visual skins.

Growers use one water for two moisture (capacity 6), one provision to seed an empty field, and harvest four provisions at growth 8. Growth advances every fourth tick while moist, spending one moisture; flood instead removes two growth levels. Households consume two provisions every 64 ticks. Engineers spend finite material on canal capacity, reservoir volume or retaining-wall level. Couriers carry up to three provisions and deliver only at foreign fields. An actor can read a nearby gauge into persistent private memory and broadcast that remembered value for four ticks; this never grants a live remote gauge feed.

## Action legend

- 0: Wait.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Quarry one construction unit with worker tool.
- 6: Upgrade nearest canal with engineer and material.
- 7: Expand nearby reservoir with engineer and material.
- 8: Build retaining wall near sluice with engineer and material.
- 9: Irrigate nearby field with grower tool.
- 10: Plant empty nearby field using its provisions.
- 11: Harvest ripe nearby field with grower tool.
- 12: Toggle nearby sluice with engineer or guard tool.
- 13: Grant nearby foreign crew entry while guarding own territory.
- 14: Revoke foreign entry while guarding own territory.
- 15: Read nearby water gauge into private memory.
- 16: Broadcast remembered gauge for four ticks.
- 17: Give one material to nearest adjacent resident.
- 18: Give one carried provision to nearest adjacent resident.
- 19: Load one provision at own field.
- 20: Deliver one provision at foreign field with courier tool.
- 21: Exchange tools with nearest adjacent resident.
- 22: Switch nearby canal between downstream and east diversion.

## Observation and role-view contract

Actors and non-agent entities use L1 radius 6; only four nearest actor slots plus self are exposed. A settlement's own provisions and all permissions granted to that settlement are shared knowledge within its crew. Remote water is absent unless a resident reads a local gauge and broadcasts its remembered value near another actor; messages may be stale. Static territory ownership is public.

The first scalar of the self block is actor ID/(A-1). Each nearest-actor slot begins with visibility then actor ID/(A-1); absent slots are all zero. Actor slots use the four smallest L1 distances with ID tie breaking; self is always visible through its self block. All other entity blocks use original entity order, visibility flag, normalized absolute x/y, then the listed attributes; hidden rows are zero. Terrain layout is fixed public knowledge, while changing object states use declared local visibility. The renderer must filter entity and event displays by role visibility; full-state diagnostics are a separate labeled view. Public scene metadata does not add policy input.

- [0, 11): Self: actor ID/383, x/95, y/63, settlement/23, tool/4, material/4, food/3, private gauge memory/12, signal/4, t/1024, own settlement grain/40 (self).
- [11, 51): Four nearest actor slots, 10 each: visible, ID/383, x/95, y/63, settlement/23, tool/4, material/4, food/3, signal/4, public remembered gauge/12 (local).
- [51, 339): 24 sluice rows, 12 each: visible, x/95, y/63, owner ID/23, water/12, open, canal/2, diversion, wall/3, permission count/24, outflow/2, flood (local).
- [339, 435): 24 reservoir rows, 4 each: visible, x/95, y/63, reservoir level/3 (local).
- [435, 579): 24 field rows, 6 each: visible, x/95, y/63, crop/8, moisture/6, grain/40 (local).
- [579, 675): 24 quarry rows, 4 each: visible, x/95, y/63, stock/48 (local).
- [675, 1035): 72 transit parcel rows, 5 each: visible, x/95, y/63, amount/2, destination/24; slot position derives from source, stored destination and travel phase (local).
- [1035, 1059): Entry permits granted to own settlement, destination order 0..23, bool as float32 (self).
- [1059, 1084): Row-major 5 by 5 edge-clamped public territory patch; settlement ID/23 (local).

## Rewards, bounds and persistent consequences

Work attempts cost 0.01 to the actor. Current flooding costs every member of the affected home settlement 0.2; a scheduled meal with fewer than two provisions costs each member 0.05. A harvest pays its actor 0.2 and splits another 1 equally among the field's 16 residents. A foreign provision delivery pays the courier 0.1 and splits 0.25 among recipient residents. Each actual construction increment splits 0.1 among the builder's home crew. Each arrived unit pays 0.02/16 to every resident of its source settlement, making water-access revenue attributable to a specific delayed source parcel. Stored outlet parcels pay no cross-settlement revenue. Inventory and grain caps, at most 16 local loaders per field and one action per actor contain these components inside [-2,4]; the bounds are analytical and unmeasured.

Rewards are float32[A]. Observations are explicitly float32[A,O]; normalized bounds are [0.0,1]. Declared reward bounds are [-2, 4], analytical containment bounds rather than empirical results or performance gates. Actor/group identity, positions, tool changes and real object levels bind to native state in manifest.json and scene.json; no scripted display transition substitutes for a kernel transition.

## Termination and declared source invariants

This is a persistent finite-window strategy game with no natural success termination: terminated remains false and H=1024 truncates. Fields and projects do not reset when harvested or completed. A live call advances t exactly once. Natural termination has priority at H. A terminal call is absorbing: all state and observation leaves, clock and terminal cause are preserved; reward is zero and every done row is true. There is no raw auto-reset.

All 384 people remain active. Material stays 0..4, carried food 0..3, quarry stock 0..48, water 0..12, canal capacity 1..2, reservoir/wall 0..3, crop 0..8, moisture 0..6 and provisions 0..40. Each settlement always permits its own residents. Parcel amounts are 0..2 and stored targets are 0..24. Transit, transit_target, packet_amount, packet_target and packet_pos describe the same native pipeline; no rendering-only motion is introduced. The map is bounded at x=95,y=63, with float32 parcel positions inside those bounds. Infrastructures, permissions and crops persist across all ordinary steps.

The manifest declares the complete fixed-key state tree. Numeric/bool state leaves use int32/float32/bool. PRNG keys remain external trajectory provenance and are not duplicated in state. The source's reset randomness is only the 24 reset soil-moisture values (integers 1 or 2) from the supplied key; raw_step consumes no additional random draws. No signed-int or PRNG distribution substitution is made. Human-readable invariants above have not been numerically checked; the optional machine invariant list is empty to avoid inventing an unsupported invariant-check dialect.

## Prospective strategies and fixed diagnostics

Upstream crews may retain water for their own growers or release it for attributable revenue; too much delivery can flood an unprepared downstream neighbor. Engineering choices can shift the tradeoff through storage, canal capacity or walls. A guard's access bargain can make courier provisions or repair labor available without granting everyone entry. The first fixed script opens a sluice briefly and broadcasts a gauge reading; the second attempts sowing and irrigation while an engineer uses a predetermined gate schedule. The pair changes only actor 2's initial sluice action. Its intended downstream difference is delayed transit/water/flood state, without a claimed magnitude or harvest outcome.

playthrough.json contains two complete literal joint-action arrays at seed 23413, plus one pair whose entire continuation differs in exactly one actor action at one step. All are UNEXECUTED_AUTHOR_HYPOTHESIS. Their finite lengths are diagnostic scopes, not claims of finishing the game. Conditions, delays and intended state changes are source reasoning; legality during rollout, effects, rewards, solvability and human usability remain unverified. There is no adaptive policy, seed search, measured effect size or expected-success threshold.

## Literal source evidence

- Complete native state and physical topology: candidate.py:56 — `s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'territory': territory, 'pos': pos, 'active': jnp.ones((384,), jnp.bool_), 'tool': ids % 16 % 5, 'material': jnp.zeros((384,), jnp.int32), 'food': jnp.zeros((384,), jnp.int32), 'memo_water': jnp.zeros((384,), jnp.int32), 'signal': jnp.zeros((384,), jnp.int32), 'message_water': jnp.zeros((384,), jnp.int32), 'seen': jnp.zeros((384, 4), jnp.int32), 'agent_vis': jnp.zeros((384, 384), jnp.bool_), 'gate_pos': gate, 'gate_open': jnp.zeros((24,), jnp.bool_), 'canal': jnp.ones((24,), jnp.int32), 'route': jnp.zeros((24,), jnp.int32), 'wall': jnp.zeros((24,), jnp.int32), 'water': jnp.where(c < 6, 6, 3).astype(jnp.int32), 'flow': jnp.zeros((24,), jnp.int32), 'flood': jnp.zeros((24,), jnp.bool_), 'permission': jnp.eye(24, dtype=jnp.bool_), 'permission_count': jnp.ones((24,), jnp.int32), 'gate_active': jnp.ones((24,), jnp.bool_), 'gate_vis': jnp.zeros((384, 24), jnp.bool_), 'reservoir_pos': centers + jnp.array([3, -3], jnp.int32), 'reservoir': jnp.zeros((24,), jnp.int32), 'reservoir_active': jnp.ones((24,), jnp.bool_), 'reservoir_vis': jnp.zeros((384, 24), jnp.bool_), 'field_pos': centers + jnp.array([3, 2], jnp.int32), 'crop': jnp.zeros((24,), jnp.int32), 'moisture': jax.random.randint(key, (24,), 1, 3).astype(jnp.int32), 'grain': jnp.full((24,), 6, jnp.int32), 'field_active': jnp.ones((24,), jnp.bool_), 'field_vis': jnp.zeros((384, 24), jnp.bool_), 'quarry_pos': centers + jnp.array([-3, 2], jnp.int32), 'quarry_stock': jnp.full((24,), 48, jnp.int32), 'quarry_active': jnp.ones((24,), jnp.bool_), 'quarry_vis': jnp.zeros((384, 24), jnp.bool_), 'transit': jnp.zeros((24, 3), jnp.int32), 'transit_target': jnp.full((24, 3), 24, jnp.int32), 'packet_pos': jnp.repeat(gate.astype(jnp.float32), 3, axis=0), 'packet_amount': jnp.zeros((72,), jnp.int32), 'packet_target': jnp.full((72,), 24, jnp.int32), 'packet_active': jnp.zeros((72,), jnp.bool_), 'packet_vis': jnp.zeros((384, 72), jnp.bool_), 'event_harvest': jnp.zeros((384,), jnp.bool_), 'event_build': jnp.zeros((384,), jnp.bool_), 'event_grant': jnp.zeros((384,), jnp.bool_), 'event_delivery': jnp.zeros((384,), jnp.bool_), 'event_water_revenue': jnp.zeros((24,), jnp.int32)}`
- Crossing permissions and flooded roads: candidate.py:108 — `authorized = s['permission'][region, ids // 16] | (region == origin)`
- Private gauge versus public message: candidate.py:24 — `s['message_water'] = jnp.where(s['signal'] > 0, s['memo_water'], 0)`
- Delayed incoming water: candidate.py:90 — `arrived = jnp.zeros((24,), jnp.int32).at[jnp.minimum(dest, 23)].add(s['transit'][:, 2] * (dest < 24))`
- Growth and flood tradeoff: candidate.py:98 — `growing = (s['crop'] > 0) & (s['crop'] < 8) & (s['moisture'] > 0) & (n['t'] % 4 == 0) & ~n['flood']`
- Local permission grant: candidate.py:152 — `grant = (a[i] == 13) & (region == crew) & (nd[foreign] <= 2)`
- Resource and infrastructure choice: candidate.py:128 — `build_c = (a[i] == 6) & (gd[g] <= 1) & (z['canal'][g] < 2) & (z['material'][i] > 0)`
- Persistent parcel destination: candidate.py:186 — `n['transit_target'] = jnp.concatenate((destination[:, None], s['transit_target'][:, :2]), axis=1).astype(jnp.int32)`
- Actual spatial water interpolation: candidate.py:193 — `n['packet_pos'] = (n['gate_pos'][:, None, :] + fraction * (endpoint - n['gate_pos'][:, None, :])).reshape((72, 2)).astype(jnp.float32)`
- Attributable delayed source revenue: candidate.py:88 — `n['event_water_revenue'] = jnp.zeros((24,), jnp.int32)`
- Absorbing finite-window semantics: candidate.py:200 — `n = jax.tree.map(lambda new, old: jnp.where(s['done'], old, new), n, s)`

## Provenance and limits

Only the 21 actual authorized input files in author_input_freeze.json were read. The controlling population/family237 input sets A and H. Older copied population notes were not read. No old source, result, IR, training/selection/heldout file, memory file, network, local model, remote model API or other author feedback was used. The inherited genre direction comes from this batch's independent_family_principles.json and design brief. No supplied upstream reference snapshot or upstream code was read or copied: Craftax mechanics and Pgx board APIs are not claimed as implementations, and PureJaxRL/Craftax PPO baselines are neither new game libraries nor replacements for the frozen learner.

Author-side checks are standard-library AST parsing, literal schema/shape declarations, JSON, byte lengths and SHA256 only. Supplied validator modules were read as input text and were not executed. No import of candidate.py, JAX, NumPy, reset, raw_step, observe or legal_actions occurred. This package establishes authored source/data identity only, not accepted R0/M0, numerical parity, learnability, strategic non-isomorphism, animation acceptance, scientific benefit or transfer.
