# Basalt Archipelago Rescue

Status: **AUTHORED_NOT_EXECUTED**. Candidate identity: game-v3-00232. Source split: train. No final test or previous candidate outcome was an input.

## Native contract and human controls

A=256, O=686, U=20, H=896. Every one of the 256 actor rows is independently controllable through one scalar int32 action. Ambient entities are not counted as agents. A human/AI/group assignment chooses the same native joint-action vector; no alternative UI dynamics are supplied. The source is the complete native kernel, and scene.json is complete declarative presentation data. JaxMARL adapter integration, actual keyboard/click controls, replay, animation, JIT/vmap, throughput, training and transfer remain unexecuted and unaccepted.

Movement is a cardinal one-cell proposal with boundary clipping. Multiple people may occupy a tile; this is an explicit broad-walkway abstraction. Visible objects and terrain govern interaction conditions. Masks describe capability/inventory permissions, not a promise that a distant or blocked operation succeeds. Out-of-range or mask-invalid codes become no-op deterministically. Spatially inapplicable legal attempts cost the stated work cost and change no targeted object. Joint actions are captured together; movement precedes ordered work, and work resolves actor IDs in ascending order. This tie rule is fixed before any execution and can affect scarce-resource competition.

Sixteen 16-person crews occupy a bounded 4 by 4 archipelago. Quarry, hammer, sail and rescue tools are transferable, so current equipment—not decorative character type—controls specialized actions. The fixed basalt map contains two-cell channels and shoal corridors; 24 raised causeway/lift gates connect neighboring islands. Causeway levels 2–3 plus an open lift gate allow channel crossing at either tide. Low tide alternates every 16 steps. Route damage removes one level every 64 steps unless a level-2 shelter lies within L1 distance 10. Shelters also protect workers within radius 4 from intense ash. Ash increases every 32 steps, blocks movement within radius 2 at intensity 3, and a hammer action within radius 4 starts a timer of 3; expiry removes two intensity units. These are persistent world changes, not visual substitutes.

A boat begins alongside each island. A single lowest-ID equipped pilot proposing movement controls each boat that step; other passengers follow. Boats move only over non-island tiles. Boarding, one-step adjacent-land disembarking and stone loading are physical operations. Passenger count never exceeds 8 minus cargo. Nearby support action 6 is necessary both to take a resident and to move while carrying one; the helper may belong to any crew. A stretcher follows its carrier's position. Delivery can use any shelter, so resources and assistance can cross crew boundaries. Signals are short-lived visible intention markers; they do not reveal remote world state.

## Action legend

- 0: Wait.
- 1: Walk or pilot north.
- 2: Walk or pilot east.
- 3: Walk or pilot south.
- 4: Walk or pilot west.
- 5: Quarry nearby stone with pick.
- 6: Brace a nearby stretcher carrier.
- 7: Build nearest causeway with hammer and stone.
- 8: Improve nearest shelter with hammer and stone.
- 9: Toggle nearby causeway lift gate.
- 10: Take nearby stretcher with rescue kit and a bracing partner.
- 11: Deliver carried resident at shelter.
- 12: Board nearby boat if capacity remains.
- 13: Leave boat onto adjacent island floor.
- 14: Give one stone to nearest adjacent actor.
- 15: Exchange tools with nearest adjacent actor.
- 16: Start delayed ash clearing with hammer.
- 17: Raise a visible three-step signal.
- 18: Load one stone into nearby boat.
- 19: Unload one boat stone into own inventory.

## Observation and role-view contract

Actor entities use four nearest rows within L1 radius 6 plus self. Every non-agent group is visible within L1 radius 6. Home shelter/rescue ledger and deterministic clock/tide are known to that crew. There is no remote ash/boat/route state in a role's flat observation.

The first scalar of the self block is actor ID/(A-1). Each nearest-actor slot begins with visibility then actor ID/(A-1); absent slots are all zero. Actor slots use the four smallest L1 distances with ID tie breaking; self is always visible through its self block. All other entity blocks use original entity order, visibility flag, normalized absolute x/y, then the listed attributes; hidden rows are zero. Terrain layout is fixed public knowledge, while changing object states use declared local visibility. The renderer must filter entity and event displays by role visibility; full-state diagnostics are a separate labeled view. Public scene metadata does not add policy input.

- [0, 13): Self: actor ID/255, x/63, y/63, crew/15, tool/3, stone/4, (boat+1)/16, carrying flag, signal/3, t/896, tidal phase 0/1, own-home rescued/32, own shelter level/3 (self).
- [13, 53): Four nearest visible actor slots, 10 each: visible, ID/255, x/63, y/63, crew/15, tool/3, stone/4, (boat+1)/16, carrying, signal/3 (local).
- [53, 117): 16 quarry rows, 4 each: visible, x/63, y/63, stock/32 (local).
- [117, 261): 24 route rows, 6 each: visible, x/63, y/63, level/3, gate open, current damage (local).
- [261, 341): 16 shelter rows, 5 each: visible, x/63, y/63, level/3, rescued/32 (local).
- [341, 421): 16 boat rows, 5 each: visible, x/63, y/63, cargo/4, passengers/8 (local).
- [421, 581): 32 resident rows, 5 each: visible, x/63, y/63, (carrier+1)/256, saved (local).
- [581, 661): 16 ash rows, 5 each: visible, x/63, y/63, intensity/3, clearing timer/3 (local).
- [661, 686): Row-major 5 by 5 terrain patch centered on actor, edge-clamped; tile code/2 (local).

## Rewards, bounds and persistent consequences

Every legal attempted work action (code >=5) costs its actor 0.01. A carrying movement attempt costs 0.005. A proposed position in unprotected intense ash costs 0.03. Each route damage event contributes a common loss of 0.02/24 to every actor that step. Delivering a resident pays the carrier 1 and distributes another 0.5 equally to the carrier's 16-person crew. Each actual stone-funded route/shelter increment distributes 0.05 equally to the builder's crew. Transfers, boarding, unloading and repeated toggles have no positive shaping reward. At most one resident can be carried/delivered by an actor and at most one funded construction increment can be performed by that actor per step; these facts contain the declared range. A shelter's rescued counter belongs to the delivery location; reward crew belongs to the delivering actor.

Rewards are float32[A]. Observations are explicitly float32[A,O]; normalized bounds are [0.0,1]. Declared reward bounds are [-2, 3], analytical containment bounds rather than empirical results or performance gates. Actor/group identity, positions, tool changes and real object levels bind to native state in manifest.json and scene.json; no scripted display transition substitutes for a kernel transition.

## Termination and declared source invariants

Natural completion occurs exactly when all 32 resident saved flags are true. Otherwise H=896 truncates the ongoing world; no reward target is a release condition. A live call advances t exactly once. Natural termination has priority at H. A terminal call is absorbing: all state and observation leaves, clock and terminal cause are preserved; reward is zero and every done row is true. There is no raw auto-reset.

All 256 actors remain independently active until joint termination. Positions stay in [0,63]^2. Quarry stock is 0..32, held stone 0..4, route/shelter/ash levels 0..3, clear/signal timers 0..3, boat cargo 0..4 and boat occupancy <=8-cargo. Each resident has at most one carrier and each actor at most one stretcher. Saved residents are inactive scene objects. Repeated stone transfers conserve stone outside quarry extraction and funded construction; no automatic resource reset occurs. People may share tiles; boats and stretchers use actual transition positions.

The manifest declares the complete fixed-key state tree. Numeric/bool state leaves use int32/float32/bool. PRNG keys remain external trajectory provenance and are not duplicated in state. The source's reset randomness is only the 16 reset ash intensities, each drawn as integer 0 or 1 from the supplied key; raw_step consumes no additional random draws. No signed-int or PRNG distribution substitution is made. Human-readable invariants above have not been numerically checked; the optional machine invariant list is empty to avoid inventing an unsupported invariant-check dialect.

## Prospective strategies and fixed diagnostics

Crews may build a durable raised route and protect it with shelter investment, or reserve stone for local ash protection and use tide windows/boats. Loading stone onto a boat trades transport inventory against room for other crews. Rescue teams can alternate leader movement and helper repositioning, while other workers maintain routes. The first script is a short source-reasoned rescue relay for actors 14 and 15; the second moves quarry stone from actor 0 to builder 1 and constructs a crossing. The pair removes only actor 14's first brace while preserving actor 15's pickup and the remaining literal movement prefix. The intended difference is resident ownership followed by coupled resident movement, not just the helper's action cost.

playthrough.json contains two complete literal joint-action arrays at seed 23207, plus one pair whose entire continuation differs in exactly one actor action at one step. All are UNEXECUTED_AUTHOR_HYPOTHESIS. Their finite lengths are diagnostic scopes, not claims of finishing the game. Conditions, delays and intended state changes are source reasoning; legality during rollout, effects, rewards, solvability and human usability remain unverified. There is no adaptive policy, seed search, measured effect size or expected-success threshold.

## Literal source evidence

- Crew topology and exact state: candidate.py:61 — `s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'terrain': terrain, 'pos': pos, 'active': jnp.ones((256,), jnp.bool_), 'tool': ids % 4, 'stone': jnp.zeros((256,), jnp.int32), 'aboard': jnp.full((256,), -1, jnp.int32), 'signal': jnp.zeros((256,), jnp.int32), 'carrying': jnp.zeros((256,), jnp.bool_), 'seen': jnp.zeros((256, 4), jnp.int32), 'agent_vis': jnp.zeros((256, 256), jnp.bool_), 'quarry_pos': center + jnp.array([-3, 0], dtype=jnp.int32), 'quarry_stock': jnp.full((16,), 32, jnp.int32), 'quarry_active': jnp.ones((16,), jnp.bool_), 'quarry_vis': jnp.zeros((256, 16), jnp.bool_), 'route_pos': routes, 'route_level': jnp.zeros((24,), jnp.int32), 'route_open': jnp.zeros((24,), jnp.bool_), 'route_damage': jnp.zeros((24,), jnp.bool_), 'route_active': jnp.ones((24,), jnp.bool_), 'route_vis': jnp.zeros((256, 24), jnp.bool_), 'shelter_pos': center + jnp.array([0, -3], dtype=jnp.int32), 'shelter_level': jnp.zeros((16,), jnp.int32), 'shelter_active': jnp.ones((16,), jnp.bool_), 'shelter_vis': jnp.zeros((256, 16), jnp.bool_), 'rescued': jnp.zeros((16,), jnp.int32), 'boat_pos': center + jnp.array([7, 0], dtype=jnp.int32), 'boat_cargo': jnp.zeros((16,), jnp.int32), 'boat_load': jnp.zeros((16,), jnp.int32), 'boat_active': jnp.ones((16,), jnp.bool_), 'boat_vis': jnp.zeros((256, 16), jnp.bool_), 'victim_pos': jnp.repeat(center, 2, axis=0) + jnp.tile(jnp.array([[2, 1], [3, 2]], jnp.int32), (16, 1)), 'victim_carrier': jnp.full((32,), -1, jnp.int32), 'victim_saved': jnp.zeros((32,), jnp.bool_), 'victim_active': jnp.ones((32,), jnp.bool_), 'victim_vis': jnp.zeros((256, 32), jnp.bool_), 'ash_pos': center + jnp.array([4, -4], dtype=jnp.int32), 'ash_level': jax.random.randint(key, (16,), 0, 2).astype(jnp.int32), 'ash_clear': jnp.zeros((16,), jnp.int32), 'ash_active': jnp.ones((16,), jnp.bool_), 'ash_vis': jnp.zeros((256, 16), jnp.bool_), 'event_rescue': jnp.zeros((256,), jnp.bool_), 'event_build': jnp.zeros((256,), jnp.bool_), 'event_transfer': jnp.zeros((256,), jnp.bool_)}`
- Local visibility and actual carrying flag: candidate.py:23 — `s['carrying'] = jnp.any(s['victim_carrier'][None, :] == jnp.arange(256, dtype=jnp.int32)[:, None], axis=1)`
- Tide, construction and lift reachability: candidate.py:104 — `passable = (tile == 1) | (tile == 2) & (s['t'] // 16 % 2 == 0) | bridge`
- Partner-dependent carrying: candidate.py:108 — `helpers = jnp.any((self.distance(s['pos'], s['pos']) <= 1) & (a[None, :] == 6) & (jnp.arange(256, dtype=jnp.int32)[:, None] != jnp.arange(256, dtype=jnp.int32)[None, :]), axis=1)`
- Finite boat capacity: candidate.py:152 — `board = (a[i] == 12) & (bd[b] <= 1) & (load < 8 - z['boat_cargo'][b])`
- Physical rescue pickup: candidate.py:142 — `pickup = (a[i] == 10) & (vd[v] <= 1) & eligible[v] & helpers[i]`
- Delayed ash clearing: candidate.py:169 — `clear = (a[i] == 16) & (ad[island] <= 4) & (z['ash_clear'][island] == 0)`
- Recipient-specific rewards: candidate.py:180 — `rew = rew.at[i].add(deliver.astype(jnp.float32))`
- Absorbing terminal state: candidate.py:191 — `n = jax.tree.map(lambda new, old: jnp.where(s['done'], old, new), n, s)`

## Provenance and limits

Only the 21 actual authorized input files in author_input_freeze.json were read. The controlling population/family237 input sets A and H. Older copied population notes were not read. No old source, result, IR, training/selection/heldout file, memory file, network, local model, remote model API or other author feedback was used. The inherited genre direction comes from this batch's independent_family_principles.json and design brief. No supplied upstream reference snapshot or upstream code was read or copied: Craftax mechanics and Pgx board APIs are not claimed as implementations, and PureJaxRL/Craftax PPO baselines are neither new game libraries nor replacements for the frozen learner.

Author-side checks are standard-library AST parsing, literal schema/shape declarations, JSON, byte lengths and SHA256 only. Supplied validator modules were read as input text and were not executed. No import of candidate.py, JAX, NumPy, reset, raw_step, observe or legal_actions occurred. This package establishes authored source/data identity only, not accepted R0/M0, numerical parity, learnability, strategic non-isomorphism, animation acceptance, scientific benefit or transfer.
