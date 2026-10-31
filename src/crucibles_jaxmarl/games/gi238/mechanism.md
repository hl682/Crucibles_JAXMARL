# Railway Commons Republics

Status: AUTHORED_NOT_EXECUTED. Source SHA256: c491d2ff054fbe8ac13430c940bb2c1e84719e3d6fe2d8d7cbb2fe5d69cbb491. All scripts/interventions are UNEXECUTED_AUTHOR_HYPOTHESIS.

Native contract: A=256, O=231, U=19, H=1024; scalar int32 actions, float32 observations, fixed int32/bool state. Geometry 64 by 32, x right/y down. Family civilization_strategy.

The four-by-four settlement graph has horizontal length-sixteen links and vertical length-eight links. Corridor cells are public geography; actual construction and damage live at physical link-center work sites. Foot agents travel through the map, and trains advance in integer world coordinates along their selected links. No browser geometry replaces these transitions.

Each settlement owns a mine, farm and station/power depot. Horizontal routes need one extra ore unit, while vertical routes are open alternatives after a switch change. Eight trains begin with one grain unit from their origins. The switch commits for six ticks. Trains reserve edges, preventing head-on entry; multiple carriers may park at a station. Speed rises to two cells/tick and red signals brake within three remaining cells. Late red changes can cause overrun. Failed bridges damage health and cargo. Broken carriers stay visible at their actual location for repairs.

Passage is a simple contestable public/home-origin declaration, not a full diplomacy interface. Foreign unloading adds two producer-stock units per cargo unit, an explicit trade/processing yield. Actors resolve in rotating order before train updates; a courier must staff arrival before a later departure or arrange a red stop.

Work costs 0.001. Unloading pays courier 0.15, each recipient-town actor 0.04 and each origin-town actor 0.02. Bridge failure or red overrun costs origin actors 0.03; a failed bridge additionally charges its real last contributor 0.02 only when such an actor exists. Empty pantries at consumption cost local actors 0.02. Rewards are clipped to [-4,4]. Alternate investment, route, access, power, pantry and repair strategies are source hypotheses; no equilibrium, prosperity, performance or transfer is proved.

## Interface and visibility

Surveyed corridor geometry is public. Only selected nearest site/link/train/agent rows reveal operational state. Home identity grants no remote treasury or signal view. Explicit decaying local bulletins convey route information. Off-view events are debug-only.

Source entry lines: reset 33, legal_actions 72, observe 91, raw_step 123. Exact columns, normalizers, shapes and scene bindings are in manifest.json. Every selected entity slot ends with presence; hidden slots are zero. Crew identities remain fixed, while tools may change. A legal intent can still be blocked by earlier actors, missing simultaneous help, material shortages or geometry. Masked or out-of-range actions become Wait. Actor priority rotates as (loop index + t) mod A.

No early natural terminal; a fixed 1024-tick settlement season truncates.

Already-done raw calls preserve every state leaf and observation, return zero rewards and true per-agent dones, and preserve terminal causes. Nonterminal calls advance t exactly once. No autoreset or external state exists. Signed identities and sentinels use int32; reset keys are not copied into state. Observation bounds [0,1] and explicit reward clipping [-4,4] are source declarations, not numerical verification.


## Literal source evidence

- candidate.py:160: build = (a == 6) & (ed[e] <= 1) & (w['cargo'][i] == 1)
- candidate.py:169: switch = (a == 7) & station & (w['commit'][c] == 0) & (w['tool'][i] == 2)
- candidate.py:182: unload = (a == 10) & station & jnp.any(parked) & (w['train_node'][b] == c) & (w['train_owner'][b] != c) & (w['train_load'][b] > 0)
- candidate.py:231: permission = (w['permit'][dest] < 0) | (w['permit'][dest] == w['train_owner'][b])
- candidate.py:233: depart = (w['train_edge'][b] < 0) & (w['train_health'][b] > 0) & jnp.any(choices) & (w['track'][e] >= 2) & (w['bridge'][e] > 0) & (w['power'][node] > 0) & ~w['red'][node] & permission & ~occupied
- candidate.py:244: brake = w['red'][dest] & (remaining <= 3)
- candidate.py:247: damaged = moving & (w['bridge'][e] == 0) & (speed > 0)
- candidate.py:251: overrun = arrive & w['red'][dest]
- candidate.py:266: storm = (t % 32 == 0) & jax.random.bernoulli(key, 0.125, (24,))
- candidate.py:270: hungry = consume & (s['grain'] == 0)


## Human operation, provenance and limits
Scene data binds actual positions, stock, tools and events. Human single-role/multiple-role control must call the unchanged native joint action interface. No UI has been implemented, opened or played by this author. Role rendering must restrict event presentation to visible entities; global event inspection is debug-only.


Four independent world families are authored, without upstream code or assets. Supplied external reference files were permitted but not read. No prior game, results, IR, selection, training, heldout, memory, network, API, local model or numerical package was read/invoked. Current population/family240 and dispatch scope override inherited counts. Parentless initial window: 240 used/reserved of 512, feedback 0.


Only bytes/SHA256, JSON/AST and literal structure are checked here. Candidate import/reset/step/observe/legal_actions and provided validators were never executed. Runtime shapes/dtypes, JIT/vmap, terminal replay, mask reachability, script success, finite numerical bounds, resource cost, animation, human playability, non-isomorphism, learning and transfer remain unverified. Independent authorized frozen-source integration must preserve actual failures. No acceptance, deployment or experimental result is authored.
