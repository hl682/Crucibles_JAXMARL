# Icebound Harbor Builders

Status: AUTHORED_NOT_EXECUTED. Source SHA256: 21d55ee55532f079c6fcd041163b2e64bdd044f859fb8cf0263ff04e1fcfba67. All scripts/interventions are UNEXECUTED_AUTHOR_HYPOTHESIS.

Native contract: A=512, O=237, U=15, H=1024; scalar int32 actions, float32 observations, fixed int32/bool state. Geometry 64 by 32, x right/y down. Family block_survival.

Thirty-two coves are connected by tundra strips and fragile ice channels. All 512 people have native action, observation, reward and done rows. This source contains people and structures, with thaw/freezing ecology; no decorative crowd is counted. Movement uses rotating serial collision resolution, while anchoring reads sanitized simultaneous intents and original positions. A cove can work with a few partners instead of a global command.

Stone, wood and freight are single carried units; freight is a heavy sled. Tools gate gathering, ferry piloting and rescue; construction and feeding require materials. Quarry/timber stock replenishes only on sixteen-tick resource ticks. Existing depots hold outward freight; new depot-site construction is not implemented. Shelters and landings start unbuilt, and both are built with stone. Each ferry is a local three-cell crossing, not a free navigation vehicle. Transfers select the first adjacent waiting eligible actor.

Expected alternatives include anchored ice hauling versus stone/fuel ferry investment, warm shelters versus immediate shipment labor, and repairing a crossing versus rescue after a break. No advantage was measured. Work costs 0.002 per non-wait intent, ice-break injury costs 0.25, foreign delivery pays carrier 0.3 and each recipient crew member 0.04, rescue pays 0.1, and a cold health loss costs 0.03. Final rewards are clipped to [-4,4]. Ice, roads, health, stock and fuel are physical state, not decorative score substitutes.

## Interface and visibility

Cove geometry is known. Dynamic map view is a radius-two square; the policy sees local integrity/road patches and selected nearest-eight entity rows. Scene visibility uses exactly those entity selections. Off-view events are debug-only.

Source entry lines: reset 29, legal_actions 56, observe 74, raw_step 101. Exact columns, normalizers, shapes and scene bindings are in manifest.json. Every selected entity slot ends with presence; hidden slots are zero. Crew identities remain fixed, while tools may change. A legal intent can still be blocked by earlier actors, missing simultaneous help, material shortages or geometry. Masked or out-of-range actions become Wait. Actor priority rotates as (loop index + t) mod A.

All 512 health values reach zero; natural completion takes precedence over the 1024-tick horizon.

Already-done raw calls preserve every state leaf and observation, return zero rewards and true per-agent dones, and preserve terminal causes. Nonterminal calls advance t exactly once. No autoreset or external state exists. Signed identities and sentinels use int32; reset keys are not copied into state. Observation bounds [0,1] and explicit reward clipping [-4,4] are source declarations, not numerical verification.


## Literal source evidence

- candidate.py:126: anchored = jnp.any((d <= 1) & anchors & (jnp.arange(512, dtype=jnp.int32) != i))
- candidate.py:131: move = (action >= 1) & (action <= 4) & ~occupied & passable & ((w['cargo'][i] != 3) | anchored) & jnp.any(target != p)
- candidate.py:133: stress = move & ~w['land'][ty, tx]
- candidate.py:148: build_sites = (sd <= 1) & ((w['site_kind'] == 2) | (w['site_kind'] == 3))
- candidate.py:158: burn = (action == 8) & jnp.any(shelters) & (w['cargo'][i] == 2) & (w['fuel'][j] < 24)
- candidate.py:171: deliver = (action == 10) & jnp.any(depots) & (w['cargo'][i] == 3)
- candidate.py:177: rescue = (action == 11) & jnp.any(fallen) & (w['cargo'][i] == 2) & (w['tool'][i] == 4)
- candidate.py:184: ferry = (action == 12) & jnp.any(docks & w['built']) & (w['tool'][i] == 3) & (w['fuel'][fuel_site] > 0) & free & w['land'][destination[1], destination[0]]
- candidate.py:204: warm = jnp.any((distance <= 2) & (s['site_kind'][None, :] == 2) & s['built'][None, :] & (s['fuel'][None, :] > 0), axis=1)
- candidate.py:212: s['terminated'] = jnp.all(s['health'] == 0)


## Human operation, provenance and limits
Scene data binds actual positions, stock, tools and events. Human single-role/multiple-role control must call the unchanged native joint action interface. No UI has been implemented, opened or played by this author. Role rendering must restrict event presentation to visible entities; global event inspection is debug-only.


Four independent world families are authored, without upstream code or assets. Supplied external reference files were permitted but not read. No prior game, results, IR, selection, training, heldout, memory, network, API, local model or numerical package was read/invoked. Current population/family240 and dispatch scope override inherited counts. Parentless initial window: 240 used/reserved of 512, feedback 0.


Only bytes/SHA256, JSON/AST and literal structure are checked here. Candidate import/reset/step/observe/legal_actions and provided validators were never executed. Runtime shapes/dtypes, JIT/vmap, terminal replay, mask reachability, script success, finite numerical bounds, resource cost, animation, human playability, non-isomorphism, learning and transfer remain unverified. Independent authorized frozen-source integration must preserve actual failures. No acceptance, deployment or experimental result is authored.
