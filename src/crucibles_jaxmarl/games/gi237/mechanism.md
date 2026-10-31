# Canal Nightmarket Kitchens

Status: AUTHORED_NOT_EXECUTED. Source SHA256: 313eebca823ab1d005942f0a52a625798ed17296f0302fa0b0091198ecbc8f00. All scripts/interventions are UNEXECUTED_AUTHOR_HYPOTHESIS.

Native contract: A=48, O=212, U=16, H=768; scalar int32 actions, float32 observations, fixed int32/bool state. Geometry 48 by 12, x right/y down. Family cooperative_kitchen.

Twelve kitchens occupy one 48 by 12 quay. Water flows east. Barges and customer docks circulate horizontally; barges refill only at their west-boundary wrap. This is an explicit finite conveyor abstraction. Workers never wrap and cannot cross canal cells except eligible lock bridges. Boats and customers are world processes, not extra agent rows.

Two recipes consume different proportions of two ingredients. A batch consumes one clean use; washing replenishes cleaning but removes water needed by downstream navigation. Closing a lock conserves local water while blocking barges and potentially raising water above bridge access. Starting pot identity is a role, not an exclusive stove permission: any prepared-item carrier may load, while knife, wash and lock operations have tool requirements.

Six barges each hold eight roots and eight grain; people carry one ingredient or plate. Unloading requires simultaneous assistance at original positions; serial handoffs use updated positions. Stove timers, heat and inventory are state. Zero-heat carried food spoils after the action phase; the ordering permits a final cold matching service at lower quality. Sector telemetry exposes hydraulic timing without remote customer requests.

Work costs 0.002. Matching service pays each home-crew actor 0.04+0.08*heat/24 and the server a further 0.08+0.12*heat/24. Spoiled carried plates cost 0.04. No reward is paid merely for washing, relay or loading. Native rewards are clipped to [-4,4]. Central cooks/runners, mobile tool exchange and different water/traffic policies are author hypotheses; success, learnability and throughput are unmeasured.

## Interface and visibility

Public quay/canal geometry; selected nearest local entity rows reveal customer requests only nearby or through explicit timed relay. Hydraulic and plate-heat gauges are available only within the current four-cell sector. Unseen event channels are debug-only.

Source entry lines: reset 26, legal_actions 66, observe 87, raw_step 121. Exact columns, normalizers, shapes and scene bindings are in manifest.json. Every selected entity slot ends with presence; hidden slots are zero. Crew identities remain fixed, while tools may change. A legal intent can still be blocked by earlier actors, missing simultaneous help, material shortages or geometry. Masked or out-of-range actions become Wait. Actor priority rotates as (loop index + t) mod A.

No early natural terminal; the 768-tick market closing truncates the episode.

Already-done raw calls preserve every state leaf and observation, return zero rewards and true per-agent dones, and preserve terminal causes. Nonterminal calls advance t exactly once. No autoreset or external state exists. Signed identities and sentinels use int32; reset keys are not copied into state. Observation bounds [0,1] and explicit reward clipping [-4,4] are source declarations, not numerical verification.


## Literal source evidence

- candidate.py:158: assist = jnp.any((jnp.sum(jnp.abs(original_pos - p), axis=1) <= 1) & (actions == 5) & (jnp.arange(48, dtype=jnp.int32) != i))
- candidate.py:159: lift = (a == 6) & (bd[b] <= 2) & available[b] & assist & (w['item'][i] == 0)
- candidate.py:167: load = (a == 8) & jnp.any(stove) & room & ((w['item'][i] == 2) | (w['item'][i] == 3)) & (w['timer'][c] == 0) & (w['meal'][c] == 0)
- candidate.py:72: wash = jnp.any(near & (s['station_kind'][None, :] == 3), axis=1)
- candidate.py:183: serve = (a == 11) & jnp.any(match) & (w['item'][i] >= 4)
- candidate.py:204: listeners = (ad <= 3) & signal
- candidate.py:217: ready = (s['veg'] + s['grain'] == 3) & (s['veg'] >= 1) & (s['grain'] >= 1) & (s['clean'] > 0) & (s['meal'] == 0)
- candidate.py:231: flow = jnp.where(s['gate'] & (water >= 2), 1, 0).astype(jnp.int32)
- candidate.py:236: advance = (t % 3 == 0) & (s['water'][sector] >= 2) & s['gate'][sector]
- candidate.py:245: expired = s['deadline'] <= 0


## Human operation, provenance and limits
Scene data binds actual positions, stock, tools and events. Human single-role/multiple-role control must call the unchanged native joint action interface. No UI has been implemented, opened or played by this author. Role rendering must restrict event presentation to visible entities; global event inspection is debug-only.


Four independent world families are authored, without upstream code or assets. Supplied external reference files were permitted but not read. No prior game, results, IR, selection, training, heldout, memory, network, API, local model or numerical package was read/invoked. Current population/family240 and dispatch scope override inherited counts. Parentless initial window: 240 used/reserved of 512, feedback 0.


Only bytes/SHA256, JSON/AST and literal structure are checked here. Candidate import/reset/step/observe/legal_actions and provided validators were never executed. Runtime shapes/dtypes, JIT/vmap, terminal replay, mask reachability, script success, finite numerical bounds, resource cost, animation, human playability, non-isomorphism, learning and transfer remain unverified. Independent authorized frozen-source integration must preserve actual failures. No acceptance, deployment or experimental result is authored.
