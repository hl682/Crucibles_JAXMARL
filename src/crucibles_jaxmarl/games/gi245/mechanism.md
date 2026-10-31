# Lantern Ferry Supper

AUTHORED_NOT_EXECUTED. Candidate game-v3-00245, attempt00, train-only, feedback0, parentless blind authoring. Source SHA256 174bcf03003597af838bc1e3deda78823f72ba639ef3eddbc95133e55f1f29d6. Native dimensions A=12, O=466, U=16, H=512. Every actor has an independently supplied scalar action, observation/reward/done row, actual position and persistent spatial consequences. There are 3 groups of 4; grouping identifies affiliation, not a single controller for that group.

## Task and interaction

Serve twelve staggered passenger orders before expiration, reusing plates and separating fish allergens while coordinating shared passages and heat.

Two opposite-bank cooks move the same actual counter, removing the old crossing and opening the new one. Shared furnace heat drives all pots and the dryer but risks burns. Carried plates, ingredients and order tickets and persistent board contamination connect three kitchens.

Place crossings near pantry paths or near service routes. Low-heat cooking leaves timing slack; hot batches require prompt plate delivery and venting. Use distinct fish/vegetable boards or spend cleaning effort for flexibility. Ticket runners can relay window information but tickets become stale when orders change.

The 23x11 grid uses integer [x,y] cell centers, y down. Positions govern movement, object work and transfer. Scene colors/glyphs are original declarative cues for the independent cooperative_kitchen family. The shared platform can select arbitrary one/multiple actors and fill their unchanged native action rows. This package does not claim a working UI, animation or human acceptance.

## Action meanings

- 0: Wait / receive adjacent handoff.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Pick pantry food, plate, plated ready pot meal or dry cabinet plate.
- 6: Chop carried raw ingredient at board.
- 7: Wash dirty plate at sink or empty-handed clean board.
- 8: Load chopped food into pot or wet plate into dryer.
- 9: Serve plated meal at dining window.
- 10: Stoke nearby shared furnace with fuel.
- 11: Vent nearby shared furnace.
- 12: Empty-handed carry gangway downward with opposite-bank partner.
- 13: Empty-handed carry gangway upward with opposite-bank partner.
- 14: Take local current order ticket.
- 15: Handoff item and ticket to adjacent empty waiting cook.

Invalid or out-of-range action IDs deterministically become action0. Wait remains legal for every actor, including disabled actors and terminal input. Individual masks expose local physical/tool conditions; partner cooperation is resolved by the joint transition. Finite-resource contention uses clock-rotating actor priority. The same masks and native actions apply to human and AI ownership.

## Transition order, rewards and termination

Old-state masks; conservative collision movement with rotating equal-target priority; two counter carries in gap order; rotating nearest-station arbitration; furnace/cabinet; orders/expiration; adjacent handoffs; meal age/burns; closure.

Clipped to [-1,2]. Correct server +0.8; each home-window kitchen member +0.2; own spill/rejection -0.15; burn -0.04; nonwait effort -0.002; home expired order -0.05; everyone -0.002 times fraction of windows holding more than three dirty plates. Plates, food and fuel are also persistent resources.

Natural termination: All twelve orders have arrived and all active orders have been served or expired. Service closure includes missed orders, so natural termination does not assert success. Otherwise H=512. Natural completion or failure takes priority if it coincides with H. truncated=(next t>=H) AND NOT terminated; done_all is their disjunction and equals episode_done. A nonterminal input advances t once. Already-terminal input is preserved by the final tree selection: clock, state, observation and terminal causes remain; reward is zero and all dones true. There is no internal reset.

The reset key is external trajectory provenance; only the supplied key drives reset stochasticity. Authored steps are deterministic and do not consume their key. Small signed values are int32, booleans bool, persistent reals float32. The uint32 key is not duplicated in world state. Exact eager/JIT/vmap behavior and numerical parity are untested.

## Exact observations and role rendering

Local entity blocks use fixed ID order and explicit presence. Hidden payloads are zero. Map patches enumerate dy=-3..3, then dx=-3..3. The local square is not occlusion-limited. Debug/full-state views must be labeled and cannot become policy observations. Actual event flags/counters must be filtered by the related visible entity; their declaration does not authorize off-screen event disclosure.

- Columns [0, 12): Time/512,ID/11,team/2,x/22,y/10,item/9,allergen,age/32,ticket/2,spill,burn,home completed orders/4.
- Columns [12, 132): 12 actor slots: x/22,y/10,team/2,item/9,allergen,age/32,ticket/2,spill,burn,presence.
- Columns [132, 348): 18 station slots: x/22,y/10,type/5,stock/16,recipe/4,progress/8,contamination,order/2,deadline/96,served,missed,presence.
- Columns [348, 356): Two gangway slots: x/22,y/10,signed displacement {-1,0,1},presence.
- Columns [356, 368): Two thermal slots: x/22,y/10,kind,level/10,stock/36,presence.
- Columns [368, 466): 7x7 dy-major patch: tile/2,in-bounds; unavailable cells zero.

All fixed-shape state leaves are listed in manifest.json. Entity attributes are represented by the corresponding native local block. The explicitly described own team score summaries are self information. Declared invariants include finite observations/rewards, coordinate bounds, resource nonnegativity under legal transitions, fixed state-tree shapes and absorbing terminal semantics. They remain source declarations, not passed runtime receipts.

## Literal diagnostic proposals

playthrough.json supplies two full-width literal scripts and one paired hypothesis. Every frame contains all 12 actions, including zeros. There is no adaptive policy, search or substitute dynamics. The branches have the same prefix and subsequent actions, and differ at exactly one actor/action/step. Intended differences come from source reasoning only. No script was imported, run, simulated, scored or accepted. Initial random details, multi-step legality, exact effects and rewards remain unverified; no solvability or learning result is claimed.

## Literal source evidence

- candidate.py:59: `def legal_actions(self, s):`
- candidate.py:92: `occupied = jnp.any(jnp.all(proposal[:, None] == s['pos'][None], axis=2) & (ids[:, None] != ids[None]), axis=1)`
- candidate.py:96: `def shift_one(k, carry):`
- candidate.py:114: `station_pick = (((typ < 2) | (typ == 4) | (typ == 5)) & (it == 0) & (s['stock'][k] > 0)) | ((typ == 3) & (it == 9) & (s['recipe'][k] > 0) & (s['progress'][k] >= 4))`
- candidate.py:129: `correct = serve & (it - 4 == s['order'][k]) & ((s['order'][k] != 1) | ~s['allergen'])`
- candidate.py:149: `heat = jnp.clip(s['thermal_level'][0] - 0.05 + 2 * stoke - 2 * vent, 0, 10)`
- candidate.py:150: `progress = jnp.where(change, 0.0, s['progress'])`
- candidate.py:169: `arrival = ((s['t'] + 1) % 48 == jnp.array([0, 16, 32], jnp.int32)) & (order[windows] == 0) & (s['remaining'] > 0)`
- candidate.py:174: `ticket = jnp.where((act == 14) & near, s['order'][k], s['ticket'])`
- candidate.py:198: `reward = 0.8 * correct + 0.2 * served[windows][s['team']] - 0.15 * spill - 0.04 * burn - 0.002 * (act > 0) - 0.05 * missed[windows][s['team']] - 0.002 * jnp.mean((stock[windows] > 3).astype(jnp.float32))`
- candidate.py:196: `terminated = jnp.all(remaining == 0) & ~jnp.any(order > 0)`
- candidate.py:207: `def raw_step(self, key, state, actions):`
- candidate.py:214: `def observe(self, s):`

## Provenance and remaining evidence

The supplied JaxMARL README informed parallel actor-interface expectations; Pgx informed legal masks, external keys and terminal states; Craftax README informed local materials/observation and no-autoreset boundaries. PureJaxRL README and Craftax_Baselines_ppo_rnn.py informed only learner/wrapper integration boundaries. No upstream code, assets, board layouts or game recipes were copied. No supplied external score was used as evidence for these authored games. Exact supplied reference bytes and uses are in author_input_freeze.json.

Authority and scope hashes were verified. ROOT_INPUT_REFERENCE_CLARIFICATION247.json explicitly supersedes the missing inherited194 reference; that absent file was not searched or read. Current247 population/family and current design/request bytes control. Actual allowed paths, byte counts, SHA256 and uses are frozen in author_input_freeze.json. No MEMORY, prior candidate/results, other coordination, training/IR, selection, heldout or final-test evidence entered authoring.

Author checks are AST parsing/whitelist-oriented text inspection, manifest/scene data inspection, literal script dimensions and single-action-difference counting, and byte/hash verification. Candidate code was never imported or executed. No NumPy/JAX/PRNG/runtime tests occurred. Runtime shape/parity/throughput, physical paired replay, native renderer/animation, arbitrary human play, learning and meta-learning transfer are UNKNOWN or UNEXECUTED. Independent immutable-source integration release is required.
