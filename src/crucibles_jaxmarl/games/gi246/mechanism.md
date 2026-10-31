# Highland Treaty Roads

AUTHORED_NOT_EXECUTED. Candidate game-v3-00246, attempt00, train-only, feedback0, parentless blind authoring. Source SHA256 5a484b4b762a57e511b0360257fcd6de1fc7047eb2676f9511b587fd8ac1c1e5. Native dimensions A=192, O=3642, U=16, H=896. Every actor has an independently supplied scalar action, observation/reward/done row, actual position and persistent spatial consequences. There are 12 groups of 16; grouping identifies affiliation, not a single controller for that group.

## Task and interaction

Trade cargo between twelve cities while maintaining highland routes, negotiating physical seal treaties and delivering timber before mine-induced collapse. Each city seeks three foreign cargo receipts.

Extraction raises quarry stress and starts delayed rockfall. Timber physically delivered to a slope prevents collapse only while brace reserve remains. A courier carries issuer seal to locally present foreign representative and then a gate. Ratification grants reciprocal foreign bridge entry; each entry transfers one actual coin from traveler to owner treasury. Long public ford routes remain alternatives.

Negotiate selective bilateral routes or keep the longer public ford. Quarry early and rely on timely timber delivery, or stage braces before extraction. Couriers choose between goods, seals and time-sensitive surveys. Representatives can assist rivals for reciprocal access, accepting more traffic on shared approaches.

The 48x24 grid uses integer [x,y] cell centers, y down. Positions govern movement, object work and transfer. Scene colors/glyphs are original declarative cues for the independent civilization_strategy family. The shared platform can select arbitrary one/multiple actors and fill their unchanged native action rows. This package does not claim a working UI, animation or human acceptance.

## Action meanings

- 0: Wait / receive cargo or local report.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Collect city grain, ripe terrace grain, or builder quarry/timber resource.
- 6: Sell carried cargo to foreign city with two treasury coins.
- 7: Builder at slope: spend timber to brace or stone to clear.
- 8: Courier at own city: obtain unsigned seal.
- 9: Present unsigned carried seal at another city for local signature.
- 10: Representative at own city: countersign locally presented seals.
- 11: Ratify signed seal at issuer or partner gate for reciprocal access.
- 12: Grower: spend grain to seed empty terrace.
- 13: Record local quarry/slope stress and countdown on survey.
- 14: Show survey to adjacent waiting actor.
- 15: Hand cargo/seal and show survey to adjacent empty waiting actor.

Invalid or out-of-range action IDs deterministically become action0. Wait remains legal for every actor, including disabled actors and terminal input. Individual masks expose local physical/tool conditions; partner cooperation is resolved by the joint transition. Finite-resource contention uses clock-rotating actor priority. The same masks and native actions apply to human and AI ownership.

## Transition order, rewards and termination

Old-state masks; movement/toll transfers; rotating finite-resource work/trade; quarry stress, countdown, material repair and fall; terraces; local seal issuance/signature and gate ratification; surveys/handoffs; injury/reward/termination. Co-occupancy is allowed.

Clipped to [-2,2]. Own trade +0.6; own-city incoming cargo +0.1; repair +0.025; work -0.002; rockfall injury -0.1; paid toll -0.02; city-member toll income +0.02/16 per credited coin; own-city collapse -0.025. Trade additionally transfers two actual city-treasury coins to seller. Toll payer/payee and material costs are explicit state changes.

Natural termination: Every city has received at least three foreign cargos, or all actors have lost health; otherwise H=896. This is an in-game completion rule, not a scientific performance gate. Natural completion or failure takes priority if it coincides with H. truncated=(next t>=H) AND NOT terminated; done_all is their disjunction and equals episode_done. A nonterminal input advances t once. Already-terminal input is preserved by the final tree selection: clock, state, observation and terminal causes remain; reward is zero and all dones true. There is no internal reset.

The reset key is external trajectory provenance; only the supplied key drives reset stochasticity. Authored steps are deterministic and do not consume their key. Small signed values are int32, booleans bool, persistent reals float32. The uint32 key is not duplicated in world state. Exact eager/JIT/vmap behavior and numerical parity are untested.

## Exact observations and role rendering

Local entity blocks use fixed ID order and explicit presence. Hidden payloads are zero. Map patches enumerate dy=-3..3, then dx=-3..3. The local square is not occlusion-limited. Debug/full-state views must be labeled and cannot become policy observations. Actual event flags/counters must be filtered by the related visible entity; their declaration does not authorize off-screen event disclosure.

- Columns [0, 16): Time/896,ID/191,city/11,tool/3,x/47,y/23,cargo/3,coins/64,health/100,(seal origin+1)/12,(partner+1)/12,(survey city+1)/12,survey stress/8,survey countdown/4,survey timestamp/896,alive.
- Columns [16, 2512): 192 actor slots,13 fields: x/47,y/23,city/11,tool/3,cargo/3,coins/64,health/100,(seal origin+1)/12,(partner+1)/12,(survey site+1)/12,survey stress/8,survey countdown/4,alive/presence.
- Columns [2512, 3352): 84 site slots: x/47,y/23,type/6,owner/11,stock/64,brace/24,crop/12,stress/8,countdown/4,presence. Separate gate marker slots inactive and zero.
- Columns [3352, 3544): 12 gate slots,16 fields: x/47,y/23,owner/11,presence,then twelve visiting-city treaty TTLs /96.
- Columns [3544, 3642): 7x7 dy-major patch: tile/3 and in-bounds/alive. Other cells zero.

All fixed-shape state leaves are listed in manifest.json. Entity attributes are represented by the corresponding native local block. The explicitly described own team score summaries are self information. Declared invariants include finite observations/rewards, coordinate bounds, resource nonnegativity under legal transitions, fixed state-tree shapes and absorbing terminal semantics. They remain source declarations, not passed runtime receipts.

## Literal diagnostic proposals

playthrough.json supplies two full-width literal scripts and one paired hypothesis. Every frame contains all 192 actions, including zeros. There is no adaptive policy, search or substitute dynamics. The branches have the same prefix and subsequent actions, and differ at exactly one actor/action/step. Intended differences come from source reasoning only. No script was imported, run, simulated, scored or accepted. Initial random details, multi-step legality, exact effects and rewards remain unverified; no solvability or learning result is claimed.

## Literal source evidence

- candidate.py:71: `def legal_actions(self, s):`
- candidate.py:115: `paid = (s['tiles'][p[:, 1], p[:, 0]] == 2) & foreign & jnp.any(p != s['pos'], axis=1)`
- candidate.py:118: `treasury = s['treasury'] + tolls`
- candidate.py:133: `mined = jnp.zeros((12,), jnp.int32).at[city].add((gather & (typ == 1)).astype(jnp.int32))`
- candidate.py:138: `trigger = (stress >= 4) & (s['fall_timer'] == 0)`
- candidate.py:141: `collapse = resolved & (brace == 0)`
- candidate.py:154: `signed_here = jnp.zeros((12,), jnp.int32).at[city].add((act == 10).astype(jnp.int32)) > 0`
- candidate.py:160: `rat = jnp.zeros((12, 12), jnp.int32).at[orig, part].add(ratified.astype(jnp.int32))`
- candidate.py:162: `treaty = jnp.where(rat > 0, 96, jnp.maximum(s['treaty'] - 1, 0)).astype(jnp.int32)`
- candidate.py:180: `def moved(v, default):`
- candidate.py:197: `reward = 0.6 * trade + 0.1 * income[s['team']] + 0.025 * repair - 0.002 * (act >= 5) - 0.1 * injury`
- candidate.py:200: `terminated = ~jnp.any(alive) | jnp.all(received_goods >= 3)`
- candidate.py:210: `def raw_step(self, key, state, actions):`
- candidate.py:217: `def observe(self, s):`

## Provenance and remaining evidence

The supplied JaxMARL README informed parallel actor-interface expectations; Pgx informed legal masks, external keys and terminal states; Craftax README informed local materials/observation and no-autoreset boundaries. PureJaxRL README and Craftax_Baselines_ppo_rnn.py informed only learner/wrapper integration boundaries. No upstream code, assets, board layouts or game recipes were copied. No supplied external score was used as evidence for these authored games. Exact supplied reference bytes and uses are in author_input_freeze.json.

Authority and scope hashes were verified. ROOT_INPUT_REFERENCE_CLARIFICATION247.json explicitly supersedes the missing inherited194 reference; that absent file was not searched or read. Current247 population/family and current design/request bytes control. Actual allowed paths, byte counts, SHA256 and uses are frozen in author_input_freeze.json. No MEMORY, prior candidate/results, other coordination, training/IR, selection, heldout or final-test evidence entered authoring.

Author checks are AST parsing/whitelist-oriented text inspection, manifest/scene data inspection, literal script dimensions and single-action-difference counting, and byte/hash verification. Candidate code was never imported or executed. No NumPy/JAX/PRNG/runtime tests occurred. Runtime shape/parity/throughput, physical paired replay, native renderer/animation, arbitrary human play, learning and meta-learning transfer are UNKNOWN or UNEXECUTED. Independent immutable-source integration release is required.
