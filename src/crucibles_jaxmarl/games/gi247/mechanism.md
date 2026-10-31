# Reedbank Market Gardens

AUTHORED_NOT_EXECUTED. Candidate game-v3-00247, attempt00, train-only, feedback0, parentless blind authoring. Source SHA256 23350458ec5a024149ae28cbdc3a06c4157f92bd0cbed2d47ce2bfa8a391ece7. Native dimensions A=80, O=2173, U=18, H=768. Every actor has an independently supplied scalar action, observation/reward/done row, actual position and persistent spatial consequences. There are 20 groups of 4; grouping identifies affiliation, not a single controller for that group.

## Task and interaction

Bring shared wetland crop/fish goods to markets while coordinating flood pulses, filtering reeds, crop diversity, shelters and tools. Every household seeks at least one sale after the first seasonal interval.

Opening a gate drains its own pool and sends water downstream. Flood pulses bring fish/nutrients but destroy ripe crops; reed cutting opens boat routes while weakening filtering. Opposite-bank rack holders create a real crossing and dry cargo; complementary net holders catch fish. Neighbor seed diversity and bird shelters change future crop growth. Local exchange moves tools, seeds and perishables.

Time gates with downstream harvest or retain local water at the cost of downstream fish. Cut reeds for boat access or keep filtration. Pair holders can maintain a shortcut and dry goods; boat routes free their labor. Exchange compatible seeds or change next planting diversity. Carry fresh products quickly or invest two workers in drying for longer routes.

The 40x32 grid uses integer [x,y] cell centers, y down. Positions govern movement, object work and transfer. Scene colors/glyphs are original declarative cues for the independent rural_community family. The shared platform can select arbitrary one/multiple actors and fill their unchanged native action rows. This package does not claim a working UI, animation or human acceptance.

## Action meanings

- 0: Wait / receive goods, seed or tool exchange.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Pick finished dried rack goods or compost portion.
- 6: Hoe holder: plant carried seed in empty plot.
- 7: Sickle holder: harvest ripe plot with empty hands.
- 8: Hold rack from bank with empty hands.
- 9: Place fresh food on empty rack.
- 10: Opposite-bank netting with water/boat tools and empty hands.
- 11: Water-tool holder: toggle local gate.
- 12: Sickle holder: cut reeds with empty hands.
- 13: Compost reed/fresh/spoiled goods or apply compost to plot.
- 14: Offer empty-handed tool exchange to adjacent waiting actor.
- 15: Offer goods, else one compatible seed, to adjacent waiting actor.
- 16: Sell fresh/dried food at local market.
- 17: Spend carried reed to repair bird shelter.

Invalid or out-of-range action IDs deterministically become action0. Wait remains legal for every actor, including disabled actors and terminal input. Individual masks expose local physical/tool conditions; partner cooperation is resolved by the joint transition. Finite-resource contention uses clock-rotating actor priority. The same masks and native actions apply to human and AI ownership.

## Transition order, rewards and termination

Old-state masks; co-occupancy-permitted movement; rotating facility work; gate/reed changes and hydrology; pollution/nutrients/crop ecology; paired rack/net processes; inventories/seeds; tool/goods/seed exchange; spoilage/market revenue; termination and visibility.

Clipped to [-2,2]. Own sale value times0.35; each household member gets household sale value times0.1; harvest+0.02; work-0.003; spoilage-0.06; own-household ripe flood loss-0.04; cut-0.002. Opening incurs another-0.04 only if newly sent runoff is sufficient to cross the downstream ripe-loss threshold on this step. Sale value=quality times(1+dried+0.25 matching variety). Ecology and labor consequences persist beyond reward.

Natural termination: At t>=96 all twenty households have at least one sale; otherwise H=768. Failed, spoiled and flood-lost harvests remain actual outcomes. Natural completion or failure takes priority if it coincides with H. truncated=(next t>=H) AND NOT terminated; done_all is their disjunction and equals episode_done. A nonterminal input advances t once. Already-terminal input is preserved by the final tree selection: clock, state, observation and terminal causes remain; reward is zero and all dones true. There is no internal reset.

The reset key is external trajectory provenance; only the supplied key drives reset stochasticity. Authored steps are deterministic and do not consume their key. Small signed values are int32, booleans bool, persistent reals float32. The uint32 key is not duplicated in world state. Exact eager/JIT/vmap behavior and numerical parity are untested.

## Exact observations and role rendering

Local entity blocks use fixed ID order and explicit presence. Hidden payloads are zero. Map patches enumerate dy=-3..3, then dx=-3..3. The local square is not occlusion-limited. Debug/full-state views must be labeled and cannot become policy observations. Actual event flags/counters must be filtered by the related visible entity; their declaration does not authorize off-screen event disclosure.

- Columns [0, 15): Time/768,ID/79,household/19,tool/3,x/39,y/31,item/8,age/64,quality,produce variety/2,seed variety/2,seeds/3,capped household sales/768,capped household revenue/1728,spoil.
- Columns [15, 1135): 80 actor slots,14 fields: x/39,y/31,household/19,tool/3,item/8,age/64,quality,produce variety/2,seed variety/2,seeds/3,harvest,sale,spoil,presence.
- Columns [1135, 1235): 20 markets: x/39,y/31,preference/2,household/19,presence.
- Columns [1235, 1375): 20 plots: x/39,y/31,crop/12,variety/2,nutrient/6,flood-loss,presence.
- Columns [1375, 1495): 20 gates: x/39,y/31,open,water/6,changed,presence.
- Columns [1495, 1615): 20 reed beds: x/39,y/31,density/6,pollution/6,cut,presence.
- Columns [1615, 1775): 20 racks: x/39,y/31,raw item/4,work/4,quality,variety/2,held,presence.
- Columns [1775, 1855): 20 compost heaps: x/39,y/31,portions/16,presence.
- Columns [1855, 1935): 20 bird shelters: x/39,y/31,health/16,presence.
- Columns [1935, 2075): 20 fish channels: x/39,y/31,stock/8,water/6,pollution/6,netted,presence.
- Columns [2075, 2173): 7x7 dy-major patch: tile/3,in-bounds; unavailable cells zero.

All fixed-shape state leaves are listed in manifest.json. Entity attributes are represented by the corresponding native local block. The explicitly described own team score summaries are self information. Declared invariants include finite observations/rewards, coordinate bounds, resource nonnegativity under legal transitions, fixed state-tree shapes and absorbing terminal semantics. They remain source declarations, not passed runtime receipts.

## Literal diagnostic proposals

playthrough.json supplies two full-width literal scripts and one paired hypothesis. Every frame contains all 80 actions, including zeros. There is no adaptive policy, search or substitute dynamics. The branches have the same prefix and subsequent actions, and differ at exactly one actor/action/step. Intended differences come from source reasoning only. No script was imported, run, simulated, scored or accepted. Initial random details, multi-step legality, exact effects and rewards remain unverified; no solvability or learning result is claimed.

## Literal source evidence

- candidate.py:66: `def legal_actions(self, s):`
- candidate.py:123: `gates = s['gate_open'] ^ changed`
- candidate.py:126: `runoff = s['water'] * gates.astype(jnp.float32) * 0.4`
- candidate.py:130: `pollution = jnp.clip(s['pollution'] * 0.65 + poisonflow + cutcount * 0.5 + totals(fertilize) * 0.2 - reeds * 0.04, 0, 6)`
- candidate.py:139: `diversity = ((c % 5 > 0) & (cropkind[left] > 0) & (cropkind[left] != cropkind)) | ((c % 5 < 4) & (cropkind[right] > 0) & (cropkind[right] != cropkind))`
- candidate.py:140: `sheltered = (bh > 0) | ((c % 5 > 0) & (bh[left] > 0)) | ((c % 5 < 4) & (bh[right] > 0))`
- candidate.py:150: `held = jnp.any(atleft & (act[:, None] == 8), axis=0) & jnp.any(atright & (act[:, None] == 8), axis=0)`
- candidate.py:153: `netted = nets & (stock > 0) & (water > 0.5)`
- candidate.py:159: `rack_progress = jnp.where(rack_changed, 0, s['rack_progress'])`
- candidate.py:184: `swap = send & (act == 14)`
- candidate.py:190: `seedgive = send & (act == 15) & ~give`
- candidate.py:218: `attributable = toggle.astype(jnp.float32) * gates[h] * flood_loss[lower] * (h < 15) * (water[lower] - runoff[h] <= 4)`
- candidate.py:219: `reward = 0.35 * value + 0.1 * earnings[s['team']] + 0.02 * harvest - 0.003 * (act > 4) - 0.06 * spoiled - 0.04 * flood_loss[s['team']] - 0.04 * attributable - 0.002 * cut`
- candidate.py:221: `terminated = (t >= 96) & jnp.all(sales > 0)`
- candidate.py:232: `def raw_step(self, key, state, actions):`
- candidate.py:239: `def observe(self, s):`

## Provenance and remaining evidence

The supplied JaxMARL README informed parallel actor-interface expectations; Pgx informed legal masks, external keys and terminal states; Craftax README informed local materials/observation and no-autoreset boundaries. PureJaxRL README and Craftax_Baselines_ppo_rnn.py informed only learner/wrapper integration boundaries. No upstream code, assets, board layouts or game recipes were copied. No supplied external score was used as evidence for these authored games. Exact supplied reference bytes and uses are in author_input_freeze.json.

Authority and scope hashes were verified. ROOT_INPUT_REFERENCE_CLARIFICATION247.json explicitly supersedes the missing inherited194 reference; that absent file was not searched or read. Current247 population/family and current design/request bytes control. Actual allowed paths, byte counts, SHA256 and uses are frozen in author_input_freeze.json. No MEMORY, prior candidate/results, other coordination, training/IR, selection, heldout or final-test evidence entered authoring.

Author checks are AST parsing/whitelist-oriented text inspection, manifest/scene data inspection, literal script dimensions and single-action-difference counting, and byte/hash verification. Candidate code was never imported or executed. No NumPy/JAX/PRNG/runtime tests occurred. Runtime shape/parity/throughput, physical paired replay, native renderer/animation, arbitrary human play, learning and meta-learning transfer are UNKNOWN or UNEXECUTED. Independent immutable-source integration release is required.
