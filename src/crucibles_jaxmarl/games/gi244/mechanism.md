# Glacier Foundry Settlements

AUTHORED_NOT_EXECUTED. Candidate game-v3-00244, attempt00, train-only, feedback0, parentless blind authoring. Source SHA256 856975d66e3c07c06c8e3daf2be1c678affa83099a26696f7ebfff4b38868ec4. Native dimensions A=384, O=4144, U=16, H=896. Every actor has an independently supplied scalar action, observation/reward/done row, actual position and persistent spatial consequences. There are 24 groups of 16; grouping identifies affiliation, not a single controller for that group.

## Task and interaction

Commission twenty-four settlements by lifting stone with a separate brake operator, then warming each delivered-load workshop for eight working ticks. Protect crews and shared routes.

A separate brake changes whether an actual moving load lands or blocks a route. Carried heat protects against cold and commissions workshops, while melting ice creates south-flowing water that weakens support and injures downstream actors. Physical resource handoffs and warning markers connect crews.

Crews can rush brake relays and heat transport, or invest early stone in safe foundations/shelters. Heating ice creates a shortcut risk for downstream workers. Scouts can move warnings beyond the originating crew. Supporting a partner spends scarce materials and action time; abandoning a lift creates a persistent obstacle.

The 48x32 grid uses integer [x,y] cell centers, y down. Positions govern movement, object work and transfer. Scene colors/glyphs are original declarative cues for the independent block_survival family. The shared platform can select arbitrary one/multiple actors and fill their unchanged native action rows. This package does not claim a working UI, animation or human acceptance.

## Action meanings

- 0: Wait / receive adjacent handoff.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Rest; heals only on shelter.
- 6: Miner: quarry self/cardinal-neighbor stone.
- 7: Fuelwright: salvage self/cardinal-neighbor fuel.
- 8: Mason: spend stone on support at feet.
- 9: Hold separate brake while standing on pedal.
- 10: Mason: attach carried stone at hoist.
- 11: Fuelwright: consume carried heat cell at feet.
- 12: Mason: spend stone to build shelter at feet.
- 13: Scout: place carried marker or pick placed marker up.
- 14: Miner: clear self/cardinal-neighbor rubble.
- 15: Offer one stone, else fuel, to adjacent waiting settler.

Invalid or out-of-range action IDs deterministically become action0. Wait remains legal for every actor, including disabled actors and terminal input. Individual masks expose local physical/tool conditions; partner cooperation is resolved by the joint transition. Finite-resource contention uses clock-rotating actor priority. The same masks and native actions apply to human and AI ownership.

## Transition order, rewards and termination

Old-state masks; co-occupancy-permitted movement; rotating finite-resource extraction/building; heat, melt and southward water; markers, hoist/brake and workshop; adjacent one-to-one inventory offers; cold/flood damage; reward and terminal causes.

Clipped to [-1,2]. Before clipping: own-settlement landing +0.5; warm-workshop completion +0.2; construction +0.025; work -0.003; carried stone/fuel each -0.001; injury -0.04; tagged water-source heater -0.001 per flood victim; all actors +0.01 times fraction of new landings. The water-owner tag is flow bookkeeping, not a statistical causal estimate.

Natural termination: All twenty-four workshops have eight warm commissioned ticks, or no settler remains alive; otherwise H=896. Dropped loads remain failed and can leave the objective unfinished. Natural completion or failure takes priority if it coincides with H. truncated=(next t>=H) AND NOT terminated; done_all is their disjunction and equals episode_done. A nonterminal input advances t once. Already-terminal input is preserved by the final tree selection: clock, state, observation and terminal causes remain; reward is zero and all dones true. There is no internal reset.

The reset key is external trajectory provenance; only the supplied key drives reset stochasticity. Authored steps are deterministic and do not consume their key. Small signed values are int32, booleans bool, persistent reals float32. The uint32 key is not duplicated in world state. Exact eager/JIT/vmap behavior and numerical parity are untested.

## Exact observations and role rendering

Local entity blocks use fixed ID order and explicit presence. Hidden payloads are zero. Map patches enumerate dy=-3..3, then dx=-3..3. The local square is not occlusion-limited. Debug/full-state views must be labeled and cannot become policy observations. Actual event flags/counters must be filtered by the related visible entity; their declaration does not authorize off-screen event disclosure.

- Columns [0, 11): Time/896,ID/383,settlement/23,tool/3,x/47,y/31,stone/3,fuel/3,health/100,marker,alive.
- Columns [11, 3467): 384 actor slots, nine fields each: x/47,y/31,team/23,tool/3,stone/3,fuel/3,health/100,marker,alive/presence. Invisible slots zero.
- Columns [3467, 3635): 24 crane slots: x/47,y/31,phase/3,progress/3,brake,workshop/8,presence.
- Columns [3635, 3731): 24 brake slots: x/47,y/31,held,presence.
- Columns [3731, 3899): 24 moving-load slots: x/47,y/31,phase/3,progress/3,drop,delivery,presence.
- Columns [3899, 4144): 7x7 dy-major patch (-3..3): each cell tiles/8,water/6,heat/8,support/24,in-bounds-and-alive. Other cells zero.

All fixed-shape state leaves are listed in manifest.json. Entity attributes are represented by the corresponding native local block. The explicitly described own team score summaries are self information. Declared invariants include finite observations/rewards, coordinate bounds, resource nonnegativity under legal transitions, fixed state-tree shapes and absorbing terminal semantics. They remain source declarations, not passed runtime receipts.

## Literal diagnostic proposals

playthrough.json supplies two full-width literal scripts and one paired hypothesis. Every frame contains all 384 actions, including zeros. There is no adaptive policy, search or substitute dynamics. The branches have the same prefix and subsequent actions, and differ at exactly one actor/action/step. Intended differences come from source reasoning only. No script was imported, run, simulated, scored or accepted. Initial random details, multi-step legality, exact effects and rewards remain unverified; no solvability or learning result is claimed.

## Literal source evidence

- candidate.py:61: `def legal_actions(self, s):`
- candidate.py:104: `gather = self.winner(target, (act == 6) | (act == 7) | (act == 14), 1536, s['t'])`
- candidate.py:111: `supports = s['support'].reshape(-1).at[cell].add(jnp.where(build & (act == 8), 12, 0)).reshape((32, 48))`
- candidate.py:124: `water = jnp.clip(s['water'] * 0.55 + incoming + melt.astype(jnp.float32) * heat * 0.7, 0, 6)`
- candidate.py:138: `drop = suspended & ~brake`
- candidate.py:147: `workshop = jnp.minimum(s['workshop'] + ((phase == 2) & (heat[s['crane_pos'][:, 1], s['crane_pos'][:, 0]] > 0.7)).astype(jnp.int32), 8)`
- candidate.py:152: `can = (act[:, None] == 15) & (act[None, :] == 0) & s['alive'][None, :] & (dist <= 1) & (idx[:, None] != idx[None, :])`
- candidate.py:167: `penalties = jnp.zeros((384,), jnp.float32).at[jnp.maximum(blamed, 0)].add((flood & s['alive'] & (blamed >= 0)).astype(jnp.float32) * 0.001)`
- candidate.py:168: `reward = 0.5 * delivery[s['team']] + 0.2 * warmed[s['team']] + 0.025 * build - 0.003 * (act >= 6) - 0.001 * (stone + fuel) - 0.04 * damage - penalties`
- candidate.py:171: `terminated = ~jnp.any(alive) | jnp.all(workshop == 8)`
- candidate.py:180: `def raw_step(self, key, state, actions):`
- candidate.py:188: `def observe(self, s):`

## Provenance and remaining evidence

The supplied JaxMARL README informed parallel actor-interface expectations; Pgx informed legal masks, external keys and terminal states; Craftax README informed local materials/observation and no-autoreset boundaries. PureJaxRL README and Craftax_Baselines_ppo_rnn.py informed only learner/wrapper integration boundaries. No upstream code, assets, board layouts or game recipes were copied. No supplied external score was used as evidence for these authored games. Exact supplied reference bytes and uses are in author_input_freeze.json.

Authority and scope hashes were verified. ROOT_INPUT_REFERENCE_CLARIFICATION247.json explicitly supersedes the missing inherited194 reference; that absent file was not searched or read. Current247 population/family and current design/request bytes control. Actual allowed paths, byte counts, SHA256 and uses are frozen in author_input_freeze.json. No MEMORY, prior candidate/results, other coordination, training/IR, selection, heldout or final-test evidence entered authoring.

Author checks are AST parsing/whitelist-oriented text inspection, manifest/scene data inspection, literal script dimensions and single-action-difference counting, and byte/hash verification. Candidate code was never imported or executed. No NumPy/JAX/PRNG/runtime tests occurred. Runtime shape/parity/throughput, physical paired replay, native renderer/animation, arbitrary human play, learning and meta-learning transfer are UNKNOWN or UNEXECUTED. Independent immutable-source integration release is required.
