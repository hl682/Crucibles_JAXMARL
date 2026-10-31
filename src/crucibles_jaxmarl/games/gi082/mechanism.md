# Lantern Flood — game-v3-00082

Status: AUTHORED_NOT_EXECUTED. Initial source attempt 0, proposal ordinal 82, window 0, training-source stream only.

A=3, O=121, U=9, H=96. Observations are float32 within [-1, 96]; rewards are float32 within [-2, 2]. Stored numbers are int32 and flags bool. These finite bounds describe rules, not performance acceptance gates.

## Spatial rules and temporal consequences
The original 9 by 7 layout has a central barrier, gated crossing at (4,3), and longer southern crossing through y=6. Roles start at (1,1), (1,3), (1,5); casualties at (7,1), (7,5), with unsupported lifetimes 54 and 60. Depot (0,3) accepts a carried casualty from distance at most one. Packs at (2,3) and (6,6) each add two supplies; initial common supply is three.

Movement updates actual carrier and casualty positions. Recover selects the lowest-index adjacent free casualty if the role is empty, then attempts a pack if no pickup occurred. Each pack is consumed once. Rigger work at gate distance at most one spends two and opens permanently. Medic work spends one on the lowest-index adjacent unstabilized casualty. Stabilization arrests countdown without delivering. Scout specialist work has no effect. Handoff requires both donor and adjacent empty receiver to choose action 7; each receiver accepts at most one casualty. Beacons stay at deployment coordinates, last six ticks and refresh when reused; their cost is the action opportunity, not supplies.

Pickup, treatment and handoff follow movement. Delivery occurs before lifetime expiration. Unresolved, unstabilized and undelivered casualties lose one lifetime tick. Rescued or expired casualties become inactive; status preserves the outcome. Any mixture of rescue and loss ends naturally when both are resolved. Reward is only shared delivery minus expiry events, without discovery or movement shaping.

## Partner dependence and strategies
Opening early lets the scout and medic use the short crossing but consumes two of three initial supplies. Treating both casualties first leaves too little to open without a recovered pack. Alternatives include south-route rescue, early supply recovery, fast transport without stabilization, or handoffs freeing a specialist. Scout sensing and well-placed beacons reveal real casualty states before teammates commit to paths. A teammate blocking a junction imposes lifetime cost on a carrier.

Prospective paired tests: compare adjacent rigger work with wait, then gate state, common supply and another role's future crossing mask; compare medic work with wait and future casualty lifetime; compare beacon versus wait and teammates' victim_vis/observations; compare mutual versus donor-only handoff and subsequent physical coordinates. These are source hypotheses, not executed effects.

Future separately specified variations in opening/treatment cost, sensor radius, beacon lifetime, casualty positions and route length could change rescue order and delegation. They are not implemented or measured here.

## Information and scene boundary
The map, landmarks, supply and aggregate outcomes are public. Sensors use Manhattan distance through walls, which block movement but not sensing. Local casualty attributes exactly match native observed life/stable/carrier/status. Only public aggregate counters, gate openness and supply appear in scene event channels. Private treatment/pickup/handoff diagnostics remain raw debug state, not public role overlays.

## Action legend
- 0: Wait.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Recover: pick adjacent casualty if empty; otherwise collect adjacent supply pack.
- 6: Specialist work: rigger opens sluice; medic stabilizes casualty; scout has no effect.
- 7: Mutual adjacent casualty handoff.
- 8: Place or renew six-tick shared sensor beacon.

## Exact observation layout
- Columns [0, 6), self: Own role one-hot(3), x,y,carrying index (-1 empty).
- Columns [6, 11), public: Clock, shared supply, gate-open flag, cumulative rescued and lost.
- Columns [11, 23), local: Visible rescuers: x,y,carrying per slot (9), then 3 visibility flags; hidden features -1.
- Columns [23, 37), local: Visible casualties: x,y,life,stable,carrier,status per slot (12), then 2 visibility flags; hidden features -1.
- Columns [37, 45), local: Visible packs: x,y,value per slot (6), then 2 visibility flags; hidden features -1.
- Columns [45, 54), public: Public beacons: x,y,TTL per slot; inactive features -1.
- Columns [54, 117), public: Public 7-by-9 map flattened row-major yx.
- Columns [117, 121), public: Public gate x,y followed by depot x,y.

## Interface and terminal boundary
The binding addendum terminal_policy is absorbing_raw_state_natural_priority_v1. A nonterminal input increments t exactly once. Natural completion has priority over horizon truncation. raw_step selects the original entire tree and zero reward when the input done flag was already true, so terminal clock, observations and causes are preserved. Pure advance expressions may be formed on terminal inputs, but their results are discarded without side effects. No raw call resets. info returns scalar done_all, terminated and truncated; dones is the joint flag repeated across A roles.

Out-of-range or masked actions deterministically become wait. Contextual commands are legal attempts, including unsuccessful attempts. Movement masks use terrain and bounds, with agent collisions resolved after simultaneous proposals. Entering another role's starting cell or a jointly proposed destination fails, so swaps and following into a just-vacated cell fail. All role slots remain physical; disabled/restrained roles keep no-op. The reset is deterministic and intentionally ignores supplied PRNG keys. No hidden mutable state, external cache, callbacks, stochastic distribution or remote service is used.

All small signed holder/status/index fields are represented losslessly as int32, as the addendum permits. Declared invariants are nonnegative fields listed in manifest.json and unique active agent positions. All source state leaves are declared. Runtime preservation, shapes, dtypes and finite bounds remain unverified by execution.

## Literal source evidence
Source SHA-256: `ad66babf039e1ebd3643e035f8cd685e5641ebd2a67714c4e683c5a91ac5af2e`.
Manifest SHA-256: `9722a9e0785236a9b080b9a0a316206a30188562b984bedb4fedac73e91eaba7`.
Scene SHA-256: `859328c2a4012664ba6a148b1028b600a78dc547b657c75b87b855b9a4bddf94`.
Witness file SHA-256: `adb673bd274375a4e7dd1570441cc2141734327c1d28786b08f5f02c049e7bd2`.

- `candidate.py:12-29`: `refresh`.
- `candidate.py:31-61`: `reset`.
- `candidate.py:63-70`: `legal_actions`.
- `candidate.py:72-84`: `observe`.
- `candidate.py:86-156`: `advance`.
- `candidate.py:158-163`: `raw_step`.

Actual transition expressions:
- `candidate.py:102`: `opened = (a[1] == 6) & near_gate & ~s['gate_open'][0] & (s['supply'] >= 2)`
- `candidate.py:124`: `stabilize = (a[2] == 6) & jnp.any(need) & (n['supply'] > 0)`
- `candidate.py:138`: `handoff = (before[v] >= 0) & (a[c] == 7) & jnp.any(willing) & ~used`
- `candidate.py:142`: `delivery = (n['carrier'] >= 0) & (jnp.sum(jnp.abs(n['victim_pos'] - s['depot_pos'][0]), axis=1) <= 1) & (s['victim_status'] == 0)`

## Community references and integration boundary
Original rules and layouts were invented for this batch. No upstream code was copied. The supplied JaxMARL README informed the separate public integration target: parallel agent-keyed actions, observations, rewards and dones with __all__. This source implements the required native flat-array interface; the implementation worker owns the adapter. The supplied Pgx README informed legal-mask and explicit termination concepts, not board-game rules. Craftax informed the distinction between actual game state and rendering and raw stepping versus external resets. PureJaxRL and Craftax_Baselines/ppo_rnn.py are algorithm/integration references; they are not a new game library or a replacement for the frozen experimental learner. Only supplied snapshots were consulted. Reference URLs and exact snapshot hashes are retained in generation_record.json; no external link was followed.

## Evidence limits
This package was not executed. Author-side AST whitelist, JSON/schema parsing, state-key comparison and hashing are static preflight only. They do not establish runtime correctness, eager/JIT/vmap parity, witness legality, successful play, solvability, strategic depth, learning, non-isomorphism, throughput, animation acceptance or meta-learning/transfer. The prospective paired-action comparisons above require actual later execution. No observed score, successful trajectory, training or test receipt is fabricated. No JAX import, model/API call, remote/SSH execution or download occurred. The proposal has one fixed ruleset/layout; possible future variations are ideas, not a hidden sampler. Repairs after submission require a separately assigned actual-feedback request.
