# Thermal Loom — game-v3-00084

Status: AUTHORED_NOT_EXECUTED. Initial source attempt 0, proposal ordinal 84, window 0, training-source stream only.

A=3, O=122, U=10, H=128. Observations are float32 within [-4, 128]; rewards are float32 within [-4, 4]. Stored numbers are int32 and flags bool. These finite bounds describe rules, not performance acceptance gates.

## Physical pipeline, output buffers and power
The 9 by 7 workshop has a middle divider with crossings at y=2 and y=4. Four separate raw pieces start at (1,1), (2,1), (1,2), (2,0). Stations are press (3,2), furnace (5,2), quench (5,4), dock (7,4), generator (1,5). Stage codes are 0 raw, 1 cut, 2 hot, 3 finished, 4 shipped, 5 scrap, 6 recycled. Owner is -1 ground, 0..2 worker, -2/-3/-4 inside press/furnace/quench. Real coordinates govern all movement, pickup, drop, load, shipping, cranking and conflict.

Each worker carries one item. Recover with empty hands selects the lowest-index eligible ground item within one; with full hands it attempts a drop at its cell. Another ground item or loaded machine at that cell blocks dropping. A completed output on a station blocks reloading, providing a single output buffer. Workers may occupy station cells; machines and workpieces are actual entities, not walls. Adjacent paired action 9 permits a handoff to an empty receiver. Agent collision rules make both crossings susceptible to queueing.

Only the cutter can load raw into press, annealer cut into furnace, and finisher hot into quench. Durations are three, four and two powered ticks. New loads may be powered immediately. At most two machines are powered per transition, each spending one common charge. Priority rotates from t modulo three, with no hidden random tie-break. Charge starts at twelve; any worker can crank four at the generator, capped at twenty-four. Scrap recycling there returns three. Completed machine output is a physical ground item at the station.

Furnace completion starts an eighteen-tick thermal window; newly heated output does not decay until the next transition. Existing hot items decay while carried, stored or processing. Quench completion precedes thermal expiry. Expiry makes scrap; if inside a machine it ejects to ground and frees the machine. Scrap can move or recycle, but cannot become usable material. Three shipments win; fewer than three possible good shipments is natural failure, retaining real partial deliveries and spoilage. The witness hypothesizes one full pipeline pass only.

## Coupled choices and alternatives
A cutter who fills its output buffer faster than the annealer collects it cannot continue. A worker holding a finished item cannot pick a hot one without shipping, dropping or handing off. Loading all three machines competes for two power channels; deferring a nonurgent press can protect a quench deadline. Sending a finisher to crank restores everybody's opportunities but risks its hot item. A worker parked in a crossing blocks a partner's load route. These dependencies propagate through holder, item position, machine occupancy, charge and deadlines.

Prospective paired tests: with all machines needed, compare press loading versus wait and trace rotating service/heat; compare output pickup versus wait and next loading eligibility; compare cranking versus wait at zero charge and all future timers; compare mutual versus one-sided handoff and the receiver's subsequent feasible load. No test was run by the author.

Separately proposed changes to station placement, recipe order, output capacity, power budget, heat window or role permissions could move the bottleneck between movement, energy and deadlines. The delivered candidate has exactly this one fixed pipeline.

## Public state and scene
All rule-relevant fields are public. Workpieces bind to real coordinates, stage, owner and heat; stations bind to actual loaded item and remaining work. Shipped/recycled entries remain in fixed arrays and observations as records but are inactive scene objects. Public event channels use observable counters, charge, timer and stage. Internal operation flags remain labelled debug information. Signed owner codes are represented losslessly as int32.

## Action legend
- 0: Wait.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Pick lowest-index adjacent ground item if empty; otherwise drop onto an unoccupied item buffer cell.
- 6: Load eligible held item into nearby role-specific machine.
- 7: Ship finished item at dock or recycle scrap at generator.
- 8: Crank four shared charge at generator.
- 9: Mutual adjacent item handoff to empty partner.

## Exact observation layout
- Columns [0, 6), self: Own role one-hot(3), x,y,carrying item index.
- Columns [6, 10), public: Clock, shared charge, cumulative delivered and spoiled.
- Columns [10, 19), public: All workers: x,y,carrying per slot.
- Columns [19, 39), public: All four pieces including retired records: x,y,stage,owner,heat.
- Columns [39, 59), public: Five stations: x,y,timer,loaded; slots press,furnace,quench,dock,generator.
- Columns [59, 122), public: Public 7-by-9 map flattened row-major yx.

## Interface and terminal boundary
The binding addendum terminal_policy is absorbing_raw_state_natural_priority_v1. A nonterminal input increments t exactly once. Natural completion has priority over horizon truncation. raw_step selects the original entire tree and zero reward when the input done flag was already true, so terminal clock, observations and causes are preserved. Pure advance expressions may be formed on terminal inputs, but their results are discarded without side effects. No raw call resets. info returns scalar done_all, terminated and truncated; dones is the joint flag repeated across A roles.

Out-of-range or masked actions deterministically become wait. Contextual commands are legal attempts, including unsuccessful attempts. Movement masks use terrain and bounds, with agent collisions resolved after simultaneous proposals. Entering another role's starting cell or a jointly proposed destination fails, so swaps and following into a just-vacated cell fail. All role slots remain physical; disabled/restrained roles keep no-op. The reset is deterministic and intentionally ignores supplied PRNG keys. No hidden mutable state, external cache, callbacks, stochastic distribution or remote service is used.

All small signed holder/status/index fields are represented losslessly as int32, as the addendum permits. Declared invariants are nonnegative fields listed in manifest.json and unique active agent positions. All source state leaves are declared. Runtime preservation, shapes, dtypes and finite bounds remain unverified by execution.

## Literal source evidence
Source SHA-256: `88d7919965eb62cd7750ba29240d0d417baf3405188b59d79089ff5bf1b60688`.
Manifest SHA-256: `6d2fc89b157de50a7794de7fae2386714c444f9df19399c273e262e1e99d2d63`.
Scene SHA-256: `bc3db240a30340af92ca5a36c19188a4bde0fabc56b163b461c59a03b97b27aa`.
Witness file SHA-256: `76c732b4d376a6ef4d85ec59e0e60384e9927665d4deca9d8b92831a8f5252e0`.

- `candidate.py:12-18`: `refresh`.
- `candidate.py:20-43`: `reset`.
- `candidate.py:45-51`: `legal_actions`.
- `candidate.py:53-60`: `observe`.
- `candidate.py:62-162`: `advance`.
- `candidate.py:164-169`: `raw_step`.

Actual transition expressions:
- `candidate.py:98`: `output_blocked = jnp.any((n['owner'] == -1) & (n['stage'] != 4) & (n['stage'] != 6) & jnp.all(n['item_pos'] == s['station_pos'][i], axis=1))`
- `candidate.py:99`: `load = (a[i] == 6) & has & close & (n['stage'][h] == i) & (n['loaded'][i] < 0) & ~output_blocked`
- `candidate.py:132`: `run = (n['loaded'][machine] >= 0) & (n['timer'][machine] > 0) & (n['charge'] > 0) & (budget > 0)`
- `candidate.py:145`: `spoil = (n['stage'] == 2) & (n['heat'] == 0)`

## Community references and integration boundary
Original rules and layouts were invented for this batch. No upstream code was copied. The supplied JaxMARL README informed the separate public integration target: parallel agent-keyed actions, observations, rewards and dones with __all__. This source implements the required native flat-array interface; the implementation worker owns the adapter. The supplied Pgx README informed legal-mask and explicit termination concepts, not board-game rules. Craftax informed the distinction between actual game state and rendering and raw stepping versus external resets. PureJaxRL and Craftax_Baselines/ppo_rnn.py are algorithm/integration references; they are not a new game library or a replacement for the frozen experimental learner. Only supplied snapshots were consulted. Reference URLs and exact snapshot hashes are retained in generation_record.json; no external link was followed.

## Evidence limits
This package was not executed. Author-side AST whitelist, JSON/schema parsing, state-key comparison and hashing are static preflight only. They do not establish runtime correctness, eager/JIT/vmap parity, witness legality, successful play, solvability, strategic depth, learning, non-isomorphism, throughput, animation acceptance or meta-learning/transfer. The prospective paired-action comparisons above require actual later execution. No observed score, successful trajectory, training or test receipt is fabricated. No JAX import, model/API call, remote/SSH execution or download occurred. The proposal has one fixed ruleset/layout; possible future variations are ideas, not a hidden sampler. Repairs after submission require a separately assigned actual-feedback request.
