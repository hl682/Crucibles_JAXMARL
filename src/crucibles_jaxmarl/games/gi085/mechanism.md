# Cipher Causeway — game-v3-00085

Status: AUTHORED_NOT_EXECUTED. Initial source attempt 0, proposal ordinal 85, window 0, training-source stream only.

A=4, O=154, U=10, H=96. Observations are float32 within [-1, 96]; rewards are float32 within [-1, 1]. Stored numbers are int32 and flags bool. These finite bounds describe rules, not performance acceptance gates.

## Asymmetric pursuit and route control
The original 9 by 7 divided chamber has crossing gates at (4,2), (4,4). Courier/engineer start at (6,1)/(6,5); wardens at (2,1)/(2,5). Cipher starts at (7,3) and must reach exactly (1,3). Both runners can carry. Initial gates remain powered for three ticks. A role standing in a gate keeps it open until vacating, even when restrained. This prevents a closing wall occupying a body and creates a contestable route foothold.

Courier pays two shared runner cells to open an adjacent gate for five ticks; engineer pays one. Either warden pays one warden cell to lock. Simultaneous gate commands resolve in role order, so later opposing control can override an earlier command, while consuming both costs. An occupied cell still stays open. Gate/device actions without sufficient resources fail harmlessly as legal attempts. Caches at (7,0), (1,6) add two cells to whichever team recovers them; resource stealing is allowed.

Warden action 5 tags an unrestrained runner at distance at most one, preferring the carrier when equally close. Restrained runners are physical bodies, observed when sensed, with no-op only. Tagging drops cipher at the actual carrier position. After tags, a still-free adjacent runner using action 9 can release its partner. Freed roles act next decision and do not automatically recover cipher. Both restrained after this rescue phase is natural failure. Extraction after tag/rescue resolution is natural success. Only those events pay +/-1. Horizon expiry gives no bonus.

## Information as an interactive resource
Public four-tick devices occupy actual deployment coordinates. Cloak buoys hide runners within one tile from wardens beyond adjacency. Scanners share radius-four agent detection across the warden team and override cloaks. A role's own live flare also defeats its concealment. Three-tick flares continuously follow their emitter, disclose that role to everyone and relay radius-three agent sensing to its own team. Gate operations and devices compete for the same team cells. Redeploying moves a role's sole device and refreshes it.

Engineer timing opens a sealed route at half the courier cost. Holding a gate while a partner approaches is useful but also obstructs that cell until the holder moves, with nearby tags a risk. Cloaks enable staging, scanners counter concealment, and a flare trades secrecy for team information. Wardens can guard a dropped cipher, conserve cells for route denial, or pursue the free rescuer. Runner partners can escort, rescue or split. All four roles are player-controlled; there is no scripted opponent or guaranteed win.

Prospective paired tests: compare engineer gate work and wait at witness step 5, then gate timer and later carrier legal mask; compare cloak deployment and wait with a runner at warden distance two, then visibility and flat observations; add scanner to check override; compare tag and wait, then captured status, cipher holder/position and next mask; compare adjacent partner rescue with waiting. All are unexecuted hypotheses.

Future separately proposed changes in gate duration, role cost, sensor range, cache position, rescue rule or exit geometry could change escort/interception. No hidden task sampler or adaptation result is implemented.

## Scene privacy
Sensors use Manhattan distance through walls, not ray-cast optics. Ground cipher, gates and caches use direct local range; carried cipher follows carrier visibility. Relays/scanners expand agent sensing only. Own-team energy is observed, opponent pool is not. Gate movement masks cannot reveal a distant gate because entry is from adjacency, within every role's local range. Entity visibility is bound to actual state masks. Public scene events include extraction and already-public device/flare lifetimes only; capture and remote gate diagnostics require a labelled full-state view.

## Action legend
- 0: Wait.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Runner picks adjacent cipher; warden tags adjacent unrestrained runner.
- 6: Runner opens nearest adjacent gate for five ticks; warden locks it.
- 7: Emit three-tick public live flare and share radius-three agent sensing with teammates.
- 8: Spend shared cell to deploy four-tick cloak buoy for runner or scanner for warden.
- 9: Runner frees adjacent captured teammate first; otherwise recover adjacent battery cache.

## Exact observation layout
- Columns [0, 9), self: Own role one-hot(4), x,y,captured,carrying,own-team energy.
- Columns [9, 12), public: Clock, extracted flag, joint done flag.
- Columns [12, 36), local: Visible roles: x,y,captured,carrying,flare TTL each (20), then four visibility flags; hidden features -1.
- Columns [36, 41), local: Visible cipher x,y,carrier,active (4), then visibility flag; hidden features -1.
- Columns [41, 51), local: Visible gates x,y,open,timer each (8), then two visibility flags; hidden features -1.
- Columns [51, 67), public: Public devices x,y,TTL,type per slot; inactive features -1.
- Columns [67, 79), public: Public live flares x,y,TTL per slot; inactive features -1.
- Columns [79, 89), local: Visible caches x,y,value,active each (8), then two visibility flags; hidden features -1.
- Columns [89, 91), public: Public extraction exit x,y.
- Columns [91, 154), public: Public 7-by-9 map flattened row-major yx.

## Interface and terminal boundary
The binding addendum terminal_policy is absorbing_raw_state_natural_priority_v1. A nonterminal input increments t exactly once. Natural completion has priority over horizon truncation. raw_step selects the original entire tree and zero reward when the input done flag was already true, so terminal clock, observations and causes are preserved. Pure advance expressions may be formed on terminal inputs, but their results are discarded without side effects. No raw call resets. info returns scalar done_all, terminated and truncated; dones is the joint flag repeated across A roles.

Out-of-range or masked actions deterministically become wait. Contextual commands are legal attempts, including unsuccessful attempts. Movement masks use terrain and bounds, with agent collisions resolved after simultaneous proposals. Entering another role's starting cell or a jointly proposed destination fails, so swaps and following into a just-vacated cell fail. All role slots remain physical; disabled/restrained roles keep no-op. The reset is deterministic and intentionally ignores supplied PRNG keys. No hidden mutable state, external cache, callbacks, stochastic distribution or remote service is used.

All small signed holder/status/index fields are represented losslessly as int32, as the addendum permits. Declared invariants are nonnegative fields listed in manifest.json and unique active agent positions. All source state leaves are declared. Runtime preservation, shapes, dtypes and finite bounds remain unverified by execution.

## Literal source evidence
Source SHA-256: `600fbe762c0281fb92813bee6baabf6f3219ad1be84c1abd4346bd1b0de72ca1`.
Manifest SHA-256: `da64bf618fa9d30803de293f1ef1879e529c26b8b5c76ef057809ea8412f8163`.
Scene SHA-256: `ba5a9ba35dba6d57a6eeb210586ed16b7d37815400900793a9091b0ed52d95bf`.
Witness file SHA-256: `2b593c231676c749d2606f1094d36726c70d9fdf5dcb0fa158b3559427cc068e`.

- `candidate.py:12-43`: `refresh`.
- `candidate.py:45-78`: `reset`.
- `candidate.py:80-89`: `legal_actions`.
- `candidate.py:91-108`: `observe`.
- `candidate.py:110-183`: `advance`.
- `candidate.py:185-190`: `raw_step`.

Actual transition expressions:
- `candidate.py:24`: `obscure = (teams[:, None] == 1) & (teams[None, :] == 0) & hidden[None, :] & ~scanned[None, :] & (distance > 1) & (s['flare_ttl'][None, :] == 0)`
- `candidate.py:135`: `control = (a[i] == 6) & (dist[g] <= 1) & (n['energy'][teams[i]] >= cost)`
- `candidate.py:156`: `tag = (a[i] == 5) & jnp.any(eligible)`
- `candidate.py:164`: `free = (a[i] == 9) & (teams[i] == 0) & ~n['captured'][i] & n['captured'][mate] & adjacent`
- `candidate.py:173`: `extraction = (n['cipher_carrier'][0] >= 0) & jnp.all(n['cipher_pos'][0] == s['exit_pos'][0])`

## Community references and integration boundary
Original rules and layouts were invented for this batch. No upstream code was copied. The supplied JaxMARL README informed the separate public integration target: parallel agent-keyed actions, observations, rewards and dones with __all__. This source implements the required native flat-array interface; the implementation worker owns the adapter. The supplied Pgx README informed legal-mask and explicit termination concepts, not board-game rules. Craftax informed the distinction between actual game state and rendering and raw stepping versus external resets. PureJaxRL and Craftax_Baselines/ppo_rnn.py are algorithm/integration references; they are not a new game library or a replacement for the frozen experimental learner. Only supplied snapshots were consulted. Reference URLs and exact snapshot hashes are retained in generation_record.json; no external link was followed.

## Evidence limits
This package was not executed. Author-side AST whitelist, JSON/schema parsing, state-key comparison and hashing are static preflight only. They do not establish runtime correctness, eager/JIT/vmap parity, witness legality, successful play, solvability, strategic depth, learning, non-isomorphism, throughput, animation acceptance or meta-learning/transfer. The prospective paired-action comparisons above require actual later execution. No observed score, successful trajectory, training or test receipt is fabricated. No JAX import, model/API call, remote/SSH execution or download occurred. The proposal has one fixed ruleset/layout; possible future variations are ideas, not a hidden sampler. Repairs after submission require a separately assigned actual-feedback request.
