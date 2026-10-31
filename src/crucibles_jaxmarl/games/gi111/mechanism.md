# Counter-Signal Garden: Relay, Guard and Disrupt

Status: **AUTHORED_NOT_EXECUTED**. Source SHA-256: 0b28fc38d89bddbf15e2acd499d1803733f583a42634a8e19828a71f042eef36. A/O/U/H = 4/146/10/80. All playthroughs and paired interventions are **UNEXECUTED_AUTHOR_HYPOTHESIS**.

## Rules and interaction

This is a simultaneous two-versus-two contest on a 9 by 9 hedge garden. Three actual scoring plots are also the openings through the central hedge. Each team owns one mobile transmitter. Either teammate can pick it up, but while carried only that holder can pulse it. Picking up an object therefore changes a teammate's feasible use of the same resource. Both teams can contest passage squares and block physical movement.

A pulse consumes one unit of the relay's four-unit battery, is emitted at its actual coordinate, and persists there for three transitions. Ordinary radius is three in Manhattan distance. An adjacent teammate who chooses guard instead of moving/pulsing increases a new pulse radius to four and blocks every simultaneous enemy jam. Guarding and pulsing require separate actions, so protecting a firing carrier typically needs its partner. A carrier is allowed to choose guard or jam, but then cannot pulse in that same step.

Enemy agents within Manhattan distance two can jam. Unprotected hits reduce the battery and the old pulse lifetime before a new pulse is considered. The mobile relay may move out of a legal jammer's original reach. Recharge is physical: both relay and recharging teammate must be within one square of the native base, and aggregate recharge is capped at one unit per transition. These timing rules permit escort, raids, withdrawal and denial without creating a separate single-agent objective for each role.

Unopposed coverage moves each plot's signed control one step toward the emitting team, bounded [-3,3]. Ties/no coverage leave control unchanged. Reaching +2 or -2 assigns ownership; prior ownership persists in the middle band. An owner scores only while its own live signal covers the plot and the opponent's does not. Thus ownership is persistent, but a stale flag alone does not keep scoring. At 24 accumulated points either team ends play; both reaching it together is allowed. The condition is a game win/end rule, not a scientific acceptance threshold.

Reward is the difference between team step points divided by four, minus the acting individual's 1/128 non-wait cost. Intermediate pulses, guards and captures are not independent reward tokens.

## Information, strategies and failure cases

Agents and plots are sensed within radius three. Each team receives its own relay and signal telemetry globally; enemy telemetry is limited to local sensing. This deliberate native-radio asymmetry is applied both to observation payloads and entity visibility. Only team point totals are public event channels. A player cannot see remote opponent battery, ownership flags or agent inventories through the scene.

A pair can keep its relay at home for short-range denial, carry it to central plots with an escort, split a jammer from a carrier, or exploit the opponent's need to recharge. Moving one relay to support a new plot leaves its existing pulse behind and may strand a teammate. Mutually protected symmetric play can stall, uncoordinated actors can drain their only battery, and a raid can fail when the target moves. None of these hypotheses is a learned policy result.

Different hedge openings, signal lifetimes, battery budgets or radio scopes could change these best responses in future separately frozen variants. Those alternatives are not silently randomized here.

## Declared invariants

Agents occupy distinct walkable cells. Each relay has one fixed team and at most one carrier; each agent can hold only its native relay. Battery stays in [0,4], pulse life in [0,3], radius in {3,4}, control in [-3,3], and owners in {-1,0,1}. All coordinates are actual transition coordinates; signals are inactive when their TTL is zero. Each scoring plot awards at most one team per transition. All 146 observation columns are labelled, and hidden payloads are zeroed rather than padded with privileged values.

## Boundary, invalid actions and terminal behavior

The native interface is GeneratedEnv.reset(key), observe(state), legal_actions(state), and raw_step(key, state, int32[A]). Native learning receives only float32[A,O] observations and bool[A,U] masks. Out-of-range or masked-illegal actions become action 0 without action cost. Joint conflicts can still prevent legal attempted actions from succeeding. All roles remain active until joint completion; there are no individually terminated roles. At joint completion, only no-op is legal. A nonterminal step increments scalar int32 t exactly once. Natural completion takes precedence over the horizon. An already-terminal raw call preserves every state leaf, clock, observation and terminal cause; it returns zero float32 rewards and all-true dones. No raw autoreset occurs.

The agent-keyed public JaxMARL adapter belongs to the separate trusted engineering layer: use agent_0 through agent_3, preserve row/action order and masks, and set the done dictionary __all__ from info.done_all. The raw array contract and frozen learner are unchanged. Adapter execution and compatibility are not claimed here.

State indices, sentinels, counts, stages and temperatures use int32; visibility/termination use bool; observations/rewards use float32. All normalization divisors are powers of two. This boundary representation does not replace a previously specified random distribution. The PRNG key is not copied into state. This source has a fixed deterministic map and dynamics; reset/step keys are accepted but unused.

The map is public knowledge. Role rendering must honor the same entity visibility masks as native observations and suppress hidden attributes. Only public channels appear in events; private source leaves are not display attributes. Full-state debug views require a separate label. Scene positions, active flags, visibility and attributes bind real state; no presentation-generated transitions or executable markup occur.

## Literal source evidence

- Native radio and local visibility are explicit state masks. candidate.py:L15-L25 (exact excerpt SHA-256 d68fc023706644779d0b99063c75c933ecfedd41f0d4614981047af4c95bd1d0).
- The legal mask reflects relay possession and physical range. candidate.py:L67-L82 (exact excerpt SHA-256 2ff7ef8f37d0799264a9aab7165dcfbc979f36f529846b3ac4517e02e150e608).
- Observation payloads are zeroed using those same masks. candidate.py:L84-L114 (exact excerpt SHA-256 6e0a675e1214e28f131d1c85c0f1afbc956e7f619df84bf3c778ebb4184382eb).
- Picking up the only relay changes teammate access. candidate.py:L129-L140 (exact excerpt SHA-256 bb3962f452342ed5277a0ef1eefd476a836799c0db56cb868b51d94bb85c1b80).
- Guard, jam, pulse and recharge act on one conserved ownership record and bounded battery. candidate.py:L147-L163 (exact excerpt SHA-256 53f09299e47486fa2b10651e3ca46623e1bcedd631c66f69c76b92d9b48606ea).
- Finite spatial coverage changes persistent control and actual scoring. candidate.py:L164-L171 (exact excerpt SHA-256 a52ce9ccef6d198632a884b7b3dfb53fad9458d78cf3586b007bb6a67d0d92bb).
- Terminal causes and absorbing raw returns are explicit. candidate.py:L186-L195 (exact excerpt SHA-256 e53b1db0a2c10009e00ee160f9f19b7a622fa11ea696b800ff9f5a5e77b8ce9c).

## Interface method locations

- moves: candidate.py:L12-L13.
- refresh: candidate.py:L15-L25.
- reset: candidate.py:L27-L57.
- move_options: candidate.py:L59-L65.
- legal_actions: candidate.py:L67-L82.
- observe: candidate.py:L84-L114.
- raw_step: candidate.py:L116-L195.

## Reference attribution and limits

Original rules and kernel were authored for this assigned design. No upstream implementation code, layouts, assets, sprites or trained parameters were copied. JaxMARL's README supplied the agent-keyed dictionary and parallel-action integration convention. Pgx's README supplied board-state, explicit legal-mask and external-random-key API reference. Craftax's README supplied the general example of executable spatial resources and staged game mechanics in JAX. PureJaxRL is an algorithm implementation reference, not a new environment library. The supplied Craftax_Baselines PPO-RNN header credits PureJaxRL; it was consulted only as baseline/integration reference and was not copied or substituted for the frozen learner. No reference performance claim is evidence about these games. Snapshot identity was checked in full; consulted passages were API/overview material and the first 120 lines of the PPO-RNN reference.

- references\JaxMARL_README.md: [official snapshot origin](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; verified 12438 bytes.
- references\Pgx_README.md: [official snapshot origin](https://github.com/sotetsuk/pgx/blob/main/README.md), SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; verified 17543 bytes.
- references\PureJaxRL_README.md: [official snapshot origin](https://github.com/luchris429/purejaxrl/blob/main/README.md), SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; verified 5023 bytes.
- references\Craftax_Baselines_ppo_rnn.py: [official snapshot origin](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py), SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; verified 18983 bytes.
- references\Craftax_README.md: [official snapshot origin](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; verified 12388 bytes.

## What was and was not checked

Only AST parsing, the supplied whitelist, JSON manifest validation, scene-schema validation, declared-state-key agreement, source byte cap, hashes and script integer-vector structure were checked. The candidate was never imported, compiled, run or numerically simulated. JAX and NumPy were absent from the checker process imports. Reset/step shapes and dtypes, eager/JIT/vmap parity, numerical bounds and conservation under execution, repeated-seed replay, trajectory legality, real paired effects, absorbing execution, rendering, playback/pause/step, public adapter, policy training, solvability and meta-learning remain unexecuted. No acceptance score, success-rate threshold, ranking or final-test information was used.
