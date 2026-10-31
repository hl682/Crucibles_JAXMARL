# Tugboat Convoy: Two Channels, One Hydraulic Selector

Status: **AUTHORED_NOT_EXECUTED**. Source SHA-256: fbc6da3df61915cf82a7b5d395befa7b60e480b958266083e6df10ed664a0c60. A/O/U/H = 4/142/12/64. All playthroughs and paired interventions are **UNEXECUTED_AUTHOR_HYPOTHESIS**.

## Rules and interaction

The actual world is an 11 by 9 harbour. Cargo centres start at (3,2) and (3,6), with two tugs bracketing each load north/south. Both can be delivered eastward, but the x=5 pier divides the routes into north and south three-row channels. Selector posts at (4,2)/(4,6), gates and docks are physical coordinates used by the kernel.

A tug can attach to one adjacent undelivered cargo. A tow action is a cardinal force vote, not an independent agent move. The dominant force axis must be unique. Two agents voting along that axis must occupy opposite positions perpendicular to travel. Every attached tug then translates with the load. This preserves attachment geometry; detached swimmers and the other cargo can block the footprint. Convoys resolve in rotating barge order, so first movement can free or consume another convoy's space.

The two cranks share one hydraulic selector. A crank resets its selected gate to four and sets the other to zero; competing requests use rotating tug priority. Timers count down before towing. Entering any square of a gate lane at x=5 requires a positive updated timer. Swimming tugs can cross a closed sluice; barges cannot. North current initially points east, south west, and signs reverse every eight steps. Entering or leaving x=5 against the current requires net force three rather than two. The observation shows the current that the next transition will use.

A barge freezes at its own dock and all attached tugs are released. Delivery is a full task event, not an attach/crank intermediate event. Each delivery yields +1 to every tug; each non-wait sanitized action costs its actor 1/64. This creates free-riding and wasted-force costs without rewarding an arbitrary cooperation variable.

## Strategies, failures and variations

A pair may finish one cargo while the other pair waits for its favourable current, or detach/reassign a third tug to overpower adverse current. Gate ownership may alternate or prioritize one cargo. Pulling while a partner cranks cannot satisfy a two-vote tow; guarding a clear path by staying out of it may outperform occupying it. Additional tug help costs that tug fuel and may block another route. None of these strategies has been executed or measured.

Blocking, opposed votes, insufficient force, wasted crank timing and timeout can leave either cargo undelivered. No success script or learnability guarantee is asserted. Alternative maps, starting anchors, current schedules or gate lifetimes could alter the needed strategy in separately specified future tasks; this source contains exactly the stated fixed map and rules.

## Declared invariants and observations

There are always four distinct tug positions and exactly two cargo identities. Each attached tug remains at Manhattan distance one from its own cargo; a successful tow translates every member equally. Cargo identities never duplicate, delivered status is monotone and only matching docks pay. Gate timers stay between zero and four and no two gates have positive timers after any crank. All map locations, attachments, delivery flags, timers and current vectors are public. Manifest observation slices enumerate all 142 columns, including every public event flag. State/scene entities are actual footprint positions, not decorative copies of unrelated scalar variables.

## Boundary, invalid actions and terminal behavior

The native interface is GeneratedEnv.reset(key), observe(state), legal_actions(state), and raw_step(key, state, int32[A]). Native learning receives only float32[A,O] observations and bool[A,U] masks. Out-of-range or masked-illegal actions become action 0 without action cost. Joint conflicts can still prevent legal attempted actions from succeeding. All roles remain active until joint completion; there are no individually terminated roles. At joint completion, only no-op is legal. A nonterminal step increments scalar int32 t exactly once. Natural completion takes precedence over the horizon. An already-terminal raw call preserves every state leaf, clock, observation and terminal cause; it returns zero float32 rewards and all-true dones. No raw autoreset occurs.

The agent-keyed public JaxMARL adapter belongs to the separate trusted engineering layer: use agent_0 through agent_3, preserve row/action order and masks, and set the done dictionary __all__ from info.done_all. The raw array contract and frozen learner are unchanged. Adapter execution and compatibility are not claimed here.

State indices, sentinels, counts, stages and temperatures use int32; visibility/termination use bool; observations/rewards use float32. All normalization divisors are powers of two. This boundary representation does not replace a previously specified random distribution. The PRNG key is not copied into state. Reset and transition are deterministic; the key is accepted but unused. Seed does not imply a hidden randomized layout.

The map is public knowledge. Role rendering must honor the same entity visibility masks as native observations and suppress hidden attributes. Only public channels appear in events; private source leaves are not display attributes. Full-state debug views require a separate label. Scene positions, active flags, visibility and attributes bind real state; no presentation-generated transitions or executable markup occur.

## Literal source evidence

- Actual map, anchors, cargo and docks initialize fixed int32 geometry. candidate.py:L15-L44 (exact excerpt SHA-256 443c75c89bcd7765517e644cc47fc39aae60b4a1566cac02b183c5a8f93b7f17).
- Complementary anchored force is required, not a cooperation label. candidate.py:L128-L133 (exact excerpt SHA-256 9ec51ea100586b25dd6b3a200e194f30dc85fa4940d79168026cd623b423770b).
- Shared gate selection changes both channel timers. candidate.py:L100-L108 (exact excerpt SHA-256 df92bd496a69b39738d1ac59d6b4e86aad42a91ed3d6de7e3ff00ba4dbe1f6de).
- Gate/current rules and physical collision checks determine cargo motion. candidate.py:L144-L154 (exact excerpt SHA-256 b29f5b4e02dbff1f7cda955bf9bd741cd1efff5a9c5a2caa641e52d6bdfcf6b6).
- Matching-dock delivery flags detach all members. candidate.py:L155-L157 (exact excerpt SHA-256 2863c931c3db61ab30b88a9446aa2a497ece4f5cc5a77c70d603e87900073ecd).
- Native observation binds public physical fields. candidate.py:L68-L81 (exact excerpt SHA-256 94c1a483d189295cac84dc6ac9f76ad327c40d06c50ba919f1da16a5a49d1898).
- Terminal calls preserve every state leaf and pay zero. candidate.py:L173-L177 (exact excerpt SHA-256 40e10d72edba7e17f3a6fbff623186a6ee8d5496fb988112e317c7f21a33c01d).

## Interface method locations

- moves: candidate.py:L12-L13.
- reset: candidate.py:L15-L44.
- free_targets: candidate.py:L46-L54.
- legal_actions: candidate.py:L56-L66.
- observe: candidate.py:L68-L81.
- raw_step: candidate.py:L83-L177.

## Reference attribution and limits

Original rules and kernel were authored for this assigned design. No upstream implementation code, layouts, assets, sprites or trained parameters were copied. JaxMARL's README supplied the agent-keyed dictionary and parallel-action integration convention. Pgx's README supplied board-state, explicit legal-mask and external-random-key API reference. Craftax's README supplied the general example of executable spatial resources and staged game mechanics in JAX. PureJaxRL is an algorithm implementation reference, not a new environment library. The supplied Craftax_Baselines PPO-RNN header credits PureJaxRL; it was consulted only as baseline/integration reference and was not copied or substituted for the frozen learner. No reference performance claim is evidence about these games. Snapshot identity was checked in full; consulted passages were API/overview material and the first 120 lines of the PPO-RNN reference.

- references\JaxMARL_README.md: [official snapshot origin](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; verified 12438 bytes.
- references\Pgx_README.md: [official snapshot origin](https://github.com/sotetsuk/pgx/blob/main/README.md), SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; verified 17543 bytes.
- references\PureJaxRL_README.md: [official snapshot origin](https://github.com/luchris429/purejaxrl/blob/main/README.md), SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; verified 5023 bytes.
- references\Craftax_Baselines_ppo_rnn.py: [official snapshot origin](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py), SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; verified 18983 bytes.
- references\Craftax_README.md: [official snapshot origin](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; verified 12388 bytes.

## What was and was not checked

Only AST parsing, the supplied whitelist, JSON manifest validation, scene-schema validation, declared-state-key agreement, source byte cap, hashes and script integer-vector structure were checked. The candidate was never imported, compiled, run or numerically simulated. JAX and NumPy were absent from the checker process imports. Reset/step shapes and dtypes, eager/JIT/vmap parity, numerical bounds and conservation under execution, repeated-seed replay, trajectory legality, real paired effects, absorbing execution, rendering, playback/pause/step, public adapter, policy training, solvability and meta-learning remain unexecuted. No acceptance score, success-rate threshold, ranking or final-test information was used.
