# Shield Relay: Escort the Supply Line

Status: AUTHORED_NOT_EXECUTED. This is train-only initial source attempt 00, not a replay, animation, training result or accepted scientific finding.

Two supply carts move along y=2 and y=4 toward x=10 in an 11 by 7 world. Three escorts have health 3 and charge 6. Movement and co-location are allowed independently for escorts; their interactions occur through coverage, cargo escorting, shared targets and energy. Four bounded projectile slots carry actual position, velocity and flight TTL. An every-other-tick schedule selects an empty slot; launch x cycles through 3,5,7,9 and launch side alternates. A blocked occupied launch slot is skipped. Shots move one cell per later tick and have a seven-tick lifetime. A newly spawned shot is first eligible for interception on the following tick.

Actions 5--8 set facing and guard for one tick. The actual intercept set is three cells one cell ahead of the escort, perpendicular to facing; outside-world points are inactive. A projectile's previous and proposed positions form the swept one-cell path. Interception requires that path to touch the arc and projectile velocity to oppose the guard direction. A shot is resolved once, by ascending slot: first the lowest indexed charged live matching guard, otherwise the lowest indexed live cart at the next position, otherwise the lowest indexed live escort. Each interception consumes one charge. No charge is consumed simply for requesting guard. A drained shield disappears in the returned frame, while the interception event retains what happened.

Rest within Manhattan distance one of a station restores two charge up to six. A transfer donor selects an adjacent active role of least charge, breaking ties by role index; at most two units move, bounded by donor stock and receiver capacity. Donors resolve in rotating order, so same-tick received charge may be forwarded by a later donor. Transfers conserve charge before recharge, repairs and hits. Repair spends two charge for one missing cart health, without reward for repairing. Escort moves each adjacent active cart at most once. Distances are tested before cart motion. Movement, transfers, repairs and cart progress precede projectile movement; delivery is checked after damage so a cart hit on arrival must survive to count.

Proposed alternatives include a moving two-sided shield formation, staggered guards supplied by a mobile donor, or a faster escort-and-repair route. Helping consumes a turn that could move or drive a cart. Shield placement can protect another escort as well as the shared supply. Formation overlap changes which role pays charge or takes damage. Dense progress and interception terms support a learning interface but may create undesirable short-horizon incentives; reward learning and strategy quality need actual evidence. Each new cart arrival gives everyone 1, each cart advance 0.125, each cart hit costs everyone 0.25, an interceptor earns 0.125 per block, and each role loses 0.25 per own hit. No shaping is added by a renderer.

Disabled escorts have only action 0 legal and take no active game actions; they remain done=false until joint termination. The script author has not tested when any role disables. Replay must check each original requested action against the actual mask and preserve any failure; kernel invalid-request sanitation is not a legal-witness certificate. Real arc-cell, health, charge, shot and cart state bind the scene. Future separately authorized changes to shot phase, station spacing or cart paths could favor different formations; they are not implemented or evaluated here.

## Contract and evidence boundaries

A=3, O=145, U=12, H=80. Observations are flat float32 and action indices int32. Every state leaf has a fixed declared shape. Source byte count is 11758 and SHA-256 is 01edcd481554df610ae6898ea2319f7235ebc6b4144ffa7298b7b013a7aab205. The manifest binds exactly these bytes; the scene binds the manifest bytes.

The raw kernel advances a nonterminal clock once, computes natural completion before truncation, and preserves a terminal input exactly with zero reward. There is no raw autoreset. Illegal indices become Wait; unavailable in-range requests are deterministic no-ops. Every active role has Wait legal. Role disablement is separate from joint done. State bounds and conservation rules are source declarations awaiting numerical R0 checks. Array updates are immutable JAX operations; local dictionary copies have no external mutable state.

The two literal scripts and paired suffixes in playthrough.json remain UNEXECUTED_AUTHOR_HYPOTHESIS. Reset uses fold_in(PRNGKey(seed),0); transition k uses fold_in(PRNGKey(seed),k+1). A paired suffix continues at global index len(prefix). Identical branch indices use identical keys. No original script is replaced by an adaptive action search or a successful-looking trace. Static shape/range checks do not establish mask legality or intended events. Stop the future replay at joint done and report any prior invalid request, unavailable interaction or unexpected outcome.

Trainability here means a supplied fixed-shape pure-JAX kernel, bounded discrete actions and rewards, explicit observations and masks, and a finite episode contract suitable for later integration. It does not mean learning has been observed. New A/O/U/H require matching policy metadata and weights. Source differences, causal hypotheses and a valid scene do not prove non-isomorphism, strategic richness, meta-learning or transfer. The independent executor must retain negative, failed, missing and indeterminate evidence.

The scene is a complete declarative mapping of native entity positions, masks, attributes and events. It contains no synthetic trajectory, decorative state transition, HTML, JavaScript or extra policy information. Actual replay, RGB frames, animation controls, adapter compatibility and learning results remain mandatory later delivery work.

## Literal source evidence

- Native three-cell directional shield geometry: candidate.py:18; literal source: `arc = s["agent_pos"][:, None, :] + facing[:, None, :] + jnp.array([-1, 0, 1], dtype=jnp.int32)[None, :, None] * perpendicular[:, None, :]`.
- Conserved transfers change another role's charge: candidate.py:104; literal source: `amount = jnp.where((a[i] == 9) & jnp.any(nearby), jnp.minimum(jnp.minimum(2, energy[i]), 6 - energy[target]), 0)`.
- Escort moves shared supply objects: candidate.py:118; literal source: `pushing = jnp.any((jnp.sum(jnp.abs(pos[:, None, :] - cp[None, :, :]), axis=2) <= 1) &`.
- Interception priority and directional matching: candidate.py:138; literal source: `can_block = arc_hit & aimed & guarding & (energy > 0) & (health > 0)`.
- Bounded telegraphed projectile schedule: candidate.py:156; literal source: `spawn = (s["t"] % 2 == 0) & ~ha[h]`.
- Damage and arrival rewards: candidate.py:177; literal source: `common = jnp.sum(arrival.astype(jnp.float32)) + 0.125 * jnp.sum(pushing.astype(jnp.float32)) - 0.25 * jnp.sum(cart_hit.astype(jnp.float32))`.
- reset interface: candidate.py:30-69.
- observe interface: candidate.py:71-82.
- legal_actions interface: candidate.py:84-85.
- raw_step interface: candidate.py:87-182.

## Reference attribution

- JaxMARL: agent-keyed public adapter concepts; author kernel remains the assigned flat functional interface.
- Pgx: explicit legal masks and terminal state concepts from its board-game API; no board implementation or benchmark imported.
- Craftax: spatial entities, persistent resources, game-like state and native rendering concepts; no Craftax game code copied.
- PureJaxRL and Craftax PPO-RNN baseline: functional vectorized rollout and recurrent-done integration concepts only. They are learner implementations, not generated game libraries and do not replace the frozen learner.

No upstream source code was copied. Only the supplied official snapshot excerpts were read; no network access, prior candidate source, project outcome, heldout implementation or final test was used.

- references\JaxMARL_README.md; SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md
- references\Pgx_README.md; SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; https://github.com/sotetsuk/pgx/blob/main/README.md
- references\PureJaxRL_README.md; SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; https://github.com/luchris429/purejaxrl/blob/main/README.md
- references\Craftax_Baselines_ppo_rnn.py; SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py
- references\Craftax_README.md; SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; https://github.com/MichaelTMatthews/Craftax/blob/main/README.md
