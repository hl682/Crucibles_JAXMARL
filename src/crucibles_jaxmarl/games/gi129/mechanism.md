# Shepherd Circles: Two Gates, One Flock

Status: AUTHORED_NOT_EXECUTED. This is train-only initial source attempt 00, not a replay, animation, training result or accepted scientific finding.

Six sheep occupy an 11 by 9 pasture. A fence at x=8 has gates at y=2 and y=6. The upper pen comprises x>=9 and y=1..3; the lower pen comprises x>=9 and y=5..7. Three sheep target each pen. Three shepherds have energy four. A movement request creates strength-one repulsion within Manhattan distance one, even if the movement itself is blocked. Press spends one energy for strength-two repulsion within distance three. Rest restores one energy and produces no pressure; Stay quiet also suppresses pressure but does not recharge.

A directional call spends one energy and places its role's beacon two clipped grid cells away. Calls attract within distance three for two frames, can act across the fence, and are replaced rather than stacked by a new call from the same role. The bounded cell vector is the sum of signed repulsion from all shepherds and signed attraction to all active calls. A 99-cell native entity field stores exactly those vectors; it is the same field indexed by sheep motion, not an illustrative overlay. The full field is present in the policy observation. A sheep with nonzero net drive also receives the sign of its nearby flock-center displacement within distance two. Zero net drive gives no cohesion-only autonomous motion. The dominant force axis determines a one-cell intent; equal magnitudes alternate by sheep index and clock. Force cancellation and direction changes are real combined-agent effects.

Shepherds cannot enter old shepherd or active sheep positions. Sheep cannot enter old active sheep positions or new shepherd positions. Eligible same-target moves use rotating priority, and swaps/follow-through moves are rejected. These rules can cause crowding and require re-positioning or quiet intervals. Sheep use freshly pulled gates while shepherd movement uses the previously visible gate state. Pulling within one cell costs one energy and refreshes a gate to three; multiple simultaneous pulls consume each puller's energy but do not add duration. Closing never teleports or kills an occupant; entry permission is tested only at a proposed cell.

Upper and lower specialists receive a 0.25 bonus for correct sheep of their target class; every role receives 1 for any correct arrival and loses 0.5 for a wrong arrival. Any pen admission resolves that sheep, so incorrect admissions remain explicit and can end the episode. Gatekeeping, synchronized pressure, attraction-dominant arcs and intentionally quiet splitting are proposed strategy families. A gatekeeper spends scarce turns and energy creating opportunities for the others; excessive parallel pressure can crowd the same gate or cancel a useful call. No script is asserted to complete either pen.

The scene binds native shepherd positions, energy, exact call beacons, timers, gate permissions, flock intents and admissions, plus the causal pressure field. Pen flags are reference centers for the regions described above. Future separately specified variations could alter target labels, gate duration, call radius or pen geometry and plausibly change coordination needs; those variations and all learning or transfer outcomes remain untested.

## Contract and evidence boundaries

A=3, O=414, U=12, H=96. Observations are flat float32 and action indices int32. Every state leaf has a fixed declared shape. Source byte count is 11787 and SHA-256 is fb02a8398d479cab6bb8e88e8bba649a6634f2f3b466610071de49e078e2ecc3. The manifest binds exactly these bytes; the scene binds the manifest bytes.

The raw kernel advances a nonterminal clock once, computes natural completion before truncation, and preserves a terminal input exactly with zero reward. There is no raw autoreset. Illegal indices become Wait; unavailable in-range requests are deterministic no-ops. Every active role has Wait legal. Role disablement is separate from joint done. State bounds and conservation rules are source declarations awaiting numerical R0 checks. Array updates are immutable JAX operations; local dictionary copies have no external mutable state.

The two literal scripts and paired suffixes in playthrough.json remain UNEXECUTED_AUTHOR_HYPOTHESIS. Reset uses fold_in(PRNGKey(seed),0); transition k uses fold_in(PRNGKey(seed),k+1). A paired suffix continues at global index len(prefix). Identical branch indices use identical keys. No original script is replaced by an adaptive action search or a successful-looking trace. Static shape/range checks do not establish mask legality or intended events. Stop the future replay at joint done and report any prior invalid request, unavailable interaction or unexpected outcome.

Trainability here means a supplied fixed-shape pure-JAX kernel, bounded discrete actions and rewards, explicit observations and masks, and a finite episode contract suitable for later integration. It does not mean learning has been observed. New A/O/U/H require matching policy metadata and weights. Source differences, causal hypotheses and a valid scene do not prove non-isomorphism, strategic richness, meta-learning or transfer. The independent executor must retain negative, failed, missing and indeterminate evidence.

The scene is a complete declarative mapping of native entity positions, masks, attributes and events. It contains no synthetic trajectory, decorative state transition, HTML, JavaScript or extra policy information. Actual replay, RGB frames, animation controls, adapter compatibility and learning results remain mandatory later delivery work.

## Literal source evidence

- Energy constrains calls, strong pressure and gate opening: candidate.py:116; literal source: `energy = energy - (calling | pressing | pulling).astype(jnp.int32)`.
- Shared causal field is also the native visual field: candidate.py:132; literal source: `field = (repulsion + attraction).astype(jnp.int32)`.
- Flock cohesion and pressure determine real spatial motion: candidate.py:139; literal source: `force = drive + jnp.where(jnp.any(drive != 0, axis=1)[:, None], cohesion, 0)`.
- Gate and occupancy constraints alter teammates' opportunities: candidate.py:145; literal source: `free = self.traversable(proposed_flock, s, gt)`.
- Wrong pen admissions remain distinct evidence: candidate.py:160; literal source: `wrong = arrival & ~correct_arrival`.
- Team and specialist rewards: candidate.py:178; literal source: `reward = jnp.full(3, common, dtype=jnp.float32) + 0.25 * jnp.sum(specialist.astype(jnp.float32) * good[None, :], axis=1)`.
- reset interface: candidate.py:25-70.
- observe interface: candidate.py:72-81.
- legal_actions interface: candidate.py:83-84.
- raw_step interface: candidate.py:94-182.

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
