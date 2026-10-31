# Bridgewrights: Hold the Span

Status: AUTHORED_NOT_EXECUTED. This is train-only initial source attempt 00, not a replay, animation, training result or accepted scientific finding.

The river occupies x=4 on a 9 by 7 grid. Three workers start on opposite banks with personal cargo; three beams and four support pins are the entire material stock. A beam has status 0 loose, 1 carried, 2 placed or 3 broken. Each worker may carry at most one beam and one pin in addition to personal cargo. Picking uses Manhattan distance at most one and the lowest material index. For simultaneous material requests, role order rotates with t. Place chooses a one-cell offset toward the river for a horizontal beam, or toward the middle row for a vertical beam; both endpoints must be in bounds and no other placed beam may have the same center. The three-cell geometry, not a cosmetic marker, defines traversable river coverage.

Pins attach to unpinned endpoints on dry ground within one cell of the installing worker; endpoints are indexed negative-axis then positive-axis. Installed pins never re-enter the loose inventory and stay attached to broken beams. Holding is evaluated from exact endpoint occupancy before movement. Actions 8 and 9 are deliberately equivalent Brace/Hold requests; action 12 is a cargo-preserving Wait alias. The latest support flags record that step's support calculation. After a fall, an old held-end flag does not promise support on the next step; the kernel recalculates it from current positions and actions. Only pins or current holders support a beam. Dry bank cells permit co-location. A bridge supported temporarily has river-load capacity one; a fully pinned bridge has capacity two. Every covered bridge bears the full occupant count; overlapping coverage is not extra load sharing. Entry onto river cells without currently supported coverage is blocked. Releasing an occupied span or overloading a supported one breaks it, and any worker lacking safe river coverage returns to that worker's own home. Carried materials follow their owner home. Cargo already delivered stays delivered.

A temporary crossing exchanges two workers' action turns for a third's travel; permanent pins exchange retrieval and installation time for later independent crossing. Building one shared route versus separating upper and lower routes, using one pin plus a helper versus spending two pins, and deciding whose cargo crosses first are distinct proposed strategies. A vertical placement changes actual endpoint geometry and may be unhelpful on this narrow river; the rules do not award progress for placement alone. Own delivery gives 2 plus a shared 0.25 per new delivery; falling costs the affected role 1. Helpers can therefore face a real opportunity cost. Broken-material scarcity and poorly timed release may lead to negative or failed attempts. No policy advantage, optimal sequence or solvability proof is claimed.

Transition order is resource pickup/placement/installation, support calculation, joint movement, load/collapse, home recovery, delivery, and terminal selection. The span-cell and endpoint entity groups are derived by the same geometry method used for crossing and support; they provide real beam orientation and held/pinned ends to native animation. Positions for inactive derived cells are clipped only for safe presentation bounds. Future separately specified variations could alter river geometry, pin supply, cargo objectives or load capacities; none is implemented or evaluated here.

## Contract and evidence boundaries

A=3, O=96, U=13, H=96. Observations are flat float32 and action indices int32. Every state leaf has a fixed declared shape. Source byte count is 12393 and SHA-256 is d08254385748360daaaca50e0a71fcb7a081663193b4da06d2835f53e5fc5269. The manifest binds exactly these bytes; the scene binds the manifest bytes.

The raw kernel advances a nonterminal clock once, computes natural completion before truncation, and preserves a terminal input exactly with zero reward. There is no raw autoreset. Illegal indices become Wait; unavailable in-range requests are deterministic no-ops. Every active role has Wait legal. Role disablement is separate from joint done. State bounds and conservation rules are source declarations awaiting numerical R0 checks. Array updates are immutable JAX operations; local dictionary copies have no external mutable state.

The two literal scripts and paired suffixes in playthrough.json remain UNEXECUTED_AUTHOR_HYPOTHESIS. Reset uses fold_in(PRNGKey(seed),0); transition k uses fold_in(PRNGKey(seed),k+1). A paired suffix continues at global index len(prefix). Identical branch indices use identical keys. No original script is replaced by an adaptive action search or a successful-looking trace. Static shape/range checks do not establish mask legality or intended events. Stop the future replay at joint done and report any prior invalid request, unavailable interaction or unexpected outcome.

Trainability here means a supplied fixed-shape pure-JAX kernel, bounded discrete actions and rewards, explicit observations and masks, and a finite episode contract suitable for later integration. It does not mean learning has been observed. New A/O/U/H require matching policy metadata and weights. Source differences, causal hypotheses and a valid scene do not prove non-isomorphism, strategic richness, meta-learning or transfer. The independent executor must retain negative, failed, missing and indeterminate evidence.

The scene is a complete declarative mapping of native entity positions, masks, attributes and events. It contains no synthetic trajectory, decorative state transition, HTML, JavaScript or extra policy information. Actual replay, RGB frames, animation controls, adapter compatibility and learning results remain mandatory later delivery work.

## Literal source evidence

- Resource custody and rotating arbitration: candidate.py:109; literal source: `for k in range(3):`.
- Placement uses spatial endpoints and orientation: candidate.py:122; literal source: `center = s["agent_pos"][i] + shift * jnp.array([1 - ba[b], ba[b]], dtype=jnp.int32)`.
- Permanent and temporary support alter crossing: candidate.py:161; literal source: `supported = (bs == 2) & jnp.all(pinned | held, axis=1)`.
- Overloading and release can harm another worker: candidate.py:170; literal source: `collapse = (bs == 2) & (load > capacity) & (load > 0)`.
- Unsafe occupants return home while material custody persists: candidate.py:175; literal source: `pos = jnp.where(fall[:, None], s["home_pos"], pos)`.
- Personal and shared delivery payoffs: candidate.py:193; literal source: `reward = 2.0 * delivered.astype(jnp.float32) + 0.25 * jnp.sum(delivered.astype(jnp.float32)) - fall.astype(jnp.float32)`.
- Native beam cells and held endpoints: candidate.py:14; literal source: `ends, cells = self.geometry(s["beam_pos"], s["beam_axis"])`.
- reset interface: candidate.py:30-79.
- observe interface: candidate.py:81-90.
- legal_actions interface: candidate.py:92-93.
- raw_step interface: candidate.py:101-197.

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
