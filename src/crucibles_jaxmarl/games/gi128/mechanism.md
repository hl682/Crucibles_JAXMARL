# Crossbar Couriers: Pass Before Expiry

Status: AUTHORED_NOT_EXECUTED. This is train-only initial source attempt 00, not a replay, animation, training result or accepted scientific finding.

The 9 by 7 map has a separator at x=4 with openings at y=1,3,5. Three couriers have unique positions and one packet buffer each. Four permanent packet identities start at four depot locations. Reset draws one independent Bernoulli choice per packet to choose between the two opposite-side depots; raw_step uses no randomness. Initial lifetimes are 36,44,36,44 ticks. Local observations expose a packet only within Manhattan distance two or while carried by the observing role. Courier positions/load are visible within distance three. Map layout, depot indices, route reservations, and packet delivery/expiry radio receipts are public. Packet attributes and entity geometry use the same packet_visible mask. Hidden fields are zero, with separate visible bits. This is an observation-access model, not a cryptographic secrecy claim.

Reservations select one neighboring road cell per courier for three frames including the creation frame. They are public, owner-bound and exclusive: other couriers cannot enter, including during creation. Reservation requests resolve in rotating order and existing live reservations retain their claim. Timers decrease before requests and movement. Moving into an old occupied courier cell is forbidden even if its owner moves away; no swaps or same-tick following are allowed. Competing eligible requests for an empty destination use rotating priority. These conservative rules make queues, detours and yielding matter without iterative collision solving.

Drop and pickup requests resolve before handoffs. A pickup requires the packet at the courier's exact position and an empty buffer; the lowest available packet index wins. A handoff requires a donor choosing Offer and an adjacent empty receiver choosing Accept. Rotating priority chooses between multiple compatible receivers or donors. Packet custody changes once per successful donor request; newly accepting roles cannot themselves Offer in the same joint action. Carried packet positions follow their current carrier. A packet arriving at its destination is delivered before the same step's expiry check. Resolved packet identities remain in the fixed state with status 2 delivered or 3 expired and carrier -1. There is no duplication, respawn, invisible handoff bonus or packet deletion.

Direct end-to-end carriage, a central consenting relay, and a route-reservation convention are proposed alternatives. A courier may pass a packet to free its own slot or retain it for the last-handler bonus; reserving a shortcut may protect a route while delaying a teammate's earlier deadline. Delivery pays everyone 0.25 plus 1 each to sponsor and final handler, which can be the same role. Expiry costs everyone 0.25 and the sponsor another 0.5. The fixed scripts cannot adapt to the private reset draw and make no successful-delivery claim. Failure, deadlock until expiry and poor incentives remain valid outcomes.

The scene omits hidden pickup/handoff flags from its global event channel. Actual custody, load and packet motion remain visible only under the observation masks; a separately labelled debug view may inspect full state. Literal action requests, branch metadata and the same seeded reset draw are frozen for independent replay. Future separately specified changes could vary deadlines, destination distributions, route geometry or sponsor incentives; none is evaluated here.

## Contract and evidence boundaries

A=3, O=83, U=13, H=64. Observations are flat float32 and action indices int32. Every state leaf has a fixed declared shape. Source byte count is 11912 and SHA-256 is c65118d8ecf1e89844561a43cfb4963d4f32076e80d6592c0d43c0df8e652f47. The manifest binds exactly these bytes; the scene binds the manifest bytes.

The raw kernel advances a nonterminal clock once, computes natural completion before truncation, and preserves a terminal input exactly with zero reward. There is no raw autoreset. Illegal indices become Wait; unavailable in-range requests are deterministic no-ops. Every active role has Wait legal. Role disablement is separate from joint done. State bounds and conservation rules are source declarations awaiting numerical R0 checks. Array updates are immutable JAX operations; local dictionary copies have no external mutable state.

The two literal scripts and paired suffixes in playthrough.json remain UNEXECUTED_AUTHOR_HYPOTHESIS. Reset uses fold_in(PRNGKey(seed),0); transition k uses fold_in(PRNGKey(seed),k+1). A paired suffix continues at global index len(prefix). Identical branch indices use identical keys. No original script is replaced by an adaptive action search or a successful-looking trace. Static shape/range checks do not establish mask legality or intended events. Stop the future replay at joint done and report any prior invalid request, unavailable interaction or unexpected outcome.

Trainability here means a supplied fixed-shape pure-JAX kernel, bounded discrete actions and rewards, explicit observations and masks, and a finite episode contract suitable for later integration. It does not mean learning has been observed. New A/O/U/H require matching policy metadata and weights. Source differences, causal hypotheses and a valid scene do not prove non-isomorphism, strategic richness, meta-learning or transfer. The independent executor must retain negative, failed, missing and indeterminate evidence.

The scene is a complete declarative mapping of native entity positions, masks, attributes and events. It contains no synthetic trajectory, decorative state transition, HTML, JavaScript or extra policy information. Actual replay, RGB frames, animation controls, adapter compatibility and learning results remain mandatory later delivery work.

## Literal source evidence

- Private reset destination draw: candidate.py:25; literal source: `right_or_bottom = jax.random.bernoulli(key, 0.5, (4,)).astype(jnp.int32)`.
- Role observation and rendering use one visibility mask: candidate.py:75; literal source: `pv = s["packet_visible"]`.
- Reservations change others' feasible movements: candidate.py:113; literal source: `reserved = jnp.any(jnp.all(proposal[:, None, :] == rp[None, :, :], axis=2) &`.
- Congestion and rotating move arbitration: candidate.py:116; literal source: `occupied = jnp.any(jnp.all(proposal[:, None, :] == s["agent_pos"][None, :, :], axis=2) &`.
- Consent and buffer capacity govern handoff: candidate.py:147; literal source: `accept = (a == 7) & ~loaded & (jnp.arange(3) != i) & (jnp.sum(jnp.abs(pos - pos[i]), axis=1) == 1)`.
- Arrival priority and packet conservation: candidate.py:156; literal source: `delivered = (status == 1) & jnp.all(pp == destination, axis=1)`.
- Sponsor and carrier incentives: candidate.py:176; literal source: `reward = 0.25 * jnp.sum(delivered_f - expired_f) + jnp.sum((own + handled) * delivered_f[None, :], axis=1) - 0.5 * jnp.sum(own * expired_f[None, :], axis=1)`.
- reset interface: candidate.py:24-60.
- observe interface: candidate.py:62-89.
- legal_actions interface: candidate.py:91-92.
- raw_step interface: candidate.py:94-180.

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
