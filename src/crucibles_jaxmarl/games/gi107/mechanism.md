# Counterweight Crossing

Status: AUTHORED_NOT_EXECUTED. A=3, O=109, U=9, H=120. This is an original physical-access coordination game.

## World and update order

A 9 by 7 map has a river at x=4, with bridge sites at (4,1) and (4,5). The west load plate is (2,1), east plate (6,5), and shelter (8,3). Three responders start at (2,1),(1,3),(2,5), with agent 0 initially opening the north bridge by standing on the west plate. Two mass-two ballast objects start at (2,2) and (6,4). Four evacuees have actual positions on both banks. Reset and all transitions are deterministic; keys are accepted without random draws.

Each active step resolves movement in role order rotated by the input clock, then cargo operations, self-evacuation, physical plate loads, brace timers and energy, bridge flags, reward and terminal causes. A destination occupied by another active responder is blocked. Earlier movement in the rotated order may vacate a cell for a later agent. River tiles are never traversable; bridge entry uses the old state's open flag. A responder already on a closed bridge can stay or leave onto a bank, preventing a hidden crushing or teleport rule.

An agent weighs one, an evacuee one, and ballast two. Carried mass is counted through its carrier; ground cargo is counted separately, avoiding double-counting. The west plate being heavier opens north; east being heavier opens south; equality closes both. A brace can independently hold either bridge open. New plate mass affects bridge access on the next transition, making release timing matter.

## Roles, objects and commitments

The engineer, agent 0, braces an adjacent bridge for three output states; the other agents brace for one. Each valid brace consumes one energy from a capacity-three pool. Waiting restores one energy. Moving away leaves only the remaining timer, so an engineer can grant a short crossing window and then leave. A person carrying a victim or ballast cannot pick up another object. Action 5 picks up or puts down an evacuee, delivering it only at shelter; action 6 handles ballast. Objects follow actual carrier coordinates. Dropped objects retain their exact location and can be retrieved later. Active responders cannot occupy the same tile, so indirect handoffs require dropping and vacating before a partner retrieves.

Agent 1, the porter, moves every tick carrying ballast. Agent 2, the escort, moves every tick carrying an evacuee. Other loaded agents move only on even input ticks; an unloaded agent always moves. At the shelter, an empty responder may use action 8 to evacuate permanently. Such a responder becomes inactive, ceases obstructing or weighing on plates, and has only the no-op action. There is no automatic self-evacuation on arrival.

## Multi-agent interaction and prospective strategies

A responder who holds a plate delays their own crossing but grants another person a route. A porter can replace that commitment with actual ballast, allowing the holder to leave. The east plate reverses which bridge is available, so delivering ballast or moving a loaded teammate can close one route while opening the other. The engineer can cover a poorly timed release with a temporary brace, consuming a recoverable but finite action resource. Waiting recovers energy while delaying delivery. These causal effects alter physical opportunities and future positions; they are not independent shortest-path tasks given an unchanging map.

One prospective strategy secures the north span with west-bank ballast, then transfers all personnel across. Another sends a bridge engineer and porter to the east plate and uses the south span for late southern evacuees, with bracing during reversal. An opportunistic empty responder can evacuate early for personal credit, reducing the remaining workforce and removing an available holder. None of these strategies has been executed or proven effective.

Every delivered evacuee grants every role +1. Each self-evacuation grants all roles +0.3 and the evacuee responder an additional +0.4. Each nonterminal input transition costs every role 0.005; inactive roles still receive shared team reward until joint completion. Natural termination requires all four civilians delivered and all three responders evacuated, including the possibility that early irreversible evacuation leaves the task impossible and only horizon truncation resolves it. No success is fabricated for such a state. Bounds [-1,6] conservatively contain one-step rewards.

## Observation and presentation

Every transition-relevant quantity is public. The 109-column observation contains own role one-hot, all responder positions/activity/energy, all cargo positions/owners/status, both plate loads, bridge-open flags/timers, clock and delivered/departed totals, and all terrain cells. Fixed plate/bridge/shelter coordinates are part of the public world rules. Scene entities bind the same actual positions and flags. The map value 2 indicates a bridge site; bridge_open alone determines whether entry is admitted. An evacuated agent is inactive in the scene rather than being animated as a still-available worker. Event counters refer to the actual transition in which access changes or cargo moves.

Declared invariants include unique active responder positions, plate loads 0..11, brace timers/energy 0..3, victim status 0..2, object owners -1..2, and one object per carrier. All action ids are legal attempts for live roles; infeasible requests are no-ops, invalid ids become wait. Absorbing terminal handling preserves the natural-versus-horizon cause.

## Prospective variation

Plate sites, ballast masses, bridge locations and brace duration would change who should commit first and when a route can be reversed. They are possible future source variants, not measured transfer results or authorization to generate another batch.

## Literal source-line evidence

- candidate.py:90: entry = (tile != 1). Actual river and bridge configuration gate movement.
- candidate.py:92: occupied = jnp.any. Active responders physically block contested destinations.
- candidate.py:81: fast =. Porter and escort have different advantages for different transported objects.
- candidate.py:136: load = (load +. Ground and carried mass are aggregated at physical plates.
- candidate.py:142: balanced_open =. The relative plate loads reversibly choose north or south access.
- candidate.py:141: timer = jnp.maximum. Temporary brace effects persist and decay over time.
- candidate.py:138: braces = (a == 7). Bracing requires an adjacent active responder with remaining energy.
- candidate.py:128: leave = (a == 8). Self-evacuation removes an empty agent from future participation.
- candidate.py:148: terminated = (delivered == 4). Natural completion requires both evacuee delivery and responder evacuation.
- candidate.py:173: nxt = jax.tree.map. Already-terminal calls preserve state and terminal causes.

## Public reference provenance

- references\JaxMARL_README.md: SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\Pgx_README.md: SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; https://github.com/sotetsuk/pgx/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\PureJaxRL_README.md: SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; https://github.com/luchris429/purejaxrl/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\Craftax_Baselines_ppo_rnn.py: SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py. API/design reference only; no upstream implementation copied.
- references\Craftax_README.md: SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; https://github.com/MichaelTMatthews/Craftax/blob/main/README.md. API/design reference only; no upstream implementation copied.

JaxMARL supplies the parallel agent-keyed API idea; Pgx supplies the distinction between game state, legal masks and terminal flags. Craftax supplies resource-world design context. PureJaxRL and Craftax_Baselines are algorithm-integration references, not substitutes for the frozen learner. No upstream implementation was copied. Source author self-report is gpt-6-astra/max; the coordinator retains dispatch evidence.

## Status and limits

AUTHORED_NOT_EXECUTED. Only stdlib AST/JSON/hashes and the supplied static code, manifest and scene validators were used. Candidate code was never imported or executed; no JAX/NumPy, numerical emulation, model API, network, SSH, training or replay ran. Playthroughs and paired interventions are UNEXECUTED_AUTHOR_HYPOTHESIS. Static acceptance establishes neither numerical correctness nor solvability, interaction quality, learning, meta-learning, transfer, browser acceptance or execution authorization. Runtime R0/M0, adapter integration, actual per-role replay and scientific evaluation remain separate. Integer quantities use int32, boolean flags use bool, observations/rewards float32. The external reset key is not duplicated in state. Already-terminal raw inputs preserve all state, clock and terminal causes, with zero reward; natural completion has priority over horizon truncation.
