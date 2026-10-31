# Phase-Lock Relay — authored attempt_00

Status: AUTHORED_NOT_EXECUTED. A=3, O=148, U=8, H=96. The package is an original prospective source, not a real replay or demonstrated solution.

## Map, cargo and local controls

The 11x7 map contains partition walls at x=4 and x=7. Each wall has actual gate cells at y=1 and y=5. Gate indices are (4,1), (4,5), (7,1), (7,5), with their respective stations at (2,2), (2,4), (6,2), (6,4). These station cells are reachable floor cells off the courier lanes.

Cargo 0 starts at (1,1) and requires dock (9,1); cargo 1 starts at (1,5) and requires dock (9,5). The three agents start at (1,1), (2,2), (6,2). The named courier/operator roles are suggested starting roles, not different abilities: any agent can carry, hold a station, shift a gate or switch jobs.

Actions 0..4 are Wait and cardinal moves. Action 5 picks up a grounded cargo at the same cell, or drops the currently carried cargo at the actual player position. Actions 6 and 7 respectively Hold or Shift the gate associated with the station the player currently occupies. Cargo interactions require no fictional remote reach.

## Timing and order

Each gate has a persistent integer offset, initially [0,3,2,5]. A Shift adds three modulo six. The phase used for this transition is (t+1+offset) modulo six. Natural entry is open at phases zero and one.

A Hold sets the forced-open override to three before movement; absent a new Hold, the old override decreases by one. Consequently the command affects the current movement and the next two calls. Hold also sets the commanding agent lock_timer to two. That agent can select only Wait on its next two calls; the timer then reaches zero and it may act again. Shift sets a one-call commitment. Neither command provides movement in the same call.

Commands and phase updates are resolved first, before courier movement. A closed-gate move is an allowed attempt whose displacement fails, because another role might open that gate in the same joint action. Legal masks do not guess teammates' pending commands. Walls and world boundaries are masked normally.

Movement uses resolved gate openness plus conservative player collision: another player's pre-move cell and every contested destination are blocked, even if a player intended to leave. A gate can close under an occupant without crushing it. The occupant can leave toward any otherwise valid cell; a stationary occupant simply remains on the closed cell. Closing does not invent a push, damage or teleport.

Cargo then follows its old carrier, processes Pick/Drop, and automatically delivers if still carried on the corresponding dock. Dropping at the dock prevents that immediate delivery until picked up again. Picking considers each cargo in index order, and recomputes who is already carrying, so one action cannot acquire two. Delivery removes the cargo from active interaction and clears ownership.

## Multi-agent mechanism and alternatives

An operator changes whether another agent and its carried object enter the next chamber at the current time. That assistance commits the operator to remaining stationary for future calls, preventing an immediate trip to another station, pickup or courier role. A Shift also changes future natural windows after the operator leaves. These are persistent changes in a partner's feasible schedule.

A team can wait for periodic openings, arrange two stations for a synchronized courier, use a short phase shift instead of a longer hold, or move between upper and lower corridors. Courier/operator exchange is possible via a physical drop and later pickup. The best timing depends on where the other players are and whether they intend to commit. None of these strategies is asserted optimal or learned.

The upper and lower literal scripts target one real delivery each and do not claim both cargo completed. Two prospective branch pairs isolate Hold versus Wait and Shift versus Wait at the first gate. The second also describes a real closure-under-occupant situation whose exit semantics require subsequent execution checking.

## Reward and termination

Each new delivery gives all three roles +1. Every nonterminal call subtracts 0.001 from each role, plus 0.005 for a role that issued Hold or Shift. Bounds [-0.007,2.0] cover all per-step returns. Both delivered flags naturally terminate; H=96 otherwise truncates, with natural completion priority.

Commitment stays in 0..2, override in 0..3, offset and phase in 0..5. Player positions stay unique and on floor/gate cells. Each of two cargo items is either on the floor, held by exactly one role, or delivered at its unique matching dock. No role can hold two and no delivered item reactivates. These are source-derived invariants pending real checks.

## Visibility and animation

All native state relevant to gameplay is public: the complete map, agent/cargo positions, ownership, commitment, gate phase/override/open state, stations and dock completion. Own role identity is supplied separately. The current offset is recoverable from public t and phase; it is not a hidden-information variable.

The scene binds gates to true gate_pos and gate_open, operators to true lock_timer, cargo to true carrier-following positions and dock status, and stations to their actual controlled gate indices. Changed and commanded are per-step events. Delivered and commitment are levels. A renderer may show the current phase and override values but must not invent future successful crossings, smooth through blocked cells, or reveal information beyond the public/native mapping. Real animation is still unproduced.

## Possible future variations

Other periods, phase-offset patterns, station placements, gate layouts and commitment costs would change scheduling and role allocation. Such changes would require separate identified sources and tests; this source implements exactly the stated fixed configuration and makes no transfer claim.

## Literal source evidence

- Operation commitment masks movement: candidate.py:68 contains exactly `free = (state["lock_timer"] == 0) & ~state["episode_done"]`.
- Persistent phase intervention: candidate.py:100 contains exactly `offset = jnp.mod(state["gate_offset"] + 3 * shift.astype(jnp.int32), 6).astype(jnp.int32)`.
- Bounded hold override: candidate.py:101 contains exactly `override = jnp.where(hold, 3, jnp.maximum(state["gate_override"] - 1, 0)).astype(jnp.int32)`.
- Post-command gate schedule: candidate.py:103 contains exactly `phase = jnp.mod(t + offset, 6).astype(jnp.int32)`.
- Real stationary opportunity cost: candidate.py:106 contains exactly `lock = jnp.where(actions == 6, 2, jnp.where(actions == 7, 1, lock)).astype(jnp.int32)`.
- Gate controls actual destination: candidate.py:112 contains exactly `gate_block = jnp.any(jnp.all(proposed[:,None,:] == state["gate_pos"][None,:,:], axis=2) & ~gate_open[None,:], axis=1)`.
- Cargo physically follows carrier: candidate.py:123 contains exactly `current_pos = jnp.where(prior_owner >= 0, pos[safe_owner], cargo_pos[c])`.
- Matching spatial delivery: candidate.py:130 contains exactly `arriving = (current_owner >= 0) & jnp.all(current_pos == state["dock_pos"][c]) & ~delivered[c]`.
- Team reward and command cost: candidate.py:161 contains exactly `reward = reward - ((actions == 6) | (actions == 7)).astype(jnp.float32) * 0.005 - 0.001`.
- Absorbing raw terminal state: candidate.py:166 contains exactly `ns = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, proposed)`.

The line references above are static source evidence, not executed transition evidence. Method spans are reset=12-54, legal_actions=56-69, observe=71-91, advance=93-162, raw_step=164-170.

## Interface, provenance and remaining acceptance

The native boundary is reset(key) -> (observation,state), observe(state), legal_actions(state), and raw_step(key,state,actions) -> (observation,state,reward,dones,info). All state leaves are fixed-shape int32/bool arrays; observations and rewards are float32. Signed sentinels use int32. The uint32 reset key belongs to external provenance and is not copied into state. Every valid nonterminal call increments t once. Natural termination has priority over a simultaneous horizon. Already-terminal calls preserve every state leaf and t, return the preserved observation and cause flags, zero reward, and all dones true. No raw auto-reset occurs. Invalid/out-of-range/masked actions deterministically become action 0. Action 0 remains legal for every role, including inactive or immobilized roles and terminal states.

A later public JaxMARL adapter must map the listed agent names to ordered action/observation/reward/done dictionaries and expose done["__all__"]. It must preserve these observations, masks, rewards, raw terminal states, and inactive-role semantics. The author did not implement or validate that separate adapter, replace the frozen learner, or reuse old policy-bank weights. The declared A/O/U/H require matching integration and policy metadata.

Only data schemas and source AST were inspected. Static checks do not establish JIT/vmap behavior, numerical parity, physical invariants under all actions, script legality, solvability, strategic richness, learned adaptation, transfer, or scientific acceptance. No real trajectories, animations, RGB frames, benchmark scores, training results or final-test data were read or generated. The scene is declarative binding data for a later real-state replay. Each source is attempt_00 and is frozen at its recorded SHA-256; a later repair requires separate actual feedback and dispatch.

Community references were supplied as hash-bound read-only files. JaxMARL informs the future parallel agent-keyed interface; Pgx informs explicit legal masks and terminated/truncated separation; Craftax demonstrates persistent spatial game state and explicit reset handling. PureJaxRL and Craftax_Baselines_ppo_rnn.py are algorithm/integration references, not game libraries or replacements for the experimental learner. No upstream game rules or code were copied into this kernel, and no upstream benchmark implementation was read. The original rules here are attributable to GPT-6-Astra/max.

- references\JaxMARL_README.md (https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), SHA-256 `6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15`.
- references\Pgx_README.md (https://github.com/sotetsuk/pgx/blob/main/README.md), SHA-256 `153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df`.
- references\PureJaxRL_README.md (https://github.com/luchris429/purejaxrl/blob/main/README.md), SHA-256 `d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0`.
- references\Craftax_Baselines_ppo_rnn.py (https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py), SHA-256 `5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011`.
- references\Craftax_README.md (https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), SHA-256 `7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff`.
