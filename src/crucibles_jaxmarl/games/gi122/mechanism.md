# Tether Turn — three ports and a shared bulky load

Status: AUTHORED_NOT_EXECUTED. This initial train-only source was written for game-v3-00122, ordinal 122, attempt 00. No candidate import, compilation, transition, rollout, JIT, vmap, numerical parity check, rendering, training or test-set use occurred. Source SHA256: 31bc46089b1cd28acd11e0749f7a2d9f2f2460bf9d7fb1023255ce684d3af220.

## Boundary and lifecycle

A=3, O=171, U=9, H=96. All state leaves have a fixed declared key, shape and int32/bool dtype; observations and rewards are float32. Reset consumes only its supplied key; no external state, time, cache, file or network is part of the kernel. The scalar t starts at zero. A nonterminal call advances it once. Natural termination takes precedence over the horizon. An already-terminal call preserves every leaf and observation, preserves its terminal cause and clock, and returns zero rewards and all dones true. There is no raw auto-reset. The final tree operation preserves each reset leaf's dtype explicitly.

Every in-range action is a legal command attempt while the episode is live. A blocked movement, absent resource, wrong recipe, unavailable object or cooldown causes the specified deterministic wait/no-effect outcome. Out-of-range integers map to action 0. At joint termination only action 0 is legal. No role is removed independently; completed contributors can still help until joint completion. The role adapter must preserve these masks and semantics.

- reset: candidate.py:42
- observe: candidate.py:83
- legal_actions: candidate.py:80
- raw_step: candidate.py:102

## Actual mechanism and timing

The world is a 13 by 11 grid, with interior coordinates x=1..11 and y=1..9. A vertical partition at x=6 has a one-row north opening at y=2 and a wider opening at y=7..9. The actual load occupies center minus axis, center, and center plus axis. Orientation has four values; although opposite orientations share a footprint, they reverse port assignments. Port 0 is the positive endpoint, port 1 the negative endpoint, and port 2 the perpendicular flank. Geometry is implemented at candidate.py:14; the wall layout is at candidate.py:46.

A toggle detaches an attached hauler or attaches a detached hauler only within distance two of its own current port. A hauler cannot choose another role's port. Each attached cardinal command contributes its unit direction to the shared force; axis ties choose x through the first argmax. A component magnitude of at least two proposes one load-cell displacement. Two equal-sign turn votes propose one quarter-turn, with turning taking priority over translation. An attached brace vetoes both. These are commands transmitted through a tether, so a blocked individual body can still exert traction; the resulting load/body configuration must satisfy all collision and rope constraints. Causal aggregation and priorities are at candidate.py:115, candidate.py:119 and candidate.py:121.

First the kernel tests the proposed load footprint and resolves each agent's cardinal move against map, proposed footprint, old occupied cells and contested targets. No swaps or follow-through into another agent's just-vacated cell occur. All attached ropes must then be at most two Manhattan units and the final load must not cover an agent. A rejected collective proposal retains the previous load pose and resolves safe individual movements against the previous footprint and anchors. That fallback lets partners reposition without silently moving the payload; see candidate.py:126, candidate.py:129 and candidate.py:130.

Wear is persistent. Two or more opposing attached pulls with no sufficient force, or a rejected collective move/turn, add one wear. Two attached brace commands remove one wear, clipped to 0..6. Brace coordination therefore trades time for risk reduction and may prevent another partner's desired translation. The reward shared by all three roles is 0.01 times the reduction in Manhattan distance to the goal, minus 0.02 for a jam, plus one for success and minus one for failure. Reward shaping telescopes with goal distance; no claimed learning gain follows. Wear and rewards are at candidate.py:138, candidate.py:140 and candidate.py:154.

The side port can obstruct the narrow north passage even when the load itself fits. One plausible plan is to detach and relocate the stabilizer while the endpoint haulers use the horizontal passage. Another is to travel through the wider south route with a different orientation and retained third tether. Neither plan has been numerically attempted. A partner may need to wait, brace, release or reposition instead of copying the leader's movement. A rotation without enough space or rope slack can be harmful; turning is an actual constrained configuration change, not a sprite rotation.

The scene binds both the real three-cell footprint and the three physical anchor coordinates. Active tether markers and the haulers' attachment/length attributes describe the actual connections; they are not invented line animation. A future renderer may connect a hauler to its assigned anchor using these exact endpoints. The current declarative glyph scheme does not itself prove that such replay animation has been reviewed.

Declared logical invariants additionally include: each attached rope length <=2, no agent/load overlap, three collinear adjacent load cells determined by orientation, unchanged map, and no external reset. Future R0 must check these against actual trajectories. Changing opening geometry, port assignment, allowed rope length or the brace rule could change useful coordination; such variants are proposed scientific factors, not implemented or tested generalization results.

## Observation contract

- Columns 0 through 5 (public): All agent xy coordinates, agent order then xy; divided by 12
- Columns 6 through 8 (public): Per-agent tether attachment flags
- Columns 9 through 11 (public): Per-agent Manhattan tether length, divided by 2; zero when detached
- Columns 12 through 13 (public): Shared load center xy divided by 12
- Columns 14 through 17 (public): Load orientation one-hot: east, south, west, north
- Columns 18 through 18 (public): Load wear divided by 6
- Columns 19 through 20 (public): Goal center xy divided by 12
- Columns 21 through 23 (self): Own role identity one-hot
- Columns 24 through 166 (public): Public 11 by 13 static tile map flattened in yx order
- Columns 167 through 167 (public): Clock divided by 96
- Columns 168 through 170 (public): Latest accepted load translation, accepted turn, and jam flags

The returned observation is the policy information boundary. Entity visibility and declared attributes refer to the same actual state and access rules. The complete static tile layout is public; a debug/full-state display must be labelled separately from a role view. Undeclared internal leaves are not an invitation to expose private values in a role renderer.

## Fixed author scripts and paired interventions

playthrough.json contains fixed joint integer commands, an explicit seed and UNEXECUTED_AUTHOR_HYPOTHESIS status for each script. Stop at the first done. These scripts were constructed from the written rules without local execution, simulation, numerical action search or feedback from other candidates. Prefix/branch comparisons state predicted causal differences only; there is no observed trace, successful episode, solvability certificate, ranking or learned policy.

## Community provenance and delivery boundary

Only this batch's hash-verified design/request, 20 listed inputs and copied public references were used. JaxMARL's README supplies multi-agent integration context, including agent-keyed public APIs; Pgx supplies JAX-native game-state and legal-action-mask design context. Craftax supplies an example of spatial compositional game mechanics. PureJaxRL and the Craftax PPO RNN baseline are algorithm/integration references, not a game library, copied environment, or replacement learner. No upstream source lines, layouts or benchmark rules were copied into this original kernel. Exact copied-input hashes and original official URLs are retained in generation_record.json.

The coordinator owns the separate public JaxMARL adapter, sealed runtime validation, actual state replay, RGB/animated presentation, playback controls and community review. This authored kernel and declarative scene do not establish those acceptance outcomes. Structural/AST screening is separate from numerical validation. Passing future interaction or interface checks would not by itself establish strategic richness, reward learning, meta-learning, non-isomorphism or transfer. No new scientific performance gate is added.
