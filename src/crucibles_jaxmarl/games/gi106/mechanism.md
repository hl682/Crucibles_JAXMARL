# Hose Relay: Firebreak Rescue

Status: AUTHORED_NOT_EXECUTED. A=3, O=242, U=11, H=96. All rules and source are original; no public implementation was copied.

## Rules and causal structure

The 9 by 7 brush world contains six physical evacuees, two initially burning cells, a finite water pump at (1,3), and shelter at (0,3). Heat blocks entry once intensity reaches two. Collocation is intentional: shared tiles enable pickup and infrastructure work. Reset is deterministic; the supplied keys are accepted but no random draws are needed. Each nonterminal raw step sanitizes action ids, attempts movement, maintains or disconnects laid hoses, computes powered hose reach, distributes finite water, sprays, cuts, performs pickups/deliveries and bilateral handoffs, advances fire, updates victim exposure, then resolves rewards and terminal causes.

A pump operator spends their action supplying others. A stationary hose tender extends reach to four Manhattan cells, whereas other endpoints reach three; moving an endpoint disconnects its laid hose. Pressure propagates through at most three nodes, so abandoning a middle node can deny another responder replenishment. Each connected recipient obtains one unit per pump tick, or up to two if agent 0 cranks; supply is clipped by tank capacity and reserve, in input-clock-rotated order. Capacities are 4,4,3. Initial water is 4,3,2 and reserve 36. Connection and pump-on attributes report pressure/operation during the last transition, not a promised next-step link.

Agent 1 cuts its current cell and four cardinal neighbors, while other roles cut only their own cell; only nonburning cells can be cut. This removes actual fuel and changes later spread. Sprays spend one water and target the highest-intensity neighboring or current fire, with lowest row-major index breaking ties. Agents 0 and 1 remove two intensity; agent 2 removes one. Fire spreads on every fourth output tick into adjacent fueled cells, intensifies, and burns one fuel per burning tick. Empty fuel extinguishes fire. Brush footprint is separate from remaining fuel; a burned-out tile may still carry brush terrain value but has zero combustible fuel.

Each responder may carry one evacuee. Pickup requires a shared tile. Delivery requires action 9 at shelter. Mutual adjacent handoff moves actual carrier ownership and victim coordinates; each responder participates in at most one handoff per tick. Agent 2 moves every tick carrying a victim; agents 0/1 move on even input ticks. This creates a reason to transfer a rescue away from an infrastructure specialist. An uncarried victim accumulates exposure when output intensity is at least two; three exposures cause irreversible loss. Exposure never automatically heals.

## Strategies, payoffs and temporal consequences

One prospective strategy keeps the mechanic at the pump and the tender coupled near the front, allowing the carrier to refill, extinguish and shuttle victims. Another uses stored water and early cross-shaped firebreaks, freeing the mechanic to rescue with slower transport. The useful choice depends on whether partners maintain pressure, where firebreaks are laid, and whether a handoff partner is ready. Long hose chains save travel but occupy people and are broken by movement; carrying everyone directly sacrifices pumping and fire control. These are rule-derived hypotheses, not measured strategic richness.

Every rescue pays all three agents +1 and the delivering carrier an extra +0.2. Every loss pays all three -1. Every active transition costs each agent 0.005. Thus infrastructure choices alter other agents' available resources and shared payoff, while individual delivery credit permits assignment conflicts. Natural termination occurs when all six people are rescued or lost; a mixed outcome is a legitimate terminal. The 96-step horizon is only an upper bound. Reward bounds [-8,8] conservatively contain at most six resolutions and one personal delivery per transition.

## Observations, scene and invariants

All transition-relevant information is public. The flat row is own role one-hot followed by responder positions/water/hose flags, 63 fire/fuel/terrain values each, victim positions/status/carriers/exposure, and reserve/operation/rescue/loss/clock values. Manifest slices enumerate every column. Map coordinates drive movement, heat lookup and rescue access. The fire entity group has exactly 63 fixed cell coordinates and is active only where actual fire is positive. Evacuee entities stay present after rescue/loss with their actual status, so resolved outcomes remain visible. Hoses are displayed through actual laid/pressure attributes; no fabricated visual path is introduced.

State invariants include water 0..capacity, reserve 0..36, fuel 0..4, fire 0..3, exposure 0..3, owner -1..2 and status 0..3. One-person capacity and transfer participation are enforced in the sequential action-resolution logic. All action ids are legal attempts before joint completion; unavailable operations have no effect. Invalid ids become wait. Terminal agents retain wait.

## Prospective variation

Changing the initial brush footprint, victim locations, reservoir or role reach would alter whether pumping, preventive cutting or rapid transport is useful. Such changes require distinct source identities and matching protocols. The current fixed design does not claim measured generalization, learnability or meta-RL performance.

## Literal source-line evidence

- candidate.py:82: pump_on = jnp.any(crank). Pump operation depends on another responder's explicit action and finite reserve.
- candidate.py:89: connected = connected |. A bounded three-hop closure connects laid hose endpoints, changing downstream access.
- candidate.py:97: amount = jnp.where(connected[i]. Water reaches only agents with actual pressure access and consumes the shared reserve.
- candidate.py:109: cut = jnp.any. The tender's physical firebreak changes available fuel rather than drawing a cosmetic line.
- candidate.py:144: ignition =. Fire spread has a four-transition schedule and depends on neighbors and remaining fuel.
- candidate.py:134: transfer = (a[i] == 10). Handoff requires both partners to choose the transfer action and an empty adjacent recipient.
- candidate.py:73: can_walk =. The rescue carrier has a role-specific advantage when transporting a person.
- candidate.py:149: distress = jnp.where. Uncarried evacuees retain irreversible fire exposure.
- candidate.py:155: terminated = jnp.all(status >= 2). Termination records all resolved evacuees, including mixed outcomes.
- candidate.py:182: nxt = jax.tree.map. Terminal inputs are absorbing without resetting their clock or causes.

## Public reference provenance

- references\JaxMARL_README.md: SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\Pgx_README.md: SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; https://github.com/sotetsuk/pgx/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\PureJaxRL_README.md: SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; https://github.com/luchris429/purejaxrl/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\Craftax_Baselines_ppo_rnn.py: SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py. API/design reference only; no upstream implementation copied.
- references\Craftax_README.md: SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; https://github.com/MichaelTMatthews/Craftax/blob/main/README.md. API/design reference only; no upstream implementation copied.

JaxMARL supplies the parallel agent-keyed API idea; Pgx supplies the distinction between game state, legal masks and terminal flags. Craftax supplies resource-world design context. PureJaxRL and Craftax_Baselines are algorithm-integration references, not substitutes for the frozen learner. No upstream implementation was copied. Source author self-report is gpt-6-astra/max; the coordinator retains dispatch evidence.

## Status and limits

AUTHORED_NOT_EXECUTED. Only stdlib AST/JSON/hashes and the supplied static code, manifest and scene validators were used. Candidate code was never imported or executed; no JAX/NumPy, numerical emulation, model API, network, SSH, training or replay ran. Playthroughs and paired interventions are UNEXECUTED_AUTHOR_HYPOTHESIS. Static acceptance establishes neither numerical correctness nor solvability, interaction quality, learning, meta-learning, transfer, browser acceptance or execution authorization. Runtime R0/M0, adapter integration, actual per-role replay and scientific evaluation remain separate. Integer quantities use int32, boolean flags use bool, observations/rewards float32. The external reset key is not duplicated in state. Already-terminal raw inputs preserve all state, clock and terminal causes, with zero reward; natural completion has priority over horizon truncation.
