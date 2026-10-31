# Escrow Street Market

Status: AUTHORED_NOT_EXECUTED. A=3, O=149, U=15, H=96. Original rules; no external game or learner implementation copied.

## Physical goods and private orders

Three traders inhabit a 9 by 7 public street. Agent 0 initially holds two grain units, agent 1 two tools, and agent 2 two medicine units. Three further individual goods, one of each type, occupy the central stall cells (3,3),(4,3),(5,3). Every one of the nine goods has a persistent id, kind, owner, position, status and lock flag. Goods are never generated or destroyed; delivery changes an object into a delivered object at an actual address. A carried object follows its owner's real position. Each trader has capacity two, so a full trader must exchange or deliver before collecting surplus. Collocation is permitted and trading requires Manhattan distance at most one.

Reset uses exactly one jax.random.randint call on the supplied key to choose, independently for each role, one of the other two commodities. Each private order needs two units of that commodity. No step randomness is used. Demand is deliberately not conditioned on success: two traders may demand four units of a commodity with only three available. Such a joint order cannot be fully fulfilled, and the unchanged world may truncate with unmet demand. This finite-supply competition is explicit, not a fabricated solvability claim or an outcome-based adjustment.

## Enforceable bargaining

Actions 5..7 select a real owned object of the posted type, choosing lowest object id. The selected good is escrowed for a countdown initially set to six. Its ownership and capacity consumption remain with its owner, but delivery is prohibited while locked. Every subsequent step decrements the timer before matching; zero releases the commitment. Reposting replaces the previous escrow. Actions 8..10 advertise a requested type. Posting a request does not reveal the true private order automatically. Action 12 withdraws both request and offer.

A trade occurs only if two roles simultaneously choose action 11, are adjacent or collocated, each has a live owned escrow good, and their public requests are reciprocal. Ownership swaps atomically, exactly one good per partner, preserving carrying capacities. Both locks clear and public requests remain. Each role can participate in at most one exchange per tick; pairs resolve in ascending role-pair order. Refusal, movement away or expiry prevents the exchange without transferring anything. An offer can therefore communicate willingness and create a temporary delivery commitment, but it cannot unilaterally seize another inventory.

After trading, ground pickup resolves in clock-rotated role order; pickup requires physical co-location and available capacity. Action 14 at the trader's own address delivers one unlocked object matching its private type and decrements remaining demand. Each delivered item pays its owner +1 and each other role +0.1. Every active transition costs each role 0.002. Trading, posting and waiting have no separate communication or exchange reward. A completed trader stays active and may broker or help partners; masks depend only on joint completion, not on hidden demand.

## Partner-dependent strategies and conflict

A direct reciprocal swap can help both parties, while a three-way demand cycle may require an intermediate trade in a commodity that is not the broker's final need. Capacity two makes such chains costly and motivates early delivery. A trader may collect a scarce public unit after freeing a slot, changing whether another partner can finish. Withholding confirmation denies the other trader a wanted resource; publishing a request can reveal an actual preference or merely express a proposed intermediary deal. Committing the same unit that could already be delivered imposes a real opportunity cost until withdrawal or expiry. These effects follow from code rules but are not empirical evidence of learned bargaining.

One prospective strategy coordinates reciprocal exchanges near the central stalls and then splits toward the public addresses. Another allows a medicine courier or another role to broker a three-party chain and uses surplus only after inspecting public inventory needs inferred from voluntary requests. When two private orders compete for only three units, agents may rationally prioritize their own two deliveries while receiving smaller benefits from helping others. No winner, successful script or learning result has been asserted.

## Observations and truthful presentation

The first seven columns encode own role, own required type and own remaining quantity. The other 142 columns encode public trader positions, all good positions/types/owners/status/locks, offer ids, requests, timers, delivery counts, destination positions, clock and terrain. Every slice is specified in the manifest. Another role's private required kind never appears in that role's flat observation or mask. The separate private-order entity group has an identity visibility matrix and binds order_kind/order_remaining only there. Public agent entities carry no hidden-order attributes. Delivered type and destination are public historical consequences, so observers may infer preferences from actual behavior without being given unobserved order cards.

The per-role scene must respect manifest visibility. A debug view may expose full state only when explicitly labelled as such by the trusted renderer. The visible surplus stall tile marks its original location even when no object remains; current availability is determined by actual object owner and position. Events identify actual swaps, pickups, prior-offer expiries and deliveries; private order values are not public event channels.

## Invariants, termination and prospective variation

Goods stay among owner codes -2 delivered, -1 ground and 0..2 carried; type and object count remain fixed. Holding capacity never exceeds two. Offer ids are -1..8 and countdowns 0..6. Remaining demand and completed count stay in 0..2. Completed goods cannot re-enter trade. All action ids are legal attempts before completion and unavailable operations have no effect; invalid ids become wait. Natural termination requires all remaining demands zero. Otherwise the 96-step horizon truncates, preserving infeasible or uncooperative outcomes. Terminal inputs are absorbing.

Changing supply counts, demand distribution, commitment duration, addresses or rewards would alter brokerage, scarcity and trust demands. Those are prospective variants requiring new source/protocol identities, not claims of observed transfer.

## Literal source-line evidence

- candidate.py:15: order_kind =. Reset samples only foreign-type private demand from the provided key.
- candidate.py:54: state["order_remaining"][:, None]. Only the observing row's private outstanding quantity enters its observation.
- candidate.py:42: "order_visible": jnp.eye. Order-card scene visibility is identity-masked rather than public.
- candidate.py:88: new_offer = jnp.where. Posting commits an actual currently owned object rather than a nominal token.
- candidate.py:102: reciprocal =. Both advertised requests must match counterpart escrow types.
- candidate.py:104: trade = (i < j). Physical proximity and simultaneous bilateral confirmation gate exchange.
- candidate.py:106: owner = jnp.where((jnp.arange(9) == gi). Successful exchange changes the real owner arrays.
- candidate.py:119: room = jnp.sum(owner == i) < 2. Finite carrying capacity constrains surplus collection and brokerage.
- candidate.py:129: eligible = (owner == i) & (state["good_kind"] == state["order_kind"][i]). Delivery consumes only the owner's private required type and excludes committed goods.
- candidate.py:165: reward = (delivery +. Rewards arise from actual deliveries, never communication or exchange alone.
- candidate.py:143: terminated = jnp.all(remaining == 0). Natural completion is actual fulfillment of all private orders.

## Public reference provenance

- references\JaxMARL_README.md: SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\Pgx_README.md: SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; https://github.com/sotetsuk/pgx/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\PureJaxRL_README.md: SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; https://github.com/luchris429/purejaxrl/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\Craftax_Baselines_ppo_rnn.py: SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py. API/design reference only; no upstream implementation copied.
- references\Craftax_README.md: SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; https://github.com/MichaelTMatthews/Craftax/blob/main/README.md. API/design reference only; no upstream implementation copied.

JaxMARL supplies the parallel agent-keyed API idea; Pgx supplies the distinction between game state, legal masks and terminal flags. Craftax supplies resource-world design context. PureJaxRL and Craftax_Baselines are algorithm-integration references, not substitutes for the frozen learner. No upstream implementation was copied. Source author self-report is gpt-6-astra/max; the coordinator retains dispatch evidence.

## Status and limits

AUTHORED_NOT_EXECUTED. Only stdlib AST/JSON/hashes and the supplied static code, manifest and scene validators were used. Candidate code was never imported or executed; no JAX/NumPy, numerical emulation, model API, network, SSH, training or replay ran. Playthroughs and paired interventions are UNEXECUTED_AUTHOR_HYPOTHESIS. Static acceptance establishes neither numerical correctness nor solvability, interaction quality, learning, meta-learning, transfer, browser acceptance or execution authorization. Runtime R0/M0, adapter integration, actual per-role replay and scientific evaluation remain separate. Integer quantities use int32, boolean flags use bool, observations/rewards float32. The external reset key is not duplicated in state. Already-terminal raw inputs preserve all state, clock and terminal causes, with zero reward; natural completion has priority over horizon truncation.
