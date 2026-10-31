# Tide Ledger: Salvage Claims

A=3, O=64, U=11, H=64. The yard has cells x=0..8, y=0..6. An impassable vertical breakwater occupies x=4 except the two openings at y=1 and y=5. Salvagers start at (1,1), (1,3), (1,5); the shared depot is (0,3); the three lots start at (7,1), (7,3), (7,5). Ground objects do not block motion. Agents have exclusive physical cells.

## Auction and transport rules

At input clocks 0,1 auction lot 0; at 2,3 auction lot 1; at 4,5 auction lot 2. Both ticks allow public commitment updates. Wait retains the previous bid. Bids 1,2,3 must strictly increase the current commitment and fit the bidder's remaining purse. Each closure considers the newly posted bids. If every bid is zero the lot remains open access. Otherwise the highest bid receives its exclusive claim; ties use the smallest (agent_id-lot_id) mod 3. The winner pays its own bid once to the treasury. Losing bids cost no tokens. Commitments remain visible by lot after closure, and spent tokens never regenerate.

Reset independently draws each valuation uniformly from integer 2..8 using fold_in(reset_key,0), and each initial purse uniformly from integer 4..7 using fold_in(reset_key,1). Values and purses are private to their owner, including in scene view. Public bids and treasury receipts permit inference but do not expose the hidden state directly. The initial token sum is state for provenance/conservation, not an observation.

Movement starts at input t=6. Directions are global compass attempts. A move fails if outside the map, into the breakwater, into another role's input cell, or toward a destination proposed by multiple roles. This deliberately excludes swaps and following into a vacated cell in the same tick. Thus yielding a bottleneck helps a partner on a later tick, rather than permitting overlap.

Interact picks the lowest-index eligible undelivered ground lot on the current cell, only when empty-handed. Eligibility requires an open claim or one's own claim. Interact while carrying delivers only at (0,3). Drop leaves the physical lot and its existing claim on the carrier's cell. Release removes the lowest-index owned, uncarried, undelivered claim; it does not refund tokens. At next clocks 28,32,36 the corresponding claim becomes open access if its lot is still uncarried and undelivered. A carried object is never confiscated on expiry. The input-state mask is applied first: a claim freed during the current transition generally becomes newly pickable on the next decision.

A delivered lot becomes inactive at the depot and cannot reenter play. Each valid delivery gives every role 1 and adds the deliverer's own lot valuation to that deliverer's reward. Auction payment contributes minus the amount paid. Completing all three lots gives each role a further 2 and naturally terminates. H=64 is the time limit, not a required duration. Per-step reward is bounded by [-3,12]; this is an interface bound, not a performance gate.

## Cross-agent mechanism and strategies

An opponent's bid can remove an affordable retrieval opportunity, change who pays, and require waiting for expiry or negotiating a release. An expensive exclusive claim can protect a nearby valuable retrieval, while a cheap bid or abstention preserves tokens and risks losing access. Claim ownership does not teleport an object or make transport independent: the north/south passages and common depot are shared physical bottlenecks.

A fast owner can deliver for its own private value, an owner can relinquish a poor-value claim so another player earns its own higher value, or a carrier can drop and release a lot for a partner to finish. Shared delivery and completion rewards can justify yielding access despite individual competition. Delaying another player may secure a favorable open-access race but also loses team completion value. These are author-described alternatives, not measured equilibria or learning findings.

The auction paired hypothesis changes only role 1's bid and keeps role 0's bid/key fixed; the intended downstream difference is claimant, payer and partner access after closure. A second pair preserves both roles' northward requests while role 0 either leaves or holds its starting route cell; old-occupancy semantics predict a delayed role 1 movement difference. Both are literal, unexecuted hypotheses. The delivery script requests one complete lot-0 route while other players wait; no successful replay is claimed.

Changing claim deadlines, purses, values, passage geometry or the tie-priority assignment would change the balance between early commitment, waiting, blocking and handoff. No such extra task has been generated here.

## Visibility, conservation and presentation

All public spatial entities have true world coordinates. Private purse/value cards travel with their actual owner and use self-only visibility. The salvage carrier and claimant are distinct fields; the scene does not equate a winning bid with possession. Delivered lots are inactive, while their delivery event and public delivered observation persist according to their declared lifetimes. Auction-close/pickup/release/delivery flags describe only the last transition; an absorbing terminal call preserves that final frame rather than creating another event.

Intended invariants: each budget is nonnegative; sum(budget)+treasury equals initial_tokens; the three lots are neither created nor duplicated; each agent carries at most one lot; a carried lot occupies its carrier's cell; delivered lots are inactive and uncarried; active agents occupy distinct walkable cells; all integer IDs and positions remain bounded. The schema expresses nonnegative resources and unique active agent positions directly. Conservation and carrier relationships are specified here for later runtime review and have not been numerically checked.

## Literal source evidence

Evidence refers to the frozen candidate.py with SHA-256 22d1dc5fbcc3633ea3cf707b9815b2aac03c8aced58f88d5177a7e0d6c0d0e6f. Line numbers were obtained by parsing/reading source text without importing it.
- candidate.py:20-56: reset implementation.
- candidate.py:58-76: legal_actions implementation.
- candidate.py:78-160: raw_step implementation.
- candidate.py:162-175: observe implementation.
- candidate.py:12-18: refresh implementation.
- candidate.py:92: rank = (ids - lot) % 3 - Rotating tie priority makes another bidder change the claimant.
- candidate.py:95: payment = jnp.where(award & (ids == winner), best, 0) - Only the winning positive bid moves tokens to treasury.
- candidate.py:115: expiry = (nt >= s["deadline"]) - Expired uncarried claims become open opportunities.
- candidate.py:109: occupied = jnp.any - Old positions create physical cross-agent blocking.
- candidate.py:141: reward = reward.at[i].add - Individual value is paid only for physical delivery.
- candidate.py:156: n = jax.tree.map - Terminal state is selected leaf by leaf without autoreset.

## Boundary and source provenance

This package is an initial author artifact, status AUTHORED_NOT_EXECUTED. It has not been imported, reset, stepped, JIT-compiled, vmapped, trained, rendered, or subjected to action search. Source AST screening and the copied data-only manifest/scene validators are separate from R0 and numerical parity. All playthroughs and paired interventions remain UNEXECUTED_AUTHOR_HYPOTHESIS; a later executor must use their literal keys and record actual legality, causes and failures.

Every nonterminal raw call increments scalar int32 t once. Natural termination has priority on the horizon: truncated=(t>=H)&~terminated. done is their union. An input whose done flag is already true preserves every state leaf and its observation, returns zero reward and all dones true, and retains both terminal-cause flags. No raw reset occurs. Invalid indices and actions disallowed by the input-state mask are replaced deterministically with wait. Attempted movement is legal even when the destination is blocked; geometry/collision resolution may leave position unchanged. All externally visible state uses int32, bool or float32 arrays of fixed shapes. The supplied reset key is not duplicated in state.

The agent-keyed JaxMARL adapter, real replay, RGB/animation, browser playback controls, sealed R0, numerical parity, learning and transfer measurement are separate integration/execution work. The author supplies actual state bindings for them, not an acceptance receipt. Inactive roles have action 0 available until joint termination.

Community sources consulted were the supplied JaxMARL README (parallel agent dictionaries and __all__), Pgx README (explicit step keys, legal masks and board-state interfaces), Craftax README (persistent spatial object interaction and a JAX-native environment boundary), and PureJaxRL README (end-to-end functional integration). These are interface/design inspiration only. No upstream source code, layout, game rule, PPO implementation, assets or weights were copied. The supplied Craftax_Baselines_ppo_rnn.py was hash-verified but its body was not used. Craftax mechanics, Pgx board APIs and PureJaxRL training implementations are distinct; no learner replacement is claimed. All source rules in candidate.py were authored prospectively for this assignment.

Only the assigned design/request, supplied source contracts/schema parsers and copied official references were inputs. The 20-file batch manifest and authorization SHA were verified. No prior candidate, project outcome, replay, ranking, training log, heldout file, result_N or other author's directory was read. This source stream is train-only and final_test_used=false. Author identity is a self-report; dispatch provenance remains with the coordinator.

Rule, map, role or resource variations discussed here are research possibilities, not generated extra tasks and not evidence of generalization. Shape/mask/schema checks cannot establish strategic richness, non-isomorphism, solvability, learning or meta-learning. Negative, failed, zero-effect and indeterminate execution findings remain valid outcomes.
