# astra-d2-00052: perishable bilateral escrow with delayed release rights

This parentless source proposal combines generic escrow, reciprocal service, and perishable inventory mechanisms. No supplied parent, benchmark source, experiment score, or feedback capsule was used. This description does not establish novelty, nonisomorphism, operational eligibility, compilation, or learning effectiveness.

## Actions and persistent transitions

Each agent has a private wallet and quotation, one persistent parcel slot, a parcel age, a release seal, and a private recipient demand. Actions are 0 replenish the wallet, 1 place a parcel, 2 witness the other agent's parcel, 3 release one's own parcel, and 4 recall one's own parcel. All five actions are legal. An unavailable operation has no direct effect, while the normal clock, demand recovery, and parcel aging still occur.

Placement requires an empty slot and at least 0.25 wallet units. It locks up to 0.8 units. A witness can seal the partner's existing unsealed parcel only after it has aged at least one tick. Release tests the previous state: a newly issued seal cannot be consumed in that same transition, and a parcel must already be at least two ticks old. Thus one agent's service changes the other's future feasible payoff-producing action. A parcel at age seven expires unless its owner releases or recalls it on that transition. The release amount is attenuated by age and the owner's private quotation. Recall returns an age-dependent fraction; expiry returns only a small fraction. Clearance draws the owner's next private quotation from the explicitly supplied random key.

The sender receives partial wallet reimbursement and a delivery reward. The other agent receives an immediate reward from the same delivery, scaled by that recipient's private demand, and its future demand decreases. Waiting rebuilds demand and, with action 0, liquidity. An exponentially decaying public reputation records delivery and witnessing history; it is an observation of history, not an additional gate or payoff multiplier.

## Information and observation columns

Each row is from that agent's perspective. The 16 columns are: time/32; own wallet/2; own parcel quantity; own parcel age/8; own release seal; own quotation; own recipient demand; other parcel quantity; other parcel age/8; other seal; own reputation; other reputation; own previous delivery; other previous delivery; own previous action/4; other previous action/4. The other wallet, quotation, and demand are not directly revealed. No observation exposes a future random draw.

## Incentives, delay, and reward bound

Replenishing, placing, witnessing, and releasing consume different action opportunities. Witnessing offers a small immediate service reward and enables the partner's delayed release; the owner still controls whether release occurs. Releasing an old parcel loses value, while recalling recovers liquidity without creating the delivery benefit. This creates a reciprocity and timing problem with private values, not a claim that cooperation is necessarily learned or optimal under every state.

The reward is the clipped sum of 0.65 times own delivery, 0.25 times incoming delivery times a bounded demand factor, a small witnessing payment, and explicit placement/recall/expiry costs. Every component uses bounded state quantities, and the final `jnp.clip(..., -1.0, 1.0)` sets the declared reward range. Both agents can change persistent state and each other's reward: a witness changes the other release seal; a delivery changes the other's demand and immediate return.

## Interface and evidence status

The intended interface is two agents, 16 float32 observation values each, five actions, and a scalar int32 clock incremented once per transition. Both done flags become true at next-state time 32; the returned terminal state and observation are preserved. The info dictionary supplies scalar `done_all` and the next-state legal mask. Source authoring and permitted syntax parsing are separate from isolated validation, duplicate review, mechanism qualification, and any learning result; those remain unestablished here.
