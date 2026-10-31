# Candidate astra-d2-00024: Perishable escrow with expiring consent

Status: original parentless source authored under the supplied v2 contract. No candidate execution, JAX initialization, isolated validation, compiler check, eligibility decision, or learning evaluation was performed by the author.

## Causal mechanism

Each role owns inventory and a fixed, privately observed component grade. Both may place inventory in public escrow, give consent that expires after three ticks, sell their own stock or escrow privately, or pay to inspect the other's grade. Escrow loses 1.5 percent of its quantity every transition. Its age tracks the existing cohort while a vault stays nonempty; adding to a nonempty vault does not rejuvenate that cohort.

A common shipment starts only when the pipeline is idle, each post-spoilage escrow contains at least 0.35, and both current consent timers are positive. Launch consumes both complete escrows and both consents. Its value depends on the smaller quantity, the mean grade, and exponential age decay. The stored value pays each role three later transitions after launch. Roles can prepare another shipment while the first is in transit. Selling one's escrow revokes that role's consent and can prevent a launch that the counterpart prepared.

These state dependencies create a conjunction across two separate inventories, an expiring authorization window, a perishable waiting cost, and a delayed common payout. Both actors can enable or obstruct the other's delivery without directly controlling the other's inventory.

## Actions and information

The five actions are 0 produce, 1 pledge, 2 authorize, 3 sell all own inventory and escrow, and 4 inspect the other grade. All actions are legal; ineffective attempts can still incur their stated action cost. Production adds 0.35 inventory up to capacity one; a pledge moves at most 0.5 and respects escrow capacity one.

The 16 observation columns are time, own inventory, own grade, own escrow, other escrow, own escrow age, other escrow age, own consent, other consent, shipment timer, shipment value, inspected other grade, inspection-valid flag, own last reward, own last action, and other last action. Ages, timers, actions, and time use the source's displayed normalizations. The counterpart's inventory is hidden. A grade inspection is permanent because grades remain fixed in the episode. A shipment's aggregate value becomes public at launch; that may itself reveal information about the other grade.

## Incentives, delay, and reward bounds

Production, pledging, authorization, and inspection cost 0.025, 0.012, 0.008, and 0.015 respectively. Private sale returns 0.10 times inventory plus 0.025 times escrow. A completed shipment pays its stored value equally to both roles. Consent timing, extra escrow that exceeds the bottleneck, and late launches can all waste resources.

Shipment value is at most 0.82, and the private sale is at most 0.125. Thus even a sale concurrent with an earlier delivery is below 1 before the explicit final reward clip to [-1,1]. The most expensive single action costs 0.025. These are source-level bounds, not measured rewards.

The episode ends on the 32nd transition, preserving that terminal state and observation. Undelivered shipments are neither cashed out nor automatically completed at the horizon.

## Relationship and limits

This intentionally relates to generic escrow, complementary production, perishability, and coordination with expiring commitments. It does not claim a new game family, nonisomorphism, or separation from any unseen proposal. No parent source, benchmark source, or outcome was supplied or consulted. The inspection action may be redundant in parts of the state space, and shared delivery incentives need not produce learning gains.
