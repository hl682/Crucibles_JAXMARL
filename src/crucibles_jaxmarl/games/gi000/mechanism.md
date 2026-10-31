# Delayed energy transfers
Status: authored proposal, not executed or validated.

Two agents own bounded energy stores. Charging increases one's store; transmitting removes a packet now and adds it to the other agent's store at the next transition. Harvesting pays only above a storage threshold and consumes the threshold amount. Interception transfers some of the other's currently available energy with a private cost. Simultaneous interceptions are calculated before either transfer, so neither agent has ordering priority.

All persistent state is public and represented from each observer's own/other perspective. Last actions, recent harvest, pending packets, time, and role identity are observable. Each action is always legal; an unsuccessful harvest simply produces no harvest. Both agents directly change stored energy and can change the other's subsequent harvest opportunity.

The delayed packet introduces a distinction between retaining resources, immediate appropriation, and voluntarily moving resources across the dyad. Harvest is bounded by 0.6 per step; charging costs 0.02 and interception costs 0.04, with rewards explicitly clipped to [-1,1]. Storage caps can waste resources and give situations in which transfers preserve otherwise wasted charging capacity. No claim is made that cooperation is optimal or learned.

This is intentionally related to generic resource-transfer and threshold-harvesting games. Distinctness from other submissions, mechanism eligibility, compilation, and learning effectiveness are unassessed. Randomness only initializes energy from the supplied key. There is no auto-reset; terminal state is retained at t >= 32.
