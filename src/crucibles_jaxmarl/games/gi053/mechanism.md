# astra-d2-00053: shared-capacity delayed liquid exchange

This parentless source proposal draws on generic queueing networks, inventory control, maintenance, and common pollution externalities. It is not adapted from an inspected game or parent source. No experiment outcome or feedback capsule was supplied. No assertion of novelty, nonisomorphism, mechanism eligibility, compilation, or learning effectiveness is made.

## Actions and persistent transitions

Each agent operates a receiving reservoir, a replenishable private cistern, local equipment wear, and a private desired sale size. A two-stage incoming pipe stores material addressed to that agent. Actions are 0 rest and replenish, 1 pump from one's cistern, 2 route reservoir liquid to the other agent, 3 sell liquid, and 4 service equipment while venting some stored liquid. All actions are legal; quantities are limited by current stocks.

Routing requests depend on the sender's current reservoir and wear. A shared per-transition capacity scales both requests proportionally when they compete. Only the allocated amount leaves the sender; unallocated material stays in the reservoir. A routed amount enters the partner's far pipe stage in the next state, advances to the near stage on the following transition, and then arrives on the next transition. Operations use the old reservoir, so an arrival cannot be sold or sent onward before it has entered persistent storage.

The reservoir has finite capacity. Arrivals and pumping can cause overflow, which penalizes both agents. Pumping, arrivals, and sales increase local wear; service reduces wear but consumes an action and vents liquid. Pumping and overflow increase a persistent shared silt level, while service by either agent lowers it. Wear limits future routing, and both wear and shared silt lower future sale quality. A sale draws that owner's next private desired quantity using the supplied key. Cisterns recharge slowly every tick and faster during rest.

## Information and observation columns

The 16 columns are: time/32; own reservoir/2; own cistern/1.5; own incoming near pipe stage/0.6; own incoming far pipe stage/0.6; own wear; own desired sale size; other reservoir/2; other wear; own previous sale; other previous sale; own previous overflow; other previous overflow; shared silt; own previous action/4; other previous action/4. The other's private cistern and desired quantity are hidden. Each recipient observes its own committed incoming material, making advance overflow avoidance possible without exposing the partner's entire state.

## Incentives, delay, and reward bound

Selling offers an immediate return based on sale quantity, match to the private desired size, and quality. Routing gives a small transfer reward, removes present stock, and creates a delayed inventory opportunity or overflow risk for the partner. A fraction of the partner's sale is also rewarded, while overflow imposes bilateral penalties. Servicing has an immediate cost and material loss but reduces both local wear and common silt. Shared transport capacity, delivery delay, and pollution allow one agent's timing to alter the other's future opportunities.

Rewards are the clipped sum of sale value, a small partner-sale benefit and routing payment, minus pumping, servicing, and own/partner overflow costs. The explicitly bounded stocks and the final `jnp.clip(..., -1.0, 1.0)` determine the reward range. This stylized transfer system is not a physical fluid simulator and does not establish an advantage for any coordination strategy.

## Interface and evidence status

The source uses two agents, 16 float32 observations per role, five all-legal actions, and fixed-shape array state including a scalar int32 clock. The clock advances once; both roles terminate when the new time reaches 32, with no reset inside `raw_step`. Info contains scalar `done_all` and the new-state legal mask. Only authoring and permitted syntax parsing belong to this handoff. Isolated compiler/runtime checks, duplicate review, mechanism qualification, and learning experiments are separate and remain unestablished.
