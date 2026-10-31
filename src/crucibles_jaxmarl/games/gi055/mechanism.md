# astra-d2-00055: cross-pollination with delayed contagious cost

This parentless source proposal combines generic mutualistic growth, staged production, common crowding, and infection externalities. The setting is a stylized transition system, not a calibrated biological model or an adaptation of an inspected benchmark. No parent source, experiment score, or feedback capsule was used. No claim is made about novelty, nonisomorphism, eligibility, compilation, or learning effectiveness.

## Actions and persistent transitions

Each agent owns biomass, moisture, persistent flowers, fruit, soil quality, a seed bank, and blight. Incoming pollen and incoming spores are stored separately. Actions are 0 rest and sanitize, 1 sow from the seed bank, 2 irrigate, 3 create flowers and emit pollen, and 4 harvest existing fruit. All actions are legal and consume available quantities where applicable.

Sowing converts seed stock to biomass and slightly depletes soil. Biomass also grows gradually according to soil, moisture, remaining growth capacity, and blight. Flowering spends some biomass, creates flowers, and sends pollen to the other plot's next state. That pollen can fertilize only the recipient's already existing flowers during the following transition. Fertilization creates persistent fruit, which is available for a later harvesting action. Unused flowers and fruit decay. Thus one plot cannot create productive fruit from its own emission alone: the other agent's flowering action supplies the cross-plot input.

An unhealthy emitter also sends spores proportional to its current blight and emission. Those spores enter the partner's next state and increase blight on the following transition. Irrigation can induce local blight when moisture before evaporation exceeds the threshold. Rest sanitizes, replenishes seeds and soil, and helps retain moisture. Harvesting returns some seed stock and weakly reduces blight. Combined biomass above a shared threshold creates shade that suppresses both plots' future biomass. Each transition receives one shared rainfall draw from the explicit random key.

## Information and observation columns

The 16 columns are: time/32; own biomass; own moisture; own flower stock; own incoming pollen; own fruit; own soil quality; own seed bank; own blight; other biomass; other flower stock; other fruit; other blight; own previous usable harvest; other previous usable harvest; other previous action/4. Other moisture, soil, and seed stock are private. Pending spores are not directly included in observations, though visible historical blight and actions can help infer transmission. Current stocks describe both immediate harvesting opportunities and the delayed pipeline of flowering, fertilization, and fruit.

## Incentives, delay, and reward bound

Own usable harvest pays 0.80 per unit, and the partner's usable harvest contributes 0.12 per unit. Usability decreases with the harvesting plot's blight. Irrigation, sowing, and flowering incur small immediate costs; existing blight incurs an ongoing penalty. Flowering can benefit the other plot through pollination while carrying an infection risk; expanding biomass can increase own productive capacity while imposing bilateral shade. Rest may sacrifice a current harvest or flowering opportunity for future health and resources.

The reward is the clipped sum of these bounded harvest benefits and costs, with `jnp.clip(..., -1.0, 1.0)` enforcing the declared range. Stock clipping represents stylized capacity and nonnegative-resource constraints rather than a claim of physical conservation. Coordination, recovery, and asymmetry are possible hypotheses; no learned response has been observed.

## Interface and evidence status

The intended interface has two agents, 16 float32 observations per agent, five actions, and fixed-shape JAX array state. Scalar int32 time advances once per transition. Both done flags are true when next-state time reaches 32, and terminal state and observations are returned without resetting. Info supplies scalar `done_all` and a next-state bool legal mask. Source authoring and any syntax parsing are the only author-side evidence. Isolated validation, structural/trajectory review, mechanism qualification, and learning results remain separate and unestablished.
