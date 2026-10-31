# Reservoir with delayed pollution and living restoration
Status: authored proposal, not executed or validated.

Agents share a bounded reservoir and pollution stock. Gentle or intensive pumps request different water amounts. If demand exceeds water, both demands are scaled by the same rationing factor. Intake pays according to the old water quality. Intensive pumping creates runoff that reaches pollution one transition later, so immediate extraction can damage future value after an observable lag.

Planting grows an agent's bounded root stock, which decays and is damaged by intensive pumping. Old roots jointly improve replenishment and reduce pollution. Filtering pays a private cost and immediately reduces next-state pollution. Root benefits therefore arrive after investment; clean water and replenishment are shared. Both agents influence common persistent state, and their simultaneous water demands compete.

All physical state is public: water, pollution, pending runoff, both root stocks, both recent intakes, both previous actions, time, and role. Actions are always legal, including pumping an empty reservoir. Benefits are at most 0.32 per agent per step; planting and filtering cost 0.07 and 0.05 respectively, so each reward stays within [-1,1].

This is intentionally a generic common-pool-resource and public-restoration mechanism with a delayed externality. No assertion of novelty, nonisomorphism, eligibility, compilation, or scientific effectiveness is made. Only initial water uses the explicit random key. The time limit is 32 transitions, with no internal reset.
