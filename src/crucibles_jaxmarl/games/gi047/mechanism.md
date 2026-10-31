# astra-d2-00047: Shared irrigation with delayed cross-patch succession

Status: original parentless source proposal. The author has not run this environment or an isolated validator.

Each role manages one patch containing two abstract plant types: a lower-yield persistent type and a higher-yield water-demanding type. The actions are rest, sow type 0, sow type 1, harvest, and toggle the irrigation gate while surveying the other patch. Sowing is effective only with sufficient local nutrient. All action indices are legal. The state contains adults, a seed bank, seedlings, local nutrient, a shared water reserve, and persistent gate settings.

Sowing enters the seed bank immediately. Only the previous seed bank contributes to next seedlings; only previous seedlings contribute to next adults. Seed-to-harvest effects therefore require multiple transitions. Adults recruit locally, while adult-derived seeds also reach the other patch when both stored gates are open. The two types differ in survival under water shortage, growth, nutrient effects, and harvest value. Patch-wide occupancy scaling makes the species compete for persistent space.

Water recharges seasonally and is allocated proportionally to the two requests when scarce. Adult composition and the previously stored gate determine request size and hydration. A closed gate still permits reduced intake. Opening a gate can improve local hydration but increases competition for shared water and permits bilateral seed exchange. Toggle effects on water and diffusion start on the next transition. Harvest removes part of the previous adult biomass and restores some local nutrient; its effect on future water demand is delayed until the changed biomass is used.

Each role sees its own adults, seed bank, seedlings, nutrient, and most recent harvest yield, plus both gates and the shared reserve. Surveying occurs only with a gate toggle and caches the other patch's post-transition total adult mass and type-1 fraction. These samples become stale. Shared water and public gate changes provide additional indirect information, so the information structure is partially observed rather than private in an absolute sense.

The 16 columns are time, two local adult quantities, two local seed-bank quantities, two local seedling quantities, nutrient, own gate, other gate, normalized shared water, last local intake, cached other adult mass, cached other type-1 fraction, cache age, and last local harvest value. Initial plants, nutrient, and water use only explicit-key randomness; later dynamics are deterministic.

The reward is tanh of harvest value minus sowing, survey/toggle, and drought costs. It is bounded in [-1, 1]. Remaining biomass and juvenile stages are preserved after transition 32 and are not assigned hypothetical future harvest value. Both roles change the shared water trajectory, and gate choices jointly control cross-patch composition over time.

This is a synthetic resource-and-succession model related to common-pool allocation, staged growth, and dispersal ecology, not a scientifically calibrated ecosystem. It offers a causal mechanism proposal rather than a uniqueness claim. Nonisomorphism, operational mechanism eligibility, compiler success, and learning effectiveness remain unasserted. No parent, evaluation task, or outcome feedback was supplied.
