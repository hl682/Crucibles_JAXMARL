# astra-d2-00028: Reciprocal collateral with private settlement clocks

This is an initial, parentless, training-context source proposal. It was authored from the supplied interface and whitelist only. It is a generic escrow and reciprocal-credit mechanism proposal; no benchmark source or other proposal was consulted. It does not establish novelty or nonisomorphism.

## Transition mechanism and actions

Each agent owns one possible active bond and a cash balance. A bond has an episode-fixed private maturity term of 3, 4 or 5 aging units and a private cash yield. Issuing locks two units of the issuer's cash. Only the other agent can place the one-unit guarantee that enables settlement. The guarantee is locked until that bond closes, so guaranteeing the partner competes with funding or accelerating one's own bond.

Actions are 0 rest and earn 0.2 cash, 1 issue a bond if the old slot is empty and cash permits, 2 guarantee the partner's existing unbacked bond if cash permits, 3 cancel one's active bond, and 4 spend 0.25 cash to add an extra aging unit to one's active bond. An unsuccessful request is a legal no-op. There is no separate cancellation fee in cash; cancellation returns only one of the two invested units.

Existing bonds age by one before closure tests. A newly accepted guarantee can enable settlement in that transition. A backed bond settles at its private term, returning principal plus yield to its issuer and principal plus 0.25 to its backer. An unbacked bond defaults at term plus two, returning one unit to its issuer. Cancellation is considered after natural settlement and default. Any accepted guarantee on a same-step cancellation is refunded. A new issue occupies an old empty slot and starts at age zero; closing a previous bond does not also issue a replacement. Cash above eight is discarded. The private cash yield affects subsequent funding capacity rather than directly scaling the settlement reward.

## Information and delayed interaction

Cash, active slots, ages, guarantees and settlement/default counts are public. Each agent observes only its own maturity term and cash yield. An agent must decide whether and when to finance a partner whose exact maturity term it does not see. Delays arise from locked collateral and accumulated age. Accelerating an unbacked bond can hasten its default, while the other agent can prevent that default through a guarantee.

The sixteen observation columns are: time/32; own and partner cash/8; own and partner active indicators; own and partner ages/8; own and partner guarantee indicators; own term/5; own cash yield/1.25; own and partner success counts/8; own and partner default counts/8; own last action/4. State is initialized with JAX arrays, and only reset samples private terms and yields.

## Incentives and reward bounds

Per-agent reward is 0.65 for its own settlement, 0.20 for the partner's settlement, minus 0.10 for issuing, 0.04 for accepting a guarantee, 0.035 for successful acceleration, 0.25 for its own default, and 0.12 for its own cancellation. Settlements require guarantees, so the partner settlement credit compensates actual bilateral cooperation. The sum of all positive terms is at most 0.85 and the sum of all possible negative coefficients is 0.545, even before accounting for mutually exclusive events. The implementation additionally clips to [-1, 1]. Cash refunds are state transitions, not extra reward.

Both agents can change their own future liquidity and directly enable the other's settlement. End-of-episode outstanding bonds receive no artificial terminal liquidation; the preserved terminal state exposes any unresolved position.

## Interface and evidence status

The intended interface is A2/O16/U5 with all actions legal, 32 transitions, float32 observations and rewards, and a terminal observation without automatic reset. The step is deterministic conditional on the initialized state and actions; its supplied key is unused.

This author has not imported or executed the candidate, initialized JAX, compiled it, run an isolated validator, measured duplicate relationships, qualified a mechanism, or assessed learning effectiveness. Static syntax parsing, if separately reported, is not any of those outcomes. Backend revision and token counts are unknown. No development or final-test material was used.
