# astra-d2-00009: a hysteretic resource commons with delayed restoration

This is an original, parentless source proposal. It has not been executed or validated by the author. Its relation to common-pool resource management, ecological hysteresis, and delayed public-good investment is intentional and generic. No nonisomorphism, eligibility, compiler result, or learning gain is claimed.

## Persistent causal structure

Both agents can extract from shared water, take from their separate reserves, invest reserve into delayed restoration, or clean shared toxin. Water and toxin jointly determine a persistent productive/collapsed regime. Productive stock collapses below water 0.16 or above toxin 0.72. A collapsed system recovers only above water 0.55 and below toxin 0.22. Consequently, crossing back over the collapse boundary does not itself restore productivity.

Actions are 0 wait, 1 extract public water, 2 take private reserve, 3 plant restoration, and 4 clean. Public requests are 0.18 each and are proportionally reduced when shared water cannot meet them. Private extraction is capped by own reserve. Planting requires 0.14 reserve and consumes that amount. A new planting enters a two-position nursery and matures on the second subsequent call after insertion. Mature plantings increase shared water and remove shared toxin, regardless of which agent paid.

Public extraction raises toxin. Cleaning removes toxin immediately. Baseline toxin decay and water regeneration also operate, but collapsed water regenerates more slowly. Own reserves recover according to the new shared water and toxin, so both agents' shared-resource actions change each other's later private opportunities. The existing regime determines current extraction quality and current baseline regeneration; the updated regime governs the following step.

## Information and incentives

Water, toxin, regime, extraction amounts, cleanup, and aggregate restoration information are public. Each agent observes its own reserve and complete nursery but not the other's reserve or full nursery. The observation supplies the two distinct regime predicates so the hysteresis is explicit rather than hidden in an undocumented threshold.

Reward is 0.7 times the fraction of a full public extraction, scaled to 35 percent in the collapsed regime; or 0.45 times the fraction of a full private extraction. Successful planting costs 0.02 reward and cleaning costs 0.04. Actions are exclusive, giving rewards in [-0.04, 0.7]. All actions remain legal: unfunded planting has no planting effect or planting cost. The public benefit of restoration arrives after its private resource cost, creating a free-riding possibility without imposing an intended winner.

## Observation columns

0 elapsed time / 32; 1 shared water; 2 shared toxin; 3 collapsed indicator; 4 own reserve; 5-6 own nursery positions; 7 total newest nursery occupancy / 2; 8 own last public extraction / 0.18; 9 other last public extraction / 0.18; 10 own last private extraction / 0.18; 11 other last cleanup indicator; 12 total last maturation / 2; 13 current recovery predicate; 14 current collapse predicate; 15 agent index.

## Scope and possible failure modes

This is a deliberately small resource-management mechanism, not an ecological model. Simple hoarding, extraction, or restoration schedules may dominate; collapse may be avoidable or recovery may be unattractive within 32 steps. Those are possible empirical outcomes, not exclusions. All terminal state is retained, and trusted validation is still required.