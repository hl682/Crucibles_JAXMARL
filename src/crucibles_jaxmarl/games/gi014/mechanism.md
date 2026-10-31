# astra-d2-00014 — Persistent signatures, complementary work, and unilateral exit

Status: authored source only. Isolated validation and duplicate/mechanism review are pending.

## Causal transition

The agents can agree to produce either of two project types. Actions are 0 wait, 1 sign for type 1, 2 sign for type 2, 3 perform work, and 4 retract one's signature and withdraw from the active project. Signatures persist until replaced, retracted, or consumed by a launch. A project launches only while inactive and when both updated signatures agree. New signatures may also be prepared during an existing project for a later agreement.

Each active project has two complementary contribution requirements. Each agent must contribute two work units for its private specialty, or three for the other project type. Each effective work action adds one unit only to that agent's contribution. Excess work has no effect or cost. A project completes only when both requirements are met. Its payoff is queued and received on the following transition.

Either role can terminate the current project through withdrawal, destroying both accumulated contributions. Withdrawal takes precedence over simultaneous completion. An incomplete project also expires after eight active transitions. Completion at the deadline is allowed unless someone withdraws. A launch transition itself cannot accumulate work. Ending a project and launching another require separate transitions.

## Information

Signatures, project type, both contribution counters, project age, and the latest resolution are public. Each agent knows its own project preference, specialty, required contribution, and pending payout. The partner's preference and specialty are not directly observed; behavior and time taken to finish can provide indirect information.

Observation columns 0–15 are time/32, remaining time/32, own signature/2, peer signature/2, active type/2, own effort/3, peer effort/3, project age/8, own preference/2, own specialty/2, own current requirement/3, own pending payout, latest completion indicator, latest withdrawal indicator, latest expiration indicator, and value of the current project to oneself.

## Incentives and reward bound

Effective work costs 0.07 and signing either type costs 0.01. These actions are mutually exclusive. A finished project pays 0.65 to an agent who prefers its type and 0.25 otherwise, after one additional transition. Rewards therefore lie in [-0.07, 0.65]. The possibility of sunk effort, deadline failure, and unilateral withdrawal makes an agreement different from completed production. There is no terminal salvage for signatures, work, or a payout queued at the final transition.

This mechanism relates to bargaining, complementary team production, and hold-up through an exit right. The proposed transition dependencies join persistent prospective consent, privately heterogeneous workloads, costly bilateral contributions, and delayed settlement. No novelty or mechanism-eligibility conclusion is asserted.

## Interface and provenance

Two agents, sixteen float32 observations each, five always-legal actions, and a 32-transition episode. Reset independently draws each agent's preference and specialty. Later transitions are deterministic conditional on actions. This is a parentless training-context source authored without parent code, experiment outcomes, or feedback.
