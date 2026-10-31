# astra-d2-00011: private sampling and delayed settlement of weighted reports

This is an original, parentless source proposal. It has not been executed or validated by the author. It is related to generic crowdsourced prediction, reputation-weighted evidence fusion, and delayed verification. It does not claim arbitrary-program nonisomorphism, mechanism eligibility, compilation success, or learning effectiveness.

## Persistent causal structure

There is a hidden binary regime and two private noisy sampling channels. Agents maintain public binary reports and public report weights. A project chooses a binary alternative from the weighted sum of the currently published reports. Any agent can launch when no project is pending; a positive sum selects 1 and a zero or negative sum selects 0.

Actions are 0 wait, 1 inspect privately, 2 publish report 0, 3 publish report 1, and 4 launch. Publishing replaces one's report. Inspection refreshes only one's private signal; publication is a separate action. A launch snapshots the pre-step reports, weights' resulting choice, and hidden regime. A report published concurrently cannot affect that launch.

Only one project can be pending. Its clock starts at two, and it settles on the second subsequent call after launch. At settlement, both agents receive the same signed project outcome. The reports locked at launch gain 0.12 weight when correct and lose 0.16 when wrong, clipped to [0.25, 1]. Agents without a report at launch have no weight update. Editing a report after launch cannot rewrite its locked verification record.

After each settlement the hidden regime flips with probability 0.25, using only the supplied step key. The public verified outcome therefore refers to the just-completed project's regime and need not reveal the new regime. Inspection in that step samples the new current regime. Reports and report weights persist across these changes, so formerly successful evidence can become stale.

## Information and incentives

Each agent sees its own signal, its age, and its own channel accuracy. The other signal, the other channel accuracy, the current hidden regime, and the pending project's stored regime are absent from observations. Reports, weights, project clock and chosen alternative, the most recently verified old regime, and the last project outcome are public. Each agent also observes its own locked report.

A completed correct project pays +0.75 to both agents and an incorrect project pays -0.75. Inspection costs 0.03, publication costs 0.01, and an effective launch costs each launching agent 0.02. With one action per agent, rewards lie in [-0.78, 0.75]. All actions remain legal. A launch action during a pending project has no effect or launch cost.

Either agent can change the shared weighted evidence, trigger a shared commitment, and incur delayed weight changes that alter the influence of later reports. Inspection and publication costs permit free-riding even though project rewards are shared.

## Observation columns

0 elapsed time / 32; 1 own private signal; 2 own signal age / 32; 3 own channel accuracy; 4 own report; 5 other report; 6 own weight; 7 other weight; 8 weighted public score / 2; 9 pending clock / 2; 10 pending choice; 11 most recently verified regime; 12 last signed project outcome; 13 number of launches / 32; 14 own report locked in the pending project; 15 agent index.

Reports, the pending choice, the last verified regime, and the locked report use 0 for absent, 0.5 for value 0, and 1 for value 1. Initially there is no report, project, or verified outcome, and the last signed outcome is zero. The own-signal column is directly binary.

## Scope and possible failure modes

The shared payoff may favor truthful coordination but does not guarantee active sampling, efficient launch timing, or useful report weighting. Persistent weights can amplify stale reports, and the tie rule can favor one alternative. Pending projects are retained at termination without forced settlement. Trusted validation and any empirical conclusions remain outside this authoring step.