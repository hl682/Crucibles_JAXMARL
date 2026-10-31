# astra-d2-00031: Asynchronous consensus with expiring commitments and stale evidence

This initial parentless proposal uses familiar decentralized inference, cheap-talk and asynchronous-agreement ingredients. Their combination is a proposed causal mechanism rather than a claim of formal novelty, nonisomorphism or benchmark independence established by testing. The author received no other proposal or evaluation source.

## Transition mechanism and actions

There is a hidden binary target, two private noisy clues, persistent public messages and two expiring public commitments. Each role has an episode-fixed private clue reliability sampled between 0.65 and 0.9. Independent clue noise is sampled only through the supplied reset or step key. The implementation draws a potential clue each step but installs it only when the role samples.

Actions are 0 obtain a fresh private clue, 1 announce zero, 2 announce one, 3 commit zero and 4 commit one. Announcements are arbitrary chosen messages: the environment does not force them to match the sender's clue. They persist until overwritten. A new commitment replaces one's old one and starts with age zero. An unrenewed commitment is available for the next two transitions and expires when its incremented age reaches three.

Once both commitments are present, agreement on the hidden target earns a common successful-resolution outcome, agreement on the wrong bit earns a common error outcome, and disagreement earns a common conflict outcome. Any such paired resolution clears the commitments and starts a new target epoch. A target epoch also ends after eight transitions even without resolution. The new target is resampled and may happen to equal the old bit; an epoch change still invalidates old evidence. Pair resolution is evaluated against the old target before a same-step time-based refresh.

Clues and messages remain stored across epoch changes. Their epoch stamps explicitly identify them as stale. Sampling and announcing occur before that transition's target refresh, so a clue or announcement obtained on a refresh transition is immediately stale in the returned observation. This ordering is intentional. New targets are never inserted directly into an observation.

## Information and delayed interaction

Private observations include only one's clue, its age/freshness and one's reliability. Both commitment values and ages are public. Messages are public and carry freshness/age information. A role can follow a partner's public commitment, which creates a timing problem for the first committer rather than requiring simultaneous actions.

The sixteen columns are: time/32; own clue; own reliability; own clue-current indicator; own clue age/32; partner message encoded as (value+1)/2; partner message-current indicator; partner message age/32; own message with the same encoding; own message-current indicator; own and partner commitments using the same encoding; own and partner commitment ages/3; epoch phase/8; previous common resolution outcome. The encoding maps missing to zero, bit zero to 0.5 and bit one to one.

An agent's message and commitment persist and influence the other's decision context. Its commitment can combine with the other's stored commitment to determine both rewards and reset the hidden target, invalidating both roles' evidence. Resampling therefore has an endogenous cause as well as the fixed epoch timeout.

## Incentives and reward bounds

The common resolution outcome is 0.75 for correct agreement, -0.55 for wrong agreement, -0.20 for disagreement and zero without paired resolution. Individually, sampling costs 0.035, announcing costs 0.01, committing costs 0.02, and an unrenewed commitment expiring costs 0.06. Role actions are mutually exclusive. A conservative expression bound is [-0.645, 0.75]; the code also clips to [-1, 1]. Resolving a pair necessarily involves at least one current commit action, so the expression's loose upper bound need not be reached by both roles simultaneously.

The incentives trade additional private evidence and communication against stale information, commitment expiry and the finite horizon. With no wait action, every action has a small cost; this is explicit. A clue's known reliability can justify uncertainty, but no policy performance or learned inference ability is asserted.

## Interface and evidence status

The source intends the exact A2/O16/U5 interface, all-legal masks, 32 transitions and preservation of terminal state/observation. Every state entry is an initialized JAX array. The supplied step key supplies independent potential clue noise and target refresh randomness, with no external randomness.

The author did not execute candidate Python, initialize JAX, invoke a compiler, run isolated validation, inspect learning results, determine duplicate eligibility or qualify this mechanism. A syntax parse is not such evidence. Backend revision and model-token counts are unknown and remain null in provenance. All source and generation-context labels are train; final_test_used is false.
