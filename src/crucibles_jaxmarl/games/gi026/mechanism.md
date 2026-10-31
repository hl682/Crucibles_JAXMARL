# Candidate astra-d2-00026: Endogenous extraction and delayed sensing

Status: original parentless source authored from the supplied contract and whitelist. The author has not executed candidate code, initialized JAX, used a compiler or isolated validator, or measured learning.

## Causal mechanism

There is one hidden, evolving scalar field and two extraction sites. Each agent knows its own fixed site offset and resource stock. Harvesting the positive or negative side pays according to the signed local field but removes local resource and moves the common field away from the harvested direction. The current reward therefore depends on a hidden quantity that both agents' previous actions helped change.

An agent can instead take a noisy private measurement, relay its most recent measurement after subtracting its known site offset, or restore resources. A relay first enters an outgoing pending state and arrives at the counterpart's next transition. Measurement ages are carried through this delay. Sampling observes the field before that transition's extraction and drift; its displayed age is already one in the returned observation.

Harvesting creates persistent interference for the counterpart's future measurements. The interference decays by half each step and receives a 0.35 increment from the other's harvest action. Restoration replenishes the restorer's site by 0.22 and the other site by 0.12, in addition to passive regeneration of 0.025. Restoration also damps the common field. This couples resource availability, information quality, and the underlying reward target rather than merely changing a payoff label.

## Actions and information

The five actions are 0 sample, 1 relay the last valid measurement, 2 harvest negative, 3 harvest positive, and 4 restore. Harvest removes at most 0.30 resource. All actions are legal; relaying without a valid sample sends nothing but still incurs its action cost.

The 16 observation columns are time, own sample, sample age, sample-valid flag, inbox message, inbox age, inbox-valid flag, own site offset, own richness, mean richness, own interference, outgoing-message-valid flag, own last extracted amount, own last reward, own last action, and other last action. The source normalizes time, samples, ages, offset, extracted amount, and actions. Both resource stocks can be reconstructed from own richness and the public mean; the field and counterpart offset remain unobserved.

The inbox stores a bias-corrected, possibly noisy and stale field estimate rather than the counterpart's raw local observation. The pending value and age are stored in state, while the sender sees only the pending-valid flag and its own measurement history. Previous harvest rewards can also provide information about the field.

## Incentives, randomness, and reward bounds

Harvest reward equals 0.45 times extracted amount times chosen sign times local field, plus 0.10 times extracted amount. Sampling, relaying, and restoration cost 0.015, 0.010, and 0.025 respectively. The clipped local field lies in [-1,1] and extraction is at most 0.30, so a harvest lies in [-0.105,0.165]. The other actions give only their stated nonpositive cost. A final explicit clip enforces [-1,1]. These are mathematical bounds from the source, not observed returns.

The explicit step key supplies bounded private sensor noise and bounded common drift through separate subkeys. Noise amplitude depends on the counterpart-induced interference. The reset key supplies the initial field, site offsets, and richness. No randomness is obtained from external state.

Current extraction can make future same-direction extraction less attractive for either role. Restoring or relaying incurs a private immediate cost and can benefit the counterpart later. Stale or noisy messages may be less useful than information inferred from actions and rewards. No beneficial communication effect is presumed.

## Relationship and limits

This relates to generic active sensing, delayed communication, renewable resource externalities, and decision making with an action-dependent latent process. It is not a benchmark adaptation or a claim of a novel game class. Parent sources and outcome data were not supplied or read. Mechanism qualification, duplicates, compiler behavior, and learning effectiveness remain unassessed. At transition 32 the terminal state is retained; pending messages and remaining resources receive no extra terminal reward.
