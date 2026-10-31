# astra-d2-00054: reciprocal sensing with a contested relay

This parentless proposal uses generic decentralized sensing, stale information, shared-channel contention, and production externalities. Its source was authored without a parent, benchmark implementation, experiment outcome, or feedback capsule. Novelty, nonisomorphism, mechanism qualification, compilation, and learning effectiveness are not established.

## Actions and hidden dynamics

Each agent has a hidden binary local production target, a battery, and strain. Crucially, an agent's private survey concerns the other agent's target. Actions are 0 recharge/rest, 1 produce on channel 0, 2 produce on channel 1, 3 survey the partner, and 4 relay the stored survey to the partner. All actions are legal; an operation lacking enough battery does nothing beyond background dynamics. Production, surveying, and relaying consume different amounts of battery.

Production succeeds when its chosen channel matches the producer's current hidden target. Every powered production action schedules a target flip for the partner. That pending flip is consumed on the following transition, so the producer changes the partner's future production conditions. Production also increases own and partner strain. Sufficient accumulated strain causes an additional target flip and partially discharges the strain. A key-controlled random flip can also occur, at a probability depending on current strain. Binary flips combine modulo two, so coincident causes can cancel.

Surveying stores an exact sample of the partner's target from before that transition's target updates. The sample is visible only to the surveying agent. Relaying sends the already stored sample, not a fresh sample, into the owner's inbox. Only one powered sender can succeed, and the shared channel must already be idle. Simultaneous senders collide and set two ticks of channel busy time; a successful relay sets one tick. Attempts while busy fail while still consuming their communication cost. New inbox values cannot affect an action already selected for that transition.

## Information and observation columns

The 16 columns are: time/32; own private survey of the other target; survey age/8; own received message encoded as unknown=0, target-0=0.5, target-1=1; message age/8; own charge; own strain; other strain; shared busy time/2; own previous powered production flag; other previous powered production flag; own previous success flag; other previous success flag; own previous reward; own previous action/4; other previous action/4. Ages saturate at eight. Raw current target bits and future random flips are not directly observed. Historical public actions and success flags can provide additional inferential information; communication is not claimed to be the only way to infer a target.

## Incentives, delay, and reward bound

Correct production pays 0.65, incorrect powered production pays -0.45, and each agent receives 0.15 times the partner's production-success flag. Survey and relay attempts incur small costs. An agent can therefore improve the partner's future information while sacrificing immediate production, or cause an extra future target change through its own production. Collision and busy periods create an endogenous timing constraint on information sharing. Recharge competes with all of those operations.

The final reward expression uses `jnp.clip(..., -1.0, 1.0)`. All production and success indicators are binary, and communication costs are bounded. These are proposed incentives, not evidence that agents exploit the communication mechanism or achieve a learned advantage.

## Interface and evidence status

The implementation intends the exact two-agent, 16-observation, five-action interface with fixed-shape JAX array state and scalar int32 time. Each step increments time once and reports both roles done when new-state time reaches 32. Terminal state is retained, and info contains scalar `done_all` plus the next-state bool legal mask. This handoff contains source and provenance only; permitted syntax parsing is not isolated validation, duplicate screening, mechanism eligibility, or a learning result.
