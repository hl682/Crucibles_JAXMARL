# astra-d2-00030: Shared inertial control with reversible queued pulses

This is a parentless training-context mechanism proposal based on generic delayed control and coupled-actuator ideas. It is not asserted to be a faithful physical simulation or a distinct isomorphism class. Only the assigned interface and numerical whitelist informed its implementation.

## Transition mechanism and actions

Two separate actuators drive one shared scalar position and velocity. Each has a battery, a gearbox sign, and a two-slot pulse queue. Queued commands store the signed pulse before application of the gearbox. The gearbox sign used at firing time therefore determines the physical direction of an already queued pulse.

Actions are 0 coast and recharge faster, 1 enqueue a positive pulse, 2 enqueue a negative pulse, 3 apply a brake, and 4 reverse one's gearbox if at least 0.08 charge is available. Positive and negative pulses consume up to 0.45 charge. Braking consumes up to 0.25 charge and reduces the shared velocity's damping multiplier; it cannot cancel the new force term directly. Reversal consumes 0.08 charge. Every step restores 0.06 charge, with another 0.12 on coast, capped at one.

The current action's accepted gearbox reversal is applied before the oldest queued pulse fires. The queue's newer slot moves to its older slot, and a newly emitted pulse enters the newer slot. A pulse emitted in one transition thus fires two transitions later, with opportunities for its sign to be altered before or at firing. Battery recharge does not resize a pulse already in the queue.

Shared velocity is the damped previous velocity plus the sum of the two fired forces, clipped to [-0.35, 0.35]. Position is clipped to [-1, 1]. Crossing a boundary leaves position at that boundary and reverses the proposed velocity with factor 0.2. This is a declared simplified collision rule.

## Information and delayed interaction

Each role has a private, episode-fixed preferred position and width. These can favor compatible or conflicting regions. Position, velocity, charges, gears and both pulse queues are visible. A role sees its own consecutive dwell counter and the partner's previous action, but not the partner's target, width or dwell counter.

The sixteen observation columns are: time/32; position; velocity/0.35; own target; own width/0.4; own and partner charge; own and partner gear; own older and newer queued pulses/0.45; partner older and newer queued pulses/0.45; own dwell/4; partner previous action/4; own previous charge expenditure/0.45.

Both agents affect the same persistent velocity and position and thereby both agents' future target quality. The queue, reversible conversion at firing, finite charge, inertia and dwell counters make effects persist beyond the current action. A reversal can reinterpret an existing commitment rather than merely change the direction of the next command.

## Incentives and reward bounds

Own quality is exp(-((position - private target)/private width)^2). A dwell counter increments, capped at four, while quality is at least 0.5 and resets to zero otherwise. Per-step reward is 0.40 times own quality, plus 0.35 while the counter is at least three, minus 0.10 times normalized absolute velocity, 0.12 times own charge expenditure, and 0.05 for a shared boundary collision.

Quality is in [0, 1], the normalized speed is at most one, and own expenditure is at most 0.45 because the actions are mutually exclusive. The expression is bounded by [-0.204, 0.75] before the additional explicit [-1, 1] clip. Dwell rewards favor holding a region, while target disagreement can make prolonged simultaneous dwell impossible for a particular episode. This geometric possibility is not a claim about trained policies or aggregate outcomes.

## Interface and evidence status

The source intends A2/O16/U5, all-legal masks, 32 transitions and an unchanged terminal state observation without automatic reset. Reset uses only its supplied key for position and private targets/widths. The step key is unused because subsequent transitions are deterministic.

The author has not executed the source, initialized JAX, compiled it, run the isolated validator, certified duplicate status, qualified the mechanism or measured learning. Any syntax parse is only a syntax parse. Requested model provenance is recorded separately; immutable backend revision and tokens are unobservable.
