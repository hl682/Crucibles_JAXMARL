# Candidate astra-d2-00027: Cross-wired registers and stability audits

Status: original parentless source authored under the supplied contract. No candidate execution, JAX initialization, compiler check, isolated validation, eligibility decision, or learning experiment was performed by the author.

## Causal mechanism

Each agent owns a three-bit register and a privately observed three-bit target. Flipping either of the first two local bits emits a delayed pulse to the other register: flipping local bit zero pulses the other's bit one, and flipping local bit one pulses the other's bit two. Pulses become the counterpart's pending input and are applied on its next transition. Addition modulo two implements the bit flips.

A role may rotate its whole register, shield against its pending pulses, or start an audit. An audit snapshots whether the post-action register matches the private target and then waits two further transitions. It succeeds only if that snapshot matched and the register has remained unchanged at the end of each intervening transition. Changes that are later undone still spoil the audit, because a persistent dirty flag records them. Exactly canceling incoming and local flips within one transition does not count as a change: stability is defined at transition endpoints.

A successful audit complements all three target bits, creating another local obligation. A failed audit leaves the target unchanged. The other agent can help construct a needed bit or spoil an audit by emitting a delayed pulse; its visible audit timer also creates an opportunity to avoid interfering.

## Actions and information

The five actions are 0 flip bit zero, 1 flip bit one, 2 rotate the register by one position, 3 shield, and 4 start an audit when idle. All actions remain legal. Action 4 while an audit is running does not restart it. Local rotation emits no pulse.

Shielding works only when charge is at least 0.35 and at least one pending pulse exists. It suppresses all current incoming pulses, consumes 0.35 charge, and incurs a cost. Charge regenerates by 0.08 each transition up to one. A shield action with no incoming pulse is a free wait; one with insufficient charge cannot block a pulse.

The 16 observation columns are time; the three own register bits; the three own target bits; the three other register bits; the three pending incoming pulse bits; own shield charge; own audit timer; and other audit timer. Time and timers are normalized. The first incoming pulse channel is structurally zero because wiring addresses only bits one and two. Targets, shield charge, and audit status details are otherwise local: the other target is not revealed, and snapshot-match and audit-dirty flags are not directly observed. The acting role can infer those audit flags from its own observation history. Incoming pulses and both registers are public to their relevant displayed roles.

## Incentives, delay, and reward bounds

A completed successful audit pays 0.70; a completed failed audit pays -0.18. A bit flip costs 0.020, rotation costs 0.010, an effective shield costs 0.015, and launching an audit costs 0.015. The action alternatives make these costs mutually exclusive for one role. Rewards therefore lie within [-0.20,0.70] before the explicit final clip to [-1,1]. These are source-derived bounds, not measured outcomes.

A pulse has a one-transition actuation delay, and an audit requires stability for two subsequent transitions. Shielding can create a temporary protected interval, but its stock replenishes more slowly than the charge consumed by repeatedly blocking pulses. Targets are private, so an action that changes the other register can be helpful or harmful depending on unobserved intent. Timing and restraint may matter; their learning effectiveness is unknown.

## Relationship and limits

This relates to generic coupled finite-state control, binary switching networks, delayed interference, and temporal stability checks. No external puzzle, benchmark source, parent candidate, or score was read. The source does not assert novelty, nonisomorphism, duplicate status, mechanism eligibility, or compiler success. Timing may be easier or harder depending on the paired policy, and generic waiting could be sufficient during some intervals. The 32nd transition returns the terminal state without resetting registers or resolving audits that would finish later.
