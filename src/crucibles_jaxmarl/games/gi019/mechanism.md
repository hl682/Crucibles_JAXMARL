# astra-d2-00019: consensus switch with irreversible route commitments

This is a parentless proposal using generic consensus, shared routing controls
and irrevocable commitments. It is not derived from a supplied parent or a
retrieved benchmark. Its relationship to these familiar abstractions is
intentional, and it carries no claim of mechanism uniqueness or eligibility.

Each role has an entrance lane equal to its role index, a private binary desired
destination, an energy reserve and at most one shipment. A shared binary switch
determines route = (role index + switch) modulo two at commitment. Once committed,
a shipment retains that route even if the shared switch changes during transit.
A committed shipment arrives after its countdown subsequently decreases from two
to one and then to zero. Targets are redrawn from the explicit step key when a
shipment arrives or expires; a redraw may equal the previous target.

Actions are 0 wait, 1 load a shipment when idle, 2 commit a previously loaded
shipment to the current route, 3 cast a switch-zero ballot, and 4 cast a
switch-one ballot. Loading costs 0.12 energy, commitment costs 0.08 and a ballot
costs 0.03. Insufficient energy makes the special effect ineffective. Energy
refills by 0.03 per transition and is capped at one. All actions remain legal.

Ballots persist until replaced or until their age reaches four. When both valid
ballots match, they are cleared and their value becomes a pending switch command.
That command applies at the beginning of the next transition, before any route
commitments on that transition. New ballots can then be cast for a later command.
Only agreement produces a command; one agent can prevent or delay a change.
This creates a delay between requesting, agreeing, applying and irreversibly
using shared infrastructure. Private targets can make demands conflict, while
public ballots can also convey desired configurations.

A shipment age starts at one on loading. An active shipment expires when its age
reaches eight unless it arrives on the same transition, in which case arrival
takes precedence. There is no automatic dispatch, no ability to reroute a
committed shipment, and no carry-over of the old shipment after completion.
The previous latched route remains observable as history while idle.

Reward is +0.80 for a correct arrival, -0.40 for an incorrect arrival, -0.45 for
expiration, -0.05 for loading, -0.015 for casting a ballot, and -0.01 for a role
whose shipment was active at the beginning of the transition. It is explicitly
clipped to [-1, 1]. The two roles affect each other's feasible route changes
through persistent votes and the shared switch, even though arrivals have
individual rewards. No assumption of coordination or learning effectiveness
is made.

Observation columns 0 through 15 are: time/32; own private target; current switch;
pending switch or -1; own stage/2; partner stage/2; own shipment age/8;
partner shipment age/8; own energy; own most recently latched route; own ballot
or -1; partner ballot or -1; own ballot age/4; partner ballot age/4;
own transit countdown/2; and own previous outcome (1 correct, -1 wrong,
-0.50 expired, 0 no outcome). Stages are 0 idle, 1 loaded and 2 committed.
Partner target, energy, latched route and previous outcome are hidden.

The intended interface is two agents, 16 observations each, five actions and
32 transitions. Terminal state and observation are preserved. This source has
author-side syntax parsing only. Isolated validation, duplicate review,
mechanism qualification and performance assessment remain unassessed.
