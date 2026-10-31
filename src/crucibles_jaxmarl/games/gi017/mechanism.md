# astra-d2-00017: relay with persistent local decoding

This parentless proposal uses generic store-and-forward communication, finite
buffers, local symbol transformations and congestion. It is not based on a
supplied game or prior proposal. Similarity to communication and queueing models
is acknowledged; no nonisomorphism or novelty certification is asserted.

Both agents are simultaneously senders to the other agent and recipients from
the other agent. Each has a persistent emitter symbol and a private decoder
offset, both in {0, 1, 2}, and a private desired decoded symbol. A launched symbol
is committed to a flight and later enters the partner's one-slot mailbox.
Receiving a new symbol overwrites an unconsumed old one. Acceptance consumes the
previous mailbox item, while a same-transition arrival can refill that mailbox.

Actions are 0 wait, 1 increment the emitter modulo three, 2 launch the current
emitter when charge is at least 0.25 and no outgoing flight exists, 3 increment
the decoder modulo three, and 4 accept the previous mailbox item when present.
Acceptance is successful exactly when (mail symbol + decoder) modulo three
equals the recipient's private target. After either successful or failed
acceptance, an explicit-key random increment of one or two changes that target.
A mailbox item arriving this transition cannot be accepted until a later step.

Launch costs 0.25 charge. Either local rotation costs up to 0.03 charge through
clipping. Charge replenishes by 0.08 per transition, capped at one. A launched
flight gets countdown two, plus one when both agents launch, plus one when
previous congestion exceeds 0.60. Countdown runs on later transitions until
arrival at its old value of one. Congestion retains 0.70 of its prior value and
adds 0.30 on a simultaneous launch. Thus joint launches affect future delays
as well as current deliveries. No launch cancels an existing flight.

The reward is +0.70 for a matching acceptance, -0.25 for a mismatch, -0.04 for a
launch, -0.08 for a mailbox overflow, and +0.10 when the partner succeeds.
It is explicitly clipped to [-1, 1]. A sender affects the partner's available
symbols and their arrival times; recipient pacing affects the consequences of
those deliveries. Cooperative and conflicting pacing incentives are possible.
No learning benefit or performance target is presumed.

Observation columns 0 through 15 are: time/32; own charge; own emitter/2;
own decoder/2; own target/2; own mailbox symbol/2 or -1 when invalid;
own mailbox validity; own most recently launched symbol/2; own outgoing
countdown/4; partner outgoing countdown/4; partner emitter/2; shared congestion;
previous incoming-delivery flag; own previous success flag; own previous
action/4; and partner previous action/4. The partner's private decoder, target,
charge and mailbox contents are hidden. Outgoing symbol state persists after
arrival but its countdown identifies whether a flight is active.

The intended interface is A2/O16/U5 over 32 transitions, preserving the terminal
state and observation. Only syntax parsing is performed by the author.
Engineering validation, structural duplicate review, mechanism eligibility and
learning outcomes are separate, unassessed steps.
