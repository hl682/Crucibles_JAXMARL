# astra-d2-00015 — Private sensing, a colliding broadcast channel, and delayed claims

Status: authored source only. Isolated validation and duplicate/mechanism review are pending.

## Causal transition

A hidden binary regime flips with probability 0.12 per transition. Actions are 0 recharge, 1 scan, 2 broadcast one's stored sample, 3 deploy a claim on regime 0, and 4 deploy a claim on regime 1. Scanning or deployment consumes one battery unit and requires a positive balance; recharging adds one unit up to capacity three. Other ineffective actions remain legal.

A scan observes the newly evolved regime with independent error probability 0.15. The private sample persists with its age. Broadcasting sends the stored sample, not a fresh scan, and publishes its true age. A single broadcaster succeeds, while simultaneous valid broadcasters collide and neither changes the published reports. Agents observe how many valid broadcasts were attempted on the latest transition.

A deployment is irrevocable and settles two subsequent transitions later against the regime then prevailing. If both agents settle claims on the same regime simultaneously, a correct claim receives half of the uncongested success payment. Thus publishing useful information can help the peer while also attracting a competing claim. Channel contention and payoff congestion are separate interactions.

## Information

The regime, peer battery, peer private sample, and peer pending claims are hidden. Each role observes its battery, private sample with a validity bit and age, both published reports with validity and ages, its two claim slots, recent channel contention, own latest gross settlement, and time. The public age is the age of the measurement itself, so rebroadcasting cannot present an old measurement as new.

Observation columns 0–15 are time/32, battery/3, private sample value, private validity, private age/32, own published value, own publication validity, own publication age/32, peer published value, peer publication validity, peer publication age/32, own far claim encoding, own near claim encoding, channel attempts/2, own last gross settlement, and remaining time/32. Claim encoding is 0 for empty, 0.5 for regime 0, and 1 for regime 1. Invalid samples use value zero together with a zero validity bit.

## Incentives and reward bound

A correct claim pays 0.55 alone or 0.275 under simultaneous same-target settlement. A wrong claim pays -0.25. An effective scan costs 0.025, a deployment costs 0.015, and a valid broadcast attempt costs 0.005 even when it collides. These three actions are mutually exclusive, so rewards lie within [-0.275, 0.55]. The battery has no terminal value and queued claims receive no settlement after the 32-transition horizon.

This is related to generic decentralized sensing, communication congestion, and costly prediction. Its proposed dependencies combine stale private measurements, honest-aged public reports, collision-mediated information availability, and commitments whose eventual payoff depends on the other agent's synchronized commitments. No novelty, compiler, pool eligibility, or learning-effectiveness claim is made.

## Interface and provenance

Two agents, sixteen float32 observations each, five always-legal actions, and a 32-transition episode. Reset uses its explicit key for the hidden regime. Each step splits its explicit key for the regime transition and measurement errors. This parentless training-context proposal used no parent source, experiment outcomes, or shared feedback capsule.
