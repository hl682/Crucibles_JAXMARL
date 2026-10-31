# Reciprocal finishing pipeline
Status: authored proposal, not executed or validated.

Each agent fabricates its own unfinished goods into a capacity-three input queue, then forwards them into the partner's capacity-three finishing queue. Only the partner can finish those goods. Finishing earns the operator a 0.2 commission and the originating producer 0.65. Fabrication costs 0.04. The finishing operator may instead purge a good at cost 0.03, destroying the producer's potential return.

Actions use old queues and heat. A newly fabricated good cannot be forwarded that step, and a newly forwarded good cannot be completed that step. Sending checks old receiver capacity conservatively, so a concurrent completion does not permit sending into a formerly full queue. Productive operations generate heat; excessive heat prevents those operations, while cooling reduces it. Purging remains possible when hot. These rules create bilateral delayed dependence and competing demands on each operator's own action time.

Both agents observe both queues, both heat values, own recent action, both recent completion indicators, time, and role. Hidden variables are absent. All actions are legal; unavailable operations have no queue effect. Rewards lie between -0.07 and 0.85 even under a loose bound, hence inside [-1,1]. Both agents can alter persistent queues and partner income.

The mechanism is related to generic production networks, hold-up, and reciprocal service exchange. It is not claimed novel in a formal sense or nonisomorphic to existing games. No source execution, compiler success, eligibility, or learning-effectiveness claim is provided. Initial raw queues are sampled with the supplied key; termination preserves state at t >= 32.
