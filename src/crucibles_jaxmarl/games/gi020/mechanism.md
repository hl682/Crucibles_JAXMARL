# Bilateral escrow with delayed authorization

Candidate: astra-d2-00020. This is a parentless source proposal with no supplied predecessor, score, or feedback.

Action 0 waits. Actions 1 and 2 deposit one unit of cash into one of two public contract channels, and require an empty escrow plus sufficient cash. Action 3 seals the current deposit. Action 4 revokes an existing deposit, recovering 0.7 cash. A role cannot change the channel while its deposit remains open.

The causal dependency is a shared settlement requiring both old-state deposits to name the same channel, both old-state seals to be present, both deposit ages to be at least two, and neither agent to revoke in the current transition. Newly authorizing a contract cannot settle it in that same transition. Settlement clears both deposits and returns each principal plus that agent's channel-dependent dividend. Revocation can prevent an otherwise ready joint settlement and penalizes a counterpart that had already sealed. Both agents therefore control the availability and timing of the other's payout, as well as their own persistent cash and commitment state.

Each agent has a private, fixed preferred channel. A settlement yields reward 0.80 on the preferred channel and 0.35 on the other channel. The preferred channel is not directly shown to the other agent; settlement rewards become public observations. Conflicting preferences create a coordination and distribution question without an imposed desired winner. Cash regenerates by 0.06 each transition and is capped at six. Deposits reduce cash immediately. Opening, sealing and revocation have small immediate reward costs, so waiting, coordinating and abandoning a stalled contract have different consequences.

The sixteen observation columns are time fraction, role sign, own cash, other cash, own channel, other channel, own seal, other seal, own escrow age, other age, own preference, whether the current own channel is preferred, own last settlement reward, other last settlement reward, own last revocation and other last revocation. Cash, channels, ages and preferences are normalized in the source. No final-test or external task information is used.

Reward equals the settlement dividend less opening, authorization, revocation and interrupted-commitment costs, with a final clip to [-1, 1]. The states underlying these terms are bounded. Both roles terminate together at transition 32; no terminal reset is performed.

This uses familiar generic escrow, commitment and coordination-game ideas. Its proposed interaction is the conjunction of individually revocable persistent deposits, separately delayed authorization, private channel values and a shared release event. The description makes no claim about nonisomorphism, pool qualification, compiler behavior or learning effectiveness. The author has not executed this environment; isolated validation belongs to the experimenter.
