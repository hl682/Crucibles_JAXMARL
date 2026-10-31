# astra-d2-00016: sequential signature escrow

This is an original source proposal based on generic voluntary public production,
escrow and staged ratification. It has no supplied parent. Similarity to those
generic mechanisms is intentional; distinctness from other proposals or any
benchmark has not been established.

Each role has a private liquid wallet, private settlement value and private work
efficiency. Bonds, project progress, project age, signatures and previous actions
are public. Both roles must hold at least 0.20 in escrow before either can do
productive work. An eligible signature is persistent, but settlement requires
both signatures to have already existed at the beginning of a later transition.
A last-step withdrawal therefore still prevents settlement and clears progress.

Actions are 0 wait, 1 deposit up to 0.24 into an escrow cap of 0.60, 2 spend up to
0.12 on work when both are bonded, 3 sign a project whose previous progress is at
least 0.75, and 4 withdraw one's bond. Withdrawal returns half the actor's bond
and transfers the other half to the partner; it invalidates all signatures and
progress. Eight active transitions without settlement expire the project,
returning half each remaining bond and destroying the rest. Settlement returns
both bonds and credits 0.20 times private value into each wallet. Wallets refill
by 0.03 per transition and are capped at 1.0, so overflow is discarded.

Work contributes through both private efficiencies to shared progress, with 0.96
retention each transition. Values, efficiencies and starting wallets are drawn
only at reset. Other transitions are deterministic given actions. All actions
remain legal; work, signing and withdrawal can have no special effect when their
prerequisites are absent. This permits delayed commitment, unrewarded effort,
asymmetric costs, and unsuccessful coordination.

The reward before clipping is a settlement payoff of
0.65 * (0.40 + 0.60 * own_value), plus 0.30 times net forfeiture received,
minus 0.25 times work spending, 0.03 times deposits and 0.35 times escrow held
when it expires. Every reward is explicitly clipped to [-1, 1]. Liquid cash
flows and scalar utility are distinct, as described above. No positive outcome
is assumed.

Observation columns 0 through 15 are: time/32; own wallet; own bond/0.60;
partner bond/0.60; progress; own signature; partner signature; age/8; own value;
own efficiency/1.20; previous settlement flag; previous breach flag;
own previous action/4; partner previous action/4; both sufficiently bonded flag;
and progress ready for signature flag. Partner wallet, value and efficiency
are hidden. Each role can alter the partner's feasible work, project lifetime,
settlement, or forfeiture receipts through persistent state.

The intended interface is two agents, 16 float32 observations per agent, five
actions and a 32-transition horizon. The terminal transition retains its state
and observation. This artifact is source authoring only. Syntax parsing is the
only author-side check; isolated engineering validation, duplicate review,
mechanism qualification and learning outcomes remain unassessed.
