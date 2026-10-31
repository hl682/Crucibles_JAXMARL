# astra-d2-00010: bilateral credit with aged obligations and productive sanctions

This is an original, parentless source proposal. It has not been executed or validated by the author. It intentionally relates to generic lending, repeated trust, and costly enforcement. It is a small abstract mechanism rather than a financial model; no novelty theorem, nonisomorphism, eligibility, compiler result, or learning advantage is asserted.

## Persistent causal structure

Each agent has a cash balance, privately known productive skill, debt owed to the other agent, a debt age, an expiring credit request, and a temporary production freeze. Both agents can simultaneously occupy borrower and lender roles.

Actions are 0 wait, 1 request credit, 2 lend, 3 repay, and 4 enforce. A request lasts three future opportunities unless renewed or funded. Lending checks the other agent's pre-step request and pre-step debt: it transfers up to 0.55 of the lender's available cash only when the recipient has a live request and effectively no existing debt. The recipient then owes 110 percent of the amount borrowed. Requests created in the same call cannot be funded immediately.

Repayment transfers as much as possible toward the existing obligation, so it can be partial. After transfers, an enforcement action can freeze an unpaid debtor whose pre-step debt age is at least four. A successful sanction sets the debtor's production-freeze clock to three; further enforcement renews it. Full repayment in that call clears the freeze and prevents enforcement against the cleared obligation. A new loan starts its age at zero.

Production is privately determined skill times the square root of post-transfer cash when the updated freeze clock is zero. Half of production is added to cash. Cash capacity is three; overflow is discarded. There is no terminal wealth or unpaid-debt settlement reward. These choices make repayment and enforcement relevant through future productive opportunities rather than through an imposed terminal success label.

## Information and incentives

Balances, obligations, obligation ages, request clocks, and freeze clocks are public. An agent sees its own skill, last income, and net transfer but does not directly observe the other agent's skill or income. Public balance movements may permit partial inference over time.

Reward is current production divided by 0.14, minus 0.05 for effective enforcement and 0.01 for requesting credit. Production is bounded by 0.07 times sqrt(3), so rewards lie within [-0.05, 0.867]. Lending and repayment have no direct reward: they change both agents' cash, future production, and enforceable obligations. All five actions are legal; an ineffective lend, repayment, or enforcement produces no corresponding transfer or enforcement cost.

## Observation columns

0 elapsed time / 32; 1 own cash / 3; 2 other cash / 3; 3 own debt / 0.61; 4 debt owed by other / 0.61; 5 own debt age / 32; 6 other debt age / 32; 7 own request clock / 3; 8 other request clock / 3; 9 own freeze clock / 3; 10 other freeze clock / 3; 11 own private skill / 0.07; 12 own last income / 0.14; 13 own last net transfer / 0.61; 14 other last effective-enforcement indicator; 15 agent index.

## Scope and possible failure modes

The finite horizon, absence of terminal settlement, and private productive capacity can make default, refusal to lend, or ineffective repeated sanctions rational. Positive credit activity is not required for the proposal to be retained. Both directions of lending and enforcement alter persistent bilateral state. Terminal balances and obligations remain present for inspection. Trusted validation remains outstanding.