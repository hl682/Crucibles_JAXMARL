# astra-d2-00008: delayed counterflow delivery and receiver gates

This is an original, parentless source proposal. Its source has not been executed or validated by the author. It is related to generic delayed transport, signaling, and receiver-controlled coordination mechanisms; novelty, nonisomorphism, eligibility, and learning effectiveness are not asserted.

## Persistent causal structure

Each agent owns a three-position outbound pipeline, energy, a receiver gate, and a fixed private preference for one of two package types. An agent's outbound pipeline terminates at the other agent. An arriving package succeeds only if its type matches the receiver's gate at the beginning of the step. The sender therefore commits before delivery, while the recipient controls acceptance later. Successful receipt also replenishes the recipient's future sending capacity.

Actions are 0 wait, 1 launch type 0, 2 launch type 1, 3 set next-step gate to 0, and 4 set next-step gate to 1. Launching requires one energy unit; an unaffordable launch produces no package and no launch cost. Every action remains legal. Each step supplies 0.2 energy, and a successful incoming delivery supplies another 0.5, with capacity three.

A new package occupies the last queue position. It is delivered on the third subsequent call after insertion. Arrivals use the old gate; a tuning action changes the gate for the following call. New launches use pre-step energy, so an arrival cannot fund a launch chosen in the same call. These timing rules create advance commitment and endogenous reciprocal capacity.

## Information and incentives

Agents see their own preference, full outbound queue, current public gates and energy levels, the type of the package arriving next at their own receiver, and only the occupancy of the following incoming position. They do not directly see the other agent's preference or the newest incoming queue position. The observed gate can serve as an imperfect signal of the recipient's preference.

A matched arrival has value 1 when preferred and 0.6 otherwise. Each reward is 0.5 times own incoming value plus 0.4 times the other agent's incoming value, minus 0.05 for a successful launch and 0.05 for an unmatched incoming package. Thus the reward lies in [-0.1, 0.9], within the declared [-1, 1] envelope. Sending can help the partner while also paying the sender after the delay; gate choices can accept or waste the partner's committed investment. Terminal queued packages and all other state are retained.

## Observation columns

0 elapsed time / 32; 1 own energy / 3; 2 own gate; 3 own private preference; 4 other gate; 5 other energy / 3; 6-8 own queue positions, encoded as 0 empty, 0.5 type 0, 1 type 1; 9 next incoming package with the same encoding; 10 occupancy of the subsequent incoming position; 11 most recent incoming value; 12 most recent outgoing value; 13 most recent unmatched incoming indicator; 14 current launch affordability; 15 agent index.

## Scope and possible failure modes

The mechanism may admit simple repeated conventions, inefficient gate oscillations, or withholding. Hidden preference is not a claim of strategic complexity. Rewards, the bilateral dependence, and the stated timing require trusted validation; no benchmark or outcome evidence was consulted.