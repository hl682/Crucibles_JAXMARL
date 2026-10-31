# Candidate astra-d2-00025: Coupled reaction chambers

Status: original parentless source authored under the supplied v2 contract. The author has not executed the candidate, initialized JAX, compiled it, applied the isolated validator, or evaluated learning.

## Causal mechanism

Each agent controls one chamber with feedstock, catalyst, product, and temperature. Reaction occurs autonomously using the state at the start of the transition. Its rate depends jointly on available feedstock, existing catalyst, and a temperature window centered at 0.5. Feed added now and catalyst added now can affect subsequent reactions; product created now cannot be sold until a later transition.

The two chambers exchange heat every step and draw cooling fluid from one regenerating reservoir. Simultaneous requests share the available coolant proportionally. Cooling lowers the requesting chamber's temperature immediately, but creates an exhaust plume that heats the other chamber on the next transition. Thus cooling can protect local production while imposing a delayed thermal externality, and requesting scarce coolant reduces the counterpart's available cooling.

Predicted temperature above 0.98 causes a local shock: the chamber cools to 0.58, feedstock falls to 35 percent, product to 20 percent, and catalyst to 50 percent of their pre-shock updated amounts. A shock also imposes a smaller reward loss on the other role. The other agent can affect both future local production and realized rewards through heat diffusion, exhaust, coolant contention, and shocks.

## Actions and information

The five actions are 0 idle, 1 feed, 2 add catalyst, 3 cool, and 4 harvest. Feed adds 0.24 up to capacity one, catalyst adds 0.30 before capacity clipping, and harvest sells at most 0.6 of the product present at the start of the transition. Cooling requests 0.24 units. The shared reservoir regenerates by 0.08 after allocation. All actions remain legal.

The 16 observation columns are time, own feedstock, own product, own temperature, other temperature, own catalyst, shared coolant, own outgoing plume, other outgoing plume, own last reaction, other last reaction, own last shock, other last shock, own last action, other last action, and own last reward. The source normalizes temperatures, plume amounts, reaction amounts, time, and actions. The other chamber's feedstock, catalyst, and product are not directly observed. Public reaction and thermal readings can still reveal information about those stocks.

## Incentives, delay, and reward bounds

Harvest earns 0.65 times the amount sold. Feeding, adding catalyst, and cooling cost 0.030, 0.020, and 0.010. A local shock costs 0.30 and a counterpart shock costs 0.06. Because sale uses old product, that sale is realized before any same-transition shock destroys the remaining updated product.

Harvest is at most 0.6, so positive reward is at most 0.39. Even conservatively adding the largest action cost and both shock penalties gives a lower bound of -0.39. The source additionally clips every reward to [-1,1]. These bounds follow from the written transition equations and are not validation results.

Investment delays, stored product, catalyst decay, temperature feedback, and delayed exhaust allow an action to matter across several future steps. A locally useful cooling action can hurt the other reactor, while avoiding a shared shock penalty can favor restraint. None of these possibilities establishes a learning advantage. Time advances exactly once per transition and termination at 32 leaves the final physical state intact without liquidation or reset.

## Relationship and limits

The proposal draws on generic coupled control, production inventories, congestion over a shared resource, and delayed negative externalities. It is an abstract numerical mechanism, not a physical chemistry model. No benchmark or parent source was consulted, and no novelty, nonisomorphism, or pool eligibility is asserted. Passive cooling and alternative operating cycles may make some interventions unnecessary in particular trajectories; this remains an empirical matter for the designated experimenter.
