# astra-d2-00012 — Escrow-backed exchange with a delivery pipeline

Status: authored source only. Isolated validation and duplicate/mechanism review are pending.

## Causal transition

Each agent manufactures a consumable good, owns money, and can post one unit of money as a public purchase escrow. Actions are 0 wait, 1 manufacture, 2 post escrow, 3 dispatch a good against the other agent's available escrow, and 4 cancel one's own unmatched escrow. Unavailable actions have no effect and remain legal.

Posting and cancellation are resolved before dispatch. A dispatch consumes the sender's stock and the buyer's free escrow. The captured payment travels with that dispatch: it cannot subsequently be cancelled. Two subsequent transitions later the recipient consumes the good and the sender receives the locked payment. The two-slot queue makes outstanding deliveries different from immediately spendable money. A payment arriving during a transition becomes spendable in the next decision.

The total of liquid money, free escrow, and money represented by all queued shipments is conserved at four units. Stock is capped at four. A sender can earn the means to buy by supplying the partner, while a buyer can create or withdraw an opportunity to supply. Both agents directly affect the other's opportunity set and delayed consumption.

## Information

An agent observes its stock and liquid money, both free escrows, both directions of the delivery pipeline, its private consumption taste and production cost, the clock, pre-action manufacture/post/dispatch feasibility, and its latest receipt. It does not observe the partner's liquid balance, inventory, taste, or production cost.

Observation columns 0–15 are time/32, own stock/4, own money/4, own escrow, peer escrow, own far shipment, own near shipment, peer far shipment, peer near shipment, taste, make cost/0.12, remaining time/32, manufacture feasibility, posting feasibility, dispatch feasibility, and last received quantity.

## Incentives and reward bound

A consumed good pays 0.6 + 0.2 times the recipient's fixed taste. Successful manufacture costs a private amount in [0.08, 0.12); posting costs 0.015 and cancellation costs 0.01. These three costly actions are mutually exclusive. At most one good arrives per role per step, so rewards lie within [-0.12, 0.8]. Money itself has no terminal utility. Pending deliveries and unused resources remain in the terminal state; the episode does not pay a post-horizon salvage value.

This mechanism is related to generic bilateral trade, escrow commitment, and delayed settlement. Its proposed causal structure is resource conservation with cancellable unmatched offers and irrevocable matched payments. This description does not establish novelty or separation from an external candidate pool.

## Interface and provenance

Two agents, sixteen float32 observations each, five always-legal actions, and a 32-transition episode. The reset key draws private tastes and costs; subsequent exchange is deterministic conditional on actions. This is a parentless training-context proposal, with no supplied parent, feedback capsule, or outcome information.
