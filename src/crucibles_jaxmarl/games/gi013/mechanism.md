# astra-d2-00013 — Jointly gated transport of delayed pollution

Status: authored source only. Isolated validation and duplicate/mechanism review are pending.

## Causal transition

Each agent owns a replenishing water reservoir, persistent contamination, a depletable filter, an outgoing plume, and a persistent gate. Actions are 0 wait, 1 extract, 2 clean local contamination, 3 replenish local filter capacity, and 4 toggle one's gate.

Extraction consumes up to 0.35 units of water and creates a plume equal to 0.8 times that volume. The plume reaches a reservoir on the next transition. The current actions first update gates. If both gates are open, each outgoing plume reaches the other reservoir; otherwise both plumes return to their source reservoir. Transport dissipates fifteen percent of each plume. Either role can therefore close the connection, jointly changing the location of both delayed externalities.

Restored filter capacity is available to absorb that transition's incoming plume and is consumed by absorption. Unfiltered pollution adds to persistent contamination after natural decay; active cleaning reduces that contamination. Extraction quality then depends on the resulting local contamination. Independent rain replenishes each reservoir after extraction.

## Information

Each role knows its own water, contamination, filter, both outgoing plumes, both gates, its last yield, rain, and unfiltered inflow. Public mean water and mean contamination additionally disclose aggregate conditions. The expected incoming-plume feature uses the currently observed gate configuration; simultaneous toggles can change the actual route. There is no hidden agent type, and aggregate water/contamination plus the local value make the corresponding peer value inferable. Peer filter capacity is not directly reported.

Observation columns 0–15 are time/32, remaining time/32, local water, local contamination, local filter/0.75, own plume/0.28, peer plume/0.28, own gate, peer gate, last normalized useful yield, last rain/0.11, last unfiltered inflow/0.28, anticipated routed plume/0.28, mean water, mean contamination, and current quality factor.

## Incentives and reward bound

Extraction pays 0.6 times normalized usable yield. Yield is extracted volume/0.35 multiplied by a quality factor in [0.15, 1]. Cleaning costs 0.05, restoring filters costs 0.055, and toggling costs 0.01. Actions are exclusive, giving rewards in [-0.055, 0.6]. There is no terminal payment for clean water, remaining filter capacity, or outstanding plumes.

The familiar relation is a pollution externality and renewable-resource game. The proposed dependencies combine controllable routing, a shared connection requiring both gates, one-step pollutant transport, and consumable local absorption. This is a mechanism hypothesis rather than evidence of novelty, eligibility, or a learning advantage.

## Interface and provenance

Two agents, sixteen float32 observations each, five always-legal actions, and a 32-transition episode. Reset randomizes initial reservoir levels, contamination, and gates. Explicit step keys supply rainfall. This parentless training-context proposal received no parent source, experiment results, or feedback capsule.
