# astra-d2-00046: Reconfigurable packet loops with cross-role reception

Status: original parentless source proposal; not executed or isolated-validated by the author.

There are four persistent transport nodes, two packet colors, and two role-controlled switches. Role 0 injects color 0 at node 0 and needs role 1 to receive it at node 2. Role 1 injects color 1 at node 2 and needs role 0 to receive it at node 0. Actions are wait, inject, flip the local switch, receive the other color at the local junction, and flush both colors at that junction. Stored local energy gates active actions. All five action indices remain legal, including ineffective unaffordable requests.

Packets advance one edge per step. Node 1 always leads to node 2 and node 3 always leads to node 0. The switch at node 0 sends packets to node 1 or node 3; the switch at node 2 sends packets to node 3 or node 1. Switch changes become effective on the following transition. This can create local loops that retain a packet until reconfiguration. Reception operates on packets present before movement, so arriving packets cannot be received during the same transition.

Colors share each node's capacity. Routed overflow is discarded proportionally across colors, then every transported quantity loses four percent. Flushing removes all colors at the local junction. Delivery removes only the opposite role's color, pays its owner, and supplies a smaller service benefit to the receiver. Both injection-induced congestion and destructive flushing can change the other role's future delivery opportunities. Loop selection, reciprocal receiving, packet aging, and shared capacity produce delays that persist beyond the immediate actions.

Both roles see the full transport state. The 16 observation columns are time, own and other energy, own and other switch, all four nodes of the role's own color, all four nodes of the other color, own and other previous delivered quantities, and a bounded aggregate previous loss indicator. Node order stays absolute; rows orient packet color and switch ownership relative to the observer. Role 0's local junction is node 0; role 1's is node 2. Randomness chooses only the initial two switch settings from the explicit key.

Reward is tanh of own delivery benefit, cross-delivery service benefit, and the negative energy-operation and own-packet-loss terms, so reward stays in [-1, 1]. The observation and queued packets after the 32nd transition are retained; packets still circulating at that point generate no later credit.

The construction is related to queueing, switching networks, store-and-forward transport, and cooperative delivery. It proposes explicit action-controlled topology and color-coupled capacity with receiver-side completion. This description does not establish novelty against all programs, nonisomorphism, operational eligibility, compiler success, or learning effectiveness. No benchmark, parent source, or feedback outcome was accessed.
