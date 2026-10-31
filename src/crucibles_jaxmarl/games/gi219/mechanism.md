# Festival Kitchens

Status: AUTHORED_NOT_EXECUTED. Parentless initial blind source, attempt_00, train only. Requested author gpt-6-astra / max; measured provider identity, tokens, cost and reasoning telemetry are unavailable.

Native dimensions are A=16, O=909, U=22, H=384. This source is an original proposal, not a claim of runtime compatibility, solvability, learned behavior or transfer.

## Equipment, stock and physical work

The 19 by 19 workshop square contains four fixed equipment stations, four service counters, a 24-cell conveyor ring with four one-cell spurs, and central finite fuel/washwater supplies. Four actors belong to each kitchen. Initial tools are chef knife, insulated porter grip, wash brush and bellows; tools can be swapped with an adjacent actor. Every actor moves and can carry one actual item. Staff cannot occupy station or belt cells.

Every pantry supplies its own flavour, index plus one, from stock eight. It replenishes one unit every twenty old-clock ticks, capped at eight. Food can be chopped only at station 0, steamed only at station 1 using one fuel, garnished only at station 2 and plated only at station 3 using one clean tray. These stages use a chef tool. A kitchen's workers may walk to another station or pass food to its crew; the source does not bind a worker permanently to their home equipment.

Steaming and reheating set warmth to 64 before that tick's cooling. Carried hot food cools by one each step, except tool 2 skips cooling on odd old-clock ticks; belt food always cools by one. Steamed, garnished and plated stages expire at zero. Expired unplated food disappears as waste. Expired or deliberately discarded plated food becomes a dirty tray instead, preserving its reusable tray.

Four trays begin split between the sink rack at station 2 and plating rack at station 3. A service leaves its dirty tray in the server's hands. A brush user can wash it at station 2 for one shared water and carry the resulting clean tray to station 3, return it to the rack, or pass it through hands/belt. The central fuel pile supplies carried fuel; bellows users feed it into station 1. Fuel and washwater renew slowly at the declared fixed rates. Preparation events do not receive completed-order reward.

## Queue routing and information

Conveyor positions are native cells. Successors follow the clockwise ring, except each corner's route=1 sends an item to its adjacent spur and holds it there. Switching back to zero bypasses that spur on the ring and lets a spur item re-enter the corner. Items move at most one grid edge in each step's conveyor phase after actor interactions. Movement requires a destination empty in the pre-belt-phase occupancy; items cannot follow into just-vacated cells. Competing sources use lowest source-cell ID. These rules allow genuine queues and deadlock under complete occupancy, recoverable by taking an item or switching a spur.

Each kitchen initially receives a uniformly sampled flavour 1-4 using the reset key. Its public service clock starts at 96. A matching warm plated delivery changes the requested flavour cyclically, resets the clock, and produces a dirty tray. Deadline expiry changes the order cyclically and applies a cost without awarding service. Offering a mismatched plate leaves it held and still cooling. The policy legal mask exposes that a carried plate can be offered at a nearby counter, but never reveals whether a hidden foreign order matches.

Only a kitchen's four members directly observe its live flavour. Any actor can tell all adjacent actors their current home order. The listener's notebook stores a snapshot and a lifetime initialized to 12 during interaction and then decremented at step end; own notes remain live at lifetime 12. The stale-report possibility matters when another crew serves or a deadline passes. No cross-kitchen radio, global live order array or privileged browser decision rule is supplied.

A rush can take the last boiler fuel, tray or open belt slot while delaying another kitchen. A crew can keep a hot plate in insulated hands, take a longer reheating route, divert freight into a spur to unblock the ring, or walk a clean tray back to support the next order. Those options change concrete inventories, item warmth and queues instead of independent work counters.

## Native action dictionary

0. Wait; recover two energy.
1. Move north.
2. Move east.
3. Move south.
4. Move west.
5. Take local pantry's raw flavour into empty hands.
6. Process carried food at its next equipment station, chef tool.
7. Place carried item onto nearest empty adjacent conveyor cell.
8. Pick item from nearest occupied adjacent conveyor cell.
9. Hand carried item to lowest-ID adjacent empty-handed actor.
10. Wash carried dirty tray at station 2, brush and shared water.
11. Take clean tray from station 2 rack.
12. Return carried clean tray to station 3 plating rack.
13. Take one fuel from the central fuel pile.
14. Stoke station 1 with carried fuel, bellows tool.
15. Toggle adjacent junction between bypass and holding spur.
16. Offer plated warm dish at adjacent service counter.
17. Discard food; preserve its tray as dirty when plated.
18. Tell adjacent actors the current home-kitchen order.
19. Exchange tools with lowest-ID adjacent actor.
20. Rest; recover five energy.
21. Reheat cooked or later food at station 1, chef tool and fuel.

## Observation, terminal and human-control contract

The map, all physical staff cargo, stations, belt occupancy/direction, fuel/water levels, service clocks, public events and reward components are public. Current order flavour is directly visible only to the four actors belonging to that kitchen: order_visible is the kitchen-equality mask. Each observation adds only that actor's own notes and note_life rows. Speaking transfers the speaker's current home order to adjacent actors; foreign notes expire and can disagree with a changed live order. Legal serving masks do not test hidden flavour equality. A full-state/debug view must be labelled separately.

Observations prepend the explicit self fields in manifest order, then flatten public_keys in their literal source order. Every column is labelled; all values are float32 and bounded by [-5, 384]. Native legal_actions is bool[A,U]; the renderer must not display hidden order/granary attributes to another role. Full-state debug views are separate from role views.

This finite world has no natural joint terminal. H=384 is a truncation window, not success. A nonterminal call increments int32 t once; terminated remains false; truncated=(t>=H)&~terminated; done is their disjunction. The absorbing raw branch returns the identical state and observation, zero float32 rewards, all dones true and unchanged causes. There is no raw autoreset.

All integer design fields use int32 and all flags bool; the supplied uint32 reset key remains external trajectory provenance. Stochastic reset variables use only that key; subsequent rules are deterministic and the step key is unused. No imported benchmark implementation, upstream game source, policy, asset or training baseline is embedded.

Human and AI controls supply the same scalar int32 actor components to the same raw_step. Actor IDs and crew/farm/settlement attributes support selecting one or multiple actors. Scene metadata supplies actual positions, attributes, visibility and actions; it does not implement a browser, camera UI, animation, JaxMARL adapter or runtime integration.

Movement is simultaneous and conservative: proposals entering another actor pre-step cell, any contested destination, a blocked tile or an outer boundary fail. Swaps and following into a just-vacated cell fail. Movement masks permit attempted blocked directions; failed movement leaves position unchanged. Interactions then resolve by ascending actor ID, with resource checks repeated against the current carried state. A mask-valid interaction can become a no-op after a lower-ID actor consumes the object or a neighbor moves. Nearest eligible object and then lowest index resolve spatial ties as defined in source. Invalid/out-of-range actions become wait; only wait remains legal after done.

## Reward allocation and analytic bounds

At most 16 actors can serve per step, so any actor receives at most 16*0.5 shared, 16*1.5 recipient-kitchen and 2 personal reward: 34. A deliberately loose waste upper count is 16 discards + 16 carried expiries + 28 belt expiries = 60, yielding at worst -3 shared waste, -0.4 wrong service, -0.3 local deadline and -0.04 effort/resource cost. Therefore [-5,40] encloses the native reward. Stage, report, utility and conveyor values are bounded by 64 or less, service clocks by 96 and native t by 384; reward components fit the declared observation bounds [-5,384].

- Component 0: Shared service: +0.5 to every actor per actual matching warm plated delivery.
- Component 1: Recipient kitchen: +1.5 to every member of the counter's kitchen per completed order.
- Component 2: Serving actor: +2.0 per completed order.
- Component 3: Effort/resources: -0.01 per move or non-rest interaction attempt, and -0.03 per wash, reheat or steaming event.
- Component 4: Quality/timing: -0.4 to the actor for an unmatched or cold offered plate, -0.05 to all per wasted food, and -0.3 to each member of a kitchen whose deadline expired.

## Literal hypotheses and evidence limits

The first script prepares a chopped item and proposes an adjacent chef-to-porter handoff, followed by the porter's walk to a spur and belt placement. The second script takes a clean tray from the sink rack, hands it to a porter and carries it to another belt spur while a chef prepares food. The paired suffixes differ only in actor 0 giving the chopped item versus waiting after the same take/chop/move prefix. Four subsequent common porter actions are intended to expose the resulting carried-item and conveyor opportunity difference. No random order matching, completed service or successful replay is assumed.

The paired suffixes have equal length 5 and differ in exactly one component: suffix step 0, actor 0. Prefixes and suffixes are literal JSON joint-action arrays. Both scripts and the pair have seed 0 selected without any trial. Their intention is a future counterfactual inspection, not an observed outcome.

Declared invariants are source obligations: actor positions are unique under conservative movement; positions/clocks and the listed inventory, resource and progress arrays have finite bounds. No conservation claim is made across extraction, growth, expiry, meals, rain, overflow, damage or explicit resource spending. Static checks cannot establish these obligations at runtime.

## Literal source evidence

- candidate.py:L13-L46: reset; exact source bytes are bound by SHA-256 34219f5e5df7753805e75886060e43fd56c8e5545eca0454601419a568d544cb.
- candidate.py:L48-L49: distances; exact source bytes are bound by SHA-256 34219f5e5df7753805e75886060e43fd56c8e5545eca0454601419a568d544cb.
- candidate.py:L51-L55: successors; exact source bytes are bound by SHA-256 34219f5e5df7753805e75886060e43fd56c8e5545eca0454601419a568d544cb.
- candidate.py:L57-L60: observe; exact source bytes are bound by SHA-256 34219f5e5df7753805e75886060e43fd56c8e5545eca0454601419a568d544cb.
- candidate.py:L62-L84: legal_actions; exact source bytes are bound by SHA-256 34219f5e5df7753805e75886060e43fd56c8e5545eca0454601419a568d544cb.
- candidate.py:L86-L171: interact; exact source bytes are bound by SHA-256 34219f5e5df7753805e75886060e43fd56c8e5545eca0454601419a568d544cb.
- candidate.py:L173-L236: advance; exact source bytes are bound by SHA-256 34219f5e5df7753805e75886060e43fd56c8e5545eca0454601419a568d544cb.
- candidate.py:L238-L240: raw_step; exact source bytes are bound by SHA-256 34219f5e5df7753805e75886060e43fd56c8e5545eca0454601419a568d544cb.

## Actual inputs and provenance

author_input_freeze.json lists all 25 actually-read current allowed inputs: two controlling authorities and 23 payloads. The four current designs and requests were read for this disjoint batch. Every byte count and SHA was checked against the current authority. The permitted PPO implementation snapshot was not read. Official README snapshots were byte-read for identity; only the recorded excerpt lines were displayed for API/design context.

Reference ideas: Craftax provides context for bounded persistent exploration; JaxMARL describes per-agent dictionaries and joint done as an integration reference; Pgx separates state, legal masks and supplied randomness; PureJaxRL is algorithm implementation context, not a game library or a substitute learner. No compatibility or performance claim is transferred from these references.

Only author-time stdlib AST/data/hash checks and the supplied trusted whitelist, manifest and scene validators were run. No candidate import or method, JAX/NumPy, JIT/vmap, rollout, render, training, seed search, external model, network or local model was executed. No prior source, feedback, diagnosis, selection or heldout evidence was accessed. Numerical, gameplay, animation, human-control, learning, throughput and transfer checks remain unperformed.
