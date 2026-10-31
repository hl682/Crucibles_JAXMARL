# Mycelium Valley

Status: AUTHORED_NOT_EXECUTED. Parentless initial blind source, attempt_00, train only. Requested author gpt-6-astra / max; measured provider identity, tokens, cost and reasoning telemetry are unavailable.

Native dimensions are A=32, O=1529, U=26, H=640. This source is an original proposal, not a claim of runtime compatibility, solvability, learned behavior or transfer.

## Farms, tools and renewable work

The 25 by 21 pastoral grid contains eight farms in a cycle, two plots per farm, eight root-bed midpoints, two wells, two compost facilities, a nursery and eight pantry destinations. Thirty-two gardeners begin as four complementary tool users per farm. Tool 1 irrigates; tool 2 prunes diseased crops or root beds; tool 3 plants, grafts, connects roots or saves seeds; tool 4 carries a larger total load and stirs/spreads compost. Tools can be exchanged when both loads fit eight, changing both actors' future available work.

Bag columns are water, seeds, compost, fruit and waste, each capped at four. Total capacity is eight, or twelve with a compost cart. Adjacency uses Manhattan distance at most one. Handoffs choose the first compatible carried resource and the lowest-ID eligible adjacent recipient; seed handoffs cannot mix different traits into an already occupied seed bag. Physical inventories, not ownership messages, determine which actor can act next. Wells, bins and the nursery occupy blocked fixture cells; plots and root corridors are walkable.

Each farm's two plots share moisture and nutrient pools capped at eight. Plot p can advance on old-clock (t+p) mod 8 equal to seven if planted, healthy, below growth eight and supplied with one water and nutrient; it consumes one of each. At growth eight, a harvest yields one fruit, or two for productive trait 2, plus one carried waste. Harvest leaves an empty bed. A graft consumes one seed, sets the crop trait and reduces growth by two; planting consumes a seed and starts an empty plot at zero. Saving seed from a developed compatible crop reduces growth by two and creates one seed without harvesting its fruit.

Trait 1 avoids the special productive-crop wet-soil outbreak. Trait 2 yields more fruit but can become infected on its growth phase when soil water is at least six. Trait 3 blocks incoming fungal exposure and a resistant graft clears current disease. Pruning an infected crop clears it while sacrificing two growth. Existing disease can worsen on that plot's phase when its soil has at least four water; severity three destroys the crop and adds one ground detritus. These are explicit costs and temporal tradeoffs, not claims of an optimal planting strategy.

Rain and drying follow an eight-step calendar with a public reset-sampled offset. Rain adds one water to soil and three to each shared well, clipped to capacity; drying subtracts one soil water. A subset of rain ticks replenishes the three seed stocks. Every 32 ticks the second plot of each farm receives one unit of leaf litter if ground detritus has room. Crop loss also supplies detritus. Gardeners collect it and carried harvest waste, transport it to a finite bin, and later take finished compost to another plot's soil. A bin naturally advances every other tick while waste is present; cart-tool stirring accelerates progress, but at most one waste converts to compost per bin per step when progress reaches four and output has space.

## Neighbor transfers and disease

Each root link joins farm e to (e+1) mod eight along the actual x-then-y map corridor, seven or eight cells long. Six links start installed/open, two are uninstalled. A grafter at a missing or cut bed can connect it for a seed and compost; only a never-installed bed earns first-construction reward. Soil-sum imbalance of at least three can launch a packet from the richer endpoint if it has both resources. Exactly one water and one nutrient are removed from the source into the packet. At most one packet occupies a link, its position advances one native cell per step, and arrival adds its actual quantities to the other farm subject to soil caps.

If that launch source is wet (water at least four) and has an infected planted crop, the packet also carries disease. At arrival, nonresistant planted crops sharing the recipient soil can be exposed. A root cut cancels the actual in-flight packet, clears its quantities and exposure, and closes the link until reconnected. It therefore blocks future resources and disease together, at the cost of the discarded resource and lost neighbor benefit. Arrivals and new launches resolve in fixed link order; a link that just arrived waits until a later step before launching again.

Richer soil can fund a neighbor's growth, while that neighbor's demand changes which way a later packet travels. Crop choice, irrigation and compost distribution alter gradients and wetness; pruning or resistant grafting alters transmission. Multiple farms also compete for the same well, nursery and finite compost input/output spaces. These native dependencies make other crews' labor and resource decisions relevant without requiring every actor to interact with all 31 others.

## Reward interests and continuing world

Pantry delivery consumes one real carried fruit. The delivering actor, its home farm and a distinct recipient pantry farm receive separate components; public rewards account for first connection and actual packet arrival. The arrival component deliberately counts infected as well as clean nutrient delivery, so immediate transport benefit does not erase later disease cost. Initial stocks, crop/soil changes, installed or cut beds, queues of compost and pantry receipts persist within the episode. Seasonal water, seed and leaf-litter renewal supports continued choices in this bounded world; no infinite map or cross-process persistence is claimed.

## Native action dictionary

0. Wait; recover two energy.
1. Move north.
2. Move east.
3. Move south.
4. Move west.
5. Fill one carried water from adjacent shared well.
6. Irrigate an adjacent plot's farm with one carried water, watering tool.
7. Harvest a healthy ripe plot into fruit and waste.
8. Plant an empty adjacent plot with a carried seed, grafting tool.
9. Graft a planted adjacent plot to carried seed trait, grafting tool.
10. Prune disease from an adjacent plot, pruning tool.
11. Cut adjacent root-bed connection and discard its in-flight packet, pruning tool.
12. Connect adjacent root bed for one seed and one compost, grafting tool.
13. Take one compost from an adjacent shared bin.
14. Put one carried waste into an adjacent shared bin.
15. Stir adjacent compost bin, compost-cart tool.
16. Spread one carried compost on an adjacent plot's soil, compost-cart tool.
17. Collect one ground detritus from an adjacent plot.
18. Deliver one fruit to an adjacent farm pantry.
19. Take a hardy seed from the central nursery.
20. Take a productive seed from the central nursery.
21. Take a resistant seed from the central nursery.
22. Give first compatible resource to lowest-ID eligible adjacent actor.
23. Exchange tools with lowest-ID adjacent actor when both loads fit eight.
24. Rest; recover six energy.
25. Save a matching seed from an adjacent developed crop, grafting tool.

## Observation, terminal and human-control contract

Fully public native physical world. Every role receives its own ID followed by the same public_keys vector: positions, tools, carried resources, seed/crop traits, soil, links, in-flight packets including infection, shared facilities, weather/calendar and real event/reward components. All entity visibility masks are true; inactive packets are not rendered as present. There is no hidden order, private farm stock or renderer-only information. Group labels and role selection use the same actor/farm state.

Observations prepend the explicit self fields in manifest order, then flatten public_keys in their literal source order. Every column is labelled; all values are float32 and bounded by [-2, 20480]. Native legal_actions is bool[A,U]; the renderer must not display hidden order/granary attributes to another role. Full-state debug views are separate from role views.

This finite world has no natural joint terminal. H=640 is a truncation window, not success. A nonterminal call increments int32 t once; terminated remains false; truncated=(t>=H)&~terminated; done is their disjunction. The absorbing raw branch returns the identical state and observation, zero float32 rewards, all dones true and unchanged causes. There is no raw autoreset.

All integer design fields use int32 and all flags bool; the supplied uint32 reset key remains external trajectory provenance. Stochastic reset variables use only that key; subsequent rules are deterministic and the step key is unused. No imported benchmark implementation, upstream game source, policy, asset or training baseline is embedded.

Human and AI controls supply the same scalar int32 actor components to the same raw_step. Actor IDs and crew/farm/settlement attributes support selecting one or multiple actors. Scene metadata supplies actual positions, attributes, visibility and actions; it does not implement a browser, camera UI, animation, JaxMARL adapter or runtime integration.

Movement is simultaneous and conservative: proposals entering another actor pre-step cell, any contested destination, a blocked tile or an outer boundary fail. Swaps and following into a just-vacated cell fail. Movement masks permit attempted blocked directions; failed movement leaves position unchanged. Interactions then resolve by ascending actor ID, with resource checks repeated against the current carried state. A mask-valid interaction can become a no-op after a lower-ID actor consumes the object or a neighbor moves. Nearest eligible object and then lowest index resolve spatial ties as defined in source. Invalid/out-of-range actions become wait; only wait remains legal after done.

## Reward allocation and analytic bounds

Each of 32 actors acts once. A loose member bound allows four home deliveries (4.8), 28 foreign deliveries to its pantry (16.8), personal harvest plus delivery (0.8), two initially missing roots (0.4) and all eight packet arrivals (0.16): 22.96, below declared 26. Per-actor negative cost is at worst 0.01 effort plus 0.08 pruning, or 0.01 plus 0.06 construction materials; at most two home plots rot in a tick, adding 0.5. Thus [-2,26] encloses reward. Per-pantry cumulative receipts cannot exceed all 32 actors delivering on all 640 steps, 20480. All other observed quantities and native component values fit [-2,20480].

- Component 0: Personal: +0.15 per harvested fruit and +0.5 per delivered fruit.
- Component 1: Home farm: +1.2 to each member of the delivering actor's home farm per fruit.
- Component 2: Foreign recipient: +0.6 to each member of a different pantry farm per fruit delivered there.
- Component 3: Public flow/infrastructure: +0.2 to every actor per first-ever root-bed installation and +0.02 per arriving resource packet, including infected packets.
- Component 4: Costs: -0.01 per move/interaction effort, -0.03 per water/compost/seed unit explicitly spent, and -0.08 per crop cure or root cut.
- Component 5: Local disease: -0.25 to each member of a farm per crop actually lost there.

## Literal hypotheses and evidence limits

The root-isolation script walks pruner actor 1 to the first root midpoint, then proposes cutting it while other gardeners wait. The common three-move prefix comes directly from native coordinates. The only paired difference is that cut versus wait; eight suffix ticks cover the seven-cell first corridor so resource retention, neighbor supply and possible infection can be inspected later. The second script proposes water/cure/seed handoff/compost work in the first farm, then an adjacent tool exchange and graft with the received seed. No seed trials, observed infection, solvability or successful harvest is claimed.

The paired suffixes have equal length 8 and differ in exactly one component: suffix step 0, actor 1. Prefixes and suffixes are literal JSON joint-action arrays. Both scripts and the pair have seed 0 selected without any trial. Their intention is a future counterfactual inspection, not an observed outcome.

Declared invariants are source obligations: actor positions are unique under conservative movement; positions/clocks and the listed inventory, resource and progress arrays have finite bounds. No conservation claim is made across extraction, growth, expiry, meals, rain, overflow, damage or explicit resource spending. Static checks cannot establish these obligations at runtime.

## Literal source evidence

- candidate.py:L13-L52: reset; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L54-L55: distances; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L57-L63: packet_positions; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L65-L67: observe; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L69-L97: legal_actions; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L99-L185: interact; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L187-L212: root_process; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L214-L230: crop_process; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L232-L280: advance; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.
- candidate.py:L282-L284: raw_step; exact source bytes are bound by SHA-256 19b2e4922d1730850ea0258b8fa2499e8044141fa86d86e5431f6866615a0d8d.

## Actual inputs and provenance

author_input_freeze.json lists all 25 actually-read current allowed inputs: two controlling authorities and 23 payloads. The four current designs and requests were read for this disjoint batch. Every byte count and SHA was checked against the current authority. The permitted PPO implementation snapshot was not read. Official README snapshots were byte-read for identity; only the recorded excerpt lines were displayed for API/design context.

Reference ideas: Craftax provides context for bounded persistent exploration; JaxMARL describes per-agent dictionaries and joint done as an integration reference; Pgx separates state, legal masks and supplied randomness; PureJaxRL is algorithm implementation context, not a game library or a substitute learner. No compatibility or performance claim is transferred from these references.

Only author-time stdlib AST/data/hash checks and the supplied trusted whitelist, manifest and scene validators were run. No candidate import or method, JAX/NumPy, JIT/vmap, rollout, render, training, seed search, external model, network or local model was executed. No prior source, feedback, diagnosis, selection or heldout evidence was accessed. Numerical, gameplay, animation, human-control, learning, throughput and transfer checks remain unperformed.
