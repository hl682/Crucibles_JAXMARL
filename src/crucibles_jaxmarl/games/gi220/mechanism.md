# Delta Federation

Status: AUTHORED_NOT_EXECUTED. Parentless initial blind source, attempt_00, train only. Requested author gpt-6-astra / max; measured provider identity, tokens, cost and reasoning telemetry are unavailable.

Native dimensions are A=24, O=1042, U=25, H=512. This source is an original proposal, not a claim of runtime compatibility, solvability, learned behavior or transfer.

## River, land and complementary work

The 25 by 19 strategy board contains six visibly distinct settlement territories, one field and renewing woodlot per settlement, six connected river nodes and five one-unit barges. The main river runs east, south and west through those nodes before the final outflow leaves at x=0. Irrigation-ditch map tiles connect each river node to its settlement field and remain passable to feet. Main-channel tiles block foot movement unless a live bridge exists at that node. One actor occupies a cell; cargo barges are separate object occupancy.

Each settlement begins with farmer, builder, courier and sluice-operator tools on four residents. Tools are exchangeable, so specialization is a current physical affordance rather than an immutable policy slot. Farmers harvest mature fields or spend a grain to advance young growth by two. Builders collect timber, build bridges, raise levees or use timber to clear sediment. Operators open/close gates, dredge two sediment or breach one levee height. Couriers alone can load barges, while any nearby resident with room can unload or hand-carry freight. Inventories hold at most six grain/timber units.

Fields start at growth four and can grow to six; harvesting needs growth at least four, yields one grain and resets growth to zero. Woodlots start at eight, replenish one per 24 ticks and cap at eight. Granaries cap at 32 and consume one grain every twelve ticks. Hunger has a local cost and does not remove actors or end the world. Food and tools can physically change hands at adjacency, and other settlements can unload a loaded barge; these rules permit cooperation or appropriation with concrete material consequences.

## Delayed flow, damage and transportation

Every fourth old-clock tick is a hydraulic phase. Each node receives the previous reach's third delay position plus its local rain. On a phase, an open gate releases one unit, a second when level is at least five, and a third when a levee is present and level is at least six, bounded by available water. Each phase also removes one positive water unit for local consumption. Node storage clips at nine; overflow is lost. The outgoing units enter the first position of that reach's three-position queue, then shift each native step. On the step after occupying position three they enter the next node. The final reach exits the world.

Visible packet positions lie along the actual channel, with declared per-position quantities. They express a three-step transport delay rather than a teleport between unrelated coordinates. A gate action changes future downstream stores, field growth and navigation. Levee height raises the flood threshold from five to five plus height, but can redirect a larger release downstream. A flood resets field growth, and a level-nine flood also removes one levee height and destroys that bridge. Repairs/reconstruction remain possible. A resident already standing on a destroyed crossing is not deleted; subsequent movement still applies the native destination rule.

A barge follows its assigned straight river reach, at most one cell per hydraulic phase. Its source gate must be open, both endpoints must hold at least two water and both sediment values must be below four. Barge movement can occur empty or loaded. Direction actions set upstream/downstream, and endpoints clamp until direction changes. Cargo stays a single actual grain or timber unit. Lowest-index eligible barge resolves a same-location handling tie. Water, sediment, routing and local construction thus change another settlement's opportunities with explicit delay and capacity.

## Local knowledge, interests and alternatives

Reset samples each settlement's rainfall phase code in 0-2 from the supplied key. Every twelve ticks, that code and the public calendar determine whether its node receives one rain. Home residents directly see their own code and granary. Speaking at distance one supplies the speaker's home code and stock snapshot to adjacent listeners; report lifetime is set to 16 then decremented at step end. Own knowledge is refreshed live. Reports may become stale as grain moves. Realized rain and public stock-change rewards are observable, so this is direct-information asymmetry, not a claim of cryptographic secrecy.

Road hand-carry can avoid a blocked barge but requires labor and safe crossings. Closing an upstream gate can preserve local water while withholding downstream navigation; raising a levee protects local growth while exporting high-water risk. Dredging and bridges provide public transport benefits, while grain stored in a particular granary contributes to that settlement's meals. Trade changes real stock and its signed reward potential, not a narrative ledger. Repeated withdrawal/deposit reverses its own potential credit; there is no positive reward simply for repeating that stock loop.

## Native action dictionary

0. Wait; recover two energy.
1. Move north.
2. Move east.
3. Move south.
4. Move west.
5. Harvest adjacent mature field with farming tool.
6. Collect one timber from adjacent woodlot.
7. Raise adjacent levee for two timber, builder tool.
8. Open adjacent sluice, operator tool.
9. Close adjacent sluice, operator tool.
10. Dredge two sediment at adjacent node, operator tool.
11. Load nearest empty adjacent barge with grain before timber, courier tool.
12. Unload nearest loaded adjacent barge into inventory.
13. Set adjacent barge direction downstream.
14. Set adjacent barge direction upstream.
15. Store one grain in adjacent granary.
16. Take one grain from own adjacent granary.
17. Give grain before timber to lowest-ID adjacent actor with room.
18. Exchange tools with lowest-ID adjacent actor.
19. Tell adjacent actors home granary and rain phase code.
20. Construct adjacent bridge for two timber, builder tool.
21. Clear one sediment for one timber, builder tool.
22. Rest; recover five energy.
23. Sow one grain to advance adjacent young field by two, farmer tool.
24. Breach one levee height at adjacent node, operator tool.

## Observation, terminal and human-control contract

All terrain, territories, actor inventories/tools, fields, woodlots, river levels/gates/levees/bridges/sediment, water packets, barges, rain, public event channels and reward components are public. A granary's current contents and rain phase code are directly visible only to its four home members via village_visible. Observations include only the observing actor's notes_food, notes_weather and note_life rows; a nearby speaker can provide a stale snapshot about their own settlement. Public actions and realized rain may permit inference; the source does not claim permanent secrecy against history. Legal foreign deposit attempts do not expose a full-granary test.

Observations prepend the explicit self fields in manifest order, then flatten public_keys in their literal source order. Every column is labelled; all values are float32 and bounded by [-6, 512]. Native legal_actions is bool[A,U]; the renderer must not display hidden order/granary attributes to another role. Full-state debug views are separate from role views.

This finite world has no natural joint terminal. H=512 is a truncation window, not success. A nonterminal call increments int32 t once; terminated remains false; truncated=(t>=H)&~terminated; done is their disjunction. The absorbing raw branch returns the identical state and observation, zero float32 rewards, all dones true and unchanged causes. There is no raw autoreset.

All integer design fields use int32 and all flags bool; the supplied uint32 reset key remains external trajectory provenance. Stochastic reset variables use only that key; subsequent rules are deterministic and the step key is unused. No imported benchmark implementation, upstream game source, policy, asset or training baseline is embedded.

Human and AI controls supply the same scalar int32 actor components to the same raw_step. Actor IDs and crew/farm/settlement attributes support selecting one or multiple actors. Scene metadata supplies actual positions, attributes, visibility and actions; it does not implement a browser, camera UI, animation, JaxMARL adapter or runtime integration.

Movement is simultaneous and conservative: proposals entering another actor pre-step cell, any contested destination, a blocked tile or an outer boundary fail. Swaps and following into a just-vacated cell fail. Movement masks permit attempted blocked directions; failed movement leaves position unchanged. Interactions then resolve by ascending actor ID, with resource checks repeated against the current carried state. A mask-valid interaction can become a no-op after a lower-ID actor consumes the object or a neighbor moves. Nearest eligible object and then lowest index resolve spatial ties as defined in source. Invalid/out-of-range actions become wait; only wait remains legal after done.

## Reward allocation and analytic bounds

Deposits and withdrawals use signed changes so taking grain out and returning it does not create new aggregate stock credit. Trade moves this potential between settlements; meal reward requires grain consumption. A loose per-actor upper bound is 24*0.8 destination potential +0.7 meal +0.2 courier +6*0.15 first bridges +5*0.02 navigability =21.1, below declared 24. Only the four home actors can withdraw from a given granary, so a member can lose at most 4*0.8 plus personal 0.2, effort/material 0.18 and local flood/hunger 1.0, greater than -6. Native observed counts are at most 512 for t and at most 32 for stocks; signed knowledge sentinels and component rewards fit [-6,512]. These are analytic enclosing bounds, not measured returns.

- Component 0: Settlement stock and meals: +0.8 to each member of the destination settlement per deposited grain, -0.8 per own-granary withdrawal, plus +0.7 per actually consumed local meal.
- Component 1: Carrier stock potential: +0.2 to the depositing actor and -0.2 to the withdrawing actor per grain.
- Component 2: Public infrastructure/navigation: +0.15 to every actor per first-ever bridge and +0.02 per currently navigable reach on a hydraulic phase.
- Component 3: Costs: -0.02 per movement/interaction effort, -0.08 per timber spent, and -0.05 per grain sown.
- Component 4: Local harm: -0.4 per home field flood and -0.6 per home meal time with an empty granary.

## Literal hypotheses and evidence limits

The delayed_flow script waits once and then proposes opening node 0 through actor 3, retaining enough literal waiting steps to cover the next hydraulic phase and the three-position delay. The grain_to_courier script harvests, walks a farmer around the field, hands grain to the adjacent courier and proposes a real granary deposit. The pair uses the identical one-wait prefix; only actor 3's first suffix action is open versus wait. Eight suffix ticks were chosen from the four-tick hydraulic clock and three-position queue, without a trial. Intended downstream differences concern packet amount, node water and subsequent field/barge opportunities; no outcome is reported.

The paired suffixes have equal length 8 and differ in exactly one component: suffix step 0, actor 3. Prefixes and suffixes are literal JSON joint-action arrays. Both scripts and the pair have seed 0 selected without any trial. Their intention is a future counterfactual inspection, not an observed outcome.

Declared invariants are source obligations: actor positions are unique under conservative movement; positions/clocks and the listed inventory, resource and progress arrays have finite bounds. No conservation claim is made across extraction, growth, expiry, meals, rain, overflow, damage or explicit resource spending. Static checks cannot establish these obligations at runtime.

## Literal source evidence

- candidate.py:L13-L55: reset; exact source bytes are bound by SHA-256 0fc73005a86b8d8efc383377d1b908d9075861b7d963d74278ad35bcf008ef07.
- candidate.py:L57-L58: distances; exact source bytes are bound by SHA-256 0fc73005a86b8d8efc383377d1b908d9075861b7d963d74278ad35bcf008ef07.
- candidate.py:L60-L63: observe; exact source bytes are bound by SHA-256 0fc73005a86b8d8efc383377d1b908d9075861b7d963d74278ad35bcf008ef07.
- candidate.py:L65-L90: legal_actions; exact source bytes are bound by SHA-256 0fc73005a86b8d8efc383377d1b908d9075861b7d963d74278ad35bcf008ef07.
- candidate.py:L92-L164: interact; exact source bytes are bound by SHA-256 0fc73005a86b8d8efc383377d1b908d9075861b7d963d74278ad35bcf008ef07.
- candidate.py:L166-L228: advance; exact source bytes are bound by SHA-256 0fc73005a86b8d8efc383377d1b908d9075861b7d963d74278ad35bcf008ef07.
- candidate.py:L230-L232: raw_step; exact source bytes are bound by SHA-256 0fc73005a86b8d8efc383377d1b908d9075861b7d963d74278ad35bcf008ef07.

## Actual inputs and provenance

author_input_freeze.json lists all 25 actually-read current allowed inputs: two controlling authorities and 23 payloads. The four current designs and requests were read for this disjoint batch. Every byte count and SHA was checked against the current authority. The permitted PPO implementation snapshot was not read. Official README snapshots were byte-read for identity; only the recorded excerpt lines were displayed for API/design context.

Reference ideas: Craftax provides context for bounded persistent exploration; JaxMARL describes per-agent dictionaries and joint done as an integration reference; Pgx separates state, legal masks and supplied randomness; PureJaxRL is algorithm implementation context, not a game library or a substitute learner. No compatibility or performance claim is transferred from these references.

Only author-time stdlib AST/data/hash checks and the supplied trusted whitelist, manifest and scene validators were run. No candidate import or method, JAX/NumPy, JIT/vmap, rollout, render, training, seed search, external model, network or local model was executed. No prior source, feedback, diagnosis, selection or heldout evidence was accessed. Numerical, gameplay, animation, human-control, learning, throughput and transfer checks remain unperformed.
