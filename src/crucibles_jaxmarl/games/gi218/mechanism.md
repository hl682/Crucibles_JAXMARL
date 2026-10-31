# Ropeway Commons

Status: AUTHORED_NOT_EXECUTED. Parentless initial blind source, attempt_00, train only. Requested author gpt-6-astra / max; measured provider identity, tokens, cost and reasoning telemetry are unavailable.

Native dimensions are A=12, O=671, U=28, H=512. This source is an original proposal, not a claim of runtime compatibility, solvability, learned behavior or transfer.

## Spatial rules and persistence

A 21 by 17 block valley contains eight finite deposits, four village depots, four pylon footings and four carriers on a clockwise square. Actors begin as four miner/rigger/hauler crews. Their current tool, rather than fixed identity, controls specialized actions; exchange tools changes which actor can mine ore, build, repair, brace or install weights. Agents carry at most eight material units in four inventory columns. Reset varies only deposit stocks from 18 through 24 using the supplied key.

Pylons 0 and 1 start built with health six. Pylons 2 and 3 require two stone and one cable; each rigger begins with three stone and one cable. Making cable consumes two fibre and one ore. Quarrying reduces visible stock permanently. Construction, counterweight placement, extraction and village receipts persist within the episode. Anchors can wear to zero but remain installed and can be repaired by three health for one stone. No resource regrowth or cross-process persistence is claimed.

A carrier is a genuine moving object on a straight span between consecutive pylons, with one cargo unit, loading-crew tag, integer progress and signed direction. It advances at most one grid cell per tick and only while loaded. At an endpoint it remains parked until reversed, unloaded or otherwise acted upon. Any nearby crew can unload it, allowing intermediate transfers onto another span and conflicting claims over shared freight. Empty carriers can be hand-positioned only by loading, moving, reversing and unloading; there is no off-scene teleport.

At each anchor, tension is twice the number of loaded incident spans plus a 0/1 gust alternating in 16-tick blocks. Support is two plus installed counterweights plus two for every current brace. Both endpoint anchors must be built, nonbroken and not overloaded for a carrier to advance. Excess tension stalls the adjacent carriers; on old-clock t mod 3 equal to 2 it removes one health from each overloaded nonbroken anchor. Unloading, bracing or redistributing stone as counterweights therefore changes another crew's movement opportunities immediately and its repair bill later.

Hand-carrying around the central rock wall avoids cable congestion but spends actor energy and time. Installing weights uses deliverable stone and can later be reversed to reclaim it, at the price of renewed overload risk. A crew can choose to support another delivery, unload a contested carrier, reverse its route or bypass the network. Deposits and carried materials are finite, and only consumed foreign-village deliveries and first construction produce positive event rewards; harvesting alone has no reward.

Objects at distance at most one in Manhattan geometry can be manipulated. Span actions explicitly select one of four carriers; other interactions choose eligible nearest objects or lowest-ID adjacent recipients. Rest and wait restore finite energy, so local mistakes and broken anchors do not naturally end the world.

## Native action dictionary

0. Wait and recover two energy.
1. Move north.
2. Move east.
3. Move south.
4. Move west.
5. Quarry adjacent stock; pick required for stone or ore.
6. Make one cable from two fibre and one ore with spanner.
7. Build adjacent pylon: two stone and one cable, spanner.
8. Brace adjacent pylon this step with hauling yoke.
9. Repair adjacent anchor by three health for one stone, spanner.
10. Load span 0 carrier; ore before stone.
11. Load span 1 carrier; ore before stone.
12. Load span 2 carrier; ore before stone.
13. Load span 3 carrier; ore before stone.
14. Unload span 0 carrier.
15. Unload span 1 carrier.
16. Unload span 2 carrier.
17. Unload span 3 carrier.
18. Reverse span 0 carrier direction.
19. Reverse span 1 carrier direction.
20. Reverse span 2 carrier direction.
21. Reverse span 3 carrier direction.
22. Install one stone counterweight, hauling yoke.
23. Withdraw one counterweight stone, hauling yoke.
24. Deliver ore or stone to an adjacent foreign village.
25. Give first carried material to lowest-ID adjacent actor with room.
26. Exchange tools with lowest-ID adjacent actor.
27. Rest and recover four energy.

## Observation, terminal and human-control contract

Full public physical state: every observer receives actor ID followed by the identical public_keys vector. Agent inventories, anchor load/support, finite deposits, carrier positions, routing direction, village receipts and component rewards are public. Visibility masks for all entity groups are true. Constant active/visibility arrays contain no private information.

Observations prepend the explicit self fields in manifest order, then flatten public_keys in their literal source order. Every column is labelled; all values are float32 and bounded by [-2, 6144]. Native legal_actions is bool[A,U]; the renderer must not display hidden order/granary attributes to another role. Full-state debug views are separate from role views.

This finite world has no natural joint terminal. H=512 is a truncation window, not success. A nonterminal call increments int32 t once; terminated remains false; truncated=(t>=H)&~terminated; done is their disjunction. The absorbing raw branch returns the identical state and observation, zero float32 rewards, all dones true and unchanged causes. There is no raw autoreset.

All integer design fields use int32 and all flags bool; the supplied uint32 reset key remains external trajectory provenance. Stochastic reset variables use only that key; subsequent rules are deterministic and the step key is unused. No imported benchmark implementation, upstream game source, policy, asset or training baseline is embedded.

Human and AI controls supply the same scalar int32 actor components to the same raw_step. Actor IDs and crew/farm/settlement attributes support selecting one or multiple actors. Scene metadata supplies actual positions, attributes, visibility and actions; it does not implement a browser, camera UI, animation, JaxMARL adapter or runtime integration.

Movement is simultaneous and conservative: proposals entering another actor pre-step cell, any contested destination, a blocked tile or an outer boundary fail. Swaps and following into a just-vacated cell fail. Movement masks permit attempted blocked directions; failed movement leaves position unchanged. Interactions then resolve by ascending actor ID, with resource checks repeated against the current carried state. A mask-valid interaction can become a no-op after a lower-ID actor consumes the object or a neighbor moves. Nearest eligible object and then lowest index resolve spatial ties as defined in source. Invalid/out-of-range actions become wait; only wait remains legal after done.

## Reward allocation and analytic bounds

All five float32 components are returned and exposed. Each actor has one action. At most three own-crew deliveries contribute 4.2 to a member and at most nine foreign deliveries to that crew contribute 9.0; personal reward is at most 0.6. Only two initially unbuilt pylons can produce first-installation events, at most 0.5 shared reward. Thus 14.3 is a loose positive per-actor bound. One action spends at most three material units plus one effort, and at most four anchors wear, so the lower bound is no smaller than -0.57. Declared [-2,20] is conservative. Village cumulative receipt cannot exceed 12*512=6144. Counts, energy, tools, progress and all observed reward components fit [-2,6144].

- Component 0: Personal: 0.6 per actor delivery.
- Component 1: Crew/recipient: 1.4 to each member of the delivering crew and 1.0 to each member of the receiving foreign crew per unit.
- Component 2: Infrastructure: 0.25 to every actor per first pylon installation.
- Component 3: Costs: -0.02 per move or non-rest interaction effort and -0.05 per material unit spent on construction, repair, counterweight installation or cable crafting.
- Component 4: Damage: -0.10 to every actor per anchor wear event.

## Literal hypotheses and evidence limits

Script brace_two_crews loads spans 0 and 1 while the other riggers construct the missing pylons; it then proposes one brace at their shared anchor. Script quarry_to_rigger walks a miner to a stone deposit, extracts one unit and carries it to an adjacent rigger. The pair shares the one-tick load/build prefix. Only actor 5's first suffix action changes from brace to wait. With two incident loads, that one support action is intended to alter both carriers' next positions, with later wear/stalling visible over four suffix ticks. This is source-derived timing only, not a measured intervention result.

The paired suffixes have equal length 4 and differ in exactly one component: suffix step 0, actor 5. Prefixes and suffixes are literal JSON joint-action arrays. Both scripts and the pair have seed 0 selected without any trial. Their intention is a future counterfactual inspection, not an observed outcome.

Declared invariants are source obligations: actor positions are unique under conservative movement; positions/clocks and the listed inventory, resource and progress arrays have finite bounds. No conservation claim is made across extraction, growth, expiry, meals, rain, overflow, damage or explicit resource spending. Static checks cannot establish these obligations at runtime.

## Literal source evidence

- candidate.py:L13-L52: reset; exact source bytes are bound by SHA-256 92695bd8eb6cd6e635e053ef152bf22ecb92b0d148a78c8246dcdd11a1844f0d.
- candidate.py:L54-L55: distances; exact source bytes are bound by SHA-256 92695bd8eb6cd6e635e053ef152bf22ecb92b0d148a78c8246dcdd11a1844f0d.
- candidate.py:L57-L66: cable_tiles; exact source bytes are bound by SHA-256 92695bd8eb6cd6e635e053ef152bf22ecb92b0d148a78c8246dcdd11a1844f0d.
- candidate.py:L68-L70: observe; exact source bytes are bound by SHA-256 92695bd8eb6cd6e635e053ef152bf22ecb92b0d148a78c8246dcdd11a1844f0d.
- candidate.py:L72-L97: legal_actions; exact source bytes are bound by SHA-256 92695bd8eb6cd6e635e053ef152bf22ecb92b0d148a78c8246dcdd11a1844f0d.
- candidate.py:L99-L162: interact; exact source bytes are bound by SHA-256 92695bd8eb6cd6e635e053ef152bf22ecb92b0d148a78c8246dcdd11a1844f0d.
- candidate.py:L164-L207: advance; exact source bytes are bound by SHA-256 92695bd8eb6cd6e635e053ef152bf22ecb92b0d148a78c8246dcdd11a1844f0d.
- candidate.py:L209-L211: raw_step; exact source bytes are bound by SHA-256 92695bd8eb6cd6e635e053ef152bf22ecb92b0d148a78c8246dcdd11a1844f0d.

## Actual inputs and provenance

author_input_freeze.json lists all 25 actually-read current allowed inputs: two controlling authorities and 23 payloads. The four current designs and requests were read for this disjoint batch. Every byte count and SHA was checked against the current authority. The permitted PPO implementation snapshot was not read. Official README snapshots were byte-read for identity; only the recorded excerpt lines were displayed for API/design context.

Reference ideas: Craftax provides context for bounded persistent exploration; JaxMARL describes per-agent dictionaries and joint done as an integration reference; Pgx separates state, legal masks and supplied randomness; PureJaxRL is algorithm implementation context, not a game library or a substitute learner. No compatibility or performance claim is transferred from these references.

Only author-time stdlib AST/data/hash checks and the supplied trusted whitelist, manifest and scene validators were run. No candidate import or method, JAX/NumPy, JIT/vmap, rollout, render, training, seed search, external model, network or local model was executed. No prior source, feedback, diagnosis, selection or heldout evidence was accessed. Numerical, gameplay, animation, human-control, learning, throughput and transfer checks remain unperformed.
