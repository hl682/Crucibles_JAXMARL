# Seasonal Pasture Commons

Status: AUTHORED_NOT_EXECUTED. Initial proposal, attempt 00, feedback_used=0. All scripts are unexecuted source-derived hypotheses.

Native interface: A=64, O=1026, U=19, H=768. Every counted actor has a scalar int32 action and its own float32 observation/reward and bool done row. The family is rural_community.

Give every homestead at least two stored grain while keeping all herds healthy and disease-free; manage competing water, pasture and corridor uses.

Herds are interactive livestock objects, not counted native actors; the 64 controllable farmers each have native rows. A farm may prioritize pantry production, herd feeding, corridor access or reserves. All row users compete for the same finite-capacity reservoir, with at most 32 replenishments. Disease is Manhattan-distance transmission, including across a fence if herds remain within distance 2; closing a gate is not magical epidemiological immunity. Reset is deterministic with herd 0 initially diseased.

Rules and physical meanings:

- 64 actors form sixteen homesteads. Tools 0 farmer, 1 herder, 2 irrigator, 3 steward can be exchanged nearby or cycled at home.
- Plant within three cells of a plot. Six moist growth ticks make a ripe crop; harvest creates two grain and one seed. Heat can wither unwatered immature crops.
- Irrigation at a trough spends two units from its row reservoir. An open downstream gate shares one moisture with the next homestead.
- Drive a nearby herd in the actor's facing direction. Grazing every four ticks consumes actual cell grass; protected reserves cannot be grazed.
- Spring and autumn restore pasture every 12 ticks; protected reserves restore it every eight. Fertilizer uses collected manure.
- Nearby herds spread disease every eight ticks. An irrigator can spend compost to treat a herd; distance and routes can limit contact.
- Gate crossings wear hinges. Occupied gates cannot close; worn empty gates close until repaired. A closed gate also stops downstream irrigation sharing.
- Store harvested grain, feed hungry herds, or keep it for later work. Household welfare is rewarded periodically; every pantry and herd must meet the stated natural finish rule.

Transition order and controls:

The native joint vector is processed in rotating actor order (k+t) modulo A. An actor checks its action against current state at its turn; a resource lost to an earlier actor converts the action to wait. Out-of-range input becomes wait. Nonterminal movement codes always allow facing even when walking is blocked; occupancy and terrain control actual movement. Ties for nearby targets choose minimum Manhattan distance, then lowest index. No direction action teleports through a blocked walking cell.
All spatial updates, resources and tool effects are in candidate.py:L157-L210 (actor). Movement/collision is in candidate.py:L54-L65 (move_actor) and candidate.py:L136-L138 (passable). Exact contextual masks are in candidate.py:L140-L155 (choices); mask assembly is in candidate.py:L23-L25 (mask_row) and candidate.py:L27-L29 (legal_actions).
The world updates production/ecology/hazards after actor actions. These delayed effects and reward components are in candidate.py:L221-L253 (raw_step).

Rewards and finite bounds:

A harvest gives actors of the plot's homestead 0.5 and all actors 0.01. Storing one grain gives its actor 0.25. Planting, irrigating, reserving, fertilizing, tending or repairing costs its actor 0.02. Every 32 ticks a healthy, fed, disease-free household herd gives its members 0.4 plus all actors 0.02 times the fraction of qualifying herds. Herd damage costs its household 0.1; crop withering costs its household 0.2. At most sixteen harvests per step plus these components fit [-1,2]. No reward clipping.
Observation entries are float32 in [-2,65536]. This loose finite envelope includes coordinates, codes, quantities and cumulative counters; at most A*H <= 49152 actor actions can contribute to a counter in one episode. It is a representational bound, never a performance threshold.

Exact observation layout:

- Columns [0,8): Self x,y,homestead,tool,seeds,grain,compost,face. Access: self.
- Columns [8,12): t,season,total harvests,total stored grain. Access: public.
- Columns [12,524): 64 actor slots width 8: visible,x,y,homestead,tool,seeds,grain,compost. Access: local.
- Columns [524,652): 16 plot slots width 8: visible,x,y,crop stage,growth,moisture,pantry,harvest count. Access: local.
- Columns [652,716): 16 trough slots width 4: visible,x,y,plot moisture. Access: local.
- Columns [716,828): 16 herd slots width 7: visible,x,y,hunger,health,disease,manure. Access: local.
- Columns [828,908): 16 gate slots width 5: visible,x,y,open,wear. Access: local.
- Columns [908,928): 4 reservoir slots width 5: visible,x,y,water stock,remaining recharges. Access: local.
- Columns [928,977): 7x7 terrain patch y-major, outside=-1; tiles include visible grass/reserve/trough/gate state. Access: local.
- Columns [977,1026): 7x7 grass patch y-major, outside=-1. Access: local.

Self fields and declared public counters/charts are exact. For each entity slot, visibility is active AND Chebyshev distance <= 3 from the role's current position; hidden slots have visible=0 and all values zero. Actor slots use the same radius. Patches cover dx,dy -3..3 in row-major order, outside=-1. Public static bounds, indexing and action meanings do not expose hidden current state. Legal masks are native affordance information; no renderer-only policy information. Human and policy actors use the same scalar action and transition.
The observer construction is candidate.py:L255-L264 (observe). Entity/map masks are regenerated in candidate.py:L31-L40 (refresh). Masked entity flattening and patch boundaries are candidate.py:L42-L44 (visible_rows) and candidate.py:L46-L52 (patch). No role-view animation may expose hidden state or use full-state event arrays as public announcements. Only the declared public counters are scene events.

Declared persistent state and invariants:

Initial arrays, positions, resource stocks and integer/bool representations are defined in candidate.py:L79-L114 (reset). The manifest declares every reset key exactly once. Every state leaf keeps a fixed shape; positions are [x,y], maps are indexed [y,x], and all actors remain active through termination. Native actor positions are unique; non-agent objects may share a cell with an actor when the mechanic permits it.
Analytical ranges used during author review: seed in [0,4]; grain in [0,4]; compost in [0,3]; grass in [0,4]; crop in [0,3]; growth in [0,6]; moisture in [0,3]; pantry in [0,8]; hunger in [0,8]; health in [1,6]; wear in [0,3]; water_stock in [0,8]. These are source assertions awaiting independent native checking.

Termination:

Natural completion when all pantry >= 2, all herd health >= 4, and no herd disease; otherwise truncate at 768. Herd health is bounded below by 1.
The terminal helper candidate.py:L67-L77 (end_step) increments t once for nonterminal input, gives natural completion priority over horizon truncation, and preserves all leaves and causes on already-terminal input with zero rewards. Joint dones agree with done_all. No reset occurs inside raw_step.

Source-derived partner intervention:

Irrigate (6) versus wait (0) at branch step 0.
Open gate 0 is intended to send one moisture to plot 1 only in branch A. At the common fourth-tick growth boundary, neighboring homestead actors can observe a changed growth value. This is a source-derived dependency hypothesis, not a replay result.
The two full literal diagnostic scripts and both paired suffixes are in playthrough.json. They are dimension/range-checked as JSON only. Prefix and branch actions were chosen from rules, not searched, simulated or replayed. They are not a solvability, success-rate or causal-effect receipt.

Presentation and provenance:

The scene binds the exact source and manifest hashes, actual position/active/visibility leaves, tile map, actions and observed attributes. Crew index, actor identity, current tool and object state support later camera/actor selection. Scene metadata is delivered; native animation, RGB rendering and a human-play UI have not been executed or accepted. Any human control must call the same native transition as a policy.
The sanitized authority and scope files govern these initial ordinals and current counts. Historical count examples are not applied. All input files are enumerated with bytes/SHA256 and truthful semantic-read scope in author_input_freeze.json. Five allowed official reference snapshots were hashed for integrity only; their contents were not used as design/code inputs. No upstream code or commercial art was copied.
All small integer quantities use int32; no uint32 PRNG key is duplicated in state. Requested model configuration is gpt-6-astra/max; this author self-report does not establish provider model identity or token/cost telemetry.

Evidence boundary:

Only source AST screening, data-schema checking, symbolic observation-width accounting, reset-key coverage and literal-script structure checking were performed. No candidate was imported or executed; no JAX/numpy simulation, native reset/step, R0, eager/JIT/vmap comparison, benchmark, training, heldout evaluation, browser or downstream feedback was used. Intra-author static corrections to whitelist-rejected constructs occurred before this initial freeze and are recorded in generation_record.json. Native compatibility, replay legality, strategic richness, solvability, learning, UI quality and transfer remain unestablished.
