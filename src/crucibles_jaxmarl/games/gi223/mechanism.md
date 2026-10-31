# Harbor Banquet

Status: AUTHORED_NOT_EXECUTED. Initial proposal, attempt 00, feedback_used=0. All scripts are unexecuted source-derived hypotheses.

Native interface: A=40, O=576, U=18, H=512. Every counted actor has a scalar int32 action and its own float32 observation/reward and bool done row. The family is cooperative_kitchen.

Complete at least one fish-and-stew feast at every kitchen; optional single meals compete for dishes, trays and return boats.

Single meals offer a faster local return, while a composite feast uses complementary stations and offers a larger local plus common reward. Queue clearing competes with cooking and carrying. Channels are shallow walkable service lanes for workers; boat collision is separate. Dock replenishment is a bounded scheduled state update, not an authored animated supply-boat subsystem. Deterministic reset; no upstream game code copied.

Rules and physical meanings:

- 40 workers in eight kitchens. Item codes: 0 empty, 1 raw fish, 2 vegetables, 3 cooked fish, 4 stew, 5 fuel crate, 6 spoiled dish.
- Cook tool 1 starts only matching raw food at a compatible station. Fuel is consumed; four ticks finish a dish and add mess.
- Pick up cooked output before 12 ticks. Carried dishes cool after 16 ticks; adjacent handoffs preserve remaining warmth.
- Serve either cooked dish locally, or contribute one of each to a composite feast. The first feast component reserves a tray for up to 24 ticks.
- A completed service launches that kitchen's return boat. It must travel a full 32-cell circuit and be unloaded near its actual location before its tray is reused.
- The junction diverts boats around recurring flotsam. Engineer tool 4 can clear the shared blockage; boats cannot move into occupied boat cells.
- Dock deliveries replace remaining dock cargo every 32 ticks, at most eight times. Assortments alternate between the two ingredients and fuel.
- Cycle tools at your home table. Each kitchen completing one composite feast ends the episode; single dishes remain an alternative use of scarce equipment.

Transition order and controls:

The native joint vector is processed in rotating actor order (k+t) modulo A. An actor checks its action against current state at its turn; a resource lost to an earlier actor converts the action to wait. Out-of-range input becomes wait. Nonterminal movement codes always allow facing even when walking is blocked; occupancy and terrain control actual movement. Ties for nearby targets choose minimum Manhattan distance, then lowest index. No direction action teleports through a blocked walking cell.
All spatial updates, resources and tool effects are in candidate.py:L138-L194 (actor). Movement/collision is in candidate.py:L54-L65 (move_actor) and candidate.py:L119-L120 (passable). Exact contextual masks are in candidate.py:L122-L136 (choices); mask assembly is in candidate.py:L23-L25 (mask_row) and candidate.py:L27-L29 (legal_actions).
The world updates production/ecology/hazards after actor actions. These delayed effects and reward components are in candidate.py:L210-L240 (raw_step).

Rewards and finite bounds:

A single meal gives each beneficiary-kitchen actor 1 for the current preferred dish or 0.5 for substitution. A completed composite gives each beneficiary-kitchen actor 2 and every actor 0.1. Each executed action with a>=6 except pickup action 7 and transfer action 14 costs its actor 0.01, including local service, cleaning and tool cycling. Spoiled station output costs each kitchen member 0.25, a cold carried dish costs its carrier 0.1, and a timed-out reserved feast component costs each kitchen member 0.3. At most one launch per table per tick and eight tables bound reward within [-1,4]. No reward clipping.
Observation entries are float32 in [-2,65536]. This loose finite envelope includes coordinates, codes, quantities and cumulative counters; at most A*H <= 49152 actor actions can contribute to a counter in one episode. It is a representational bound, never a performance threshold.

Exact observation layout:

- Columns [0,7): Self x,y,kitchen,tool,item,warmth,face. Access: self.
- Columns [7,10): Clock t; total single meals; total composite feasts. Access: public.
- Columns [10,290): 40 actor slots width 7: visible,x,y,kitchen,tool,item,warmth. Access: local.
- Columns [290,362): 8 station slots width 9: visible,x,y,specialty,fuel,mess,timer,output,age. Access: local.
- Columns [362,410): 8 dock slots width 6: visible,x,y,stock,assortment,remaining deliveries. Access: local.
- Columns [410,482): 8 table slots width 9: visible,x,y,trays,fish,stew,feast age,single serves,feasts. Access: local.
- Columns [482,522): 8 boat slots width 5: visible,x,y,moved cells,ready. Access: local.
- Columns [522,527): Junction slot: visible,x,y,diversion,blockage. Access: local.
- Columns [527,576): 7x7 terrain patch y-major, outside=-1. Access: local.

Self fields and declared public counters/charts are exact. For each entity slot, visibility is active AND Chebyshev distance <= 3 from the role's current position; hidden slots have visible=0 and all values zero. Actor slots use the same radius. Patches cover dx,dy -3..3 in row-major order, outside=-1. Public static bounds, indexing and action meanings do not expose hidden current state. Legal masks are native affordance information; no renderer-only policy information. Human and policy actors use the same scalar action and transition.
The observer construction is candidate.py:L242-L251 (observe). Entity/map masks are regenerated in candidate.py:L31-L40 (refresh). Masked entity flattening and patch boundaries are candidate.py:L42-L44 (visible_rows) and candidate.py:L46-L52 (patch). No role-view animation may expose hidden state or use full-state event arrays as public announcements. Only the declared public counters are scene events.

Declared persistent state and invariants:

Initial arrays, positions, resource stocks and integer/bool representations are defined in candidate.py:L79-L107 (reset). The manifest declares every reset key exactly once. Every state leaf keeps a fixed shape; positions are [x,y], maps are indexed [y,x], and all actors remain active through termination. Native actor positions are unique; non-agent objects may share a cell with an actor when the mechanic permits it.
Analytical ranges used during author review: item in [0,6]; warmth in [0,16]; fuel in [0,11]; mess in [0,3]; timer in [0,4]; trays in [0,2]; stock in [0,4]; deliveries_left in [0,8]. These are source assertions awaiting independent native checking.

Termination:

Natural completion when every table has feasts >= 1; otherwise truncate at 512.
The terminal helper candidate.py:L67-L77 (end_step) increments t once for nonterminal input, gives natural completion priority over horizon truncation, and preserves all leaves and causes on already-terminal input with zero rewards. Joint dones agree with done_all. No reset occurs inside raw_step.

Source-derived partner intervention:

Start cooking (6) versus wait (0), at branch step 0.
After the identical three waiting steps, the source intends station 0 output/mess/fuel to differ. Actor 0's local station observation and future pickup opportunities depend on its partner's initial action. This is unexecuted.
The two full literal diagnostic scripts and both paired suffixes are in playthrough.json. They are dimension/range-checked as JSON only. Prefix and branch actions were chosen from rules, not searched, simulated or replayed. They are not a solvability, success-rate or causal-effect receipt.

Presentation and provenance:

The scene binds the exact source and manifest hashes, actual position/active/visibility leaves, tile map, actions and observed attributes. Crew index, actor identity, current tool and object state support later camera/actor selection. Scene metadata is delivered; native animation, RGB rendering and a human-play UI have not been executed or accepted. Any human control must call the same native transition as a policy.
The sanitized authority and scope files govern these initial ordinals and current counts. Historical count examples are not applied. All input files are enumerated with bytes/SHA256 and truthful semantic-read scope in author_input_freeze.json. Five allowed official reference snapshots were hashed for integrity only; their contents were not used as design/code inputs. No upstream code or commercial art was copied.
All small integer quantities use int32; no uint32 PRNG key is duplicated in state. Requested model configuration is gpt-6-astra/max; this author self-report does not establish provider model identity or token/cost telemetry.

Evidence boundary:

Only source AST screening, data-schema checking, symbolic observation-width accounting, reset-key coverage and literal-script structure checking were performed. No candidate was imported or executed; no JAX/numpy simulation, native reset/step, R0, eager/JIT/vmap comparison, benchmark, training, heldout evaluation, browser or downstream feedback was used. Intra-author static corrections to whitelist-rejected constructs occurred before this initial freeze and are recorded in generation_record.json. Native compatibility, replay legality, strategic richness, solvability, learning, UI quality and transfer remain unestablished.
