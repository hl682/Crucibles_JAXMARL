# Cavern Rescue Guilds

Status: AUTHORED_NOT_EXECUTED. Initial proposal, attempt 00, feedback_used=0. All scripts are unexecuted source-derived hypotheses.

Native interface: A=20, O=344, U=18, H=512. Every counted actor has a scalar int32 action and its own float32 observation/reward and bool done row. The family is block_survival.

Evacuate all five casualties through excavated caverns; sustain shared air and infrastructure while provisioning other crews.

Interdependence: a miner can improve access while imposing dust on nearby workers; a ventilator can spend wood on a duct instead of donating it; a carpenter can trade support safety for faster hauling. Winch/rail routes and walking rescue routes are alternatives. Donations are bounded first receipts, not a repeatable pickup/redeposit reward. Layout and initial supplies are deterministic; key is intentionally unused.

Rules and physical meanings:

- 20 miners in five crews share a connected cavern. Move or face, mine, build and carry people through the actual map.
- Tools 0 pick, 1 builder, 2 ventilator, 3 medic can be cycled at a depot. Build actions use one wood.
- Dig only the facing rock cell. Dust spreads through open cardinal neighbors; high dust injures workers and can collapse unsupported empty floor every 24 ticks.
- Stand on a duct to pump. Support beams, rails and winches replace ordinary floor and persist. Rail haul needs two clear rail/winch cells ahead.
- A medic lifts a nearby casualty only after its rock cell is cleared; a winch extends lift reach to three cells. Arrival near any depot evacuates the passenger.
- Adjacent transfers choose the nearest actor, then lowest index. Foreign depots accept at most two rewarded wood donations.
- All actors remain actionable. Health 1-2 slows movement; treatment restores health to 5. All five rescues finish the episode.

Transition order and controls:

The native joint vector is processed in rotating actor order (k+t) modulo A. An actor checks its action against current state at its turn; a resource lost to an earlier actor converts the action to wait. Out-of-range input becomes wait. Nonterminal movement codes always allow facing even when walking is blocked; occupancy and terrain control actual movement. Ties for nearby targets choose minimum Manhattan distance, then lowest index. No direction action teleports through a blocked walking cell.
All spatial updates, resources and tool effects are in candidate.py:L137-L184 (actor). Movement/collision is in candidate.py:L54-L65 (move_actor) and candidate.py:L116-L118 (passable). Exact contextual masks are in candidate.py:L120-L135 (choices); mask assembly is in candidate.py:L23-L25 (mask_row) and candidate.py:L27-L29 (legal_actions).
The world updates production/ecology/hazards after actor actions. These delayed effects and reward components are in candidate.py:L186-L215 (raw_step).

Rewards and finite bounds:

Each successful evacuation adds 5 to every member of the carrier's crew and 0.4 to every actor. A foreign-depot wood donation adds 0.4 to its donor, with two receipts per depot. A build or medical treatment costs its actor 0.02. A dust-injury tick costs the injured actor 0.1 and every actor 0.005 per injured worker. At most five rescues and twenty injury flags occur in a step; [-1,30] safely contains these components. No reward clipping is used.
Observation entries are float32 in [-2,65536]. This loose finite envelope includes coordinates, codes, quantities and cumulative counters; at most A*H <= 49152 actor actions can contribute to a counter in one episode. It is a representational bound, never a performance threshold.

Exact observation layout:

- Columns [0,9): Self x,y,crew,tool,wood,medicine,health,carrying,face. Access: self.
- Columns [9,11): Clock t; total rescued. Access: public.
- Columns [11,191): 20 actor slots, width 9: visible,x,y,crew,tool,wood,medicine,health,carrying. Access: local.
- Columns [191,221): 5 depot slots, width 6: visible,x,y,wood,medicine,foreign-aid receipts. Access: local.
- Columns [221,246): 5 casualty slots, width 5: visible,x,y,bearer,active. Access: local.
- Columns [246,295): 7x7 terrain patch in y-major order, dx/dy -3..3; outside=-1. Access: local.
- Columns [295,344): 7x7 dust patch, same ordering; outside=-1. Access: local.

Self fields and declared public counters/charts are exact. For each entity slot, visibility is active AND Chebyshev distance <= 3 from the role's current position; hidden slots have visible=0 and all values zero. Actor slots use the same radius. Patches cover dx,dy -3..3 in row-major order, outside=-1. Public static bounds, indexing and action meanings do not expose hidden current state. Legal masks are native affordance information; no renderer-only policy information. Human and policy actors use the same scalar action and transition.
The observer construction is candidate.py:L217-L223 (observe). Entity/map masks are regenerated in candidate.py:L31-L40 (refresh). Masked entity flattening and patch boundaries are candidate.py:L42-L44 (visible_rows) and candidate.py:L46-L52 (patch). No role-view animation may expose hidden state or use full-state event arrays as public announcements. Only the declared public counters are scene events.

Declared persistent state and invariants:

Initial arrays, positions, resource stocks and integer/bool representations are defined in candidate.py:L79-L105 (reset). The manifest declares every reset key exactly once. Every state leaf keeps a fixed shape; positions are [x,y], maps are indexed [y,x], and all actors remain active through termination. Native actor positions are unique; non-agent objects may share a cell with an actor when the mechanic permits it.
Analytical ranges used during author review: wood in [0,4]; med in [0,2]; health in [1,5]; dust in [0,8]; depot_aid in [0,2]; rescued in [0,5]. These are source assertions awaiting independent native checking.

Termination:

Natural completion exactly when rescued == 5; otherwise truncate at 512. Health never falls below 1, so injured actors remain controllable.
The terminal helper candidate.py:L67-L77 (end_step) increments t once for nonterminal input, gives natural completion priority over horizon truncation, and preserves all leaves and causes on already-terminal input with zero rewards. Joint dones agree with done_all. No reset occurs inside raw_step.

Source-derived partner intervention:

Excavate (5) versus wait (0), at the first branch step.
Only branch A is intended to open cell [3,2] and emit dust. Common waiting lets the local dust field change other actors' patch observations and makes a cleared casualty available to a later medic. No effect magnitude or successful rescue is claimed.
The two full literal diagnostic scripts and both paired suffixes are in playthrough.json. They are dimension/range-checked as JSON only. Prefix and branch actions were chosen from rules, not searched, simulated or replayed. They are not a solvability, success-rate or causal-effect receipt.

Presentation and provenance:

The scene binds the exact source and manifest hashes, actual position/active/visibility leaves, tile map, actions and observed attributes. Crew index, actor identity, current tool and object state support later camera/actor selection. Scene metadata is delivered; native animation, RGB rendering and a human-play UI have not been executed or accepted. Any human control must call the same native transition as a policy.
The sanitized authority and scope files govern these initial ordinals and current counts. Historical count examples are not applied. All input files are enumerated with bytes/SHA256 and truthful semantic-read scope in author_input_freeze.json. Five allowed official reference snapshots were hashed for integrity only; their contents were not used as design/code inputs. No upstream code or commercial art was copied.
All small integer quantities use int32; no uint32 PRNG key is duplicated in state. Requested model configuration is gpt-6-astra/max; this author self-report does not establish provider model identity or token/cost telemetry.

Evidence boundary:

Only source AST screening, data-schema checking, symbolic observation-width accounting, reset-key coverage and literal-script structure checking were performed. No candidate was imported or executed; no JAX/numpy simulation, native reset/step, R0, eager/JIT/vmap comparison, benchmark, training, heldout evaluation, browser or downstream feedback was used. Intra-author static corrections to whitelist-rejected constructs occurred before this initial freeze and are recorded in generation_record.json. Native compatibility, replay legality, strategic richness, solvability, learning, UI quality and transfer remain unestablished.
