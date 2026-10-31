# Archipelago Councils

Status: AUTHORED_NOT_EXECUTED. Initial proposal, attempt 00, feedback_used=0. All scripts are unexecuted source-derived hypotheses.

Native interface: A=48, O=681, U=18, H=640. Every counted actor has a scalar int32 action and its own float32 observation/reward and bool done row. The family is civilization_strategy.

Develop all eight towns through two timber and two grain deliveries each while negotiating shared transport and drainage.

Reset samples seven binary reef flags from the supplied key; all other initial geometry/resources are deterministic. Signal publication reveals a deliberately chosen local reading, while arbitrary remote state remains hidden. Ferry entities are actual service-crossing locations with capacity/cooldown, not independently animated vessels. Town 7's chart reports crossing 6. Bridge safety checks preserve passability under occupants.

Rules and physical meanings:

- 48 settlers in eight towns. Every actor moves and carries materials; tools change at the actor's own town.
- Two timber investments make a bridge walkable. A canal prevents its surge damage and downstream water diversion; each investment uses one carried timber.
- A ferry moves an actor across its two-cell channel. One shared fuel unit and cooldown are consumed; town owners can close their port to outsiders.
- At high tide a reef crossing requires a navigator; a failed crossing attempt stays in place. Any actor can refuel a ferry.
- A messenger spends a fuel pack to broadcast its town's reef/tide chart for 16 ticks. Charts are public and can become stale.
- A complete bridge is protected from collapse while occupied. Uncanalized construction can reduce the next town's seasonal farm renewal.
- Deliver two timber and two grain to each town for development. Aid rewards the beneficiary town and the delivering actor.
- Grove/farm renewal is bounded to eight cycles. Finite town fuel and the longer southern coastal road make local investment and reciprocal transport distinct choices.

Transition order and controls:

The native joint vector is processed in rotating actor order (k+t) modulo A. An actor checks its action against current state at its turn; a resource lost to an earlier actor converts the action to wait. Out-of-range input becomes wait. Nonterminal movement codes always allow facing even when walking is blocked; occupancy and terrain control actual movement. Ties for nearby targets choose minimum Manhattan distance, then lowest index. No direction action teleports through a blocked walking cell.
All spatial updates, resources and tool effects are in candidate.py:L150-L196 (actor). Movement/collision is in candidate.py:L54-L65 (move_actor) and candidate.py:L125-L128 (passable). Exact contextual masks are in candidate.py:L130-L148 (choices); mask assembly is in candidate.py:L23-L25 (mask_row) and candidate.py:L27-L29 (legal_actions).
The world updates production/ecology/hazards after actor actions. These delayed effects and reward components are in candidate.py:L198-L220 (raw_step).

Rewards and finite bounds:

Each capped timber/grain delivery gives beneficiary-town actors 0.8 and the delivering actor 0.3. First use of each ferry gives every actor 0.1. Bridge/canal investment, refueling or chart broadcast costs the acting actor 0.04. An uncanalized surge-exposed bridge costs downstream-town actors 0.2 even when occupancy temporarily protects the span. Four deliveries per town, seven first uses and one personal action bound reward within [-1,5]. No reward clipping.
Observation entries are float32 in [-2,65536]. This loose finite envelope includes coordinates, codes, quantities and cumulative counters; at most A*H <= 49152 actor actions can contribute to a counter in one episode. It is a representational bound, never a performance threshold.

Exact observation layout:

- Columns [0,8): Self x,y,town,tool,wood,grain,fuel packs,face. Access: self.
- Columns [8,28): t,tide phase,developed towns,used crossings; 8 chart codes; 8 chart ages. Chart code=3*reef+tide, -1 absent. Access: public.
- Columns [28,412): 48 actor slots width 8: visible,x,y,town,tool,wood,grain,fuel packs. Access: local.
- Columns [412,468): 8 town slots width 7: visible,x,y,timber delivered,grain delivered,fuel store,port access. Access: local.
- Columns [468,503): 7 bridge slots width 5: visible,x,y,bridge level,canal. Access: local.
- Columns [503,552): 7 ferry slots width 7: visible,x,y,fuel,cooldown,reef,ever used. Access: local.
- Columns [552,592): 8 grove slots width 5: visible,x,y,timber stock,remaining renewals. Access: local.
- Columns [592,632): 8 farm slots width 5: visible,x,y,grain stock,water. Access: local.
- Columns [632,681): 7x7 base terrain patch y-major, outside=-1. Bridge entities override water passability at their cell when level 2. Access: local.

Self fields and declared public counters/charts are exact. For each entity slot, visibility is active AND Chebyshev distance <= 3 from the role's current position; hidden slots have visible=0 and all values zero. Actor slots use the same radius. Patches cover dx,dy -3..3 in row-major order, outside=-1. Public static bounds, indexing and action meanings do not expose hidden current state. Legal masks are native affordance information; no renderer-only policy information. Human and policy actors use the same scalar action and transition.
The observer construction is candidate.py:L222-L231 (observe). Entity/map masks are regenerated in candidate.py:L31-L40 (refresh). Masked entity flattening and patch boundaries are candidate.py:L42-L44 (visible_rows) and candidate.py:L46-L52 (patch). No role-view animation may expose hidden state or use full-state event arrays as public announcements. Only the declared public counters are scene events.

Declared persistent state and invariants:

Initial arrays, positions, resource stocks and integer/bool representations are defined in candidate.py:L79-L112 (reset). The manifest declares every reset key exactly once. Every state leaf keeps a fixed shape; positions are [x,y], maps are indexed [y,x], and all actors remain active through termination. Native actor positions are unique; non-agent objects may share a cell with an actor when the mechanic permits it.
Analytical ranges used during author review: wood in [0,4]; grain in [0,4]; fuel_pack in [0,3]; bridge in [0,2]; ferry_fuel in [0,8]; cooldown in [0,4]; signal in [-1,5]; signal_age in [0,17]; water in [0,3]; timber_delivered in [0,2]; grain_delivered in [0,2]. These are source assertions awaiting independent native checking.

Termination:

Natural completion when all eight towns have both delivery counters equal to 2; otherwise truncate at 640.
The terminal helper candidate.py:L67-L77 (end_step) increments t once for nonterminal input, gives natural completion priority over horizon truncation, and preserves all leaves and causes on already-terminal input with zero rewards. Joint dones agree with done_all. No reset occurs inside raw_step.

Source-derived partner intervention:

Build (10) versus wait (0), at the first branch step. Both branches then use the same build and westward movement actions.
The branch-A bridge is intended to reach level 2 while branch B reaches level 1; the same actor 6 west action can then have different movement feasibility. No native transition was evaluated.
The two full literal diagnostic scripts and both paired suffixes are in playthrough.json. They are dimension/range-checked as JSON only. Prefix and branch actions were chosen from rules, not searched, simulated or replayed. They are not a solvability, success-rate or causal-effect receipt.

Presentation and provenance:

The scene binds the exact source and manifest hashes, actual position/active/visibility leaves, tile map, actions and observed attributes. Crew index, actor identity, current tool and object state support later camera/actor selection. Scene metadata is delivered; native animation, RGB rendering and a human-play UI have not been executed or accepted. Any human control must call the same native transition as a policy.
The sanitized authority and scope files govern these initial ordinals and current counts. Historical count examples are not applied. All input files are enumerated with bytes/SHA256 and truthful semantic-read scope in author_input_freeze.json. Five allowed official reference snapshots were hashed for integrity only; their contents were not used as design/code inputs. No upstream code or commercial art was copied.
All small integer quantities use int32; no uint32 PRNG key is duplicated in state. Requested model configuration is gpt-6-astra/max; this author self-report does not establish provider model identity or token/cost telemetry.

Evidence boundary:

Only source AST screening, data-schema checking, symbolic observation-width accounting, reset-key coverage and literal-script structure checking were performed. No candidate was imported or executed; no JAX/numpy simulation, native reset/step, R0, eager/JIT/vmap comparison, benchmark, training, heldout evaluation, browser or downstream feedback was used. Intra-author static corrections to whitelist-rejected constructs occurred before this initial freeze and are recorded in generation_record.json. Native compatibility, replay legality, strategic richness, solvability, learning, UI quality and transfer remain unestablished.
