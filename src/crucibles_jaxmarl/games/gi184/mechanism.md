# Rotating Keep — original source-only mechanism

Status: AUTHORED_NOT_EXECUTED. This is a source specification with original unexecuted scripts. Static source/data checks do not establish runtime correctness, successful gameplay, animation, causal validity or learning.

Three mobile couriers occupy a 9 × 7 fortress separated by a wall at x=4. Its only crossings are the north threshold (4,2) and south threshold (4,4). Phase 0 desires north open and south closed; phase 1 desires the reverse. The initial phase is 1. Actors start at (2,3), (3,2), (6,3). No phase change moves a coordinate. The map's threshold value is architectural metadata; actual traversal consults the corresponding door_open flag.

The amber console at (2,3) requires key 0; the azure console at (6,3) requires key 1. Each has three finite request charges. Keys initially belong to actors 0 and 2, respectively. A key has exactly one carrier or a ground coordinate. An actor can carry at most one key. Ground pickup requires the same cell; dropping leaves the key underfoot; handoff requires an adjacent waiting keyless actor. Carried key positions track their owner's actual position. Keys are never consumed or duplicated.

Action 5 at a matching console requests the other phase. There is one shared pending slot. All requests are admitted from the old state, and cyclic actor priority picks at most one; only the selected console spends one charge. A new request sets timer 2. On each subsequent transition the existing timer decreases. When it reaches zero, the phase changes after movement. No request is accepted on a call whose input still has a pending request, including its maturity call. This avoids instant cascaded rotations and fixes the earliest peer-use time.

A desired-closed threshold remains actually open while occupied by any actor. The door_jammed flag means occupancy defers closure. Movement cannot enter an input-state occupied threshold, so an occupancy hold does not let a second actor overlap its holder. Once the occupant exits, the end-of-transition openness becomes the desired state and the door closes. Actors can leave the threshold normally without teleportation, pushing or coordinate repair. Holding a door therefore costs the holder's location and future movement opportunity.

Movement is assessed against the old topology and occupied positions. Following into an old occupied cell and swapping are disallowed. Competing empty destinations use cyclic priority (actor minus clock modulo three). Key transactions and claims are then processed in cyclic order with current ownership checks. Pending timers mature after movement; a new admitted request is installed; actual door openness is recomputed from the resulting phase and final occupancy. All out-of-range or illegal actions become wait.

Destinations are the east pad (7,1), identity 0, and west pad (1,5), identity 1. Each courier can claim one pad by standing exactly there and using action 7. Its own preferred pad gives 2 instead of 1, plus every actor receives 0.25 per claim in that joint transition. A completed courier stays mobile and may still operate a console, carry a key or hold a threshold. Joint natural completion requires all three claims, not a fixed reward. H=96 bounds unfinished cases.

The reset key samples one bit b and fixes preferences [b,1-b,b], creating conflicting destination priorities without changing the physical layout. Each row observes only its own preference. Public information is the map, actor/key coordinates, key holders, claims, actual doors, current/pending phase, timer and console charges. O=88 comprises 86 such public fields plus own preference and own index. Desired door openness and jam flags shown in the scene are exact public derivations of phase and occupancy, not extra private observations. Private preference and realized private utility are not scene attributes or events.

Complementary keys, finite charges and delayed global topology create partner dependence. One actor may spend a charge to let another cross, lend a key and pursue its own pad, or wait while a peer holds an occupied threshold. An opposing preference may favor spending the next request on the other corridor, or conserving charges for a return trip. Both north and south routes are physical alternatives, with travel and occupancy costs. These are prospective strategies, not tested best responses or proven strategic depth.

The paired intervention changes only actor 0's first console action into wait. With a request on call 1, the timer ages on calls 2 and 3, so actor 1's east mask can first use the opening on call 4. Both branches retain the same following east actions; an unavailable move is converted to wait. The comparison targets another actor's later legal movement, not a reward counter. A second original script asks a peer to occupy the north threshold during a later south rotation, then leave so deferred closure can occur. Neither script was executed.

Source invariants awaiting numerical checks are: unique walkable actor positions; exactly two conserved keys; one key per holder at most; charges 0–3; pending -1/0/1; timer 0–2; phase 0/1; and actual door openness equal to desired openness OR occupancy. A closed threshold cannot contain an actor, by that same closure rule. Completion does not remove or teleport actors. The per-step reward bound [0,3] contains the source-derived maximum 2.75; observations lie in [-1,96].

Natural completion takes priority if the final claim also reaches H. An already-terminal raw call preserves state, clock and observation exactly, returns zero reward and all dones true, and preserves its natural/truncated cause. The raw kernel never performs autoreset. int32 stores small signed integer fields, float32 carries observations/rewards, and bool stores flags; the external reset key is not duplicated in state.

No code was copied from supplied official references; they were byte/hash checked only. No unsupplied game, prior source, training/selection/heldout result or memory was consulted. The proposed adapter, actual JAX parity tests, policy-bank dimensions, native recorded animation and scientific learning/transfer evaluation belong to independent later integration. None is an author-side completion claim.

## Literal source evidence

- candidate.py:64: `request = atswitch & holds & (s["switch_charge"][None,:]>0) & (s["pending"]<0)`
- candidate.py:128: `due = (state["pending"]>=0)&(state["timer"]==1)`
- candidate.py:138: `s["timer"] = jnp.where(accepted,2,s["timer"])`
- candidate.py:139: `s["switch_charge"] = s["switch_charge"].at[console].add(-accepted.astype(jnp.int32))`
- candidate.py:144: `s["door_open"] = target | occupied`
- candidate.py:145: `s["door_jammed"] = ~target & occupied`
- candidate.py:151: `s["natural"] = jnp.all(s["delivered"])`
- candidate.py:153: `s = jax.tree.map(lambda new,old:jnp.where(state["done"],old,new),s,state)`

Interface anchors from static AST: reset at candidate.py:12–47; legal_actions at candidate.py:73–78; observe at candidate.py:80–86; raw_step at candidate.py:88–159.

The literal anchors and declared shapes were inspected statically. They are not execution traces or a proof that the implementation satisfies these claims under JAX.
