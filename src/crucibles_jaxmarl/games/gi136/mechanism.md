# Turnhouse Compact

A=3, O=168, U=10, H=96. The map is a 13 by 7 bounded grid. The three 3 by 3 room interiors are x=1..3,5..7,9..11 at y=2..4; their centre stations are (2,3),(6,3),(10,3). The only other potentially walkable cells are six north/south bays at each centre x and y=1,5, plus bridges (4,3) and (8,3). Remaining cells are walls.

## Exact rotating geometry and passage semantics

A room orientation is an integer 0..3. Orientation 0 opens north/east, 1 east/south, 2 south/west, 3 west/north. Initial orientations are [0,2,2]. A bay's requested openness equals that room's corresponding port. A bridge's requested openness is the conjunction of the facing east and west ports of its two rooms. The state tile map is rebuilt from these exact booleans; the renderer uses this map rather than inventing a visual rotation.

A courier within Manhattan distance one of a station may request clockwise or counterclockwise rotation if it has energy. Requests are simultaneous. If a room receives both signs, it does not move and nobody pays. With one sign only, the room executes exactly one quarter turn and every requesting supporter in that room pays one energy. The three roles each start with four energy and there is no recharge.

For an interior point with offset (dx,dy) from the centre, clockwise is (-dy,dx) and counterclockwise is (dy,-dx), with y pointing down. All active couriers and undelivered cargo inside that room undergo this same bijection, including non-requesting partners and a carrier's cargo. Centre stations, bays and bridges do not move. Delivered cargo and escaped couriers retain their final coordinates.

A passage requested closed stays open if its actual cell contains an active courier or any undelivered cargo. This occupancy safety interlock prevents a rotation from turning a live occupant's cell into a wall. It also permits deliberate holding-open with a person or dropped crate. The map/interlock is refreshed after rotation and again after movement, delivery and escape. A passage vacated during a tick may close before the next decision; an occupant can leave into a neighbouring room even if its port orientation is no longer aligned.

Input masks are applied before any transitions. Valid control requests rotate rooms first. Then compass movement attempts are interpreted from the rotated positions using the new map. Moves into another active courier's post-rotation starting cell or into a contested destination fail; swaps and same-tick following are excluded. Cargo on the floor does not obstruct movement. Escapees no longer block cells. Floor cargo can overlap after drops; interact deterministically chooses the lowest-index available item.

## Cargo, incentives and completion

Cargo 0 starts at (1,2) and belongs to role 0, with destination (6,5). Cargo 1 starts at (5,4), belongs to role 1, and must reach (10,1). Cargo 2 starts at (11,2), belongs to role 2, and must reach (2,1). Ownership identifies reward recipients; it does not restrict who can carry. Each role can hold one item. Interact while empty-handed picks an undelivered ground item on the same cell; while carrying it delivers only at that item's specified destination. Drop leaves the item where the carrier stands. Neither ownership nor destination changes under turns or handoff.

Every delivery gives all roles 1, the physical deliverer 1 additional, and the owner 3 additional. Thus an agent can help a partner while benefiting itself, but owners have unequal interests in which route is opened first. Escape is optional from any north/south bay when empty-handed. Escaping removes the role's movement/control contribution. The episode ends naturally when all three have escaped, whether or not the cargo mission succeeded. If all cargo was delivered, this final step also gives every role 4. Otherwise there is no completion bonus. The H=96 time limit truncates remaining work. Per-step reward is within [0,12].

Early evacuation is deliberately a legal zero-delivery terminal outcome. It is not an acceptance failure invented by a score threshold. The provided short evacuation script is an unexecuted hypothesis about this branch. A separate cargo/rotation script requests a pickup, a carried turn, a partner crossing and a second turn that physically relocates that partner, followed by a drop.

## Causal interaction and possible conventions

A controller can rotate a non-acting partner, open its route, close its future exit, or cause its compass action to start from a different cell. Opposing requests cancel each other and preserve energy, while redundant agreeing requests waste multiple agents' finite energy for one room turn. The cross-room bridge requires compatible orientations at both ends, so an isolated courier cannot always set its own route independently.

Couriers can agree on passage schedules, send a cheap operator ahead while another hauls, or stage a crate as a temporary door holder before collecting it again. Conversely, a courier may favor its own destination, withhold control cooperation or leave early. Keeping a route open and advancing a cargo can conflict when the very crate serving as the interlock must be moved. These are partner-dependent alternatives; no equilibrium, learned schedule or success rate is claimed.

One pair compares clockwise and counterclockwise requests by role 0 while role 1 makes the same westward attempts near the first bridge; the intended difference is role 1's later reachable cells. Another pair puts two roles near the same station and compares agreement versus opposite requests; the intended effects are actual orientation, displaced partner position and energy spending. Both share literal keys and remain unexecuted.

Alternative destinations, initial orientations, energy endowments or room-port patterns could alter useful schedules. Such variants would require their own frozen definitions and evidence.

## Rendering and intended invariants

Room-station entities bind public centres and actual orientation categories. Gate entities bind actual passage cells, gate_open and the persistent jammed level. Courier attributes expose only public energy, carried IDs and last control request. Cargo/goal pairs preserve their owner-index binding and show physical transport rather than score-driven animation. Escaped couriers and delivered cargo are inactive; their final positions remain in the fully observable state. All native state needed for visibility and map interpretation is public.

Intended invariants: turn energy remains nonnegative; active courier cells are unique and walkable; no live object rotates outside its room; carried objects coincide with their carrier; each courier carries at most one item; cargo is neither created nor duplicated; ownership and destination are constant; delivered cargo is uncarried/inactive; occupied passages cannot close; orientations remain modulo four. Rigid rotation preserves interior uniqueness by construction, but these are source-level claims awaiting independent numerical validation.

## Literal source evidence

Evidence refers to the frozen candidate.py with SHA-256 cc7856a8a488e1142d45a1afda813dffde0080112144fca73f95123eb4938895. Line numbers were obtained by parsing/reading source text without importing it.
- candidate.py:46-76: reset implementation.
- candidate.py:78-92: legal_actions implementation.
- candidate.py:94-166: raw_step implementation.
- candidate.py:168-181: observe implementation.
- candidate.py:24-44: refresh implementation.
- candidate.py:106: turns = jnp.where(clockwise & ~counter - Opposite simultaneous control requests cancel.
- candidate.py:107: paid = (command != 0) - Every supporter of an executed turn spends energy.
- candidate.py:33: opened = desired | agent_here | cargo_here - Physical occupancy holds a closing passage open.
- candidate.py:112: rotated_agents = self.rotate_points - One agent's control physically moves room partners.
- candidate.py:144: reward = reward.at[s["owner"][h]].add - Cargo ownership changes incentives independently of carrier identity.
- candidate.py:156: natural = jnp.all(n["escaped"]) - Natural termination permits incomplete-cargo evacuation.

## Boundary and source provenance

This package is an initial author artifact, status AUTHORED_NOT_EXECUTED. It has not been imported, reset, stepped, JIT-compiled, vmapped, trained, rendered, or subjected to action search. Source AST screening and the copied data-only manifest/scene validators are separate from R0 and numerical parity. All playthroughs and paired interventions remain UNEXECUTED_AUTHOR_HYPOTHESIS; a later executor must use their literal keys and record actual legality, causes and failures.

Nonterminal raw steps advance int32 t once. Natural termination wins at H; truncated=(t>=H)&~terminated and done is their union. Already-done inputs preserve all state and observation leaves, clock and causes, return zero reward and all dones true. No raw reset occurs. Invalid/masked actions become wait; movement attempts can be blocked. State uses fixed-shape int32/bool/float32 arrays; keys remain external.

A trusted JaxMARL dictionary adapter, real replay/animation, browser controls, sealed R0, numerical parity and scientific comparisons remain independent work. No acceptance or learnability claim is made.

The supplied JaxMARL README informed parallel-agent API integration, Pgx README explicit keys/masks, Craftax README persistent spatial state, and PureJaxRL README functional training integration. No upstream code, game rules, layouts, assets or weights were copied. Craftax_Baselines_ppo_rnn.py was hash-verified only. These references do not replace the frozen learner.

Only the assigned batch design/request, contracts/parsers and copied references were read. All 20 input hashes/byte sizes and the authorization SHA matched. No outcomes, candidate sources, replays, logs, rankings, heldout files, result_N or another author's directory were read. Source split is train; final_test_used=false. The coordinator retains dispatch identity. Potential variants are ungenerated hypotheses. Static screening cannot prove solvability, non-isomorphism, strategic richness, learning or transfer; failures and indeterminate results remain valid.
