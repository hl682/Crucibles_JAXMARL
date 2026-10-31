# Winter Road Confederation

Status: AUTHORED_NOT_EXECUTED; repair attempt remains unvalidated. Candidate game-v3-00230, source attempt 1, train-only prospective source stream. Assigned author identity: gpt-6-astra / max; root dispatch is the authority for actual model execution. A/O/U/H = 192/224/23/896. Source SHA-256: 4131c3b22a610ef51cb5cd6ee32ff15fc21f4355ce0d90c7c98906edd7635d39.

## Engineering repair and prior failed attempt

This is the separately requested first engineering repair of the same existing game. The original attempt_00 source SHA-256 is 2b8f2fa14a225ee7fbbec05505a3b8548e5aada3124fd0f28db39965460ede70, and its manifest SHA-256 is 6998bc19652aba7da432245c160c0ab151d7e2182bac31cb9ec6b0cefacb4da2. The supplied sealed engineering feedback SHA-256 is 786d4b757cb376603cbba12a82167723d022e5dab5bc0b1cff5ae350c57616b3. It records FAILED_OR_INCOMPLETE: 8 reset calls, 0 completed joint steps and 0 agent steps. The first raw_step JIT compilation failed in act at candidate.py line 260 with TypeError: neg does not accept dtype bool. The unfinished raw-step operation is not counted as a completed transition. The original failure remains part of the record.

The sole source edit is on line 260: the numeric reward term now uses -move.astype(jnp.float32) instead of -move. The comparison-defined movement flag remains boolean elsewhere; only its reward indicator is converted to float32 before unary negation. All constants and all other source bytes are identical. This encodes the already documented movement-cost meaning without changing its coefficients, eligible actions, map, state, observations, update order or terminal logic. It is a source-level engineering correction, not a measured reward change or an easier replacement.

Both original literal scripts, every seed, the paired prefix and both branch action arrays are retained byte for byte except their top-level source hash binding. Manifest content changes only its source hash; scene content changes only its source and manifest hashes. No script or paired intervention has been executed for attempt_01, and no successful compilation, replay, R0, solvability, training or scientific result is claimed.

## Native game and dependencies

Sixteen settlements occupy a connected 64 by 48 square-grid seasonal plain. Each has twelve separately controlled actors, with two initial copies of six complementary tool roles: survey, build, ferry operation, haul, scout and diplomacy. These are people in one world, not independent copies of a small environment. Each settlement's first hauler begins at the previous settlement's checkpoint carrying one provision; the other actors begin near their own roads, ferry and depot. Perimeter routes around river segments remain possible alternatives.

Plain movement sets a recovery counter to 2, which falls to 1 in the same environmental pass; the following tick must therefore recover before another move. Road movement avoids that delay. Entering a road with provision cargo consumes one durability. Bridges use that same actual road grid: strength at least 2 permits entry, except deep frost requires 3. Actors already on a degraded bridge may leave; the rule blocks new entry rather than retroactively teleporting or killing its occupant.

The public season is floor(t/64) modulo 3. Season 0 permits ferry trips, season 1 is deep frost, and season 2 is thaw. On the transition into thaw, at ticks 128, 320, 512 and 704, every positive road loses one durability. Material gathered from the home depot can build roads or maintain bridges, or a ferry operator can consume it to add two ferry charges. Each ferry trip consumes a charge and foreign riders pay one coin.

Checkpoint g belongs to settlement g. Only its owner's diplomat can change its open state or toll. A closed checkpoint can still admit a team holding a local temporary permit; a visitor buys that permit for one coin. A purchase writes TTL 8 before the end-tick decrement, so the next returned state contains 7. Permit ownership is by team, creating a benefit for other members of the same visiting settlement. Foreign entry still pays the currently posted toll, even with a permit. Toll rates cycle through 0, 1 and 2. Actor purses are finite; receipt banks cap at 4096.

A home depot provides finite construction stock and provision loads. A carried provision keeps its origin through handoff; a different settlement may accept it until its imported-order count reaches two. Scouts can raid another settlement's provision stock or reduce a facing road's durability. A survey samples a cell four steps ahead and stores its site and coarse condition. A later broadcast delivers that stored report to actors within Manhattan distance 8. The stored report can become stale because the world can change.

Plausible choices include maintaining roads for everyone, operating a toll crossing, granting short permits to a visiting team, using a ferry instead of a degrading bridge, carrying materials rather than provisions, taking slower perimeter routes, and exploiting opened passages for raids. Investment and conflict alter real connectivity, costs, local information and payoffs. No strategy has been shown superior.

## Actions, observations and presentation

Actions 0..8 are common movement/facing controls. Actions 9..22 survey, gather material, build, load/deliver provisions, ride ferry, supply ferry, toggle checkpoint, set toll, buy permit, raid, sabotage, hand off, broadcast and swap tools. The manifest describes each exact label and eligibility. Crossing fees and collision checks happen when that actor's cyclic priority turn resolves.

The 224-column observation has 16 self values; a 49-cell three-channel terrain/road/cargo-occupancy patch; four detailed nearby actors; local checkpoint, settlement, ferry and bridge records; and a four-field heard report. All local structure records are zero outside radius 3. Other teams' permit TTLs remain hidden and are deliberately absent from rendered checkpoint attributes. The current season is public; remote queues and map damage are not.

The scene uses settlement flags, territorial checkpoint objects, ferry landings and the actual road-cell entities. Its square-grid topology is preserved; no hex-map geometry or separate strategic dashboard substitutes for native movement. Road objects have capacity 3072, but only cells with positive durability are active. They are infrastructure, not additional agents.

## Rewards and termination

reward_parts has six columns with explicit recipients.

- Column 0 charges the actor 0.005 for a road move or 0.02 for a plain move, 0.02 for build/ferry supply/sabotage, and 0.005 for an unavailable requested move. A legal action at the pre-step boundary can become unavailable during cyclic contention.
- Column 1 gives the actor 0.3 per foreign provision delivery and 0.1 per raid.
- Column 2 gives every member of the receiving settlement 0.1 per delivery, or charges every member of the raided settlement 0.1 per raid.
- Column 3 gives actors within Chebyshev distance 3 of a bridge a 0.03 milestone component when a builder changes that bridge from strength 1 to 2. This is an actual repair event; it is not an arbitrary vote.
- Column 4 charges the paying actor 0.02 per crossing/permit coin paid.
- Column 5 gives each member of the corresponding owner settlement 0.002 per collected checkpoint, ferry or permit coin.

The declared interval [-64, 64] conservatively bounds at most 192 action resolutions and these bounded event contributions per actor. It is an interface range, not a target. Deposits, purse caps and completed-order caps keep observation normalizers finite. Provision handoff preserves its origin; road strength remains 0..3.

Natural completion is two imported provisions received at all sixteen settlements. Otherwise H=896 truncates, with natural priority at the same tick. All 192 actors remain active until joint termination.

## Hypotheses and limits

The first literal script attempts a diplomat's open/toll decision and a neighboring settlement hauler's paid passage, with actual recovery ticks and a later survey broadcast. The second attempts a built bridge, a loaded crossing that consumes durability and a ferry trip made possible by another actor's supply.

The paired hypothesis changes only owner diplomat 5's open action, keeps both suffixes as identical waits, and inspects visitor actor 15's feasible east move. It expressly does not claim collected tolls without actual entry. Source syntax and metadata checks do not execute that comparison. JIT/vmap/scan parity, costs, human play, non-isomorphism, strategic quality, learning and transfer remain unverified.

## Literal source evidence

These line references identify authored implementation expressions, not measured effects.

- Repaired numeric movement-cost indicator: candidate.py line 260, literal source: `rp = rp.at[i, 0].add(-move.astype(jnp.float32) * jnp.where(road > 0, 0.005, 0.02) - 0.02 * (build | operate | damage) - 0.005 * blocked)`.
- Gate/bridge topology and fees: candidate.py line 25, literal source: `def walkable(self, s, i, xy):`.
- Roles and opportunity conditions: candidate.py line 39, literal source: `def legal_row(self, s, i):`.
- Foreign hauler placement: candidate.py line 100, literal source: `pp = pp.at[:, 3, :].set(base[(g + 15) % 16] + jnp.array([7, 5], jnp.int32))`.
- Loaded road wear: candidate.py line 209, literal source: `wear = move & (road > 0) & (cargo > 0)`.
- Actual bridge investment event: candidate.py line 263, literal source: `milestone = build & (s["tiles"][q[1], q[0]] == 3) & (s["road_grid"][q[1], q[0]] == 1)`.
- Team permit state: candidate.py line 248, literal source: `z["permit"] = z["permit"].at[g, team].set(jnp.where(negotiate, 8, z["permit"][g, team]))`.
- Scheduled infrastructure loss: candidate.py line 276, literal source: `thaw = ((s["t"] + 1) % 192) == 128`.
- Absorbing native boundary: candidate.py line 291, literal source: `def raw_step(self, key, state, actions):`.

## Read-only reference use for this repair

The current read set is exactly the 17 physically supplied files recorded in author_input_freeze.json: INPUT_MANIFEST248.json, request.json, feedback.json, the seven files under inputs/, and the seven copied files under previous_attempt00/. Every input was rehashed against the sealed packet or the dispatch identities. The current request overrides the copied original-author contract only for this specifically assigned attempt_01 repair, allowed previous-attempt input and seventh output.

The original generation record and input freeze are historical evidence read as provided files. Their historical external paths, former batch scope, reference names and no-execution statements describe the prior author turn; they are neither current read permissions nor a current execution status. None of those external paths or reference links was followed. The current supplied static whitelist is inputs/trusted/code_contract.py, SHA-256 d8702b586664c9bd6828ce5a0393dbbb97606a63e1404043b69a3c25cc28b09e. No other game, memory, training/development/heldout material, browser, network or remote system was accessed.

## Boundary and evidence status

The native interface is reset(key), observe(state), legal_actions(state), and raw_step(key,state,actions). All state leaves are fixed-shape int32, bool or float32 arrays declared in manifest.json. Keys are accepted but intentionally unused: these original rules and layouts are deterministic. Small integer quantities use int32 at the boundary; there is no hidden PRNG key in state.

The joint input is one int32 action per actor. Resolution uses rotating priority (k + input t) modulo A. Each actor's action is rechecked against the current intermediate state, so contention can turn a previously legal intention into a deterministic wait. Environmental processes run once afterward. This is a simultaneous input interface with explicitly sequential conflict resolution, not a claim that actions commute.

A nonterminal step advances the int32 clock once. Natural completion takes priority over H. An already-terminal raw call preserves the entire state, observation, clock and terminal causes and returns zero reward; it never auto-resets. Event flags refer to the last actual nonabsorbing step and are not new events on repeated absorbing calls. A separate trusted wrapper or public adapter remains outside this author package.

During this repair author turn, no candidate was imported or executed. No JAX/NumPy runtime, provided validator, trajectory script, PRNG test, local model, remote job, HTTP endpoint, browser, training or evaluation was run. Standard-library AST parsing, exact source-byte comparison, JSON-field comparison and SHA-256 checks are not native validation. The prior attempt did undergo the failed sealed validation recorded above; these no-execution statements apply to the new repair authoring only. This is one authorized repair (1 of 2 allowed) of existing game-v3-00230, with no new proposal, mutation, game ID or production reservation. Source authoring is not an R0, causal, animation, solvability, learning or transfer receipt. Further execution requires the separately controlled sealed process and is outside this author assignment.

