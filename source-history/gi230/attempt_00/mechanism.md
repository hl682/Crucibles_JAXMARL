# Winter Road Confederation

Status: AUTHORED_NOT_EXECUTED. Candidate game-v3-00230, source attempt 0, train-only prospective source stream. Assigned author identity: gpt-6-astra / max; root dispatch is the authority for actual model execution. A/O/U/H = 192/224/23/896. Source SHA-256: 2b8f2fa14a225ee7fbbec05505a3b8548e5aada3124fd0f28db39965460ede70.

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

- Gate/bridge topology and fees: candidate.py line 25, literal source: `def walkable(self, s, i, xy):`.
- Roles and opportunity conditions: candidate.py line 39, literal source: `def legal_row(self, s, i):`.
- Foreign hauler placement: candidate.py line 100, literal source: `pp = pp.at[:, 3, :].set(base[(g + 15) % 16] + jnp.array([7, 5], jnp.int32))`.
- Loaded road wear: candidate.py line 209, literal source: `wear = move & (road > 0) & (cargo > 0)`.
- Actual bridge investment event: candidate.py line 263, literal source: `milestone = build & (s["tiles"][q[1], q[0]] == 3) & (s["road_grid"][q[1], q[0]] == 1)`.
- Team permit state: candidate.py line 248, literal source: `z["permit"] = z["permit"].at[g, team].set(jnp.where(negotiate, 8, z["permit"][g, team]))`.
- Scheduled infrastructure loss: candidate.py line 276, literal source: `thaw = ((s["t"] + 1) % 192) == 128`.
- Absorbing native boundary: candidate.py line 291, literal source: `def raw_step(self, key, state, actions):`.

## Read-only reference use

Only the current batch's authorized files were read. author_input_freeze.json records all 26 actual input files, bytes, hashes and use. The 24 payload files include this batch's four current designs/requests, shared contracts, current population/family control, general design and human-play requirements, and five named official reference snapshots. Three optional historical population addenda were not read. Earlier copied example numbers are noncontrolling.

JaxMARL README informed the distinction between a simultaneous native action-row boundary and a later agent-keyed dictionary adapter. Pgx README informed explicit legal masks, separate termination causes and external key provenance; no Pgx game was copied. Craftax README informed bounded persistent world state and the separation between mechanics and rendering. PureJaxRL README and Craftax_Baselines_ppo_rnn.py were read only for JAX scan/vectorization and wrapper/algorithm separation. No upstream game code, PPO code, assets or performance claims were copied into this kernel. No reference links were opened, dependencies installed, local models invoked or scientific outcomes read.

## Boundary and evidence status

The native interface is reset(key), observe(state), legal_actions(state), and raw_step(key,state,actions). All state leaves are fixed-shape int32, bool or float32 arrays declared in manifest.json. Keys are accepted but intentionally unused: these original rules and layouts are deterministic. Small integer quantities use int32 at the boundary; there is no hidden PRNG key in state.

The joint input is one int32 action per actor. Resolution uses rotating priority (k + input t) modulo A. Each actor's action is rechecked against the current intermediate state, so contention can turn a previously legal intention into a deterministic wait. Environmental processes run once afterward. This is a simultaneous input interface with explicitly sequential conflict resolution, not a claim that actions commute.

A nonterminal step advances the int32 clock once. Natural completion takes priority over H. An already-terminal raw call preserves the entire state, observation, clock and terminal causes and returns zero reward; it never auto-resets. Event flags refer to the last actual nonabsorbing step and are not new events on repeated absorbing calls. A separate trusted wrapper or public adapter remains outside this author package.

No candidate was imported or executed. No JAX/NumPy runtime, validator, trajectory script, local model, remote job, HTTP endpoint, browser, training or evaluation was run. Standard-library AST parsing and data-only JSON/static consistency checks are not native validation. Source authoring is not an R0, causal, animation, solvability, learning or transfer receipt. No additional proposal or repair attempt was created.

