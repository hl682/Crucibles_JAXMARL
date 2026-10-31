# Echo Passage — lures, patrols and shared exposure

Status: AUTHORED_NOT_EXECUTED. This initial train-only source was written for game-v3-00123, ordinal 123, attempt 00. No candidate import, compilation, transition, rollout, JIT, vmap, numerical parity check, rendering, training or test-set use occurred. Source SHA256: a68e16ea61f5486530ed5dbdd03805b0985192578932dbf91634ce11ae39e814.

## Boundary and lifecycle

A=3, O=356, U=8, H=112. All state leaves have a fixed declared key, shape and int32/bool dtype; observations and rewards are float32. Reset consumes only its supplied key; no external state, time, cache, file or network is part of the kernel. The scalar t starts at zero. A nonterminal call advances it once. Natural termination takes precedence over the horizon. An already-terminal call preserves every leaf and observation, preserves its terminal cause and clock, and returns zero rewards and all dones true. There is no raw auto-reset. The final tree operation preserves each reset leaf's dtype explicitly.

Every in-range action is a legal command attempt while the episode is live. A blocked movement, absent resource, wrong recipe, unavailable object or cooldown causes the specified deterministic wait/no-effect outcome. Out-of-range integers map to action 0. At joint termination only action 0 is legal. No role is removed independently; completed contributors can still help until joint completion. The role adapter must preserve these masks and semantics.

- reset: candidate.py:45
- observe: candidate.py:85
- legal_actions: candidate.py:82
- raw_step: candidate.py:118

## Actual mechanism and timing

The 13 by 11 public map has a partition at x=6, permanent passages at y=2 and y=8, and a locked shortcut at (6,5). Crew starts at (2,4), (2,5), (2,6); two cases start at (10,3), (10,7), and extraction is at (1,5). The gate tile is static map information; its changing open state has separate local/radio visibility. Geometry and passability are at candidate.py:14 and candidate.py:49.

Each step first handles a valid role-0 unlock adjacent to the gate, then simultaneous movement. Destinations occupied at the old step and multiply requested destinations are blocked, including swaps. Interact-to-interact handoffs use a snapshot of original carriers/empty recipients, choose each donor's lowest-index eligible receiver, and then the lowest-index competing donor. This snapshot prevents a case from being passed around multiple times in one transition. Ground pickup and extraction follow in fixed object/agent order. A case is always either on the floor, carried by one role, or extracted; carried coordinates follow the actual carrier. See candidate.py:122, candidate.py:135, candidate.py:137, candidate.py:146 and candidate.py:159.

Every crew member has three lure charges and two radio charges. A lure emits strength 6; a movement made while carrying at the start of the step emits 4; a newly opened radio emits 3; unlocking emits 5. The strongest new emission for an actor becomes its source at its current coordinate; with no new emission its previous source stays in place and loses one strength. The acoustic field at each cell is the maximum of max(0, source strength minus Manhattan distance). It deliberately propagates through walls and reveals an audible source's coordinate exactly, which is an explicit information model rather than physical acoustics. Emission and the actual field are at candidate.py:164, candidate.py:165, candidate.py:168 and candidate.py:19.

Each patrol chooses the strongest positive audible score, with source-index tie breaking. It retains an unrefreshed target for its attention countdown; when attention is exhausted it follows a fixed four-waypoint schedule, offset between patrols. It tries an x-directed step first, then a y-directed fallback if blocked; it has no hidden route search or optimal pathfinder. Patrols can share a cell. One-step motion and wall/gate checks are literal at candidate.py:170, candidate.py:175, candidate.py:180 and candidate.py:182.

After patrol movement, a crew member is exposed if Manhattan distance to a patrol is <=1, or if that distance is <=3 and the actual acoustic intensity at the crew member is >=2. Thus one actor's lure can reveal another actor. Shared alarm increases by the number exposed, clipped to five. It decreases by one only when all three choose wait and none are exposed. Luring and carrying can buy a route for one teammate while increasing another's future risk. Exposure, quiet coordination and alarm are at candidate.py:184, candidate.py:186 and candidate.py:188.

A radio is a three-tick live channel, not a cached invented sensor. Everyone receives that sender's current position and carried-object state; everyone also receives any patrol or gate information inside an active sender's own radius. Outside those conditions, the returned observation zeroes hidden entity blocks and includes visibility flags. The public map does not leak current gate openness. Source nodes are shown only if audible or owned; positive acoustic cells are shown only within the observer's radius-two area. Patrol targets, unseen inventories and hidden gate events are not scene channels. See candidate.py:23, candidate.py:28, candidate.py:34, candidate.py:36, candidate.py:38 and candidate.py:85. Local radius uses Chebyshev distance without wall occlusion, exactly as documented.

Each extraction pays 0.5 to every role; each exposed crew member subtracts 0.03 from everyone's reward. The lure user additionally pays 0.01. Joint success adds one and alarm failure subtracts one. Positions, resources, carried cases, noise, alarm and gate state persist. There is no independent agent elimination, and no reward threshold is an acceptance rule.

Plausible alternatives include a quiet two-route infiltration, a locksmith shortcut with coordinated distraction, or relays that change who bears noisy carrying movement. A radio can replace risky scouting but spends a charge and emits sound. Case transfer changes route and exposure responsibility. Fixed hypotheses compare a lure with silence and radio disclosure with silence; they do not establish successful extraction. Prospective variations in hearing attenuation, patrol attention, gate placement or radio access could change partner best responses. No such transfer result is claimed.

## Observation contract

- Columns 0 through 1 (self): Own xy divided by 12
- Columns 2 through 4 (self): Own lure charges /3, radio charges /2 and locksmith tool remaining
- Columns 5 through 7 (self): Own carried-object one-hot: empty, object 0, object 1
- Columns 8 through 25 (local): Three agent blocks: xy /12, visible flag, carried-object one-hot; whole block zero unless Chebyshev range <=3 or that agent has an open radio
- Columns 26 through 33 (local): Two patrol blocks: xy /12, visible flag, attention /3; visible locally or through an active sender local to that patrol
- Columns 34 through 45 (local): Three sound-source blocks: xy /12, strength /6, visible flag; zero unless active and own or strictly audible
- Columns 46 through 53 (local): Two mission-object blocks: xy /12, visible flag, (owner+2)/4; zero unless active and locally seen or carried by a visible agent
- Columns 54 through 57 (local): Gate xy /12, visible flag, open flag; zero unless local or visible via an active radio sender
- Columns 58 through 59 (public): Public exit xy /12
- Columns 60 through 62 (self): Own role identity one-hot
- Columns 63 through 63 (public): Shared alarm /5
- Columns 64 through 64 (public): Clock /112
- Columns 65 through 66 (public): Public extraction-completed flags per mission object
- Columns 67 through 69 (public): Public radio-channel TTL per agent /3
- Columns 70 through 212 (public): Public static 11 by 13 map in yx order /2; the gate tile denotes location, not current open state
- Columns 213 through 355 (local): Actual acoustic intensity /6 at cells within own Chebyshev radius 2, zero elsewhere; visible domain follows own position

The returned observation is the policy information boundary. Entity visibility and declared attributes refer to the same actual state and access rules. The complete static tile layout is public; a debug/full-state display must be labelled separately from a role view. Undeclared internal leaves are not an invitation to expose private values in a role renderer.

## Fixed author scripts and paired interventions

playthrough.json contains fixed joint integer commands, an explicit seed and UNEXECUTED_AUTHOR_HYPOTHESIS status for each script. Stop at the first done. These scripts were constructed from the written rules without local execution, simulation, numerical action search or feedback from other candidates. Prefix/branch comparisons state predicted causal differences only; there is no observed trace, successful episode, solvability certificate, ranking or learned policy.

## Community provenance and delivery boundary

Only this batch's hash-verified design/request, 20 listed inputs and copied public references were used. JaxMARL's README supplies multi-agent integration context, including agent-keyed public APIs; Pgx supplies JAX-native game-state and legal-action-mask design context. Craftax supplies an example of spatial compositional game mechanics. PureJaxRL and the Craftax PPO RNN baseline are algorithm/integration references, not a game library, copied environment, or replacement learner. No upstream source lines, layouts or benchmark rules were copied into this original kernel. Exact copied-input hashes and original official URLs are retained in generation_record.json.

The coordinator owns the separate public JaxMARL adapter, sealed runtime validation, actual state replay, RGB/animated presentation, playback controls and community review. This authored kernel and declarative scene do not establish those acceptance outcomes. Structural/AST screening is separate from numerical validation. Passing future interaction or interface checks would not by itself establish strategic richness, reward learning, meta-learning, non-isomorphism or transfer. No new scientific performance gate is added.
