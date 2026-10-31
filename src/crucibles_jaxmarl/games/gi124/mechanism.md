# Bankline — four-player ricochet possession arena

Status: AUTHORED_NOT_EXECUTED. This initial train-only source was written for game-v3-00124, ordinal 124, attempt 00. No candidate import, compilation, transition, rollout, JIT, vmap, numerical parity check, rendering, training or test-set use occurred. Source SHA256: be2e366123e40eb86d3b71476fa3807775d75ba255178d0e118891d2634041c9.

## Boundary and lifecycle

A=4, O=169, U=12, H=120. All state leaves have a fixed declared key, shape and int32/bool dtype; observations and rewards are float32. Reset consumes only its supplied key; no external state, time, cache, file or network is part of the kernel. The scalar t starts at zero. A nonterminal call advances it once. Natural termination takes precedence over the horizon. An already-terminal call preserves every leaf and observation, preserves its terminal cause and clock, and returns zero rewards and all dones true. There is no raw auto-reset. The final tree operation preserves each reset leaf's dtype explicitly.

Every in-range action is a legal command attempt while the episode is live. A blocked movement, absent resource, wrong recipe, unavailable object or cooldown causes the specified deterministic wait/no-effect outcome. Out-of-range integers map to action 0. At joint termination only action 0 is legal. No role is removed independently; completed contributors can still help until joint completion. The role adapter must preserve these masks and semantics.

- reset: candidate.py:18
- observe: candidate.py:48
- legal_actions: candidate.py:45
- raw_step: candidate.py:67

## Actual mechanism and timing

The 13 by 9 arena uses interior x=1..11, y=1..7. Boundary walls and four interior obstacles constrain all player and ball movement. The western and eastern goal mouths are x=1 and x=11 with y=3..5; their scene centers are (1,4), (11,4). Teams are fixed to players [0,1] and [2,3]. The shared ball begins loose at (3,3), and every player has one scalar action per simultaneous transition. Geometry is at candidate.py:21 and movement is at candidate.py:74.

Cardinal movement resolves first. A target occupied by another player's old coordinate cannot be entered even when that player moves away; all multiple-target contests are blocked. This simple no-swap rule makes body blocking a real opportunity constraint without hidden randomized arbitration. All special commands are attempts, and cooldown does not disable ordinary movement.

A carried ball follows the possessor's actual post-movement position. Ready opponents using claim/tackle within Manhattan distance one may take it unless the possessor is using a ready shield. Contests select the smallest (player index minus t) modulo four. This rotating priority also resolves blockers and catchers and is fully determined by the public clock. A new tackler cannot release in the same transition. See candidate.py:80, candidate.py:88, candidate.py:90 and candidate.py:99.

A ready possessor's pass sets velocity to the component-wise sign of the current teammate displacement. This allows diagonal flight but does not predict future receiver motion. A shot uses one of four cardinal unit vectors. Both start a six-tick flight timer and release possession. The first flight update occurs immediately. After every update the timer decreases, and reaching zero stops the ball. There is no teleportation to a teammate or a goal; see candidate.py:102, candidate.py:103, candidate.py:101 and candidate.py:112.

A flight step checks its x-only, y-only and full target cells against actual obstacles. Blocked axes reverse. A diagonal-only corner obstruction reverses both axes; if even the reflected target is blocked, the ball stays and its velocity becomes zero. A ready blocking player within distance one of the post-wall target has priority over catches: that block returns the ball to its pre-flight coordinate for this step and reverses the new velocity. This is a discrete defensive contact rule, not continuous rigid-body physics. All branches are written at candidate.py:113, candidate.py:118, candidate.py:119, candidate.py:127 and candidate.py:130.

An unblocked loose ball is caught on exact body contact, or within distance one by a ready claim command. A just-releasing player is excluded for that transition. Passive exact-contact catches do not need a special action; deliberate extended catches do. The latter, releases, tackles and blocks all set cooldown to two. A catch by the opposite last-touch team or a successful tackle sets the interception event. A teammate's readiness and chosen movement therefore change whether a pass is received, lost or carried into a ricochet. See candidate.py:135, candidate.py:139 and candidate.py:157.

Contacts resolve before scoring. A loose ball in the eastern mouth scores for team 0, and one in the western mouth for team 1, regardless of who last touched it; own goals therefore count against that side. One goal pays +1 to both scoring teammates and -1 to both opponents. No auxiliary pass/catch reward creates a score-farming incentive. On a nonterminal goal only ball position, velocity, possession, timer and touch attribution reset to the arena center. Player positions and cooldowns continue. Reaching score three ends naturally. See candidate.py:147, candidate.py:150 and candidate.py:168.

Candidate strategies include direct carrying plus a coordinated finish, a receiver drawing away a defender before a pass, ricochet shots around obstacles, and alternating a shielded carrier with a ready receiver. A defender must choose between movement, extended tackling and a one-step block; cooldown creates a temporal window for opponents. These are strategy hypotheses, not a solved game or a richness measurement. Proposed variations in obstacles, flight duration or cooldown would change interception timing, but no sampled-rule transfer or meta-learning result is claimed.

The scene binds actual ball position, velocity, owner and timer, all player locations/team/cooldown/block state, and both goal coordinates/scores. The generic goal glyph denotes the center of the three-cell mouth; the rule text declares its extent. Pass, catch, interception, bounce and score flags correspond to source transitions and are not inferred from visual motion.

## Observation contract

- Columns 0 through 7 (public): All four player xy coordinates divided by 12
- Columns 8 through 15 (public): Team identity for each player, two-way one-hot
- Columns 16 through 19 (public): Player special-action cooldowns /2
- Columns 20 through 23 (public): Active blocking/shield flags for the last transition
- Columns 24 through 25 (public): Ball xy /12
- Columns 26 through 27 (public): Ball integer velocity components in -1..1
- Columns 28 through 32 (public): Ball owner one-hot: loose, player 0, 1, 2, 3
- Columns 33 through 33 (public): Remaining flight timer /6
- Columns 34 through 36 (public): Last ball-touch team one-hot: none, team 0, team 1
- Columns 37 through 38 (public): Team scores /3
- Columns 39 through 42 (public): Western and eastern goal center xy /12
- Columns 43 through 159 (public): Public 9 by 13 arena map in yx order
- Columns 160 through 163 (self): Own player identity one-hot
- Columns 164 through 164 (public): Clock /120
- Columns 165 through 168 (public): Latest pass-release, catch, interception/tackle and bounce flags

The returned observation is the policy information boundary. Entity visibility and declared attributes refer to the same actual state and access rules. The complete static tile layout is public; a debug/full-state display must be labelled separately from a role view. Undeclared internal leaves are not an invitation to expose private values in a role renderer.

## Fixed author scripts and paired interventions

playthrough.json contains fixed joint integer commands, an explicit seed and UNEXECUTED_AUTHOR_HYPOTHESIS status for each script. Stop at the first done. These scripts were constructed from the written rules without local execution, simulation, numerical action search or feedback from other candidates. Prefix/branch comparisons state predicted causal differences only; there is no observed trace, successful episode, solvability certificate, ranking or learned policy.

## Community provenance and delivery boundary

Only this batch's hash-verified design/request, 20 listed inputs and copied public references were used. JaxMARL's README supplies multi-agent integration context, including agent-keyed public APIs; Pgx supplies JAX-native game-state and legal-action-mask design context. Craftax supplies an example of spatial compositional game mechanics. PureJaxRL and the Craftax PPO RNN baseline are algorithm/integration references, not a game library, copied environment, or replacement learner. No upstream source lines, layouts or benchmark rules were copied into this original kernel. Exact copied-input hashes and original official URLs are retained in generation_record.json.

The coordinator owns the separate public JaxMARL adapter, sealed runtime validation, actual state replay, RGB/animated presentation, playback controls and community review. This authored kernel and declarative scene do not establish those acceptance outcomes. Structural/AST screening is separate from numerical validation. Passing future interaction or interface checks would not by itself establish strategic richness, reward learning, meta-learning, non-isomorphism or transfer. No new scientific performance gate is added.
