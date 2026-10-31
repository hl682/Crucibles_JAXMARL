# Common Kiln — private orders in a coupled workshop

Status: AUTHORED_NOT_EXECUTED. This initial train-only source was written for game-v3-00125, ordinal 125, attempt 00. No candidate import, compilation, transition, rollout, JIT, vmap, numerical parity check, rendering, training or test-set use occurred. Source SHA256: 892828a25bc9e7eee939ebe10ebde9c3d16ebcfc21767839a8ad32de2b3d7847.

## Boundary and lifecycle

A=3, O=351, U=11, H=144. All state leaves have a fixed declared key, shape and int32/bool dtype; observations and rewards are float32. Reset consumes only its supplied key; no external state, time, cache, file or network is part of the kernel. The scalar t starts at zero. A nonterminal call advances it once. Natural termination takes precedence over the horizon. An already-terminal call preserves every leaf and observation, preserves its terminal cause and clock, and returns zero rewards and all dones true. There is no raw auto-reset. The final tree operation preserves each reset leaf's dtype explicitly.

Every in-range action is a legal command attempt while the episode is live. A blocked movement, absent resource, wrong recipe, unavailable object or cooldown causes the specified deterministic wait/no-effect outcome. Out-of-range integers map to action 0. At joint termination only action 0 is legal. No role is removed independently; completed contributors can still help until joint completion. The role adapter must preserve these masks and semantics.

- reset: candidate.py:26
- observe: candidate.py:59
- legal_actions: candidate.py:56
- raw_step: candidate.py:84

## Actual mechanism and timing

The 13 by 11 workshop has a partial central wall, a kiln at (3,3), cooling at (9,3), a shared press at (6,7), dispatch at (6,9), six individually tracked raw pieces and one pair of tongs. Every worker can access both alternative corridors, but physical travel and one-piece carrying constrain production sequences. Reset permutes the three order kinds, using only the supplied key, so exactly one worker needs annealed material, one tempered material and one two-unit composite. Assignment is private; the scene does not expose it. Reset order assignment is at candidate.py:36; public/private observation separation is at candidate.py:59.

Movements resolve against actual floor, old occupied positions and contested targets. Then thermal timers progress, completed machine operations eject their output/tool, and resource commands are processed in rotating order i=(t+offset) modulo three. That order is a deterministic simultaneous-conflict rule: a scarce item, station or last fuel unit may be awarded to an earlier worker; all agents know the clock. It is not hidden asynchronous scheduling. See candidate.py:94, candidate.py:107, candidate.py:118, candidate.py:126 and candidate.py:140.

Ownership codes are -3 delivered, -2 consumed, -1 floor, 0..2 carrier, 3 kiln and 4 press. A worker carries at most one workpiece; tongs have their own unique owner. Pick/drop targets the lowest-index eligible floor piece within distance one. An offer transfers the carried piece to a lowest-index adjacent empty-handed waiting worker; if the sender has no piece, the same command can transfer its tongs to a waiting neighbour. Waiting as the receiver prevents same-tick transfer cycles. Moving, handing off and station occupancy all update actual object coordinates. See candidate.py:145, candidate.py:148, candidate.py:154, candidate.py:161 and candidate.py:14.

The kiln starts at heat three with six fuel. Every fourth live tick it loses one heat. A reachable stoke consumes one fuel and adds three heat, clipped to eight; reachable venting subtracts two. Coolant starts at four and regains one every eighth tick up to four. These thermal changes affect everyone, including a different worker's item already being processed. A command-phase stoke/vent affects that stored item at the next timer phase, making the ordering explicit. The fuel user pays 0.02 while partners can benefit, providing a mixed-incentive cost allocation problem. See candidate.py:109, candidate.py:170, candidate.py:137, candidate.py:173 and candidate.py:136.

Loading a raw piece requires temperature >=3 and an empty kiln. Heat above four also requires tongs; lower heat permits a slower manual route. Loading with tongs reserves the tool inside the kiln for two ticks; manual loading takes three. A stored item's heat and peak track the maximum temperature encountered, rather than cooling below an already absorbed peak while in the kiln. Completion ejects the still-hot piece to the actual station coordinate and releases reserved tongs there. See candidate.py:115, candidate.py:179, candidate.py:185 and candidate.py:121.

Outside the kiln, hot pieces lose one heat on global ticks divisible by three; a piece carried by the current tongs holder instead loses heat on ticks divisible by six. Reaching zero makes it annealed. This cadence is part of the source rules and was chosen in the initial author draft, without numeric trial or performance feedback. At the cooling station a hot carried piece may consume one coolant if its two-tick station cooldown is clear. A peak >=6 and current heat >=3 produce tempered output; otherwise quenching produces annealed output. Thus tongs are both a handling bottleneck and a temporal resource for transport, while passive air cooling is an alternative annealing route. See candidate.py:101, candidate.py:102, candidate.py:190 and candidate.py:191.

Composite assembly requires one carried annealed one-unit piece and another annealed one-unit piece on the floor within distance one of the press. The primary slot receives the sum of both masses; the secondary becomes consumed with mass zero. The press holds the primary for two ticks with tongs or three manually and then sets its stage to composite. Tongs used in pressing are unavailable elsewhere until completion. No mass is created, and delivered mass remains in its inactive slot, so the sum over all six mass slots stays six. This conservation statement is a source-level invariant to check later, not an executed test. The atomic merge is at candidate.py:197, candidate.py:200, candidate.py:201 and candidate.py:203.

At dispatch, a worker can deliver only stage order_kind+2: annealed=2, tempered=3, composite=4. A completed worker retains movement and assistance opportunities but cannot earn another personal delivery. Each new completion gives its own worker one, and gives every worker 0.2; a worker may therefore want a partner to succeed while still preferring scarce tools or coolant for its private order. All orders completed ends naturally. Exhausted fuel or an awkward material allocation does not produce a fabricated impossibility certificate; the episode can reach the horizon. See candidate.py:212, candidate.py:218 and candidate.py:227.

Possible plans include a worker specializing in heating while another performs hauling/cooling, exchanging the output suited to another private order, shared tongs transport, or a slower manual process that leaves tongs available. Stoking and venting can help or harm a partner's needed schedule; spending all coolant on rapid annealing can delay a tempering order. Press assembly explicitly needs a second material source and coordinated placement. These are hypotheses about incentives and coordination, not observed equilibria, learned adaptation or measured strategic richness.

Scene attributes expose actual item ownership/stage/heat/peak/mass, machine occupation/timers, tool owner and worker completion. Current kiln heat/fuel/coolant are level channels, not falsely repeated event flags. Private orders are omitted from scene bindings. Proposed rule changes in cooling cadence, shared rewards, wall geometry or tool handling could change coordination; none is an already validated transfer experiment.

## Observation contract

- Columns 0 through 5 (public): All worker xy /12
- Columns 6 through 26 (public): Each worker carried item one-hot: empty then six item slots
- Columns 27 through 29 (public): Each worker completed-order flag
- Columns 30 through 31 (public): Shared tongs xy /12
- Columns 32 through 37 (public): Tongs owner one-hot: floor, worker 0,1,2, kiln, press
- Columns 38 through 45 (public): Kiln, cooling, press and dispatch station xy /12
- Columns 46 through 49 (public): Station operation/cooldown timers /3
- Columns 50 through 77 (public): Each station item one-hot: empty then six item slots
- Columns 78 through 78 (public): Kiln temperature /8
- Columns 79 through 79 (public): Shared remaining fuel /6
- Columns 80 through 80 (public): Shared coolant /4
- Columns 81 through 92 (public): All six item-slot xy /12; inactive slots retained for provenance
- Columns 93 through 140 (public): Each item owner one-hot: delivered, consumed, floor, worker 0,1,2, kiln, press
- Columns 141 through 170 (public): Each item stage one-hot: raw, hot, annealed, tempered, composite
- Columns 171 through 176 (public): Item heat /8
- Columns 177 through 182 (public): Item peak heat /8
- Columns 183 through 188 (public): Material units per item slot /2, including delivered units
- Columns 189 through 194 (public): Item active flags; consumed and delivered slots inactive
- Columns 195 through 197 (self): Own private order one-hot: annealed, tempered, composite
- Columns 198 through 200 (self): Own worker identity one-hot
- Columns 201 through 201 (public): Clock /144
- Columns 202 through 344 (public): Public 11 by 13 static tile map in yx order
- Columns 345 through 347 (public): Latest fuel-or-vent, quench and assembly-start flags
- Columns 348 through 350 (public): Latest delivery flags per worker

The returned observation is the policy information boundary. Entity visibility and declared attributes refer to the same actual state and access rules. The complete static tile layout is public; a debug/full-state display must be labelled separately from a role view. Undeclared internal leaves are not an invitation to expose private values in a role renderer.

## Fixed author scripts and paired interventions

playthrough.json contains fixed joint integer commands, an explicit seed and UNEXECUTED_AUTHOR_HYPOTHESIS status for each script. Stop at the first done. These scripts were constructed from the written rules without local execution, simulation, numerical action search or feedback from other candidates. Prefix/branch comparisons state predicted causal differences only; there is no observed trace, successful episode, solvability certificate, ranking or learned policy.

## Community provenance and delivery boundary

Only this batch's hash-verified design/request, 20 listed inputs and copied public references were used. JaxMARL's README supplies multi-agent integration context, including agent-keyed public APIs; Pgx supplies JAX-native game-state and legal-action-mask design context. Craftax supplies an example of spatial compositional game mechanics. PureJaxRL and the Craftax PPO RNN baseline are algorithm/integration references, not a game library, copied environment, or replacement learner. No upstream source lines, layouts or benchmark rules were copied into this original kernel. Exact copied-input hashes and original official URLs are retained in generation_record.json.

The coordinator owns the separate public JaxMARL adapter, sealed runtime validation, actual state replay, RGB/animated presentation, playback controls and community review. This authored kernel and declarative scene do not establish those acceptance outcomes. Structural/AST screening is separate from numerical validation. Passing future interaction or interface checks would not by itself establish strategic richness, reward learning, meta-learning, non-isomorphism or transfer. No new scientific performance gate is added.
