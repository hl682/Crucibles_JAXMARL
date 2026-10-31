Switchback Repair Crew — game-v3-00103

Initial window 0, attempt 00. A=3, O=110, U=10, H=96. Roles 0, 1 and 2 are a structural engineer, electrician and courier. They share the task reward and differ in physical repair capabilities. There is exactly one wrench and one power cell; any worker can transport either, and each worker has capacity one. All objects and the evolving public map have literal state bindings.

The 9 by 7 map has a river-bank divider at x=4, a permanently open upper switchback at (4,1), and a broken central crossing at (4,3). The bridge console is at (3,3), the pump at (6,2), and the transmitter at (6,4). Before repair, the central crossing tile is 2 and is not walkable; the bridge completion changes that actual tile to 0. Movement in the completion tick used the old map, so crossing becomes available on the next decision.

Actions and work rules

Actions 0–4 are wait and cardinal movement. Action 5 drops the worker's carried tool at its position; with empty hands it picks the nearest uncarried tool within Manhattan distance one. Equal-distance tools use the lower tool index. These toggles resolve in worker id order, making same-step ground-object contests deterministic. A later worker can pick up a tool dropped earlier in that same resolution. Actions 8 and 9 implement explicit lending: an offering carrier transfers to an adjacent, empty-handed worker who chooses receive. Tool id order resolves two offers; the lowest eligible role id receives, and capacity is rechecked after each transfer. Tool positions follow their actual carrier after movement and transfers.

Action 6 works and 7 assists. Each addresses the nearest unfinished device within distance one; equal distances choose bridge before pump before transmitter. Actions have no effect if no valid nearby target exists. The engineer carrying the wrench adds one bridge progress per work tick, or two when another worker simultaneously assists the bridge. Bridge progress persists and completion opens the crossing.

The electrician carrying the power cell primes the pump only while a different worker simultaneously assists that same device. Two consecutive effective ticks are required. While incomplete, a missed joint work tick loses one pump progress, so someone arriving late or leaving early changes the outcome. Once complete the pump remains operational. The transmitter requires the already repaired pump plus engineer-with-wrench and electrician-with-cell both choosing work beside it on the same tick. It needs three joint work ticks; these transmitter ticks need not be consecutive.

Every device newly completed gives every worker +1 once. The reward is the count of newly completed devices, bounded by [0,3]. All three completions naturally end play; otherwise t=96 truncates. There is no reward for passing, movement, work attempts or repeated use of an already repaired device.

Causal interactions and plausible approaches

A tool loan changes which worker can satisfy a later work precondition and removes that opportunity from its former holder. A bridge repair changes every worker's available movement route. A courier can carry a needed tool to a waiting specialist, assist a pump while the engineer moves the wrench toward the transmitter, or accelerate the bridge instead. The pump's decaying partial progress makes partner timing consequential rather than a cosmetic shared score. The final transmitter work physically requires both scarce tools with their matching roles in the same neighbourhood.

One possible approach is bridge-first coordination, then send the electrician and engineer through different lanes, stabilize the pump together, and converge on the transmitter. Another is to use the permanently open upper route early, let courier and electrician prepare the pump, and delay the bridge/tool delivery until the engineer can exploit the central crossing. Body blocking and one-tool capacity make a worker's route and willingness to receive relevant to the others. These are plausible strategies from rules, without a claim that either is optimal, learned or empirically richer.

The main action hypothesis attempts a full three-device repair against no stochasticity. Separate scripts exercise an explicit courier-to-engineer loan. Paired branches compare assistant versus wait for bridge speed, sustained versus interrupted pump assistance, and receiving versus refusing an offered tool. They have not been executed, and no successful trajectory or terminal receipt exists.

Observations, rendering and invariants

All 110 values are exact integer features divided by 128. Observations contain own identity, clock/terminal causes, every worker's position/role/held tool, every tool position/type/carrier, every device position/progress/required progress/done flag, the live 7x9 tile map, and newly completed/handoff flags. Every entity is active and visible to every role; no private attribute is added by the scene. The repaired state is a persistent attribute, and event_repair only marks the transition where it first changes. Work progress is a level, not an event counted again each frame.

The declared invariants include unique active worker cells and nonnegative clock/work requirements/progress. Further source-level bounds for independent checks are one holder per tool, at most one tool per worker, carriers in -1..2, bridge/pump/transmitter progress in 0..3/0..2/0..3, permanent device completion, positions on passable tiles, and t in 0..96. Tools can legitimately share a tile with their carrier or another grounded object, so object-position uniqueness is not asserted. Variation in which role can stabilize which device, work decay, tool placement or switchback geometry could change the useful division of labour; no such variant or adaptation result is claimed.

Interface and evidence boundary

The only candidate imports are jax and jax.numpy. All state leaves are fixed-shape int32 or bool arrays; observations and rewards are float32. All masks describe permitted attempts: a legal action may fail because of simultaneous contention or an unmet interaction condition. Out-of-range actions and masked actions deterministically become action 0. Destination contention cancels every conflicting move; no agent may enter another agent's start-of-step cell, including swaps or following a vacated cell in the same tick. No role is removed before joint termination. At terminal states only wait is legal. A nonterminal raw call advances t once; natural termination takes priority over the horizon; terminated and truncated are mutually exclusive. On an already terminal input every state leaf and observation remains unchanged, the clock is frozen, reward is zero and all dones remain true. There is no raw auto-reset.

The fixed initial layout and rules are deterministic. reset(key) and raw_step(key, ...) accept the trusted key but consume no randomness; no PRNG state is duplicated or replaced. The design did not prescribe narrow signed integer widths, so int32 is an explicit boundary representation choice. Power-of-two observation scaling is intended to keep represented values exact, but no eager/JIT/vmap parity has been executed or accepted.

The manifest binds every state leaf and every observation column. Entity coordinates affect the actual rules, and scene.json only labels those bound objects. The public tile map is explicitly declared public. Full-state debug rendering must remain labelled separately from each role view. No JaxMARL adapter, animation, replay, training, benchmark, scientific acceptance or transfer claim is supplied by this author. The coordinator must supply its separate agent-keyed adapter and independently test legal masks, inactive-role and __all__ behavior without modifying this frozen kernel.

Community reference attribution

The supplied official JaxMARL README informed the distinction between parallel agent-keyed dictionaries and this flat trusted kernel: https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md . The supplied official Pgx README informed explicit external-key, legality and termination API considerations, not board-game mechanics or heldout layouts: https://github.com/sotetsuk/pgx/blob/main/README.md . The supplied official Craftax README provided broad inspiration for persistent spatial objects and game state, and for separating raw transitions from outer auto-reset: https://github.com/MichaelTMatthews/Craftax/blob/main/README.md . The supplied official PureJaxRL README describes end-to-end JAX training and single-file algorithm implementations, not a new game library: https://github.com/luchris429/purejaxrl/blob/main/README.md . No upstream code was copied. The supplied Craftax_Baselines_ppo_rnn.py was not read for code or used; its bytes are hashed only to verify the supplied reference manifest. No external network request, dependency installation, candidate import, JAX/NumPy numerical execution or model/API call was made by the source author.

All action scripts and paired interventions in playthrough.json are UNEXECUTED_AUTHOR_HYPOTHESIS. Only source-byte, stdlib AST/JSON and supplied data-schema checks were performed. Static screening does not establish executability, exact numerical parity, legal trajectories, solvability, strategic richness, learning or generalization. Failed, negative, missing or inconclusive results from later independent evaluation must be retained.

Literal source evidence

- Crossing map is changed by actual repair: candidate.py:23: `n["map"] = s["map"].at[3, 4].set(jnp.where(n["device_done"][0], 0, 2))`
- Explicit bilateral physical lending: candidate.py:98: `transfer = (owner >= 0) & (a[owner_index] == 8) & jnp.any(ready)`
- Assistant changes bridge work: candidate.py:113: `bridge_assist = jnp.any(assists & (target == 0))`
- Pump pair requirement: candidate.py:116: `pump_work = (jnp.any(works & electrician & (target == 1))`
- Delayed transmitter dependency: candidate.py:121: `radio_work = (s["device_done"][1] & jnp.any(works & engineer & (target == 2))`
- Actual completion reward: candidate.py:130: `reward = jnp.full((3,), jnp.sum(n["event_repair"].astype(jnp.int32)), jnp.float32)`
- Absorbing terminal state: candidate.py:152: `n = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), proposed, state)`

Method line index

- candidate.py:12-13: distances.
- candidate.py:15-24: refresh.
- candidate.py:26-51: reset.
- candidate.py:53-61: legal_actions.
- candidate.py:63-69: move.
- candidate.py:71-131: advance.
- candidate.py:133-144: observe.
- candidate.py:146-156: raw_step.
