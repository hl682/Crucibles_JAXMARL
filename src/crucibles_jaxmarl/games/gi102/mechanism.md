Signal Veil Relay — game-v3-00102

Authored initial window 0, attempt 00. A=4, O=181, U=11, H=96. All four agents are active throughout an episode; a stunned role can only wait. Team membership is (0,1) versus (2,3); even-numbered roles are scouts. The world is an 11 by 7 bounded grid with two central passages at y=2 and y=4, physical beacons at (1,3) and (9,3), and distinct moving bodies.

The task is to bring the enemy beacon onto the team's home cell while its own beacon is home and uncarried. The first capture gives each winning teammate +1 and each opponent -1; all other transitions reward zero. Capture is natural termination, with no threshold external to the game's rules. The actual terminal carrier/beacon positions are retained.

Actions and simultaneous resolution

Actions 0–4 are wait and north/east/south/west. Action 5 handles adjacent beacons: it returns a grounded own beacon home and picks up a grounded enemy beacon. It can perform both if both satisfy the rules. Action 6 fires at the nearest hostile body or decoy on the same row/column within two tiles, with the intermediate tile checked for a wall. Firing consumes a three-tick cooldown even on a miss. A hit imposes two future wait ticks and drops carried cargo at the actual hit position. All shooters use one common post-movement, post-deployment snapshot, so reciprocal hits are possible. Equal-distance decoys beat bodies; otherwise fixed entity order breaks ties.

Action 7 transmits for three ticks: it shares the sender's current radar with its teammate and reveals the sender's position to every role. Action 8 spends the role's only decoy stock and places a four-tick physical reflector at its current cell. The reflector does not block feet. It absorbs an automatic nearest-target beam and disappears on a hit. Its radius-one interference hides bodies and beacons from hostile observers farther than one tile away. The decoy is not claimed to be an indistinguishable fake identity: the role scene labels the visible reflector truthfully; the deception mechanism is diverted tag targeting and obscured nearby cargo/body locations.

Action 9 offers a carried enemy beacon; the adjacent teammate must simultaneously choose 10 to receive. Passing cannot occur if either role is stunned after this tick's tags. Every transition sanitizes actions, resolves collision-constrained movement, decays/creates signals and reflectors, fires all beams, drops hit cargo, resolves adjacent pickups/returns in ascending role id, resolves bilateral handoffs, then tests capture. Interaction attempts are legal even when no object is in reach, so masks do not reveal concealed objects or bodies. Movement masks check walls and own stun only; collisions resolve in the transition.

Information and state binding

All 181 observation values are integer features divided by 128, with bounds [-1,1]. Visible object blocks contain their visibility bit and actual values; hidden blocks are zero. Empty/ground carrier sentinels are -1 before scaling. Scouts sense Manhattan radius four, couriers radius two, without optical wall occlusion. Radar observations are first masked by hostile interference, then shared only from actively transmitting teammates; a publicly transmitting body remains visible even in interference. Every event id is intentionally announced publicly and included in the last 20 columns, so renderer event labels add no privileged policy information. These event announcements reveal identities and event types, not unobserved positions. The 7x11 map, both home coordinates and capture counts are public. No hidden future randomness is exposed.

Agents, beacon positions, home coordinates and reflector positions are actual state arrays. Carried beacon coordinates follow their carrier; active reflector masks equal positive remaining lifetime. Agent-carry attributes are derived from actual carrier ownership. A hidden reflector is not drawn; a hit or expired reflector becomes inactive. Remaining signal/decoy/stun values are levels, whereas the separately declared event flags describe the actual latest transition. In an absorbing call those flags are preserved as part of the terminal snapshot, not new events.

Causal strategies and tradeoffs

A scout can move ahead and transmit an enemy's live position to a courier that cannot sense it, at the cost of exposing its own position. A courier can retain a decoy for the return trip, while an escort occupies a chokepoint, receives a beacon when the carrier is threatened, or fires a tag to recover its own beacon before a return can score. Alternatively both teammates can scout different passages, coordinate a late handoff and give up continuous signalling to reduce public exposure. A defender can suppress a route with tags or walk close enough to overcome interference. Success of a pass depends on the recipient's action and position; success of a return depends on the partner keeping or restoring the home beacon. These are rule-based interaction hypotheses, not measured strategic richness.

The playthrough proposes a complete capture via an explicit final handoff against waiting defenders, plus shorter signal and decoy demonstrations. Paired branches isolate teammate receive versus wait, signal versus wait, and decoy deployment versus no deployment against an identical incoming beam. No branch was run. Different pass ranges, team-role assignments, decoy budgets, radar radii or passage layouts could change escort and interception needs, but none of those variants is implemented or claimed to transfer here.

Declared invariants include distinct active agent positions and nonnegative clocks, stocks, timers and capture counts. Source-level bounds are: positions stay on walkable interior tiles; each beacon carrier is -1 or a role id; a team can carry only its enemy beacon; stock is 0 or 1; stun is 0..2; tag/signal timers 0..3; decoy lifetime 0..4; score is 0 or 1; t is 0..96. These are declarations for independent checking, not observed passes.

Interface and evidence boundary

The only candidate imports are jax and jax.numpy. All state leaves are fixed-shape int32 or bool arrays; observations and rewards are float32. All masks describe permitted attempts: a legal action may fail because of simultaneous contention or an unmet interaction condition. Out-of-range actions and masked actions deterministically become action 0. Destination contention cancels every conflicting move; no agent may enter another agent's start-of-step cell, including swaps or following a vacated cell in the same tick. No role is removed before joint termination. At terminal states only wait is legal. A nonterminal raw call advances t once; natural termination takes priority over the horizon; terminated and truncated are mutually exclusive. On an already terminal input every state leaf and observation remains unchanged, the clock is frozen, reward is zero and all dones remain true. There is no raw auto-reset.

The fixed initial layout and rules are deterministic. reset(key) and raw_step(key, ...) accept the trusted key but consume no randomness; no PRNG state is duplicated or replaced. The design did not prescribe narrow signed integer widths, so int32 is an explicit boundary representation choice. Power-of-two observation scaling is intended to keep represented values exact, but no eager/JIT/vmap parity has been executed or accepted.

The manifest binds every state leaf and every observation column. Entity coordinates affect the actual rules, and scene.json only labels those bound objects. The public tile map is explicitly declared public. Full-state debug rendering must remain labelled separately from each role view. No JaxMARL adapter, animation, replay, training, benchmark, scientific acceptance or transfer claim is supplied by this author. The coordinator must supply its separate agent-keyed adapter and independently test legal masks, inactive-role and __all__ behavior without modifying this frozen kernel.

Community reference attribution

The supplied official JaxMARL README informed the distinction between parallel agent-keyed dictionaries and this flat trusted kernel: https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md . The supplied official Pgx README informed explicit external-key, legality and termination API considerations, not board-game mechanics or heldout layouts: https://github.com/sotetsuk/pgx/blob/main/README.md . The supplied official Craftax README provided broad inspiration for persistent spatial objects and game state, and for separating raw transitions from outer auto-reset: https://github.com/MichaelTMatthews/Craftax/blob/main/README.md . The supplied official PureJaxRL README describes end-to-end JAX training and single-file algorithm implementations, not a new game library: https://github.com/luchris429/purejaxrl/blob/main/README.md . No upstream code was copied. The supplied Craftax_Baselines_ppo_rnn.py was not read for code or used; its bytes are hashed only to verify the supplied reference manifest. No external network request, dependency installation, candidate import, JAX/NumPy numerical execution or model/API call was made by the source author.

All action scripts and paired interventions in playthrough.json are UNEXECUTED_AUTHOR_HYPOTHESIS. Only source-byte, stdlib AST/JSON and supplied data-schema checks were performed. Static screening does not establish executability, exact numerical parity, legal trajectories, solvability, strategic richness, learning or generalization. Failed, negative, missing or inconclusive results from later independent evaluation must be retained.

Literal source evidence

- Local interference and relay: candidate.py:25: `return local | jnp.any(relay[:, :, None] & local[None, :, :], axis=1)`
- Physical nearest-target competition: candidate.py:124: `costs = jnp.where(eligible, 2 * dist + (jnp.arange(8)[None, :] >= 4), 30)`
- Tag drops actual cargo: candidate.py:133: `drop = (n["beacon_carrier"] >= 0) & hit_agent[jnp.clip(n["beacon_carrier"], 0, 3)]`
- Bilateral handoff: candidate.py:158: `handoff = ((a[i] == 9) & (a[peer] == 10) & (n["beacon_carrier"][enemy] == i)`
- Home-beacon capture dependency: candidate.py:172: `capture = capture.at[team].set((carrier >= 0) & at_base & home[team])`
- Absorbing raw state: candidate.py:211: `n = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), proposed, state)`

Method line index

- candidate.py:12-13: distances.
- candidate.py:15-25: sight.
- candidate.py:27-39: refresh.
- candidate.py:41-75: reset.
- candidate.py:77-89: legal_actions.
- candidate.py:91-97: move.
- candidate.py:99-181: advance.
- candidate.py:183-203: observe.
- candidate.py:205-215: raw_step.
