Two-Stride Relay Arena — game-v3-00105

Initial window 0, attempt 00. A=4, O=182, U=11, H=96. Roles 0 and 1 form team 0, roles 2 and 3 team 1. The court is a bounded 13 by 9 grid. Team 0 attacks x=11 at y=3..5; team 1 attacks x=1 at y=3..5. A goal marker is the center of that actual three-cell zone, and the entire zone is encoded in the map. Initial positions are (3,3), (5,5), (9,3), (7,5); role 0 starts with the token. This fixed kickoff advantage is explicit and has not been measured or balanced. Side-swapping is a possible future variant, not part of this reset.

The physical token must be caught from a teammate's in-flight pass to arm a goal. Merely picking up a grounded token, self-recovery, tackling or intercepting does not arm it. A team scores only while a player holds an armed token on that team's target zone. The first such goal ends the joint episode and gives each teammate +1, each opponent -1; all other rewards are zero. Flying a ball through the goal without a holder does not score.

Possession, flight and simultaneous actions

Actions 0–4 are wait and cardinal movement. A carrier can make only two successful moves during one possession; blocked attempts consume time but not travel allowance. A held token drops automatically after four held transitions without a release, with the former carrier locked out of reacquisition on the immediately following tick. Newly caught or stolen possession has age zero. Catching and throwing are distinct actions on distinct ticks; there is no catch-and-throw shortcut.

Actions 5–8 throw north/east/south/west. The ball starts at the post-movement carrier position and advances one actual cell during the throw tick, then one per later transition for at most six cells. A wall stops it on the preceding valid cell; exhaustion of flight leaves a loose token. The flight state is position, integer velocity, remaining steps and last thrower/team, all shown in the scene and policy input. A throw imposes pickup cooldown two on the thrower, and the last thrower cannot catch their own token while it is flying.

Every unlocked player automatically catches a loose or flying token at their exact cell. Action 10 adds one-tile Manhattan reaching distance for an allied or neutral token. Action 9 adds that reach for a hostile token. At simultaneous contact the closest player wins; equal distance prefers a hostile guarding player, then priority rotates by the previous clock modulo four. Reaching pulls the token to the actual catcher cell. An allied catch during flight arms the token, while other catches clear arming. Flight completion on the catch cell still counts as an in-flight catch.

After catch resolution, a guard beside an opposing held carrier can steal the token. New catches are protected from tackling for their acquisition tick, and a throw releases before any tackle attempt. Tackle ties use rotating role priority, possession follows the tackler position, and arming, age and travel reset. A carrier may therefore pass away from an adjacent defender, but that pass can still be intercepted on its actual trajectory. A stolen carrier has no automatic immunity on the next tick; contested counter-tackles are possible.

Order is action sanitation, simultaneous collision-constrained player movement, clock/cooldown update, release, one token flight step, contact/catch, held-token tackle, four-tick forced drop, goal and horizon checks. All direct references are to actual state; the scene never invents a parabola or unseen in-between contact.

Causal strategies

Two-stride travel and the four-tick clock prevent a lone uninterrupted carrier from simply running across the court. A receiver's support position and choice of receive determine whether an otherwise identical throw is caught early or continues into defender space. An opponent who commits to guard cannot move that tick but can beat an equally close receiver at the contact point. Screeners can block lanes through the same collision rules; they do not receive magical immunity or a separate screen reward.

A possible attacking approach is short give-and-go passes with a receiving teammate just beside the flight path. Another is to move a support runner around defenders while the carrier uses the allowed hold window, then throw a longer pass into open space. Defenders can press the held carrier to force a release or reserve a guard at an expected landing lane. The timing tradeoff is explicit: moving improves future position, whereas receiving or guarding increases this tick's contact reach. No strategy has been optimized, learned or compared.

The principal action hypothesis attempts several mutual relays and a team 0 goal while defenders wait. One intended move is legally attempted into a teammate's start cell and should be blocked, demonstrating the simultaneous occupancy rule rather than assuming the cell is vacated early. A paired receiving branch compares early catch versus continued flight; another changes only the defender guard at a later contested pass. A separate wait script targets the holding-limit drop. All are unexecuted hypotheses, including the expected goal.

Observation and presentation fidelity

The full 182-dimensional observation consists of own identity, clock/terminal causes, all player positions/teams/cooldowns, token position/carrier/velocity/flight/arming/age/travel/last thrower and team, target goal centers, public court tiles, team score and all event flags. Every integer is divided by 128; -1 sentinels remain distinguishable, and values stay within declared [-1,1] bounds. All entity visibility masks are true. No future path, opponent intent or planned landing position is added to observations or the scene. The real ball position can be airborne, loose or at a carrier, as indicated by its bound attributes.

Declared invariants include unique active player cells and nonnegative time, cooldown, flight, age, travel and score. Further bounds for independent checking are carrier in -1..3, last thrower in -1..3, last team in -1..1, velocity cardinal or zero, flight remaining 0..5 after a processed transition (six is only an intermediate release value), returned held age 0..3, travel 0..2, score 0..1 and t 0..96. Interceptions reset arming; a carried ball shares its actual carrier position; a grounded or airborne ball may share a cell with a pickup-locked player without violating unique-player occupancy. Changing throw range, receiving radius, hold budget, target zones or team assignments could alter coordination requirements, but no measured transfer or fairness claim is made.

Interface and evidence boundary

The only candidate imports are jax and jax.numpy. All state leaves are fixed-shape int32 or bool arrays; observations and rewards are float32. All masks describe permitted attempts: a legal action may fail because of simultaneous contention or an unmet interaction condition. Out-of-range actions and masked actions deterministically become action 0. Destination contention cancels every conflicting move; no agent may enter another agent's start-of-step cell, including swaps or following a vacated cell in the same tick. No role is removed before joint termination. At terminal states only wait is legal. A nonterminal raw call advances t once; natural termination takes priority over the horizon; terminated and truncated are mutually exclusive. On an already terminal input every state leaf and observation remains unchanged, the clock is frozen, reward is zero and all dones remain true. There is no raw auto-reset.

The fixed initial layout and rules are deterministic. reset(key) and raw_step(key, ...) accept the trusted key but consume no randomness; no PRNG state is duplicated or replaced. The design did not prescribe narrow signed integer widths, so int32 is an explicit boundary representation choice. Power-of-two observation scaling is intended to keep represented values exact, but no eager/JIT/vmap parity has been executed or accepted.

The manifest binds every state leaf and every observation column. Entity coordinates affect the actual rules, and scene.json only labels those bound objects. The public tile map is explicitly declared public. Full-state debug rendering must remain labelled separately from each role view. No JaxMARL adapter, animation, replay, training, benchmark, scientific acceptance or transfer claim is supplied by this author. The coordinator must supply its separate agent-keyed adapter and independently test legal masks, inactive-role and __all__ behavior without modifying this frozen kernel.

Community reference attribution

The supplied official JaxMARL README informed the distinction between parallel agent-keyed dictionaries and this flat trusted kernel: https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md . The supplied official Pgx README informed explicit external-key, legality and termination API considerations, not board-game mechanics or heldout layouts: https://github.com/sotetsuk/pgx/blob/main/README.md . The supplied official Craftax README provided broad inspiration for persistent spatial objects and game state, and for separating raw transitions from outer auto-reset: https://github.com/MichaelTMatthews/Craftax/blob/main/README.md . The supplied official PureJaxRL README describes end-to-end JAX training and single-file algorithm implementations, not a new game library: https://github.com/luchris429/purejaxrl/blob/main/README.md . No upstream code was copied. The supplied Craftax_Baselines_ppo_rnn.py was not read for code or used; its bytes are hashed only to verify the supplied reference manifest. No external network request, dependency installation, candidate import, JAX/NumPy numerical execution or model/API call was made by the source author.

All action scripts and paired interventions in playthrough.json are UNEXECUTED_AUTHOR_HYPOTHESIS. Only source-byte, stdlib AST/JSON and supplied data-schema checks were performed. Static screening does not establish executability, exact numerical parity, legal trajectories, solvability, strategic richness, learning or generalization. Failed, negative, missing or inconclusive results from later independent evaluation must be retained.

Literal source evidence

- Movement budget binds possession: candidate.py:49: `& (~owns | (s["ball_travel"][0] < 2))[:, None])`
- One-cell physical flight: candidate.py:94: `pos = jnp.where(advances, next_pos, pos)`
- Defensive reach: candidate.py:101: `guard = (a == 9) & hostile & (distance <= 1)`
- Actual contact ordering: candidate.py:106: `costs = distance * 16 + jnp.where(guard, 0, 4) + rank`
- Team pass arms scoring: candidate.py:110: `allied_pass = allied_pass & (last_team == n["agent_team"][catcher])`
- Opponent tackle changes carrier: candidate.py:129: `owner = jnp.where(tackled, tackler, owner)`
- Holding clock forces release: candidate.py:135: `forced_drop = (owner >= 0) & (age >= 4)`
- Carried armed goal requirement: candidate.py:153: `goal = (owner >= 0) & relay & (pos[0] == goal_x) & (pos[1] >= 3) & (pos[1] <= 5)`
- Absorbing terminal state: candidate.py:183: `n = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), proposed, state)`

Method line index

- candidate.py:12-39: reset.
- candidate.py:41-52: legal_actions.
- candidate.py:54-60: move.
- candidate.py:62-161: advance.
- candidate.py:163-175: observe.
- candidate.py:177-187: raw_step.
