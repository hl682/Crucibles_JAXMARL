# Territorial patrol and mobile decoys

AUTHORED_NOT_EXECUTED. This document describes source rules and author hypotheses; it is not a validation, animation, successful play, learning or transfer receipt.

## Task and state transition order

The 13 by 9 map has a central wall at x=6 with passages at y=2 and y=6. West and east refuges are x<=2 and x>=10; opponents cannot enter them. Four initially ground objectives lie beside the passages. Teams may take any objective and bank it within distance one of their own home. Every patrol can carry one objective, and handoff to an adjacent empty teammate changes its physical carrier and subsequent position.

A guard is a one-tick stationary action, not an automatic role label. After motion and decoy updates, a hostile decoy within distance two takes precedence: the guard destroys the nearest such decoy, using decoy id to break ties, and cannot capture. An undisrupted guard captures all adjacent awake opponents outside refuges. Guards act simultaneously, so mutually adjacent guards can capture one another. A captured carrier drops its object at the encounter position, teleports to its fixed berth and receives stun=2. Its next two transitions accept only hold. Home berth co-location after a teleport is allowed; normal movement still blocks occupied targets.

Each decoy moves one cell along its heading, ignores physical bodies, and reflects by reversing its heading while holding position at a wall. Launching places it one step ahead if that cell is floor, otherwise on the launcher cell. One fixed slot per role is replaced on re-launch; launch consumes one charge, starts cooldown=3, and sets ttl=6. Holding awake within one cell of home restores one charge up to two. Separate from guard distraction, an active hostile decoy within distance two reduces the observer radius to one. A scout pulse takes the whole action and overrides this to radius five, broadcasting only its position/team to allies.

The west team reward is delivery_delta_west-delivery_delta_east+0.1*(east_captures-west_captures), with the opposite payoff for east. Both teammates receive their team value. Objects and captures, not visual signals, generate payoffs.

## Actions and inactive-role semantics

- 0: Hold; recharge one decoy charge near own home while awake.
- 1: Move east.
- 2: Move west.
- 3: Move south.
- 4: Move north.
- 5: Collect an adjacent ground objective when empty, or bank carried objective near own home.
- 6: Hand carried objective to adjacent empty teammate.
- 7: Guard this tick; capture adjacent awake enemies unless distracted by a hostile decoy.
- 8: Scout pulse this tick; observation radius five and position signal.
- 9: Deploy or replace own mobile decoy heading east.
- 10: Deploy or replace own mobile decoy heading west.
- 11: Deploy or replace own mobile decoy heading south.
- 12: Deploy or replace own mobile decoy heading north.
- 13: Drop carried objective at current cell.

All fourteen intents are legal for an awake role; inapplicable interactions are deterministic no-ops. A stunned role and a terminal state allow only 0. Out-of-range or masked actions map to 0. Movement resolves simultaneously against walls, previous occupied cells, duplicate targets and enemy refuge admission; swaps are blocked. Existing decoys move before new launches. Guards resolve simultaneously after movement and decoy updates: each distracted guard destroys its nearest hostile decoy and captures nobody; other guards tag every adjacent awake enemy outside its refuge. Mutual captures are allowed. Captured cargo drops before teleport. Pickup, banking, handoff and voluntary drop then resolve in cyclic role order starting at t mod 4. Capture returns to a fixed refuge berth, where temporary friendly co-location is permitted; no global unique-agent-position invariant is asserted.

## Information and rendering

Terrain, home locations, score totals and clock are public. Patrol/objective/decoy state is visible within Manhattan radius three, or one when a hostile active decoy is within distance two; a scout pulse raises it to five. Self is always visible. The owner additionally sees its active decoy remotely. Friendly scout signals broadcast only sender position/team globally; hostile signals are visible within distance five. A signal does not reveal the sender inventory or other hidden attributes. Every invisible record has an explicit visibility bit and zero data fields.

Per-step event arrays bind agent_visible. Role views must hide events attributed to invisible patrols, even though public scores reveal delivery totals. Full capture/deployment/handoff event identity belongs only to the labelled debug view when its participant is hidden. A scout signal must not be treated as full agent visibility.

Agent positions, stuns, guard/scout flags, charges and actual carried ids are source-bound. Decoys are visible known physical jammers, not falsely labelled anonymous agents; their effect is real range reduction and guard diversion. Decoy headings and TTL drive movement. Objective ground/carried/banked state governs actual positions and visibility. Signal markers have their own visibility masks and must not unmask patrol attributes. Captures show actual refuge return and cargo drop from recorded states; no interpolated invented battle outcome is authorized.

## Native observation columns

- [0, 2): Own xy position divided by 12 (self).
- [2, 3): Own team id (self).
- [3, 4): Own recovery ticks divided by 2 (self).
- [4, 5): Own carried objective id divided by 3; negative means none (self).
- [5, 6): Own decoy charges divided by 2 (self).
- [6, 7): Own launch cooldown divided by 3 (self).
- [7, 11): Own patrol one-hot id (self).
- [11, 47): Four ordered patrol records, stride 9: visible, x/12, y/12, team, stun/2, guard, scout, objective id/3, charges/2. Invisible data values are zero. (local).
- [47, 75): Four decoy records, stride 7: visible, x/12, y/12, heading dx, heading dy, owner/3, ttl/6. Invisible data are zero; an owner has active-decoy telemetry. (local).
- [75, 99): Four objective records, stride 6: visible, x/12, y/12, mode/2, carrier/3, bank team. Invisible data are zero; banked objectives are inactive. (local).
- [99, 115): Four scout-signal records, stride 4: visible active pulse, x/12, y/12, sender team. Friendly pulses are audible globally; enemy pulses only within distance five. No other sender fields are broadcast. (local).
- [115, 232): Public 9 by 13 wall map in yx row-major order (public).
- [232, 236): Two public home xy positions divided by 12, ordered by team (public).
- [236, 238): Public team banked-objective totals divided by 4 (public).
- [238, 239): Clock divided by 96 (public).

## Invariants and conservation

There are always four objective identities, each in exactly one mode: ground, carried or banked. Pickup, drop, capture, handoff and banking change mode/carrier without creating objects, and carriers are admitted only when empty. Banked objects never reactivate; sum(score) equals the number of mode-2 objectives and is at most four. Each role has at most one active decoy because its fixed slot is overwritten; charges stay 0..2, cooldown 0..3, TTL 0..6 and stun 0..2. Movement and launch checks keep all positions within map floor cells; teleport locations are fixed valid refuges. Team rewards sum to zero across the four agents. Per-role entity visibility is constructed from the same radii used by observe; signal and owner-telemetry exceptions are explicit. Source reasoning is not a runtime invariant receipt.

## Partner-dependent strategies and bounded variation

Teams may send a guarded carrier through one passage while a partner launches a moving screen, use a scout to see through jamming, or switch to the other route to avoid a blocking patrol. Guarding denies adjacent transport but sacrifices movement and is vulnerable to a decoy; scouting gains information but provides no capture that tick. A carrier can hand cargo to an adjacent partner before an interception, while defenders can wait for a decoy to expire or force charge expenditure. Public scores support timing decisions without exposing cargo locations. Prospective changes to passage layout, TTL, guard radius, charge regeneration or objective placement would alter coordination requirements; none have been evaluated.

## Terminal, RNG and numeric representation contract

A nonterminal call advances scalar int32 t once. Natural termination takes priority at H: truncated=(t>=H)&~terminated; done=terminated|truncated. An already-done input returns the same entire state and observation, zero float32[A] rewards, all dones true and the preserved causes. No raw reset occurs. All roles continue to receive the joint done flag; an arrived or stunned role is not independently terminated.
Reset and all transitions are deterministic and ignore the supplied keys. The exact declared seed/fold-in schedule remains replay provenance. No local model, hidden RNG or process state is consulted.
The author declares seed 120 with reset_key=jax.random.PRNGKey(seed) and raw_step_key=jax.random.fold_in(reset_key, step_index+1), step_index starting at zero. The same keys are required for both intervention branches, including suffix steps. This schedule has not been executed.
All signed integer state uses int32, boolean state uses bool, and the observation/reward boundary is float32. No smaller signed integer design field or stored reset key is required. Probability draws use only the supplied explicit key. Token counts, compute cost and elapsed model-compute cost are unknown.

## Literal source evidence

- Decoys change role information: candidate.py:20; literal source: `radius = jnp.where(s['scout'], 5, jnp.where(jammed, 1, 3))`.
- Decoy telemetry visibility: candidate.py:22; literal source: `s['decoy_visible'] = ((decoy_distance <= radius[:, None]) | jnp.eye(4, dtype=jnp.bool_)) & active_decoy[None, :]`.
- Scouts broadcast position without full agent visibility: candidate.py:32; literal source: `s['signal_visible'] = ((s['team'][:, None] == s['team'][None, :]) | (distance <= 5)) & s['scout'][None, :]`.
- Guard diversion is a transition: candidate.py:140; literal source: `distraction = guard & jnp.any(decoy_targets, axis=1)`.
- Simultaneous physical capture: candidate.py:147; literal source: `captured = jnp.any(tagged, axis=0) & awake & ~refuge`.
- Captured cargo drops before teleport: candidate.py:151; literal source: `dropped = carried & captured[old_carrier]`.
- Actual teammate handoff: candidate.py:183; literal source: `handoff = (a[i] == 6) & usable & has_object & jnp.any(neighbor)`.
- Competitive shared team payoff: candidate.py:193; literal source: `left_reward = (delta_score[0] - delta_score[1]).astype(jnp.float32) + 0.1 * (captured_right - captured_left)`.

## Method line map

- decorate: candidate.py:12-34.
- reset: candidate.py:36-73.
- observe: candidate.py:75-96.
- legal_actions: candidate.py:98-99.
- advance: candidate.py:101-204.
- absorb: candidate.py:206-208.
- raw_step: candidate.py:210-212.

## Reference attribution

- JaxMARL_README.md: Reviewed the copied parallel agent-keyed API description and reset/step call sites. The native author kernel remains a distinct interface requiring a trusted adapter. Supplied SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; origin https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md.
- Pgx_README.md: Reviewed the copied legal-action-mask and explicit-key board-game API discussion. No board rule, implementation, layout or evaluation was copied. Supplied SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; origin https://github.com/sotetsuk/pgx/blob/main/README.md.
- PureJaxRL_README.md: Reviewed the copied description of a JAX-native PPO training pipeline. This is algorithm integration background, not a game library and not a learner replacement. Supplied SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; origin https://github.com/luchris429/purejaxrl/blob/main/README.md.
- Craftax_Baselines_ppo_rnn.py: Read-only search of reset/step/scan call sites in the supplied recurrent PPO integration code. No upstream source was copied or executed. Supplied SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; origin https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py.
- Craftax_README.md: Reviewed copied JAX spatial game, gymnax-interface and optional auto-reset descriptions. Original mechanics are authored here; raw terminal handling follows the assigned contract. Supplied SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; origin https://github.com/MichaelTMatthews/Craftax/blob/main/README.md.

No upstream game code is copied. JaxMARL API ideas, Pgx board APIs, Craftax game/environment ideas and PureJaxRL/Craftax PPO algorithm integration are distinct sources of conceptual context. No literature claim is treated as evidence that this game works.

## Acceptance limits

Only the supplied stdlib AST whitelist and data-only manifest/scene validators are used. Static checks do not test runtime array shapes, eager/JIT/vmap parity, terminal behavior, conservation over rollouts, paired causal outcomes, solvability, rich strategy, trainability, animation, optimality, non-isomorphism or meta-learning. Source authoring alone is not community delivery. The coordinator must retain independent dispatch provenance and perform any later numeric work only under its separately authorized, hash-bound execution boundary.
All scripts and paired branches are UNEXECUTED_AUTHOR_HYPOTHESIS. Negative, failed and indeterminate later outcomes must remain visible; no performance threshold or scientific acceptance is introduced.
