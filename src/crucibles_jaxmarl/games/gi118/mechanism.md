# Sailboat race with shared wind shadows

AUTHORED_NOT_EXECUTED. This document describes source rules and author hypotheses; it is not a validation, animation, successful play, learning or transfer receipt.

## Task and state transition order

The two vertical island segments at x=5, y=2..3 and y=5..6 leave north, middle and south alternatives. Starting boats 0 and 1 are adjacent on the east-wind line; boat 2 approaches from the lower lane. Buoy 0 at (7,2) is shared by boats 0 and 2; boat 1 uses buoy 1 at (7,6). Docks are (9,1), (9,7), (1,4). Raised sails cast shadows even while their skipper holds or handles a tow. A low-sail neighbor can therefore remove a real movement constraint or consent to being pulled. Shadow calculation uses all unmoored high sails and the literal relative coordinates. Low sails ignore the shadow but consume stamina when a row actually moves.

After sail changes and deterministic tow arbitration, leaders perform one simultaneous movement. Each moved helper pulls its recipient into its old position. Unattached high-sail boats with an eligible downwind boost then attempt a second cell. Rowing and towing consume one stamina only when their corresponding move occurs. Goal checks use the final position, so a two-cell move that passes over a buoy without ending on it does not count. Reward is -0.01 for each skipper that had not already arrived, +0.25 for its first buoy, +1 for arrival, plus 0.5 divided among the first simultaneous arrivals. There is no reward for displaying a cooperation flag. End-of-step weather and derived visibility are then updated.

## Actions and inactive-role semantics

- 0: Heave to; recover one stamina if not arrived.
- 1: Steer east.
- 2: Steer west.
- 3: Steer south.
- 4: Steer north.
- 5: Raise sail; leave any incoming tow.
- 6: Lower sail; permit an adjacent high-sail boat to tow.
- 7: Attach tow to lowest-index eligible adjacent low-sail boat.
- 8: Release own outgoing and incoming tow.

All nine intent actions are legal for unfinished boats; blocked, inapplicable or malformed actions deterministically have no effect or map to action 0. Directional actions on a currently towed recipient are suppressed while the link persists. Arrived boats and an absorbing terminal state allow only action 0. Sail/release changes precede shadow and tow allocation. New links use cyclic priority beginning at t mod 3 and lowest eligible recipient id. Normal movement cannot enter any other boat previous cell; duplicate proposed destinations cancel. Tow followers alone enter their helper vacated cell. A second downwind movement uses a new simultaneous occupancy check.

## Information and rendering

The terrain, buoy/dock locations, wind and clock are public. Another boat is visible exactly within Manhattan distance 3; its displayed attributes are present in the masked peer record. Own goal, sail, stamina and route state are always available. Tow entities are visible only when both endpoints are visible. Invisible data are zero with an explicit visibility bit; no privileged full state is a policy input.

Per-step event arrays carry an explicit boat_visible binding. A role view may disclose only visible participants; full event arrays belong to a labelled debug view. No global arrival or tow event panel may reveal invisible boats.

Boat coordinates and island tiles govern transitions. Sail, stamina, shadow, route stage and arrival attributes bind real leaves. Each tow marker lies at the midpoint of its two actual endpoint positions and carries the actual displacement and endpoint ids; it is inactive without a real link. The public wind indicator is a fixed map annotation for the actual global wind, not an extra simulated vessel. Buoys and docks bind the positions used for task checks. Rendering must not replace a role view with a full-state view.

## Native observation columns

- [0, 2): Own x,y divided by 10 (self).
- [2, 3): Own stamina divided by 4 (self).
- [3, 4): Own sail raised flag (self).
- [4, 5): Own route progress divided by 2 (self).
- [5, 6): Own arrived flag (self).
- [6, 7): Own outgoing tow recipient id divided by 2; -0.5 means none (self).
- [7, 9): Current own required goal x,y divided by 10 (self).
- [9, 12): Own skipper one-hot id (self).
- [12, 39): Three ordered boat records, stride 9: visibility, x/10, y/10, sail, stamina/4, progress/2, arrived, outgoing recipient/2, current shadow. All eight data values zero when invisible. Includes self. (local).
- [39, 138): Public 9 by 11 terrain in yx row-major order; 0 water, 1 island (public).
- [138, 142): Two public buoy xy positions in buoy-id order divided by 10 (public).
- [142, 148): Three public assigned dock xy positions in skipper-id order divided by 10 (public).
- [148, 150): Public wind unit dx,dy (public).
- [150, 151): Public wind strength divided by 2 (public).
- [151, 152): Public clock divided by 80 (public).

## Invariants and conservation

Position updates are admitted only for water cells inside [0,10] by [0,8]. Conservative occupied-cell checks plus duplicate cancellation maintain distinct boat cells; a tow follower uses only its own helper vacated cell. Each high-sail helper has at most one recipient, recipients are allocated once, and low-sail recipients cannot themselves be high-sail helpers, excluding tow cycles. Stamina is clipped to [0,4], route progress advances only 0 to 1 to 2, and arrivals never revert. Buoy/dock arrays and course assignments are fixed. Rewards and normalized observations have the stated finite bounds. These are source-level invariants, not measured rollout results.

## Partner-dependent strategies and bounded variation

A skipper can keep a raised sail to stall a pursuer, lower it to let traffic through, row around a shadow, or take a longer island lane with cleaner wind. Towing trades the helper speed boost and stamina for recipient movement; a partner can repay later, decline by raising sail, or release a link. Waiting and first-arrival rewards make help costly in some states without preventing mutual benefit. Public wind shifts create timing choices and simultaneous occupancy creates passing conflicts. Prospective variations in wind persistence, island geometry, route assignments or tow cost would alter partner-dependent choices; no such variants were executed and no non-isomorphism or transfer claim is made.

## Terminal, RNG and numeric representation contract

A nonterminal call advances scalar int32 t once. Natural termination takes priority at H: truncated=(t>=H)&~terminated; done=terminated|truncated. An already-done input returns the same entire state and observation, zero float32[A] rewards, all dones true and the preserved causes. No raw reset occurs. All roles continue to receive the joint done flag; an arrived or stunned role is not independently terminated.
Reset fixes wind east and draws strength with jax.random.randint(reset_key,(1,),1,3,dtype=int32). Each nonterminal step splits its supplied key into two keys; wind direction index 0..3 and strength 1..2 are applied only when the new clock is divisible by six. Sailing uses the input-state wind; the newly sampled wind is the next-state wind. No hidden RNG, sampled route, seed remapping or key in state exists.
The author declares seed 118 with reset_key=jax.random.PRNGKey(seed) and raw_step_key=jax.random.fold_in(reset_key, step_index+1), step_index starting at zero. The same keys are required for both intervention branches, including suffix steps. This schedule has not been executed.
All signed integer state uses int32, boolean state uses bool, and the observation/reward boundary is float32. No smaller signed integer design field or stored reset key is required. Probability draws use only the supplied explicit key. Token counts, compute cost and elapsed model-compute cost are unknown.

## Literal source evidence

- Causal wind casting: candidate.py:16; literal source: `casting = sail[None, :] & ~arrived[None, :]`.
- Actual shadow movement constraint: candidate.py:122; literal source: `sailing_speed = jnp.where(shadow, 0, 1 + (downwind & (s['wind_strength'][0] == 2)).astype(jnp.int32))`.
- Tow recipients use helper previous position: candidate.py:127; literal source: `pos1 = jnp.where(pulled[:, None], s['pos'][helper], pos1)`.
- Towing resource cost: candidate.py:131; literal source: `towing_cost = helper_moved.astype(jnp.int32)`.
- Shared first-arrival incentive: candidate.py:140; literal source: `bonus = 0.5 * first.astype(jnp.float32) / jnp.maximum(jnp.sum(arrival.astype(jnp.float32)), 1.0)`.
- Role-local visibility: candidate.py:22; literal source: `s['boat_visible'] = distance <= 3`.
- Native absorption: candidate.py:173; literal source: `return jax.lax.cond(state['done'], lambda _: self.absorb(state),`.

## Method line map

- shadows: candidate.py:12-17.
- decorate: candidate.py:19-30.
- reset: candidate.py:32-64.
- observe: candidate.py:66-78.
- legal_actions: candidate.py:80-81.
- move_once: candidate.py:83-92.
- advance: candidate.py:94-166.
- absorb: candidate.py:168-170.
- raw_step: candidate.py:172-174.

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
