# Lantern Flag Rescue — authored attempt_00

Status: AUTHORED_NOT_EXECUTED. A=4, O=131, U=8, H=72. The author used only the hash-verified batch inputs and official supplied reference snapshots. All strategies, scripts and interventions below are hypotheses from the rules.

## World and actions

The board has 11 columns and 7 rows, coordinates [x,y], y down. The central wall is x=5 at y=0,2,4,6; three actual openings remain at y=1,3,5. Team 0 uses agents 0,1 and home (1,3); team 1 uses agents 2,3 and home (9,3). Agents start at (1,2), (1,4), (9,2), (8,4). A flag is either on one actual ground cell or carried by one enemy agent; carried flag positions follow the carrier.

Actions are 0 Wait, 1 North, 2 East, 3 South, 4 West, 5 Tag, 6 Rescue and 7 Guard. A normal move advances one cell. Wall/out-of-bounds directions are masked. A direction blocked only by another player is a legal attempt with no displacement. All players stay present while captured.

## Ordered transition and conflicts

First sanitize actions, then resolve movement simultaneously. Entering any other player's pre-move cell is rejected even if that player plans to leave. All duplicate proposed destinations are rejected. This conservative rule prevents swaps, overlap and ambiguous movement chains.

Tags, protection and rescues are then computed together at the resolved positions. Manhattan distance at most one is the interaction radius. A free Guard player protects the other member of its team, never itself. A free Tag player tags every adjacent enemy who began the step free and is not protected. Simultaneous mutual tags are possible. An agent tagged on this step still contributes its already-selected tag or rescue, because these effects are simultaneous.

A fresh tag sets the target capture timer to four. Old positive timers decrement once; a Rescue action by the adjacent free ally instead clears an old prisoner's timer. A newly tagged agent cannot be rescued on the same transition, and an old prisoner released on this transition had only Wait available at its start. Release therefore restores movement for the next call. Timer expiry guarantees bounded disablement with no permanently disabled role.

Flags follow old carriers, then captured carriers drop them. Free owners touching a ground flag return it home. Free enemies on the resulting ground position pick it up; lowest agent index is the explicit pickup tie-break, although unique player positions make that tie unreachable from the declared reset. Pickup and return are automatic, spatial interactions. Finally, a carrier at its own base scores only if its own flag is home and uncarried. The two flag checks use the same completed pickup state.

## Why another player changes the decision

An escort spending Guard forgoes movement and tagging; without the adjacent escort, an enemy interception disables the carrier and drops the prize. A rescuer restores the ally's action set early but must travel to the real capture cell and spend a stationary action. A defender can tag a runner, block an opening, retrieve its own dropped flag, or chase an escort. These alternatives change which route and timing a teammate should choose.

Direct unopposed return, a guarded advance with pauses, a rescue detour, and counter-capture of the opposing flag are distinct prospective tactics. Guarding everywhere is not assumed optimal: it sacrifices progress, leaves the guard itself exposed, and requires adjacency. The three lanes allow interceptions and detours. No strategic richness, best response, learnability or non-isomorphism has been experimentally established.

## Rewards, endings and state invariants

Each team's newly scored return minus the other team's newly scored return is paid to both teammates, with 0.001 subtracted for every role on each nonterminal call. A natural finish happens on any return; the horizon is 72. Returned per-step rewards lie inside [-1.01,1.0]. If simultaneous scoring were reached, its differential is zero. There is no reward for repetitive tagging or rescue.

Player positions are unique walkable cells. Capture timers stay in 0..4; no flag is duplicated; a flag owner is -1 or an opposing player; each player can hold only the opposite team's single flag. Team score can only move from zero to one before absorbing termination. These are declared/source-derived invariants for later real tests, not executed evidence.

## Native information and scene

Every role receives the same public dynamic tables and public 7x11 terrain, plus its own one-hot identity. Observation fields in manifest.json enumerate every column. The scene binds actual agent positions, current flags/carriers, actual bases/scores, and a capture overlay at each currently immobilized agent's real position. The overlay is not a teleport jail. Guard, tag and rescue markers are the true latest-transition fields. Capture countdown is a level; score is cumulative; tagged/released/rescued are per-step flags.

The two prospective branch pairs isolate Guard versus Wait and Rescue versus Wait while keeping the prefix and other players' actions literal and identical. A later executor must record the real states and masks; it must not fabricate an animation from the text's intended events.

## Possible future variation

Changing openings, capture duration, escort radius, flag-home scoring conditions or team/role assignments would change interception and support schedules. Such modifications would be new source versions/tasks with new provenance and acceptance; none are implemented or claimed to generalize here.

## Literal source evidence

- Conservative spatial collision: candidate.py:85 contains exactly `occupied = jnp.any(jnp.all(proposed[:,None,:] == old_pos[None,:,:], axis=2) & others, axis=1)`.
- Actual ally protection: candidate.py:92 contains exactly `protected = jnp.any((distance <= 1) & ally & others & guard[None,:], axis=1)`.
- Simultaneous enemy tags: candidate.py:94 contains exactly `tagged = jnp.any((distance <= 1) & ~ally & attack[:,None], axis=0) & was_free & ~protected`.
- Rescue of a previously captured partner: candidate.py:96 contains exactly `rescued = jnp.any((distance <= 1) & ally & others & rescuer[:,None], axis=0) & ~was_free`.
- Bounded capture cost: candidate.py:99 contains exactly `timer = jnp.where(tagged, 4, timer).astype(jnp.int32)`.
- Dropped flag after capture: candidate.py:107 contains exactly `dropped = (old_owner >= 0) & (timer[owner_index] > 0)`.
- Physical return prerequisite: candidate.py:118 contains exactly `home = (carrier < 0) & jnp.all(flag_pos == state["base_pos"], axis=1)`.
- Shared competitive payoff: candidate.py:141 contains exactly `team_reward = (scored - scored[::-1]).astype(jnp.float32)`.
- Raw absorbing contract: candidate.py:147 contains exactly `ns = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, proposed)`.

The line references above are static source evidence, not executed transition evidence. Method spans are reset=12-46, legal_actions=48-56, observe=58-75, advance=77-143, raw_step=145-151.

## Interface, provenance and remaining acceptance

The native boundary is reset(key) -> (observation,state), observe(state), legal_actions(state), and raw_step(key,state,actions) -> (observation,state,reward,dones,info). All state leaves are fixed-shape int32/bool arrays; observations and rewards are float32. Signed sentinels use int32. The uint32 reset key belongs to external provenance and is not copied into state. Every valid nonterminal call increments t once. Natural termination has priority over a simultaneous horizon. Already-terminal calls preserve every state leaf and t, return the preserved observation and cause flags, zero reward, and all dones true. No raw auto-reset occurs. Invalid/out-of-range/masked actions deterministically become action 0. Action 0 remains legal for every role, including inactive or immobilized roles and terminal states.

A later public JaxMARL adapter must map the listed agent names to ordered action/observation/reward/done dictionaries and expose done["__all__"]. It must preserve these observations, masks, rewards, raw terminal states, and inactive-role semantics. The author did not implement or validate that separate adapter, replace the frozen learner, or reuse old policy-bank weights. The declared A/O/U/H require matching integration and policy metadata.

Only data schemas and source AST were inspected. Static checks do not establish JIT/vmap behavior, numerical parity, physical invariants under all actions, script legality, solvability, strategic richness, learned adaptation, transfer, or scientific acceptance. No real trajectories, animations, RGB frames, benchmark scores, training results or final-test data were read or generated. The scene is declarative binding data for a later real-state replay. Each source is attempt_00 and is frozen at its recorded SHA-256; a later repair requires separate actual feedback and dispatch.

Community references were supplied as hash-bound read-only files. JaxMARL informs the future parallel agent-keyed interface; Pgx informs explicit legal masks and terminated/truncated separation; Craftax demonstrates persistent spatial game state and explicit reset handling. PureJaxRL and Craftax_Baselines_ppo_rnn.py are algorithm/integration references, not game libraries or replacements for the experimental learner. No upstream game rules or code were copied into this kernel, and no upstream benchmark implementation was read. The original rules here are attributable to GPT-6-Astra/max.

- references\JaxMARL_README.md (https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), SHA-256 `6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15`.
- references\Pgx_README.md (https://github.com/sotetsuk/pgx/blob/main/README.md), SHA-256 `153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df`.
- references\PureJaxRL_README.md (https://github.com/luchris429/purejaxrl/blob/main/README.md), SHA-256 `d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0`.
- references\Craftax_Baselines_ppo_rnn.py (https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py), SHA-256 `5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011`.
- references\Craftax_README.md (https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), SHA-256 `7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff`.
