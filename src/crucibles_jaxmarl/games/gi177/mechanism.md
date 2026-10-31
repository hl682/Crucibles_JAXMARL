# Tag Exchange

Four controlled players form fixed pairs (0,1) and (2,3) on an 11 by 7 arena. Base/flag homes are (1,3) and (9,3). Central wall cells at x=5, y in {1,2,4,5} leave a middle passage and outer routes. Positions govern every tag, rescue, pickup, handoff, charge and score. Co-location, swaps and crossing are deliberately allowed; a crossing move alone is not a tag. This base instance is deterministic: the public reset key is accepted but unused, and no design-mandated random distribution is being replaced.

Actions are 0 wait, 1 north, 2 east, 3 south, 4 west, 5 tag attempt, 6 rescue attempt, 7 flag interaction, 8 flag pass, 9 depot recharge, 10 self-thaw. All four agents choose one action simultaneously. Unfrozen tag/rescue attempts with energy and flag-interaction attempts are legal even without a current target; an unsuccessful attempt has no effect or cost. This permits anticipating another player's movement. Pass requires possession, and movement/recharge/self-thaw have explicit masks. A frozen player remains an active controlled role with wait and, if affordable, self-thaw; it is not an independently finished agent.

After all legal moves, tags resolve simultaneously from starting-unfrozen attackers and starting-unfrozen enemies at Manhattan distance at most one. The target is nearest, then lowest index. Each successful attacker spends one energy even if a teammate tags the same enemy. Mutual tags both succeed. A new tag sets frozen=3; otherwise the previous timer decreases by one. Tagging a previously frozen target is disallowed. The new tag drops every carried flag at the actual post-movement contact cell and prevents that player's same-tick object/recharge/rescue actions.

An already-frozen player can spend two energy to clear its timer. An untagged, starting-free teammate may spend one energy to clear its adjacent teammate's still-positive timer. A fresh same-tick tag is not rescuable. A timer that naturally expires on this tick needs no rescue and costs no rescue energy. A released player could not move on the release tick, since its action was chosen under the old frozen mask; its next boundary opens movement and interaction choices. Depot recharge requires own-base distance at most one, room below energy three, and remaining team stock. Each depot starts with four units, shared by its two players; lower player index wins the last unit. No automatic energy/stock regeneration occurs.

Flag interaction uses exact cells. An actor returns its own displaced unheld flag to its base and can pick up its unheld enemy flag if it carries none; ascending actor order resolves simultaneous requests. Returning a flag already at home is not an event. A carrier may pass to its only teammate at distance at most one if the recipient is free, untagged and empty-handed. Possession immediately follows the recipient's actual position. A team scores automatically when its enemy flag is at its own exact base in a free teammate's hands while its own flag is home and unheld. The scored flag returns to the opposing base, and play continues until a team reaches two captures. Scores, event flags and possession are native state, not renderer-generated transitions.

Both scoring teammates receive +1 and both opponents -1 on a capture. There is no reward for cosmetic tagging or rescue. Finite energy creates an opportunity cost for duplicate tags, premature self-thaw and repeated rescues; recharge competes for a shared finite depot. Carrier/support splits, escorting, intercepting, returning a stolen home flag, rescuing or handing possession across a threatened cell are plausible alternatives. A player's best response can depend on teammate position, opponent energy, flag state and whether support will act. No policy, payoff ranking or learning outcome has been measured.

The flat observation exposes the full dynamic strategic state and public topology, with observer identity/team codes. Every entity has true role-visibility masks. Agent attributes bind team, energy and freeze timers; flag positions/holders follow actual possession; base attributes bind actual depot stock and capture scores. Tag/rescue/pass/score flashes are per-step flags, frozen is a level, and scores is a cumulative counter. Intended invariants include energy 0..3, depot stock 0..4, frozen 0..3, nonnegative scores, each flag at one position with at most one holder, and only opposing-team possession. Co-location is legal, so unique active positions is not declared.

The literal seed-17701 playthrough intends to show an opponent tag, teammate rescue, a carrier route, a handoff and a capture. It is an unexecuted controlled joint-action script; even its intended score is not an execution receipt. The paired hypothesis reuses its three-step prefix, changes only player 1's rescue to wait, and then waits identically. It is intended to change player 0's remaining freeze and next legal movement opportunities, without issuing a suffix move that would be legal in only one branch. Prospective obstacles, energy, team/flag tasks or sight restrictions would require new frozen identities and tests.

Status: AUTHORED_NOT_EXECUTED. All strategies, scripts and paired interventions are source-reasoned hypotheses. No successful replay, effect magnitude, learnability, reward gain, optimality, non-isomorphism or meta-RL transfer is claimed.

The source uses only JAX/JAX NumPy imports; it was never imported or called. The author ran only the supplied trusted stdlib AST whitelist and manifest/scene data validators, plus data-only hashes, state-key/action-range/one-action-difference checks. Shapes, dtypes, jit/vmap parity, numerical bounds, gameplay, rendering, legal script execution and terminal absorption require independent frozen-source integration.

All agents are controlled; there is no scripted teammate, pursuer, planner or oracle. Every step receives a joint action vector. Invalid/out-of-range/masked actions become wait. A live raw step increments t once. A terminal input preserves every state leaf and observation, returns zero rewards and true dones, and preserves causes. Natural completion takes priority over H=96 truncation. The outer wrapper must perform auto-reset.

Representation: counters, identifiers and inventories use int32; flags use bool; observations/rewards use float32. The external reset key is not duplicated in state. Stochasticity, where present, uses only that key. No design-required distribution was removed to satisfy the interface.

Provenance: only the 22 current closed inputs listed in author_input_freeze.json were read. The five official reference files were read as bytes for integrity only; their text/code was not semantically consulted, copied or executed. These are original game kernels, with no copied upstream code or incorporated upstream license text. Reference URLs remain input-bundle provenance. No prior candidates, selection/training outcomes, heldout inputs, network sources or external models were used. Adapters, real trajectories and animation belong to independent integration. New A/O/U contracts do not license reuse of old policy weights or two-player ranking statistics.

Literal source evidence:
- candidate.py:L80: `targets = (distance <= 1) & enemy & free[None, :]`
- candidate.py:L82: `attack = (actions == 5) & free & (state['energy'] > 0) & jnp.any(targets, axis=1)`
- candidate.py:L85: `frozen = jnp.where(tagged, 3, jnp.maximum(state['frozen'] - 1, 0))`
- candidate.py:L91: `self_thaw = (actions == 10) & ~free & (energy >= 2)`
- candidate.py:L96: `rescue = rescue & (distance[jnp.arange(4), mate] <= 1) & (frozen[mate] > 0) & ~tagged[mate]`
- candidate.py:L111: `stock = stock.at[team].add(-charge.astype(jnp.int32))`
- candidate.py:L122: `holder = jnp.where(dropped, -1, holder)`
- candidate.py:L133: `pickup = interact & ~jnp.any(holder == i) & (holder[enemy_flag] < 0) & jnp.all(flag_pos[enemy_flag] == pos[i])`
- candidate.py:L142: `pass_flag = pass_flag & (distance[safe_holder, receiver] <= 1) & (frozen[receiver] == 0) & ~tagged[receiver]`
- candidate.py:L153: `home = (holder < 0) & jnp.all(flag_pos == state['base_pos'], axis=1)`
- candidate.py:L156: `goal = (enemy_holder >= 0) & home & (state['team'][scorer] == jnp.arange(2)) & (frozen[scorer] == 0)`
- candidate.py:L165: `s['terminated'] = jnp.any(s['scores'] >= 2)`
- candidate.py:L168: `rewards = goal[state['team']].astype(jnp.float32) - goal[1 - state['team']].astype(jnp.float32)`
- candidate.py:L190: `next_state = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), proposed, state)`
