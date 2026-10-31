# Charter Frontier

Four controlled settlers share an 11 by 7 grid. The wall at x=5, y=1..5 contains one gate at (5,3); the always-open top and bottom rows are longer detours. The council is (3,3), and four claimable plots lie on the eastern frontier. Co-location, swaps and crossing are allowed. Every movement action is checked against the actual pre-step gate state. Closing a lease does not teleport or evict a settler on the gate: exit to a floor cell remains possible, while new gate entry is blocked.

Actions are 0 wait, 1 north, 2 east, 3 south, 4 west, 5 claim, 6 build, 7 harvest, 8 donate, 9 propose, 10 vote yes, 11 vote no, 12 abstain. Claim/build/harvest require Manhattan distance at most one from the relevant plot; a settler may own at most one plot. Unclaimed plots do not produce. Claims use ascending actor priority and the lowest-index eligible nearby plot. Construction costs two materials and makes a persistent building. Built plots gain one maturity per tick, capped at three, including the construction tick. A legal harvest requires mature pre-step crop, ownership and proximity; collection resets maturity and gives one harvest up to two.

Each settler starts with three materials. At council distance at most one, donation voluntarily moves one unit to a shared capacity-four fund. Lower actor index resolves the last free fund slot. Materials never regenerate; eight materials are needed to build all four plots, so donations compete with eventual private construction. No donation pledge is imposed on a non-consenting agent.

The sole governance mechanism is a temporary public corridor charter. A proposal requires council proximity, at least two existing fund units, and no active ballot, pending activation or lease. If several propose, the lowest-index proposer wins. Launch starts a four-tick ballot, clears old votes and records the proposer's yes. During each of the four subsequent ticks, every settler can independently change its public vote remotely to yes, no or abstain; unchanged votes persist. Evaluation happens only at the deadline, after that tick's vote updates. At least three yes votes and two available fund units ratify; two units are then consumed. Otherwise the ballot rejects without consuming fund. There is no privileged single-person veto; abstention and no both fail to contribute to quorum, while their distinct public signals persist until changed or next launch.

Ratification sets a two-transition activation delay. After those two further raw transitions, the gate has a twelve-tick lease. Movement on the activation transition still uses the pre-step closed mask; entry is available from the next observation/action boundary. The lease declines once per transition and expires naturally. Movement on the final open lease transition is legal, even if the resulting snapshot closes the gate. A new proposal can begin only after expiry and any new funding. Delay, ballot and lease are distinct state fields and level channels, not decorative animation timers.

Reset samples off-diagonal plot values in {1,2}, with own-index value three. Harvest rewards only the actual plot owner by its private value. Each material spent incurs 0.02; full harvest completion gives one to every settler. No reward is given merely for proposing, voting or opening. The intended tradeoff is private production/material conservation versus shorter shared access: one can take the detour, help fund a charter, support or block its quorum, time a crossing before expiry, or claim a different available plot. Other settlers' votes, contribution and timing change these choices. These strategic possibilities are untested source deductions.

The native observation includes public positions, map, materials, ownership, buildings, fund, votes, proposer and all public timers. Only own plot values appear. Settler sprites carry materials and vote attributes; plot sprites bind actual claims/buildings/crop; the barrier binds gate_open and lease; council binds real fund. Private values are not renderer attributes. Invariants include nonnegative materials/fund/growth/harvests/lease, one claim per settler, at most one owner per plot, finite original materials, vote values -1/0/1 and intended timer bounds. No unique-position invariant is asserted because co-location is legal.

The single-action paired hypothesis uses seed 17601 and literal prefix donations, proposal and one additional yes vote while settler 3 walks to the gate approach. The branch changes only settler 2's next action from wait to yes. Four identical wait transitions follow. The source predicts a difference in delayed gate entry availability to settler 3; it does not report a measured effect or completed run. Prospective quorum, costs, lease durations or frontier layouts are possible future task axes only under separately frozen protocols.

Status: AUTHORED_NOT_EXECUTED. All strategies, scripts and paired interventions are source-reasoned hypotheses. No successful replay, effect magnitude, learnability, reward gain, optimality, non-isomorphism or meta-RL transfer is claimed.

The source uses only JAX/JAX NumPy imports; it was never imported or called. The author ran only the supplied trusted stdlib AST whitelist and manifest/scene data validators, plus data-only hashes, state-key/action-range/one-action-difference checks. Shapes, dtypes, jit/vmap parity, numerical bounds, gameplay, rendering, legal script execution and terminal absorption require independent frozen-source integration.

All agents are controlled; there is no scripted teammate, pursuer, planner or oracle. Every step receives a joint action vector. Invalid/out-of-range/masked actions become wait. A live raw step increments t once. A terminal input preserves every state leaf and observation, returns zero rewards and true dones, and preserves causes. Natural completion takes priority over H=96 truncation. The outer wrapper must perform auto-reset.

Representation: counters, identifiers and inventories use int32; flags use bool; observations/rewards use float32. The external reset key is not duplicated in state. Stochasticity, where present, uses only that key. No design-required distribution was removed to satisfy the interface.

Provenance: only the 22 current closed inputs listed in author_input_freeze.json were read. The five official reference files were read as bytes for integrity only; their text/code was not semantically consulted, copied or executed. These are original game kernels, with no copied upstream code or incorporated upstream license text. Reference URLs remain input-bundle provenance. No prior candidates, selection/training outcomes, heldout inputs, network sources or external models were used. Adapters, real trajectories and animation belong to independent integration. New A/O/U contracts do not license reuse of old policy weights or two-player ranking statistics.

Literal source evidence:
- candidate.py:L68: `movement = inside & (tiles != 1) & ((tiles != 2) | state['gate_open'][0])`
- candidate.py:L75: `~jnp.any(own, axis=1) & jnp.any(nearby & (state['owner'][None, :] < 0), axis=1),`
- candidate.py:L110: `construct = (actions[i] == 6) & (materials[i] >= 2) & jnp.any(buildable)`
- candidate.py:L114: `contribute = (actions[i] == 8) & near_council[i] & (materials[i] > 0) & (fund < 4)`
- candidate.py:L131: `ratified = closing & (jnp.sum(votes == 1) >= 3) & (fund >= 2)`
- candidate.py:L139: `s['ballot_remaining'] = jnp.where(launch, 4, jnp.maximum(state['ballot_remaining'] - 1, 0))`
- candidate.py:L140: `fund = fund - 2 * ratified.astype(jnp.int32)`
- candidate.py:L144: `opened = state['pending'] == 1`
- candidate.py:L145: `lease = jnp.where(opened, 12, jnp.maximum(state['lease'][0] - 1, 0))`
- candidate.py:L146: `s['pending'] = jnp.where(ratified, 2, jnp.maximum(state['pending'] - 1, 0))`
- candidate.py:L153: `s['expired'] = (state['lease'][0] == 1) & ~opened`
- candidate.py:L162: `s['terminated'] = jnp.all(s['harvests'] >= 2)`
- candidate.py:L183: `return jnp.concatenate([jnp.eye(4, dtype=jnp.float32), state['private_value'] / 3.0,`
- candidate.py:L191: `next_state = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), proposed, state)`
