"""Freight canal locks and shared water: original unexecuted spatial rules."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 100
    num_actions = 11
    max_steps = 128

    def reset(self, key):
        x = jnp.arange(11)[None, :]
        y = jnp.arange(5)[:, None]
        water = (y >= 1) & (y <= 3) & ((x <= 3) | (x >= 7) | (y == 1) | (y == 3))
        tiles = jnp.where(water, 0, 1).astype(jnp.int32)
        tiles = tiles.at[1, 9].set(2).at[3, 9].set(2).at[2, 1].set(2)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False),
             'terminated': jnp.array(False), 'truncated': jnp.array(False), 'tiles': tiles,
             'turn_offset': jax.random.randint(key, (), 0, 3, dtype=jnp.int32),
             'boat_pos': jnp.array([[2, 1], [2, 3], [8, 2]], jnp.int32),
             'boat_active': jnp.ones(3, jnp.bool_), 'boat_visible': jnp.ones((3, 3), jnp.bool_),
             'cargo': jnp.ones(3, jnp.int32), 'delivered': jnp.zeros(3, jnp.int32),
             'lock_pos': jnp.array([[5, 1], [5, 3]], jnp.int32),
             'lock_active': jnp.ones(2, jnp.bool_), 'lock_visible': jnp.ones((3, 2), jnp.bool_),
             'level': jnp.zeros(2, jnp.int32), 'flow': jnp.zeros(2, jnp.int32),
             'last_operator': jnp.full(2, -1, jnp.int32),
             'gate_pos': jnp.array([[4, 1], [6, 1], [4, 3], [6, 3]], jnp.int32),
             'gate_active': jnp.ones(4, jnp.bool_), 'gate_visible': jnp.ones((3, 4), jnp.bool_),
             'gate_open': jnp.array([True, False, True, False]),
             'dock_pos': jnp.array([[9, 1], [9, 3], [1, 2]], jnp.int32),
             'dock_active': jnp.ones(3, jnp.bool_), 'dock_visible': jnp.ones((3, 3), jnp.bool_),
             'dock_served': jnp.zeros(3, jnp.int32),
             'reservoir_pos': jnp.array([[5, 2]], jnp.int32), 'reservoir_active': jnp.ones(1, jnp.bool_),
             'reservoir_visible': jnp.ones((3, 1), jnp.bool_), 'stock': jnp.array([2], jnp.int32),
             'water_total': jnp.array([2], jnp.int32), 'delivery_count': jnp.array(0, jnp.int32)}
        return self.observe(s), s

    def observe(self, s):
        boats = jnp.concatenate((s['boat_pos'] / 10.0, s['cargo'][:, None], s['delivered'][:, None]), axis=1)
        locks = jnp.concatenate((s['lock_pos'] / 10.0, s['level'][:, None] / 2.0,
            s['gate_open'].reshape(2, 2), s['flow'][:, None] / 2.0,
            (s['last_operator'][:, None] + 1) / 3.0), axis=1)
        docks = jnp.concatenate((s['dock_pos'] / 10.0, s['dock_served'][:, None]), axis=1)
        reservoir = jnp.concatenate((s['reservoir_pos'] / 10.0, s['stock'][:, None] / 2.0), axis=1)
        public = jnp.concatenate((jnp.array([s['t'] / 128.0]), (s['tiles'] / 2.0).reshape(55),
            boats.reshape(12), locks.reshape(14), docks.reshape(9), reservoir.reshape(3),
            jax.nn.one_hot((s['t'] + s['turn_offset']) % 3, 3)))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, 97))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        return (jnp.arange(11)[None, :] == 0) | ((~s['done'] & s['boat_active'])[:, None])

    def raw_step(self, key, s, actions):
        a = jnp.asarray(actions, jnp.int32)
        a = jnp.where((a >= 0) & (a < 11) & s['boat_active'] & ~s['done'], a, 0)
        n = dict(s)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        target = jnp.clip(s['boat_pos'] + delta[a], jnp.array([0, 0]), jnp.array([10, 4]))
        lock_index = (s['boat_pos'][:, 1] == 3).astype(jnp.int32)
        low_cross = ((s['boat_pos'][:, 0] == 3) & (target[:, 0] == 4)) | ((s['boat_pos'][:, 0] == 4) & (target[:, 0] == 3))
        high_cross = ((s['boat_pos'][:, 0] == 6) & (target[:, 0] == 7)) | ((s['boat_pos'][:, 0] == 7) & (target[:, 0] == 6))
        pass_gate = (~low_cross | s['gate_open'][2 * lock_index]) & (~high_cross | s['gate_open'][2 * lock_index + 1])
        can_move = (s['tiles'][target[:, 1], target[:, 0]] != 1) & pass_gate & s['boat_active']
        # Ordered occupancy arbitration: a vacated cell is available later in the order.
        for offset in range(3):
            i = (s['t'] + s['turn_offset'] + offset) % 3
            occupied = jnp.any(n['boat_active'] & (jnp.arange(3) != i) & jnp.all(n['boat_pos'] == target[i], axis=1))
            n['boat_pos'] = n['boat_pos'].at[i].set(jnp.where(can_move[i] & ~occupied, target[i], n['boat_pos'][i]))
        # One nominated command per chamber, selected from boats at its gate/inside it.
        operating = (n['boat_pos'][:, 0] >= 3) & (n['boat_pos'][:, 0] <= 7) & (
            (n['boat_pos'][:, 1] == 1) | (n['boat_pos'][:, 1] == 3)) & s['boat_active'] & (a >= 5)
        boat_lock = (n['boat_pos'][:, 1] == 3).astype(jnp.int32)
        rank = (jnp.arange(3) - s['t'] - s['turn_offset']) % 3
        commands = jnp.zeros(2, jnp.int32)
        operators = jnp.full(2, -1, jnp.int32)
        for k in range(2):
            eligible = operating & (boat_lock == k)
            winner = jnp.argmin(jnp.where(eligible, rank, 9)).astype(jnp.int32)
            commands = commands.at[k].set(jnp.where(jnp.any(eligible), a[winner], 0))
            operators = operators.at[k].set(jnp.where(jnp.any(eligible), winner, -1))
        # Alternating chamber order settles simultaneous claims on the shared two units.
        for offset in range(2):
            k = (s['t'] + offset) % 2
            other = 1 - k
            cmd = commands[k]
            lo = n['gate_open'][2 * k]
            hi = n['gate_open'][2 * k + 1]
            closed = ~lo & ~hi
            fill = (cmd == 8) & closed & (n['level'][k] < 2) & (n['stock'][0] > 0)
            drain = (cmd == 9) & closed & (n['level'][k] > 0)
            transfer = (cmd == 10) & closed & ~n['gate_open'][2 * other] & ~n['gate_open'][2 * other + 1] & (
                n['level'][k] > 0) & (n['level'][other] < 2)
            change = fill.astype(jnp.int32) - drain.astype(jnp.int32) - transfer.astype(jnp.int32)
            n['level'] = n['level'].at[k].add(change).at[other].add(transfer.astype(jnp.int32))
            n['stock'] = n['stock'] - fill.astype(jnp.int32) + drain.astype(jnp.int32)
            open_low = (cmd == 5) & (n['level'][k] == 0) & ~hi
            open_high = (cmd == 6) & (n['level'][k] == 2) & ~lo
            close = cmd == 7
            n['gate_open'] = n['gate_open'].at[2 * k].set((lo | open_low) & ~close)
            n['gate_open'] = n['gate_open'].at[2 * k + 1].set((hi | open_high) & ~close)
        n['flow'] = n['level'] - s['level']
        n['last_operator'] = operators
        delivered_now = s['boat_active'] & jnp.all(n['boat_pos'] == s['dock_pos'], axis=1)
        n['delivered'] = s['delivered'] + delivered_now.astype(jnp.int32)
        n['cargo'] = s['cargo'] - delivered_now.astype(jnp.int32)
        n['boat_active'] = n['delivered'] == 0
        n['boat_visible'] = jnp.broadcast_to(n['boat_active'][None, :], (3, 3))
        n['dock_served'] = n['delivered']
        n['delivery_count'] = jnp.sum(n['delivered'])
        n['water_total'] = jnp.array([jnp.sum(n['level']) + n['stock'][0]], jnp.int32)
        n['t'] = s['t'] + jnp.array(1, jnp.int32)
        n['terminated'] = jnp.all(n['delivered'] == 1)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        # Self freight matters more, while each completed delivery benefits the team.
        pumped = jnp.any((operators[None, :] == jnp.arange(3)[:, None]) & (commands[None, :] >= 8), axis=1)
        reward = (6.0 * delivered_now + jnp.sum(delivered_now.astype(jnp.float32)) -
                  0.02 * s['boat_active'] - 0.03 * pumped).astype(jnp.float32)
        n = {k: jnp.where(s['done'], s[k], n[k]) for k in s}
        reward = jnp.where(s['done'], jnp.zeros(3, jnp.float32), reward)
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), {
            'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
