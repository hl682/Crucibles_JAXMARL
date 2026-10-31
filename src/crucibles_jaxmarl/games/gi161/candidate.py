"""Tetherwake: integer cable constraints, anchors, drift and shared power."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 14
    observation_size = 210
    max_steps = 160

    def gates(self, s):
        phase = (s['t'] // 4) % 2
        occupied = jnp.any(jnp.all(s['gate_pos'][:, None, :] == s['pos'][None, :, :], axis=2), axis=1)
        occupied = occupied | jnp.any(jnp.all(s['gate_pos'][:, None, :] == s['cargo_pos'][None, :, :], axis=2) & s['cargo_active'][None, :], axis=1)
        return (jnp.arange(2) == phase) | (s['gate_timer'] > 0) | occupied

    def clear(self, s, q):
        q = jnp.clip(q, jnp.array([0, 0]), jnp.array([10, 8]))
        door = jnp.all(q[..., None, :] == s['gate_pos'], axis=-1)
        return (s['tile'][q[..., 1], q[..., 0]] != 1) & ~jnp.any(door & ~s['gate_open'], axis=-1)

    def refresh(self, s):
        s = dict(s)
        s['cargo_active'] = ~s['delivered']
        attached = s['attached'] >= 0
        cargo = s['cargo_pos'][jnp.maximum(s['attached'], 0)]
        s['tension'] = jnp.where(attached, jnp.sum(jnp.abs(s['pos'] - cargo), axis=1), 0)
        anchor = s['anchor_pos'][jnp.maximum(s['anchored'], 0)]
        s['cable_start'] = jnp.concatenate((s['pos'], s['cargo_pos']), axis=0)
        ends = jnp.concatenate((cargo, anchor), axis=0)
        s['cable_vector'] = ends - s['cable_start']
        s['cable_active'] = jnp.concatenate((attached, s['anchored'] >= 0))
        s['cable_length'] = jnp.sum(jnp.abs(s['cable_vector']), axis=1)
        s['cable_taut'] = s['cable_active'] & (s['cable_length'] >= jnp.array([3, 2, 2, 3, 3]))
        s['cable_vis'] = jnp.broadcast_to(s['cable_active'][None, :], (3, 5))
        s['endpoint_pos'] = jnp.stack((s['cable_start'], ends), axis=1).reshape(10, 2)
        s['endpoint_active'] = jnp.repeat(s['cable_active'], 2)
        s['endpoint_vis'] = jnp.broadcast_to(s['endpoint_active'][None, :], (3, 10))
        s['cargo_vis'] = jnp.broadcast_to(s['cargo_active'][None, :], (3, 2))
        s['chip_vis'] = jnp.broadcast_to(s['chip_active'][None, :], (3, 3))
        s['gate_open'] = self.gates(s)
        return s

    def reset(self, key):
        tile = jnp.zeros((9, 11), jnp.int32)
        tile = tile.at[0, :].set(1).at[8, :].set(1).at[:, 0].set(1).at[:, 10].set(1)
        tile = tile.at[1:8, 5].set(1).at[3, 5].set(2).at[5, 5].set(2)
        s = {'t': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False),
             'terminated': jnp.array(False), 'truncated': jnp.array(False), 'tile': tile,
             'pos': jnp.array([[6, 2], [6, 4], [8, 3]], jnp.int32),
             'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
             'role': jnp.arange(3, dtype=jnp.int32), 'attached': jnp.full(3, -1, jnp.int32),
             'rope_limit': jnp.array([3, 2, 2], jnp.int32), 'tension': jnp.zeros(3, jnp.int32),
             'last_action': jnp.zeros(3, jnp.int32), 'personal': jnp.zeros(3, jnp.int32),
             'cargo_pos': jnp.array([[7, 3], [8, 6]], jnp.int32), 'cargo_active': jnp.ones(2, jnp.bool_),
             'cargo_vis': jnp.ones((3, 2), jnp.bool_), 'delivered': jnp.zeros(2, jnp.bool_),
             'anchored': jnp.full(2, -1, jnp.int32), 'pulled': jnp.zeros(2, jnp.bool_),
             'drifted': jnp.zeros(2, jnp.bool_), 'strained': jnp.zeros(2, jnp.bool_),
             'snapped': jnp.zeros(3, jnp.bool_), 'delivered_step': jnp.zeros(2, jnp.bool_),
             'anchor_pos': jnp.array([[8, 2], [8, 6]], jnp.int32),
             'anchor_active': jnp.ones(2, jnp.bool_), 'anchor_vis': jnp.ones((3, 2), jnp.bool_),
             'gate_pos': jnp.array([[5, 3], [5, 5]], jnp.int32), 'gate_active': jnp.ones(2, jnp.bool_),
             'gate_vis': jnp.ones((3, 2), jnp.bool_), 'gate_open': jnp.array([True, False]),
             'gate_timer': jnp.zeros(2, jnp.int32), 'power': jnp.array([20], jnp.int32),
             'dock_pos': jnp.array([[2, 3], [2, 5]], jnp.int32), 'dock_active': jnp.ones(2, jnp.bool_),
             'dock_vis': jnp.ones((3, 2), jnp.bool_), 'charger_pos': jnp.array([[7, 4]], jnp.int32),
             'charger_active': jnp.ones(1, jnp.bool_), 'charger_vis': jnp.ones((3, 1), jnp.bool_),
             'chip_pos': jnp.array([[9, 2], [7, 7], [3, 5]], jnp.int32),
             'chip_active': jnp.ones(3, jnp.bool_), 'chip_vis': jnp.ones((3, 3), jnp.bool_),
             'cable_start': jnp.zeros((5, 2), jnp.int32), 'cable_vector': jnp.zeros((5, 2), jnp.int32),
             'cable_active': jnp.zeros(5, jnp.bool_), 'cable_vis': jnp.zeros((3, 5), jnp.bool_),
             'cable_length': jnp.zeros(5, jnp.int32), 'cable_taut': jnp.zeros(5, jnp.bool_),
             'endpoint_pos': jnp.zeros((10, 2), jnp.int32), 'endpoint_active': jnp.zeros(10, jnp.bool_),
             'endpoint_vis': jnp.zeros((3, 10), jnp.bool_), 'endpoint_link': jnp.repeat(jnp.arange(5, dtype=jnp.int32), 2),
             'endpoint_side': jnp.tile(jnp.array([0, 1], jnp.int32), 5)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(3)
        m = jnp.ones((3, 14), jnp.bool_)
        near_cargo = jnp.any((jnp.sum(jnp.abs(s['pos'][:, None, :] - s['cargo_pos'][None, :, :]), axis=2) <= s['rope_limit'][:, None]) & s['cargo_active'][None, :], axis=1)
        m = m.at[:, 5].set((s['attached'] >= 0) | near_cargo)
        m = m.at[:, 6:10].set(jnp.broadcast_to((s['attached'] >= 0)[:, None], (3, 4)))
        near_gate = jnp.any(jnp.sum(jnp.abs(s['pos'][:, None, :] - s['gate_pos'][None, :, :]), axis=2) <= 1, axis=1)
        m = m.at[:, 10].set((ids == 2) & near_gate & (s['power'][0] >= 3))
        m = m.at[:, 11].set((ids == 2) & (jnp.sum(jnp.abs(s['pos'] - s['charger_pos'][0]), axis=1) <= 1))
        m = m.at[:, 13].set(ids == 0)
        return (m & ~s['episode_done']).at[:, 0].set(True)

    def raw_step(self, key, state, actions):
        s = dict(state)
        ids = jnp.arange(3)
        ac = jnp.clip(actions, 0, 13)
        ac = jnp.where((actions >= 0) & (actions < 14) & self.legal_actions(state)[ids, ac], ac, 0)
        s['last_action'] = ac
        s['pulled'] = jnp.zeros(2, jnp.bool_)
        s['drifted'] = jnp.zeros(2, jnp.bool_)
        s['strained'] = jnp.zeros(2, jnp.bool_)
        s['snapped'] = jnp.zeros(3, jnp.bool_)
        s['delivered_step'] = jnp.zeros(2, jnp.bool_)
        s['gate_timer'] = jnp.maximum(0, s['gate_timer'] - 1)
        near_gate = jnp.sum(jnp.abs(s['gate_pos'] - s['pos'][2]), axis=1)
        g = jnp.argmin(near_gate)
        override = ac[2] == 10
        s['gate_timer'] = s['gate_timer'].at[g].set(jnp.where(override, 3, s['gate_timer'][g]))
        s['power'] = s['power'].at[0].add(-3 * override.astype(jnp.int32))
        s['power'] = s['power'].at[0].set(jnp.minimum(24, s['power'][0] + 4 * (ac[2] == 11).astype(jnp.int32)))
        s['gate_open'] = self.gates(s)
        for i in range(3):
            distance = jnp.sum(jnp.abs(s['cargo_pos'] - s['pos'][i]), axis=1)
            eligible = s['cargo_active'] & (distance <= s['rope_limit'][i])
            c = jnp.argmin(jnp.where(eligible, distance, 99))
            link = jnp.where(s['attached'][i] >= 0, -1, jnp.where(jnp.any(eligible), c, -1))
            s['attached'] = s['attached'].at[i].set(jnp.where(ac[i] == 5, link, s['attached'][i]))
        distance = jnp.sum(jnp.abs(s['cargo_pos'] - s['pos'][0]), axis=1)
        c = jnp.argmin(jnp.where(s['cargo_active'], distance, 99))
        a = jnp.argmin(jnp.sum(jnp.abs(s['anchor_pos'] - s['pos'][0]), axis=1))
        near_anchor = jnp.sum(jnp.abs(s['anchor_pos'][a] - s['pos'][0])) <= 1
        anchor_reach = jnp.sum(jnp.abs(s['cargo_pos'][c] - s['anchor_pos'][a])) <= 3
        anchor_action = (ac[0] == 13) & s['cargo_active'][c] & (distance[c] <= 3)
        anchor_ok = (s['anchored'][c] >= 0) | (near_anchor & anchor_reach)
        s['anchored'] = s['anchored'].at[c].set(jnp.where(anchor_action & anchor_ok, jnp.where(s['anchored'][c] >= 0, -1, a), s['anchored'][c]))
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = jnp.clip(s['pos'] + directions[jnp.where(ac <= 4, ac, 0)], jnp.array([0, 0]), jnp.array([10, 8]))
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        cargo_block = jnp.any(jnp.all(q[:, None, :] == s['cargo_pos'][None, :, :], axis=2) & s['cargo_active'][None, :], axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        attached_cargo = s['cargo_pos'][jnp.maximum(s['attached'], 0)]
        rope_ok = (s['attached'] < 0) | (jnp.sum(jnp.abs(q - attached_cargo), axis=1) <= s['rope_limit'])
        moving = (ac >= 1) & (ac <= 4) & self.clear(s, q) & ~occupied & ~cargo_block & ~conflict & rope_ok
        s['pos'] = jnp.where(moving[:, None], q, s['pos'])
        reward = jnp.full(3, -0.001, jnp.float32)
        for k in range(3):
            i = (state['t'] + k) % 3
            near = (jnp.sum(jnp.abs(s['chip_pos'] - s['pos'][i]), axis=1) <= 1) & s['chip_active']
            chip = jnp.argmax(near.astype(jnp.int32))
            take = (ac[i] == 12) & jnp.any(near)
            s['chip_active'] = s['chip_active'].at[chip].set(s['chip_active'][chip] & ~take)
            s['personal'] = s['personal'].at[i].add(take.astype(jnp.int32))
            s['power'] = s['power'].at[0].set(jnp.minimum(24, s['power'][0] + 2 * take.astype(jnp.int32)))
            reward = reward.at[i].add(0.3 * take.astype(jnp.float32))
        for c in range(2):
            participants = (s['attached'] == c) & (ac >= 6) & (ac <= 9)
            votes = jnp.sum((participants[:, None] & (ac[:, None] == jnp.arange(6, 10)[None, :])).astype(jnp.int32), axis=0)
            direction = jnp.argmax(votes)
            crew = (votes[direction] >= 2) & participants[1] & (ac[1] == direction + 6)
            q = s['cargo_pos'][c] + directions[direction + 1]
            tether_ok = jnp.all((s['attached'] != c) | (jnp.sum(jnp.abs(s['pos'] - q), axis=1) <= s['rope_limit']))
            anchor = s['anchor_pos'][jnp.maximum(s['anchored'][c], 0)]
            anchor_ok = (s['anchored'][c] < 0) | (jnp.sum(jnp.abs(q - anchor)) <= 3)
            occupied = jnp.any(jnp.all(s['pos'] == q, axis=1)) | jnp.any(jnp.all(s['cargo_pos'] == q, axis=1) & s['cargo_active'])
            pull = s['cargo_active'][c] & crew & (s['power'][0] >= 2) & tether_ok & anchor_ok & self.clear(s, q) & ~occupied
            s['cargo_pos'] = s['cargo_pos'].at[c].set(jnp.where(pull, q, s['cargo_pos'][c]))
            s['power'] = s['power'].at[0].add(-2 * pull.astype(jnp.int32))
            s['pulled'] = s['pulled'].at[c].set(pull)
            drift_target = s['cargo_pos'][c] + jnp.array([1, 0], jnp.int32)
            occupied = jnp.any(jnp.all(s['pos'] == drift_target, axis=1)) | jnp.any(jnp.all(s['cargo_pos'] == drift_target, axis=1) & s['cargo_active'])
            drift = s['cargo_active'][c] & ~pull & (s['anchored'][c] < 0) & ((state['t'] + 1) % 3 == 0) & self.clear(s, drift_target) & ~occupied
            tether_ok = jnp.all((s['attached'] != c) | (jnp.sum(jnp.abs(s['pos'] - drift_target), axis=1) <= s['rope_limit']))
            strain = drift & ~tether_ok
            powered_hold = strain & (s['power'][0] > 0)
            snap = strain & ~powered_hold
            broken = (s['attached'] == c) & snap
            s['snapped'] = s['snapped'] | broken
            s['attached'] = jnp.where(broken, -1, s['attached'])
            s['power'] = s['power'].at[0].add(-powered_hold.astype(jnp.int32))
            s['cargo_pos'] = s['cargo_pos'].at[c].set(jnp.where(drift & ~powered_hold, drift_target, s['cargo_pos'][c]))
            s['drifted'] = s['drifted'].at[c].set(drift & ~powered_hold)
            s['strained'] = s['strained'].at[c].set(strain)
            delivered = s['cargo_active'][c] & jnp.all(s['cargo_pos'][c] == s['dock_pos'][c])
            s['delivered'] = s['delivered'].at[c].set(s['delivered'][c] | delivered)
            s['cargo_active'] = s['cargo_active'].at[c].set(~s['delivered'][c])
            s['delivered_step'] = s['delivered_step'].at[c].set(delivered)
            s['attached'] = jnp.where(delivered & (s['attached'] == c), -1, s['attached'])
            s['anchored'] = s['anchored'].at[c].set(jnp.where(delivered, -1, s['anchored'][c]))
            reward = reward + delivered.astype(jnp.float32)
        s['t'] = state['t'] + 1
        s['terminated'] = jnp.all(s['delivered'])
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['episode_done'] = s['terminated'] | s['truncated']
        s = self.refresh(s)
        s = jax.tree.map(lambda new, old: jnp.where(state['episode_done'], old, new), s, state)
        reward = jnp.where(state['episode_done'], jnp.zeros(3, jnp.float32), reward)
        return self.observe(s), s, reward, jnp.full(3, s['episode_done']), {'done_all': s['episode_done'], 'terminated': s['terminated'], 'truncated': s['truncated']}

    def observe(self, s):
        common = jnp.concatenate((s['tile'].reshape(-1), s['pos'].reshape(-1), s['role'], s['attached'], s['rope_limit'],
            s['tension'], s['last_action'], s['personal'], s['cargo_pos'].reshape(-1), s['delivered'], s['anchored'],
            s['pulled'], s['drifted'], s['strained'], s['snapped'], s['delivered_step'], s['anchor_pos'].reshape(-1),
            s['gate_pos'].reshape(-1), s['gate_open'], s['gate_timer'], s['power'], s['dock_pos'].reshape(-1),
            s['charger_pos'].reshape(-1), s['chip_pos'].reshape(-1), s['chip_active'], s['cable_start'].reshape(-1),
            s['cable_vector'].reshape(-1), s['cable_active'], s['cable_length'], s['cable_taut'],
            jnp.array([s['t'], s['episode_done']], jnp.int32))).astype(jnp.float32)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(common, (3, common.shape[0]))), axis=1)
