"""Pressure Relay: an original discrete, fictional expedition kernel."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 15
    observation_size = 155
    max_steps = 160

    def room(self, pos):
        return jnp.where(pos[..., 0] < 4, 0, jnp.where(pos[..., 0] < 8, 1, 2)).astype(jnp.int32)

    def refresh(self, s):
        s = dict(s)
        held = s['carrier'] >= 0
        s['cargo_pos'] = jnp.where(held[:, None], s['pos'][jnp.maximum(s['carrier'], 0)], s['cargo_pos'])
        s['cargo_active'] = s['cargo_status'] < 2
        s['cargo_vis'] = jnp.broadcast_to(s['cargo_active'][None, :], (3, 2))
        s['room_id'] = self.room(s['pos'])
        s['carrying'] = jnp.sum((s['carrier'][None, :] == jnp.arange(3)[:, None]).astype(jnp.int32), axis=1)
        return s

    def reset(self, key):
        tile = jnp.zeros((7, 13), jnp.int32)
        tile = tile.at[0, :].set(1).at[6, :].set(1).at[:, 0].set(1).at[:, 12].set(1)
        tile = tile.at[:, 4].set(1).at[:, 8].set(1).at[3, 4].set(2).at[3, 8].set(2)
        s = {'t': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False),
             'terminated': jnp.array(False), 'truncated': jnp.array(False), 'tile': tile,
             'pos': jnp.array([[3, 3], [6, 3], [9, 3]], jnp.int32),
             'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
             'oxygen': jnp.full(3, 24, jnp.int32), 'role': jnp.arange(3, dtype=jnp.int32),
             'room_id': jnp.array([0, 1, 2], jnp.int32), 'carrying': jnp.zeros(3, jnp.int32),
             'last_action': jnp.zeros(3, jnp.int32), 'air_given': jnp.zeros(3, jnp.int32),
             'room_pos': jnp.array([[2, 2], [6, 2], [10, 2]], jnp.int32),
             'room_active': jnp.ones(3, jnp.bool_), 'room_vis': jnp.ones((3, 3), jnp.bool_),
             'pressure': jnp.array([12, 6, 2], jnp.int32),
             'door_pos': jnp.array([[4, 3], [8, 3]], jnp.int32),
             'door_active': jnp.ones(2, jnp.bool_), 'door_vis': jnp.ones((3, 2), jnp.bool_),
             'door_open': jnp.zeros(2, jnp.bool_), 'door_changed': jnp.zeros(2, jnp.bool_),
             'flow': jnp.zeros(2, jnp.int32), 'reservoir': jnp.array([32], jnp.int32),
             'leak_pos': jnp.array([[10, 5]], jnp.int32), 'leak_active': jnp.ones(1, jnp.bool_),
             'leak_vis': jnp.ones((3, 1), jnp.bool_), 'leaking': jnp.array([True]),
             'patches': jnp.array([1], jnp.int32), 'dock_pos': jnp.array([[2, 3]], jnp.int32),
             'dock_active': jnp.ones(1, jnp.bool_), 'dock_vis': jnp.ones((3, 1), jnp.bool_),
             'cargo_pos': jnp.array([[10, 2], [10, 4]], jnp.int32),
             'cargo_active': jnp.ones(2, jnp.bool_), 'cargo_vis': jnp.ones((3, 2), jnp.bool_),
             'carrier': jnp.full(2, -1, jnp.int32), 'cargo_status': jnp.zeros(2, jnp.int32),
             'integrity': jnp.full(2, 3, jnp.int32), 'shock': jnp.zeros(2, jnp.bool_),
             'delivered_step': jnp.zeros(2, jnp.bool_), 'pumped': jnp.array(0, jnp.int32)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        a = jnp.arange(3)
        near_door = jnp.any(jnp.sum(jnp.abs(s['pos'][:, None, :] - s['door_pos'][None, :, :]), axis=2) <= 1, axis=1)
        near_panel = jnp.any(jnp.sum(jnp.abs(s['pos'][:, None, :] - s['room_pos'][None, :, :]), axis=2) <= 1, axis=1)
        m = jnp.ones((3, 15), jnp.bool_)
        m = m.at[:, 5].set(near_door).at[:, 6].set(near_door & (s['oxygen'] >= 2))
        m = m.at[:, 7].set((a == 1) & near_panel & (s['reservoir'][0] > 0))
        m = m.at[:, 8].set((a == 0) & s['leaking'][0] & (s['patches'][0] > 0) & (jnp.sum(jnp.abs(s['pos'] - s['leak_pos'][0]), axis=1) <= 1))
        m = m.at[:, 9].set(near_panel & (s['reservoir'][0] > 0) & (s['oxygen'] < 24))
        m = m.at[:, 10].set(s['carrying'] == 0).at[:, 11].set(s['carrying'] > 0)
        m = m.at[:, 12].set(s['oxygen'] > 3).at[:, 13].set((s['carrying'] > 0) & (jnp.sum(jnp.abs(s['pos'] - s['dock_pos'][0]), axis=1) <= 1))
        m = m.at[:, 14].set(near_door)
        return (m & ~s['episode_done']).at[:, 0].set(True)

    def raw_step(self, key, state, actions):
        s = dict(state)
        ids = jnp.arange(3)
        ac = jnp.clip(actions, 0, 14)
        ac = jnp.where((actions >= 0) & (actions < 15) & self.legal_actions(state)[ids, ac], ac, 0)
        s['last_action'] = ac
        s['air_given'] = jnp.zeros(3, jnp.int32)
        s['shock'] = jnp.zeros(2, jnp.bool_)
        s['delivered_step'] = jnp.zeros(2, jnp.bool_)
        s['pumped'] = jnp.array(0, jnp.int32)
        vent = jnp.zeros(2, jnp.bool_)
        for d in range(2):
            nearby = jnp.sum(jnp.abs(s['pos'] - s['door_pos'][d]), axis=1) <= 1
            request = nearby & ((ac == 5) | (ac == 6) | (ac == 14))
            i = jnp.argmin(jnp.where(request, (ids - state['t']) % 3, 99))
            use = jnp.any(request)
            occupied = jnp.any(jnp.all(s['pos'] == s['door_pos'][d], axis=1))
            safe = jnp.abs(s['pressure'][d] - s['pressure'][d + 1]) <= 4
            toggle = use & (ac[i] == 5) & jnp.where(s['door_open'][d], ~occupied, safe)
            forced = use & (ac[i] == 6)
            s['door_open'] = s['door_open'].at[d].set(jnp.where(forced, True, s['door_open'][d] ^ toggle))
            s['oxygen'] = s['oxygen'].at[i].add(-2 * forced.astype(jnp.int32))
            vent = vent.at[d].set(use & (ac[i] == 14))
        s['door_changed'] = s['door_open'] != state['door_open']
        repair = ac[0] == 8
        s['leaking'] = s['leaking'].at[0].set(s['leaking'][0] & ~repair)
        s['patches'] = s['patches'].at[0].add(-repair.astype(jnp.int32))
        for k in range(3):
            i = (state['t'] + k) % 3
            r = self.room(s['pos'][i])
            pump = jnp.minimum(jnp.minimum(2, 12 - s['pressure'][r]), s['reservoir'][0]) * (ac[i] == 7)
            fill = jnp.minimum(jnp.minimum(6, 24 - s['oxygen'][i]), s['reservoir'][0]) * (ac[i] == 9)
            s['pressure'] = s['pressure'].at[r].add(pump)
            s['oxygen'] = s['oxygen'].at[i].add(fill)
            s['reservoir'] = s['reservoir'].at[0].add(-pump - fill)
            s['pumped'] = s['pumped'] + pump
        delta = s['pressure'][:2] - s['pressure'][1:]
        magnitude = jnp.minimum(jnp.abs(delta) // 2, jnp.where(s['door_open'], 3, vent.astype(jnp.int32)))
        s['flow'] = jnp.sign(delta).astype(jnp.int32) * magnitude
        s['pressure'] = s['pressure'] + jnp.array([-s['flow'][0], s['flow'][0] - s['flow'][1], s['flow'][1]], jnp.int32)
        s['pressure'] = s['pressure'].at[2].set(jnp.maximum(0, s['pressure'][2] - s['leaking'][0].astype(jnp.int32)))
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposal = s['pos'] + directions[jnp.where(ac <= 4, ac, 0)]
        q = jnp.clip(proposal, jnp.array([0, 0]), jnp.array([12, 6]))
        door_at = jnp.all(q[:, None, :] == s['door_pos'][None, :, :], axis=2)
        blocked = (s['tile'][q[:, 1], q[:, 0]] == 1) | jnp.any(door_at & ~s['door_open'][None, :], axis=1)
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        move = (ac >= 1) & (ac <= 4) & ~blocked & ~occupied & ~conflict
        s['pos'] = jnp.where(move[:, None], q, s['pos'])
        reward = jnp.full(3, -0.001, jnp.float32)
        for k in range(3):
            i = (state['t'] + k) % 3
            holding = jnp.any(s['carrier'] == i)
            c = jnp.argmax((s['carrier'] == i).astype(jnp.int32))
            floor = (s['cargo_status'] == 0) & (jnp.sum(jnp.abs(s['cargo_pos'] - s['pos'][i]), axis=1) <= 1)
            take = (ac[i] == 10) & ~holding & jnp.any(floor)
            f = jnp.argmax(floor.astype(jnp.int32))
            s['carrier'] = s['carrier'].at[f].set(jnp.where(take, i, s['carrier'][f]))
            s['cargo_status'] = s['cargo_status'].at[f].set(jnp.where(take, 1, s['cargo_status'][f]))
            empty = ~jnp.any(s['carrier'][None, :] == ids[:, None], axis=1)
            near = (jnp.sum(jnp.abs(s['pos'] - s['pos'][i]), axis=1) == 1) & (ids != i)
            recipient = near & empty
            j = jnp.argmin(jnp.where(recipient, (ids - state['t']) % 3, 99))
            hand = (ac[i] == 11) & holding
            s['carrier'] = s['carrier'].at[c].set(jnp.where(hand, jnp.where(jnp.any(recipient), j, -1), s['carrier'][c]))
            s['cargo_status'] = s['cargo_status'].at[c].set(jnp.where(hand & ~jnp.any(recipient), 0, s['cargo_status'][c]))
            s['cargo_pos'] = s['cargo_pos'].at[c].set(jnp.where(hand, s['pos'][i], s['cargo_pos'][c]))
            gift_target = near & (s['oxygen'] < 24)
            j = jnp.argmin(jnp.where(gift_target, (ids - state['t']) % 3, 99))
            gift = jnp.minimum(3, jnp.minimum(jnp.maximum(s['oxygen'][i] - 1, 0), 24 - s['oxygen'][j])) * ((ac[i] == 12) & jnp.any(gift_target))
            s['oxygen'] = s['oxygen'].at[i].add(-gift).at[j].add(gift)
            s['air_given'] = s['air_given'].at[i].set(gift)
            deliver = (ac[i] == 13) & holding
            s['carrier'] = s['carrier'].at[c].set(jnp.where(deliver, -1, s['carrier'][c]))
            s['cargo_status'] = s['cargo_status'].at[c].set(jnp.where(deliver, 2, s['cargo_status'][c]))
            s['cargo_pos'] = s['cargo_pos'].at[c].set(jnp.where(deliver, s['dock_pos'][0], s['cargo_pos'][c]))
            s['delivered_step'] = s['delivered_step'].at[c].set(s['delivered_step'][c] | deliver)
            reward = reward + 0.6 * deliver.astype(jnp.float32)
            reward = reward.at[i].add(0.4 * deliver.astype(jnp.float32))
        s = self.refresh(s)
        holder = jnp.maximum(s['carrier'], 0)
        before = state['pressure'][self.room(state['pos'][holder])]
        after = s['pressure'][s['room_id'][holder]]
        s['shock'] = (s['carrier'] >= 0) & (s['carrier'] != 2) & (jnp.abs(after - before) >= 3)
        s['integrity'] = jnp.maximum(0, s['integrity'] - s['shock'].astype(jnp.int32))
        shattered = s['integrity'] == 0
        s['cargo_status'] = jnp.where(shattered, 3, s['cargo_status'])
        s['carrier'] = jnp.where(shattered, -1, s['carrier'])
        pressure_here = s['pressure'][s['room_id']]
        breath = jnp.where(pressure_here >= 6, 0, jnp.where(pressure_here >= 3, 1, 2))
        s['oxygen'] = jnp.maximum(0, s['oxygen'] - breath)
        success = jnp.all(s['cargo_status'] == 2)
        failure = (jnp.any(s['oxygen'] == 0) | jnp.any(shattered)) & ~success
        reward = reward - 0.5 * failure.astype(jnp.float32)
        s['t'] = state['t'] + 1
        s['terminated'] = success | failure
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['episode_done'] = s['terminated'] | s['truncated']
        s = self.refresh(s)
        s = jax.tree.map(lambda new, old: jnp.where(state['episode_done'], old, new), s, state)
        reward = jnp.where(state['episode_done'], jnp.zeros(3, jnp.float32), reward)
        return self.observe(s), s, reward, jnp.full(3, s['episode_done']), {'done_all': s['episode_done'], 'terminated': s['terminated'], 'truncated': s['truncated']}

    def observe(self, s):
        common = jnp.concatenate((s['tile'].reshape(-1), s['pos'].reshape(-1), s['oxygen'], s['role'],
            s['pressure'], s['room_pos'].reshape(-1), s['door_pos'].reshape(-1), s['door_open'],
            s['reservoir'], s['leaking'], s['patches'], s['leak_pos'].reshape(-1), s['dock_pos'].reshape(-1),
            s['cargo_pos'].reshape(-1), s['carrier'], s['cargo_status'], s['integrity'], s['last_action'],
            s['air_given'], s['door_changed'], s['flow'], s['shock'], s['delivered_step'],
            jnp.array([s['t'], s['episode_done'], s['pumped']], jnp.int32))).astype(jnp.float32)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(common, (3, common.shape[0]))), axis=1)
