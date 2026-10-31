"""Canal Compact. Original blind-authored, unexecuted source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 237
    num_actions = 13
    max_steps = 150

    def sync(self, s):
        s = dict(s)
        s['agent_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=2) <= 3
        s['lock_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['lock_pos'][None, :, :]), axis=2) <= 3
        s['berth_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['berth_pos'][None, :, :]), axis=2) <= 3
        s['field_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['field_pos'][None, :, :]), axis=2) <= 3
        s['message_active'] = s['message_ttl'] > 0
        s['message_vis'] = (jnp.arange(3)[:, None] == s['message_recipient'][None, :]) & s['message_active'][None, :]
        return s

    def reset(self, key):
        grid = jnp.zeros((7, 13), jnp.int32).at[3, :].set(1).at[1:4, 2].set(1).at[3:6, 6].set(1).at[1:4, 10].set(1)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False), 'success': jnp.array(False), 'failure': jnp.array(False), 'map': grid,
            'agent_pos': jnp.array([[3, 3], [5, 3], [9, 3]], jnp.int32), 'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_), 'reservoir': jnp.full(3, 5, jnp.int32), 'cargo': jnp.ones(3, jnp.int32), 'lost': jnp.zeros(3, jnp.int32), 'work': jnp.zeros(3, jnp.int32), 'blocked': jnp.zeros(3, jnp.bool_),
            'lock_pos': jnp.array([[4, 3], [8, 3]], jnp.int32), 'lock_active': jnp.ones(2, jnp.bool_), 'lock_vis': jnp.ones((3, 2), jnp.bool_), 'level': jnp.ones(2, jnp.int32), 'phase': jnp.zeros(2, jnp.int32), 'surge': jnp.zeros(2, jnp.bool_),
            'berth_pos': jnp.array([[2, 1], [6, 5], [10, 1]], jnp.int32), 'berth_active': jnp.ones(3, jnp.bool_), 'berth_vis': jnp.ones((3, 3), jnp.bool_), 'reservation': jnp.full(3, -1, jnp.int32), 'reserve_ttl': jnp.zeros(3, jnp.int32),
            'field_pos': jnp.array([[2, 0], [6, 6], [10, 0]], jnp.int32), 'field_active': jnp.ones(3, jnp.bool_), 'field_vis': jnp.ones((3, 3), jnp.bool_), 'moisture': jnp.full(3, 2, jnp.int32), 'growth': jnp.zeros(3, jnp.int32), 'harvested': jnp.zeros(3, jnp.int32),
            'dock_pos': jnp.array([[12, 3], [0, 3], [0, 3]], jnp.int32), 'dock_active': jnp.ones(3, jnp.bool_), 'dock_vis': jnp.ones((3, 3), jnp.bool_),
            'message_pos': jnp.zeros((9, 2), jnp.int32), 'message_active': jnp.zeros(9, jnp.bool_), 'message_vis': jnp.zeros((3, 9), jnp.bool_), 'message_recipient': jnp.repeat(jnp.arange(3, dtype=jnp.int32), 3), 'message_sender': jnp.tile(jnp.arange(3, dtype=jnp.int32), 3), 'message_level': jnp.zeros(9, jnp.int32), 'message_time': jnp.zeros(9, jnp.int32), 'message_ttl': jnp.zeros(9, jnp.int32),
            'delivered': jnp.zeros(3, jnp.int32), 'delivery_now': jnp.zeros(3, jnp.bool_), 'flood': jnp.array(0, jnp.int32), 'released': jnp.array(0, jnp.int32), 'irrigated': jnp.array(0, jnp.int32),
        }
        s = self.sync(s)
        return self.observe(s), s

    def targets(self, s):
        p = s['agent_pos'][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        inside = jnp.all((p >= 0) & (p <= jnp.array([12, 6], jnp.int32)), axis=2)
        canal = s['map'][jnp.clip(p[:, :, 1], 0, 6), jnp.clip(p[:, :, 0], 0, 12)] == 1
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=3), axis=2)
        return p, inside & canal & ~occupied

    def legal_actions(self, s):
        p = s['agent_pos']
        ln = jnp.sum(jnp.abs(p[:, None, :] - s['lock_pos'][None, :, :]), axis=2) <= 1
        bn = jnp.sum(jnp.abs(p[:, None, :] - s['berth_pos'][None, :, :]), axis=2) <= 1
        ownfield = jnp.sum(jnp.abs(p - s['field_pos']), axis=1) <= 1
        _, walk = self.targets(s)
        m = jnp.zeros((3, 13), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(walk)
        m = m.at[:, 5].set((s['reservoir'] > 0) & jnp.any(ln & (s['level'] < 2)[None, :], axis=1))
        m = m.at[:, 6].set(jnp.any(ln & (s['level'] > 0)[None, :], axis=1))
        m = m.at[:, 7].set(jnp.any(bn & ((s['reservation'] < 0)[None, :] | (s['reservation'][None, :] == jnp.arange(3)[:, None])), axis=1))
        m = m.at[:, 8].set(jnp.any(bn & (s['reservation'][None, :] == jnp.arange(3)[:, None]), axis=1))
        m = m.at[:, 9].set(ownfield & (s['reservoir'] > 0) & (s['moisture'] < 4) & (s['harvested'] < 2))
        m = m.at[:, 10].set((s['cargo'] > 0) & jnp.all(p == s['dock_pos'], axis=1))
        m = m.at[:, 11].set(ownfield & (s['growth'] >= 4) & (s['harvested'] < 2))
        m = m.at[:, 12].set(jnp.any(ln, axis=1))
        return jnp.where(s['done'], jnp.arange(13)[None, :] == 0, m)

    def observe(self, s):
        public = jnp.concatenate((jnp.stack((s['t'], s['done'], s['terminated'], s['truncated'], s['success'], s['failure'], s['flood'])), s['map'].ravel(), s['dock_pos'].ravel(), s['delivered'], s['delivery_now']))
        boats = jnp.concatenate((s['agent_pos'], s['reservoir'][:, None], s['cargo'][:, None]), axis=1)
        locks = jnp.concatenate((s['lock_pos'], s['level'][:, None], s['phase'][:, None], s['surge'][:, None]), axis=1)
        berths = jnp.concatenate((s['berth_pos'], s['reservation'][:, None], s['reserve_ttl'][:, None]), axis=1)
        fields = jnp.concatenate((s['field_pos'], s['moisture'][:, None], s['growth'][:, None], s['harvested'][:, None]), axis=1)
        messages = jnp.concatenate((s['message_pos'], s['message_sender'][:, None], s['message_level'][:, None], s['message_time'][:, None], s['message_ttl'][:, None]), axis=1)
        rows = []
        for i in range(3):
            own = jnp.stack((i, s['work'][i], s['blocked'][i], s['lost'][i]))
            row = jnp.concatenate((public, own, s['agent_vis'][i], jnp.where(s['agent_vis'][i, :, None], boats, -1).ravel(), s['lock_vis'][i], jnp.where(s['lock_vis'][i, :, None], locks, -1).ravel(), s['berth_vis'][i], jnp.where(s['berth_vis'][i, :, None], berths, -1).ravel(), s['field_vis'][i], jnp.where(s['field_vis'][i, :, None], fields, -1).ravel(), s['message_vis'][i], jnp.where(s['message_vis'][i, :, None], messages, -1).ravel()))
            rows = rows + [row.astype(jnp.float32)]
        return jnp.stack(rows)

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 12)
        a = jnp.where((actions >= 0) & (actions < 13) & self.legal_actions(s)[jnp.arange(3), clipped], clipped, 0)
        n['t'] = s['t'] + 1
        n['blocked'] = actions != a
        n['work'] = s['work'] + (a != 0).astype(jnp.int32)
        n['delivery_now'] = jnp.zeros(3, jnp.bool_)
        n['reserve_ttl'] = jnp.maximum(s['reserve_ttl'] - 1, 0)
        n['reservation'] = jnp.where(n['reserve_ttl'] > 0, s['reservation'], -1)
        n['message_ttl'] = jnp.maximum(s['message_ttl'] - 1, 0)
        near = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['lock_pos'][None, :, :]), axis=2) <= 1
        fill_target = jnp.argmax(near & (s['level'] < 2)[None, :], axis=1)
        drain_target = jnp.argmax(near & (s['level'] > 0)[None, :], axis=1)
        fills = (a == 5)[:, None] & (fill_target[:, None] == jnp.arange(2)[None, :])
        drains = (a == 6)[:, None] & (drain_target[:, None] == jnp.arange(2)[None, :])
        fc = jnp.sum(fills, axis=0).astype(jnp.int32)
        dc = jnp.sum(drains, axis=0).astype(jnp.int32)
        n['surge'] = (fc > 0) & (dc > 0)
        capacity = jnp.minimum(s['level'] + fc, 2)
        drained = jnp.minimum(capacity, dc)
        ordinary = capacity - drained
        n['level'] = jnp.where(n['surge'], 0, ordinary).astype(jnp.int32)
        n['released'] = s['released'] + jnp.sum(s['level'] + fc - n['level']).astype(jnp.int32)
        n['reservoir'] = n['reservoir'] - (a == 5).astype(jnp.int32)
        n['phase'] = jnp.where(n['surge'], 2, jnp.where(fc > 0, 1, jnp.where(dc > 0, -1, 0))).astype(jnp.int32)
        n['flood'] = s['flood'] + jnp.sum(n['surge']).astype(jnp.int32)
        bonus = jnp.zeros(3, jnp.float32)
        for k in range(3):
            i = (s['t'] + k) % 3
            bn = jnp.sum(jnp.abs(s['berth_pos'] - s['agent_pos'][i]), axis=1) <= 1
            available = bn & ((n['reservation'] < 0) | (n['reservation'] == i))
            b = jnp.argmax(available)
            reserve = (a[i] == 7) & jnp.any(available)
            n['reservation'] = n['reservation'].at[b].set(jnp.where(reserve, i, n['reservation'][b]))
            n['reserve_ttl'] = n['reserve_ttl'].at[b].set(jnp.where(reserve, 6, n['reserve_ttl'][b]))
            release = (a[i] == 8) & bn & (n['reservation'] == i)
            n['reservation'] = jnp.where(release, -1, n['reservation'])
            n['reserve_ttl'] = jnp.where(release, 0, n['reserve_ttl'])
            irrigate = a[i] == 9
            n['reservoir'] = n['reservoir'].at[i].add(-irrigate.astype(jnp.int32))
            n['moisture'] = n['moisture'].at[i].set(jnp.where(irrigate, jnp.minimum(n['moisture'][i] + 2, 4), n['moisture'][i]))
            n['irrigated'] = n['irrigated'] + irrigate.astype(jnp.int32)
            ship = a[i] == 10
            harvest = a[i] == 11
            n['cargo'] = n['cargo'].at[i].add(-ship.astype(jnp.int32))
            n['delivered'] = n['delivered'].at[i].add(ship.astype(jnp.int32))
            n['delivery_now'] = n['delivery_now'].at[i].set(ship)
            n['harvested'] = n['harvested'].at[i].add(harvest.astype(jnp.int32))
            n['growth'] = n['growth'].at[i].set(jnp.where(harvest, 0, n['growth'][i]))
            bonus = bonus.at[i].add(1.5 * ship.astype(jnp.float32) + harvest.astype(jnp.float32))
            q = jnp.argmax(near[i])
            report = a[i] == 12
            for recipient in range(3):
                j = recipient * 3 + i
                n['message_pos'] = n['message_pos'].at[j].set(jnp.where(report, s['lock_pos'][q], n['message_pos'][j]))
                n['message_level'] = n['message_level'].at[j].set(jnp.where(report, n['level'][q], n['message_level'][j]))
                n['message_time'] = n['message_time'].at[j].set(jnp.where(report, n['t'], n['message_time'][j]))
                n['message_ttl'] = n['message_ttl'].at[j].set(jnp.where(report, 5, n['message_ttl'][j]))
        targets, geometry = self.targets(s)
        target = targets[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        wants = (a >= 1) & (a <= 4) & geometry[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        enters = jnp.all(target[:, None, :] == s['lock_pos'][None, :, :], axis=2)
        exits = jnp.all(s['agent_pos'][:, None, :] == s['lock_pos'][None, :, :], axis=2)
        needed_enter = jnp.where(s['agent_pos'][:, 0, None] < s['lock_pos'][None, :, 0], 2, 0)
        needed_exit = jnp.where(target[:, 0, None] < s['lock_pos'][None, :, 0], 2, 0)
        access = ~jnp.any((enters & (n['level'][None, :] != needed_enter)) | (exits & (n['level'][None, :] != needed_exit)) | ((enters | exits) & n['surge'][None, :]), axis=1)
        berth = jnp.all(target[:, None, :] == s['berth_pos'][None, :, :], axis=2)
        access = access & ~jnp.any(berth & (n['reservation'] >= 0)[None, :] & (n['reservation'][None, :] != jnp.arange(3)[:, None]), axis=1)
        wants = wants & access
        rank = (jnp.arange(3) - s['t']) % 3
        rivals = jnp.all(target[:, None, :] == target[None, :, :], axis=2) & wants[None, :]
        win = wants & ~jnp.any(rivals & (rank[None, :] < rank[:, None]), axis=1)
        n['agent_pos'] = jnp.where(win[:, None], target, s['agent_pos'])
        n['blocked'] = n['blocked'] | (((a >= 1) & (a <= 4)) & ~win)
        caught = jnp.any(jnp.all(s['agent_pos'][:, None, :] == s['lock_pos'][None, :, :], axis=2) & n['surge'][None, :], axis=1)
        pushed = s['agent_pos'] + jnp.array([1, 0], jnp.int32)
        collision = jnp.any(jnp.all(pushed[:, None, :] == n['agent_pos'][None, :, :], axis=2), axis=1)
        drift = caught & ~collision
        n['agent_pos'] = jnp.where(drift[:, None], pushed, n['agent_pos'])
        lost = caught & collision & (n['cargo'] > 0)
        n['cargo'] = n['cargo'] - lost.astype(jnp.int32)
        n['lost'] = n['lost'] + lost.astype(jnp.int32)
        damaged = jnp.array([n['surge'][0], False, n['surge'][1]], jnp.bool_)
        n['moisture'] = jnp.where(damaged, 0, jnp.maximum(n['moisture'] - (n['t'] % 4 == 0).astype(jnp.int32), 0))
        n['growth'] = jnp.where(damaged, 0, jnp.minimum(n['growth'] + ((n['moisture'] > 0) & (n['harvested'] < 2)).astype(jnp.int32), 4))
        n['success'] = jnp.all(n['delivered'] >= 1)
        n['failure'] = ((n['flood'] >= 3) | jnp.any(n['lost'] > 0)) & ~n['success']
        n['terminated'] = n['success'] | n['failure']
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.sync(n)
        reward = bonus + 0.2 * jnp.sum(n['delivery_now']).astype(jnp.float32) - 0.01 * (a != 0).astype(jnp.float32) - jnp.sum(n['surge']).astype(jnp.float32) - n['failure'].astype(jnp.float32)
        return n, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
