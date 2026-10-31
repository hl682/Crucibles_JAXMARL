"""Proofing House: original unexecuted spatial source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 167
    num_actions = 16
    max_steps = 180

    def sync(self, s):
        s = dict(s)
        s['tray_pos'] = jnp.where((s['tray_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['tray_holder'], 0)], s['tray_pos'])
        s['map'] = s['map'].at[2, 5].set(jnp.where(s['door_open'][0], 0, 2))
        return s

    def reset(self, key):
        grid = jnp.zeros((7, 11), jnp.int32)
        grid = grid.at[0, 4:8].set(1).at[1:3, 4].set(1).at[1:3, 7].set(1).at[2, 4:8].set(1)
        grid = grid.at[2, 5].set(2).at[2, 2:4].set(3).at[2, 8].set(4)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'success': jnp.array(False), 'failure': jnp.array(False), 'map': grid,
            'agent_pos': jnp.array([[3, 3], [5, 3], [8, 3]], jnp.int32),
            'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
            'facing': jnp.zeros(3, jnp.int32), 'held': jnp.full(3, -1, jnp.int32),
            'work': jnp.zeros(3, jnp.int32), 'blocked': jnp.zeros(3, jnp.bool_),
            'tray_pos': jnp.array([[5, 1], [3, 2], [2, 2]], jnp.int32),
            'tray_active': jnp.ones(3, jnp.bool_), 'tray_vis': jnp.ones((3, 3), jnp.bool_),
            'tray_holder': jnp.full(3, -1, jnp.int32), 'stage': jnp.array([2, 1, 0], jnp.int32),
            'recipe': jnp.array([0, 1, -1], jnp.int32), 'trace': jnp.zeros(3, jnp.bool_),
            'proof': jnp.array([4, 0, 0], jnp.int32), 'bake': jnp.zeros(3, jnp.int32),
            'batch_owner': jnp.array([0, 1, -1], jnp.int32),
            'board_pos': jnp.array([[2, 2], [3, 2]], jnp.int32),
            'board_active': jnp.ones(2, jnp.bool_), 'board_vis': jnp.ones((3, 2), jnp.bool_),
            'board_trace': jnp.zeros(2, jnp.bool_), 'board_dirty': jnp.zeros(2, jnp.bool_),
            'door_pos': jnp.array([[5, 2]], jnp.int32), 'door_open': jnp.zeros(1, jnp.bool_),
            'door_active': jnp.ones(1, jnp.bool_), 'door_vis': jnp.ones((3, 1), jnp.bool_),
            'vent': jnp.array(False), 'temperature': jnp.array(3, jnp.int32),
            'station_pos': jnp.array([[1, 1], [8, 2], [9, 5], [1, 5]], jnp.int32),
            'station_active': jnp.ones(4, jnp.bool_), 'station_vis': jnp.ones((3, 4), jnp.bool_),
            'pantry': jnp.array([2, 2], jnp.int32), 'wash_water': jnp.array(8, jnp.int32),
            'served': jnp.zeros(2, jnp.int32), 'waste': jnp.zeros(2, jnp.int32),
            'served_now': jnp.zeros(3, jnp.bool_), 'cleaned': jnp.zeros(2, jnp.bool_),
            'spoiled': jnp.zeros(3, jnp.bool_), 'wasted_now': jnp.zeros(3, jnp.bool_),
        }
        s = self.sync(s)
        return self.observe(s), s

    def inside(self, p):
        return (p[..., 1] == 1) & ((p[..., 0] == 5) | (p[..., 0] == 6))

    def walk_targets(self, s):
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['agent_pos'][:, None, :] + delta[None, :, :]
        inside = jnp.all((p >= 0) & (p <= jnp.array([10, 6], jnp.int32)), axis=2)
        tile = s['map'][jnp.clip(p[:, :, 1], 0, 6), jnp.clip(p[:, :, 0], 0, 10)]
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=3), axis=2)
        return p, inside & (tile != 1) & (tile != 2) & ~occupied

    def front(self, s):
        return s['agent_pos'] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[s['facing']]

    def legal_actions(self, s):
        p = s['agent_pos']
        free = s['held'] < 0
        h = jnp.maximum(s['held'], 0)
        near = jnp.sum(jnp.abs(p[:, None, :] - s['tray_pos'][None, :, :]), axis=2) <= 1
        accessible = ~self.inside(s['tray_pos']) | s['door_open'][0]
        loose = (s['tray_holder'] < 0) & (s['stage'] != 3) & accessible
        f = self.front(s)
        fbound = jnp.all((f >= 0) & (f <= jnp.array([10, 6], jnp.int32)), axis=1)
        tile = s['map'][jnp.clip(f[:, 1], 0, 6), jnp.clip(f[:, 0], 0, 10)]
        occupied = jnp.any(jnp.all(f[:, None, :] == s['agent_pos'][None, :, :], axis=2), axis=1)
        traycell = jnp.any(jnp.all(f[:, None, :] == s['tray_pos'][None, :, :], axis=2) & (s['tray_holder'] < 0)[None, :], axis=1)
        boardtray = jnp.all(s['tray_pos'][:, None, :] == s['board_pos'][None, :, :], axis=2)
        kneadable = (s['tray_holder'] < 0) & (s['stage'] == 1) & jnp.any(boardtray, axis=1)
        ds = jnp.sum(jnp.abs(p[:, None, :] - s['station_pos'][None, :, :]), axis=2) <= 1
        doornear = jnp.sum(jnp.abs(p - s['door_pos'][0]), axis=1) <= 1
        dooroccupied = jnp.any(jnp.all(s['agent_pos'] == s['door_pos'][0], axis=1)) | jnp.any(jnp.all(s['tray_pos'] == s['door_pos'][0], axis=1) & (s['tray_holder'] < 0))
        bnear = jnp.sum(jnp.abs(p[:, None, :] - s['board_pos'][None, :, :]), axis=2) <= 1
        onboard = jnp.any(boardtray & (s['tray_holder'] < 0)[:, None], axis=0)
        ovenfree = ~jnp.any(jnp.all(s['tray_pos'] == s['station_pos'][1], axis=1) & (s['tray_holder'] < 0))
        accept = (s['proof'][h] >= jnp.where(s['recipe'][h] == 0, 5, 7)) & (s['proof'][h] <= jnp.where(s['recipe'][h] == 0, 8, 10)) & ((s['recipe'][h] != 0) | ~s['trace'][h])
        _, walk = self.walk_targets(s)
        m = jnp.zeros((3, 16), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(walk)
        m = m.at[:, 5].set(free & jnp.any(near & loose[None, :], axis=1))
        m = m.at[:, 6].set(~free & fbound & (tile != 1) & (tile != 2) & ~occupied & ~traycell)
        m = m.at[:, 7].set(~free & (s['stage'][h] == 0) & ds[:, 0] & (s['pantry'][0] > 0))
        m = m.at[:, 8].set(~free & (s['stage'][h] == 0) & ds[:, 0] & (s['pantry'][1] > 0))
        m = m.at[:, 9].set(free & jnp.any(near & kneadable[None, :], axis=1))
        m = m.at[:, 10].set(doornear & (~s['door_open'][0] | ~dooroccupied))
        m = m.at[:, 11].set(doornear)
        m = m.at[:, 12].set(~free & (s['stage'][h] == 2) & ds[:, 1] & ovenfree)
        m = m.at[:, 13].set(~free & (s['stage'][h] == 4) & ds[:, 2] & accept & (s['served'][jnp.maximum(s['recipe'][h], 0)] < 2))
        m = m.at[:, 14].set(free & (s['wash_water'] > 0) & jnp.any(bnear & ~onboard[None, :] & s['board_dirty'][None, :], axis=1))
        m = m.at[:, 15].set(~free & (s['stage'][h] > 0) & ds[:, 3] & (s['wash_water'] > 0))
        return jnp.where(s['done'], jnp.arange(16)[None, :] == 0, m)

    def observe(self, s):
        f = jnp.concatenate((
            jnp.stack((s['t'], s['done'], s['terminated'], s['truncated'], s['success'], s['failure'], s['vent'], s['temperature'], s['wash_water'])),
            s['map'].ravel(), s['agent_pos'].ravel(), s['facing'], s['held'], s['work'], s['blocked'],
            s['tray_pos'].ravel(), s['tray_holder'], s['stage'], s['recipe'], s['trace'], s['proof'], s['bake'], s['batch_owner'],
            s['board_pos'].ravel(), s['board_trace'], s['board_dirty'], s['door_pos'].ravel(), s['door_open'],
            s['station_pos'].ravel(), s['pantry'], s['served'], s['waste'], s['served_now'], s['cleaned'], s['spoiled'], s['wasted_now'])).astype(jnp.float32)
        return jnp.broadcast_to(f, (3, self.observation_size))

    def advance(self, s, actions):
        n = dict(s)
        mask = self.legal_actions(s)
        clipped = jnp.clip(actions, 0, 15)
        a = jnp.where((actions >= 0) & (actions < 16) & mask[jnp.arange(3), clipped], clipped, 0)
        n['t'] = s['t'] + 1
        n['blocked'] = actions != a
        n['served_now'] = jnp.zeros(3, jnp.bool_)
        n['cleaned'] = jnp.zeros(2, jnp.bool_)
        n['spoiled'] = jnp.zeros(3, jnp.bool_)
        n['wasted_now'] = jnp.zeros(3, jnp.bool_)
        n['work'] = s['work'] + (a != 0).astype(jnp.int32)
        targets, walk = self.walk_targets(s)
        target = targets[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        wants = (a >= 1) & (a <= 4) & walk[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        rank = (jnp.arange(3) - s['t']) % 3
        conflict = jnp.all(target[:, None, :] == target[None, :, :], axis=2) & wants[None, :]
        win = wants & ~jnp.any(conflict & (rank[None, :] < rank[:, None]), axis=1)
        n['agent_pos'] = jnp.where(win[:, None], target, s['agent_pos'])
        n['facing'] = jnp.where((actions >= 1) & (actions <= 4), actions - 1, s['facing']).astype(jnp.int32)
        n['blocked'] = n['blocked'] | (wants & ~win)
        n = self.sync(n)
        bonus = jnp.zeros(3, jnp.float32)
        for k in range(3):
            i = (s['t'] + k) % 3
            p = n['agent_pos'][i]
            h = jnp.maximum(n['held'][i], 0)
            near = jnp.sum(jnp.abs(n['tray_pos'] - p), axis=1) <= 1
            loose = (n['tray_holder'] < 0) & (n['stage'] != 3) & (~self.inside(n['tray_pos']) | n['door_open'][0])
            takeable = near & loose
            q = jnp.argmax(takeable)
            take = (a[i] == 5) & (n['held'][i] < 0) & jnp.any(takeable)
            n['tray_holder'] = n['tray_holder'].at[q].set(jnp.where(take, i, n['tray_holder'][q]))
            n['held'] = n['held'].at[i].set(jnp.where(take, q, n['held'][i]))
            f = self.front(n)[i]
            cellbusy = jnp.any(jnp.all(n['tray_pos'] == f, axis=1) & (n['tray_holder'] < 0)) | jnp.any(jnp.all(n['agent_pos'] == f, axis=1))
            drop = (a[i] == 6) & (n['held'][i] >= 0) & ~cellbusy & (~jnp.all(f == n['door_pos'][0]) | n['door_open'][0])
            n['tray_pos'] = n['tray_pos'].at[h].set(jnp.where(drop, f, n['tray_pos'][h]))
            n['tray_holder'] = n['tray_holder'].at[h].set(jnp.where(drop, -1, n['tray_holder'][h]))
            n['held'] = n['held'].at[i].set(jnp.where(drop, -1, n['held'][i]))
            r = jnp.clip(a[i] - 7, 0, 1)
            fill = ((a[i] == 7) | (a[i] == 8)) & (n['pantry'][r] > 0) & (n['stage'][h] == 0)
            n['pantry'] = n['pantry'].at[r].add(-fill.astype(jnp.int32))
            n['stage'] = n['stage'].at[h].set(jnp.where(fill, 1, n['stage'][h]))
            n['recipe'] = n['recipe'].at[h].set(jnp.where(fill, r, n['recipe'][h]))
            n['batch_owner'] = n['batch_owner'].at[h].set(jnp.where(fill, i, n['batch_owner'][h]))
            bt = jnp.all(n['tray_pos'][:, None, :] == n['board_pos'][None, :, :], axis=2)
            choices = near & (n['tray_holder'] < 0) & (n['stage'] == 1) & jnp.any(bt, axis=1)
            q = jnp.argmax(choices)
            b = jnp.argmax(bt[q])
            knead = (a[i] == 9) & jnp.any(choices)
            trace = n['trace'][q] | n['board_trace'][b] | (n['recipe'][q] == 1)
            n['trace'] = n['trace'].at[q].set(jnp.where(knead, trace, n['trace'][q]))
            n['board_trace'] = n['board_trace'].at[b].set(jnp.where(knead, trace, n['board_trace'][b]))
            n['board_dirty'] = n['board_dirty'].at[b].set(n['board_dirty'][b] | knead)
            n['stage'] = n['stage'].at[q].set(jnp.where(knead, 2, n['stage'][q]))
            occupieddoor = jnp.any(jnp.all(n['agent_pos'] == n['door_pos'][0], axis=1)) | jnp.any(jnp.all(n['tray_pos'] == n['door_pos'][0], axis=1) & (n['tray_holder'] < 0))
            toggle = (a[i] == 10) & (~n['door_open'][0] | ~occupieddoor)
            n['door_open'] = n['door_open'].at[0].set(n['door_open'][0] ^ toggle)
            n['vent'] = n['vent'] ^ (a[i] == 11)
            ovenfree = ~jnp.any(jnp.all(n['tray_pos'] == n['station_pos'][1], axis=1) & (n['tray_holder'] < 0))
            bake = (a[i] == 12) & ovenfree
            n['stage'] = n['stage'].at[h].set(jnp.where(bake, 3, n['stage'][h]))
            n['bake'] = n['bake'].at[h].set(jnp.where(bake, 0, n['bake'][h]))
            n['tray_pos'] = n['tray_pos'].at[h].set(jnp.where(bake, n['station_pos'][1], n['tray_pos'][h]))
            n['tray_holder'] = n['tray_holder'].at[h].set(jnp.where(bake, -1, n['tray_holder'][h]))
            n['held'] = n['held'].at[i].set(jnp.where(bake, -1, n['held'][i]))
            r = jnp.maximum(n['recipe'][h], 0)
            serve = (a[i] == 13) & (n['served'][r] < 2)
            wash = (a[i] == 15) & (n['wash_water'] > 0)
            n['served'] = n['served'].at[r].add(serve.astype(jnp.int32))
            n['waste'] = n['waste'].at[r].add(wash.astype(jnp.int32))
            n['served_now'] = n['served_now'].at[i].set(serve)
            n['wasted_now'] = n['wasted_now'].at[i].set(wash)
            owner = jnp.maximum(n['batch_owner'][h], 0)
            bonus = bonus.at[owner].add(0.3 * serve.astype(jnp.float32)).at[i].add(0.2 * serve.astype(jnp.float32))
            clear = serve | wash
            n['stage'] = n['stage'].at[h].set(jnp.where(clear, 0, n['stage'][h]))
            n['recipe'] = n['recipe'].at[h].set(jnp.where(clear, -1, n['recipe'][h]))
            n['trace'] = n['trace'].at[h].set(jnp.where(clear, False, n['trace'][h]))
            n['proof'] = n['proof'].at[h].set(jnp.where(clear, 0, n['proof'][h]))
            n['bake'] = n['bake'].at[h].set(jnp.where(clear, 0, n['bake'][h]))
            n['batch_owner'] = n['batch_owner'].at[h].set(jnp.where(clear, -1, n['batch_owner'][h]))
            nearboard = jnp.sum(jnp.abs(n['board_pos'] - p), axis=1) <= 1
            on = jnp.any(jnp.all(n['tray_pos'][:, None, :] == n['board_pos'][None, :, :], axis=2) & (n['tray_holder'] < 0)[:, None], axis=0)
            eligible = nearboard & ~on & n['board_dirty']
            b = jnp.argmax(eligible)
            clean = (a[i] == 14) & jnp.any(eligible) & (n['wash_water'] > 0)
            n['board_dirty'] = n['board_dirty'].at[b].set(jnp.where(clean, False, n['board_dirty'][b]))
            n['board_trace'] = n['board_trace'].at[b].set(jnp.where(clean, False, n['board_trace'][b]))
            n['cleaned'] = n['cleaned'].at[b].set(n['cleaned'][b] | clean)
            n['wash_water'] = n['wash_water'] - wash.astype(jnp.int32) - clean.astype(jnp.int32)
        targettemp = jnp.where(n['door_open'][0], 1, jnp.where(n['vent'], 2, 3))
        n['temperature'] = s['temperature'] + jnp.sign(targettemp - s['temperature'])
        sheltered = self.inside(n['tray_pos']) & ~n['door_open'][0]
        gain = jnp.where(sheltered, jnp.maximum(n['temperature'] - 1, 0), (n['t'] % 3 == 0).astype(jnp.int32))
        n['proof'] = jnp.minimum(n['proof'] + (n['stage'] == 2).astype(jnp.int32) * gain, 15)
        onoven = jnp.all(n['tray_pos'] == n['station_pos'][1], axis=1) & (n['tray_holder'] < 0)
        hot = onoven & ((n['stage'] == 3) | (n['stage'] == 4))
        n['bake'] = jnp.minimum(n['bake'] + hot.astype(jnp.int32), 7)
        ready = (n['stage'] == 3) & (n['bake'] >= 3)
        spoiled = ((n['stage'] == 2) & (n['proof'] > 12)) | (hot & (n['bake'] >= 7))
        n['stage'] = jnp.where(spoiled, 5, jnp.where(ready, 4, n['stage'])).astype(jnp.int32)
        n['spoiled'] = spoiled
        available = jnp.sum(((n['recipe'][:, None] == jnp.arange(2)[None, :]) & (n['stage'][:, None] > 0) & (n['stage'][:, None] < 5)).astype(jnp.int32), axis=0)
        n['success'] = jnp.all(n['served'] >= 2)
        n['failure'] = jnp.any(n['served'] + n['pantry'] + available < 2) & ~n['success']
        n['terminated'] = n['success'] | n['failure']
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.sync(n)
        reward = jnp.sum(n['served_now']).astype(jnp.float32) + bonus - 0.01 * (a != 0).astype(jnp.float32) - 0.1 * n['wasted_now'].astype(jnp.float32) - n['failure'].astype(jnp.float32)
        return n, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), info
