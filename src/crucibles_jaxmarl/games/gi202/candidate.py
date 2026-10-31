"""Counterweight Foundry: original unexecuted spatial source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 182
    num_actions = 13
    max_steps = 160

    def beam_cells(self, ends):
        x = jnp.arange(3, 8, dtype=jnp.int32)
        y = ((7 - x) * ends[0] + (x - 3) * ends[1] + 2) // 4
        return jnp.stack((x, y), axis=1)

    def sync(self, s):
        s = dict(s)
        s['beam_pos'] = self.beam_cells(s['ends'])
        s['pad_pos'] = jnp.stack((jnp.array([2, 8], jnp.int32), s['ends']), axis=1)
        endpoint = jnp.stack((jnp.array([3, 7], jnp.int32), s['ends']), axis=1)
        s['weight_pos'] = jnp.where((s['weight_mount'] >= 0)[:, None], endpoint[jnp.maximum(s['weight_mount'], 0)], s['weight_pos'])
        s['weight_pos'] = jnp.where((s['weight_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['weight_holder'], 0)], s['weight_pos'])
        s['cargo_pos'] = jnp.where((s['cargo_status'] == 1)[:, None], s['agent_pos'][jnp.maximum(s['cargo_holder'], 0)], s['cargo_pos'])
        s['cargo_pos'] = jnp.where((s['cargo_status'] == 2)[:, None], s['beam_pos'][2], s['cargo_pos'])
        s['cargo_active'] = s['cargo_status'] < 3
        s['cargo_vis'] = jnp.broadcast_to(s['cargo_active'], (3, 3))
        return s

    def reset(self, key):
        grid = jnp.zeros((9, 11), jnp.int32)
        grid = grid.at[1:8, 3].set(1).at[1:8, 7].set(1)
        grid = grid.at[1, 4:7].set(2).at[7, 4:7].set(3)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'success': jnp.array(False), 'failure': jnp.array(False), 'map': grid,
            'agent_pos': jnp.array([[2, 3], [8, 3], [5, 2]], jnp.int32),
            'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
            'held_kind': jnp.zeros(3, jnp.int32), 'held_id': jnp.full(3, -1, jnp.int32),
            'work': jnp.zeros(3, jnp.int32), 'blocked': jnp.zeros(3, jnp.bool_),
            'ends': jnp.array([3, 3], jnp.int32), 'tension': jnp.array(0, jnp.int32),
            'braced': jnp.array(False), 'beam_pos': jnp.zeros((5, 2), jnp.int32),
            'beam_active': jnp.ones(5, jnp.bool_), 'beam_vis': jnp.ones((3, 5), jnp.bool_),
            'pad_pos': jnp.zeros((2, 2), jnp.int32), 'pad_active': jnp.ones(2, jnp.bool_),
            'pad_vis': jnp.ones((3, 2), jnp.bool_),
            'weight_pos': jnp.array([[3, 3], [7, 3], [1, 5], [9, 5]], jnp.int32),
            'weight_holder': jnp.full(4, -1, jnp.int32),
            'weight_mount': jnp.array([0, 1, -1, -1], jnp.int32),
            'weight_active': jnp.ones(4, jnp.bool_), 'weight_vis': jnp.ones((3, 4), jnp.bool_),
            'cargo_pos': jnp.array([[5, 3], [4, 1], [6, 1]], jnp.int32),
            'cargo_status': jnp.array([2, 0, 0], jnp.int32),
            'cargo_holder': jnp.full(3, -1, jnp.int32), 'cargo_owner': jnp.arange(3, dtype=jnp.int32),
            'cargo_active': jnp.ones(3, jnp.bool_), 'cargo_vis': jnp.ones((3, 3), jnp.bool_),
            'delivered': jnp.zeros(3, jnp.int32), 'shipment': jnp.zeros(3, jnp.bool_),
            'dropped': jnp.zeros(3, jnp.bool_), 'jam': jnp.array(False),
        }
        s = self.sync(s)
        return self.observe(s), s

    def walk_targets(self, s):
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['agent_pos'][:, None, :] + delta[None, :, :]
        inside = jnp.all((p >= 0) & (p <= jnp.array([10, 8], jnp.int32)), axis=2)
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=3), axis=2)
        beam = jnp.any(jnp.all(p[:, :, None, :] == s['beam_pos'][None, None, :, :], axis=3), axis=2)
        return p, inside & ~occupied & ~beam

    def legal_actions(self, s):
        p = s['agent_pos']
        free = s['held_kind'] == 0
        cp = jnp.sum(jnp.abs(p[:, None, :] - s['cargo_pos'][None, :, :]), axis=2) <= 1
        wp = jnp.sum(jnp.abs(p[:, None, :] - s['weight_pos'][None, :, :]), axis=2) <= 1
        atpad = jnp.all(p[:, None, :] == s['pad_pos'][None, :, :], axis=2)
        nearbeam = jnp.min(jnp.sum(jnp.abs(p[:, None, :] - s['beam_pos'][None, :, :]), axis=2), axis=1) <= 1
        _, walk = self.walk_targets(s)
        m = jnp.zeros((3, 13), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(walk)
        m = m.at[:, 5].set(free & jnp.any(cp & (s['cargo_status'] == 0)[None, :], axis=1))
        m = m.at[:, 6].set(free & jnp.any(wp & (s['weight_holder'] < 0)[None, :], axis=1))
        m = m.at[:, 7].set(~free)
        m = m.at[:, 8].set(free & jnp.any(atpad & (s['ends'] > 1)[None, :], axis=1))
        m = m.at[:, 9].set(free & jnp.any(atpad & (s['ends'] < 7)[None, :], axis=1))
        m = m.at[:, 10].set(free & nearbeam)
        m = m.at[:, 11].set((s['held_kind'] == 1) & (jnp.sum(jnp.abs(p - s['beam_pos'][2]), axis=1) <= 1) & ~jnp.any(s['cargo_status'] == 2) & (jnp.abs(s['ends'][0] - s['ends'][1]) <= 1))
        counts = jnp.sum(s['weight_mount'][:, None] == jnp.arange(2)[None, :], axis=0)
        m = m.at[:, 12].set((s['held_kind'] == 2) & jnp.any(atpad & (counts < 2)[None, :], axis=1))
        return jnp.where(s['done'], jnp.arange(13)[None, :] == 0, m)

    def observe(self, s):
        f = jnp.concatenate((
            jnp.stack((s['t'], s['done'], s['terminated'], s['truncated'], s['success'], s['failure'], s['tension'], s['braced'], s['jam'])),
            s['map'].ravel(), s['agent_pos'].ravel(), s['held_kind'], s['held_id'], s['work'], s['blocked'],
            s['beam_pos'].ravel(), s['ends'], s['pad_pos'].ravel(), s['weight_pos'].ravel(),
            s['weight_holder'], s['weight_mount'], s['cargo_pos'].ravel(), s['cargo_status'], s['cargo_holder'],
            s['cargo_owner'], s['delivered'], s['shipment'], s['dropped'])).astype(jnp.float32)
        return jnp.broadcast_to(f, (3, self.observation_size))

    def advance(self, s, actions):
        n = dict(s)
        mask = self.legal_actions(s)
        clipped = jnp.clip(actions, 0, 12)
        a = jnp.where((actions >= 0) & (actions < 13) & mask[jnp.arange(3), clipped], clipped, 0)
        n['t'] = s['t'] + 1
        n['shipment'] = jnp.zeros(3, jnp.bool_)
        n['dropped'] = jnp.zeros(3, jnp.bool_)
        n['jam'] = jnp.array(False)
        n['work'] = s['work'] + (a != 0).astype(jnp.int32)
        n['blocked'] = actions != a
        targets, walk = self.walk_targets(s)
        target = targets[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        wants = (a >= 1) & (a <= 4) & walk[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        rank = (jnp.arange(3) - s['t']) % 3
        rivals = jnp.all(target[:, None, :] == target[None, :, :], axis=2) & wants[None, :]
        win = wants & ~jnp.any(rivals & (rank[None, :] < rank[:, None]), axis=1)
        n['agent_pos'] = jnp.where(win[:, None], target, s['agent_pos'])
        n['blocked'] = n['blocked'] | (wants & ~win)
        n['braced'] = jnp.any(a == 10)
        n = self.sync(n)
        for k in range(3):
            i = (s['t'] + k) % 3
            p = n['agent_pos'][i]
            free = n['held_kind'][i] == 0
            eligc = (n['cargo_status'] == 0) & (jnp.sum(jnp.abs(n['cargo_pos'] - p), axis=1) <= 1)
            c = jnp.argmax(eligc)
            takec = (a[i] == 5) & free & jnp.any(eligc)
            eligw = (n['weight_holder'] < 0) & (jnp.sum(jnp.abs(n['weight_pos'] - p), axis=1) <= 1)
            w = jnp.argmax(eligw)
            takew = (a[i] == 6) & free & jnp.any(eligw)
            n['cargo_status'] = n['cargo_status'].at[c].set(jnp.where(takec, 1, n['cargo_status'][c]))
            n['cargo_holder'] = n['cargo_holder'].at[c].set(jnp.where(takec, i, n['cargo_holder'][c]))
            n['weight_holder'] = n['weight_holder'].at[w].set(jnp.where(takew, i, n['weight_holder'][w]))
            n['weight_mount'] = n['weight_mount'].at[w].set(jnp.where(takew, -1, n['weight_mount'][w]))
            n['held_kind'] = n['held_kind'].at[i].set(jnp.where(takec, 1, jnp.where(takew, 2, n['held_kind'][i])))
            n['held_id'] = n['held_id'].at[i].set(jnp.where(takec, c, jnp.where(takew, w, n['held_id'][i])))
            hid = jnp.maximum(n['held_id'][i], 0)
            cid = jnp.minimum(hid, 2)
            dc = (a[i] == 7) & (n['held_kind'][i] == 1)
            dw = (a[i] == 7) & (n['held_kind'][i] == 2)
            load = (a[i] == 11) & (n['held_kind'][i] == 1) & ~jnp.any(n['cargo_status'] == 2)
            at = jnp.all(n['pad_pos'] == p, axis=1)
            e = jnp.argmax(at)
            count = jnp.sum(n['weight_mount'] == e)
            mount = (a[i] == 12) & (n['held_kind'][i] == 2) & jnp.any(at) & (count < 2)
            n['cargo_status'] = n['cargo_status'].at[cid].set(jnp.where(dc, 0, jnp.where(load, 2, n['cargo_status'][cid])))
            n['cargo_holder'] = n['cargo_holder'].at[cid].set(jnp.where(dc | load, -1, n['cargo_holder'][cid]))
            n['cargo_pos'] = n['cargo_pos'].at[cid].set(jnp.where(dc, p, n['cargo_pos'][cid]))
            n['weight_holder'] = n['weight_holder'].at[hid].set(jnp.where(dw | mount, -1, n['weight_holder'][hid]))
            n['weight_mount'] = n['weight_mount'].at[hid].set(jnp.where(mount, e, n['weight_mount'][hid]))
            n['weight_pos'] = n['weight_pos'].at[hid].set(jnp.where(dw, p, n['weight_pos'][hid]))
            release = dc | dw | load | mount
            n['held_kind'] = n['held_kind'].at[i].set(jnp.where(release, 0, n['held_kind'][i]))
            n['held_id'] = n['held_id'].at[i].set(jnp.where(release, -1, n['held_id'][i]))
        counts = jnp.sum(n['weight_mount'][:, None] == jnp.arange(2)[None, :], axis=0)
        at = jnp.all(s['agent_pos'][:, None, :] == s['pad_pos'][None, :, :], axis=2)
        request = jnp.sum(at * ((a == 9).astype(jnp.int32) - (a == 8).astype(jnp.int32))[:, None], axis=0).astype(jnp.int32)
        support = (counts >= 1) | ~jnp.any(n['cargo_status'] == 2) | n['braced']
        proposed = jnp.clip(s['ends'] + request * support.astype(jnp.int32), 1, 7)
        cells = self.beam_cells(proposed)
        collision = jnp.any(jnp.all(cells[:, None, :] == n['agent_pos'][None, :, :], axis=2))
        n['jam'] = collision & jnp.any(proposed != s['ends'])
        n['ends'] = jnp.where(collision, s['ends'], proposed)
        moved = jnp.any(n['ends'] != s['ends'])
        tilt = jnp.abs(n['ends'][0] - n['ends'][1])
        strain = tilt + jnp.abs(counts[0] - counts[1])
        change = jnp.where(moved & (strain > 0), strain, -1) - 2 * n['braced'].astype(jnp.int32)
        n['tension'] = jnp.clip(s['tension'] + change, 0, 8).astype(jnp.int32)
        n = self.sync(n)
        loaded = n['cargo_status'] == 2
        drop = loaded & (tilt >= 2) & ~n['braced']
        broken = loaded & (n['tension'] >= 8)
        ship = loaded & (tilt == 0) & (n['ends'][0] == 7) & ~broken
        n['cargo_status'] = jnp.where(broken, 4, jnp.where(ship, 3, jnp.where(drop, 0, n['cargo_status']))).astype(jnp.int32)
        n['dropped'] = drop | broken
        n['shipment'] = ship
        n['delivered'] = s['delivered'] + ship.astype(jnp.int32)
        n['success'] = jnp.all(n['cargo_status'] == 3)
        n['failure'] = (jnp.any(n['cargo_status'] == 4) | (n['tension'] >= 8)) & ~n['success']
        n['terminated'] = n['success'] | n['failure']
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.sync(n)
        reward = -0.01 * (a != 0).astype(jnp.float32) - 0.02 * ((a == 8) | (a == 9) | (a == 10)).astype(jnp.float32)
        reward = reward + jnp.sum(ship).astype(jnp.float32) + 0.5 * ship.astype(jnp.float32) - 0.2 * jnp.sum(n['dropped']).astype(jnp.float32) - 2.0 * n['failure'].astype(jnp.float32)
        return n, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), info
