"""Echo Quarry. Original blind-authored, unexecuted source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 210
    num_actions = 11
    max_steps = 150
    obs_keys = ('t', 'done', 'terminated', 'truncated', 'success', 'failure', 'map', 'agent_pos', 'held', 'work', 'blocked', 'ore_pos', 'ore_owner', 'ore_status', 'ore_holder', 'drill_progress', 'screen_pos', 'screen_holder', 'slab_pos', 'stress', 'load', 'wave', 'braced', 'collapsed', 'rubble', 'depot_pos', 'delivered', 'drilled', 'collapse_now', 'shipment', 'injury')

    def sync(self, s):
        s = dict(s)
        s['ore_pos'] = jnp.where((s['ore_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['ore_holder'], 0)], s['ore_pos'])
        s['screen_pos'] = jnp.where((s['screen_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['screen_holder'], 0)], s['screen_pos'])
        s['ore_active'] = s['ore_status'] < 3
        s['ore_vis'] = jnp.broadcast_to(s['ore_active'], (3, 6))
        workers = jnp.all(s['slab_pos'][:, None, :] == s['agent_pos'][None, :, :], axis=2)
        s['load'] = jnp.sum(workers * (1 + ((s['held'] >= 0) & (s['held'] < 6)).astype(jnp.int32))[None, :], axis=1).astype(jnp.int32)
        return s

    def reset(self, key):
        grid = jnp.zeros((9, 11), jnp.int32).at[1:8, 5].set(1)
        grid = grid.at[jnp.array([2, 2, 6, 6, 3, 5]), jnp.array([3, 7, 3, 7, 5, 5])].set(2)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False), 'success': jnp.array(False), 'failure': jnp.array(False), 'map': grid,
            'agent_pos': jnp.array([[2, 2], [3, 3], [8, 6]], jnp.int32), 'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
            'held': jnp.full(3, -1, jnp.int32), 'work': jnp.zeros(3, jnp.int32), 'blocked': jnp.zeros(3, jnp.bool_),
            'ore_pos': jnp.array([[3, 2], [7, 2], [3, 6], [7, 6], [5, 3], [5, 5]], jnp.int32), 'ore_owner': jnp.array([0, 1, 2, 0, 1, 2], jnp.int32), 'ore_status': jnp.zeros(6, jnp.int32), 'ore_holder': jnp.full(6, -1, jnp.int32), 'drill_progress': jnp.zeros(6, jnp.int32), 'ore_active': jnp.ones(6, jnp.bool_), 'ore_vis': jnp.ones((3, 6), jnp.bool_),
            'screen_pos': jnp.array([[2, 3], [8, 3]], jnp.int32), 'screen_holder': jnp.full(2, -1, jnp.int32), 'screen_active': jnp.ones(2, jnp.bool_), 'screen_vis': jnp.ones((3, 2), jnp.bool_),
            'slab_pos': jnp.array([[3, 3], [7, 3], [4, 6]], jnp.int32), 'slab_active': jnp.ones(3, jnp.bool_), 'slab_vis': jnp.ones((3, 3), jnp.bool_), 'stress': jnp.zeros(3, jnp.int32), 'load': jnp.zeros(3, jnp.int32), 'wave': jnp.zeros(3, jnp.int32), 'braced': jnp.zeros(3, jnp.bool_), 'collapsed': jnp.zeros(3, jnp.bool_), 'rubble': jnp.zeros(3, jnp.int32),
            'depot_pos': jnp.array([[1, 4], [9, 4], [5, 8]], jnp.int32), 'depot_active': jnp.ones(3, jnp.bool_), 'depot_vis': jnp.ones((3, 3), jnp.bool_),
            'delivered': jnp.zeros(3, jnp.int32), 'drilled': jnp.zeros(6, jnp.bool_), 'collapse_now': jnp.zeros(3, jnp.bool_), 'shipment': jnp.zeros(3, jnp.int32), 'injury': jnp.zeros(3, jnp.bool_),
        }
        s = self.sync(s)
        return self.observe(s), s

    def targets(self, s):
        p = s['agent_pos'][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        bound = jnp.all((p >= 0) & (p <= jnp.array([10, 8], jnp.int32)), axis=2)
        floor = s['map'][jnp.clip(p[:, :, 1], 0, 8), jnp.clip(p[:, :, 0], 0, 10)] == 0
        worker = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=3), axis=2)
        screen = jnp.any(jnp.all(p[:, :, None, :] == s['screen_pos'][None, None, :, :], axis=3) & (s['screen_holder'] < 0)[None, None, :], axis=2)
        return p, bound & floor & ~worker & ~screen

    def legal_actions(self, s):
        nearore = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['ore_pos'][None, :, :]), axis=2) <= 1
        nearslab = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['slab_pos'][None, :, :]), axis=2) <= 1
        nearscreen = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['screen_pos'][None, :, :]), axis=2) <= 1
        free = s['held'] < 0
        _, walk = self.targets(s)
        h = jnp.clip(s['held'], 0, 5)
        dest = s['depot_pos'][s['ore_owner'][h]]
        m = jnp.zeros((3, 11), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(walk)
        m = m.at[:, 5].set(free & jnp.any(nearore & (s['ore_status'] == 0)[None, :], axis=1))
        m = m.at[:, 6].set(free & jnp.any(nearslab & ~s['collapsed'][None, :], axis=1))
        m = m.at[:, 7].set(free & (jnp.any(nearore & (s['ore_status'] == 1)[None, :], axis=1) | jnp.any(nearscreen & (s['screen_holder'] < 0)[None, :], axis=1)))
        m = m.at[:, 8].set(~free)
        m = m.at[:, 9].set((s['held'] >= 0) & (s['held'] < 6) & (jnp.sum(jnp.abs(s['agent_pos'] - dest), axis=1) <= 1))
        m = m.at[:, 10].set(free & jnp.any(nearslab & (s['rubble'] > 0)[None, :], axis=1))
        return jnp.where(s['done'], jnp.arange(11)[None, :] == 0, m)

    def observe(self, s):
        flat = jnp.concatenate([s[k].ravel() for k in self.obs_keys]).astype(jnp.float32)
        return jnp.broadcast_to(flat, (3, self.observation_size))

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 10)
        a = jnp.where((actions >= 0) & (actions < 11) & self.legal_actions(s)[jnp.arange(3), clipped], clipped, 0)
        n['t'] = s['t'] + 1
        n['work'] = s['work'] + (a != 0).astype(jnp.int32)
        n['blocked'] = actions != a
        n['drilled'] = jnp.zeros(6, jnp.bool_)
        n['braced'] = jnp.zeros(3, jnp.bool_)
        n['shipment'] = jnp.zeros(3, jnp.int32)
        tp, ok = self.targets(s)
        target = tp[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        wants = (a >= 1) & (a <= 4) & ok[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        rank = (jnp.arange(3) - s['t']) % 3
        rivals = jnp.all(target[:, None, :] == target[None, :, :], axis=2) & wants[None, :]
        win = wants & ~jnp.any(rivals & (rank[None, :] < rank[:, None]), axis=1)
        n['agent_pos'] = jnp.where(win[:, None], target, s['agent_pos'])
        n['blocked'] = n['blocked'] | (wants & ~win)
        bonus = jnp.zeros(3, jnp.float32)
        n = self.sync(n)
        for k in range(3):
            i = (s['t'] + k) % 3
            p = n['agent_pos'][i]
            ore = (jnp.sum(jnp.abs(n['ore_pos'] - p), axis=1) <= 1)
            q = jnp.argmax(ore & (n['ore_status'] == 0))
            drill = (a[i] == 5) & jnp.any(ore & (n['ore_status'] == 0))
            n['drill_progress'] = n['drill_progress'].at[q].add(drill.astype(jnp.int32))
            n['drilled'] = n['drilled'].at[q].set(n['drilled'][q] | drill)
            slab = (jnp.sum(jnp.abs(n['slab_pos'] - p), axis=1) <= 1)
            q = jnp.argmax(slab & ~n['collapsed'])
            brace = (a[i] == 6) & jnp.any(slab & ~n['collapsed'])
            n['braced'] = n['braced'].at[q].set(n['braced'][q] | brace)
            loose = ore & (n['ore_status'] == 1)
            scr = (jnp.sum(jnp.abs(n['screen_pos'] - p), axis=1) <= 1) & (n['screen_holder'] < 0)
            q = jnp.argmax(loose)
            w = jnp.argmax(scr)
            takeore = (a[i] == 7) & jnp.any(loose)
            takescreen = (a[i] == 7) & ~jnp.any(loose) & jnp.any(scr)
            n['ore_status'] = n['ore_status'].at[q].set(jnp.where(takeore, 2, n['ore_status'][q]))
            n['ore_holder'] = n['ore_holder'].at[q].set(jnp.where(takeore, i, n['ore_holder'][q]))
            n['screen_holder'] = n['screen_holder'].at[w].set(jnp.where(takescreen, i, n['screen_holder'][w]))
            n['held'] = n['held'].at[i].set(jnp.where(takeore, q, jnp.where(takescreen, w + 6, n['held'][i])))
            q = jnp.clip(n['held'][i], 0, 5)
            w = jnp.clip(n['held'][i] - 6, 0, 1)
            dropore = (a[i] == 8) & (n['held'][i] < 6)
            dropscreen = (a[i] == 8) & (n['held'][i] >= 6)
            ship = a[i] == 9
            owner = n['ore_owner'][q]
            n['ore_status'] = n['ore_status'].at[q].set(jnp.where(ship, 3, jnp.where(dropore, 1, n['ore_status'][q])))
            n['ore_holder'] = n['ore_holder'].at[q].set(jnp.where(ship | dropore, -1, n['ore_holder'][q]))
            n['ore_pos'] = n['ore_pos'].at[q].set(jnp.where(dropore, p, n['ore_pos'][q]))
            n['screen_holder'] = n['screen_holder'].at[w].set(jnp.where(dropscreen, -1, n['screen_holder'][w]))
            n['screen_pos'] = n['screen_pos'].at[w].set(jnp.where(dropscreen, p, n['screen_pos'][w]))
            n['held'] = n['held'].at[i].set(jnp.where((a[i] == 8) | ship, -1, n['held'][i]))
            n['delivered'] = n['delivered'].at[owner].add(ship.astype(jnp.int32))
            n['shipment'] = n['shipment'].at[i].add(ship.astype(jnp.int32))
            bonus = bonus.at[owner].add(ship.astype(jnp.float32)).at[i].add(0.2 * ship.astype(jnp.float32))
            q = jnp.argmax(slab & (n['rubble'] > 0))
            clear = (a[i] == 10) & jnp.any(slab & (n['rubble'] > 0))
            n['rubble'] = n['rubble'].at[q].add(-clear.astype(jnp.int32))
        excavated = (n['ore_status'] == 0) & (n['drill_progress'] >= 2)
        n['ore_status'] = jnp.where(excavated, 1, n['ore_status'])
        n['map'] = n['map'].at[n['ore_pos'][:, 1], n['ore_pos'][:, 0]].set(jnp.where(excavated, 0, n['map'][n['ore_pos'][:, 1], n['ore_pos'][:, 0]]))
        n = self.sync(n)
        hit = (jnp.sum(jnp.abs(n['slab_pos'][:, None, :] - s['ore_pos'][None, :, :]), axis=2) <= 4) & n['drilled'][None, :]
        shields = (jnp.sum(jnp.abs(n['slab_pos'][:, None, :] - n['screen_pos'][None, :, :]), axis=2) <= 1) & (n['screen_holder'] < 0)[None, :]
        n['wave'] = jnp.maximum(2 * jnp.sum(hit, axis=1) - 2 * jnp.sum(shields, axis=1), 0).astype(jnp.int32)
        workers = jnp.all(n['slab_pos'][:, None, :] == n['agent_pos'][None, :, :], axis=2)
        n['load'] = jnp.sum(workers * (1 + ((n['held'] >= 0) & (n['held'] < 6)).astype(jnp.int32))[None, :], axis=1).astype(jnp.int32)
        n['stress'] = jnp.where(n['collapsed'], n['stress'], jnp.clip(s['stress'] + n['wave'] + n['load'] * (n['wave'] > 0).astype(jnp.int32) - 1 - 2 * n['braced'].astype(jnp.int32), 0, 8)).astype(jnp.int32)
        n['collapse_now'] = (n['stress'] >= 8) & ~n['collapsed']
        n['collapsed'] = n['collapsed'] | n['collapse_now']
        n['rubble'] = jnp.where(n['collapse_now'], 2, n['rubble']).astype(jnp.int32)
        n['map'] = n['map'].at[n['slab_pos'][:, 1], n['slab_pos'][:, 0]].set(jnp.where(n['rubble'] > 0, 3, 0))
        n['injury'] = jnp.any(workers & n['collapse_now'][:, None], axis=0)
        lost = jnp.any(jnp.all(n['ore_pos'][:, None, :] == n['slab_pos'][None, :, :], axis=2) & n['collapse_now'][None, :], axis=1) & (n['ore_status'] < 3)
        n['ore_status'] = jnp.where(lost, 4, n['ore_status'])
        n['ore_holder'] = jnp.where(lost, -1, n['ore_holder'])
        lost_held = (n['held'] >= 0) & (n['held'] < 6) & lost[jnp.clip(n['held'], 0, 5)]
        n['held'] = jnp.where(lost_held, -1, n['held'])
        n['success'] = jnp.all(n['ore_status'] == 3)
        n['failure'] = (jnp.any(n['injury']) | jnp.any(lost)) & ~n['success']
        n['terminated'] = n['success'] | n['failure']
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.sync(n)
        reward = bonus + 0.1 * jnp.sum(n['shipment']).astype(jnp.float32) - 0.01 * (a != 0).astype(jnp.float32) - 0.03 * (a == 5).astype(jnp.float32) - 0.5 * jnp.sum(n['collapse_now']).astype(jnp.float32) - 2.0 * n['failure'].astype(jnp.float32)
        return n, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
