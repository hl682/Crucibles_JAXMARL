"""Compost Commons. Original blind-authored, unexecuted source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 234
    num_actions = 19
    max_steps = 150
    obs_keys = ('t', 'done', 'terminated', 'truncated', 'success', 'failure', 'map', 'agent_pos', 'held', 'flask', 'tool_dirty', 'produce', 'origin', 'work', 'blocked', 'cart_pos', 'cart_holder', 'cart_units', 'cart_age', 'cart_moisture', 'screen_pos', 'screen_holder', 'crop_pos', 'crop_owner', 'stage', 'growth', 'moisture', 'stress', 'pest_pos', 'cloud_pos', 'cloud_strength', 'cloud_active', 'wind', 'station_pos', 'organic_stock', 'well_water', 'applied', 'harvested', 'community', 'private_stock', 'turned', 'pest_hits', 'donated_now', 'stored_now')

    def sync(self, s):
        s = dict(s)
        s['cart_pos'] = jnp.where((s['cart_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['cart_holder'], 0)], s['cart_pos'])
        s['screen_pos'] = jnp.where((s['screen_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['screen_holder'], 0)], s['screen_pos'])
        s['cloud_active'] = s['cloud_strength'] > 0
        s['cloud_vis'] = jnp.broadcast_to(s['cloud_active'], (3, 1))
        s['wind'] = jnp.where((s['t'] // 8) % 2 == 0, 1, -1).astype(jnp.int32)
        return s

    def reset(self, key):
        grid = jnp.zeros((9, 11), jnp.int32).at[4, :].set(1).at[4, jnp.array([2, 5, 8])].set(2)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False), 'success': jnp.array(False), 'failure': jnp.array(False), 'map': grid,
            'agent_pos': jnp.array([[2, 1], [5, 1], [8, 7]], jnp.int32), 'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_), 'held': jnp.full(3, -1, jnp.int32), 'flask': jnp.ones(3, jnp.int32), 'tool_dirty': jnp.zeros(3, jnp.bool_), 'produce': jnp.zeros(3, jnp.int32), 'origin': jnp.full(3, -1, jnp.int32), 'work': jnp.zeros(3, jnp.int32), 'blocked': jnp.zeros(3, jnp.bool_),
            'cart_pos': jnp.array([[5, 2]], jnp.int32), 'cart_holder': jnp.full(1, -1, jnp.int32), 'cart_units': jnp.array([3], jnp.int32), 'cart_age': jnp.array([1], jnp.int32), 'cart_moisture': jnp.array([2], jnp.int32), 'cart_active': jnp.ones(1, jnp.bool_), 'cart_vis': jnp.ones((3, 1), jnp.bool_),
            'screen_pos': jnp.array([[3, 2], [7, 6]], jnp.int32), 'screen_holder': jnp.full(2, -1, jnp.int32), 'screen_active': jnp.ones(2, jnp.bool_), 'screen_vis': jnp.ones((3, 2), jnp.bool_),
            'crop_pos': jnp.array([[2, 2], [5, 2], [8, 2], [2, 6], [5, 6], [8, 6]], jnp.int32), 'crop_owner': jnp.array([0, 1, 2, 0, 1, 2], jnp.int32), 'crop_active': jnp.ones(6, jnp.bool_), 'crop_vis': jnp.ones((3, 6), jnp.bool_), 'stage': jnp.zeros(6, jnp.int32), 'growth': jnp.array([2, 1, 2, 1, 2, 1], jnp.int32), 'moisture': jnp.full(6, 3, jnp.int32), 'stress': jnp.zeros(6, jnp.int32),
            'pest_pos': jnp.array([[2, 3], [8, 3], [5, 7]], jnp.int32), 'pest_active': jnp.ones(3, jnp.bool_), 'pest_vis': jnp.ones((3, 3), jnp.bool_),
            'cloud_pos': jnp.array([[5, 2]], jnp.int32), 'cloud_strength': jnp.zeros(1, jnp.int32), 'cloud_active': jnp.zeros(1, jnp.bool_), 'cloud_vis': jnp.zeros((3, 1), jnp.bool_), 'wind': jnp.array(1, jnp.int32),
            'station_pos': jnp.array([[1, 1], [5, 4], [1, 7], [5, 7], [9, 7], [5, 0]], jnp.int32), 'station_active': jnp.ones(6, jnp.bool_), 'station_vis': jnp.ones((3, 6), jnp.bool_),
            'organic_stock': jnp.array(3, jnp.int32), 'well_water': jnp.array(18, jnp.int32), 'applied': jnp.array(0, jnp.int32), 'harvested': jnp.zeros(3, jnp.int32), 'community': jnp.zeros(3, jnp.int32), 'private_stock': jnp.zeros(3, jnp.int32), 'turned': jnp.array(False), 'pest_hits': jnp.zeros(6, jnp.int32), 'donated_now': jnp.zeros(3, jnp.int32), 'stored_now': jnp.zeros(3, jnp.int32),
        }
        return self.observe(s), s

    def targets(self, s):
        p = s['agent_pos'][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        bound = jnp.all((p >= 0) & (p <= jnp.array([10, 8], jnp.int32)), axis=2)
        floor = s['map'][jnp.clip(p[:, :, 1], 0, 8), jnp.clip(p[:, :, 0], 0, 10)] != 1
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=3), axis=2)
        return p, bound & floor & ~occupied

    def legal_actions(self, s):
        p = s['agent_pos']
        free = (s['held'] < 0) & (s['produce'] == 0)
        cart = jnp.sum(jnp.abs(p - s['cart_pos'][0]), axis=1) <= 1
        screen = jnp.sum(jnp.abs(p[:, None, :] - s['screen_pos'][None, :, :]), axis=2) <= 1
        crop = jnp.sum(jnp.abs(p[:, None, :] - s['crop_pos'][None, :, :]), axis=2) <= 1
        station = jnp.sum(jnp.abs(p[:, None, :] - s['station_pos'][None, :, :]), axis=2) <= 1
        _, walk = self.targets(s)
        m = jnp.zeros((3, 19), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(walk)
        m = m.at[:, 5].set(free & cart & (s['cart_holder'][0] < 0))
        m = m.at[:, 6].set(s['held'] >= 3)
        m = m.at[:, 7].set(free & jnp.any(screen & (s['screen_holder'] < 0)[None, :], axis=1))
        m = m.at[:, 8].set(cart & (s['flask'] > 0) & (s['cart_units'][0] > 0) & (s['cart_moisture'][0] < 6))
        m = m.at[:, 9].set((s['flask'] > 0) & jnp.any(crop & (s['stage'] < 2)[None, :] & (s['moisture'] < 4)[None, :], axis=1))
        m = m.at[:, 10].set(cart & station[:, 0] & (s['organic_stock'] > 0) & (s['cart_units'][0] < 3))
        m = m.at[:, 11].set(station[:, 1] & (s['flask'] < 2) & (s['well_water'] > 0))
        m = m.at[:, 12].set(free & cart & (s['cart_units'][0] > 0) & ~s['tool_dirty'])
        m = m.at[:, 13].set(((s['held'] == 1) | (s['held'] == 2)) & jnp.any(crop & (s['stage'] == 0)[None, :], axis=1))
        m = m.at[:, 14].set(station[:, 1] & s['tool_dirty'] & (s['flask'] > 0))
        m = m.at[:, 15].set(free & jnp.any(crop & (s['stage'] == 1)[None, :], axis=1))
        m = m.at[:, 16].set((s['produce'] > 0) & (jnp.sum(jnp.abs(p - s['station_pos'][2:5]), axis=1) <= 1))
        m = m.at[:, 17].set((s['produce'] > 0) & station[:, 5])
        m = m.at[:, 18].set(cart & (s['cart_units'][0] > 0))
        return jnp.where(s['done'], jnp.arange(19)[None, :] == 0, m)

    def observe(self, s):
        flat = jnp.concatenate([s[k].ravel() for k in self.obs_keys]).astype(jnp.float32)
        return jnp.broadcast_to(flat, (3, self.observation_size))

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 18)
        a = jnp.where((actions >= 0) & (actions < 19) & self.legal_actions(s)[jnp.arange(3), clipped], clipped, 0)
        n['t'] = s['t'] + 1
        n['work'] = s['work'] + (a != 0).astype(jnp.int32)
        n['blocked'] = actions != a
        n['turned'] = jnp.array(False)
        n['donated_now'] = jnp.zeros(3, jnp.int32)
        n['stored_now'] = jnp.zeros(3, jnp.int32)
        tp, ok = self.targets(s)
        target = tp[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        wants = (a >= 1) & (a <= 4) & ok[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        rank = (jnp.arange(3) - s['t']) % 3
        rivals = jnp.all(target[:, None, :] == target[None, :, :], axis=2) & wants[None, :]
        win = wants & ~jnp.any(rivals & (rank[None, :] < rank[:, None]), axis=1)
        n['agent_pos'] = jnp.where(win[:, None], target, s['agent_pos'])
        n['blocked'] = n['blocked'] | (wants & ~win)
        n = self.sync(n)
        bonus = jnp.zeros(3, jnp.float32)
        for k in range(3):
            i = (s['t'] + k) % 3
            p = n['agent_pos'][i]
            cartnear = jnp.sum(jnp.abs(p - n['cart_pos'][0])) <= 1
            grab = (a[i] == 5) & (n['cart_holder'][0] < 0) & cartnear
            n['cart_holder'] = n['cart_holder'].at[0].set(jnp.where(grab, i, n['cart_holder'][0]))
            n['held'] = n['held'].at[i].set(jnp.where(grab, 3, n['held'][i]))
            scr = (jnp.sum(jnp.abs(n['screen_pos'] - p), axis=1) <= 1) & (n['screen_holder'] < 0)
            q = jnp.argmax(scr)
            pickup = (a[i] == 7) & jnp.any(scr)
            n['screen_holder'] = n['screen_holder'].at[q].set(jnp.where(pickup, i, n['screen_holder'][q]))
            n['held'] = n['held'].at[i].set(jnp.where(pickup, q + 4, n['held'][i]))
            releasecart = (a[i] == 6) & (n['held'][i] == 3)
            releasescreen = (a[i] == 6) & (n['held'][i] >= 4)
            q = jnp.clip(n['held'][i] - 4, 0, 1)
            n['cart_holder'] = n['cart_holder'].at[0].set(jnp.where(releasecart, -1, n['cart_holder'][0]))
            n['cart_pos'] = n['cart_pos'].at[0].set(jnp.where(releasecart, p, n['cart_pos'][0]))
            n['screen_holder'] = n['screen_holder'].at[q].set(jnp.where(releasescreen, -1, n['screen_holder'][q]))
            n['screen_pos'] = n['screen_pos'].at[q].set(jnp.where(releasescreen, p, n['screen_pos'][q]))
            n['held'] = n['held'].at[i].set(jnp.where(releasecart | releasescreen, -1, n['held'][i]))
            waterpile = (a[i] == 8) & cartnear & (n['cart_units'][0] > 0) & (n['cart_moisture'][0] < 6)
            n['cart_moisture'] = n['cart_moisture'].at[0].set(jnp.where(waterpile, jnp.minimum(n['cart_moisture'][0] + 2, 6), n['cart_moisture'][0]))
            cn = jnp.sum(jnp.abs(n['crop_pos'] - p), axis=1) <= 1
            thirsty = cn & (n['stage'] < 2) & (n['moisture'] < 4)
            q = jnp.argmax(thirsty)
            watercrop = (a[i] == 9) & jnp.any(thirsty)
            n['moisture'] = n['moisture'].at[q].set(jnp.where(watercrop, jnp.minimum(n['moisture'][q] + 2, 4), n['moisture'][q]))
            raw = (a[i] == 10) & cartnear & (n['organic_stock'] > 0) & (n['cart_units'][0] < 3)
            n['cart_units'] = n['cart_units'].at[0].add(raw.astype(jnp.int32))
            n['cart_age'] = n['cart_age'].at[0].set(jnp.where(raw, 0, n['cart_age'][0]))
            n['cart_moisture'] = n['cart_moisture'].at[0].set(jnp.where(raw, jnp.maximum(n['cart_moisture'][0], 1), n['cart_moisture'][0]))
            n['organic_stock'] = n['organic_stock'] - raw.astype(jnp.int32)
            fill = (a[i] == 11) & (n['well_water'] > 0) & (n['flask'][i] < 2)
            n['well_water'] = n['well_water'] - fill.astype(jnp.int32)
            scoop = (a[i] == 12) & cartnear & (n['cart_units'][0] > 0)
            quality = jnp.where(n['cart_age'][0] >= 6, 2, 1)
            n['held'] = n['held'].at[i].set(jnp.where(scoop, quality, n['held'][i]))
            n['tool_dirty'] = n['tool_dirty'].at[i].set(n['tool_dirty'][i] | scoop)
            n['cart_units'] = n['cart_units'].at[0].add(-scoop.astype(jnp.int32))
            crop = cn & (n['stage'] == 0)
            q = jnp.argmax(crop)
            apply = (a[i] == 13) & jnp.any(crop)
            immature = n['held'][i] == 1
            n['growth'] = n['growth'].at[q].set(jnp.where(apply, jnp.minimum(n['growth'][q] + jnp.where(immature, 2, 3), 8), n['growth'][q]))
            n['stress'] = n['stress'].at[q].set(jnp.where(apply & immature, jnp.minimum(n['stress'][q] + 2, 4), n['stress'][q]))
            n['moisture'] = n['moisture'].at[q].set(jnp.where(apply & ~immature, jnp.minimum(n['moisture'][q] + 1, 4), n['moisture'][q]))
            n['held'] = n['held'].at[i].set(jnp.where(apply, -1, n['held'][i]))
            n['applied'] = n['applied'] + apply.astype(jnp.int32)
            clean = a[i] == 14
            n['tool_dirty'] = n['tool_dirty'].at[i].set(jnp.where(clean, False, n['tool_dirty'][i]))
            n['flask'] = n['flask'].at[i].add(fill.astype(jnp.int32) - waterpile.astype(jnp.int32) - watercrop.astype(jnp.int32) - clean.astype(jnp.int32))
            ripe = cn & (n['stage'] == 1)
            q = jnp.argmax(ripe)
            harvest = (a[i] == 15) & jnp.any(ripe)
            owner = n['crop_owner'][q]
            n['stage'] = n['stage'].at[q].set(jnp.where(harvest, 2, n['stage'][q]))
            n['produce'] = n['produce'].at[i].set(jnp.where(harvest, 2, n['produce'][i]))
            n['origin'] = n['origin'].at[i].set(jnp.where(harvest, owner, n['origin'][i]))
            n['harvested'] = n['harvested'].at[owner].add(2 * harvest.astype(jnp.int32))
            qty = n['produce'][i]
            owner = jnp.maximum(n['origin'][i], 0)
            private = a[i] == 16
            donate = a[i] == 17
            n['private_stock'] = n['private_stock'].at[owner].add(jnp.where(private, qty, 0))
            n['community'] = n['community'].at[owner].add(jnp.where(donate, qty, 0))
            n['stored_now'] = n['stored_now'].at[i].set(jnp.where(private, qty, 0))
            n['donated_now'] = n['donated_now'].at[i].set(jnp.where(donate, qty, 0))
            bonus = bonus.at[i].add(0.5 * qty.astype(jnp.float32) * private.astype(jnp.float32)).at[owner].add(0.2 * qty.astype(jnp.float32) * private.astype(jnp.float32))
            n['produce'] = n['produce'].at[i].set(jnp.where(private | donate, 0, qty))
            n['origin'] = n['origin'].at[i].set(jnp.where(private | donate, -1, n['origin'][i]))
            turn = (a[i] == 18) & cartnear & (n['cart_units'][0] > 0)
            n['turned'] = n['turned'] | turn
            n['cart_age'] = n['cart_age'].at[0].set(jnp.minimum(n['cart_age'][0] + 2 * turn.astype(jnp.int32), 8))
        n = self.sync(n)
        moist = (n['cart_moisture'] >= 2) & (n['cart_moisture'] <= 4) & (n['cart_units'] > 0)
        n['cart_age'] = jnp.where(n['cart_units'] > 0, jnp.minimum(n['cart_age'] + moist.astype(jnp.int32), 8), 0)
        n['cart_moisture'] = jnp.maximum(n['cart_moisture'] - ((n['t'] % 4 == 0) & (n['cart_units'] > 0)).astype(jnp.int32), 0)
        emission = (n['cart_units'][0] > 0) & (n['turned'] | ((n['t'] % 3 == 0) & ((n['cart_age'][0] < 6) | (n['cart_moisture'][0] > 4))))
        drift = jnp.array([n['wind'], 0], jnp.int32) * (n['t'] % 2 == 0).astype(jnp.int32)
        movedcloud = jnp.clip(s['cloud_pos'][0] + drift, jnp.array([0, 0], jnp.int32), jnp.array([10, 8], jnp.int32))
        n['cloud_pos'] = n['cloud_pos'].at[0].set(jnp.where(emission, n['cart_pos'][0], movedcloud))
        n['cloud_strength'] = n['cloud_strength'].at[0].set(jnp.where(emission, jnp.where(n['turned'], 5, 3), jnp.maximum(s['cloud_strength'][0] - 1, 0)))
        for b in range(3):
            p = n['pest_pos'][b]
            d = jnp.sum(jnp.abs(n['crop_pos'] - p), axis=1)
            q = jnp.argmin(jnp.where(n['stage'] < 2, d, 999))
            goal = jnp.where(n['cloud_strength'][0] > 0, n['cloud_pos'][0], n['crop_pos'][q])
            delta = jnp.array([[1, 0], [-1, 0], [0, -1], [0, 1], [0, 0]], jnp.int32).at[0, 0].set(n['wind']).at[1, 0].set(-n['wind'])
            options = p + delta
            bound = jnp.all((options >= 0) & (options <= jnp.array([10, 8], jnp.int32)), axis=1)
            floor = n['map'][jnp.clip(options[:, 1], 0, 8), jnp.clip(options[:, 0], 0, 10)] != 1
            screened = jnp.any((jnp.sum(jnp.abs(options[:, None, :] - n['screen_pos'][None, :, :]), axis=2) <= 1) & (n['screen_holder'] < 0)[None, :], axis=1)
            score = jnp.where(bound & floor & ~screened, jnp.sum(jnp.abs(options - goal), axis=1), 999)
            direction = jnp.argmin(score)
            n['pest_pos'] = n['pest_pos'].at[b].set(jnp.where(jnp.min(score) < 999, options[direction], p))
        protected = jnp.any((jnp.sum(jnp.abs(n['crop_pos'][:, None, :] - n['screen_pos'][None, :, :]), axis=2) <= 1) & (n['screen_holder'] < 0)[None, :], axis=1)
        n['pest_hits'] = jnp.sum(jnp.all(n['crop_pos'][:, None, :] == n['pest_pos'][None, :, :], axis=2), axis=1).astype(jnp.int32) * (~protected & (n['stage'] < 2)).astype(jnp.int32)
        live = n['stage'] < 2
        n['moisture'] = jnp.maximum(n['moisture'] - n['pest_hits'] - (live & (n['t'] % 4 == 0)).astype(jnp.int32), 0)
        recovery = live & (n['moisture'] > 0) & (n['t'] % 4 == 0)
        n['stress'] = jnp.clip(n['stress'] + n['pest_hits'] + (live & (n['moisture'] == 0)).astype(jnp.int32) - recovery.astype(jnp.int32), 0, 4)
        n['growth'] = jnp.minimum(n['growth'] + ((n['stage'] == 0) & (n['moisture'] > 0) & (n['t'] % 2 == 0)).astype(jnp.int32), 8)
        dead = live & (n['stress'] >= 4)
        n['stage'] = jnp.where(dead, 3, jnp.where((n['stage'] == 0) & (n['growth'] >= 6), 1, n['stage'])).astype(jnp.int32)
        n['success'] = jnp.sum(n['community']) >= 6
        potential = 2 * jnp.sum(n['stage'] < 2) + jnp.sum(n['produce']) + jnp.sum(n['community'])
        n['failure'] = (potential < 6) & ~n['success']
        n['terminated'] = n['success'] | n['failure']
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.sync(n)
        reward = bonus + 0.5 * jnp.sum(n['donated_now']).astype(jnp.float32) - 0.01 * (a != 0).astype(jnp.float32) - n['failure'].astype(jnp.float32)
        return n, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
