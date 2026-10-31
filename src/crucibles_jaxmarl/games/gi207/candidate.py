"""Steam Basket Kitchen. Original blind-authored, unexecuted source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 172
    num_actions = 17
    max_steps = 180
    obs_keys = ('t', 'done', 'terminated', 'truncated', 'success', 'failure', 'map', 'agent_pos', 'held', 'flask', 'work', 'blocked', 'basket_pos', 'basket_holder', 'tier_of', 'stage', 'filling', 'prep', 'cook', 'wet', 'batch_owner', 'tool_pos', 'tool_holder', 'tier_pos', 'tier_open', 'station_pos', 'pressure', 'water', 'fuel', 'burner', 'danger', 'pantry', 'wrappers', 'well_water', 'served', 'waste', 'water_lost', 'served_now', 'vented', 'spill', 'spoiled')

    def sync(self, s):
        s = dict(s)
        s['basket_pos'] = jnp.where((s['tier_of'] >= 0)[:, None], s['tier_pos'][jnp.maximum(s['tier_of'], 0)], s['basket_pos'])
        s['basket_pos'] = jnp.where((s['basket_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['basket_holder'], 0)], s['basket_pos'])
        s['tool_pos'] = jnp.where((s['tool_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['tool_holder'], 0)], s['tool_pos'])
        return s

    def reset(self, key):
        grid = jnp.zeros((7, 11), jnp.int32).at[3, 3:5].set(1)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False), 'success': jnp.array(False), 'failure': jnp.array(False), 'map': grid,
            'agent_pos': jnp.array([[5, 2], [5, 4], [2, 3]], jnp.int32), 'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_), 'held': jnp.full(3, -1, jnp.int32), 'flask': jnp.ones(3, jnp.int32), 'work': jnp.zeros(3, jnp.int32), 'blocked': jnp.zeros(3, jnp.bool_),
            'basket_pos': jnp.array([[6, 2], [6, 4], [2, 2]], jnp.int32), 'basket_active': jnp.ones(3, jnp.bool_), 'basket_vis': jnp.ones((3, 3), jnp.bool_), 'basket_holder': jnp.full(3, -1, jnp.int32), 'tier_of': jnp.array([0, 1, -1], jnp.int32), 'stage': jnp.array([2, 2, 1], jnp.int32), 'filling': jnp.array([0, 1, 0], jnp.int32), 'prep': jnp.array([1, 2, 0], jnp.int32), 'cook': jnp.array([1, 1, 0], jnp.int32), 'wet': jnp.zeros(3, jnp.int32), 'batch_owner': jnp.arange(3, dtype=jnp.int32),
            'tool_pos': jnp.array([[2, 3]], jnp.int32), 'tool_holder': jnp.full(1, -1, jnp.int32), 'tool_active': jnp.ones(1, jnp.bool_), 'tool_vis': jnp.ones((3, 1), jnp.bool_),
            'tier_pos': jnp.array([[6, 2], [6, 4]], jnp.int32), 'tier_open': jnp.zeros(2, jnp.bool_), 'tier_active': jnp.ones(2, jnp.bool_), 'tier_vis': jnp.ones((3, 2), jnp.bool_),
            'station_pos': jnp.array([[1, 1], [2, 2], [6, 3], [1, 5], [9, 3]], jnp.int32), 'station_active': jnp.ones(5, jnp.bool_), 'station_vis': jnp.ones((3, 5), jnp.bool_),
            'pressure': jnp.array(3, jnp.int32), 'water': jnp.array(3, jnp.int32), 'fuel': jnp.array(24, jnp.int32), 'burner': jnp.array(True), 'danger': jnp.array(0, jnp.int32), 'pantry': jnp.array([1, 2], jnp.int32), 'wrappers': jnp.array(4, jnp.int32), 'well_water': jnp.array(12, jnp.int32), 'served': jnp.zeros(2, jnp.int32), 'waste': jnp.zeros(2, jnp.int32), 'water_lost': jnp.array(0, jnp.int32), 'served_now': jnp.zeros(3, jnp.bool_), 'vented': jnp.array(False), 'spill': jnp.array(False), 'spoiled': jnp.zeros(3, jnp.bool_),
        }
        return self.observe(s), s

    def targets(self, s):
        p = s['agent_pos'][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        inside = jnp.all((p >= 0) & (p <= jnp.array([10, 6], jnp.int32)), axis=2)
        floor = s['map'][jnp.clip(p[:, :, 1], 0, 6), jnp.clip(p[:, :, 0], 0, 10)] == 0
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=3), axis=2)
        return p, inside & floor & ~occupied

    def legal_actions(self, s):
        p = s['agent_pos']
        h = jnp.clip(s['held'], 0, 2)
        basket = (s['held'] >= 0) & (s['held'] < 3)
        free = s['held'] < 0
        near = jnp.sum(jnp.abs(p[:, None, :] - s['basket_pos'][None, :, :]), axis=2) <= 1
        access = (s['tier_of'] < 0) | s['tier_open'][jnp.maximum(s['tier_of'], 0)]
        toolnear = jnp.sum(jnp.abs(p - s['tool_pos'][0]), axis=1) <= 1
        st = jnp.sum(jnp.abs(p[:, None, :] - s['station_pos'][None, :, :]), axis=2) <= 1
        tn = jnp.sum(jnp.abs(p[:, None, :] - s['tier_pos'][None, :, :]), axis=2) <= 1
        used = jnp.any(s['tier_of'][:, None] == jnp.arange(2)[None, :], axis=0)
        oncounter = jnp.all(s['basket_pos'] == s['station_pos'][1], axis=1) & (s['basket_holder'] < 0) & (s['stage'] == 1)
        occupied = jnp.any(jnp.all(p[:, None, :] == s['basket_pos'][None, :, :], axis=2) & (s['basket_holder'] < 0)[None, :], axis=1)
        _, walk = self.targets(s)
        m = jnp.zeros((3, 17), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(walk)
        m = m.at[:, 5].set(free & (jnp.any(near & access[None, :] & (s['basket_holder'] < 0)[None, :], axis=1) | (toolnear & (s['tool_holder'][0] < 0))))
        m = m.at[:, 6].set((basket & ~occupied) | (s['held'] == 3))
        m = m.at[:, 7].set(basket & (s['stage'][h] == 0) & st[:, 0] & (s['pantry'][0] > 0))
        m = m.at[:, 8].set(basket & (s['stage'][h] == 0) & st[:, 0] & (s['pantry'][1] > 0))
        m = m.at[:, 9].set((s['held'] == 3) & st[:, 1] & jnp.any(oncounter) & (s['wrappers'] > 0))
        m = m.at[:, 10].set(jnp.any(tn, axis=1))
        m = m.at[:, 11].set(st[:, 2] & (s['burner'] | (s['fuel'] > 0)))
        m = m.at[:, 12].set(st[:, 2] & (s['flask'] > 0))
        m = m.at[:, 13].set(basket & (s['stage'][h] == 2) & jnp.any(tn & s['tier_open'][None, :] & ~used[None, :], axis=1))
        r = jnp.maximum(s['filling'][h], 0)
        m = m.at[:, 14].set(basket & (s['stage'][h] == 3) & st[:, 4] & (s['served'][r] < 2))
        m = m.at[:, 15].set(basket & (s['stage'][h] != 0) & st[:, 3] & (s['flask'] > 0))
        m = m.at[:, 16].set(st[:, 3] & (s['flask'] < 2) & (s['well_water'] > 0))
        return jnp.where(s['done'], jnp.arange(17)[None, :] == 0, m)

    def observe(self, s):
        f = jnp.concatenate([s[k].ravel() for k in self.obs_keys]).astype(jnp.float32)
        return jnp.broadcast_to(f, (3, self.observation_size))

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 16)
        a = jnp.where((actions >= 0) & (actions < 17) & self.legal_actions(s)[jnp.arange(3), clipped], clipped, 0)
        n['t'] = s['t'] + 1
        n['work'] = s['work'] + (a != 0).astype(jnp.int32)
        n['blocked'] = actions != a
        n['served_now'] = jnp.zeros(3, jnp.bool_)
        n['spill'] = jnp.array(False)
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
            bd = jnp.sum(jnp.abs(n['basket_pos'] - p), axis=1)
            access = (n['tier_of'] < 0) | n['tier_open'][jnp.maximum(n['tier_of'], 0)]
            eligible = (bd <= 1) & (n['basket_holder'] < 0) & access
            td = jnp.sum(jnp.abs(n['tool_pos'][0] - p))
            score = jnp.concatenate((jnp.where(eligible, bd * 10 + jnp.arange(3), 999), jnp.where((td <= 1) & (n['tool_holder'][0] < 0), td * 10 + 3, 999)[None]))
            obj = jnp.argmin(score)
            take = (a[i] == 5) & (jnp.min(score) < 999)
            h = jnp.minimum(obj, 2)
            n['basket_holder'] = n['basket_holder'].at[h].set(jnp.where(take & (obj < 3), i, n['basket_holder'][h]))
            n['tier_of'] = n['tier_of'].at[h].set(jnp.where(take & (obj < 3), -1, n['tier_of'][h]))
            n['tool_holder'] = n['tool_holder'].at[0].set(jnp.where(take & (obj == 3), i, n['tool_holder'][0]))
            n['held'] = n['held'].at[i].set(jnp.where(take, obj, n['held'][i]))
            h = jnp.clip(n['held'][i], 0, 2)
            db = (a[i] == 6) & (n['held'][i] < 3) & ~jnp.any(jnp.all(n['basket_pos'] == p, axis=1) & (n['basket_holder'] < 0))
            dt = (a[i] == 6) & (n['held'][i] == 3)
            n['basket_holder'] = n['basket_holder'].at[h].set(jnp.where(db, -1, n['basket_holder'][h]))
            n['basket_pos'] = n['basket_pos'].at[h].set(jnp.where(db, p, n['basket_pos'][h]))
            n['tool_holder'] = n['tool_holder'].at[0].set(jnp.where(dt, -1, n['tool_holder'][0]))
            n['tool_pos'] = n['tool_pos'].at[0].set(jnp.where(dt, p, n['tool_pos'][0]))
            n['held'] = n['held'].at[i].set(jnp.where(db | dt, -1, n['held'][i]))
            r = jnp.clip(a[i] - 7, 0, 1)
            fill = ((a[i] == 7) | (a[i] == 8)) & (n['pantry'][r] > 0)
            n['pantry'] = n['pantry'].at[r].add(-fill.astype(jnp.int32))
            n['stage'] = n['stage'].at[h].set(jnp.where(fill, 1, n['stage'][h]))
            n['filling'] = n['filling'].at[h].set(jnp.where(fill, r, n['filling'][h]))
            n['batch_owner'] = n['batch_owner'].at[h].set(jnp.where(fill, i, n['batch_owner'][h]))
            raw = jnp.all(n['basket_pos'] == n['station_pos'][1], axis=1) & (n['basket_holder'] < 0) & (n['stage'] == 1)
            q = jnp.argmax(raw)
            roll = (a[i] == 9) & jnp.any(raw) & (n['wrappers'] > 0)
            n['prep'] = n['prep'].at[q].add(roll.astype(jnp.int32))
            wrapped = roll & (n['prep'][q] >= 1 + n['filling'][q])
            n['stage'] = n['stage'].at[q].set(jnp.where(wrapped, 2, n['stage'][q]))
            n['cook'] = n['cook'].at[q].set(jnp.where(wrapped, 0, n['cook'][q]))
            n['wrappers'] = n['wrappers'] - wrapped.astype(jnp.int32)
            near = jnp.sum(jnp.abs(n['tier_pos'] - p), axis=1) <= 1
            tier = jnp.argmax(near)
            n['tier_open'] = n['tier_open'].at[tier].set(n['tier_open'][tier] ^ (a[i] == 10))
            n['burner'] = n['burner'] ^ (a[i] == 11)
            pour = jnp.where(a[i] == 12, n['flask'][i], 0)
            excess = jnp.maximum(n['water'] + pour - 6, 0)
            n['water'] = jnp.minimum(n['water'] + pour, 6)
            n['flask'] = n['flask'].at[i].add(-pour)
            n['water_lost'] = n['water_lost'] + excess
            n['spill'] = n['spill'] | (excess > 0)
            available = near & n['tier_open'] & ~jnp.any(n['tier_of'][:, None] == jnp.arange(2)[None, :], axis=0)
            tier = jnp.argmax(available)
            load = (a[i] == 13) & jnp.any(available)
            n['tier_of'] = n['tier_of'].at[h].set(jnp.where(load, tier, n['tier_of'][h]))
            n['basket_holder'] = n['basket_holder'].at[h].set(jnp.where(load, -1, n['basket_holder'][h]))
            n['held'] = n['held'].at[i].set(jnp.where(load, -1, n['held'][i]))
            r = jnp.maximum(n['filling'][h], 0)
            serve = (a[i] == 14) & (n['served'][r] < 2)
            wash = a[i] == 15
            discard = wash & (n['filling'][h] >= 0)
            n['served'] = n['served'].at[r].add(serve.astype(jnp.int32))
            n['waste'] = n['waste'].at[r].add(discard.astype(jnp.int32))
            n['served_now'] = n['served_now'].at[i].set(serve)
            bonus = bonus.at[jnp.maximum(n['batch_owner'][h], 0)].add(0.2 * serve.astype(jnp.float32)).at[i].add(0.2 * serve.astype(jnp.float32) - 0.1 * discard.astype(jnp.float32))
            clear = serve | wash
            n['stage'] = n['stage'].at[h].set(jnp.where(wash, 0, jnp.where(serve, 5, n['stage'][h])))
            for field in ('filling', 'batch_owner'):
                n[field] = n[field].at[h].set(jnp.where(clear, -1, n[field][h]))
            for field in ('prep', 'cook', 'wet'):
                n[field] = n[field].at[h].set(jnp.where(clear, 0, n[field][h]))
            n['flask'] = n['flask'].at[i].add(-wash.astype(jnp.int32))
            refill = (a[i] == 16) & (n['well_water'] > 0) & (n['flask'][i] < 2)
            n['flask'] = n['flask'].at[i].add(refill.astype(jnp.int32))
            n['well_water'] = n['well_water'] - refill.astype(jnp.int32)
        heating = n['burner'] & (n['fuel'] > 0)
        n['fuel'] = n['fuel'] - heating.astype(jnp.int32)
        n['burner'] = n['burner'] & (n['fuel'] > 0)
        n['vented'] = jnp.any(n['tier_open'])
        n['pressure'] = jnp.clip(s['pressure'] + jnp.where(heating & (n['water'] > 0), 1, -1) - 2 * n['vented'].astype(jnp.int32), 0, 3).astype(jnp.int32)
        n['danger'] = jnp.clip(s['danger'] + jnp.where(heating & (n['water'] == 0), 1, -1), 0, 3).astype(jnp.int32)
        on = n['tier_of'] >= 0
        closed = ~n['tier_open'][jnp.maximum(n['tier_of'], 0)]
        n['wet'] = jnp.clip(n['wet'] + (on & heating & (n['water'] > 4)).astype(jnp.int32) + 2 * (on & n['spill']).astype(jnp.int32), 0, 4)
        cooking = on & closed & (n['pressure'] >= 2) & (n['water'] > 0) & (n['water'] <= 4) & ((n['stage'] == 2) | (n['stage'] == 3))
        n['cook'] = jnp.minimum(n['cook'] + cooking.astype(jnp.int32), 10)
        minimum = jnp.where(n['filling'] == 0, 4, 7)
        maximum = minimum + 2
        ready = (n['stage'] == 2) & (n['cook'] >= minimum)
        n['spoiled'] = ((n['stage'] == 2) | (n['stage'] == 3)) & ((n['cook'] > maximum) | (n['wet'] >= 3))
        n['stage'] = jnp.where(n['spoiled'], 4, jnp.where(ready, 3, n['stage'])).astype(jnp.int32)
        evaporated = heating & (n['water'] > 0) & (n['t'] % 3 == 0)
        n['water'] = n['water'] - evaporated.astype(jnp.int32)
        n['water_lost'] = n['water_lost'] + evaporated.astype(jnp.int32)
        remaining = jnp.sum(((n['filling'][:, None] == jnp.arange(2)[None, :]) & (n['stage'][:, None] < 4)).astype(jnp.int32), axis=0)
        n['success'] = jnp.all(n['served'] >= 2)
        n['failure'] = ((n['danger'] >= 3) | jnp.any(n['served'] + n['pantry'] + remaining < 2)) & ~n['success']
        n['terminated'] = n['success'] | n['failure']
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.sync(n)
        reward = jnp.sum(n['served_now']).astype(jnp.float32) + bonus - 0.01 * (a != 0).astype(jnp.float32) - n['failure'].astype(jnp.float32)
        return n, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
