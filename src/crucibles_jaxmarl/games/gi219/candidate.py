"""Festival Kitchens: four workshops sharing trays, heat and a routed belt."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 16
    num_actions = 22
    observation_size = 909
    max_steps = 384
    public_keys = ('t', 'done', 'terminated', 'truncated', 'terrain', 'pos', 'kitchen', 'tool', 'cargo', 'flavour', 'warmth', 'energy', 'station_pos', 'station_type', 'fuel', 'trays', 'pantry_stock', 'belt_pos', 'belt_item', 'belt_flavour', 'belt_warmth', 'belt_flow', 'junction_pos', 'route', 'service_pos', 'deadline', 'utility_pos', 'water', 'fuel_stock', 'served', 'wasted', 'processed', 'handed', 'belt_moved', 'reward_parts')

    def reset(self, key):
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False, jnp.bool_), 'terminated': jnp.array(False, jnp.bool_), 'truncated': jnp.array(False, jnp.bool_),
            'terrain': jnp.zeros((19, 19), jnp.int32),
            'pos': jnp.array([[3, 2], [4, 3], [3, 4], [2, 3], [15, 2], [14, 3], [15, 4], [16, 3], [15, 16], [14, 15], [15, 14], [16, 15], [3, 16], [4, 15], [3, 14], [2, 15]], jnp.int32),
            'agent_active': jnp.ones((16,), jnp.bool_), 'agent_visible': jnp.ones((16, 16), jnp.bool_),
            'kitchen': jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], jnp.int32),
            'tool': jnp.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], jnp.int32),
            'cargo': jnp.zeros((16,), jnp.int32), 'flavour': jnp.zeros((16,), jnp.int32), 'warmth': jnp.zeros((16,), jnp.int32), 'energy': jnp.full((16,), 16, jnp.int32),
            'station_pos': jnp.array([[3, 3], [15, 3], [15, 15], [3, 15]], jnp.int32), 'station_type': jnp.array([0, 1, 2, 3], jnp.int32),
            'station_active': jnp.ones((4,), jnp.bool_), 'station_visible': jnp.ones((16, 4), jnp.bool_),
            'fuel': jnp.array([0, 4, 0, 0], jnp.int32), 'trays': jnp.array([0, 0, 2, 2], jnp.int32), 'pantry_stock': jnp.full((4,), 8, jnp.int32),
            'belt_pos': jnp.array([[6, 6], [7, 6], [8, 6], [9, 6], [10, 6], [11, 6], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [11, 12], [10, 12], [9, 12], [8, 12], [7, 12], [6, 12], [6, 11], [6, 10], [6, 9], [6, 8], [6, 7], [5, 6], [12, 5], [13, 12], [6, 13]], jnp.int32),
            'belt_active': jnp.ones((28,), jnp.bool_), 'belt_visible': jnp.ones((16, 28), jnp.bool_),
            'belt_item': jnp.zeros((28,), jnp.int32), 'belt_flavour': jnp.zeros((28,), jnp.int32), 'belt_warmth': jnp.zeros((28,), jnp.int32), 'belt_flow': jnp.zeros((28, 2), jnp.int32),
            'junction_pos': jnp.array([[6, 6], [12, 6], [12, 12], [6, 12]], jnp.int32),
            'junction_active': jnp.ones((4,), jnp.bool_), 'junction_visible': jnp.ones((16, 4), jnp.bool_), 'route': jnp.zeros((4,), jnp.int32),
            'service_pos': jnp.array([[4, 1], [14, 1], [14, 17], [4, 17]], jnp.int32),
            'order_active': jnp.ones((4,), jnp.bool_), 'order_visible': jnp.zeros((16, 4), jnp.bool_),
            'order': jnp.ones((4,), jnp.int32), 'deadline': jnp.full((4,), 96, jnp.int32),
            'notes': jnp.zeros((16, 4), jnp.int32), 'note_life': jnp.zeros((16, 4), jnp.int32),
            'utility_pos': jnp.array([[9, 9], [10, 9]], jnp.int32), 'utility_active': jnp.ones((2,), jnp.bool_), 'utility_visible': jnp.ones((16, 2), jnp.bool_),
            'water': jnp.array(24, jnp.int32), 'fuel_stock': jnp.array(16, jnp.int32),
            'served': jnp.zeros((16,), jnp.int32), 'wasted': jnp.zeros((1,), jnp.int32), 'processed': jnp.zeros((16,), jnp.int32), 'handed': jnp.zeros((16,), jnp.int32), 'belt_moved': jnp.zeros((28,), jnp.int32),
            'reward_parts': jnp.zeros((16, 5), jnp.float32)
        }
        s['order'] = jax.random.randint(key, (4,), 1, 5, dtype=jnp.int32)
        s['order_visible'] = s['kitchen'][:, None] == jnp.arange(4, dtype=jnp.int32)[None, :]
        s['notes'] = jnp.where(s['order_visible'], s['order'][None, :], 0)
        s['note_life'] = jnp.where(s['order_visible'], 12, 0).astype(jnp.int32)
        s['terrain'] = s['terrain'].at[1:6, 1:6].set(1).at[1:6, 13:18].set(2).at[13:18, 13:18].set(3).at[13:18, 1:6].set(4)
        s['terrain'] = s['terrain'].at[s['belt_pos'][:, 1], s['belt_pos'][:, 0]].set(5)
        s['belt_flow'] = s['belt_pos'][self.successors(s)] - s['belt_pos']
        return self.observe(s), s

    def distances(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=2, dtype=jnp.int32)

    def successors(self, s):
        nxt = jnp.concatenate(((jnp.arange(24, dtype=jnp.int32) + 1) % 24, jnp.array([0, 6, 12, 18], jnp.int32)))
        corners = jnp.array([0, 6, 12, 18], jnp.int32)
        nxt = nxt.at[corners].set(jnp.where(s['route'] == 1, jnp.arange(24, 28, dtype=jnp.int32), corners + 1))
        return nxt.at[24:28].set(jnp.where(s['route'] == 1, jnp.arange(24, 28, dtype=jnp.int32), corners))

    def observe(self, s):
        public = jnp.concatenate([s[k].astype(jnp.float32).ravel() for k in self.public_keys])
        own = jnp.concatenate((jnp.arange(16, dtype=jnp.int32)[:, None], s['notes'], s['note_life']), axis=1).astype(jnp.float32)
        return jnp.concatenate((own, jnp.broadcast_to(public, (16, public.shape[0]))), axis=1)

    def legal_actions(self, s):
        ds = self.distances(s['pos'], s['station_pos'])
        db = self.distances(s['pos'], s['belt_pos'])
        near = ds <= 1
        empty = s['cargo'] == 0
        process = (s['tool'] == 1) & jnp.any(near & (((s['station_type'][None, :] == 0) & (s['cargo'][:, None] == 1)) | ((s['station_type'][None, :] == 1) & (s['cargo'][:, None] == 2) & (s['fuel'][None, :] > 0)) | ((s['station_type'][None, :] == 2) & (s['cargo'][:, None] == 3)) | ((s['station_type'][None, :] == 3) & (s['cargo'][:, None] == 4) & (s['trays'][None, :] > 0))), axis=1)
        peers = self.distances(s['pos'], s['pos']) == 1
        take_raw = empty & jnp.any(near & (s['pantry_stock'][None, :] > 0), axis=1)
        put_belt = ~empty & jnp.any((db <= 1) & (s['belt_item'][None, :] == 0), axis=1)
        take_belt = empty & jnp.any((db <= 1) & (s['belt_item'][None, :] > 0), axis=1)
        hand = ~empty & jnp.any(peers & empty[None, :], axis=1)
        wash = (s['tool'] == 3) & (s['cargo'] == 6) & near[:, 2] & (s['water'] > 0)
        tray = empty & near[:, 2] & (s['trays'][2] > 0)
        rack = (s['cargo'] == 7) & near[:, 3] & (s['trays'][3] < 16)
        fuel = empty & (self.distances(s['pos'], s['utility_pos'])[:, 0] <= 1) & (s['fuel_stock'] > 0)
        stoke = (s['cargo'] == 8) & (s['tool'] == 4) & near[:, 1] & (s['fuel'][1] < 12)
        switch = jnp.any(self.distances(s['pos'], s['junction_pos']) <= 1, axis=1)
        serve = (s['cargo'] == 5) & jnp.any(self.distances(s['pos'], s['service_pos']) <= 1, axis=1)
        discard = (s['cargo'] >= 1) & (s['cargo'] <= 5)
        reheat = (s['tool'] == 1) & near[:, 1] & (s['cargo'] >= 3) & (s['cargo'] <= 5) & (s['fuel'][1] > 0)
        extra = jnp.stack((take_raw, process, put_belt, take_belt, hand, wash, tray, rack, fuel, stoke, switch, serve, discard, jnp.any(peers, axis=1), jnp.any(peers, axis=1), jnp.ones((16,), jnp.bool_), reheat), axis=1)
        mask = jnp.concatenate((jnp.ones((16, 5), jnp.bool_), extra), axis=1) & (s['energy'][:, None] > 0) & ~s['done']
        return mask.at[:, 0].set(True).at[:, 20].set(~s['done'])

    def interact(self, i, s, a):
        b = a[i]
        ds = jnp.sum(jnp.abs(s['station_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        st = jnp.argmin(ds)
        close = ds[st] <= 1
        code = s['cargo'][i]
        take = (b == 5) & close & (code == 0) & (s['pantry_stock'][st] > 0)
        s['pantry_stock'] = s['pantry_stock'].at[st].add(-take.astype(jnp.int32))
        s['cargo'] = s['cargo'].at[i].set(jnp.where(take, 1, code))
        s['flavour'] = s['flavour'].at[i].set(jnp.where(take, st + 1, s['flavour'][i]))
        process = (b == 6) & close & (s['tool'][i] == 1) & (code == st + 1) & ((st != 1) | (s['fuel'][st] > 0)) & ((st != 3) | (s['trays'][st] > 0))
        s['cargo'] = s['cargo'].at[i].add(process.astype(jnp.int32))
        s['processed'] = s['processed'].at[i].set(process.astype(jnp.int32))
        s['fuel'] = s['fuel'].at[st].add(-(process & (st == 1)).astype(jnp.int32))
        s['trays'] = s['trays'].at[st].add(-(process & (st == 3)).astype(jnp.int32))
        s['warmth'] = s['warmth'].at[i].set(jnp.where(process & (st == 1), 64, s['warmth'][i]))
        db = jnp.sum(jnp.abs(s['belt_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        bi = jnp.argmin(jnp.where((db <= 1) & (s['belt_item'] == 0), db, 999))
        put = (b == 7) & (db[bi] <= 1) & (s['belt_item'][bi] == 0) & (code > 0)
        s['belt_item'] = s['belt_item'].at[bi].set(jnp.where(put, code, s['belt_item'][bi]))
        s['belt_flavour'] = s['belt_flavour'].at[bi].set(jnp.where(put, s['flavour'][i], s['belt_flavour'][bi]))
        s['belt_warmth'] = s['belt_warmth'].at[bi].set(jnp.where(put, s['warmth'][i], s['belt_warmth'][bi]))
        s['cargo'] = s['cargo'].at[i].set(jnp.where(put, 0, s['cargo'][i]))
        s['flavour'] = s['flavour'].at[i].set(jnp.where(put, 0, s['flavour'][i]))
        s['warmth'] = s['warmth'].at[i].set(jnp.where(put, 0, s['warmth'][i]))
        bi = jnp.argmin(jnp.where((db <= 1) & (s['belt_item'] > 0), db, 999))
        pick = (b == 8) & (db[bi] <= 1) & (s['belt_item'][bi] > 0) & (code == 0)
        s['cargo'] = s['cargo'].at[i].set(jnp.where(pick, s['belt_item'][bi], s['cargo'][i]))
        s['flavour'] = s['flavour'].at[i].set(jnp.where(pick, s['belt_flavour'][bi], s['flavour'][i]))
        s['warmth'] = s['warmth'].at[i].set(jnp.where(pick, s['belt_warmth'][bi], s['warmth'][i]))
        s['belt_item'] = s['belt_item'].at[bi].set(jnp.where(pick, 0, s['belt_item'][bi]))
        s['belt_flavour'] = s['belt_flavour'].at[bi].set(jnp.where(pick, 0, s['belt_flavour'][bi]))
        s['belt_warmth'] = s['belt_warmth'].at[bi].set(jnp.where(pick, 0, s['belt_warmth'][bi]))
        pd = jnp.sum(jnp.abs(s['pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        pi = jnp.argmin(jnp.where((pd == 1) & (s['cargo'] == 0), jnp.arange(16, dtype=jnp.int32), 999))
        give = (b == 9) & (pd[pi] == 1) & (s['cargo'][pi] == 0) & (code > 0)
        for field in ('cargo', 'flavour', 'warmth'):
            val = s[field][i]
            s[field] = s[field].at[pi].set(jnp.where(give, val, s[field][pi])).at[i].set(jnp.where(give, 0, val))
        s['handed'] = s['handed'].at[i].set(give.astype(jnp.int32))
        wash = (b == 10) & (code == 6) & (s['tool'][i] == 3) & (ds[2] <= 1) & (s['water'] > 0)
        tray = (b == 11) & (code == 0) & (ds[2] <= 1) & (s['trays'][2] > 0)
        rack = (b == 12) & (code == 7) & (ds[3] <= 1) & (s['trays'][3] < 16)
        fuel = (b == 13) & (code == 0) & (jnp.sum(jnp.abs(s['pos'][i] - s['utility_pos'][0]), dtype=jnp.int32) <= 1) & (s['fuel_stock'] > 0)
        stoke = (b == 14) & (code == 8) & (s['tool'][i] == 4) & (ds[1] <= 1) & (s['fuel'][1] < 12)
        s['cargo'] = s['cargo'].at[i].set(jnp.where(wash | tray, 7, jnp.where(fuel, 8, jnp.where(rack | stoke, 0, s['cargo'][i]))))
        s['water'] = s['water'] - wash.astype(jnp.int32)
        s['fuel_stock'] = s['fuel_stock'] - fuel.astype(jnp.int32)
        s['fuel'] = s['fuel'].at[1].add(stoke.astype(jnp.int32))
        s['trays'] = s['trays'].at[2].add(-tray.astype(jnp.int32)).at[3].add(rack.astype(jnp.int32))
        jd = jnp.sum(jnp.abs(s['junction_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        ji = jnp.argmin(jd)
        switch = (b == 15) & (jd[ji] <= 1)
        s['route'] = s['route'].at[ji].set(jnp.where(switch, 1 - s['route'][ji], s['route'][ji]))
        sd = jnp.sum(jnp.abs(s['service_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        si = jnp.argmin(sd)
        attempt = (b == 16) & (code == 5) & (sd[si] <= 1)
        serve = attempt & (s['flavour'][i] == s['order'][si]) & (s['warmth'][i] > 0)
        s['cargo'] = s['cargo'].at[i].set(jnp.where(serve, 6, s['cargo'][i]))
        s['flavour'] = s['flavour'].at[i].set(jnp.where(serve, 0, s['flavour'][i]))
        s['warmth'] = s['warmth'].at[i].set(jnp.where(serve, 0, s['warmth'][i]))
        s['served'] = s['served'].at[i].set(serve.astype(jnp.int32))
        s['order'] = s['order'].at[si].set(jnp.where(serve, s['order'][si] % 4 + 1, s['order'][si]))
        s['deadline'] = s['deadline'].at[si].set(jnp.where(serve, 96, s['deadline'][si]))
        s['reward_parts'] = s['reward_parts'].at[:, 0].add(0.5 * serve.astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[:, 1].add(serve.astype(jnp.float32) * (s['kitchen'] == si).astype(jnp.float32) * 1.5)
        s['reward_parts'] = s['reward_parts'].at[i, 2].add(serve.astype(jnp.float32) * 2.0)
        s['reward_parts'] = s['reward_parts'].at[i, 4].add(-0.4 * (attempt & ~serve).astype(jnp.float32))
        discard = (b == 17) & (code >= 1) & (code <= 5)
        s['cargo'] = s['cargo'].at[i].set(jnp.where(discard, jnp.where(code == 5, 6, 0), s['cargo'][i]))
        s['flavour'] = s['flavour'].at[i].set(jnp.where(discard, 0, s['flavour'][i]))
        s['warmth'] = s['warmth'].at[i].set(jnp.where(discard, 0, s['warmth'][i]))
        s['wasted'] = s['wasted'] + discard.astype(jnp.int32)
        speak = (b == 18) & (pd == 1)
        k = s['kitchen'][i]
        s['notes'] = s['notes'].at[:, k].set(jnp.where(speak, s['order'][k], s['notes'][:, k]))
        s['note_life'] = s['note_life'].at[:, k].set(jnp.where(speak, 12, s['note_life'][:, k]))
        pi = jnp.argmin(jnp.where(pd == 1, jnp.arange(16, dtype=jnp.int32), 999))
        swap = (b == 19) & (pd[pi] == 1)
        ti, tp = s['tool'][i], s['tool'][pi]
        s['tool'] = s['tool'].at[i].set(jnp.where(swap, tp, ti)).at[pi].set(jnp.where(swap, ti, tp))
        reheat = (b == 21) & (s['tool'][i] == 1) & (ds[1] <= 1) & (code >= 3) & (code <= 5) & (s['fuel'][1] > 0)
        s['fuel'] = s['fuel'].at[1].add(-reheat.astype(jnp.int32))
        s['warmth'] = s['warmth'].at[i].set(jnp.where(reheat, 64, s['warmth'][i]))
        s['reward_parts'] = s['reward_parts'].at[i, 3].add(-0.03 * (wash | reheat | (process & (st == 1))).astype(jnp.float32))
        return s

    def advance(self, key, state, actions):
        s = dict(state)
        a = jnp.clip(actions, 0, 21)
        valid = (actions >= 0) & (actions < 22) & self.legal_actions(s)[jnp.arange(16, dtype=jnp.int32), a]
        a = jnp.where(valid, a, 0)
        s['served'] = jnp.zeros((16,), jnp.int32)
        s['processed'] = jnp.zeros((16,), jnp.int32)
        s['handed'] = jnp.zeros((16,), jnp.int32)
        s['wasted'] = jnp.zeros((1,), jnp.int32)
        s['reward_parts'] = jnp.zeros((16, 5), jnp.float32)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['pos'] + delta[jnp.where(a < 5, a, 0)]
        inside = jnp.all((p >= 0) & (p <= 18), axis=1)
        q = jnp.clip(p, 0, 18)
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(16, dtype=jnp.bool_), axis=1)
        clash = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1, dtype=jnp.int32) > 1
        blocked = jnp.any(jnp.all(q[:, None, :] == s['station_pos'][None, :, :], axis=2), axis=1) | (s['terrain'][q[:, 1], q[:, 0]] == 5)
        move = (a > 0) & (a < 5) & inside & ~occupied & ~clash & ~blocked
        s['pos'] = jnp.where(move[:, None], q, s['pos'])
        effort = move.astype(jnp.int32) + ((a >= 5) & (a != 20)).astype(jnp.int32)
        s['energy'] = jnp.clip(s['energy'] - effort + 2 * (a == 0).astype(jnp.int32) + 5 * (a == 20).astype(jnp.int32), 0, 16)
        s['reward_parts'] = s['reward_parts'].at[:, 3].add(-0.01 * effort.astype(jnp.float32))
        s = jax.lax.fori_loop(0, 16, lambda i, carry: self.interact(i, carry, a), s)
        nxt = self.successors(s)
        ids = jnp.arange(28, dtype=jnp.int32)
        can = (s['belt_item'] > 0) & (s['belt_item'][nxt] == 0) & (nxt != ids)
        prior = jnp.any((nxt[:, None] == nxt[None, :]) & can[None, :] & (ids[None, :] < ids[:, None]), axis=1)
        moving = can & ~prior
        incoming = (nxt[:, None] == ids[None, :]) & moving[:, None]
        source = jnp.argmax(incoming, axis=0)
        arrivals = jnp.any(incoming, axis=0)
        for field in ('belt_item', 'belt_flavour', 'belt_warmth'):
            old = s[field]
            s[field] = jnp.where(arrivals, old[source], jnp.where(moving, 0, old))
        s['belt_moved'] = moving.astype(jnp.int32)
        s['belt_flow'] = s['belt_pos'][nxt] - s['belt_pos']
        hot = (s['cargo'] >= 3) & (s['cargo'] <= 5)
        cooling = hot & ((s['tool'] != 2) | ((s['t'] % 2) == 0))
        s['warmth'] = jnp.maximum(0, s['warmth'] - cooling.astype(jnp.int32))
        expired = hot & (s['warmth'] == 0)
        s['cargo'] = jnp.where(expired, jnp.where(s['cargo'] == 5, 6, 0), s['cargo'])
        s['flavour'] = jnp.where(expired, 0, s['flavour'])
        belt_hot = (s['belt_item'] >= 3) & (s['belt_item'] <= 5)
        s['belt_warmth'] = jnp.maximum(0, s['belt_warmth'] - belt_hot.astype(jnp.int32))
        rotten = belt_hot & (s['belt_warmth'] == 0)
        s['belt_item'] = jnp.where(rotten, jnp.where(s['belt_item'] == 5, 6, 0), s['belt_item'])
        s['belt_flavour'] = jnp.where(rotten, 0, s['belt_flavour'])
        s['wasted'] = s['wasted'] + jnp.sum(expired, dtype=jnp.int32) + jnp.sum(rotten, dtype=jnp.int32)
        late = s['deadline'] <= 1
        s['deadline'] = jnp.where(late, 96, s['deadline'] - 1)
        s['order'] = jnp.where(late, s['order'] % 4 + 1, s['order'])
        s['reward_parts'] = s['reward_parts'].at[:, 4].add(-0.05 * s['wasted'][0].astype(jnp.float32) - 0.3 * late[s['kitchen']].astype(jnp.float32))
        s['water'] = jnp.minimum(24, s['water'] + ((s['t'] % 6) == 5).astype(jnp.int32))
        s['fuel_stock'] = jnp.minimum(16, s['fuel_stock'] + ((s['t'] % 12) == 11).astype(jnp.int32))
        s['pantry_stock'] = jnp.minimum(8, s['pantry_stock'] + ((s['t'] % 20) == 19).astype(jnp.int32))
        s['note_life'] = jnp.maximum(0, s['note_life'] - 1)
        s['notes'] = jnp.where(s['note_life'] > 0, s['notes'], 0)
        s['notes'] = jnp.where(s['order_visible'], s['order'][None, :], s['notes'])
        s['note_life'] = jnp.where(s['order_visible'], 12, s['note_life'])
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.array(False, jnp.bool_)
        s['truncated'] = (s['t'] >= 384) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        return s, jnp.sum(s['reward_parts'], axis=1, dtype=jnp.float32)

    def raw_step(self, key, state, actions):
        s, rewards = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros((16,), jnp.float32)), lambda _: self.advance(key, state, actions), operand=None)
        return self.observe(s), s, rewards, jnp.full((16,), s['done'], jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
