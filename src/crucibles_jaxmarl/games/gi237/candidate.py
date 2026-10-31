"""Canal Nightmarket Kitchens. Original, unexecuted author kernel."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 48
    observation_size = 212
    num_actions = 16
    max_steps = 768

    def nearest(self, p, q, active, k, radius):
        d = jnp.sum(jnp.abs(p[:, None, :] - q[None, :, :]), axis=2)
        ix = jnp.argsort(d + (~active[None, :]) * 10000, axis=1)[:, :k]
        v = (jnp.take_along_axis(d, ix, axis=1) <= radius) & active[ix]
        return ix.astype(jnp.int32), jnp.any(jax.nn.one_hot(ix, q.shape[0], dtype=jnp.bool_) & v[:, :, None], axis=1)

    def refresh(self, s):
        s = dict(s)
        s['peer'], s['agent_visible'] = self.nearest(s['pos'], s['pos'], s['active'], 6, 4)
        s['near'], s['station_visible'] = self.nearest(s['pos'], s['station_pos'], s['station_active'], 8, 4)
        s['boats'], s['barge_visible'] = self.nearest(s['pos'], s['barge_pos'], s['barge_active'], 2, 3)
        s['clients'], s['customer_visible'] = self.nearest(s['pos'], s['customer_pos'], s['customer_active'], 3, 3)
        return s

    def reset(self, key):
        a = jnp.arange(48, dtype=jnp.int32)
        team, role = a // 4, a % 4
        pos = jnp.stack((team * 4 + 1 + role % 2, jnp.where(role < 2, 2, 7)), axis=1)
        z = jnp.arange(60, dtype=jnp.int32)
        offsets = jnp.array([[1, 4], [1, 2], [2, 2], [2, 8], [0, 5]], jnp.int32)
        sp = jnp.stack(((z // 5) * 4, jnp.zeros(60, jnp.int32)), axis=1) + offsets[z % 5]
        x, y = jnp.arange(48, dtype=jnp.int32)[None, :], jnp.arange(12, dtype=jnp.int32)[:, None]
        terrain = jnp.where((y == 5) | (y == 6), jnp.where(x % 4 == 0, 2, 1), 0).astype(jnp.int32)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
             'pos': pos, 'active': jnp.ones(48, jnp.bool_), 'team': team, 'tool': role,
             'item': jnp.where(role == 0, 1, jnp.where(role == 1, 2, 0)).astype(jnp.int32), 'heat': jnp.zeros(48, jnp.int32),
             'message': jnp.zeros(48, jnp.int32), 'message_ttl': jnp.zeros(48, jnp.int32), 'event': jnp.zeros(48, jnp.int32),
             'station_pos': sp, 'station_active': jnp.ones(60, jnp.bool_), 'station_kind': z % 5,
             'veg': jnp.ones(12, jnp.int32), 'grain': jnp.zeros(12, jnp.int32), 'clean': jnp.ones(12, jnp.int32),
             'timer': jnp.zeros(12, jnp.int32), 'meal': jnp.zeros(12, jnp.int32), 'plate_heat': jnp.zeros(12, jnp.int32),
             'water': jnp.full(12, 4, jnp.int32), 'gate': jnp.ones(12, jnp.bool_), 'water_pipe': jnp.zeros((12, 2), jnp.int32),
             'station_level': jnp.zeros(60, jnp.int32), 'station_aux': jnp.zeros(60, jnp.int32), 'station_state': jnp.zeros(60, jnp.int32),
             'barge_pos': jnp.stack((jnp.arange(6, dtype=jnp.int32) * 8 + 1, jnp.full(6, 5, jnp.int32)), axis=1),
             'barge_active': jnp.ones(6, jnp.bool_), 'roots': jnp.full(6, 8, jnp.int32), 'rice': jnp.full(6, 8, jnp.int32), 'barge_event': jnp.zeros(6, jnp.int32),
             'customer_pos': jnp.stack((jnp.arange(12, dtype=jnp.int32) * 4 + 1, jnp.full(12, 9, jnp.int32)), axis=1),
             'customer_active': jnp.ones(12, jnp.bool_), 'request': jax.random.randint(key, (12,), 4, 6, dtype=jnp.int32),
             'deadline': jnp.full(12, 48, jnp.int32), 'served': jnp.zeros(12, jnp.int32), 'customer_event': jnp.zeros(12, jnp.int32),
             'terrain': terrain, 'peer': jnp.zeros((48, 6), jnp.int32), 'near': jnp.zeros((48, 8), jnp.int32),
             'boats': jnp.zeros((48, 2), jnp.int32), 'clients': jnp.zeros((48, 3), jnp.int32),
             'agent_visible': jnp.zeros((48, 48), jnp.bool_), 'station_visible': jnp.zeros((48, 60), jnp.bool_),
             'barge_visible': jnp.zeros((48, 6), jnp.bool_), 'customer_visible': jnp.zeros((48, 12), jnp.bool_)}
        s = self.display_stations(s)
        s = self.refresh(s)
        return self.observe(s), s

    def display_stations(self, s):
        s = dict(s)
        c = jnp.arange(60, dtype=jnp.int32) // 5
        k = s['station_kind']
        s['station_level'] = jnp.where(k == 2, s['veg'][c], jnp.where((k == 3) | (k == 4), s['water'][c], 0)).astype(jnp.int32)
        s['station_aux'] = jnp.where(k == 2, s['grain'][c], jnp.where(k == 3, s['clean'][c], s['gate'][c])).astype(jnp.int32)
        s['station_state'] = jnp.where(k == 2, s['meal'][c] * 10 + s['timer'][c], 0).astype(jnp.int32)
        return s

    def legal_actions(self, s):
        on = ~s['done']
        d = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['station_pos'][None, :, :]), axis=2)
        near = d <= 1
        cut = jnp.any(near & (s['station_kind'][None, :] == 1), axis=1)
        stove = jnp.any(near & (s['station_kind'][None, :] == 2), axis=1)
        wash = jnp.any(near & (s['station_kind'][None, :] == 3), axis=1)
        gate = jnp.any(near & (s['station_kind'][None, :] == 4), axis=1)
        bd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['barge_pos'][None, :, :]), axis=2)
        cd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['customer_pos'][None, :, :]), axis=2)
        ad = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2)
        peer = jnp.any((ad <= 1) & (ad > 0), axis=1)
        yes = jnp.full(48, on, jnp.bool_)
        return jnp.stack((jnp.ones(48, jnp.bool_), yes, yes, yes, yes, yes,
                          yes & (s['item'] == 0) & jnp.any(bd <= 2, axis=1),
                          yes & cut & (s['item'] == 1) & (s['tool'] == 0),
                          yes & stove & ((s['item'] == 2) | (s['item'] == 3)), yes & stove & (s['item'] == 0),
                          yes & wash & (s['tool'] == 3), yes & (s['item'] >= 4) & jnp.any(cd <= 1, axis=1),
                          yes & gate & (s['tool'] == 2), yes & peer & (s['item'] > 0),
                          yes & jnp.any(cd <= 3, axis=1), yes & peer), axis=1)

    def observe(self, s):
        a = jnp.arange(48, dtype=jnp.int32)
        own = jnp.stack((s['pos'][:, 0] / 47.0, s['pos'][:, 1] / 11.0, s['team'] / 11.0, s['tool'] / 3.0,
                         s['item'] / 5.0, s['heat'] / 24.0, s['message'] / 5.0, s['message_ttl'] / 16.0,
                         s['event'] / 8.0, a / 47.0, jnp.full(48, s['t'] / 768.0), jnp.full(48, (s['t'] % 4) / 3.0)), axis=1)
        pdata = jnp.stack((s['pos'][:, 0] / 47.0, s['pos'][:, 1] / 11.0, s['tool'] / 3.0, s['item'] / 5.0,
                           s['heat'] / 24.0, s['message'] / 5.0, s['message_ttl'] / 16.0, s['team'] / 11.0, a / 47.0, jnp.ones(48)), axis=1)
        pv = jnp.take_along_axis(s['agent_visible'], s['peer'], axis=1)
        peers = (pdata[s['peer']] * pv[:, :, None]).reshape((48, 60))
        q = s['station_pos']
        st = jnp.stack((q[:, 0] / 47.0, q[:, 1] / 11.0, s['station_kind'] / 4.0, s['station_level'] / 6.0,
                        s['station_aux'] / 3.0, s['station_state'] / 54.0, (jnp.arange(60, dtype=jnp.int32) // 5) / 11.0, jnp.ones(60)), axis=1)
        sv = jnp.take_along_axis(s['station_visible'], s['near'], axis=1)
        stations = (st[s['near']] * sv[:, :, None]).reshape((48, 64))
        q = s['barge_pos']
        bd = jnp.stack((q[:, 0] / 47.0, q[:, 1] / 11.0, s['roots'] / 8.0, s['rice'] / 8.0, jnp.arange(6, dtype=jnp.int32) / 5.0, jnp.ones(6)), axis=1)
        bv = jnp.take_along_axis(s['barge_visible'], s['boats'], axis=1)
        boats = (bd[s['boats']] * bv[:, :, None]).reshape((48, 12))
        q = s['customer_pos']
        cd = jnp.stack((q[:, 0] / 47.0, q[:, 1] / 11.0, s['request'] / 5.0, s['deadline'] / 48.0,
                        s['served'] / 768.0, jnp.arange(12, dtype=jnp.int32) / 11.0, jnp.ones(12)), axis=1)
        cv = jnp.take_along_axis(s['customer_visible'], s['clients'], axis=1)
        clients = (cd[s['clients']] * cv[:, :, None]).reshape((48, 21))
        sector = jnp.clip(s['pos'][:, 0] // 4, 0, 11)
        local = jnp.stack((s['water'][sector] / 6.0, s['gate'][sector], s['water_pipe'][sector, 0] / 2.0,
                           s['water_pipe'][sector, 1] / 2.0, s['plate_heat'][sector] / 24.0), axis=1)
        extra = jnp.concatenate((self.legal_actions(s).astype(jnp.float32), local), axis=1)
        padding = jnp.zeros((48, 22), jnp.float32)
        return jnp.concatenate((own, peers, stations, boats, clients, extra, padding), axis=1).astype(jnp.float32)

    def absorb(self, args):
        key, s, actions = args
        return self.observe(s), s, jnp.zeros(48, jnp.float32), jnp.ones(48, jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state['done'], self.absorb, self.advance, (key, state, actions))

    def advance(self, args):
        key, s, actions = args
        s = dict(s)
        z = jnp.clip(actions, 0, 15)
        actions = jnp.where(self.legal_actions(s)[jnp.arange(48, dtype=jnp.int32), z] & (actions >= 0) & (actions < 16), z, 0)
        original_pos = s['pos']
        s['event'] = jnp.zeros(48, jnp.int32)
        s['barge_event'] = jnp.zeros(6, jnp.int32)
        s['customer_event'] = jnp.zeros(12, jnp.int32)
        s['message_ttl'] = jnp.maximum(0, s['message_ttl'] - 1)
        s['message'] = jnp.where(s['message_ttl'] > 0, s['message'], 0)
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        reward = jnp.zeros(48, jnp.float32)

        def actor(k, carry):
            w, r = carry
            w = dict(w)
            i = (k + s['t']) % 48
            a = actions[i]
            p = w['pos'][i]
            target = jnp.clip(p + dirs[jnp.clip(a, 0, 4)], jnp.array([0, 0], dtype=jnp.int32), jnp.array([47, 11], dtype=jnp.int32))
            sector = target[0] // 4
            tile = w['terrain'][target[1], target[0]]
            passable = (tile == 0) | ((tile == 2) & (w['water'][sector] <= 3))
            free = ~jnp.any(jnp.all(w['pos'] == target, axis=1) & (jnp.arange(48, dtype=jnp.int32) != i))
            move = (a >= 1) & (a <= 4) & passable & free & jnp.any(target != p)
            w['pos'] = w['pos'].at[i].set(jnp.where(move, target, p))
            dist = jnp.sum(jnp.abs(w['station_pos'] - p), axis=1)
            stove = (dist <= 1) & (w['station_kind'] == 2)
            c = jnp.argmax(stove) // 5
            bd = jnp.sum(jnp.abs(w['barge_pos'] - p), axis=1)
            root = (w['tool'][i] == 0) | (w['tool'][i] == 2)
            available = jnp.where(root, w['roots'], w['rice']) > 0
            b = jnp.argmin(bd + (~available) * 1000)
            assist = jnp.any((jnp.sum(jnp.abs(original_pos - p), axis=1) <= 1) & (actions == 5) & (jnp.arange(48, dtype=jnp.int32) != i))
            lift = (a == 6) & (bd[b] <= 2) & available[b] & assist & (w['item'][i] == 0)
            w['roots'] = w['roots'].at[b].add(-(lift & root).astype(jnp.int32))
            w['rice'] = w['rice'].at[b].add(-(lift & ~root).astype(jnp.int32))
            w['item'] = w['item'].at[i].set(jnp.where(lift, jnp.where(root, 1, 2), w['item'][i]))
            w['barge_event'] = w['barge_event'].at[b].set(jnp.where(lift, 1, w['barge_event'][b]))
            cut = (a == 7) & (w['item'][i] == 1) & (w['tool'][i] == 0) & jnp.any((dist <= 1) & (w['station_kind'] == 1))
            w['item'] = w['item'].at[i].set(jnp.where(cut, 3, w['item'][i]))
            room = jnp.where(w['item'][i] == 3, w['veg'][c] < 2, w['grain'][c] < 2)
            load = (a == 8) & jnp.any(stove) & room & ((w['item'][i] == 2) | (w['item'][i] == 3)) & (w['timer'][c] == 0) & (w['meal'][c] == 0)
            w['veg'] = w['veg'].at[c].add((load & (w['item'][i] == 3)).astype(jnp.int32))
            w['grain'] = w['grain'].at[c].add((load & (w['item'][i] == 2)).astype(jnp.int32))
            w['item'] = w['item'].at[i].set(jnp.where(load, 0, w['item'][i]))
            collect = (a == 9) & jnp.any(stove) & (w['meal'][c] > 0) & (w['timer'][c] == 0) & (w['item'][i] == 0)
            w['item'] = w['item'].at[i].set(jnp.where(collect, w['meal'][c], w['item'][i]))
            w['heat'] = w['heat'].at[i].set(jnp.where(collect, w['plate_heat'][c], w['heat'][i]))
            w['meal'] = w['meal'].at[c].set(jnp.where(collect, 0, w['meal'][c]))
            washes = (dist <= 1) & (w['station_kind'] == 3)
            c = jnp.argmax(washes) // 5
            wash = (a == 10) & jnp.any(washes) & (w['tool'][i] == 3) & (w['water'][c] > 0) & (w['clean'][c] < 3)
            w['water'] = w['water'].at[c].add(-wash.astype(jnp.int32))
            w['clean'] = w['clean'].at[c].add(wash.astype(jnp.int32))
            cd = jnp.sum(jnp.abs(w['customer_pos'] - p), axis=1)
            match = (cd <= 1) & (w['request'] == w['item'][i])
            customer = jnp.argmax(match)
            serve = (a == 11) & jnp.any(match) & (w['item'][i] >= 4)
            quality = w['heat'][i] / 24.0
            w['served'] = w['served'].at[customer].set(jnp.minimum(768, w['served'][customer] + serve))
            w['customer_event'] = w['customer_event'].at[customer].set(jnp.where(serve, 1, w['customer_event'][customer]))
            w['deadline'] = w['deadline'].at[customer].set(jnp.where(serve, 48, w['deadline'][customer]))
            w['request'] = w['request'].at[customer].set(jnp.where(serve, 9 - w['request'][customer], w['request'][customer]))
            w['item'] = w['item'].at[i].set(jnp.where(serve, 0, w['item'][i]))
            r = r + serve * (w['team'] == w['team'][i]).astype(jnp.float32) * (0.04 + 0.08 * quality)
            gates = (dist <= 1) & (w['station_kind'] == 4)
            c = jnp.argmax(gates) // 5
            gate = (a == 12) & jnp.any(gates) & (w['tool'][i] == 2)
            w['gate'] = w['gate'].at[c].set(jnp.where(gate, ~w['gate'][c], w['gate'][c]))
            ad = jnp.sum(jnp.abs(w['pos'] - p), axis=1)
            peers = (ad <= 1) & (jnp.arange(48, dtype=jnp.int32) != i) & (actions == 0)
            empty = peers & (w['item'] == 0)
            j = jnp.argmax(empty)
            hand = (a == 13) & jnp.any(empty) & (w['item'][i] > 0)
            w['item'] = w['item'].at[j].set(jnp.where(hand, w['item'][i], w['item'][j])).at[i].set(jnp.where(hand, 0, w['item'][i]))
            w['heat'] = w['heat'].at[j].set(jnp.where(hand, w['heat'][i], w['heat'][j]))
            customer = jnp.argmin(cd)
            signal = (a == 14) & (cd[customer] <= 3)
            listeners = (ad <= 3) & signal
            w['message'] = jnp.where(listeners, w['request'][customer], w['message'])
            w['message_ttl'] = jnp.where(listeners, 16, w['message_ttl'])
            j = jnp.argmax(peers)
            swap = (a == 15) & jnp.any(peers)
            ti, tj = w['tool'][i], w['tool'][j]
            w['tool'] = w['tool'].at[i].set(jnp.where(swap, tj, ti)).at[j].set(jnp.where(swap, ti, tj))
            w['event'] = w['event'].at[i].set(jnp.where(lift, 1, jnp.where(cut, 2, jnp.where(load, 3, jnp.where(collect, 4, jnp.where(wash, 5, jnp.where(serve, 6, jnp.where(gate, 7, jnp.where(hand | signal | swap, 8, 0)))))))))
            r = r.at[i].add(-0.002 * (a != 0) + serve * (0.08 + 0.12 * quality))
            return w, r

        s, reward = jax.lax.fori_loop(0, 48, actor, (s, reward))
        t = s['t'] + 1
        ready = (s['veg'] + s['grain'] == 3) & (s['veg'] >= 1) & (s['grain'] >= 1) & (s['clean'] > 0) & (s['meal'] == 0)
        s['timer'] = jnp.where(ready, 4, jnp.maximum(0, s['timer'] - 1)).astype(jnp.int32)
        s['meal'] = jnp.where(ready, jnp.where(s['veg'] == 2, 4, 5), s['meal']).astype(jnp.int32)
        s['clean'] = s['clean'] - ready.astype(jnp.int32)
        s['veg'] = jnp.where(ready, 0, s['veg'])
        s['grain'] = jnp.where(ready, 0, s['grain'])
        s['plate_heat'] = jnp.where(ready | (s['timer'] > 0), 24, jnp.maximum(0, s['plate_heat'] - 1)).astype(jnp.int32)
        spoiled = (s['item'] >= 4) & (s['heat'] == 0)
        reward = reward - 0.04 * spoiled
        s['item'] = jnp.where(spoiled, 0, s['item'])
        s['heat'] = jnp.maximum(0, s['heat'] - 1)
        stale = (s['meal'] > 0) & (s['timer'] == 0) & (s['plate_heat'] == 0)
        s['meal'] = jnp.where(stale, 0, s['meal'])
        water = jnp.minimum(6, s['water'] + s['water_pipe'][:, 0] + ((jnp.arange(12, dtype=jnp.int32) == 0) & (t % 2 == 0)))
        flow = jnp.where(s['gate'] & (water >= 2), 1, 0).astype(jnp.int32)
        s['water'] = water - flow
        arrivals = jnp.concatenate((jnp.zeros(1, jnp.int32), flow[:-1]))
        s['water_pipe'] = jnp.stack((s['water_pipe'][:, 1], arrivals), axis=1)
        sector = s['barge_pos'][:, 0] // 4
        advance = (t % 3 == 0) & (s['water'][sector] >= 2) & s['gate'][sector]
        nx = (s['barge_pos'][:, 0] + advance.astype(jnp.int32)) % 48
        s['barge_pos'] = s['barge_pos'].at[:, 0].set(nx)
        refill = advance & (nx == 0)
        s['roots'] = jnp.where(refill, 8, s['roots'])
        s['rice'] = jnp.where(refill, 8, s['rice'])
        s['barge_event'] = jnp.where(advance, 2, s['barge_event']).astype(jnp.int32)
        s['customer_pos'] = s['customer_pos'].at[:, 0].set((s['customer_pos'][:, 0] + (t % 4 == 0)) % 48)
        s['deadline'] = s['deadline'] - 1
        expired = s['deadline'] <= 0
        s['deadline'] = jnp.where(expired, 48, s['deadline'])
        s['request'] = jnp.where(expired, 9 - s['request'], s['request'])
        s['customer_event'] = jnp.where(expired, 2, s['customer_event']).astype(jnp.int32)
        s['t'] = t
        s['terminated'] = jnp.array(False)
        s['truncated'] = t >= 768
        s['done'] = s['truncated']
        s = self.refresh(self.display_stations(s))
        return self.observe(s), s, jnp.clip(reward, -4.0, 4.0).astype(jnp.float32), jnp.full(48, s['done'], jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
