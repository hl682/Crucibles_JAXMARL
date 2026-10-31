"""Floodplain Seed Cooperative. Original unexecuted author source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 96
    observation_size = 238
    num_actions = 19
    max_steps = 896

    def nearest(self, p, q, active, k, radius):
        d = jnp.sum(jnp.abs(p[:, None, :] - q[None, :, :]), axis=2)
        ix = jnp.argsort(d + (~active[None, :]) * 10000, axis=1)[:, :k]
        ok = (jnp.take_along_axis(d, ix, axis=1) <= radius) & active[ix]
        return ix.astype(jnp.int32), jnp.any(jax.nn.one_hot(ix, q.shape[0], dtype=jnp.bool_) & ok[:, :, None], axis=1)

    def refresh(self, s):
        s = dict(s)
        s['peer'], s['agent_visible'] = self.nearest(s['pos'], s['pos'], s['active'], 8, 4)
        s['fields'], s['field_visible'] = self.nearest(s['pos'], s['field_pos'], s['field_active'], 4, 5)
        s['near'], s['facility_visible'] = self.nearest(s['pos'], s['facility_pos'], s['facility_active'], 6, 5)
        s['animals'], s['animal_visible'] = self.nearest(s['pos'], s['animal_pos'], s['animal_active'], 2, 4)
        s['trays'], s['tray_visible'] = self.nearest(s['pos'], s['tray_pos'], s['tray_active'], 2, 4)
        h = jnp.arange(96, dtype=jnp.int32) // 4
        kind = s['facility_kind']
        s['facility_stock'] = jnp.where(kind == 0, s['reserve'][h], jnp.where(kind == 1, s['store'][h], jnp.where(kind == 2, s['levee'][h], s['animal_health'][h]))).astype(jnp.int32)
        s['facility_gate'] = (kind == 2) & s['gate'][h]
        return s

    def reset(self, key):
        h = jnp.arange(24, dtype=jnp.int32)
        row = h // 6
        col = jnp.where(row % 2 == 0, h % 6, 5 - h % 6)
        origin = jnp.stack((col * 8, row * 6), axis=1)
        a = jnp.arange(96, dtype=jnp.int32)
        team, role = a // 4, a % 4
        offsets = jnp.array([[3, 3], [4, 4], [5, 3], [5, 4]], jnp.int32)
        fp = (origin[:, None, :] + jnp.array([[4, 3], [4, 1]], jnp.int32)[None, :, :]).reshape((48, 2))
        facilities = (origin[:, None, :] + jnp.array([[1, 3], [2, 1], [6, 3], [6, 1]], jnp.int32)[None, :, :]).reshape((96, 2))
        x, y = jnp.arange(48, dtype=jnp.int32)[None, :], jnp.arange(24, dtype=jnp.int32)[:, None]
        terrain = jnp.where(y % 6 == 5, jnp.where(x % 8 == 6, 2, 1), 0).astype(jnp.int32)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
             'pos': origin[team] + offsets[role], 'active': jnp.ones(96, jnp.bool_), 'team': team, 'tool': role,
             'cargo': jnp.where(role == 0, 1, 0).astype(jnp.int32), 'message': jnp.zeros(96, jnp.int32), 'event': jnp.zeros(96, jnp.int32),
             'field_pos': fp, 'field_active': jnp.ones(48, jnp.bool_), 'field_kind': jnp.arange(48, dtype=jnp.int32) % 2,
             'variety': jax.random.randint(key, (48,), 0, 2, dtype=jnp.int32), 'growth': jnp.where(jnp.arange(48, dtype=jnp.int32) % 6 == 0, 8, 0).astype(jnp.int32),
             'planted': jnp.arange(48, dtype=jnp.int32) % 6 == 0, 'water': jnp.full(48, 3, jnp.int32), 'salt': jnp.zeros(48, jnp.int32),
             'soil': jnp.full(48, 4, jnp.int32), 'care': jnp.where(jnp.arange(48, dtype=jnp.int32) % 6 == 0, 3, 0).astype(jnp.int32),
             'field_event': jnp.zeros(48, jnp.int32), 'facility_pos': facilities, 'facility_active': jnp.ones(96, jnp.bool_),
             'facility_kind': jnp.arange(96, dtype=jnp.int32) % 4, 'facility_stock': jnp.zeros(96, jnp.int32), 'facility_gate': jnp.zeros(96, jnp.bool_),
             'reserve': jnp.full(24, 6, jnp.int32), 'store': jnp.full(24, 3, jnp.int32), 'levee': jnp.full(24, 3, jnp.int32),
             'gate': jnp.zeros(24, jnp.bool_), 'gate_owner': jnp.full(24, -1, jnp.int32),
             'water_pipe': jnp.zeros((24, 3), jnp.int32), 'salt_pipe': jnp.zeros((24, 3), jnp.int32),
             'flow_owner': jnp.full((24, 3), -1, jnp.int32),
             'animal_pos': origin + jnp.array([6, 3], dtype=jnp.int32), 'animal_active': jnp.ones(24, jnp.bool_), 'animal_health': jnp.full(24, 6, jnp.int32),
             'animal_safe': jnp.zeros(24, jnp.bool_), 'animal_event': jnp.zeros(24, jnp.int32),
             'tray_pos': origin + jnp.array([6, 3], dtype=jnp.int32), 'tray_active': jnp.zeros(24, jnp.bool_), 'tray_variety': jnp.zeros(24, jnp.int32),
             'tray_progress': jnp.zeros(24, jnp.int32), 'tray_dest': jnp.minimum(23, h + 1), 'tray_event': jnp.zeros(24, jnp.int32),
             'terrain': terrain, 'peer': jnp.zeros((96, 8), jnp.int32), 'fields': jnp.zeros((96, 4), jnp.int32), 'near': jnp.zeros((96, 6), jnp.int32),
             'animals': jnp.zeros((96, 2), jnp.int32), 'trays': jnp.zeros((96, 2), jnp.int32),
             'agent_visible': jnp.zeros((96, 96), jnp.bool_), 'field_visible': jnp.zeros((96, 48), jnp.bool_),
             'facility_visible': jnp.zeros((96, 96), jnp.bool_), 'animal_visible': jnp.zeros((96, 24), jnp.bool_), 'tray_visible': jnp.zeros((96, 24), jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        yes = jnp.full(96, ~s['done'], jnp.bool_)
        fd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['field_pos'][None, :, :]), axis=2)
        sd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['facility_pos'][None, :, :]), axis=2)
        near = sd <= 1
        field = jnp.any(fd <= 1, axis=1)
        nursery = jnp.any(near & (s['facility_kind'][None, :] == 0), axis=1)
        granary = jnp.any(near & (s['facility_kind'][None, :] == 1), axis=1)
        sluice = jnp.any(near & (s['facility_kind'][None, :] == 2), axis=1)
        td = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['tray_pos'][None, :, :]), axis=2)
        animal = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['animal_pos'][None, :, :]), axis=2)
        ad = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2)
        peer = jnp.any((ad <= 1) & (ad > 0), axis=1)
        seed = (s['cargo'] == 1) | (s['cargo'] == 2)
        return jnp.stack((jnp.ones(96, jnp.bool_), yes, yes, yes, yes,
                          yes & field & seed, yes & field & (s['tool'] <= 1), yes & field & (s['tool'] == 2) & (s['cargo'] == 0),
                          yes & sluice, yes & sluice & (s['cargo'] == 3),
                          yes & (s['cargo'] == 0) & (nursery | jnp.any((td <= 1) & s['tray_active'][None, :], axis=1)),
                          yes & sluice & seed & (s['tool'] == 3), yes & granary & (s['cargo'] == 3),
                          yes & nursery & (s['cargo'] > 0), yes & jnp.any(animal <= 1, axis=1),
                          yes & peer & (s['cargo'] > 0), yes & peer, yes & field, yes), axis=1)

    def observe(self, s):
        a = jnp.arange(96, dtype=jnp.int32)
        own = jnp.stack((s['pos'][:, 0] / 47.0, s['pos'][:, 1] / 23.0, s['team'] / 23.0, s['tool'] / 3.0,
                         s['cargo'] / 3.0, s['message'] / 8.0, s['event'] / 9.0, a / 95.0,
                         jnp.full(96, s['t'] / 896.0), jnp.full(96, (s['t'] % 64) / 63.0)), axis=1)
        pd = jnp.stack((s['pos'][:, 0] / 47.0, s['pos'][:, 1] / 23.0, s['team'] / 23.0, s['tool'] / 3.0,
                        s['cargo'] / 3.0, s['message'] / 8.0, a / 95.0, jnp.ones(96)), axis=1)
        pv = jnp.take_along_axis(s['agent_visible'], s['peer'], axis=1)
        peers = (pd[s['peer']] * pv[:, :, None]).reshape((96, 64))
        q = s['field_pos']
        fd = jnp.stack((q[:, 0] / 47.0, q[:, 1] / 23.0, s['field_kind'], s['variety'], s['growth'] / 8.0,
                        s['planted'], s['water'] / 8.0, s['salt'] / 4.0, s['soil'] / 4.0, s['care'] / 3.0,
                        (jnp.arange(48, dtype=jnp.int32) // 2) / 23.0, jnp.ones(48)), axis=1)
        fv = jnp.take_along_axis(s['field_visible'], s['fields'], axis=1)
        fields = (fd[s['fields']] * fv[:, :, None]).reshape((96, 48))
        q = s['facility_pos']
        sd = jnp.stack((q[:, 0] / 47.0, q[:, 1] / 23.0, s['facility_kind'] / 3.0, s['facility_stock'] / 24.0,
                        s['facility_gate'], (jnp.arange(96, dtype=jnp.int32) // 4) / 23.0, jnp.ones(96)), axis=1)
        sv = jnp.take_along_axis(s['facility_visible'], s['near'], axis=1)
        sites = (sd[s['near']] * sv[:, :, None]).reshape((96, 42))
        q = s['animal_pos']
        ad = jnp.stack((q[:, 0] / 47.0, q[:, 1] / 23.0, s['animal_health'] / 6.0, s['animal_safe'], jnp.arange(24, dtype=jnp.int32) / 23.0, jnp.ones(24)), axis=1)
        av = jnp.take_along_axis(s['animal_visible'], s['animals'], axis=1)
        animals = (ad[s['animals']] * av[:, :, None]).reshape((96, 12))
        q = s['tray_pos']
        td = jnp.stack((q[:, 0] / 47.0, q[:, 1] / 23.0, s['tray_variety'], s['tray_progress'] / 3.0, s['tray_dest'] / 23.0, jnp.ones(24)), axis=1)
        tv = jnp.take_along_axis(s['tray_visible'], s['trays'], axis=1)
        trays = (td[s['trays']] * tv[:, :, None]).reshape((96, 12))
        return jnp.concatenate((own, peers, fields, sites, animals, trays, self.legal_actions(s).astype(jnp.float32), jnp.zeros((96, 31), jnp.float32)), axis=1).astype(jnp.float32)

    def absorb(self, args):
        key, s, actions = args
        return self.observe(s), s, jnp.zeros(96, jnp.float32), jnp.ones(96, jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state['done'], self.absorb, self.advance, (key, state, actions))

    def advance(self, args):
        key, s, actions = args
        s = dict(s)
        z = jnp.clip(actions, 0, 18)
        actions = jnp.where(self.legal_actions(s)[jnp.arange(96, dtype=jnp.int32), z] & (actions >= 0) & (actions < 19), z, 0)
        original = s['pos']
        s['event'] = jnp.zeros(96, jnp.int32)
        s['field_event'] = jnp.zeros(48, jnp.int32)
        s['animal_event'] = jnp.zeros(24, jnp.int32)
        s['tray_event'] = jnp.zeros(24, jnp.int32)
        s['message'] = jnp.maximum(0, s['message'] - 1)
        reward = jnp.zeros(96, jnp.float32)
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

        def actor(k, carry):
            w, r = carry
            w = dict(w)
            i = (k + s['t']) % 96
            a = actions[i]
            p = w['pos'][i]
            target = jnp.clip(p + dirs[jnp.clip(a, 0, 4)], jnp.array([0, 0], dtype=jnp.int32), jnp.array([47, 23], dtype=jnp.int32))
            tile = w['terrain'][target[1], target[0]]
            row, col = target[1] // 6, target[0] // 8
            cell = row * 6 + jnp.where(row % 2 == 0, col, 5 - col)
            passable = (tile == 0) | ((tile == 2) & (w['levee'][cell] > 0)) | (w['tool'][i] == 3)
            free = ~jnp.any(jnp.all(w['pos'] == target, axis=1) & (jnp.arange(96, dtype=jnp.int32) != i))
            move = (a >= 1) & (a <= 4) & free & passable & jnp.any(target != p)
            w['pos'] = w['pos'].at[i].set(jnp.where(move, target, p))
            fd = jnp.sum(jnp.abs(w['field_pos'] - p), axis=1)
            f = jnp.argmin(fd)
            close = fd[f] <= 1
            sow = (a == 5) & close & ~w['planted'][f] & ((w['cargo'][i] == 1) | (w['cargo'][i] == 2))
            w['variety'] = w['variety'].at[f].set(jnp.where(sow, w['cargo'][i] - 1, w['variety'][f]))
            w['planted'] = w['planted'].at[f].set(w['planted'][f] | sow)
            w['growth'] = w['growth'].at[f].set(jnp.where(sow, 0, w['growth'][f]))
            w['care'] = w['care'].at[f].set(jnp.where(sow, 0, w['care'][f]))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(sow, 0, w['cargo'][i]))
            care = (a == 6) & close & (w['tool'][i] <= 1)
            bit = jnp.where(w['tool'][i] == 0, 1, 2)
            w['care'] = w['care'].at[f].set(jnp.where(care, w['care'][f] | bit, w['care'][f]))
            w['water'] = w['water'].at[f].set(jnp.minimum(8, w['water'][f] + (care & (w['tool'][i] == 1))))
            assist = jnp.any((jnp.sum(jnp.abs(original - w['field_pos'][f]), axis=1) <= 2) & (actions == 18) & (jnp.arange(96, dtype=jnp.int32) != i))
            harvest = (a == 7) & close & w['planted'][f] & (w['growth'][f] >= 8) & (w['tool'][i] == 2) & (w['cargo'][i] == 0) & assist
            w['cargo'] = w['cargo'].at[i].set(jnp.where(harvest, 3, w['cargo'][i]))
            w['growth'] = w['growth'].at[f].set(jnp.where(harvest, 0, w['growth'][f]))
            w['planted'] = w['planted'].at[f].set(w['planted'][f] & ~harvest)
            w['soil'] = w['soil'].at[f].set(jnp.maximum(0, w['soil'][f] - harvest))
            w['field_event'] = w['field_event'].at[f].set(jnp.where(harvest, 3, jnp.where(sow, 1, jnp.where(care, 2, w['field_event'][f]))))
            sd = jnp.sum(jnp.abs(w['facility_pos'] - p), axis=1)
            sluices = (sd <= 1) & (w['facility_kind'] == 2)
            c = jnp.argmax(sluices) // 4
            gate = (a == 8) & jnp.any(sluices)
            w['gate'] = w['gate'].at[c].set(jnp.where(gate, ~w['gate'][c], w['gate'][c]))
            w['gate_owner'] = w['gate_owner'].at[c].set(jnp.where(gate, i, w['gate_owner'][c]))
            repair = (a == 9) & jnp.any(sluices) & (w['cargo'][i] == 3) & (w['levee'][c] < 6)
            w['levee'] = w['levee'].at[c].set(jnp.minimum(6, w['levee'][c] + repair * 3))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(repair, 0, w['cargo'][i]))
            launch = (a == 11) & jnp.any(sluices) & ((w['cargo'][i] == 1) | (w['cargo'][i] == 2)) & (w['tool'][i] == 3) & ~w['tray_active'][c] & (c < 23)
            w['tray_variety'] = w['tray_variety'].at[c].set(jnp.where(launch, w['cargo'][i] - 1, w['tray_variety'][c]))
            w['tray_active'] = w['tray_active'].at[c].set(w['tray_active'][c] | launch)
            w['tray_progress'] = w['tray_progress'].at[c].set(jnp.where(launch, 0, w['tray_progress'][c]))
            w['tray_pos'] = w['tray_pos'].at[c].set(jnp.where(launch, w['facility_pos'][c * 4 + 2], w['tray_pos'][c]))
            w['tray_event'] = w['tray_event'].at[c].set(jnp.where(launch, 1, w['tray_event'][c]))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(launch, 0, w['cargo'][i]))
            td = jnp.sum(jnp.abs(w['tray_pos'] - p), axis=1)
            available = (td <= 1) & w['tray_active']
            tr = jnp.argmax(available)
            pick = (a == 10) & jnp.any(available) & (w['cargo'][i] == 0)
            w['cargo'] = w['cargo'].at[i].set(jnp.where(pick, w['tray_variety'][tr] + 1, w['cargo'][i]))
            w['tray_active'] = w['tray_active'].at[tr].set(w['tray_active'][tr] & ~pick)
            w['tray_event'] = w['tray_event'].at[tr].set(jnp.where(pick, 3, w['tray_event'][tr]))
            nurseries = (sd <= 1) & (w['facility_kind'] == 0)
            c = jnp.argmax(nurseries) // 4
            withdraw = (a == 10) & ~pick & jnp.any(nurseries) & (w['cargo'][i] == 0) & (w['reserve'][c] > 0)
            w['reserve'] = w['reserve'].at[c].add(-withdraw.astype(jnp.int32))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(withdraw, 1 + (s['t'] // 16 + c) % 2, w['cargo'][i]))
            donate = (a == 13) & jnp.any(nurseries) & (w['cargo'][i] > 0) & (w['reserve'][c] < 24)
            donation_harvest = donate & (w['cargo'][i] == 3)
            amount = jnp.where(w['cargo'][i] == 3, 3, 1)
            w['reserve'] = w['reserve'].at[c].set(jnp.minimum(24, w['reserve'][c] + donate * amount))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(donate, 0, w['cargo'][i]))
            granaries = (sd <= 1) & (w['facility_kind'] == 1)
            c = jnp.argmax(granaries) // 4
            store = (a == 12) & jnp.any(granaries) & (w['cargo'][i] == 3) & (w['store'][c] < 24)
            w['store'] = w['store'].at[c].set(jnp.minimum(24, w['store'][c] + store * 3))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(store, 0, w['cargo'][i]))
            r = r + store * (w['team'] == c).astype(jnp.float32) * 0.025
            ad = jnp.sum(jnp.abs(w['animal_pos'] - p), axis=1)
            animal = jnp.argmin(ad + (~w['animal_active']) * 1000)
            herd = (a == 14) & (ad[animal] <= 1) & w['animal_active'][animal]
            barn = w['facility_pos'][animal * 4 + 3]
            delta = barn - w['animal_pos'][animal]
            shift = jnp.where(delta[0] != 0, jnp.array([jnp.sign(delta[0]), 0], dtype=jnp.int32), jnp.array([0, jnp.sign(delta[1])], dtype=jnp.int32))
            w['animal_pos'] = w['animal_pos'].at[animal].set(w['animal_pos'][animal] + herd * shift)
            w['animal_event'] = w['animal_event'].at[animal].set(jnp.where(herd, 1, w['animal_event'][animal]))
            peer_dist = jnp.sum(jnp.abs(w['pos'] - p), axis=1)
            peers = (peer_dist <= 1) & (jnp.arange(96, dtype=jnp.int32) != i) & (actions == 0)
            empty = peers & (w['cargo'] == 0)
            j = jnp.argmax(empty)
            hand = (a == 15) & jnp.any(empty) & (w['cargo'][i] > 0)
            w['cargo'] = w['cargo'].at[j].set(jnp.where(hand, w['cargo'][i], w['cargo'][j])).at[i].set(jnp.where(hand, 0, w['cargo'][i]))
            j = jnp.argmax(peers)
            swap = (a == 16) & jnp.any(peers)
            ti, tj = w['tool'][i], w['tool'][j]
            w['tool'] = w['tool'].at[i].set(jnp.where(swap, tj, ti)).at[j].set(jnp.where(swap, ti, tj))
            signal = (a == 17) & close
            w['message'] = jnp.where((peer_dist <= 4) & signal, w['water'][f], w['message'])
            ev = jnp.where(sow, 1, jnp.where(care, 2, jnp.where(harvest, 3, jnp.where(gate | repair, 4, jnp.where(pick | withdraw | launch, 5, jnp.where(store, 6, jnp.where(donate, 7, jnp.where(herd, 8, jnp.where(hand | swap | signal, 9, 0)))))))))
            w['event'] = w['event'].at[i].set(ev)
            r = r.at[i].add(-0.002 * (a != 0) + 0.08 * harvest + 0.15 * store + 0.06 * donation_harvest)
            return w, r

        s, reward = jax.lax.fori_loop(0, 96, actor, (s, reward))
        t = s['t'] + 1
        flood = (t % 64 >= 40) & (t % 64 < 48)
        rain = (t % 8 == 0) & jax.random.bernoulli(key, 0.5, (48,))
        incoming = s['water_pipe'][:, 0]
        salt_in = s['salt_pipe'][:, 0]
        s['water'] = jnp.clip(s['water'] + rain - (t % 8 == 0), 0, 8).astype(jnp.int32)
        s['water'] = s['water'].at[::2].set(jnp.clip(s['water'][::2] + incoming + flood * (s['levee'] < 4), 0, 8))
        s['salt'] = s['salt'].at[::2].set(jnp.clip(s['salt'][::2] + salt_in + flood * (s['levee'] < 2), 0, 4))
        wet = s['water'][::2]
        flow = jnp.where(s['gate'] & (wet >= 2), 2, 0).astype(jnp.int32)
        carried_salt = jnp.where(flow > 0, jnp.minimum(1, s['salt'][::2]), 0).astype(jnp.int32)
        s['water'] = s['water'].at[::2].add(-flow)
        s['salt'] = s['salt'].at[::2].add(-carried_salt)
        bad = s['planted'] & ((s['water'] > 6) | (s['salt'] > 2))
        charged = (incoming > 0) & bad[::2] & (s['flow_owner'][:, 0] >= 0)
        owners = jnp.maximum(0, s['flow_owner'][:, 0])
        reward = reward.at[owners].add(-0.03 * charged.astype(jnp.float32))
        s['water_pipe'] = jnp.concatenate((s['water_pipe'][:, 1:], jnp.concatenate((jnp.zeros(1, jnp.int32), flow[:-1]))[:, None]), axis=1)
        s['salt_pipe'] = jnp.concatenate((s['salt_pipe'][:, 1:], jnp.concatenate((jnp.zeros(1, jnp.int32), carried_salt[:-1]))[:, None]), axis=1)
        new_owner = jnp.concatenate((jnp.full(1, -1, jnp.int32), s['gate_owner'][:-1]))
        s['flow_owner'] = jnp.concatenate((s['flow_owner'][:, 1:], new_owner[:, None]), axis=1)
        growing = s['planted'] & (s['care'] == 3) & (s['soil'] > 0) & (s['salt'] <= jnp.where(s['variety'] == 0, 2, 1))
        growing = growing & jnp.where(s['variety'] == 0, s['water'] >= 3, (s['water'] >= 1) & (s['water'] <= 3))
        growth_tick = (t % 4 == 0) & ((s['field_kind'] == 0) | (t % 8 == 0))
        s['growth'] = jnp.clip(s['growth'] + (growing & growth_tick) - bad, 0, 8).astype(jnp.int32)
        s['field_event'] = jnp.where(bad, 4, s['field_event']).astype(jnp.int32)
        reward = reward - bad[(s['team'] * 2)] * 0.005
        s['salt'] = jnp.maximum(0, s['salt'] - (t % 16 == 0)).astype(jnp.int32)
        s['animal_safe'] = jnp.all(s['animal_pos'] == s['facility_pos'][3::4], axis=1)
        animal_hurt = flood & ~s['animal_safe'] & (s['water'][::2] > 5) & s['animal_active']
        s['animal_health'] = jnp.maximum(0, s['animal_health'] - animal_hurt).astype(jnp.int32)
        s['animal_active'] = s['animal_health'] > 0
        s['animal_event'] = jnp.where(animal_hurt, 2, s['animal_event']).astype(jnp.int32)
        soil_bonus = jnp.repeat(s['animal_safe'] & s['animal_active'], 2)
        s['soil'] = jnp.minimum(4, s['soil'] + ((~s['planted'] | soil_bonus) & (t % 16 == 0))).astype(jnp.int32)
        s['levee'] = jnp.maximum(0, s['levee'] - (t % 64 == 47)).astype(jnp.int32)
        drift = s['tray_active'] & s['gate'] & (s['tray_progress'] < 3) & (wet >= 2)
        s['tray_progress'] = jnp.minimum(3, s['tray_progress'] + drift).astype(jnp.int32)
        start = s['facility_pos'][2::4]
        end = s['facility_pos'][s['tray_dest'] * 4]
        s['tray_pos'] = jnp.where(s['tray_active'][:, None], start + (end - start) * s['tray_progress'][:, None] // 3, s['tray_pos']).astype(jnp.int32)
        s['tray_event'] = jnp.where(drift & (s['tray_progress'] == 3), 2, s['tray_event']).astype(jnp.int32)
        eat = t % 32 == 0
        hungry = eat & (s['store'] == 0)
        reward = reward - hungry[s['team']] * 0.03
        s['store'] = jnp.maximum(0, s['store'] - eat).astype(jnp.int32)
        s['t'] = t
        s['terminated'] = jnp.array(False)
        s['truncated'] = t >= 896
        s['done'] = s['truncated']
        s = self.refresh(s)
        return self.observe(s), s, jnp.clip(reward, -4.0, 4.0).astype(jnp.float32), jnp.full(96, s['done'], jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
