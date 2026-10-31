"""Railway Commons Republics. Original graph-geometric JAX author source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 256
    observation_size = 231
    num_actions = 19
    max_steps = 1024

    def nearest(self, p, q, active, k, radius):
        d = jnp.sum(jnp.abs(p[:, None, :] - q[None, :, :]), axis=2)
        ix = jnp.argsort(d + (~active[None, :]) * 10000, axis=1)[:, :k]
        ok = (jnp.take_along_axis(d, ix, axis=1) <= radius) & active[ix]
        return ix.astype(jnp.int32), jnp.any(jax.nn.one_hot(ix, q.shape[0], dtype=jnp.bool_) & ok[:, :, None], axis=1)

    def refresh(self, s):
        s = dict(s)
        s['peer'], s['agent_visible'] = self.nearest(s['pos'], s['pos'], s['active'], 8, 4)
        s['near'], s['site_visible'] = self.nearest(s['pos'], s['site_pos'], s['site_active'], 6, 6)
        s['edges'], s['edge_visible'] = self.nearest(s['pos'], s['edge_pos'], s['edge_active'], 4, 9)
        s['trains'], s['train_visible'] = self.nearest(s['pos'], s['train_pos'], s['train_active'], 2, 6)
        c = jnp.arange(48, dtype=jnp.int32) // 3
        k = s['site_kind']
        s['site_stock'] = jnp.where(k == 0, s['power'][c], jnp.where(k == 1, s['ore'][c], s['grain'][c])).astype(jnp.int32)
        s['site_route'] = jnp.where(k == 0, s['switch'][c], 0).astype(jnp.int32)
        s['site_permit'] = jnp.where(k == 0, s['permit'][c] + 1, 0).astype(jnp.int32)
        s['site_red'] = (k == 0) & s['red'][c]
        s['site_timer'] = jnp.where(k == 0, s['commit'][c], 0).astype(jnp.int32)
        return s

    def reset(self, key):
        ids = jnp.arange(256, dtype=jnp.int32)
        team, local = ids // 16, ids % 16
        origin = jnp.stack(((team % 4) * 16, (team // 4) * 8), axis=1)
        pos = origin + jnp.stack((local % 4 + 5, local // 4), axis=1)
        pos = jnp.where((local == 0)[:, None], origin + jnp.array([2, 2], dtype=jnp.int32), pos)
        pos = jnp.where((local == 1)[:, None], origin + jnp.array([15, 4], dtype=jnp.int32), pos)
        pos = jnp.where((local == 3)[:, None], origin + jnp.array([8, 5], dtype=jnp.int32), pos)
        nodes = jnp.stack(((jnp.arange(16, dtype=jnp.int32) % 4) * 16 + 8, (jnp.arange(16, dtype=jnp.int32) // 4) * 8 + 4), axis=1).astype(jnp.int32)
        ha = (jnp.arange(12, dtype=jnp.int32) // 3) * 4 + jnp.arange(12, dtype=jnp.int32) % 3
        va = jnp.arange(12, dtype=jnp.int32)
        ea = jnp.concatenate((ha, va)).astype(jnp.int32)
        eb = jnp.concatenate((ha + 1, va + 4)).astype(jnp.int32)
        sp = (nodes[:, None, :] + jnp.array([[0, 0], [-6, -2], [4, 2]], jnp.int32)[None, :, :]).reshape((48, 2))
        tn = jnp.arange(8, dtype=jnp.int32) * 2
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
             'pos': pos, 'active': jnp.ones(256, jnp.bool_), 'team': team, 'tool': local % 6,
             'cargo': jnp.where(local == 1, 1, jnp.where(local == 4, 2, 0)).astype(jnp.int32),
             'message': jnp.zeros(256, jnp.int32), 'event': jnp.zeros(256, jnp.int32),
             'node_pos': nodes, 'ore': jnp.full(16, 12, jnp.int32), 'grain': jnp.full(16, 8, jnp.int32),
             'power': jnp.full(16, 4, jnp.int32), 'switch': jnp.zeros(16, jnp.int32), 'red': jnp.zeros(16, jnp.bool_),
             'commit': jnp.zeros(16, jnp.int32), 'permit': jnp.full(16, -1, jnp.int32), 'received': jnp.zeros(16, jnp.int32),
             'site_pos': sp, 'site_active': jnp.ones(48, jnp.bool_), 'site_kind': jnp.arange(48, dtype=jnp.int32) % 3,
             'site_stock': jnp.zeros(48, jnp.int32), 'site_route': jnp.zeros(48, jnp.int32), 'site_permit': jnp.zeros(48, jnp.int32),
             'site_red': jnp.zeros(48, jnp.bool_), 'site_timer': jnp.zeros(48, jnp.int32),
             'edge_a': ea, 'edge_b': eb, 'edge_pos': (nodes[ea] + nodes[eb]) // 2, 'edge_active': jnp.ones(24, jnp.bool_),
             'track': jnp.where(jnp.arange(24, dtype=jnp.int32) < 12, 1, 2).astype(jnp.int32), 'bridge': jnp.full(24, 6, jnp.int32),
             'edge_event': jnp.zeros(24, jnp.int32), 'edge_builder': jnp.full(24, -1, jnp.int32),
             'train_pos': nodes[tn], 'train_active': jnp.ones(8, jnp.bool_), 'train_node': tn, 'train_dest': tn,
             'train_edge': jnp.full(8, -1, jnp.int32), 'progress': jnp.zeros(8, jnp.int32), 'speed': jnp.zeros(8, jnp.int32),
             'train_kind': jnp.full(8, 2, jnp.int32), 'train_load': jnp.ones(8, jnp.int32), 'train_owner': tn,
             'train_health': jnp.full(8, 4, jnp.int32), 'train_event': jnp.zeros(8, jnp.int32),
             'terrain': ((((jnp.arange(32, dtype=jnp.int32)[:, None] % 8 == 4) & (jnp.arange(64, dtype=jnp.int32)[None, :] >= 8) & (jnp.arange(64, dtype=jnp.int32)[None, :] <= 56)) | ((jnp.arange(64, dtype=jnp.int32)[None, :] % 16 == 8) & (jnp.arange(32, dtype=jnp.int32)[:, None] >= 4) & (jnp.arange(32, dtype=jnp.int32)[:, None] <= 28))).astype(jnp.int32)), 'peer': jnp.zeros((256, 8), jnp.int32), 'near': jnp.zeros((256, 6), jnp.int32),
             'edges': jnp.zeros((256, 4), jnp.int32), 'trains': jnp.zeros((256, 2), jnp.int32),
             'agent_visible': jnp.zeros((256, 256), jnp.bool_), 'site_visible': jnp.zeros((256, 48), jnp.bool_),
             'edge_visible': jnp.zeros((256, 24), jnp.bool_), 'train_visible': jnp.zeros((256, 8), jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        yes = jnp.full(256, ~s['done'], jnp.bool_)
        sd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['site_pos'][None, :, :]), axis=2)
        nd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['node_pos'][None, :, :]), axis=2)
        ed = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['edge_pos'][None, :, :]), axis=2)
        td = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['train_pos'][None, :, :]), axis=2)
        ad = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2)
        node = jnp.any(nd <= 1, axis=1)
        peer = jnp.any((ad <= 1) & (ad > 0), axis=1)
        mine = jnp.any((sd <= 1) & (s['site_kind'][None, :] == 1), axis=1) & (s['tool'] == 0)
        farm = jnp.any((sd <= 1) & (s['site_kind'][None, :] == 2), axis=1) & (s['tool'] == 4)
        train = jnp.any(td <= 1, axis=1)
        return jnp.stack((jnp.ones(256, jnp.bool_), yes, yes, yes, yes,
                          yes & (s['cargo'] == 0) & (mine | farm), yes & jnp.any(ed <= 1, axis=1) & (s['cargo'] == 1),
                          yes & node & (s['tool'] == 2), yes & node & (s['tool'] == 2),
                          yes & train & (s['cargo'] > 0), yes & train & node, yes & node, yes & node,
                          yes & node & (s['cargo'] == 1), yes & peer & (s['cargo'] > 0), yes & peer,
                          yes & node, yes & node & (s['cargo'] == 2), yes & train & (s['cargo'] == 1) & (s['tool'] == 5)), axis=1)

    def observe(self, s):
        ids = jnp.arange(256, dtype=jnp.int32)
        own = jnp.stack((s['pos'][:, 0] / 63.0, s['pos'][:, 1] / 31.0, s['team'] / 15.0, s['tool'] / 5.0,
                         s['cargo'] / 2.0, s['message'] / 6.0, s['event'] / 10.0, ids / 255.0,
                         jnp.full(256, s['t'] / 1024.0), jnp.full(256, (s['t'] % 32) / 31.0)), axis=1)
        pd = jnp.stack((s['pos'][:, 0] / 63.0, s['pos'][:, 1] / 31.0, s['team'] / 15.0, s['tool'] / 5.0,
                        s['cargo'] / 2.0, s['message'] / 6.0, ids / 255.0, jnp.ones(256)), axis=1)
        pv = jnp.take_along_axis(s['agent_visible'], s['peer'], axis=1)
        peers = (pd[s['peer']] * pv[:, :, None]).reshape((256, 64))
        q = s['site_pos']
        sd = jnp.stack((q[:, 0] / 63.0, q[:, 1] / 31.0, s['site_kind'] / 2.0, s['site_stock'] / 24.0,
                        s['site_route'] / 3.0, s['site_permit'] / 16.0, s['site_red'].astype(jnp.float32), s['site_timer'] / 6.0,
                        (jnp.arange(48, dtype=jnp.int32) // 3) / 15.0, jnp.ones(48)), axis=1)
        sv = jnp.take_along_axis(s['site_visible'], s['near'], axis=1)
        sites = (sd[s['near']] * sv[:, :, None]).reshape((256, 60))
        q = s['edge_pos']
        ed = jnp.stack((q[:, 0] / 63.0, q[:, 1] / 31.0, s['edge_a'] / 15.0, s['edge_b'] / 15.0,
                        s['track'] / 3.0, s['bridge'] / 6.0, (s['edge_builder'] + 1) / 256.0, jnp.ones(24)), axis=1)
        ev = jnp.take_along_axis(s['edge_visible'], s['edges'], axis=1)
        edges = (ed[s['edges']] * ev[:, :, None]).reshape((256, 32))
        q = s['train_pos']
        td = jnp.stack((q[:, 0] / 63.0, q[:, 1] / 31.0, s['train_kind'] / 2.0, s['train_load'] / 4.0,
                        s['train_owner'] / 15.0, s['train_health'] / 4.0, s['speed'] / 2.0, s['train_dest'] / 15.0,
                        s['progress'] / 16.0, jnp.ones(8)), axis=1)
        tv = jnp.take_along_axis(s['train_visible'], s['trains'], axis=1)
        trains = (td[s['trains']] * tv[:, :, None]).reshape((256, 20))
        return jnp.concatenate((own, peers, sites, edges, trains, self.legal_actions(s).astype(jnp.float32), jnp.zeros((256, 26), jnp.float32)), axis=1).astype(jnp.float32)

    def absorb(self, args):
        key, s, actions = args
        return self.observe(s), s, jnp.zeros(256, jnp.float32), jnp.ones(256, jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state['done'], self.absorb, self.advance, (key, state, actions))

    def advance(self, args):
        key, s, actions = args
        s = dict(s)
        z = jnp.clip(actions, 0, 18)
        actions = jnp.where(self.legal_actions(s)[jnp.arange(256, dtype=jnp.int32), z] & (actions >= 0) & (actions < 19), z, 0)
        s['event'] = jnp.zeros(256, jnp.int32)
        s['edge_event'] = jnp.zeros(24, jnp.int32)
        s['train_event'] = jnp.zeros(8, jnp.int32)
        s['message'] = jnp.maximum(0, s['message'] - 1)
        s['commit'] = jnp.maximum(0, s['commit'] - 1)
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        rewards = jnp.zeros(256, jnp.float32)

        def actor(k, carry):
            w, r = carry
            w = dict(w)
            i = (k + s['t']) % 256
            a = actions[i]
            p = w['pos'][i]
            target = jnp.clip(p + dirs[jnp.clip(a, 0, 4)], jnp.array([0, 0], dtype=jnp.int32), jnp.array([63, 31], dtype=jnp.int32))
            free = ~jnp.any(jnp.all(w['pos'] == target, axis=1) & (jnp.arange(256, dtype=jnp.int32) != i))
            move = (a >= 1) & (a <= 4) & free & jnp.any(target != p)
            w['pos'] = w['pos'].at[i].set(jnp.where(move, target, p))
            sd = jnp.sum(jnp.abs(w['site_pos'] - p), axis=1)
            kind = jnp.where(w['tool'][i] == 0, 1, 2)
            available = (sd <= 1) & (w['site_kind'] == kind)
            c = jnp.argmax(available) // 3
            stock = jnp.where(kind == 1, w['ore'][c], w['grain'][c])
            gather = (a == 5) & jnp.any(available) & (stock > 0) & (w['cargo'][i] == 0)
            w['ore'] = w['ore'].at[c].add(-(gather & (kind == 1)).astype(jnp.int32))
            w['grain'] = w['grain'].at[c].add(-(gather & (kind == 2)).astype(jnp.int32))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(gather, kind, w['cargo'][i]))
            ed = jnp.sum(jnp.abs(w['edge_pos'] - p), axis=1)
            e = jnp.argmin(ed)
            build = (a == 6) & (ed[e] <= 1) & (w['cargo'][i] == 1)
            w['track'] = w['track'].at[e].set(jnp.minimum(3, w['track'][e] + build))
            w['bridge'] = w['bridge'].at[e].set(jnp.minimum(6, w['bridge'][e] + build * 3))
            w['edge_builder'] = w['edge_builder'].at[e].set(jnp.where(build, i, w['edge_builder'][e]))
            w['edge_event'] = w['edge_event'].at[e].set(jnp.where(build, 1, w['edge_event'][e]))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(build, 0, w['cargo'][i]))
            nd = jnp.sum(jnp.abs(w['node_pos'] - p), axis=1)
            c = jnp.argmin(nd)
            station = nd[c] <= 1
            switch = (a == 7) & station & (w['commit'][c] == 0) & (w['tool'][i] == 2)
            w['switch'] = w['switch'].at[c].set(jnp.where(switch, (w['switch'][c] + 1) % 4, w['switch'][c]))
            w['commit'] = w['commit'].at[c].set(jnp.where(switch, 6, w['commit'][c]))
            signal = (a == 8) & station & (w['tool'][i] == 2)
            w['red'] = w['red'].at[c].set(jnp.where(signal, ~w['red'][c], w['red'][c]))
            td = jnp.sum(jnp.abs(w['train_pos'] - p), axis=1)
            parked = (td <= 1) & (w['train_edge'] < 0) & (w['train_health'] > 0)
            b = jnp.argmax(parked)
            load = (a == 9) & jnp.any(parked) & (w['cargo'][i] > 0) & (w['train_load'][b] < 4) & ((w['train_load'][b] == 0) | ((w['train_kind'][b] == w['cargo'][i]) & (w['train_owner'][b] == w['team'][i])))
            w['train_owner'] = w['train_owner'].at[b].set(jnp.where(load & (w['train_load'][b] == 0), w['team'][i], w['train_owner'][b]))
            w['train_kind'] = w['train_kind'].at[b].set(jnp.where(load, w['cargo'][i], w['train_kind'][b]))
            w['train_load'] = w['train_load'].at[b].add(load.astype(jnp.int32))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(load, 0, w['cargo'][i]))
            unload = (a == 10) & station & jnp.any(parked) & (w['train_node'][b] == c) & (w['train_owner'][b] != c) & (w['train_load'][b] > 0)
            w['ore'] = w['ore'].at[c].set(jnp.minimum(24, w['ore'][c] + (unload & (w['train_kind'][b] == 1)) * 2))
            w['grain'] = w['grain'].at[c].set(jnp.minimum(24, w['grain'][c] + (unload & (w['train_kind'][b] == 2)) * 2))
            w['train_load'] = w['train_load'].at[b].add(-unload.astype(jnp.int32))
            w['received'] = w['received'].at[c].set(jnp.minimum(1024, w['received'][c] + unload))
            r = r + unload * ((w['team'] == c) * 0.04 + (w['team'] == w['train_owner'][b]) * 0.02)
            public = (a == 11) & station
            claim = (a == 12) & station & (w['team'][i] == c)
            w['permit'] = w['permit'].at[c].set(jnp.where(public, -1, jnp.where(claim, w['team'][i], w['permit'][c])))
            power = (a == 13) & station & (w['cargo'][i] == 1) & (w['power'][c] < 8)
            w['power'] = w['power'].at[c].set(jnp.minimum(8, w['power'][c] + power * 3))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(power, 0, w['cargo'][i]))
            ad = jnp.sum(jnp.abs(w['pos'] - p), axis=1)
            peers = (ad <= 1) & (jnp.arange(256, dtype=jnp.int32) != i) & (actions == 0)
            empty = peers & (w['cargo'] == 0)
            j = jnp.argmax(empty)
            hand = (a == 14) & jnp.any(empty) & (w['cargo'][i] > 0)
            w['cargo'] = w['cargo'].at[j].set(jnp.where(hand, w['cargo'][i], w['cargo'][j])).at[i].set(jnp.where(hand, 0, w['cargo'][i]))
            j = jnp.argmax(peers)
            swap = (a == 15) & jnp.any(peers)
            ti, tj = w['tool'][i], w['tool'][j]
            w['tool'] = w['tool'].at[i].set(jnp.where(swap, tj, ti)).at[j].set(jnp.where(swap, ti, tj))
            bulletin = (a == 16) & station
            w['message'] = jnp.where((ad <= 4) & bulletin, 2 + w['switch'][c] + w['red'][c], w['message']).astype(jnp.int32)
            feed = (a == 17) & station & (w['cargo'][i] == 2) & (w['grain'][c] < 24)
            w['grain'] = w['grain'].at[c].set(jnp.minimum(24, w['grain'][c] + feed * 2))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(feed, 0, w['cargo'][i]))
            b = jnp.argmin(td)
            repair = (a == 18) & (td[b] <= 1) & (w['cargo'][i] == 1) & (w['tool'][i] == 5) & (w['train_health'][b] < 4)
            w['train_health'] = w['train_health'].at[b].set(jnp.minimum(4, w['train_health'][b] + repair * 2))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(repair, 0, w['cargo'][i]))
            event = jnp.where(gather, 1, jnp.where(build, 2, jnp.where(switch | signal, 3, jnp.where(load, 4, jnp.where(unload, 5, jnp.where(public | claim, 6, jnp.where(power, 7, jnp.where(hand | swap, 8, jnp.where(feed, 9, jnp.where(repair, 10, 0))))))))))
            w['event'] = w['event'].at[i].set(event)
            r = r.at[i].add(-0.001 * (a != 0) + 0.15 * unload)
            return w, r

        s, rewards = jax.lax.fori_loop(0, 256, actor, (s, rewards))
        routes = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], jnp.int32)

        def train(b, carry):
            w, r = carry
            w = dict(w)
            node = w['train_node'][b]
            other = jnp.where(w['edge_a'] == node, w['edge_b'], w['edge_a'])
            connected = (w['edge_a'] == node) | (w['edge_b'] == node)
            direction = jnp.sign(w['node_pos'][other] - w['node_pos'][node])
            choices = connected & jnp.all(direction == routes[w['switch'][node]], axis=1)
            e = jnp.argmax(choices)
            dest = other[e]
            permission = (w['permit'][dest] < 0) | (w['permit'][dest] == w['train_owner'][b])
            occupied = jnp.any((w['train_edge'] == e) & (jnp.arange(8, dtype=jnp.int32) != b))
            depart = (w['train_edge'][b] < 0) & (w['train_health'][b] > 0) & jnp.any(choices) & (w['track'][e] >= 2) & (w['bridge'][e] > 0) & (w['power'][node] > 0) & ~w['red'][node] & permission & ~occupied
            w['train_edge'] = w['train_edge'].at[b].set(jnp.where(depart, e, w['train_edge'][b]))
            w['train_dest'] = w['train_dest'].at[b].set(jnp.where(depart, dest, w['train_dest'][b]))
            w['progress'] = w['progress'].at[b].set(jnp.where(depart, 0, w['progress'][b]))
            w['power'] = w['power'].at[node].add(-depart.astype(jnp.int32))
            moving = (w['train_edge'][b] >= 0) & (w['train_health'][b] > 0)
            e = jnp.maximum(0, w['train_edge'][b])
            dest = w['train_dest'][b]
            start, end = w['node_pos'][node], w['node_pos'][dest]
            length = jnp.maximum(1, jnp.sum(jnp.abs(end - start)))
            remaining = length - w['progress'][b]
            brake = w['red'][dest] & (remaining <= 3)
            speed = jnp.where(brake, jnp.maximum(0, w['speed'][b] - 1), jnp.minimum(2, w['speed'][b] + 1))
            speed = jnp.where(moving, speed, 0).astype(jnp.int32)
            damaged = moving & (w['bridge'][e] == 0) & (speed > 0)
            speed = jnp.where(damaged, 0, speed)
            progress = jnp.where(moving, jnp.minimum(length, w['progress'][b] + speed), w['progress'][b])
            arrive = moving & (progress >= length)
            overrun = arrive & w['red'][dest]
            w['train_health'] = w['train_health'].at[b].set(jnp.maximum(0, w['train_health'][b] - damaged * 2 - overrun))
            w['train_load'] = w['train_load'].at[b].set(jnp.maximum(0, w['train_load'][b] - damaged - overrun))
            w['train_pos'] = w['train_pos'].at[b].set(jnp.where(moving, start + (end - start) * progress // length, w['train_pos'][b]))
            w['progress'] = w['progress'].at[b].set(jnp.where(arrive, 0, progress))
            w['speed'] = w['speed'].at[b].set(jnp.where(arrive, 0, speed))
            w['train_edge'] = w['train_edge'].at[b].set(jnp.where(arrive, -1, w['train_edge'][b]))
            w['train_node'] = w['train_node'].at[b].set(jnp.where(arrive, dest, node))
            w['train_event'] = w['train_event'].at[b].set(jnp.where(damaged, 4, jnp.where(overrun, 5, jnp.where(arrive, 3, jnp.where(depart, 1, jnp.where(moving & (speed == 0), 2, 0))))))
            r = r - (damaged | overrun) * (w['team'] == w['train_owner'][b]).astype(jnp.float32) * 0.03
            r = r.at[jnp.maximum(0, w['edge_builder'][e])].add(-0.02 * damaged * (w['edge_builder'][e] >= 0))
            return w, r

        s, rewards = jax.lax.fori_loop(0, 8, train, (s, rewards))
        t = s['t'] + 1
        storm = (t % 32 == 0) & jax.random.bernoulli(key, 0.125, (24,))
        s['bridge'] = jnp.maximum(0, s['bridge'] - storm * 2).astype(jnp.int32)
        s['edge_event'] = jnp.where(storm, 2, s['edge_event']).astype(jnp.int32)
        consume = t % 16 == 0
        hungry = consume & (s['grain'] == 0)
        rewards = rewards - hungry[s['team']] * 0.02
        s['grain'] = jnp.clip(s['grain'] - consume + (t % 24 == 0) * 2, 0, 24).astype(jnp.int32)
        s['ore'] = jnp.minimum(24, s['ore'] + (t % 16 == 0)).astype(jnp.int32)
        s['t'] = t
        s['terminated'] = jnp.array(False)
        s['truncated'] = t >= 1024
        s['done'] = s['truncated']
        s = self.refresh(s)
        return self.observe(s), s, jnp.clip(rewards, -4.0, 4.0).astype(jnp.float32), jnp.full(256, s['done'], jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
