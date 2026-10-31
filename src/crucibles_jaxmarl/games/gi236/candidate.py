"""Icebound Harbor Builders. Original author source; never executed by author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 512
    observation_size = 237
    num_actions = 15
    max_steps = 1024

    def nearest(self, p, q, active, k, radius):
        d = jnp.sum(jnp.abs(p[:, None, :] - q[None, :, :]), axis=2)
        ix = jnp.argsort(d + (~active[None, :]) * 10000, axis=1)[:, :k]
        valid = (jnp.take_along_axis(d, ix, axis=1) <= radius) & active[ix]
        vis = jnp.any(jax.nn.one_hot(ix, q.shape[0], dtype=jnp.bool_) & valid[:, :, None], axis=1)
        return ix.astype(jnp.int32), vis

    def refresh(self, s):
        s = dict(s)
        s['peer'], s['agent_visible'] = self.nearest(s['pos'], s['pos'], s['active'], 8, 4)
        s['near'], s['site_visible'] = self.nearest(s['pos'], s['site_pos'], s['site_active'], 8, 5)
        x = jnp.arange(64, dtype=jnp.int32)[None, None, :]
        y = jnp.arange(32, dtype=jnp.int32)[None, :, None]
        s['map_visible'] = (jnp.abs(x - s['pos'][:, 0, None, None]) <= 2) & (jnp.abs(y - s['pos'][:, 1, None, None]) <= 2)
        s['terrain'] = jnp.where(s['land'], 0, jnp.where(s['road'] > 0, 3, jnp.where(s['ice'] > 2, 1, jnp.where(s['ice'] > 0, 2, 4)))).astype(jnp.int32)
        return s

    def reset(self, key):
        ids = jnp.arange(512, dtype=jnp.int32)
        team = ids // 16
        local = ids % 16
        pos = jnp.stack(((team % 8) * 8 + local % 4, (team // 8) * 8 + local // 4), axis=1)
        sid = jnp.arange(160, dtype=jnp.int32)
        crew = sid // 5
        kind = sid % 5
        offsets = jnp.array([[1, 6], [4, 6], [2, 2], [5, 4], [4, 2]], dtype=jnp.int32)
        sp = jnp.stack(((crew % 8) * 8, (crew // 8) * 8), axis=1) + offsets[kind]
        xx = jnp.arange(64, dtype=jnp.int32)[None, :]
        yy = jnp.arange(32, dtype=jnp.int32)[:, None]
        land = jnp.broadcast_to((xx % 8 < 6) | (yy % 8 == 0), (32, 64))
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
             'pos': pos, 'active': jnp.ones(512, jnp.bool_), 'team': team, 'tool': local % 5,
             'cargo': jnp.where(local % 5 == 2, 3, 0).astype(jnp.int32), 'health': jnp.full(512, 8, jnp.int32),
             'signal': jnp.zeros(512, jnp.int32), 'anchor': jnp.zeros(512, jnp.bool_), 'event': jnp.zeros(512, jnp.int32),
             'site_pos': sp, 'site_active': jnp.ones(160, jnp.bool_), 'site_kind': kind, 'site_crew': crew,
             'stock': jnp.where(kind == 4, 4, jnp.where(kind < 2, 12, 0)).astype(jnp.int32),
             'built': (kind != 3) & (kind != 2), 'fuel': jnp.where(kind == 2, 4, 0).astype(jnp.int32), 'delivered': jnp.zeros(160, jnp.int32),
             'land': land, 'ice': jnp.where(land, 8, jax.random.randint(key, (32, 64), 5, 9, dtype=jnp.int32)).astype(jnp.int32),
             'road': jnp.zeros((32, 64), jnp.int32), 'terrain': jnp.zeros((32, 64), jnp.int32),
             'map_visible': jnp.zeros((512, 32, 64), jnp.bool_), 'peer': jnp.zeros((512, 8), jnp.int32),
             'near': jnp.zeros((512, 8), jnp.int32), 'agent_visible': jnp.zeros((512, 512), jnp.bool_), 'site_visible': jnp.zeros((512, 160), jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        alive = (s['health'] > 0) & ~s['done']
        sd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['site_pos'][None, :, :]), axis=2)
        near = sd <= 1
        quarry = jnp.any(near & (s['site_kind'][None, :] == 0) & (s['stock'][None, :] > 0), axis=1)
        wood = jnp.any(near & (s['site_kind'][None, :] == 1) & (s['stock'][None, :] > 0), axis=1)
        depot = jnp.any(near & (s['site_kind'][None, :] == 4) & (s['stock'][None, :] > 0), axis=1)
        shelter = jnp.any(near & (s['site_kind'][None, :] == 2), axis=1)
        dock = jnp.any(near & (s['site_kind'][None, :] == 3) & s['built'][None, :], axis=1)
        ad = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2)
        peer = jnp.any((ad <= 1) & (ad > 0), axis=1)
        gather = (s['cargo'] == 0) & ((quarry & (s['tool'] == 0)) | (wood & (s['tool'] == 1)) | (depot & (s['tool'] == 2)))
        columns = [jnp.ones(512, jnp.bool_), alive, alive, alive, alive, alive,
                   alive & gather, alive & (s['cargo'] == 1), alive & shelter & (s['cargo'] == 2),
                   alive & peer & (s['cargo'] > 0), alive & (s['cargo'] == 3), alive & peer & (s['cargo'] == 2) & (s['tool'] == 4),
                   alive & dock & (s['tool'] == 3), alive & peer, alive]
        return jnp.stack(columns, axis=1)

    def observe(self, s):
        a = jnp.arange(512, dtype=jnp.int32)
        own = jnp.stack((s['pos'][:, 0] / 63.0, s['pos'][:, 1] / 31.0, s['team'] / 31.0, s['tool'] / 4.0,
                         s['cargo'] / 3.0, s['health'] / 8.0, s['signal'] / 4.0, s['anchor'].astype(jnp.float32),
                         s['event'] / 9.0, jnp.full(512, s['t'] / 1024.0), jnp.full(512, (s['t'] // 32) % 2), a / 511.0), axis=1)
        pdata = jnp.stack((s['pos'][:, 0] / 63.0, s['pos'][:, 1] / 31.0, s['team'] / 31.0, s['tool'] / 4.0,
                           s['cargo'] / 3.0, s['health'] / 8.0, s['signal'] / 4.0, s['anchor'].astype(jnp.float32), a / 511.0, jnp.ones(512)), axis=1)
        pv = jnp.take_along_axis(s['agent_visible'], s['peer'], axis=1)
        peers = (pdata[s['peer']] * pv[:, :, None]).reshape((512, 80))
        q = s['site_pos']
        data = jnp.stack((q[:, 0] / 63.0, q[:, 1] / 31.0, s['site_kind'] / 4.0, s['site_crew'] / 31.0,
                          s['stock'] / 12.0, s['built'].astype(jnp.float32), s['fuel'] / 24.0, s['delivered'] / 1024.0,
                          jnp.arange(160, dtype=jnp.int32) / 159.0, jnp.ones(160)), axis=1)
        ov = jnp.take_along_axis(s['site_visible'], s['near'], axis=1)
        objects = (data[s['near']] * ov[:, :, None]).reshape((512, 80))
        offsets = jnp.arange(25, dtype=jnp.int32)
        px = jnp.clip(s['pos'][:, 0, None] + offsets[None, :] % 5 - 2, 0, 63)
        py = jnp.clip(s['pos'][:, 1, None] + offsets[None, :] // 5 - 2, 0, 31)
        patch = jnp.concatenate((s['ice'][py, px] / 8.0, s['road'][py, px] / 6.0), axis=1)
        terrain = s['terrain'][s['pos'][:, 1], s['pos'][:, 0]] / 4.0
        mask_summary = self.legal_actions(s).astype(jnp.float32)[:, 1:15]
        return jnp.concatenate((own, peers, objects, patch, terrain[:, None], mask_summary), axis=1).astype(jnp.float32)

    def absorb(self, args):
        key, s, actions = args
        return self.observe(s), s, jnp.zeros(512, jnp.float32), jnp.ones(512, jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state['done'], self.absorb, self.advance, (key, state, actions))

    def advance(self, args):
        key, s, actions = args
        s = dict(s)
        safe = jnp.clip(actions, 0, 14)
        legal = self.legal_actions(s)[jnp.arange(512, dtype=jnp.int32), safe]
        actions = jnp.where(legal & (actions >= 0) & (actions < 15), safe, 0)
        s['event'] = jnp.zeros(512, jnp.int32)
        s['signal'] = jnp.maximum(s['signal'] - 1, 0)
        s['anchor'] = (actions == 5) & (s['health'] > 0)
        anchors = s['anchor']
        initial_pos = s['pos']
        rewards = jnp.zeros(512, jnp.float32)
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

        def actor(k, carry):
            w, r = carry
            w = dict(w)
            i = (k + s['t']) % 512
            action = actions[i]
            p = w['pos'][i]
            sd = jnp.sum(jnp.abs(w['site_pos'] - p), axis=1)
            d = jnp.sum(jnp.abs(initial_pos - p), axis=1)
            anchored = jnp.any((d <= 1) & anchors & (jnp.arange(512, dtype=jnp.int32) != i))
            target = jnp.clip(p + directions[jnp.clip(action, 0, 4)], jnp.array([0, 0], dtype=jnp.int32), jnp.array([63, 31], dtype=jnp.int32))
            occupied = jnp.any(jnp.all(w['pos'] == target, axis=1) & (jnp.arange(512, dtype=jnp.int32) != i))
            tx, ty = target[0], target[1]
            passable = w['land'][ty, tx] | (w['ice'][ty, tx] > 0) | (w['road'][ty, tx] > 0)
            move = (action >= 1) & (action <= 4) & ~occupied & passable & ((w['cargo'][i] != 3) | anchored) & jnp.any(target != p)
            w['pos'] = w['pos'].at[i].set(jnp.where(move, target, p))
            stress = move & ~w['land'][ty, tx]
            weight = jnp.where(w['cargo'][i] == 3, 3, 1)
            w['ice'] = w['ice'].at[ty, tx].set(jnp.maximum(0, w['ice'][ty, tx] - stress * weight))
            w['road'] = w['road'].at[ty, tx].set(jnp.maximum(0, w['road'][ty, tx] - stress * (w['cargo'][i] == 3)))
            crack = stress & (w['ice'][ty, tx] == 0) & (w['road'][ty, tx] == 0)
            w['health'] = w['health'].at[i].set(jnp.maximum(0, w['health'][i] - crack * 4))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(crack, 0, w['cargo'][i]))
            wanted = jnp.where(w['tool'][i] == 0, 0, jnp.where(w['tool'][i] == 1, 1, 4))
            candidates = (sd <= 1) & (w['site_kind'] == wanted) & (w['stock'] > 0)
            j = jnp.argmax(candidates)
            take = (action == 6) & jnp.any(candidates) & (w['cargo'][i] == 0)
            w['stock'] = w['stock'].at[j].add(-take.astype(jnp.int32))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(take, jnp.where(wanted == 0, 1, jnp.where(wanted == 1, 2, 3)), w['cargo'][i]))
            front = jnp.clip(p + jnp.array([1, 0], dtype=jnp.int32), jnp.array([0, 0], dtype=jnp.int32), jnp.array([63, 31], dtype=jnp.int32))
            docks = (sd <= 1) & (w['site_kind'] == 3)
            build_sites = (sd <= 1) & ((w['site_kind'] == 2) | (w['site_kind'] == 3))
            j = jnp.argmax(build_sites)
            build = (action == 7) & (w['cargo'][i] == 1)
            dockbuild = build & jnp.any(build_sites)
            roadbuild = build & ~jnp.any(build_sites) & ~w['land'][front[1], front[0]]
            w['built'] = w['built'].at[j].set(w['built'][j] | dockbuild)
            w['road'] = w['road'].at[front[1], front[0]].set(jnp.where(roadbuild, 6, w['road'][front[1], front[0]]))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(dockbuild | roadbuild, 0, w['cargo'][i]))
            shelters = (sd <= 1) & (w['site_kind'] == 2)
            j = jnp.argmax(shelters)
            burn = (action == 8) & jnp.any(shelters) & (w['cargo'][i] == 2) & (w['fuel'][j] < 24)
            w['fuel'] = w['fuel'].at[j].set(jnp.minimum(24, w['fuel'][j] + burn * 8))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(burn, 0, w['cargo'][i]))
            ad = jnp.sum(jnp.abs(w['pos'] - p), axis=1)
            neighbors = (ad <= 1) & (jnp.arange(512, dtype=jnp.int32) != i) & (actions == 0)
            empty = neighbors & (w['cargo'] == 0) & (w['health'] > 0)
            n = jnp.argmax(empty)
            hand = (action == 9) & jnp.any(empty) & (w['cargo'][i] > 0)
            item = w['cargo'][i]
            w['cargo'] = w['cargo'].at[n].set(jnp.where(hand, item, w['cargo'][n]))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(hand, 0, w['cargo'][i]))
            depots = (sd <= 1) & (w['site_kind'] == 4) & (w['site_crew'] != w['team'][i])
            j = jnp.argmax(depots)
            deliver = (action == 10) & jnp.any(depots) & (w['cargo'][i] == 3)
            w['delivered'] = w['delivered'].at[j].set(jnp.minimum(1024, w['delivered'][j] + deliver))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(deliver, 0, w['cargo'][i]))
            r = r + deliver * (w['team'] == w['site_crew'][j]).astype(jnp.float32) * 0.04
            fallen = (ad <= 1) & (w['health'] == 0) & (jnp.arange(512, dtype=jnp.int32) != i)
            n = jnp.argmax(fallen)
            rescue = (action == 11) & jnp.any(fallen) & (w['cargo'][i] == 2) & (w['tool'][i] == 4)
            w['health'] = w['health'].at[n].set(jnp.where(rescue, 4, w['health'][n]))
            w['cargo'] = w['cargo'].at[i].set(jnp.where(rescue, 0, w['cargo'][i]))
            j = jnp.argmax(docks & w['built'])
            destination = jnp.minimum(w['site_pos'][j] + jnp.array([3, 0], dtype=jnp.int32), jnp.array([63, 31], dtype=jnp.int32))
            fuel_site = (j // 5) * 5 + 2
            free = ~jnp.any(jnp.all(w['pos'] == destination, axis=1) & (jnp.arange(512, dtype=jnp.int32) != i))
            ferry = (action == 12) & jnp.any(docks & w['built']) & (w['tool'][i] == 3) & (w['fuel'][fuel_site] > 0) & free & w['land'][destination[1], destination[0]]
            w['pos'] = w['pos'].at[i].set(jnp.where(ferry, destination, w['pos'][i]))
            w['fuel'] = w['fuel'].at[fuel_site].add(-ferry.astype(jnp.int32))
            n = jnp.argmax(neighbors)
            swap = (action == 13) & jnp.any(neighbors)
            ti, tn = w['tool'][i], w['tool'][n]
            w['tool'] = w['tool'].at[i].set(jnp.where(swap, tn, ti)).at[n].set(jnp.where(swap, ti, tn))
            w['signal'] = w['signal'].at[i].set(jnp.where(action == 14, 4, w['signal'][i]))
            event = jnp.where(crack, 1, jnp.where(take, 2, jnp.where(dockbuild | roadbuild, 3, jnp.where(burn, 4, jnp.where(hand, 5, jnp.where(deliver, 6, jnp.where(rescue, 7, jnp.where(ferry, 8, jnp.where(swap, 9, 0)))))))))
            w['event'] = w['event'].at[i].set(event)
            r = r.at[i].add(-0.002 * (action != 0) - 0.25 * crack + 0.3 * deliver + 0.1 * rescue)
            return w, r

        s, rewards = jax.lax.fori_loop(0, 512, actor, (s, rewards))
        t = s['t'] + 1
        thaw = (t % 32 == 0) & ((t // 32) % 2 == 1)
        freeze = (t % 32 == 0) & ((t // 32) % 2 == 0)
        s['ice'] = jnp.where(s['land'], 8, jnp.clip(s['ice'] - thaw + freeze * 2, 0, 8)).astype(jnp.int32)
        s['road'] = jnp.maximum(0, s['road'] - thaw).astype(jnp.int32)
        distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['site_pos'][None, :, :]), axis=2)
        warm = jnp.any((distance <= 2) & (s['site_kind'][None, :] == 2) & s['built'][None, :] & (s['fuel'][None, :] > 0), axis=1)
        cold_tick = t % 16 == 0
        old = s['health']
        s['health'] = jnp.where(old > 0, jnp.clip(old + cold_tick * jnp.where(warm, 1, -1), 0, 8), 0).astype(jnp.int32)
        rewards = rewards - 0.03 * ((old > 0) & (s['health'] < old))
        s['fuel'] = jnp.maximum(0, s['fuel'] - (t % 8 == 0)).astype(jnp.int32)
        s['stock'] = jnp.minimum(12, s['stock'] + ((s['site_kind'] < 2) & (t % 16 == 0))).astype(jnp.int32)
        s['t'] = t
        s['terminated'] = jnp.all(s['health'] == 0)
        s['truncated'] = (t >= 1024) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        s = self.refresh(s)
        return self.observe(s), s, jnp.clip(rewards, -4.0, 4.0).astype(jnp.float32), jnp.full(512, s['done'], jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
