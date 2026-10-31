import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 256
    observation_size = 686
    num_actions = 20
    max_steps = 896

    def distance(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=-1)

    def refresh(self, s):
        s = dict(s)
        d = self.distance(s['pos'], s['pos'])
        ix = jnp.argsort(d + jnp.arange(256, dtype=jnp.int32)[None, :] / 1024.0, axis=1)[:, :4]
        seen = jnp.take_along_axis(d, ix, axis=1) <= 6
        s['seen'] = ix.astype(jnp.int32)
        s['agent_vis'] = jnp.zeros((256, 256), jnp.bool_).at[jnp.arange(256, dtype=jnp.int32)[:, None], ix].set(seen)
        s['agent_vis'] = s['agent_vis'].at[jnp.arange(256, dtype=jnp.int32), jnp.arange(256, dtype=jnp.int32)].set(True)
        for name in ('quarry', 'route', 'shelter', 'boat', 'victim', 'ash'):
            s[name + '_vis'] = self.distance(s['pos'], s[name + '_pos']) <= 6
        s['carrying'] = jnp.any(s['victim_carrier'][None, :] == jnp.arange(256, dtype=jnp.int32)[:, None], axis=1)
        return s

    def packed(self, pos, data, vis):
        rows = jnp.concatenate((pos.astype(jnp.float32) / 63.0, data), axis=1)
        out = jnp.where(vis[:, :, None], rows[None, :, :], 0.0)
        return jnp.concatenate((vis[:, :, None].astype(jnp.float32), out), axis=2).reshape((256, -1))

    def observe(self, s):
        ids = jnp.arange(256, dtype=jnp.int32)
        crew = ids // 16
        own = jnp.stack((jnp.arange(256, dtype=jnp.int32) / 255.0, s['pos'][:, 0] / 63.0, s['pos'][:, 1] / 63.0, crew / 15.0, s['tool'] / 3.0, s['stone'] / 4.0, (s['aboard'] + 1) / 16.0, s['carrying'].astype(jnp.float32), s['signal'] / 3.0, jnp.full((256,), s['t'] / 896.0), jnp.full((256,), s['t'] // 16 % 2), s['rescued'][crew] / 32.0, s['shelter_level'][crew] / 3.0), axis=1)
        data = jnp.stack((jnp.arange(256, dtype=jnp.int32) / 255.0, s['pos'][:, 0] / 63.0, s['pos'][:, 1] / 63.0, crew / 15.0, s['tool'] / 3.0, s['stone'] / 4.0, (s['aboard'] + 1) / 16.0, s['carrying'].astype(jnp.float32), s['signal'] / 3.0), axis=1)
        v = jnp.take_along_axis(s['agent_vis'], s['seen'], axis=1)
        others = jnp.concatenate((v[:, :, None], jnp.where(v[:, :, None], data[s['seen']], 0)), axis=2).reshape((256, 40))
        q = self.packed(s['quarry_pos'], s['quarry_stock'][:, None] / 32.0, s['quarry_vis'])
        r = self.packed(s['route_pos'], jnp.stack((s['route_level'] / 3.0, s['route_open'], s['route_damage']), axis=1), s['route_vis'])
        h = self.packed(s['shelter_pos'], jnp.stack((s['shelter_level'] / 3.0, s['rescued'] / 32.0), axis=1), s['shelter_vis'])
        b = self.packed(s['boat_pos'], jnp.stack((s['boat_cargo'] / 4.0, s['boat_load'] / 8.0), axis=1), s['boat_vis'])
        p = self.packed(s['victim_pos'], jnp.stack(((s['victim_carrier'] + 1) / 256.0, s['victim_saved']), axis=1), s['victim_vis'])
        a = self.packed(s['ash_pos'], jnp.stack((s['ash_level'] / 3.0, s['ash_clear'] / 3.0), axis=1), s['ash_vis'])
        off = jnp.stack((jnp.tile(jnp.arange(-2, 3, dtype=jnp.int32), 5), jnp.repeat(jnp.arange(-2, 3, dtype=jnp.int32), 5)), axis=1)
        xy = jnp.clip(s['pos'][:, None, :] + off[None, :, :], 0, 63)
        patch = s['terrain'][xy[:, :, 1], xy[:, :, 0]].astype(jnp.float32) / 2.0
        return jnp.concatenate((own, others, q, r, h, b, p, a, patch), axis=1).astype(jnp.float32)

    def reset(self, key):
        ids = jnp.arange(256, dtype=jnp.int32)
        c = jnp.arange(16, dtype=jnp.int32)
        center = jnp.stack((8 + 16 * (c % 4), 8 + 16 * (c // 4)), axis=1)
        pos = center[ids // 16] + jnp.stack((ids % 4 - 2, ids % 16 // 4 - 2), axis=1)
        xx = jnp.arange(64, dtype=jnp.int32)[None, :]
        yy = jnp.arange(64, dtype=jnp.int32)[:, None]
        land = (xx % 16 >= 1) & (xx % 16 <= 14) & (yy % 16 >= 1) & (yy % 16 <= 14)
        shoal = ((xx % 16 == 0) | (xx % 16 == 15)) & (yy % 16 == 8) | ((yy % 16 == 0) | (yy % 16 == 15)) & (xx % 16 == 8)
        terrain = jnp.where(land, 1, jnp.where(shoal, 2, 0)).astype(jnp.int32)
        k = jnp.arange(12, dtype=jnp.int32)
        routes = jnp.concatenate((jnp.stack((15 + 16 * (k % 3), 8 + 16 * (k // 3)), axis=1), jnp.stack((8 + 16 * (k % 4), 15 + 16 * (k // 4)), axis=1)), axis=0)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'terrain': terrain, 'pos': pos, 'active': jnp.ones((256,), jnp.bool_), 'tool': ids % 4, 'stone': jnp.zeros((256,), jnp.int32), 'aboard': jnp.full((256,), -1, jnp.int32), 'signal': jnp.zeros((256,), jnp.int32), 'carrying': jnp.zeros((256,), jnp.bool_), 'seen': jnp.zeros((256, 4), jnp.int32), 'agent_vis': jnp.zeros((256, 256), jnp.bool_), 'quarry_pos': center + jnp.array([-3, 0], dtype=jnp.int32), 'quarry_stock': jnp.full((16,), 32, jnp.int32), 'quarry_active': jnp.ones((16,), jnp.bool_), 'quarry_vis': jnp.zeros((256, 16), jnp.bool_), 'route_pos': routes, 'route_level': jnp.zeros((24,), jnp.int32), 'route_open': jnp.zeros((24,), jnp.bool_), 'route_damage': jnp.zeros((24,), jnp.bool_), 'route_active': jnp.ones((24,), jnp.bool_), 'route_vis': jnp.zeros((256, 24), jnp.bool_), 'shelter_pos': center + jnp.array([0, -3], dtype=jnp.int32), 'shelter_level': jnp.zeros((16,), jnp.int32), 'shelter_active': jnp.ones((16,), jnp.bool_), 'shelter_vis': jnp.zeros((256, 16), jnp.bool_), 'rescued': jnp.zeros((16,), jnp.int32), 'boat_pos': center + jnp.array([7, 0], dtype=jnp.int32), 'boat_cargo': jnp.zeros((16,), jnp.int32), 'boat_load': jnp.zeros((16,), jnp.int32), 'boat_active': jnp.ones((16,), jnp.bool_), 'boat_vis': jnp.zeros((256, 16), jnp.bool_), 'victim_pos': jnp.repeat(center, 2, axis=0) + jnp.tile(jnp.array([[2, 1], [3, 2]], jnp.int32), (16, 1)), 'victim_carrier': jnp.full((32,), -1, jnp.int32), 'victim_saved': jnp.zeros((32,), jnp.bool_), 'victim_active': jnp.ones((32,), jnp.bool_), 'victim_vis': jnp.zeros((256, 32), jnp.bool_), 'ash_pos': center + jnp.array([4, -4], dtype=jnp.int32), 'ash_level': jax.random.randint(key, (16,), 0, 2).astype(jnp.int32), 'ash_clear': jnp.zeros((16,), jnp.int32), 'ash_active': jnp.ones((16,), jnp.bool_), 'ash_vis': jnp.zeros((256, 16), jnp.bool_), 'event_rescue': jnp.zeros((256,), jnp.bool_), 'event_build': jnp.zeros((256,), jnp.bool_), 'event_transfer': jnp.zeros((256,), jnp.bool_)}
        s = self.refresh(s)
        return (self.observe(s), s)

    def legal_actions(self, s):
        foot = s['aboard'] < 0
        mask = jnp.ones((256, 20), jnp.bool_)
        mask = mask.at[:, 5].set(foot & (s['tool'] == 0) & (s['stone'] < 4))
        mask = mask.at[:, 7].set(foot & (s['tool'] == 1) & (s['stone'] > 0))
        mask = mask.at[:, 8].set(foot & (s['tool'] == 1) & (s['stone'] > 0))
        mask = mask.at[:, 10].set(foot & (s['tool'] == 3) & ~s['carrying'])
        mask = mask.at[:, 11].set(foot & s['carrying'])
        mask = mask.at[:, 12].set(foot & ~s['carrying'])
        mask = mask.at[:, 13].set(~foot)
        mask = mask.at[:, 14].set(foot & (s['stone'] > 0))
        mask = mask.at[:, 15].set(foot)
        mask = mask.at[:, 16].set(foot & (s['tool'] == 1))
        mask = mask.at[:, 18].set(foot & (s['stone'] > 0))
        mask = mask.at[:, 19].set(foot & (s['stone'] < 4))
        return mask & (~s['done'] | (jnp.arange(20, dtype=jnp.int32)[None, :] == 0))

    def raw_step(self, key, state, actions):
        s = state
        clipped = jnp.clip(actions, 0, 19)
        a = jnp.where((actions >= 0) & (actions < 20) & self.legal_actions(s)[jnp.arange(256, dtype=jnp.int32), clipped], clipped, 0)
        n = dict(s)
        n['t'] = s['t'] + 1
        n['signal'] = jnp.maximum(s['signal'] - 1, 0)
        n['event_rescue'] = jnp.zeros((256,), jnp.bool_)
        n['event_build'] = jnp.zeros((256,), jnp.bool_)
        n['event_transfer'] = jnp.zeros((256,), jnp.bool_)
        n['ash_clear'] = jnp.maximum(s['ash_clear'] - 1, 0)
        eruption = (n['t'] % 32 == 0).astype(jnp.int32)
        n['ash_level'] = jnp.clip(s['ash_level'] + eruption - 2 * (s['ash_clear'] == 1).astype(jnp.int32), 0, 3)
        protect = jnp.min(self.distance(s['route_pos'], s['shelter_pos']) + jnp.where(s['shelter_level'][None, :] >= 2, 0, 1000), axis=1) <= 10
        n['route_damage'] = (n['t'] % 64 == 0) & (s['route_level'] > 0) & ~protect
        n['route_level'] = jnp.maximum(s['route_level'] - n['route_damage'].astype(jnp.int32), 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        motion = delta[jnp.minimum(a, 4)] * ((a >= 1) & (a <= 4))[:, None]
        target = jnp.clip(s['pos'] + motion, 0, 63)
        rd = self.distance(target, s['route_pos'])
        bridge = jnp.any((rd <= 1) & (n['route_level'][None, :] >= 2) & s['route_open'][None, :], axis=1)
        tile = s['terrain'][target[:, 1], target[:, 0]]
        passable = (tile == 1) | (tile == 2) & (s['t'] // 16 % 2 == 0) | bridge
        ad = self.distance(target, s['ash_pos'])
        covered = jnp.any((self.distance(target, s['shelter_pos']) <= 4) & (s['shelter_level'][None, :] >= 2), axis=1)
        hot = jnp.any((ad <= 2) & (n['ash_level'][None, :] >= 3), axis=1) & ~covered
        helpers = jnp.any((self.distance(s['pos'], s['pos']) <= 1) & (a[None, :] == 6) & (jnp.arange(256, dtype=jnp.int32)[:, None] != jnp.arange(256, dtype=jnp.int32)[None, :]), axis=1)
        walk = passable & ~hot & (~s['carrying'] | helpers) & (s['aboard'] < 0)
        n['pos'] = jnp.where(walk[:, None], target, s['pos'])
        pilot = (s['aboard'][:, None] == jnp.arange(16, dtype=jnp.int32)[None, :]) & (s['tool'][:, None] == 2) & ((a >= 1) & (a <= 4))[:, None]
        lead = jnp.min(jnp.where(pilot, jnp.arange(256, dtype=jnp.int32)[:, None], 256), axis=0)
        bt = jnp.clip(s['boat_pos'] + motion[jnp.minimum(lead, 255)], 0, 63)
        sail = (lead < 256) & (s['terrain'][bt[:, 1], bt[:, 0]] != 1)
        n['boat_pos'] = jnp.where(sail[:, None], bt, s['boat_pos'])
        n['pos'] = jnp.where((s['aboard'] >= 0)[:, None], n['boat_pos'][jnp.maximum(s['aboard'], 0)], n['pos'])
        reward = -0.02 * jnp.sum(n['route_damage']).astype(jnp.float32) / 24.0 - 0.01 * (a >= 5).astype(jnp.float32) - 0.005 * (s['carrying'] & jnp.any(motion != 0, axis=1)).astype(jnp.float32) - 0.03 * hot.astype(jnp.float32)

        def work(i, pair):
            z, rew = pair
            z = dict(z)
            here = z['pos'][i]
            qd = jnp.sum(jnp.abs(z['quarry_pos'] - here), axis=1)
            q = jnp.argmin(qd)
            quarry = (a[i] == 5) & (qd[q] <= 1) & (z['quarry_stock'][q] > 0) & (z['stone'][i] < 4)
            z['quarry_stock'] = z['quarry_stock'].at[q].add(-quarry.astype(jnp.int32))
            z['stone'] = z['stone'].at[i].add(quarry.astype(jnp.int32))
            rd = jnp.sum(jnp.abs(z['route_pos'] - here), axis=1)
            r = jnp.argmin(rd)
            build = (a[i] == 7) & (rd[r] <= 2) & (z['route_level'][r] < 3) & (z['stone'][i] > 0)
            z['route_level'] = z['route_level'].at[r].add(build.astype(jnp.int32))
            gate = (a[i] == 9) & (rd[r] <= 2)
            z['route_open'] = z['route_open'].at[r].set(z['route_open'][r] ^ gate)
            hd = jnp.sum(jnp.abs(z['shelter_pos'] - here), axis=1)
            h = jnp.argmin(hd)
            shelter = (a[i] == 8) & (hd[h] <= 1) & (z['shelter_level'][h] < 3) & (z['stone'][i] > 0)
            z['shelter_level'] = z['shelter_level'].at[h].add(shelter.astype(jnp.int32))
            z['stone'] = z['stone'].at[i].add(-build.astype(jnp.int32) - shelter.astype(jnp.int32))
            vd = jnp.sum(jnp.abs(z['victim_pos'] - here), axis=1)
            eligible = ~z['victim_saved'] & (z['victim_carrier'] < 0)
            v = jnp.argmin(jnp.where(eligible, vd, 1000))
            pickup = (a[i] == 10) & (vd[v] <= 1) & eligible[v] & helpers[i]
            z['victim_carrier'] = z['victim_carrier'].at[v].set(jnp.where(pickup, i, z['victim_carrier'][v]))
            mine = z['victim_carrier'] == i
            deliver = (a[i] == 11) & (hd[h] <= 1) & jnp.any(mine)
            z['victim_saved'] = z['victim_saved'] | mine & deliver
            z['victim_carrier'] = jnp.where(mine & deliver, -1, z['victim_carrier'])
            z['rescued'] = z['rescued'].at[h].add(deliver.astype(jnp.int32))
            bd = jnp.sum(jnp.abs(z['boat_pos'] - here), axis=1)
            b = jnp.argmin(bd)
            load = jnp.sum(z['aboard'] == b)
            board = (a[i] == 12) & (bd[b] <= 1) & (load < 8 - z['boat_cargo'][b])
            oldb = jnp.maximum(z['aboard'][i], 0)
            exits = jnp.clip(z['boat_pos'][oldb] + delta[1:], 0, 63)
            ok = z['terrain'][exits[:, 1], exits[:, 0]] == 1
            leave = (a[i] == 13) & jnp.any(ok)
            z['aboard'] = z['aboard'].at[i].set(jnp.where(board, b, jnp.where(leave, -1, z['aboard'][i])))
            z['pos'] = z['pos'].at[i].set(jnp.where(board, z['boat_pos'][b], jnp.where(leave, exits[jnp.argmax(ok)], here)))
            pd = jnp.sum(jnp.abs(z['pos'] - here), axis=1)
            friend = jnp.argmin(jnp.where(jnp.arange(256, dtype=jnp.int32) != i, pd, 1000))
            give = (a[i] == 14) & (pd[friend] <= 1) & (z['stone'][i] > 0) & (z['stone'][friend] < 4)
            z['stone'] = z['stone'].at[i].add(-give.astype(jnp.int32)).at[friend].add(give.astype(jnp.int32))
            lend = (a[i] == 15) & (pd[friend] <= 1)
            oldtool = z['tool'][i]
            newtool = z['tool'][friend]
            z['tool'] = z['tool'].at[i].set(jnp.where(lend, newtool, oldtool)).at[friend].set(jnp.where(lend, oldtool, newtool))
            ad = jnp.sum(jnp.abs(z['ash_pos'] - here), axis=1)
            island = jnp.argmin(ad)
            clear = (a[i] == 16) & (ad[island] <= 4) & (z['ash_clear'][island] == 0)
            z['ash_clear'] = z['ash_clear'].at[island].set(jnp.where(clear, 3, z['ash_clear'][island]))
            z['signal'] = z['signal'].at[i].set(jnp.where(a[i] == 17, 3, z['signal'][i]))
            put = (a[i] == 18) & (bd[b] <= 1) & (z['boat_cargo'][b] < 4) & (z['stone'][i] > 0) & (load < 8 - z['boat_cargo'][b])
            take = (a[i] == 19) & (bd[b] <= 1) & (z['boat_cargo'][b] > 0) & (z['stone'][i] < 4)
            change = put.astype(jnp.int32) - take.astype(jnp.int32)
            z['boat_cargo'] = z['boat_cargo'].at[b].add(change)
            z['stone'] = z['stone'].at[i].add(-change)
            z['event_rescue'] = z['event_rescue'].at[i].set(deliver)
            z['event_build'] = z['event_build'].at[i].set(build | shelter)
            z['event_transfer'] = z['event_transfer'].at[i].set(give | lend | put | take)
            rew = rew.at[i].add(deliver.astype(jnp.float32))
            rew = rew + (jnp.arange(256, dtype=jnp.int32) // 16 == i // 16).astype(jnp.float32) * (0.5 * deliver + 0.05 * (build | shelter)) / 16.0
            return (z, rew)
        n, reward = jax.lax.fori_loop(0, 256, work, (n, reward))
        vc = jnp.maximum(n['victim_carrier'], 0)
        n['victim_pos'] = jnp.where((n['victim_carrier'] >= 0)[:, None], n['pos'][vc], n['victim_pos'])
        n['victim_active'] = ~n['victim_saved']
        n['boat_load'] = jnp.sum(n['aboard'][:, None] == jnp.arange(16, dtype=jnp.int32)[None, :], axis=0).astype(jnp.int32)
        n['terminated'] = jnp.all(n['victim_saved'])
        n['done'] = n['terminated'] | (n['t'] >= 896)
        n = self.refresh(n)
        n = jax.tree.map(lambda new, old: jnp.where(s['done'], old, new), n, s)
        reward = jnp.where(s['done'], jnp.zeros_like(reward), reward).astype(jnp.float32)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': (n['t'] >= 896) & ~n['terminated']}
        return (self.observe(n), n, reward, jnp.full((256,), n['done'], jnp.bool_), info)
