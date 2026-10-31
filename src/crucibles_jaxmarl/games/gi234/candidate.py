import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 384
    observation_size = 1084
    num_actions = 23
    max_steps = 1024

    def distance(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=-1)

    def refresh(self, s):
        s = dict(s)
        d = self.distance(s['pos'], s['pos'])
        ix = jnp.argsort(d + jnp.arange(384, dtype=jnp.int32)[None, :] / 1536.0, axis=1)[:, :4]
        v = jnp.take_along_axis(d, ix, axis=1) <= 6
        s['seen'] = ix.astype(jnp.int32)
        s['agent_vis'] = jnp.zeros((384, 384), jnp.bool_).at[jnp.arange(384, dtype=jnp.int32)[:, None], ix].set(v)
        s['agent_vis'] = s['agent_vis'].at[jnp.arange(384, dtype=jnp.int32), jnp.arange(384, dtype=jnp.int32)].set(True)
        for group in ('gate', 'reservoir', 'field', 'quarry', 'packet'):
            s[group + '_vis'] = self.distance(s['pos'], s[group + '_pos']) <= 6
        s['permission_count'] = jnp.sum(s['permission'], axis=1).astype(jnp.int32)
        s['message_water'] = jnp.where(s['signal'] > 0, s['memo_water'], 0)
        return s

    def packed(self, pos, data, vis):
        rows = jnp.concatenate((pos.astype(jnp.float32) / jnp.array([95.0, 63.0], jnp.float32), data), axis=1)
        return jnp.concatenate((vis[:, :, None], jnp.where(vis[:, :, None], rows[None, :, :], 0)), axis=2).reshape((384, -1))

    def observe(self, s):
        ids = jnp.arange(384, dtype=jnp.int32)
        crew = ids // 16
        own = jnp.stack((jnp.arange(384, dtype=jnp.int32) / 383.0, s['pos'][:, 0] / 95.0, s['pos'][:, 1] / 63.0, crew / 23.0, s['tool'] / 4.0, s['material'] / 4.0, s['food'] / 3.0, s['memo_water'] / 12.0, s['signal'] / 4.0, jnp.full((384,), s['t'] / 1024.0), s['grain'][crew] / 40.0), axis=1)
        data = jnp.stack((jnp.arange(384, dtype=jnp.int32) / 383.0, s['pos'][:, 0] / 95.0, s['pos'][:, 1] / 63.0, crew / 23.0, s['tool'] / 4.0, s['material'] / 4.0, s['food'] / 3.0, s['signal'] / 4.0, s['message_water'] / 12.0), axis=1)
        vis = jnp.take_along_axis(s['agent_vis'], s['seen'], axis=1)
        people = jnp.concatenate((vis[:, :, None], jnp.where(vis[:, :, None], data[s['seen']], 0)), axis=2).reshape((384, 40))
        gate = self.packed(s['gate_pos'], jnp.stack((jnp.arange(24, dtype=jnp.int32) / 23.0, s['water'] / 12.0, s['gate_open'], s['canal'] / 2.0, s['route'], s['wall'] / 3.0, s['permission_count'] / 24.0, s['flow'] / 2.0, s['flood']), axis=1), s['gate_vis'])
        reservoir = self.packed(s['reservoir_pos'], s['reservoir'][:, None] / 3.0, s['reservoir_vis'])
        field = self.packed(s['field_pos'], jnp.stack((s['crop'] / 8.0, s['moisture'] / 6.0, s['grain'] / 40.0), axis=1), s['field_vis'])
        quarry = self.packed(s['quarry_pos'], s['quarry_stock'][:, None] / 48.0, s['quarry_vis'])
        packet = self.packed(s['packet_pos'], jnp.stack((s['packet_amount'] / 2.0, s['packet_target'] / 24.0), axis=1), s['packet_vis'])
        permission = s['permission'][:, crew].T.astype(jnp.float32)
        off = jnp.stack((jnp.tile(jnp.arange(-2, 3, dtype=jnp.int32), 5), jnp.repeat(jnp.arange(-2, 3, dtype=jnp.int32), 5)), axis=1)
        xy = jnp.clip(s['pos'][:, None, :] + off[None, :, :], jnp.array([0, 0], dtype=jnp.int32), jnp.array([95, 63], dtype=jnp.int32))
        patch = s['territory'][xy[:, :, 1], xy[:, :, 0]].astype(jnp.float32) / 23.0
        return jnp.concatenate((own, people, gate, reservoir, field, quarry, packet, permission, patch), axis=1).astype(jnp.float32)

    def reset(self, key):
        ids = jnp.arange(384, dtype=jnp.int32)
        c = jnp.arange(24, dtype=jnp.int32)
        centers = jnp.stack((8 + 16 * (c % 6), 8 + 16 * (c // 6)), axis=1)
        pos = centers[ids // 16] + jnp.stack((ids % 4 - 2, ids % 16 // 4 - 2), axis=1)
        gate = centers + jnp.array([0, -3], jnp.int32)
        territory = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32)[:, None] // 16 * 6 + jnp.arange(96, dtype=jnp.int32)[None, :] // 16, (64, 96)).astype(jnp.int32)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'territory': territory, 'pos': pos, 'active': jnp.ones((384,), jnp.bool_), 'tool': ids % 16 % 5, 'material': jnp.zeros((384,), jnp.int32), 'food': jnp.zeros((384,), jnp.int32), 'memo_water': jnp.zeros((384,), jnp.int32), 'signal': jnp.zeros((384,), jnp.int32), 'message_water': jnp.zeros((384,), jnp.int32), 'seen': jnp.zeros((384, 4), jnp.int32), 'agent_vis': jnp.zeros((384, 384), jnp.bool_), 'gate_pos': gate, 'gate_open': jnp.zeros((24,), jnp.bool_), 'canal': jnp.ones((24,), jnp.int32), 'route': jnp.zeros((24,), jnp.int32), 'wall': jnp.zeros((24,), jnp.int32), 'water': jnp.where(c < 6, 6, 3).astype(jnp.int32), 'flow': jnp.zeros((24,), jnp.int32), 'flood': jnp.zeros((24,), jnp.bool_), 'permission': jnp.eye(24, dtype=jnp.bool_), 'permission_count': jnp.ones((24,), jnp.int32), 'gate_active': jnp.ones((24,), jnp.bool_), 'gate_vis': jnp.zeros((384, 24), jnp.bool_), 'reservoir_pos': centers + jnp.array([3, -3], jnp.int32), 'reservoir': jnp.zeros((24,), jnp.int32), 'reservoir_active': jnp.ones((24,), jnp.bool_), 'reservoir_vis': jnp.zeros((384, 24), jnp.bool_), 'field_pos': centers + jnp.array([3, 2], jnp.int32), 'crop': jnp.zeros((24,), jnp.int32), 'moisture': jax.random.randint(key, (24,), 1, 3).astype(jnp.int32), 'grain': jnp.full((24,), 6, jnp.int32), 'field_active': jnp.ones((24,), jnp.bool_), 'field_vis': jnp.zeros((384, 24), jnp.bool_), 'quarry_pos': centers + jnp.array([-3, 2], jnp.int32), 'quarry_stock': jnp.full((24,), 48, jnp.int32), 'quarry_active': jnp.ones((24,), jnp.bool_), 'quarry_vis': jnp.zeros((384, 24), jnp.bool_), 'transit': jnp.zeros((24, 3), jnp.int32), 'transit_target': jnp.full((24, 3), 24, jnp.int32), 'packet_pos': jnp.repeat(gate.astype(jnp.float32), 3, axis=0), 'packet_amount': jnp.zeros((72,), jnp.int32), 'packet_target': jnp.full((72,), 24, jnp.int32), 'packet_active': jnp.zeros((72,), jnp.bool_), 'packet_vis': jnp.zeros((384, 72), jnp.bool_), 'event_harvest': jnp.zeros((384,), jnp.bool_), 'event_build': jnp.zeros((384,), jnp.bool_), 'event_grant': jnp.zeros((384,), jnp.bool_), 'event_delivery': jnp.zeros((384,), jnp.bool_), 'event_water_revenue': jnp.zeros((24,), jnp.int32)}
        s = self.refresh(s)
        return (self.observe(s), s)

    def legal_actions(self, s):
        tool = s['tool']
        mask = jnp.ones((384, 23), jnp.bool_)
        mask = mask.at[:, 5].set((tool == 0) & (s['material'] < 4))
        for action in (6, 7, 8):
            mask = mask.at[:, action].set((tool == 2) & (s['material'] > 0))
        for action in (9, 10, 11):
            mask = mask.at[:, action].set(tool == 3)
        mask = mask.at[:, 12].set((tool == 2) | (tool == 4))
        mask = mask.at[:, 13].set(tool == 4)
        mask = mask.at[:, 14].set(tool == 4)
        mask = mask.at[:, 17].set(s['material'] > 0)
        mask = mask.at[:, 18].set(s['food'] > 0)
        mask = mask.at[:, 19].set(s['food'] < 3)
        mask = mask.at[:, 20].set((tool == 1) & (s['food'] > 0))
        mask = mask.at[:, 22].set(tool == 2)
        return mask & (~s['done'] | (jnp.arange(23, dtype=jnp.int32)[None, :] == 0))

    def raw_step(self, key, state, actions):
        s = state
        ids = jnp.arange(384, dtype=jnp.int32)
        clipped = jnp.clip(actions, 0, 22)
        a = jnp.where((actions >= 0) & (actions < 23) & self.legal_actions(s)[ids, clipped], clipped, 0)
        n = dict(s)
        n['t'] = s['t'] + 1
        n['signal'] = jnp.maximum(s['signal'] - 1, 0)
        for name in ('event_harvest', 'event_build', 'event_grant', 'event_delivery'):
            n[name] = jnp.zeros((384,), jnp.bool_)
        n['event_water_revenue'] = jnp.zeros((24,), jnp.int32)
        dest = s['transit_target'][:, 2]
        arrived = jnp.zeros((24,), jnp.int32).at[jnp.minimum(dest, 23)].add(s['transit'][:, 2] * (dest < 24))
        c = jnp.arange(24, dtype=jnp.int32)
        rain = ((n['t'] % 8 == 0) & (c < 6)).astype(jnp.int32) * 2
        drought = (n['t'] % 48 == 0).astype(jnp.int32)
        incoming = jnp.maximum(s['water'] + arrived + rain - drought, 0)
        capacity = 6 + 2 * s['reservoir']
        n['flood'] = incoming > capacity + s['wall']
        n['water'] = jnp.minimum(incoming, capacity)
        growing = (s['crop'] > 0) & (s['crop'] < 8) & (s['moisture'] > 0) & (n['t'] % 4 == 0) & ~n['flood']
        n['crop'] = jnp.where(n['flood'], jnp.maximum(s['crop'] - 2, 0), jnp.minimum(s['crop'] + growing.astype(jnp.int32), 8))
        n['moisture'] = jnp.clip(s['moisture'] - growing.astype(jnp.int32) - drought + n['flood'].astype(jnp.int32), 0, 6)
        meal = (n['t'] % 64 == 0).astype(jnp.int32)
        n['grain'] = jnp.maximum(s['grain'] - 2 * meal, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        motion = delta[jnp.minimum(a, 4)] * ((a >= 1) & (a <= 4))[:, None]
        target = jnp.clip(s['pos'] + motion, jnp.array([0, 0], dtype=jnp.int32), jnp.array([95, 63], dtype=jnp.int32))
        region = s['territory'][target[:, 1], target[:, 0]]
        origin = s['territory'][s['pos'][:, 1], s['pos'][:, 0]]
        authorized = s['permission'][region, ids // 16] | (region == origin)
        road = (target[:, 1] % 16 == 8) | (target[:, 0] % 16 == 8)
        blocked = road & n['flood'][region] & (s['wall'][region] < 2)
        n['pos'] = jnp.where((authorized & ~blocked)[:, None], target, s['pos'])
        reward = -0.01 * (a >= 5).astype(jnp.float32) - 0.2 * n['flood'][ids // 16].astype(jnp.float32) - 0.05 * (meal & (s['grain'][ids // 16] < 2)).astype(jnp.float32)

        def work(i, pair):
            z, rew = pair
            z = dict(z)
            here = z['pos'][i]
            crew = i // 16
            qd = jnp.sum(jnp.abs(z['quarry_pos'] - here), axis=1)
            quarry = jnp.argmin(qd)
            dig = (a[i] == 5) & (qd[quarry] <= 1) & (z['quarry_stock'][quarry] > 0) & (z['material'][i] < 4)
            z['quarry_stock'] = z['quarry_stock'].at[quarry].add(-dig.astype(jnp.int32))
            z['material'] = z['material'].at[i].add(dig.astype(jnp.int32))
            gd = jnp.sum(jnp.abs(z['gate_pos'] - here), axis=1)
            g = jnp.argmin(gd)
            rd = jnp.sum(jnp.abs(z['reservoir_pos'] - here), axis=1)
            r = jnp.argmin(rd)
            build_c = (a[i] == 6) & (gd[g] <= 1) & (z['canal'][g] < 2) & (z['material'][i] > 0)
            build_r = (a[i] == 7) & (rd[r] <= 1) & (z['reservoir'][r] < 3) & (z['material'][i] > 0)
            build_w = (a[i] == 8) & (gd[g] <= 1) & (z['wall'][g] < 3) & (z['material'][i] > 0)
            z['canal'] = z['canal'].at[g].add(build_c.astype(jnp.int32))
            z['reservoir'] = z['reservoir'].at[r].add(build_r.astype(jnp.int32))
            z['wall'] = z['wall'].at[g].add(build_w.astype(jnp.int32))
            built = build_c | build_r | build_w
            z['material'] = z['material'].at[i].add(-built.astype(jnp.int32))
            toggle = (a[i] == 12) & (gd[g] <= 1)
            z['gate_open'] = z['gate_open'].at[g].set(z['gate_open'][g] ^ toggle)
            redirect = (a[i] == 22) & (gd[g] <= 1) & (g % 6 < 5)
            z['route'] = z['route'].at[g].set(jnp.where(redirect, 1 - z['route'][g], z['route'][g]))
            fd = jnp.sum(jnp.abs(z['field_pos'] - here), axis=1)
            field = jnp.argmin(fd)
            irrigate = (a[i] == 9) & (fd[field] <= 1) & (z['water'][field] > 0) & (z['moisture'][field] <= 4)
            z['water'] = z['water'].at[field].add(-irrigate.astype(jnp.int32))
            z['moisture'] = z['moisture'].at[field].add(2 * irrigate.astype(jnp.int32))
            plant = (a[i] == 10) & (fd[field] <= 1) & (z['crop'][field] == 0) & (z['grain'][field] > 0)
            harvest = (a[i] == 11) & (fd[field] <= 1) & (z['crop'][field] == 8) & (z['grain'][field] <= 36)
            z['crop'] = z['crop'].at[field].set(jnp.where(plant, 1, jnp.where(harvest, 0, z['crop'][field])))
            z['grain'] = z['grain'].at[field].add(-plant.astype(jnp.int32) + 4 * harvest.astype(jnp.int32))
            nd = jnp.sum(jnp.abs(z['pos'] - here), axis=1)
            foreign = jnp.argmin(jnp.where(jnp.arange(384, dtype=jnp.int32) // 16 != crew, nd, 1000))
            region = z['territory'][here[1], here[0]]
            grant = (a[i] == 13) & (region == crew) & (nd[foreign] <= 2)
            revoke = (a[i] == 14) & (region == crew)
            row = jnp.where(revoke, jnp.arange(24, dtype=jnp.int32) == crew, z['permission'][crew])
            row = row.at[foreign // 16].set(row[foreign // 16] | grant)
            z['permission'] = z['permission'].at[crew].set(row)
            read = (a[i] == 15) & (gd[g] <= 1)
            z['memo_water'] = z['memo_water'].at[i].set(jnp.where(read, z['water'][g], z['memo_water'][i]))
            z['signal'] = z['signal'].at[i].set(jnp.where(a[i] == 16, 4, z['signal'][i]))
            friend = jnp.argmin(jnp.where(jnp.arange(384, dtype=jnp.int32) != i, nd, 1000))
            give = (a[i] == 17) & (nd[friend] <= 1) & (z['material'][i] > 0) & (z['material'][friend] < 4)
            z['material'] = z['material'].at[i].add(-give.astype(jnp.int32)).at[friend].add(give.astype(jnp.int32))
            share = (a[i] == 18) & (nd[friend] <= 1) & (z['food'][i] > 0) & (z['food'][friend] < 3)
            z['food'] = z['food'].at[i].add(-share.astype(jnp.int32)).at[friend].add(share.astype(jnp.int32))
            load = (a[i] == 19) & (fd[field] <= 1) & (field == crew) & (z['grain'][field] > 0) & (z['food'][i] < 3)
            deliver = (a[i] == 20) & (fd[field] <= 1) & (field != crew) & (z['food'][i] > 0) & (z['grain'][field] < 40)
            z['grain'] = z['grain'].at[field].add(deliver.astype(jnp.int32) - load.astype(jnp.int32))
            z['food'] = z['food'].at[i].add(load.astype(jnp.int32) - deliver.astype(jnp.int32))
            lend = (a[i] == 21) & (nd[friend] <= 1)
            ta, tb = (z['tool'][i], z['tool'][friend])
            z['tool'] = z['tool'].at[i].set(jnp.where(lend, tb, ta)).at[friend].set(jnp.where(lend, ta, tb))
            z['event_harvest'] = z['event_harvest'].at[i].set(harvest)
            z['event_build'] = z['event_build'].at[i].set(built)
            z['event_grant'] = z['event_grant'].at[i].set(grant | revoke)
            z['event_delivery'] = z['event_delivery'].at[i].set(deliver)
            rew = rew.at[i].add(0.2 * harvest + 0.1 * deliver)
            rew = rew + (jnp.arange(384, dtype=jnp.int32) // 16 == field).astype(jnp.float32) * (harvest + 0.25 * deliver) / 16.0
            rew = rew + (jnp.arange(384, dtype=jnp.int32) // 16 == crew).astype(jnp.float32) * 0.1 * built / 16.0
            return (z, rew)
        n, reward = jax.lax.fori_loop(0, 384, work, (n, reward))
        flow = jnp.where(n['gate_open'], jnp.minimum(n['water'], n['canal']), 0)
        destination = jnp.where((n['route'] == 1) & (c % 6 < 5), c + 1, jnp.minimum(c + 6, 24))
        n['water'] = n['water'] - flow
        n['flow'] = flow
        n['transit'] = jnp.concatenate((flow[:, None], s['transit'][:, :2]), axis=1).astype(jnp.int32)
        n['transit_target'] = jnp.concatenate((destination[:, None], s['transit_target'][:, :2]), axis=1).astype(jnp.int32)
        n['event_water_revenue'] = jnp.where((dest < 24) & (dest != c), s['transit'][:, 2], 0)
        reward = reward + 0.02 * n['event_water_revenue'][ids // 16].astype(jnp.float32) / 16.0
        endpoint = n['gate_pos'][jnp.minimum(n['transit_target'], 23)].astype(jnp.float32)
        outlet = jnp.stack((n['gate_pos'][:, 0], jnp.full((24,), 63)), axis=1).astype(jnp.float32)
        endpoint = jnp.where((n['transit_target'] < 24)[:, :, None], endpoint, outlet[:, None, :])
        fraction = jnp.array([0.25, 0.5, 0.75], jnp.float32)[None, :, None]
        n['packet_pos'] = (n['gate_pos'][:, None, :] + fraction * (endpoint - n['gate_pos'][:, None, :])).reshape((72, 2)).astype(jnp.float32)
        n['packet_amount'] = n['transit'].reshape((72,))
        n['packet_target'] = n['transit_target'].reshape((72,))
        n['packet_active'] = n['packet_amount'] > 0
        n['terminated'] = jnp.array(False)
        n['done'] = n['t'] >= 1024
        n = self.refresh(n)
        n = jax.tree.map(lambda new, old: jnp.where(s['done'], old, new), n, s)
        reward = jnp.where(s['done'], jnp.zeros_like(reward), reward).astype(jnp.float32)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': (n['t'] >= 1024) & ~n['terminated']}
        return (self.observe(n), n, reward, jnp.full((384,), n['done'], jnp.bool_), info)
