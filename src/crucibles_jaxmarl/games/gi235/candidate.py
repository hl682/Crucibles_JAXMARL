import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 64
    observation_size = 910
    num_actions = 27
    max_steps = 896

    def distance(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=-1)

    def passable(self, s, positions):
        tile = s['tiles'][positions[:, 1], positions[:, 0]]
        gate = jnp.any((self.distance(positions, s['gate_pos']) <= 1) & (s['gate_open'] | (s['gate_health'] == 0))[None, :], axis=1)
        return (tile == 0) | gate

    def refresh(self, s):
        s = dict(s)
        d = self.distance(s['pos'], s['pos'])
        ix = jnp.argsort(d + jnp.arange(64, dtype=jnp.int32)[None, :] / 256.0, axis=1)[:, :4]
        v = jnp.take_along_axis(d, ix, axis=1) <= 6
        s['seen'] = ix.astype(jnp.int32)
        s['agent_vis'] = jnp.zeros((64, 64), jnp.bool_).at[jnp.arange(64, dtype=jnp.int32)[:, None], ix].set(v)
        s['agent_vis'] = s['agent_vis'].at[jnp.arange(64, dtype=jnp.int32), jnp.arange(64, dtype=jnp.int32)].set(True)
        for group in ('pasture', 'orchard', 'barn', 'trough', 'gate', 'facility', 'sheep'):
            s[group + '_vis'] = self.distance(s['pos'], s[group + '_pos']) <= 6
        s['facility_load'] = jnp.sum(s['sheep_pen'][:, None] == jnp.arange(4, dtype=jnp.int32)[None, :], axis=0).astype(jnp.int32)
        return s

    def packed(self, pos, data, vis):
        row = jnp.concatenate((pos.astype(jnp.float32) / 63.0, data), axis=1)
        return jnp.concatenate((vis[:, :, None], jnp.where(vis[:, :, None], row[None, :, :], 0)), axis=2).reshape((64, -1))

    def observe(self, s):
        crew = jnp.arange(64, dtype=jnp.int32) // 4
        own = jnp.stack((jnp.arange(64, dtype=jnp.int32) / 63.0, s['pos'][:, 0] / 63.0, s['pos'][:, 1] / 63.0, crew / 15.0, s['tool'] / 3.0, s['wood'] / 4.0, s['fodder'] / 4.0, s['water'] / 3.0, s['face'] / 4.0, jnp.full((64,), s['t'] / 896.0), jnp.full((64,), s['t'] // 128 % 3 / 2.0), s['wool_stock'][crew] / 40.0, s['fruit_stock'][crew] / 40.0), axis=1)
        data = jnp.stack((jnp.arange(64, dtype=jnp.int32) / 63.0, s['pos'][:, 0] / 63.0, s['pos'][:, 1] / 63.0, crew / 15.0, s['tool'] / 3.0, s['wood'] / 4.0, s['fodder'] / 4.0, s['water'] / 3.0, s['face'] / 4.0), axis=1)
        v = jnp.take_along_axis(s['agent_vis'], s['seen'], axis=1)
        people = jnp.concatenate((v[:, :, None], jnp.where(v[:, :, None], data[s['seen']], 0)), axis=2).reshape((64, 40))
        pasture = self.packed(s['pasture_pos'], jnp.stack((s['grass'] / 12.0, s['rotation'] / 2.0, s['grazed'] / 32.0), axis=1), s['pasture_vis'])
        orchard = self.packed(s['orchard_pos'], jnp.stack((s['orchard_growth'] / 8.0, s['branches'] / 8.0, s['orchard_water'] / 4.0), axis=1), s['orchard_vis'])
        barn = self.packed(s['barn_pos'], jnp.stack((s['shelter'] / 3.0, s['fodder_stock'] / 24.0, s['wool_stock'] / 40.0, s['fruit_stock'] / 40.0), axis=1), s['barn_vis'])
        trough = self.packed(s['trough_pos'], s['trough_water'][:, None] / 8.0, s['trough_vis'])
        gate = self.packed(s['gate_pos'], jnp.stack((s['gate_open'], s['gate_health'] / 3.0), axis=1), s['gate_vis'])
        facility = self.packed(s['facility_pos'], jnp.stack((s['facility_water'] / 12.0, s['facility_dirty'], s['facility_load'] / 4.0), axis=1), s['facility_vis'])
        sheep = self.packed(s['sheep_pos'], jnp.stack((s['sheep_owner'] / 15.0, s['sheep_wool'] / 6.0, s['sheep_hunger'] / 8.0, s['sheep_health'] / 3.0, s['sheep_sick'], (s['sheep_pen'] + 1) / 4.0, s['sheep_timer'] / 8.0), axis=1), s['sheep_vis'])
        off = jnp.stack((jnp.tile(jnp.arange(-2, 3, dtype=jnp.int32), 5), jnp.repeat(jnp.arange(-2, 3, dtype=jnp.int32), 5)), axis=1)
        xy = jnp.clip(s['pos'][:, None, :] + off[None, :, :], 0, 63)
        patch = s['tiles'][xy[:, :, 1], xy[:, :, 0]].astype(jnp.float32)
        return jnp.concatenate((own, people, pasture, orchard, barn, trough, gate, facility, sheep, patch), axis=1).astype(jnp.float32)

    def reset(self, key):
        c = jnp.arange(16, dtype=jnp.int32)
        center = jnp.stack((8 + 16 * (c % 4), 8 + 16 * (c // 4)), axis=1)
        ids = jnp.arange(64, dtype=jnp.int32)
        pos = center[ids // 4] + jnp.stack((ids % 2, ids % 4 // 2), axis=1)
        xx = jnp.arange(64, dtype=jnp.int32)[None, :]
        yy = jnp.arange(64, dtype=jnp.int32)[:, None]
        tiles = ((xx % 16 == 0) | (xx % 16 == 15) | (yy % 16 == 0) | (yy % 16 == 15)).astype(jnp.int32)
        k = jnp.arange(12, dtype=jnp.int32)
        gates = jnp.concatenate((jnp.stack((15 + 16 * (k % 3), 8 + 16 * (k // 3)), axis=1), jnp.stack((8 + 16 * (k % 4), 15 + 16 * (k // 4)), axis=1)), axis=0)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'tiles': tiles, 'pos': pos, 'active': jnp.ones((64,), jnp.bool_), 'tool': ids % 4, 'face': jnp.full((64,), 2, jnp.int32), 'wood': jnp.zeros((64,), jnp.int32), 'fodder': jnp.zeros((64,), jnp.int32), 'water': jnp.zeros((64,), jnp.int32), 'seen': jnp.zeros((64, 4), jnp.int32), 'agent_vis': jnp.zeros((64, 64), jnp.bool_), 'pasture_pos': center, 'grass': jax.random.randint(key, (16,), 6, 11).astype(jnp.int32), 'rotation': jnp.zeros((16,), jnp.int32), 'grazed': jnp.zeros((16,), jnp.int32), 'pasture_active': jnp.ones((16,), jnp.bool_), 'pasture_vis': jnp.zeros((64, 16), jnp.bool_), 'orchard_pos': center + jnp.array([-3, -3], jnp.int32), 'orchard_growth': jnp.full((16,), 4, jnp.int32), 'branches': jnp.full((16,), 8, jnp.int32), 'orchard_water': jnp.full((16,), 2, jnp.int32), 'orchard_active': jnp.ones((16,), jnp.bool_), 'orchard_vis': jnp.zeros((64, 16), jnp.bool_), 'barn_pos': center + jnp.array([0, -3], jnp.int32), 'shelter': jnp.zeros((16,), jnp.int32), 'fodder_stock': jnp.zeros((16,), jnp.int32), 'wool_stock': jnp.zeros((16,), jnp.int32), 'fruit_stock': jnp.zeros((16,), jnp.int32), 'barn_active': jnp.ones((16,), jnp.bool_), 'barn_vis': jnp.zeros((64, 16), jnp.bool_), 'trough_pos': center + jnp.array([3, 2], jnp.int32), 'trough_water': jnp.full((16,), 4, jnp.int32), 'trough_active': jnp.ones((16,), jnp.bool_), 'trough_vis': jnp.zeros((64, 16), jnp.bool_), 'gate_pos': gates, 'gate_open': jnp.zeros((24,), jnp.bool_), 'gate_health': jnp.full((24,), 3, jnp.int32), 'gate_active': jnp.ones((24,), jnp.bool_), 'gate_vis': jnp.zeros((64, 24), jnp.bool_), 'facility_pos': jnp.array([[10, 10], [42, 10], [10, 42], [42, 42]], jnp.int32), 'facility_water': jnp.full((4,), 12, jnp.int32), 'facility_dirty': jnp.zeros((4,), jnp.bool_), 'facility_load': jnp.zeros((4,), jnp.int32), 'facility_active': jnp.ones((4,), jnp.bool_), 'facility_vis': jnp.zeros((64, 4), jnp.bool_), 'sheep_pos': jnp.repeat(center, 2, axis=0) + jnp.tile(jnp.array([[1, 1], [2, 1]], jnp.int32), (16, 1)), 'sheep_owner': jnp.repeat(c, 2), 'sheep_wool': jnp.full((32,), 3, jnp.int32), 'sheep_hunger': jnp.zeros((32,), jnp.int32), 'sheep_health': jnp.full((32,), 3, jnp.int32), 'sheep_sick': jnp.arange(32, dtype=jnp.int32) == 31, 'sheep_pen': jnp.full((32,), -1, jnp.int32), 'sheep_timer': jnp.zeros((32,), jnp.int32), 'sheep_active': jnp.ones((32,), jnp.bool_), 'sheep_vis': jnp.zeros((64, 32), jnp.bool_), 'event_shear': jnp.zeros((64,), jnp.bool_), 'event_feed': jnp.zeros((64,), jnp.bool_), 'event_quarantine': jnp.zeros((64,), jnp.bool_), 'event_fruit': jnp.zeros((64,), jnp.bool_), 'event_rescue': jnp.zeros((64,), jnp.bool_)}
        s = self.refresh(s)
        return (self.observe(s), s)

    def legal_actions(self, s):
        t = s['tool']
        mask = jnp.ones((64, 27), jnp.bool_)
        for action in (5, 10, 13, 25):
            mask = mask.at[:, action].set(t == 0)
        for action in (7, 18):
            mask = mask.at[:, action].set((t == 2) & (s['wood'] > 0))
        mask = mask.at[:, 8].set(s['wood'] < 4)
        mask = mask.at[:, 10].set((t == 0) & (s['fodder'] < 4))
        mask = mask.at[:, 11].set((t == 3) & (s['water'] < 3))
        for action in (12, 22, 26):
            mask = mask.at[:, action].set((t == 3) & (s['water'] > 0))
        mask = mask.at[:, 14].set((t == 0) | (t == 3))
        mask = mask.at[:, 15].set(t == 1)
        for action in (17, 21, 23):
            mask = mask.at[:, action].set(s['fodder'] > 0)
        mask = mask.at[:, 24].set(s['fodder'] < 4)
        mask = mask.at[:, 25].set((t == 0) & (s['fodder'] > 0))
        return mask & (~s['done'] | (jnp.arange(27, dtype=jnp.int32)[None, :] == 0))

    def raw_step(self, key, state, actions):
        s = state
        ids = jnp.arange(64, dtype=jnp.int32)
        clipped = jnp.clip(actions, 0, 26)
        a = jnp.where((actions >= 0) & (actions < 27) & self.legal_actions(s)[ids, clipped], clipped, 0)
        n = dict(s)
        n['t'] = s['t'] + 1
        for name in ('event_shear', 'event_feed', 'event_quarantine', 'event_fruit', 'event_rescue'):
            n[name] = jnp.zeros((64,), jnp.bool_)
        n['grazed'] = jnp.zeros((16,), jnp.int32)
        drought = n['t'] // 128 % 3 == 2
        regrow = (n['t'] % 8 == 0) & ~drought
        n['grass'] = jnp.minimum(s['grass'] + regrow.astype(jnp.int32) * (1 + (s['rotation'] == 1).astype(jnp.int32)), 12)
        grow = (n['t'] % 16 == 0) & (s['orchard_water'] > 0)
        n['orchard_growth'] = jnp.minimum(s['orchard_growth'] + grow.astype(jnp.int32), 8)
        n['orchard_water'] = jnp.maximum(s['orchard_water'] - grow.astype(jnp.int32), 0)
        n['branches'] = jnp.minimum(s['branches'] + (n['t'] % 32 == 0).astype(jnp.int32), 8)
        n['facility_water'] = jnp.minimum(s['facility_water'] + 4 * ((n['t'] % 32 == 0) & ~drought).astype(jnp.int32), 12)
        n['gate_health'] = jnp.maximum(s['gate_health'] - ((n['t'] % 96 == 0) & s['gate_open']).astype(jnp.int32), 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        moving = (a >= 1) & (a <= 4)
        target = jnp.clip(s['pos'] + delta[jnp.minimum(a, 4)] * moving[:, None], 0, 63)
        n['pos'] = jnp.where(self.passable(n, target)[:, None], target, s['pos'])
        n['face'] = jnp.where(moving, a, s['face']).astype(jnp.int32)
        reward = -0.01 * (a >= 5).astype(jnp.float32)

        def work(i, pair):
            z, rew = pair
            z = dict(z)
            here = z['pos'][i]
            crew = i // 4
            sd = jnp.sum(jnp.abs(z['sheep_pos'] - here), axis=1)
            sheep = jnp.argmin(jnp.where(z['sheep_active'], sd, 1000))
            near = (sd[sheep] <= 2) & z['sheep_active'][sheep]
            helpers = jnp.any((jnp.sum(jnp.abs(z['pos'] - z['sheep_pos'][sheep]), axis=1) <= 1) & (a == 16) & (jnp.arange(64, dtype=jnp.int32) != i))
            herd = (a[i] == 5) & near & (z['sheep_pen'][sheep] < 0)
            push = jnp.clip(z['sheep_pos'][sheep] + delta[z['face'][i]], 0, 63)
            pushed = herd & self.passable(z, push[None, :])[0]
            z['sheep_pos'] = z['sheep_pos'].at[sheep].set(jnp.where(pushed, push, z['sheep_pos'][sheep]))
            gd = jnp.sum(jnp.abs(z['gate_pos'] - here), axis=1)
            gate = jnp.argmin(gd)
            toggle = (a[i] == 6) & (gd[gate] <= 2)
            repair = (a[i] == 7) & (gd[gate] <= 2) & (z['gate_health'][gate] < 3) & (z['wood'][i] > 0)
            z['gate_open'] = z['gate_open'].at[gate].set(z['gate_open'][gate] ^ toggle)
            z['gate_health'] = z['gate_health'].at[gate].add(repair.astype(jnp.int32))
            od = jnp.sum(jnp.abs(z['orchard_pos'] - here), axis=1)
            orchard = jnp.argmin(od)
            gather = (a[i] == 8) & (od[orchard] <= 1) & (z['branches'][orchard] > 0) & (z['wood'][i] < 4)
            fruit = (a[i] == 9) & (od[orchard] <= 1) & (z['orchard_growth'][orchard] == 8) & (z['fruit_stock'][crew] < 40)
            tend = (a[i] == 22) & (od[orchard] <= 1) & (z['orchard_water'][orchard] < 4) & (z['water'][i] > 0)
            z['branches'] = z['branches'].at[orchard].add(-gather.astype(jnp.int32))
            z['wood'] = z['wood'].at[i].add(gather.astype(jnp.int32) - repair.astype(jnp.int32))
            z['orchard_growth'] = z['orchard_growth'].at[orchard].set(jnp.where(fruit, 0, z['orchard_growth'][orchard]))
            z['orchard_water'] = z['orchard_water'].at[orchard].add(tend.astype(jnp.int32))
            z['fruit_stock'] = z['fruit_stock'].at[crew].add(fruit.astype(jnp.int32))
            pd = jnp.sum(jnp.abs(z['pasture_pos'] - here), axis=1)
            pasture = jnp.argmin(pd)
            cut_cost = jnp.where(z['rotation'][pasture] == 2, 1, 2)
            cut = (a[i] == 10) & (pd[pasture] <= 2) & (z['grass'][pasture] >= cut_cost) & (z['fodder'][i] < 4)
            z['grass'] = z['grass'].at[pasture].add(-cut_cost * cut.astype(jnp.int32))
            z['fodder'] = z['fodder'].at[i].add(cut.astype(jnp.int32))
            rotate = (a[i] == 19) & (pasture == crew) & (pd[pasture] <= 2)
            z['rotation'] = z['rotation'].at[pasture].set(jnp.where(rotate, (z['rotation'][pasture] + 1) % 3, z['rotation'][pasture]))
            fd = jnp.sum(jnp.abs(z['facility_pos'] - here), axis=1)
            facility = jnp.argmin(fd)
            draw = (a[i] == 11) & (fd[facility] <= 1) & (z['facility_water'][facility] > 0) & (z['water'][i] < 3)
            z['facility_water'] = z['facility_water'].at[facility].add(-draw.astype(jnp.int32))
            td = jnp.sum(jnp.abs(z['trough_pos'] - here), axis=1)
            trough = jnp.argmin(td)
            water = (a[i] == 12) & (td[trough] <= 1) & (z['trough_water'][trough] < 8) & (z['water'][i] > 0)
            z['trough_water'] = z['trough_water'].at[trough].add(water.astype(jnp.int32))
            clean = (a[i] == 26) & (fd[facility] <= 1) & (z['water'][i] > 0)
            z['facility_dirty'] = z['facility_dirty'].at[facility].set(z['facility_dirty'][facility] & ~clean)
            z['water'] = z['water'].at[i].add(draw.astype(jnp.int32) - water.astype(jnp.int32) - tend.astype(jnp.int32) - clean.astype(jnp.int32))
            pen_load = jnp.sum(z['sheep_pen'] == facility)
            confine = (a[i] == 13) & near & (fd[facility] <= 3) & (z['sheep_pen'][sheep] < 0) & (pen_load < 4)
            penxy = z['facility_pos'][facility] + jnp.stack((pen_load % 2, pen_load // 2))
            z['sheep_pen'] = z['sheep_pen'].at[sheep].set(jnp.where(confine, facility, z['sheep_pen'][sheep]))
            z['sheep_timer'] = z['sheep_timer'].at[sheep].set(jnp.where(confine, 8, z['sheep_timer'][sheep]))
            z['sheep_pos'] = z['sheep_pos'].at[sheep].set(jnp.where(confine, penxy, z['sheep_pos'][sheep]))
            release = (a[i] == 14) & near & (z['sheep_pen'][sheep] >= 0) & (z['sheep_timer'][sheep] == 0)
            z['sheep_pen'] = z['sheep_pen'].at[sheep].set(jnp.where(release, -1, z['sheep_pen'][sheep]))
            rackdist = jnp.sum(jnp.abs(z['facility_pos'] - z['sheep_pos'][sheep]), axis=1)
            rack = jnp.argmin(rackdist)
            shear = (a[i] == 15) & (sd[sheep] <= 1) & z['sheep_active'][sheep] & (rackdist[rack] <= 2) & (z['sheep_wool'][sheep] >= 3) & helpers & (z['wool_stock'][crew] < 40)
            contaminated = shear & z['facility_dirty'][rack]
            z['facility_dirty'] = z['facility_dirty'].at[rack].set(z['facility_dirty'][rack] | shear & z['sheep_sick'][sheep])
            z['sheep_sick'] = z['sheep_sick'].at[sheep].set(z['sheep_sick'][sheep] | contaminated)
            z['sheep_wool'] = z['sheep_wool'].at[sheep].add(-3 * shear.astype(jnp.int32))
            z['wool_stock'] = z['wool_stock'].at[crew].add(shear.astype(jnp.int32))
            feed = (a[i] == 17) & (sd[sheep] <= 1) & z['sheep_active'][sheep] & (z['fodder'][i] > 0)
            rescue = (a[i] == 25) & near & (z['sheep_hunger'][sheep] >= 6) & helpers & (z['fodder'][i] > 0)
            z['sheep_hunger'] = z['sheep_hunger'].at[sheep].set(jnp.where(feed | rescue, 0, z['sheep_hunger'][sheep]))
            z['sheep_health'] = z['sheep_health'].at[sheep].set(jnp.minimum(z['sheep_health'][sheep] + feed.astype(jnp.int32), 3))
            z['sheep_pos'] = z['sheep_pos'].at[sheep].set(jnp.where(rescue, here, z['sheep_pos'][sheep]))
            z['sheep_pen'] = z['sheep_pen'].at[sheep].set(jnp.where(rescue, -1, z['sheep_pen'][sheep]))
            z['fodder'] = z['fodder'].at[i].add(-feed.astype(jnp.int32) - rescue.astype(jnp.int32))
            bd = jnp.sum(jnp.abs(z['barn_pos'] - here), axis=1)
            barn = jnp.argmin(bd)
            build = (a[i] == 18) & (bd[barn] <= 1) & (z['shelter'][barn] < 3) & (z['wood'][i] > 0)
            z['shelter'] = z['shelter'].at[barn].add(build.astype(jnp.int32))
            z['wood'] = z['wood'].at[i].add(-build.astype(jnp.int32))
            store = (a[i] == 23) & (bd[barn] <= 1) & (z['fodder_stock'][barn] < 24) & (z['fodder'][i] > 0)
            take = (a[i] == 24) & (bd[barn] <= 1) & (z['fodder_stock'][barn] > 0) & (z['fodder'][i] < 4)
            change = store.astype(jnp.int32) - take.astype(jnp.int32)
            z['fodder_stock'] = z['fodder_stock'].at[barn].add(change)
            z['fodder'] = z['fodder'].at[i].add(-change)
            nd = jnp.sum(jnp.abs(z['pos'] - here), axis=1)
            friend = jnp.argmin(jnp.where(jnp.arange(64, dtype=jnp.int32) != i, nd, 1000))
            lend = (a[i] == 20) & (nd[friend] <= 1)
            ta, tb = (z['tool'][i], z['tool'][friend])
            z['tool'] = z['tool'].at[i].set(jnp.where(lend, tb, ta)).at[friend].set(jnp.where(lend, ta, tb))
            give = (a[i] == 21) & (nd[friend] <= 1) & (z['fodder'][i] > 0) & (z['fodder'][friend] < 4)
            z['fodder'] = z['fodder'].at[i].add(-give.astype(jnp.int32)).at[friend].add(give.astype(jnp.int32))
            z['event_shear'] = z['event_shear'].at[i].set(shear)
            z['event_feed'] = z['event_feed'].at[i].set(feed)
            z['event_quarantine'] = z['event_quarantine'].at[i].set(confine | release)
            z['event_fruit'] = z['event_fruit'].at[i].set(fruit)
            z['event_rescue'] = z['event_rescue'].at[i].set(rescue)
            rew = rew.at[i].add(0.75 * shear + 0.4 * fruit + 0.15 * rescue - 0.25 * contaminated)
            rew = rew + (jnp.arange(64, dtype=jnp.int32) // 4 == crew).astype(jnp.float32) * (0.25 * shear + 0.2 * fruit) / 4.0
            return (z, rew)
        n, reward = jax.lax.fori_loop(0, 64, work, (n, reward))
        score = 4 * n['grass'][None, :] - self.distance(n['sheep_pos'], n['pasture_pos'])
        target_pasture = jnp.argmax(score, axis=1)
        gap = n['pasture_pos'][target_pasture] - n['sheep_pos']
        dx = jnp.sign(gap[:, 0]) * (jnp.abs(gap[:, 0]) >= jnp.abs(gap[:, 1]))
        dy = jnp.sign(gap[:, 1]) * (jnp.abs(gap[:, 0]) < jnp.abs(gap[:, 1]))
        move = jnp.stack((dx, dy), axis=1).astype(jnp.int32)
        to = jnp.clip(n['sheep_pos'] + move, 0, 63)
        roam = (n['t'] % 2 == 0) & (n['sheep_pen'] < 0) & n['sheep_active'] & self.passable(n, to)
        n['sheep_pos'] = jnp.where(roam[:, None], to, n['sheep_pos'])
        n['sheep_timer'] = jnp.maximum(n['sheep_timer'] - 1, 0)
        cured = (n['sheep_pen'] >= 0) & (n['sheep_timer'] == 0) & ~n['facility_dirty'][jnp.maximum(n['sheep_pen'], 0)]
        contact = self.distance(n['sheep_pos'], n['sheep_pos']) <= 1
        spread = jnp.any(contact & n['sheep_sick'][None, :] & (n['sheep_pen'][None, :] < 0) & n['sheep_active'][None, :], axis=1)
        n['sheep_sick'] = (n['sheep_sick'] | spread & (n['t'] % 8 == 0) & (n['sheep_pen'] < 0)) & ~cured

        def graze(j, pair):
            z, rew = pair
            z = dict(z)
            region = z['sheep_pos'][j, 1] // 16 * 4 + z['sheep_pos'][j, 0] // 16
            hungry_tick = (z['t'] % 4 == 0) & z['sheep_active'][j] & (z['sheep_pen'][j] < 0)
            eat = hungry_tick & (z['grass'][region] > 0)
            z['grass'] = z['grass'].at[region].add(-eat.astype(jnp.int32))
            z['grazed'] = z['grazed'].at[region].add(eat.astype(jnp.int32))
            hunger = jnp.clip(z['sheep_hunger'][j] + hungry_tick.astype(jnp.int32) - 2 * eat.astype(jnp.int32), 0, 8)
            drink = hungry_tick & (z['trough_water'][region] > 0) & (jnp.sum(jnp.abs(z['sheep_pos'][j] - z['trough_pos'][region])) <= 3)
            z['trough_water'] = z['trough_water'].at[region].add(-drink.astype(jnp.int32))
            sheltered = (z['shelter'][region] >= 2) & (jnp.sum(jnp.abs(z['sheep_pos'][j] - z['barn_pos'][region])) <= 4)
            hurt = (z['t'] % 16 == 0) & z['sheep_active'][j] & ((hunger >= 7) | drought & ~drink & ~sheltered | z['sheep_sick'][j])
            z['sheep_health'] = z['sheep_health'].at[j].set(jnp.maximum(z['sheep_health'][j] - hurt.astype(jnp.int32), 0))
            z['sheep_hunger'] = z['sheep_hunger'].at[j].set(hunger)
            wool = (z['t'] % 16 == 0) & (hunger <= 3) & ~z['sheep_sick'][j] & z['sheep_active'][j]
            z['sheep_wool'] = z['sheep_wool'].at[j].set(jnp.minimum(z['sheep_wool'][j] + wool.astype(jnp.int32), 6))
            rew = rew - (jnp.arange(64, dtype=jnp.int32) // 4 == region).astype(jnp.float32) * 0.04 * eat
            rew = rew - (jnp.arange(64, dtype=jnp.int32) // 4 == z['sheep_owner'][j]).astype(jnp.float32) * 0.1 * hurt
            return (z, rew)
        n, reward = jax.lax.fori_loop(0, 32, graze, (n, reward))
        n['sheep_active'] = n['sheep_health'] > 0
        n['terminated'] = ~jnp.any(n['sheep_active'])
        n['done'] = n['terminated'] | (n['t'] >= 896)
        n = self.refresh(n)
        n = jax.tree.map(lambda new, old: jnp.where(s['done'], old, new), n, s)
        reward = jnp.where(s['done'], jnp.zeros_like(reward), reward).astype(jnp.float32)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': (n['t'] >= 896) & ~n['terminated']}
        return (self.observe(n), n, reward, jnp.full((64,), n['done'], jnp.bool_), info)
