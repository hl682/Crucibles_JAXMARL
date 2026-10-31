"""Pollinator Paths: original unexecuted spatial source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 317
    num_actions = 16
    max_steps = 168

    def sync(self, s):
        s = dict(s)
        s['shelter_pos'] = jnp.where((s['shelter_holder'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['shelter_holder'], 0)], s['shelter_pos'])
        s['daylight'] = s['t'] % 24 < 18
        s['agent_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=2) <= 4
        s['crop_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['crop_pos'][None, :, :]), axis=2) <= 4
        s['gate_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['gate_pos'][None, :, :]), axis=2) <= 4
        s['shelter_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['shelter_pos'][None, :, :]), axis=2) <= 4
        s['bee_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['bee_pos'][None, :, :]), axis=2) <= 4
        s['strip_vis'] = (jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['strip_pos'][None, :, :]), axis=2) <= 4) & s['strip_active'][None, :]
        s['well_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['well_pos'][None, :, :]), axis=2) <= 4
        return s

    def reset(self, key):
        grid = jnp.zeros((9, 13), jnp.int32).at[:, 4].set(1).at[:, 8].set(1)
        grid = grid.at[2, 4].set(2).at[6, 8].set(2)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'success': jnp.array(False), 'failure': jnp.array(False), 'daylight': jnp.array(True), 'map': grid,
            'agent_pos': jnp.array([[3, 2], [6, 4], [9, 6]], jnp.int32),
            'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
            'held_shelter': jnp.full(3, -1, jnp.int32), 'water': jnp.zeros(3, jnp.int32),
            'produce': jnp.zeros(3, jnp.int32), 'produce_origin': jnp.full(3, -1, jnp.int32),
            'seeds': jnp.full(3, 3, jnp.int32), 'wood': jnp.full(3, 2, jnp.int32),
            'work': jnp.zeros(3, jnp.int32), 'blocked': jnp.zeros(3, jnp.bool_),
            'crop_pos': jnp.array([[2, 2], [2, 7], [6, 2], [6, 6], [10, 2], [10, 7]], jnp.int32),
            'crop_active': jnp.ones(6, jnp.bool_), 'crop_vis': jnp.ones((3, 6), jnp.bool_),
            'crop_owner': jnp.array([0, 0, 1, 1, 2, 2], jnp.int32), 'crop_stage': jnp.ones(6, jnp.int32),
            'moisture': jnp.full(6, 3, jnp.int32), 'visits': jnp.zeros(6, jnp.int32),
            'nectar': jnp.full(6, 3, jnp.int32), 'growth': jnp.zeros(6, jnp.int32),
            'bloom_age': jnp.zeros(6, jnp.int32), 'stress': jnp.zeros(6, jnp.int32),
            'visited_now': jnp.zeros(6, jnp.bool_),
            'gate_pos': jnp.array([[4, 2], [8, 6]], jnp.int32),
            'gate_active': jnp.ones(2, jnp.bool_), 'gate_vis': jnp.ones((3, 2), jnp.bool_),
            'gate_open': jnp.zeros(2, jnp.bool_), 'gate_health': jnp.full(2, 3, jnp.int32),
            'shelter_pos': jnp.array([[2, 4], [10, 4]], jnp.int32),
            'shelter_active': jnp.ones(2, jnp.bool_), 'shelter_vis': jnp.ones((3, 2), jnp.bool_),
            'shelter_holder': jnp.full(2, -1, jnp.int32),
            'bee_pos': jnp.array([[2, 4], [2, 4], [10, 4], [10, 4]], jnp.int32),
            'bee_active': jnp.ones(4, jnp.bool_), 'bee_vis': jnp.ones((3, 4), jnp.bool_),
            'bee_home': jnp.array([0, 0, 1, 1], jnp.int32), 'bee_budget': jnp.full(4, 6, jnp.int32),
            'bee_return': jnp.zeros(4, jnp.bool_),
            'strip_pos': jnp.array([[3, 2], [6, 4], [9, 6]], jnp.int32),
            'strip_active': jnp.zeros(3, jnp.bool_), 'strip_vis': jnp.zeros((3, 3), jnp.bool_),
            'strip_nectar': jnp.zeros(3, jnp.int32), 'strip_age': jnp.zeros(3, jnp.int32),
            'well_pos': jnp.array([[1, 4], [6, 4], [11, 4]], jnp.int32),
            'well_active': jnp.ones(3, jnp.bool_), 'well_vis': jnp.ones((3, 3), jnp.bool_),
            'well_water': jnp.full(3, 6, jnp.int32),
            'stall_pos': jnp.array([[1, 7], [6, 7], [11, 7], [6, 0]], jnp.int32),
            'stall_active': jnp.ones(4, jnp.bool_), 'stall_vis': jnp.ones((3, 4), jnp.bool_),
            'harvested': jnp.zeros(3, jnp.int32), 'community': jnp.zeros(3, jnp.int32),
            'private_stock': jnp.zeros(3, jnp.int32), 'planted': jnp.zeros(3, jnp.int32),
            'repairs': jnp.zeros(3, jnp.int32), 'donated_now': jnp.zeros(3, jnp.int32),
            'stored_now': jnp.zeros(3, jnp.int32),
        }
        s = self.sync(s)
        return self.observe(s), s

    def walk_grid(self, s):
        walk = s['map'] != 1
        walk = walk.at[s['gate_pos'][:, 1], s['gate_pos'][:, 0]].set(s['gate_open'])
        return walk

    def walk_targets(self, s):
        p = s['agent_pos'][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        inside = jnp.all((p >= 0) & (p <= jnp.array([12, 8], jnp.int32)), axis=2)
        floor = self.walk_grid(s)[jnp.clip(p[:, :, 1], 0, 8), jnp.clip(p[:, :, 0], 0, 12)]
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=3), axis=2)
        return p, inside & floor & ~occupied

    def legal_actions(self, s):
        p = s['agent_pos']
        free = (s['held_shelter'] < 0) & (s['produce'] == 0)
        shnear = jnp.sum(jnp.abs(p[:, None, :] - s['shelter_pos'][None, :, :]), axis=2) <= 1
        gn = jnp.sum(jnp.abs(p[:, None, :] - s['gate_pos'][None, :, :]), axis=2) <= 1
        gnfree = ~jnp.any(jnp.all(s['agent_pos'][:, None, :] == s['gate_pos'][None, :, :], axis=2), axis=0)
        beeongate = jnp.any(jnp.all(s['bee_pos'][:, None, :] == s['gate_pos'][None, :, :], axis=2), axis=0)
        gnfree = gnfree & ~beeongate
        cn = jnp.sum(jnp.abs(p[:, None, :] - s['crop_pos'][None, :, :]), axis=2) <= 1
        sn = jnp.sum(jnp.abs(p[:, None, :] - s['strip_pos'][None, :, :]), axis=2) <= 1
        wn = jnp.sum(jnp.abs(p[:, None, :] - s['well_pos'][None, :, :]), axis=2) <= 1
        plantcell = ~jnp.any(jnp.all(p[:, None, :] == s['crop_pos'][None, :, :], axis=2), axis=1) & ~jnp.any(jnp.all(p[:, None, :] == s['strip_pos'][None, :, :], axis=2) & s['strip_active'][None, :], axis=1) & (s['map'][p[:, 1], p[:, 0]] == 0)
        wetcrop = cn & (s['crop_stage'] < 3)[None, :] & ((s['moisture'] < 4) | ((s['crop_stage'] == 1) & (s['nectar'] < 3)))[None, :]
        wetstrip = sn & s['strip_active'][None, :] & (s['strip_nectar'] < 4)[None, :]
        harvest = cn & ((s['crop_stage'] == 1) | (s['crop_stage'] == 2))[None, :]
        _, walk = self.walk_targets(s)
        m = jnp.zeros((3, 16), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(walk)
        m = m.at[:, 5].set(free & jnp.any(shnear & (s['shelter_holder'] < 0)[None, :], axis=1))
        shelterhere = jnp.any(jnp.all(p[:, None, :] == s['shelter_pos'][None, :, :], axis=2) & (s['shelter_holder'] < 0)[None, :], axis=1)
        m = m.at[:, 6].set((s['held_shelter'] >= 0) & ~shelterhere & (s['map'][p[:, 1], p[:, 0]] == 0))
        m = m.at[:, 7].set(jnp.any(gn & (s['gate_health'] > 0)[None, :] & (~s['gate_open'] | gnfree)[None, :], axis=1))
        m = m.at[:, 8].set((s['wood'] > 0) & jnp.any(gn & (s['gate_health'] < 3)[None, :], axis=1))
        m = m.at[:, 9].set((s['seeds'] > 0) & ~s['strip_active'] & plantcell)
        m = m.at[:, 10].set((s['water'] > 0) & (jnp.any(wetcrop, axis=1) | jnp.any(wetstrip, axis=1)))
        m = m.at[:, 11].set((s['water'] < 2) & jnp.any(wn & (s['well_water'] > 0)[None, :], axis=1))
        m = m.at[:, 12].set(free & jnp.any(harvest, axis=1))
        m = m.at[:, 13].set((s['seeds'] > 0) & jnp.any(cn & (s['crop_stage'] >= 3)[None, :], axis=1))
        m = m.at[:, 14].set((s['produce'] > 0) & (jnp.sum(jnp.abs(p - s['stall_pos'][:3]), axis=1) <= 1))
        m = m.at[:, 15].set((s['produce'] > 0) & (jnp.sum(jnp.abs(p - s['stall_pos'][3]), axis=1) <= 1))
        return jnp.where(s['done'], jnp.arange(16)[None, :] == 0, m)

    def distances(self, origin, walk):
        d = jnp.full((9, 13), 99, jnp.int32).at[origin[1], origin[0]].set(0)
        def relax(k, field):
            north = jnp.pad(field[:-1, :], ((1, 0), (0, 0)), constant_values=99)
            south = jnp.pad(field[1:, :], ((0, 1), (0, 0)), constant_values=99)
            west = jnp.pad(field[:, :-1], ((0, 0), (1, 0)), constant_values=99)
            east = jnp.pad(field[:, 1:], ((0, 0), (0, 1)), constant_values=99)
            best = jnp.minimum(jnp.minimum(north, south), jnp.minimum(west, east)) + 1
            return jnp.where(walk, jnp.minimum(field, best), 99)
        return jax.lax.fori_loop(0, 24, relax, d)

    def observe(self, s):
        public = jnp.concatenate((jnp.stack((s['t'], s['done'], s['terminated'], s['truncated'], s['success'], s['failure'], s['daylight'])), s['map'].ravel(), s['community'], s['private_stock'], s['donated_now'], s['stored_now'], s['stall_pos'].ravel()))
        agents = jnp.concatenate((s['agent_pos'], s['held_shelter'][:, None], s['water'][:, None], s['produce'][:, None], s['produce_origin'][:, None], s['seeds'][:, None], s['wood'][:, None]), axis=1)
        crops = jnp.concatenate((s['crop_pos'], s['crop_owner'][:, None], s['crop_stage'][:, None], s['moisture'][:, None], s['visits'][:, None], s['nectar'][:, None], s['growth'][:, None], s['bloom_age'][:, None], s['stress'][:, None], s['visited_now'][:, None]), axis=1)
        gates = jnp.concatenate((s['gate_pos'], s['gate_open'][:, None], s['gate_health'][:, None]), axis=1)
        shelters = jnp.concatenate((s['shelter_pos'], s['shelter_holder'][:, None]), axis=1)
        bees = jnp.concatenate((s['bee_pos'], s['bee_home'][:, None], s['bee_budget'][:, None], s['bee_return'][:, None]), axis=1)
        strips = jnp.concatenate((s['strip_pos'], s['strip_nectar'][:, None], s['strip_age'][:, None]), axis=1)
        wells = jnp.concatenate((s['well_pos'], s['well_water'][:, None]), axis=1)
        rows = []
        for i in range(3):
            own = jnp.stack((i, s['work'][i], s['blocked'][i], s['planted'][i], s['repairs'][i]))
            row = jnp.concatenate((public, own, s['agent_vis'][i], jnp.where(s['agent_vis'][i, :, None], agents, -1).ravel(), s['crop_vis'][i], jnp.where(s['crop_vis'][i, :, None], crops, -1).ravel(), s['gate_vis'][i], jnp.where(s['gate_vis'][i, :, None], gates, -1).ravel(), s['shelter_vis'][i], jnp.where(s['shelter_vis'][i, :, None], shelters, -1).ravel(), s['bee_vis'][i], jnp.where(s['bee_vis'][i, :, None], bees, -1).ravel(), s['strip_vis'][i], jnp.where(s['strip_vis'][i, :, None], strips, -1).ravel(), s['well_vis'][i], jnp.where(s['well_vis'][i, :, None], wells, -1).ravel()))
            rows = rows + [row.astype(jnp.float32)]
        return jnp.stack(rows)

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 15)
        mask = self.legal_actions(s)
        a = jnp.where((actions >= 0) & (actions < 16) & mask[jnp.arange(3), clipped], clipped, 0)
        n['t'] = s['t'] + 1
        n['work'] = s['work'] + (a != 0).astype(jnp.int32)
        n['blocked'] = actions != a
        n['visited_now'] = jnp.zeros(6, jnp.bool_)
        n['donated_now'] = jnp.zeros(3, jnp.int32)
        n['stored_now'] = jnp.zeros(3, jnp.int32)
        targets, walk = self.walk_targets(s)
        target = targets[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        wants = (a >= 1) & (a <= 4) & walk[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
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
            sh = (n['shelter_holder'] < 0) & (jnp.sum(jnp.abs(n['shelter_pos'] - p), axis=1) <= 1)
            h = jnp.argmax(sh)
            take = (a[i] == 5) & jnp.any(sh)
            n['shelter_holder'] = n['shelter_holder'].at[h].set(jnp.where(take, i, n['shelter_holder'][h]))
            n['held_shelter'] = n['held_shelter'].at[i].set(jnp.where(take, h, n['held_shelter'][i]))
            h = jnp.maximum(n['held_shelter'][i], 0)
            occupied = jnp.any(jnp.all(n['shelter_pos'] == p, axis=1) & (n['shelter_holder'] < 0))
            drop = (a[i] == 6) & ~occupied
            n['shelter_holder'] = n['shelter_holder'].at[h].set(jnp.where(drop, -1, n['shelter_holder'][h]))
            n['shelter_pos'] = n['shelter_pos'].at[h].set(jnp.where(drop, p, n['shelter_pos'][h]))
            n['held_shelter'] = n['held_shelter'].at[i].set(jnp.where(drop, -1, n['held_shelter'][i]))
            near = jnp.sum(jnp.abs(n['gate_pos'] - p), axis=1) <= 1
            occupiedgate = jnp.any(jnp.all(n['agent_pos'][:, None, :] == n['gate_pos'][None, :, :], axis=2), axis=0) | jnp.any(jnp.all(n['bee_pos'][:, None, :] == n['gate_pos'][None, :, :], axis=2), axis=0)
            usable = near & (n['gate_health'] > 0) & (~n['gate_open'] | ~occupiedgate)
            g = jnp.argmax(usable)
            toggle = (a[i] == 7) & jnp.any(usable)
            n['gate_open'] = n['gate_open'].at[g].set(n['gate_open'][g] ^ toggle)
            damaged = near & (n['gate_health'] < 3)
            g = jnp.argmax(damaged)
            repair = (a[i] == 8) & jnp.any(damaged) & (n['wood'][i] > 0)
            n['gate_health'] = n['gate_health'].at[g].set(jnp.where(repair, 3, n['gate_health'][g]))
            n['wood'] = n['wood'].at[i].add(-repair.astype(jnp.int32))
            n['repairs'] = n['repairs'].at[i].add(repair.astype(jnp.int32))
            stripbusy = jnp.any(jnp.all(n['strip_pos'] == p, axis=1) & n['strip_active'])
            plant = (a[i] == 9) & ~stripbusy & ~n['strip_active'][i] & (n['seeds'][i] > 0)
            n['strip_pos'] = n['strip_pos'].at[i].set(jnp.where(plant, p, n['strip_pos'][i]))
            n['strip_active'] = n['strip_active'].at[i].set(n['strip_active'][i] | plant)
            n['strip_age'] = n['strip_age'].at[i].set(jnp.where(plant, 0, n['strip_age'][i]))
            n['strip_nectar'] = n['strip_nectar'].at[i].set(jnp.where(plant, 4, n['strip_nectar'][i]))
            cn = jnp.sum(jnp.abs(n['crop_pos'] - p), axis=1) <= 1
            wetcrop = cn & (n['crop_stage'] < 3) & ((n['moisture'] < 4) | ((n['crop_stage'] == 1) & (n['nectar'] < 3)))
            q = jnp.argmax(wetcrop)
            watercrop = (a[i] == 10) & jnp.any(wetcrop) & (n['water'][i] > 0)
            n['moisture'] = n['moisture'].at[q].set(jnp.where(watercrop, jnp.minimum(n['moisture'][q] + 2, 4), n['moisture'][q]))
            n['nectar'] = n['nectar'].at[q].set(jnp.where(watercrop & (n['crop_stage'][q] == 1), jnp.minimum(n['nectar'][q] + 1, 3), n['nectar'][q]))
            sn = (jnp.sum(jnp.abs(n['strip_pos'] - p), axis=1) <= 1) & n['strip_active'] & (n['strip_nectar'] < 4)
            q = jnp.argmax(sn)
            waterstrip = (a[i] == 10) & ~watercrop & jnp.any(sn) & (n['water'][i] > 0)
            n['strip_nectar'] = n['strip_nectar'].at[q].set(jnp.where(waterstrip, jnp.minimum(n['strip_nectar'][q] + 2, 4), n['strip_nectar'][q]))
            n['water'] = n['water'].at[i].add(-watercrop.astype(jnp.int32) - waterstrip.astype(jnp.int32))
            wn = (jnp.sum(jnp.abs(n['well_pos'] - p), axis=1) <= 1) & (n['well_water'] > 0)
            q = jnp.argmax(wn)
            fill = (a[i] == 11) & jnp.any(wn) & (n['water'][i] < 2)
            n['water'] = n['water'].at[i].add(fill.astype(jnp.int32))
            n['well_water'] = n['well_water'].at[q].add(-fill.astype(jnp.int32))
            fruit = cn & ((n['crop_stage'] == 1) | (n['crop_stage'] == 2))
            q = jnp.argmax(fruit)
            harvest = (a[i] == 12) & jnp.any(fruit) & (n['produce'][i] == 0)
            amount = jnp.where(n['crop_stage'][q] == 2, 2, 1)
            origin = n['crop_owner'][q]
            n['produce'] = n['produce'].at[i].set(jnp.where(harvest, amount, n['produce'][i]))
            n['produce_origin'] = n['produce_origin'].at[i].set(jnp.where(harvest, origin, n['produce_origin'][i]))
            n['harvested'] = n['harvested'].at[origin].add(jnp.where(harvest, amount, 0))
            n['crop_stage'] = n['crop_stage'].at[q].set(jnp.where(harvest, 3, n['crop_stage'][q]))
            n['nectar'] = n['nectar'].at[q].set(jnp.where(harvest, 0, n['nectar'][q]))
            empty = cn & (n['crop_stage'] >= 3)
            q = jnp.argmax(empty)
            reseed = (a[i] == 13) & jnp.any(empty) & (n['seeds'][i] > 0)
            n['crop_stage'] = n['crop_stage'].at[q].set(jnp.where(reseed, 0, n['crop_stage'][q]))
            n['moisture'] = n['moisture'].at[q].set(jnp.where(reseed, 2, n['moisture'][q]))
            n['visits'] = n['visits'].at[q].set(jnp.where(reseed, 0, n['visits'][q]))
            n['growth'] = n['growth'].at[q].set(jnp.where(reseed, 0, n['growth'][q]))
            n['bloom_age'] = n['bloom_age'].at[q].set(jnp.where(reseed, 0, n['bloom_age'][q]))
            n['stress'] = n['stress'].at[q].set(jnp.where(reseed, 0, n['stress'][q]))
            n['seeds'] = n['seeds'].at[i].add(-plant.astype(jnp.int32) - reseed.astype(jnp.int32))
            n['planted'] = n['planted'].at[i].add(plant.astype(jnp.int32) + reseed.astype(jnp.int32))
            origin = jnp.maximum(n['produce_origin'][i], 0)
            stash = a[i] == 14
            donate = a[i] == 15
            qty = n['produce'][i]
            n['private_stock'] = n['private_stock'].at[origin].add(jnp.where(stash, qty, 0))
            n['community'] = n['community'].at[origin].add(jnp.where(donate, qty, 0))
            n['stored_now'] = n['stored_now'].at[i].set(jnp.where(stash, qty, 0))
            n['donated_now'] = n['donated_now'].at[i].set(jnp.where(donate, qty, 0))
            bonus = bonus.at[i].add(qty.astype(jnp.float32) * (0.6 * stash.astype(jnp.float32) + 0.1 * donate.astype(jnp.float32))).at[origin].add(0.2 * qty.astype(jnp.float32) * stash.astype(jnp.float32))
            n['produce'] = n['produce'].at[i].set(jnp.where(stash | donate, 0, qty))
            n['produce_origin'] = n['produce_origin'].at[i].set(jnp.where(stash | donate, -1, n['produce_origin'][i]))
        n = self.sync(n)
        walk = self.walk_grid(n)
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]], jnp.int32)
        flowers = jnp.concatenate((n['crop_pos'], n['strip_pos']), axis=0)
        for k in range(4):
            b = (s['t'] + k) % 4
            nest = n['shelter_pos'][n['bee_home'][b]]
            held = n['shelter_holder'][n['bee_home'][b]] >= 0
            atnest = jnp.all(n['bee_pos'][b] == nest)
            fresh = atnest & ~held & n['daylight']
            returning = jnp.where(fresh, False, n['bee_return'][b]) | held | ~n['daylight'] | ((n['bee_budget'][b] <= 0) & ~fresh)
            budget = jnp.where(fresh, 6, n['bee_budget'][b])
            field = self.distances(n['bee_pos'][b], walk)
            d = field[flowers[:, 1], flowers[:, 0]]
            available = jnp.concatenate(((n['crop_stage'] == 1) & (n['nectar'] > 0), n['strip_active'] & (n['strip_nectar'] > 0)))
            eligible = available & (jnp.sum(jnp.abs(flowers - nest), axis=1) <= 6) & (d < 99) & n['daylight'] & ~held
            choice = jnp.argmin(jnp.where(eligible, d * 10 + jnp.arange(9), 9999))
            returning = returning | ~jnp.any(eligible)
            goal = jnp.where(returning, nest, flowers[choice])
            togoal = self.distances(goal, walk)
            options = n['bee_pos'][b] + delta
            inside = jnp.all((options >= 0) & (options <= jnp.array([12, 8], jnp.int32)), axis=1)
            ox = jnp.clip(options[:, 0], 0, 12)
            oy = jnp.clip(options[:, 1], 0, 8)
            score = jnp.where(inside & walk[oy, ox], togoal[oy, ox], 999)
            direction = jnp.argmin(score)
            canstep = jnp.min(score) < 99
            pos = jnp.where(canstep, options[direction], n['bee_pos'][b])
            visit = ~returning & jnp.all(pos == flowers[choice]) & eligible[choice]
            c = jnp.minimum(choice, 5)
            q = jnp.clip(choice - 6, 0, 2)
            cv = visit & (choice < 6)
            sv = visit & (choice >= 6)
            n['visits'] = n['visits'].at[c].set(jnp.minimum(n['visits'][c] + cv.astype(jnp.int32), 3))
            n['nectar'] = n['nectar'].at[c].add(-cv.astype(jnp.int32))
            n['visited_now'] = n['visited_now'].at[c].set(n['visited_now'][c] | cv)
            n['strip_nectar'] = n['strip_nectar'].at[q].add(-sv.astype(jnp.int32))
            n['bee_pos'] = n['bee_pos'].at[b].set(pos)
            n['bee_budget'] = n['bee_budget'].at[b].set(jnp.maximum(budget - (~returning).astype(jnp.int32), 0))
            n['bee_return'] = n['bee_return'].at[b].set(returning)
        activecrop = n['crop_stage'] < 3
        n['moisture'] = jnp.maximum(n['moisture'] - (activecrop & (n['t'] % 5 == 0)).astype(jnp.int32), 0)
        n['stress'] = jnp.where(activecrop, jnp.clip(n['stress'] + jnp.where(n['moisture'] == 0, 1, -1), 0, 3), n['stress']).astype(jnp.int32)
        grow = (n['moisture'] > 0) & (((n['crop_stage'] == 0) & (n['t'] % 3 == 0)) | ((n['crop_stage'] == 1) & (n['visits'] >= 2)))
        n['growth'] = jnp.minimum(n['growth'] + grow.astype(jnp.int32), 3)
        bloom = (n['crop_stage'] == 0) & (n['growth'] >= 3)
        ripe = (n['crop_stage'] == 1) & (n['growth'] >= 3)
        n['bloom_age'] = jnp.where(bloom, 0, jnp.minimum(n['bloom_age'] + (n['crop_stage'] == 1).astype(jnp.int32), 18))
        dead = activecrop & ((n['stress'] >= 3) | ((n['crop_stage'] == 1) & (n['bloom_age'] >= 18)))
        n['crop_stage'] = jnp.where(dead, 4, jnp.where(bloom, 1, jnp.where(ripe, 2, n['crop_stage']))).astype(jnp.int32)
        n['growth'] = jnp.where(bloom, 0, n['growth'])
        n['nectar'] = jnp.where(bloom & ~dead, 3, jnp.where((n['crop_stage'] != 1), 0, n['nectar'])).astype(jnp.int32)
        n['strip_age'] = jnp.minimum(n['strip_age'] + n['strip_active'].astype(jnp.int32), 12)
        n['strip_active'] = n['strip_active'] & (n['strip_age'] < 12)
        n['strip_nectar'] = jnp.where(n['strip_active'], n['strip_nectar'], 0)
        n['well_water'] = jnp.minimum(n['well_water'] + (n['t'] % 4 == 0).astype(jnp.int32), 6)
        n['gate_health'] = jnp.maximum(n['gate_health'] - (n['gate_open'] & (n['t'] % 4 == 0)).astype(jnp.int32), 0)
        occupiedgate = jnp.any(jnp.all(n['agent_pos'][:, None, :] == n['gate_pos'][None, :, :], axis=2), axis=0) | jnp.any(jnp.all(n['bee_pos'][:, None, :] == n['gate_pos'][None, :, :], axis=2), axis=0)
        n['gate_open'] = n['gate_open'] & ((n['gate_health'] > 0) | occupiedgate)
        n['success'] = jnp.sum(n['community']) >= 6
        n['failure'] = jnp.all(n['crop_stage'] >= 3) & (jnp.sum(n['seeds']) == 0) & (jnp.sum(n['produce']) == 0) & ~n['success']
        n['terminated'] = n['success'] | n['failure']
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.sync(n)
        reward = 0.5 * jnp.sum(n['donated_now']).astype(jnp.float32) + bonus - 0.01 * (a != 0).astype(jnp.float32) - n['failure'].astype(jnp.float32)
        return n, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), info
