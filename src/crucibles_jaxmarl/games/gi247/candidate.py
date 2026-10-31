"""Reedbank Market Gardens. Native original rules; no play or training receipt."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 80
    num_actions = 18
    observation_size = 2173
    max_steps = 768

    def refresh(self, s):
        p = s['pos']
        av = jnp.max(jnp.abs(p[:, None] - p[None]), axis=2) <= 3
        nv = jnp.max(jnp.abs(p[:, None] - s['node_pos'][None]), axis=2) <= 3
        x = jnp.arange(40, dtype=jnp.int32)[None, :]
        y = jnp.arange(32, dtype=jnp.int32)[:, None]
        owner = (y // 8) * 5 + x // 8
        channel = x % 8 == 5
        tiles = jnp.where(channel, jnp.where(s['reeds'][owner] >= 3, 1, 2), 0).astype(jnp.int32)
        tiles = jnp.where(channel & (y % 8 == 5) & s['held'][owner], 3, tiles)
        mv = (jnp.abs(x[None] - p[:, 0, None, None]) <= 3) & (jnp.abs(y[None] - p[:, 1, None, None]) <= 3)
        out = dict(s, agent_vis=av, node_vis=nv, map_vis=mv, tiles=tiles)
        names = ('market', 'plot', 'gate', 'reed', 'rack', 'compost', 'bird', 'fish')
        for i, name in enumerate(names):
            out[name + '_pos'] = s['node_pos'][i::8]
            out[name + '_vis'] = nv[:, i::8]
        return out

    def reset(self, key):
        a = jnp.arange(80, dtype=jnp.int32)
        c = jnp.arange(20, dtype=jnp.int32)
        base = jnp.stack(((c % 5) * 8, (c // 5) * 8), axis=1)
        spawn = jnp.array([[4, 5], [6, 5], [4, 2], [1, 1]], jnp.int32)
        p = base[a // 4] + spawn[a % 4]
        n = jnp.arange(160, dtype=jnp.int32)
        layout = jnp.array([[1, 1], [3, 3], [4, 2], [5, 4], [5, 5], [2, 5], [1, 5], [5, 6]], jnp.int32)
        np = base[n // 8] + layout[n % 8]
        s = {'t': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
             'pos': p, 'active': jnp.ones((80,), jnp.bool_), 'team': a // 4, 'tool': a % 4,
             'item': jnp.zeros((80,), jnp.int32), 'age': jnp.zeros((80,), jnp.int32), 'quality': jnp.zeros((80,), jnp.float32),
             'produce_kind': jnp.zeros((80,), jnp.int32), 'seed_kind': 1 + (a // 4) % 2, 'seeds': jnp.full((80,), 3, jnp.int32),
             'node_pos': np, 'site_active': jnp.ones((20,), jnp.bool_), 'household': c,
             'water': jnp.ones((20,), jnp.float32), 'pollution': jnp.zeros((20,), jnp.float32),
             'nutrient': jax.random.uniform(key, (20,), minval=1.0, maxval=2.0, dtype=jnp.float32),
             'gate_open': jnp.zeros((20,), jnp.bool_), 'reeds': jnp.full((20,), 4, jnp.int32),
             'crop': jnp.zeros((20,), jnp.float32), 'crop_kind': jnp.zeros((20,), jnp.int32),
             'rack_item': jnp.zeros((20,), jnp.int32), 'rack_progress': jnp.zeros((20,), jnp.int32),
             'rack_quality': jnp.zeros((20,), jnp.float32), 'rack_kind': jnp.zeros((20,), jnp.int32),
             'held': jnp.zeros((20,), jnp.bool_), 'fish_stock': jnp.full((20,), 2, jnp.int32),
             'compost': jnp.zeros((20,), jnp.int32), 'bird_health': jnp.zeros((20,), jnp.int32),
             'sales': jnp.zeros((20,), jnp.int32), 'revenue': jnp.zeros((20,), jnp.float32),
             'preference': 1 + c % 2, 'flood_loss': jnp.zeros((20,), jnp.bool_), 'netted': jnp.zeros((20,), jnp.bool_),
             'harvested': jnp.zeros((80,), jnp.bool_), 'sold': jnp.zeros((80,), jnp.bool_), 'spoiled': jnp.zeros((80,), jnp.bool_),
             'gate_changed': jnp.zeros((20,), jnp.bool_), 'cut': jnp.zeros((20,), jnp.bool_),
             'agent_vis': jnp.zeros((80, 80), jnp.bool_), 'node_vis': jnp.zeros((80, 160), jnp.bool_),
             'tiles': jnp.zeros((32, 40), jnp.int32), 'map_vis': jnp.zeros((80, 32, 40), jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def nearest(self, s, p):
        dd = jnp.sum(jnp.abs(p[:, None] - s['node_pos'][None]), axis=2)
        k = jnp.argmin(dd, axis=1)
        return k, jnp.min(dd, axis=1) <= 1

    def legal_actions(self, s):
        p = s['pos']
        d = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = p[:, None] + d[None]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 40) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 32)
        tile = s['tiles'][jnp.clip(q[:, :, 1], 0, 31), jnp.clip(q[:, :, 0], 0, 39)]
        walk = inside & ((tile == 0) | (tile == 3) | ((tile == 2) & (s['tool'][:, None] == 3)))
        k, near = self.nearest(s, p)
        typ = k % 8
        h = k // 8
        it = s['item']
        empty = it == 0
        pick = empty & (((typ == 4) & (s['rack_progress'][h] >= 4) & (s['rack_item'][h] > 0)) | ((typ == 5) & (s['compost'][h] > 0)))
        plant = (typ == 1) & (s['tool'] == 0) & (s['seeds'] > 0) & (s['crop'][h] == 0)
        harvest = (typ == 1) & (s['tool'] == 1) & empty & (s['crop'][h] >= 12)
        banks = (jnp.abs(p[:, None, 0] - s['rack_pos'][None, :, 0]) == 1) & (p[:, None, 1] == s['rack_pos'][None, :, 1])
        bank = jnp.any(banks, axis=1)
        load = (typ == 4) & ((it == 3) | (it == 4)) & (s['rack_item'][h] == 0)
        net = bank & empty & (s['tool'] >= 2)
        gate = (typ == 2) & (s['tool'] == 2)
        cut = (typ == 3) & (s['tool'] == 1) & empty & (s['reeds'][h] > 0)
        compost = ((typ == 5) & ((it == 2) | (it == 3) | (it == 4) | (it == 7))) | ((typ == 1) & (it == 8))
        sell = (typ == 0) & (it >= 3) & (it <= 6)
        bird = (typ == 6) & (it == 2)
        rest = jnp.stack((near & pick, near & plant, near & harvest, bank & empty, near & load, net,
                 near & gate, near & cut, near & compost, empty, (it > 0) | (s['seeds'] > 0), near & sell, near & bird), axis=1)
        return (jnp.concatenate((walk, rest), axis=1) & ~s['episode_done']).at[:, 0].set(True)

    def winner(self, target, wanted, count, t):
        rank = (jnp.arange(80, dtype=jnp.int32) - t) % 80
        costs = jnp.where(wanted[:, None] & (target[:, None] == jnp.arange(count)[None]), rank[:, None], 81)
        return wanted & (rank == jnp.min(costs, axis=0)[target])

    def advance(self, s, actions):
        ids = jnp.arange(80, dtype=jnp.int32)
        aa = jnp.clip(actions, 0, 17)
        act = jnp.where((aa == actions) & self.legal_actions(s)[ids, aa], aa, 0)
        d = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['pos'] + d[jnp.minimum(act, 4)] * (act <= 4)[:, None]
        k, near = self.nearest(s, p)
        typ = k % 8
        h = k // 8
        interacting = (act == 5) | (act == 6) | (act == 7) | (act == 9) | (act == 11) | (act == 12) | (act == 13) | (act == 16) | (act == 17)
        win = self.winner(k, near & interacting, 160, s['t'])
        pick = win & (act == 5)
        plant = win & (act == 6)
        harvest = win & (act == 7)
        load = win & (act == 9)
        toggle = win & (act == 11)
        cut = win & (act == 12)
        composting = win & (act == 13) & (typ == 5)
        fertilize = win & (act == 13) & (typ == 1)
        sold = win & (act == 16)
        bird = win & (act == 17)
        def totals(flags):
            return jnp.zeros((20,), jnp.int32).at[h].add(flags.astype(jnp.int32))
        changed = totals(toggle) > 0
        gates = s['gate_open'] ^ changed
        cutcount = totals(cut)
        reeds = jnp.clip(s['reeds'] - cutcount + ((s['t'] % 48 == 47) & (s['water'] > 0.5)).astype(jnp.int32), 0, 6)
        runoff = s['water'] * gates.astype(jnp.float32) * 0.4
        inflow = jnp.concatenate((jnp.full((5,), jnp.where(s['t'] % 32 < 4, 2.4, 0.2), jnp.float32), runoff[:-5]))
        water = jnp.clip(s['water'] * (0.75 - gates.astype(jnp.float32) * 0.25) + inflow, 0, 6)
        poisonflow = jnp.concatenate((jnp.zeros((5,), jnp.float32), s['pollution'][:-5] * gates[:-5].astype(jnp.float32) * 0.4))
        pollution = jnp.clip(s['pollution'] * 0.65 + poisonflow + cutcount * 0.5 + totals(fertilize) * 0.2 - reeds * 0.04, 0, 6)
        nutrient = jnp.clip(s['nutrient'] * 0.98 + inflow * 0.15 + totals(fertilize) * 1.5 - pollution * 0.04, 0, 6)
        bh = jnp.clip(s['bird_health'] + totals(bird) * 8 - (s['t'] % 24 == 23).astype(jnp.int32), 0, 16)
        cropkind = s['crop_kind'] + jnp.zeros((20,), jnp.int32).at[h].add(jnp.where(plant, s['seed_kind'] - s['crop_kind'][h], 0))
        crop = jnp.where(totals(plant) > 0, 0.5, s['crop'])
        crop = jnp.where(totals(harvest) > 0, 0.0, crop)
        c = jnp.arange(20, dtype=jnp.int32)
        left = jnp.maximum(c - 1, 0)
        right = jnp.minimum(c + 1, 19)
        diversity = ((c % 5 > 0) & (cropkind[left] > 0) & (cropkind[left] != cropkind)) | ((c % 5 < 4) & (cropkind[right] > 0) & (cropkind[right] != cropkind))
        sheltered = (bh > 0) | ((c % 5 > 0) & (bh[left] > 0)) | ((c % 5 < 4) & (bh[right] > 0))
        growth = ((water >= 0.5) & (water <= 4) & (nutrient > 0.5) & (crop > 0)).astype(jnp.float32) * (0.15 + 0.1 * diversity + 0.05 * sheltered)
        pests = (s['t'] % 16 == 15) & ~sheltered & ~diversity & (crop > 0)
        crop = jnp.clip(crop + growth - pests.astype(jnp.float32) * 0.5, 0, 12)
        flood_loss = (water > 4) & (crop >= 10)
        crop = jnp.where(flood_loss, 0.0, crop).astype(jnp.float32)
        lp = s['rack_pos'] + jnp.array([-1, 0], jnp.int32)
        rp = s['rack_pos'] + jnp.array([1, 0], jnp.int32)
        atleft = jnp.all(p[:, None] == lp[None], axis=2)
        atright = jnp.all(p[:, None] == rp[None], axis=2)
        held = jnp.any(atleft & (act[:, None] == 8), axis=0) & jnp.any(atright & (act[:, None] == 8), axis=0)
        nets = jnp.any(atleft & (act[:, None] == 10), axis=0) & jnp.any(atright & (act[:, None] == 10), axis=0)
        stock = jnp.clip(s['fish_stock'] + ((inflow > 0.5) & (pollution < 2)).astype(jnp.int32), 0, 8)
        netted = nets & (stock > 0) & (water > 0.5)
        nk = jnp.argmax(atleft, axis=1)
        takefish = self.winner(nk, (act == 10) & jnp.any(atleft, axis=1) & netted[nk], 20, s['t'])
        stock = stock - netted.astype(jnp.int32)
        rack_item = s['rack_item'] + jnp.zeros((20,), jnp.int32).at[h].add(jnp.where(load, s['item'], jnp.where(pick & (typ == 4), -s['rack_item'][h], 0)))
        rack_changed = totals(load | (pick & (typ == 4))) > 0
        rack_progress = jnp.where(rack_changed, 0, s['rack_progress'])
        rack_progress = jnp.minimum(rack_progress + (held & (rack_item > 0) & (water < 5)).astype(jnp.int32), 4)
        rackq = s['rack_quality'] + jnp.zeros((20,), jnp.float32).at[h].add(jnp.where(load, s['quality'] - s['rack_quality'][h], 0))
        rackkind = s['rack_kind'] + jnp.zeros((20,), jnp.int32).at[h].add(jnp.where(load, s['produce_kind'] - s['rack_kind'][h], 0))
        item = jnp.where(harvest, 3, jnp.where(cut, 2, s['item']))
        item = jnp.where(pick, jnp.where(typ == 4, s['rack_item'][h] + 2, 8), item)
        item = jnp.where(load | composting | fertilize | sold | bird, 0, item)
        item = jnp.where(takefish, 4, item).astype(jnp.int32)
        quality = jnp.where(harvest, jnp.clip(s['nutrient'][h] / 3, 0.1, 1) * jnp.where(s['pollution'][h] < 1, 1.0, 0.5), s['quality'])
        quality = jnp.where(pick & (typ == 4), s['rack_quality'][h], quality)
        quality = jnp.where(takefish, jnp.where(pollution[nk] < 1, 1.0, 0.4), quality).astype(jnp.float32)
        pk = jnp.where(harvest, s['crop_kind'][h], s['produce_kind'])
        pk = jnp.where(pick & (typ == 4), s['rack_kind'][h], pk)
        compost = jnp.clip(s['compost'] + totals(composting) - totals(pick & (typ == 5)), 0, 16)
        seeds = s['seeds'] - plant.astype(jnp.int32)
        gainseed = harvest & ((seeds == 0) | (s['seed_kind'] == s['crop_kind'][h])) & (seeds < 3)
        seedkind = jnp.where(gainseed & (seeds == 0), s['crop_kind'][h], s['seed_kind'])
        seeds = seeds + gainseed.astype(jnp.int32)
        dist = jnp.sum(jnp.abs(p[:, None] - p[None]), axis=2)
        adjacent = (dist <= 1) & (ids[:, None] != ids[None])
        compatible = (s['seeds'][None] < 3) & ((s['seeds'][None] == 0) | (s['seed_kind'][:, None] == s['seed_kind'][None]))
        can = adjacent & (act[None] == 0) & (s['item'][None] == 0)
        can = can & (((act[:, None] == 14) & (s['item'][:, None] == 0)) | ((act[:, None] == 15) & ((s['item'][:, None] > 0) | ((s['seeds'][:, None] > 0) & compatible))))
        dest = jnp.argmin(jnp.where(can, ids[None], 81), axis=1)
        send = self.winner(dest, jnp.any(can, axis=1), 80, s['t'])
        swap = send & (act == 14)
        swaprecv = jnp.zeros((80,), jnp.int32).at[dest].add(swap.astype(jnp.int32)) > 0
        tool = jnp.where(swap, s['tool'][dest], s['tool'])
        senttool = jnp.zeros((80,), jnp.int32).at[dest].add(jnp.where(swap, s['tool'], 0))
        tool = jnp.where(swaprecv, senttool, tool)
        give = send & (act == 15) & (s['item'] > 0)
        seedgive = send & (act == 15) & ~give
        recv = jnp.zeros((80,), jnp.int32).at[dest].add(give.astype(jnp.int32)) > 0
        def passint(v):
            incoming = jnp.zeros((80,), jnp.int32).at[dest].add(jnp.where(give, v, 0))
            return jnp.where(give, 0, jnp.where(recv, incoming, v))
        item = passint(item)
        pk = passint(pk)
        age = passint(s['age'])
        qi = jnp.zeros((80,), jnp.float32).at[dest].add(jnp.where(give, quality, 0))
        quality = jnp.where(give, 0.0, jnp.where(recv, qi, quality))
        sr = jnp.zeros((80,), jnp.int32).at[dest].add(seedgive.astype(jnp.int32))
        sk = jnp.zeros((80,), jnp.int32).at[dest].add(jnp.where(seedgive, seedkind, 0))
        seedkind = jnp.where((sr > 0) & (seeds == 0), sk, seedkind)
        seeds = seeds - seedgive.astype(jnp.int32) + sr
        fresh = (item == 3) | (item == 4)
        dry = (item == 5) | (item == 6)
        age = jnp.where(fresh | dry, age + 1, 0).astype(jnp.int32)
        age = jnp.where(harvest | takefish | (pick & (typ == 4)), 0, age)
        spoiled = (fresh & (age > 24)) | (dry & (age > 64))
        item = jnp.where(spoiled, 7, item).astype(jnp.int32)
        quality = jnp.where((item == 0) | spoiled, 0.0, quality).astype(jnp.float32)
        age = jnp.where(spoiled, 0, age)
        preference = 1 + ((jnp.arange(20, dtype=jnp.int32) + (s['t'] + 1) // 96) % 2)
        value = sold.astype(jnp.float32) * s['quality'] * (1 + (s['item'] >= 5).astype(jnp.float32) + 0.25 * (s['produce_kind'] == s['preference'][h]))
        earnings = jnp.zeros((20,), jnp.float32).at[s['team']].add(value)
        sales = s['sales'] + jnp.zeros((20,), jnp.int32).at[s['team']].add(sold.astype(jnp.int32))
        revenue = s['revenue'] + earnings
        lower = jnp.minimum(h + 5, 19)
        attributable = toggle.astype(jnp.float32) * gates[h] * flood_loss[lower] * (h < 15) * (water[lower] - runoff[h] <= 4)
        reward = 0.35 * value + 0.1 * earnings[s['team']] + 0.02 * harvest - 0.003 * (act > 4) - 0.06 * spoiled - 0.04 * flood_loss[s['team']] - 0.04 * attributable - 0.002 * cut
        t = s['t'] + jnp.array(1, jnp.int32)
        terminated = (t >= 96) & jnp.all(sales > 0)
        truncated = (t >= 768) & ~terminated
        n = dict(s, t=t, episode_done=terminated | truncated, terminated=terminated, truncated=truncated,
                 pos=p, tool=tool, item=item, age=age, quality=quality, produce_kind=pk, seed_kind=seedkind, seeds=seeds,
                 water=water, pollution=pollution, nutrient=nutrient, gate_open=gates, reeds=reeds,
                 crop=crop, crop_kind=cropkind, rack_item=rack_item, rack_progress=rack_progress, rack_quality=rackq, rack_kind=rackkind,
                 held=held, fish_stock=stock, compost=compost, bird_health=bh, sales=sales, revenue=revenue,
                 preference=preference, flood_loss=flood_loss, netted=netted, harvested=harvest, sold=sold, spoiled=spoiled,
                 gate_changed=changed, cut=cutcount > 0)
        return self.refresh(n), jnp.clip(reward, -2, 2).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = self.advance(state, actions)
        n = jax.tree.map(lambda old, new: jnp.where(state['episode_done'], old, new), state, n)
        reward = jnp.where(state['episode_done'], jnp.zeros((80,), jnp.float32), reward)
        info = {'done_all': n['episode_done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((80,), n['episode_done'], jnp.bool_), info

    def observe(self, s):
        p = s['pos']
        own = jnp.stack((jnp.full((80,), s['t'] / 768), jnp.arange(80) / 79, s['team'] / 19, s['tool'] / 3,
             p[:, 0] / 39, p[:, 1] / 31, s['item'] / 8, s['age'] / 64, s['quality'], s['produce_kind'] / 2,
             s['seed_kind'] / 2, s['seeds'] / 3, jnp.minimum(s['sales'][s['team']], 768) / 768,
             jnp.minimum(s['revenue'][s['team']], 1728) / 1728, s['spoiled'].astype(jnp.float32)), axis=1)
        actors = jnp.stack((p[:, 0] / 39, p[:, 1] / 31, s['team'] / 19, s['tool'] / 3, s['item'] / 8,
             s['age'] / 64, s['quality'], s['produce_kind'] / 2, s['seed_kind'] / 2, s['seeds'] / 3,
             s['harvested'].astype(jnp.float32), s['sold'].astype(jnp.float32), s['spoiled'].astype(jnp.float32), jnp.ones((80,))), axis=1)
        parts = []
        fields = (jnp.stack((s['preference'] / 2, s['household'] / 19), axis=1),
             jnp.stack((s['crop'] / 12, s['crop_kind'] / 2, s['nutrient'] / 6, s['flood_loss']), axis=1),
             jnp.stack((s['gate_open'], s['water'] / 6, s['gate_changed']), axis=1),
             jnp.stack((s['reeds'] / 6, s['pollution'] / 6, s['cut']), axis=1),
             jnp.stack((s['rack_item'] / 4, s['rack_progress'] / 4, s['rack_quality'], s['rack_kind'] / 2, s['held']), axis=1),
             jnp.stack((s['compost'] / 16,), axis=1), jnp.stack((s['bird_health'] / 16,), axis=1),
             jnp.stack((s['fish_stock'] / 8, s['water'] / 6, s['pollution'] / 6, s['netted']), axis=1))
        for i, f in enumerate(fields):
            point = s['node_pos'][i::8]
            features = jnp.concatenate((point / jnp.array([39, 31], jnp.float32), f, jnp.ones((20, 1))), axis=1)
            parts = parts + [(features[None] * s['node_vis'][:, i::8, None]).reshape((80, -1))]
        k = jnp.arange(49, dtype=jnp.int32)
        q = p[:, None] + jnp.stack((k % 7 - 3, k // 7 - 3), axis=1)[None]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 40) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 32)
        tile = s['tiles'][jnp.clip(q[:, :, 1], 0, 31), jnp.clip(q[:, :, 0], 0, 39)] / 3
        patch = jnp.stack((jnp.where(inside, tile, 0), inside.astype(jnp.float32)), axis=2)
        return jnp.concatenate([own, (actors[None] * s['agent_vis'][:, :, None]).reshape((80, 1120))] + parts + [patch.reshape((80, 98))], axis=1).astype(jnp.float32)
