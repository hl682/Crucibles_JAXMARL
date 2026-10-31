"""Mycelium Valley. Eight farms share tools, compost and living root conduits."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 32
    num_actions = 26
    observation_size = 1529
    max_steps = 640
    public_keys = ('t', 'done', 'terminated', 'truncated', 'terrain', 'pos', 'farm', 'tool', 'bag', 'seed_trait', 'energy', 'farm_pos', 'pantry_pos', 'water_soil', 'nutrient', 'fruit_delivered', 'plot_pos', 'plot_farm', 'planted', 'trait', 'growth', 'disease', 'rot', 'root_pos', 'root_open', 'root_installed', 'root_length', 'packet_pos', 'packet_active', 'packet_water', 'packet_nutrient', 'packet_virus', 'packet_progress', 'packet_direction', 'well_pos', 'well_water', 'bin_pos', 'bin_waste', 'bin_progress', 'bin_compost', 'nursery_pos', 'seed_stock', 'weather_shift', 'rain', 'harvested', 'delivered', 'rotted', 'littered', 'arrived', 'built_now', 'handed', 'reward_parts')

    def reset(self, key):
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False, jnp.bool_), 'terminated': jnp.array(False, jnp.bool_), 'truncated': jnp.array(False, jnp.bool_),
            'terrain': jnp.zeros((21, 25), jnp.int32),
            'pos': jnp.array([[4, 4], [5, 3], [6, 4], [6, 5], [11, 4], [12, 3], [13, 4], [13, 5], [18, 4], [19, 3], [20, 4], [20, 5], [20, 10], [21, 9], [22, 10], [22, 11], [18, 16], [19, 15], [20, 16], [20, 17], [11, 16], [12, 15], [13, 16], [13, 17], [4, 16], [5, 15], [6, 16], [6, 17], [2, 10], [3, 9], [4, 10], [4, 11]], jnp.int32),
            'agent_active': jnp.ones((32,), jnp.bool_), 'agent_visible': jnp.ones((32, 32), jnp.bool_),
            'farm': jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7], jnp.int32),
            'tool': jnp.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], jnp.int32),
            'bag': jnp.zeros((32, 5), jnp.int32), 'seed_trait': jnp.ones((32,), jnp.int32), 'energy': jnp.full((32,), 20, jnp.int32),
            'farm_pos': jnp.array([[5, 4], [12, 4], [19, 4], [21, 10], [19, 16], [12, 16], [5, 16], [3, 10]], jnp.int32),
            'pantry_pos': jnp.array([[5, 6], [12, 6], [19, 6], [21, 12], [19, 18], [12, 18], [5, 18], [3, 12]], jnp.int32),
            'farm_active': jnp.ones((8,), jnp.bool_), 'farm_visible': jnp.ones((32, 8), jnp.bool_),
            'water_soil': jnp.array([6, 2, 4, 3, 2, 4, 3, 2], jnp.int32), 'nutrient': jnp.array([5, 1, 3, 2, 2, 4, 2, 1], jnp.int32), 'fruit_delivered': jnp.zeros((8,), jnp.int32),
            'plot_pos': jnp.array([[5, 4], [5, 5], [12, 4], [12, 5], [19, 4], [19, 5], [21, 10], [21, 11], [19, 16], [19, 17], [12, 16], [12, 17], [5, 16], [5, 17], [3, 10], [3, 11]], jnp.int32),
            'plot_active': jnp.ones((16,), jnp.bool_), 'plot_visible': jnp.ones((32, 16), jnp.bool_), 'plot_farm': jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], jnp.int32),
            'planted': jnp.ones((16,), jnp.bool_), 'trait': jnp.array([1, 2, 1, 3, 2, 1, 3, 2, 1, 2, 1, 3, 2, 1, 3, 2], jnp.int32), 'growth': jnp.full((16,), 4, jnp.int32),
            'disease': jnp.zeros((16,), jnp.int32), 'rot': jnp.zeros((16,), jnp.int32),
            'root_pos': jnp.array([[8, 4], [15, 4], [21, 6], [19, 12], [16, 16], [9, 16], [3, 14], [5, 8]], jnp.int32),
            'root_active': jnp.ones((8,), jnp.bool_), 'root_visible': jnp.ones((32, 8), jnp.bool_), 'root_open': jnp.array([True, True, True, True, True, True, False, False], jnp.bool_), 'root_installed': jnp.array([True, True, True, True, True, True, False, False], jnp.bool_), 'root_length': jnp.array([7, 7, 8, 8, 7, 7, 8, 8], jnp.int32),
            'packet_pos': jnp.array([[5, 4], [12, 4], [19, 4], [21, 10], [19, 16], [12, 16], [5, 16], [3, 10]], jnp.int32),
            'packet_active': jnp.zeros((8,), jnp.bool_), 'packet_visible': jnp.ones((32, 8), jnp.bool_), 'packet_water': jnp.zeros((8,), jnp.int32), 'packet_nutrient': jnp.zeros((8,), jnp.int32), 'packet_virus': jnp.zeros((8,), jnp.int32), 'packet_progress': jnp.zeros((8,), jnp.int32), 'packet_direction': jnp.ones((8,), jnp.int32),
            'well_pos': jnp.array([[7, 9], [17, 11]], jnp.int32), 'well_active': jnp.ones((2,), jnp.bool_), 'well_visible': jnp.ones((32, 2), jnp.bool_), 'well_water': jnp.full((2,), 20, jnp.int32),
            'bin_pos': jnp.array([[9, 10], [15, 10]], jnp.int32), 'bin_active': jnp.ones((2,), jnp.bool_), 'bin_visible': jnp.ones((32, 2), jnp.bool_), 'bin_waste': jnp.full((2,), 3, jnp.int32), 'bin_progress': jnp.zeros((2,), jnp.int32), 'bin_compost': jnp.full((2,), 2, jnp.int32),
            'nursery_pos': jnp.array([[12, 10]], jnp.int32), 'nursery_active': jnp.ones((1,), jnp.bool_), 'nursery_visible': jnp.ones((32, 1), jnp.bool_), 'seed_stock': jnp.full((1, 3), 6, jnp.int32),
            'weather_shift': jnp.array(0, jnp.int32), 'rain': jnp.array(False, jnp.bool_),
            'harvested': jnp.zeros((32,), jnp.int32), 'delivered': jnp.zeros((32,), jnp.int32), 'rotted': jnp.zeros((16,), jnp.int32), 'arrived': jnp.zeros((8,), jnp.int32), 'built_now': jnp.zeros((8,), jnp.int32), 'handed': jnp.zeros((32,), jnp.int32),
            'littered': jnp.zeros((16,), jnp.int32), 'reward_parts': jnp.zeros((32, 6), jnp.float32)
        }
        s['weather_shift'] = jax.random.randint(key, (), 0, 8, dtype=jnp.int32)
        s['disease'] = s['disease'].at[0].set(1)
        s['bag'] = s['bag'].at[0::4, 0].set(2).at[2::4, 1].set(2).at[3::4, 2].set(2)
        s['seed_trait'] = (s['farm'] % 3 + 1).astype(jnp.int32)
        xx, yy = jnp.arange(25, dtype=jnp.int32)[None, :], jnp.arange(21, dtype=jnp.int32)[:, None]
        for e in range(8):
            f, z = s['farm_pos'][e], s['farm_pos'][(e + 1) % 8]
            corridor = ((yy == f[1]) & (xx >= jnp.minimum(f[0], z[0])) & (xx <= jnp.maximum(f[0], z[0]))) | ((xx == z[0]) & (yy >= jnp.minimum(f[1], z[1])) & (yy <= jnp.maximum(f[1], z[1])))
            s['terrain'] = jnp.where(corridor, 1, s['terrain'])
        s['terrain'] = s['terrain'].at[s['plot_pos'][:, 1], s['plot_pos'][:, 0]].set(2)
        s['terrain'] = s['terrain'].at[s['pantry_pos'][:, 1], s['pantry_pos'][:, 0]].set(3)
        return self.observe(s), s

    def distances(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=2, dtype=jnp.int32)

    def packet_positions(self, s):
        start = s['farm_pos']
        end = jnp.roll(start, -1, axis=0)
        dx = end[:, 0] - start[:, 0]
        dy = end[:, 1] - start[:, 1]
        p = s['packet_progress']
        return jnp.stack((start[:, 0] + jnp.sign(dx) * jnp.minimum(p, jnp.abs(dx)), start[:, 1] + jnp.sign(dy) * jnp.maximum(0, p - jnp.abs(dx))), axis=1)

    def observe(self, s):
        public = jnp.concatenate([s[k].astype(jnp.float32).ravel() for k in self.public_keys])
        return jnp.concatenate((jnp.arange(32, dtype=jnp.int32)[:, None].astype(jnp.float32), jnp.broadcast_to(public, (32, public.shape[0]))), axis=1)

    def legal_actions(self, s):
        near = self.distances(s['pos'], s['plot_pos']) <= 1
        root = self.distances(s['pos'], s['root_pos']) <= 1
        bins = self.distances(s['pos'], s['bin_pos']) <= 1
        free = jnp.sum(s['bag'], axis=1, dtype=jnp.int32) < jnp.where(s['tool'] == 4, 12, 8)
        irrigator, pruner, grafter, carter = s['tool'] == 1, s['tool'] == 2, s['tool'] == 3, s['tool'] == 4
        fill = free & (s['bag'][:, 0] < 4) & jnp.any((self.distances(s['pos'], s['well_pos']) <= 1) & (s['well_water'][None, :] > 0), axis=1)
        water = irrigator & (s['bag'][:, 0] > 0) & jnp.any(near & (s['water_soil'][s['plot_farm']][None, :] < 8), axis=1)
        harvest = (s['bag'][:, 3] <= 2) & (s['bag'][:, 4] < 4) & jnp.any(near & s['planted'][None, :] & (s['growth'][None, :] >= 8) & (s['disease'][None, :] == 0), axis=1) & (jnp.sum(s['bag'], axis=1, dtype=jnp.int32) <= jnp.where(carter, 9, 5))
        plant = grafter & (s['bag'][:, 1] > 0) & jnp.any(near & ~s['planted'][None, :], axis=1)
        graft = grafter & (s['bag'][:, 1] > 0) & jnp.any(near & s['planted'][None, :], axis=1)
        cure = pruner & jnp.any(near & (s['disease'][None, :] > 0), axis=1)
        cut = pruner & jnp.any(root & s['root_open'][None, :], axis=1)
        connect = grafter & (s['bag'][:, 1] > 0) & (s['bag'][:, 2] > 0) & jnp.any(root & ~s['root_open'][None, :], axis=1)
        compost = free & (s['bag'][:, 2] < 4) & jnp.any(bins & (s['bin_compost'][None, :] > 0), axis=1)
        waste = (s['bag'][:, 4] > 0) & jnp.any(bins & (s['bin_waste'][None, :] < 8), axis=1)
        stir = carter & jnp.any(bins & (s['bin_waste'][None, :] > 0), axis=1)
        spread = carter & (s['bag'][:, 2] > 0) & jnp.any(near & (s['nutrient'][s['plot_farm']][None, :] < 8), axis=1)
        rot = free & (s['bag'][:, 4] < 4) & jnp.any(near & (s['rot'][None, :] > 0), axis=1)
        deliver = (s['bag'][:, 3] > 0) & jnp.any(self.distances(s['pos'], s['pantry_pos']) <= 1, axis=1)
        nursery = (self.distances(s['pos'], s['nursery_pos'])[:, 0] <= 1) & free & (s['bag'][:, 1] < 4)
        seeds = nursery[:, None] & (s['seed_stock'] > 0) & ((s['bag'][:, 1:2] == 0) | (s['seed_trait'][:, None] == jnp.arange(1, 4, dtype=jnp.int32)[None, :]))
        peer = self.distances(s['pos'], s['pos']) == 1
        hand = (jnp.sum(s['bag'], axis=1, dtype=jnp.int32) > 0) & jnp.any(peer & free[None, :], axis=1)
        save = grafter & free & (s['bag'][:, 1] < 4) & jnp.any(near & s['planted'][None, :] & (s['growth'][None, :] >= 6) & ((s['bag'][:, 1:2] == 0) | (s['seed_trait'][:, None] == s['trait'][None, :])), axis=1)
        extra = jnp.stack((fill, water, harvest, plant, graft, cure, cut, connect, compost, waste, stir, spread, rot, deliver), axis=1)
        tail = jnp.stack((hand, jnp.any(peer, axis=1), jnp.ones((32,), jnp.bool_), save), axis=1)
        mask = jnp.concatenate((jnp.ones((32, 5), jnp.bool_), extra, seeds, tail), axis=1) & (s['energy'][:, None] > 0) & ~s['done']
        return mask.at[:, 0].set(True).at[:, 24].set(~s['done'])

    def interact(self, i, s, a):
        b = a[i]
        bag = s['bag'][i]
        capacity = jnp.where(s['tool'][i] == 4, 12, 8)
        free = jnp.sum(bag, dtype=jnp.int32) < capacity
        wd = jnp.sum(jnp.abs(s['well_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        wi = jnp.argmin(wd)
        fill = (b == 5) & (wd[wi] <= 1) & free & (bag[0] < 4) & (s['well_water'][wi] > 0)
        s['well_water'] = s['well_water'].at[wi].add(-fill.astype(jnp.int32))
        s['bag'] = s['bag'].at[i, 0].add(fill.astype(jnp.int32))
        pd = jnp.sum(jnp.abs(s['plot_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        eligible = (pd <= 1) & (((b == 7) & s['planted'] & (s['growth'] >= 8) & (s['disease'] == 0)) | ((b == 8) & ~s['planted']) | ((b == 9) & s['planted']) | ((b == 10) & (s['disease'] > 0)) | ((b == 17) & (s['rot'] > 0)) | ((b == 25) & s['planted'] & (s['growth'] >= 6) & ((bag[1] == 0) | (s['trait'] == s['seed_trait'][i]))) | (b == 6) | (b == 16))
        pi = jnp.argmin(jnp.where(eligible, pd, 999))
        close = eligible[pi]
        farm = s['plot_farm'][pi]
        irrigate = (b == 6) & close & (s['tool'][i] == 1) & (bag[0] > 0) & (s['water_soil'][farm] < 8)
        spread = (b == 16) & close & (s['tool'][i] == 4) & (bag[2] > 0) & (s['nutrient'][farm] < 8)
        s['water_soil'] = s['water_soil'].at[farm].add(irrigate.astype(jnp.int32))
        s['nutrient'] = s['nutrient'].at[farm].add(spread.astype(jnp.int32))
        s['bag'] = s['bag'].at[i, 0].add(-irrigate.astype(jnp.int32)).at[i, 2].add(-spread.astype(jnp.int32))
        amount = 1 + (s['trait'][pi] == 2).astype(jnp.int32)
        harvest = (b == 7) & close & (bag[3] + amount <= 4) & (bag[4] < 4) & (jnp.sum(bag, dtype=jnp.int32) + amount + 1 <= capacity)
        plant = (b == 8) & close & (s['tool'][i] == 3) & (bag[1] > 0)
        graft = (b == 9) & close & (s['tool'][i] == 3) & (bag[1] > 0)
        cure = (b == 10) & close & (s['tool'][i] == 2)
        collect = (b == 17) & close & free & (bag[4] < 4)
        save = (b == 25) & close & (s['tool'][i] == 3) & free & (bag[1] < 4)
        s['seed_trait'] = s['seed_trait'].at[i].set(jnp.where(save, s['trait'][pi], s['seed_trait'][i]))
        s['bag'] = s['bag'].at[i, 3].add(amount * harvest.astype(jnp.int32)).at[i, 4].add(harvest.astype(jnp.int32) + collect.astype(jnp.int32)).at[i, 1].add(save.astype(jnp.int32) - (plant | graft).astype(jnp.int32))
        s['planted'] = s['planted'].at[pi].set(jnp.where(harvest, False, s['planted'][pi] | plant))
        s['trait'] = s['trait'].at[pi].set(jnp.where(plant | graft, s['seed_trait'][i], s['trait'][pi]))
        s['growth'] = s['growth'].at[pi].set(jnp.where(harvest | plant, 0, jnp.maximum(0, s['growth'][pi] - 2 * (graft | cure | save).astype(jnp.int32))))
        s['disease'] = s['disease'].at[pi].set(jnp.where(cure | harvest | plant | (graft & (s['seed_trait'][i] == 3)), 0, s['disease'][pi]))
        s['rot'] = s['rot'].at[pi].add(-collect.astype(jnp.int32))
        s['harvested'] = s['harvested'].at[i].set(amount * harvest.astype(jnp.int32))
        rd = jnp.sum(jnp.abs(s['root_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        ri = jnp.argmin(rd)
        cut = (b == 11) & (rd[ri] <= 1) & (s['tool'][i] == 2) & s['root_open'][ri]
        connect = (b == 12) & (rd[ri] <= 1) & (s['tool'][i] == 3) & ~s['root_open'][ri] & (bag[1] > 0) & (bag[2] > 0)
        s['built_now'] = s['built_now'].at[ri].add((connect & ~s['root_installed'][ri]).astype(jnp.int32))
        s['root_installed'] = s['root_installed'].at[ri].set(s['root_installed'][ri] | connect)
        s['root_open'] = s['root_open'].at[ri].set(jnp.where(cut, False, s['root_open'][ri] | connect))
        s['packet_active'] = s['packet_active'].at[ri].set(s['packet_active'][ri] & ~cut)
        s['packet_water'] = s['packet_water'].at[ri].set(jnp.where(cut, 0, s['packet_water'][ri]))
        s['packet_nutrient'] = s['packet_nutrient'].at[ri].set(jnp.where(cut, 0, s['packet_nutrient'][ri]))
        s['packet_virus'] = s['packet_virus'].at[ri].set(jnp.where(cut, 0, s['packet_virus'][ri]))
        s['bag'] = s['bag'].at[i, 1].add(-connect.astype(jnp.int32)).at[i, 2].add(-connect.astype(jnp.int32))
        bd = jnp.sum(jnp.abs(s['bin_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        bi = jnp.argmin(bd)
        compost = (b == 13) & (bd[bi] <= 1) & free & (bag[2] < 4) & (s['bin_compost'][bi] > 0)
        waste = (b == 14) & (bd[bi] <= 1) & (bag[4] > 0) & (s['bin_waste'][bi] < 8)
        stir = (b == 15) & (bd[bi] <= 1) & (s['tool'][i] == 4) & (s['bin_waste'][bi] > 0)
        s['bin_compost'] = s['bin_compost'].at[bi].add(-compost.astype(jnp.int32))
        s['bin_waste'] = s['bin_waste'].at[bi].add(waste.astype(jnp.int32))
        s['bin_progress'] = s['bin_progress'].at[bi].set(jnp.minimum(4, s['bin_progress'][bi] + stir.astype(jnp.int32)))
        s['bag'] = s['bag'].at[i, 2].add(compost.astype(jnp.int32)).at[i, 4].add(-waste.astype(jnp.int32))
        vd = jnp.sum(jnp.abs(s['pantry_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        vi = jnp.argmin(vd)
        deliver = (b == 18) & (vd[vi] <= 1) & (bag[3] > 0)
        s['bag'] = s['bag'].at[i, 3].add(-deliver.astype(jnp.int32))
        s['fruit_delivered'] = s['fruit_delivered'].at[vi].add(deliver.astype(jnp.int32))
        s['delivered'] = s['delivered'].at[i].set(deliver.astype(jnp.int32))
        si = jnp.clip(b - 19, 0, 2)
        seed = (b >= 19) & (b <= 21) & (jnp.sum(jnp.abs(s['nursery_pos'][0] - s['pos'][i]), dtype=jnp.int32) <= 1) & free & (bag[1] < 4) & (s['seed_stock'][0, si] > 0) & ((bag[1] == 0) | (s['seed_trait'][i] == si + 1))
        s['seed_stock'] = s['seed_stock'].at[0, si].add(-seed.astype(jnp.int32))
        s['bag'] = s['bag'].at[i, 1].add(seed.astype(jnp.int32))
        s['seed_trait'] = s['seed_trait'].at[i].set(jnp.where(seed, si + 1, s['seed_trait'][i]))
        ad = jnp.sum(jnp.abs(s['pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        transferable = (bag[None, :] > 0) & (s['bag'] < 4) & ((jnp.arange(5, dtype=jnp.int32)[None, :] != 1) | (s['bag'][:, 1:2] == 0) | (s['seed_trait'][:, None] == s['seed_trait'][i]))
        room = jnp.sum(s['bag'], axis=1, dtype=jnp.int32) < jnp.where(s['tool'] == 4, 12, 8)
        eligible_peer = (ad == 1) & room & jnp.any(transferable, axis=1)
        ai = jnp.argmin(jnp.where(eligible_peer, jnp.arange(32, dtype=jnp.int32), 999))
        item = jnp.argmax(transferable[ai])
        give = (b == 22) & eligible_peer[ai]
        s['bag'] = s['bag'].at[i, item].add(-give.astype(jnp.int32)).at[ai, item].add(give.astype(jnp.int32))
        s['seed_trait'] = s['seed_trait'].at[ai].set(jnp.where(give & (item == 1), s['seed_trait'][i], s['seed_trait'][ai]))
        s['handed'] = s['handed'].at[i].set(give.astype(jnp.int32))
        ai = jnp.argmin(jnp.where(ad == 1, jnp.arange(32, dtype=jnp.int32), 999))
        swap = (b == 23) & (ad[ai] == 1) & (jnp.sum(s['bag'][i], dtype=jnp.int32) <= 8) & (jnp.sum(s['bag'][ai], dtype=jnp.int32) <= 8)
        ti, ta = s['tool'][i], s['tool'][ai]
        s['tool'] = s['tool'].at[i].set(jnp.where(swap, ta, ti)).at[ai].set(jnp.where(swap, ti, ta))
        s['reward_parts'] = s['reward_parts'].at[i, 0].add(0.15 * (amount * harvest.astype(jnp.int32)).astype(jnp.float32) + 0.5 * deliver.astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[:, 1].add(1.2 * deliver.astype(jnp.float32) * (s['farm'] == s['farm'][i]).astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[:, 2].add(0.6 * (deliver & (vi != s['farm'][i])).astype(jnp.float32) * (s['farm'] == vi).astype(jnp.float32))
        used = irrigate.astype(jnp.int32) + spread.astype(jnp.int32) + (plant | graft).astype(jnp.int32) + 2 * connect.astype(jnp.int32)
        s['reward_parts'] = s['reward_parts'].at[i, 4].add(-0.03 * used.astype(jnp.float32) - 0.08 * (cure | cut).astype(jnp.float32))
        return s

    def root_process(self, e, s):
        other = (e + 1) % 8
        p = jnp.clip(s['packet_progress'][e] + s['packet_direction'][e] * s['packet_active'][e].astype(jnp.int32), 0, s['root_length'][e])
        arrived = s['packet_active'][e] & (((s['packet_direction'][e] == 1) & (p == s['root_length'][e])) | ((s['packet_direction'][e] == -1) & (p == 0)))
        destination = jnp.where(s['packet_direction'][e] == 1, other, e)
        s['water_soil'] = s['water_soil'].at[destination].set(jnp.minimum(8, s['water_soil'][destination] + s['packet_water'][e] * arrived.astype(jnp.int32)))
        s['nutrient'] = s['nutrient'].at[destination].set(jnp.minimum(8, s['nutrient'][destination] + s['packet_nutrient'][e] * arrived.astype(jnp.int32)))
        exposed = arrived & (s['packet_virus'][e] > 0) & (s['plot_farm'] == destination) & s['planted'] & (s['trait'] != 3)
        s['disease'] = jnp.maximum(s['disease'], exposed.astype(jnp.int32))
        s['arrived'] = s['arrived'].at[e].set(arrived.astype(jnp.int32))
        s['packet_active'] = s['packet_active'].at[e].set(s['packet_active'][e] & ~arrived)
        s['packet_progress'] = s['packet_progress'].at[e].set(p)
        score = s['water_soil'] + s['nutrient']
        source = jnp.where(score[e] >= score[other], e, other)
        target = jnp.where(source == e, other, e)
        launch = s['root_open'][e] & ~s['packet_active'][e] & ~arrived & (score[source] >= score[target] + 3) & (s['water_soil'][source] > 0) & (s['nutrient'][source] > 0)
        infected = (s['water_soil'][source] >= 4) & jnp.any((s['plot_farm'] == source) & (s['disease'] > 0) & s['planted'])
        s['packet_active'] = s['packet_active'].at[e].set(s['packet_active'][e] | launch)
        s['packet_direction'] = s['packet_direction'].at[e].set(jnp.where(launch, jnp.where(source == e, 1, -1), s['packet_direction'][e]))
        s['packet_progress'] = s['packet_progress'].at[e].set(jnp.where(launch, jnp.where(source == e, 0, s['root_length'][e]), s['packet_progress'][e]))
        s['packet_water'] = s['packet_water'].at[e].set(jnp.where(launch, 1, jnp.where(arrived, 0, s['packet_water'][e])))
        s['packet_nutrient'] = s['packet_nutrient'].at[e].set(jnp.where(launch, 1, jnp.where(arrived, 0, s['packet_nutrient'][e])))
        s['packet_virus'] = s['packet_virus'].at[e].set(jnp.where(launch, infected.astype(jnp.int32), jnp.where(arrived, 0, s['packet_virus'][e])))
        s['water_soil'] = s['water_soil'].at[source].add(-launch.astype(jnp.int32))
        s['nutrient'] = s['nutrient'].at[source].add(-launch.astype(jnp.int32))
        return s

    def crop_process(self, p, s):
        f = s['plot_farm'][p]
        phase = ((s['t'] + p) % 8) == 7
        grow = phase & s['planted'][p] & (s['disease'][p] == 0) & (s['growth'][p] < 8) & (s['water_soil'][f] > 0) & (s['nutrient'][f] > 0)
        s['growth'] = s['growth'].at[p].add(grow.astype(jnp.int32))
        s['water_soil'] = s['water_soil'].at[f].add(-grow.astype(jnp.int32))
        s['nutrient'] = s['nutrient'].at[f].add(-grow.astype(jnp.int32))
        outbreak = phase & s['planted'][p] & (s['trait'][p] == 2) & (s['water_soil'][f] >= 6)
        worsening = phase & s['planted'][p] & (s['disease'][p] > 0) & (s['water_soil'][f] >= 4)
        s['disease'] = s['disease'].at[p].set(jnp.minimum(3, jnp.maximum(s['disease'][p], outbreak.astype(jnp.int32)) + worsening.astype(jnp.int32)))
        dead = s['planted'][p] & (s['disease'][p] >= 3)
        s['rotted'] = s['rotted'].at[p].set(dead.astype(jnp.int32))
        s['planted'] = s['planted'].at[p].set(s['planted'][p] & ~dead)
        s['growth'] = s['growth'].at[p].set(jnp.where(dead, 0, s['growth'][p]))
        s['disease'] = s['disease'].at[p].set(jnp.where(dead, 0, s['disease'][p]))
        s['rot'] = s['rot'].at[p].set(jnp.minimum(4, s['rot'][p] + dead.astype(jnp.int32)))
        return s

    def advance(self, key, state, actions):
        s = dict(state)
        a = jnp.clip(actions, 0, 25)
        valid = (actions >= 0) & (actions < 26) & self.legal_actions(s)[jnp.arange(32, dtype=jnp.int32), a]
        a = jnp.where(valid, a, 0)
        s['harvested'] = jnp.zeros((32,), jnp.int32)
        s['delivered'] = jnp.zeros((32,), jnp.int32)
        s['rotted'] = jnp.zeros((16,), jnp.int32)
        s['built_now'] = jnp.zeros((8,), jnp.int32)
        s['handed'] = jnp.zeros((32,), jnp.int32)
        s['reward_parts'] = jnp.zeros((32, 6), jnp.float32)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = s['pos'] + delta[jnp.where(a < 5, a, 0)]
        inside = jnp.all((proposed >= 0) & (proposed <= jnp.array([24, 20], jnp.int32)), axis=1)
        q = jnp.clip(proposed, jnp.array([0, 0], jnp.int32), jnp.array([24, 20], jnp.int32))
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(32, dtype=jnp.bool_), axis=1)
        clash = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1, dtype=jnp.int32) > 1
        fixed = jnp.concatenate((s['well_pos'], s['bin_pos'], s['nursery_pos']), axis=0)
        obstacle = jnp.any(jnp.all(q[:, None, :] == fixed[None, :, :], axis=2), axis=1)
        move = (a > 0) & (a < 5) & inside & ~occupied & ~clash & ~obstacle
        s['pos'] = jnp.where(move[:, None], q, s['pos'])
        effort = move.astype(jnp.int32) + ((a >= 5) & (a != 24)).astype(jnp.int32)
        s['energy'] = jnp.clip(s['energy'] - effort + 2 * (a == 0).astype(jnp.int32) + 6 * (a == 24).astype(jnp.int32), 0, 20)
        s['reward_parts'] = s['reward_parts'].at[:, 4].add(-0.01 * effort.astype(jnp.float32))
        s = jax.lax.fori_loop(0, 32, lambda i, carry: self.interact(i, carry, a), s)
        s['rain'] = ((s['t'] + s['weather_shift']) % 8) == 0
        dry = ((s['t'] + s['weather_shift']) % 8) == 4
        s['water_soil'] = jnp.clip(s['water_soil'] + s['rain'].astype(jnp.int32) - dry.astype(jnp.int32), 0, 8)
        s['well_water'] = jnp.minimum(24, s['well_water'] + 3 * s['rain'].astype(jnp.int32))
        s['seed_stock'] = jnp.minimum(6, s['seed_stock'] + (s['rain'] & ((s['t'] % 24) < 8)).astype(jnp.int32))
        s = jax.lax.fori_loop(0, 8, lambda e, carry: self.root_process(e, carry), s)
        s['packet_pos'] = self.packet_positions(s)
        s = jax.lax.fori_loop(0, 16, lambda p, carry: self.crop_process(p, carry), s)
        litter = ((s['t'] % 32) == 31) & ((jnp.arange(16, dtype=jnp.int32) % 2) == 1) & (s['rot'] < 4)
        s['littered'] = litter.astype(jnp.int32)
        s['rot'] = s['rot'] + s['littered']
        s['bin_progress'] = jnp.minimum(4, s['bin_progress'] + ((s['bin_waste'] > 0) & ((s['t'] % 2) == 1)).astype(jnp.int32))
        composted = (s['bin_waste'] > 0) & (s['bin_progress'] >= 4) & (s['bin_compost'] < 8)
        s['bin_waste'] = s['bin_waste'] - composted.astype(jnp.int32)
        s['bin_compost'] = s['bin_compost'] + composted.astype(jnp.int32)
        s['bin_progress'] = jnp.where(composted, 0, s['bin_progress'])
        local_rot = jnp.sum(s['rotted'].reshape((8, 2)), axis=1, dtype=jnp.int32)
        s['reward_parts'] = s['reward_parts'].at[:, 3].add(0.2 * jnp.sum(s['built_now'], dtype=jnp.int32).astype(jnp.float32) + 0.02 * jnp.sum(s['arrived'], dtype=jnp.int32).astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[:, 5].add(-0.25 * local_rot[s['farm']].astype(jnp.float32))
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.array(False, jnp.bool_)
        s['truncated'] = (s['t'] >= 640) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        return s, jnp.sum(s['reward_parts'], axis=1, dtype=jnp.float32)

    def raw_step(self, key, state, actions):
        s, rewards = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros((32,), jnp.float32)), lambda _: self.advance(key, state, actions), operand=None)
        return self.observe(s), s, rewards, jnp.full((32,), s['done'], jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
