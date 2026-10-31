"""Ropeway Commons. Original blind-authored rules; never executed by author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 12
    num_actions = 28
    observation_size = 671
    max_steps = 512
    public_keys = ('t', 'done', 'terminated', 'truncated', 'terrain', 'pos', 'crew', 'tool', 'inventory', 'energy', 'village_pos', 'received', 'node_pos', 'node_kind', 'stock', 'anchor_pos', 'built', 'health', 'counterweight', 'tension', 'braced', 'carrier_pos', 'load', 'cargo_crew', 'progress', 'direction', 'span_from', 'span_to', 'span_length', 'wind', 'delivered', 'constructed', 'damaged', 'transferred', 'reward_parts')

    def reset(self, key):
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False, jnp.bool_),
            'terminated': jnp.array(False, jnp.bool_), 'truncated': jnp.array(False, jnp.bool_),
            'terrain': jnp.zeros((17, 21), jnp.int32),
            'pos': jnp.array([[3, 4], [5, 3], [5, 5], [17, 4], [15, 3], [15, 5], [17, 12], [15, 13], [15, 11], [3, 12], [5, 13], [5, 11]], jnp.int32),
            'agent_active': jnp.ones((12,), jnp.bool_), 'agent_visible': jnp.ones((12, 12), jnp.bool_),
            'crew': jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], jnp.int32),
            'tool': jnp.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], jnp.int32),
            'inventory': jnp.zeros((12, 4), jnp.int32), 'energy': jnp.full((12,), 12, jnp.int32),
            'village_pos': jnp.array([[3, 3], [17, 3], [17, 13], [3, 13]], jnp.int32),
            'village_active': jnp.ones((4,), jnp.bool_), 'village_visible': jnp.ones((12, 4), jnp.bool_),
            'received': jnp.zeros((4,), jnp.int32),
            'node_pos': jnp.array([[2, 5], [7, 2], [18, 5], [13, 2], [18, 11], [13, 14], [2, 11], [7, 14]], jnp.int32),
            'node_kind': jnp.array([0, 1, 2, 1, 0, 1, 2, 1], jnp.int32),
            'stock': jnp.full((8,), 18, jnp.int32),
            'node_active': jnp.ones((8,), jnp.bool_), 'node_visible': jnp.ones((12, 8), jnp.bool_),
            'anchor_pos': jnp.array([[5, 4], [15, 4], [15, 12], [5, 12]], jnp.int32),
            'anchor_active': jnp.ones((4,), jnp.bool_), 'anchor_visible': jnp.ones((12, 4), jnp.bool_),
            'built': jnp.array([True, True, False, False], jnp.bool_),
            'health': jnp.array([6, 6, 0, 0], jnp.int32), 'counterweight': jnp.zeros((4,), jnp.int32),
            'tension': jnp.zeros((4,), jnp.int32), 'braced': jnp.zeros((4,), jnp.int32),
            'carrier_pos': jnp.array([[5, 4], [15, 4], [15, 12], [5, 12]], jnp.int32),
            'carrier_active': jnp.ones((4,), jnp.bool_), 'carrier_visible': jnp.ones((12, 4), jnp.bool_),
            'load': jnp.zeros((4,), jnp.int32), 'cargo_crew': jnp.full((4,), -1, jnp.int32),
            'progress': jnp.zeros((4,), jnp.int32), 'direction': jnp.ones((4,), jnp.int32),
            'span_from': jnp.array([0, 1, 2, 3], jnp.int32), 'span_to': jnp.array([1, 2, 3, 0], jnp.int32),
            'span_length': jnp.array([10, 8, 10, 8], jnp.int32), 'wind': jnp.array(0, jnp.int32),
            'delivered': jnp.zeros((12,), jnp.int32), 'constructed': jnp.zeros((4,), jnp.int32),
            'damaged': jnp.zeros((4,), jnp.int32), 'transferred': jnp.zeros((12,), jnp.int32),
            'reward_parts': jnp.zeros((12, 5), jnp.float32)
        }
        s['stock'] = s['stock'] + jax.random.randint(key, (8,), 0, 7, dtype=jnp.int32)
        s['inventory'] = s['inventory'].at[:, 0].set(1).at[1::3, 0].set(3).at[1::3, 3].set(1)
        s['terrain'] = s['terrain'].at[7:10, 8:13].set(1).at[8, 8:13].set(0)
        s['terrain'] = s['terrain'].at[s['anchor_pos'][:, 1], s['anchor_pos'][:, 0]].set(2)
        s['terrain'] = s['terrain'].at[s['village_pos'][:, 1], s['village_pos'][:, 0]].set(3)
        s['terrain'] = s['terrain'].at[s['node_pos'][:, 1], s['node_pos'][:, 0]].set(4)
        s['terrain'] = self.cable_tiles(s)
        return self.observe(s), s

    def distances(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=2, dtype=jnp.int32)

    def cable_tiles(self, s):
        tiles = s['terrain']
        xx, yy = jnp.arange(21, dtype=jnp.int32)[None, :], jnp.arange(17, dtype=jnp.int32)[:, None]
        for e in range(4):
            f, z = s['anchor_pos'][e], s['anchor_pos'][(e + 1) % 4]
            path = (xx >= jnp.minimum(f[0], z[0])) & (xx <= jnp.maximum(f[0], z[0])) & (yy >= jnp.minimum(f[1], z[1])) & (yy <= jnp.maximum(f[1], z[1]))
            installed = s['built'][e] & s['built'][(e + 1) % 4]
            sound = (s['health'][e] > 0) & (s['health'][(e + 1) % 4] > 0)
            tiles = jnp.where(path & (tiles != 2), jnp.where(installed, jnp.where(sound, 5, 6), 0), tiles)
        return tiles

    def observe(self, s):
        public = jnp.concatenate([s[k].astype(jnp.float32).ravel() for k in self.public_keys])
        return jnp.concatenate((jnp.arange(12, dtype=jnp.int32)[:, None].astype(jnp.float32), jnp.broadcast_to(public, (12, public.shape[0]))), axis=1)

    def legal_actions(self, s):
        da = self.distances(s['pos'], s['anchor_pos'])
        dc = self.distances(s['pos'], s['carrier_pos'])
        dn = self.distances(s['pos'], s['node_pos'])
        dv = self.distances(s['pos'], s['village_pos'])
        peers = (self.distances(s['pos'], s['pos']) == 1)
        free = jnp.sum(s['inventory'], axis=1, dtype=jnp.int32) < 8
        near = da <= 1
        mine = jnp.any((dn <= 1) & (s['stock'][None, :] > 0) & ((s['node_kind'][None, :] == 1) | (s['tool'][:, None] == 1)), axis=1) & free
        craft = (s['tool'] == 2) & (s['inventory'][:, 1] >= 2) & (s['inventory'][:, 2] >= 1)
        build = (s['tool'] == 2) & (s['inventory'][:, 0] >= 2) & (s['inventory'][:, 3] > 0) & jnp.any(near & ~s['built'][None, :], axis=1)
        brace = (s['tool'] == 3) & jnp.any(near & s['built'][None, :], axis=1)
        repair = (s['tool'] == 2) & (s['inventory'][:, 0] > 0) & jnp.any(near & s['built'][None, :] & (s['health'][None, :] < 6), axis=1)
        canload = (dc <= 1) & (s['load'][None, :] == 0) & ((s['inventory'][:, 0] + s['inventory'][:, 2])[:, None] > 0)
        cantake = (dc <= 1) & (s['load'][None, :] > 0) & free[:, None]
        reverse = dc <= 1
        install = (s['tool'] == 3) & (s['inventory'][:, 0] > 0) & jnp.any(near & s['built'][None, :] & (s['counterweight'][None, :] < 3), axis=1)
        withdraw = (s['tool'] == 3) & free & jnp.any(near & (s['counterweight'][None, :] > 0), axis=1)
        delivery = ((s['inventory'][:, 0] + s['inventory'][:, 2]) > 0) & jnp.any((dv <= 1) & (jnp.arange(4, dtype=jnp.int32)[None, :] != s['crew'][:, None]), axis=1)
        handoff = jnp.any(peers & free[None, :], axis=1) & (jnp.sum(s['inventory'], axis=1, dtype=jnp.int32) > 0)
        head = jnp.ones((12, 5), jnp.bool_)
        mid = jnp.stack((mine, craft, build, brace, repair), axis=1)
        tail = jnp.stack((install, withdraw, delivery, handoff, jnp.any(peers, axis=1), jnp.ones((12,), jnp.bool_)), axis=1)
        mask = jnp.concatenate((head, mid, canload, cantake, reverse, tail), axis=1)
        mask = mask & (s['energy'][:, None] > 0) & ~s['done']
        return mask.at[:, 0].set(True).at[:, 27].set(~s['done'])

    def interact(self, i, s, a):
        b = a[i]
        inv = s['inventory'][i]
        dn = jnp.sum(jnp.abs(s['node_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        eligible = (dn <= 1) & (s['stock'] > 0) & ((s['node_kind'] == 1) | (s['tool'][i] == 1))
        ni = jnp.argmin(jnp.where(eligible, dn, 999))
        mine = (b == 5) & eligible[ni] & (jnp.sum(inv, dtype=jnp.int32) < 8)
        amount = mine.astype(jnp.int32)
        s['inventory'] = s['inventory'].at[i, s['node_kind'][ni]].add(amount)
        s['stock'] = s['stock'].at[ni].add(-amount)
        craft = (b == 6) & (s['tool'][i] == 2) & (inv[1] >= 2) & (inv[2] >= 1)
        s['inventory'] = s['inventory'].at[i].add(craft.astype(jnp.int32) * jnp.array([0, -2, -1, 1], jnp.int32))
        da = jnp.sum(jnp.abs(s['anchor_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        ai = jnp.argmin(da)
        close = da[ai] <= 1
        build = (b == 7) & close & ~s['built'][ai] & (s['tool'][i] == 2) & (inv[0] >= 2) & (inv[3] > 0)
        repair = (b == 9) & close & s['built'][ai] & (s['health'][ai] < 6) & (s['tool'][i] == 2) & (inv[0] > 0)
        brace = (b == 8) & close & s['built'][ai] & (s['tool'][i] == 3)
        install = (b == 22) & close & s['built'][ai] & (s['counterweight'][ai] < 3) & (s['tool'][i] == 3) & (inv[0] > 0)
        withdraw = (b == 23) & close & (s['counterweight'][ai] > 0) & (s['tool'][i] == 3) & (jnp.sum(inv, dtype=jnp.int32) < 8)
        s['built'] = s['built'].at[ai].set(s['built'][ai] | build)
        s['health'] = s['health'].at[ai].set(jnp.where(build, 6, jnp.minimum(6, s['health'][ai] + 3 * repair.astype(jnp.int32))))
        s['constructed'] = s['constructed'].at[ai].add(build.astype(jnp.int32))
        s['counterweight'] = s['counterweight'].at[ai].add(install.astype(jnp.int32) - withdraw.astype(jnp.int32))
        s['braced'] = s['braced'].at[ai].add(2 * brace.astype(jnp.int32))
        s['inventory'] = s['inventory'].at[i, 0].add(-2 * build.astype(jnp.int32) - repair.astype(jnp.int32) - install.astype(jnp.int32) + withdraw.astype(jnp.int32)).at[i, 3].add(-build.astype(jnp.int32))
        e = jnp.clip(b - 10, 0, 3)
        closec = jnp.sum(jnp.abs(s['carrier_pos'][e] - s['pos'][i]), dtype=jnp.int32) <= 1
        resource = jnp.where(inv[2] > 0, 2, 0)
        load = (b >= 10) & (b < 14) & closec & (s['load'][e] == 0) & (inv[resource] > 0)
        s['load'] = s['load'].at[e].set(jnp.where(load, resource + 1, s['load'][e]))
        s['cargo_crew'] = s['cargo_crew'].at[e].set(jnp.where(load, s['crew'][i], s['cargo_crew'][e]))
        s['inventory'] = s['inventory'].at[i, resource].add(-load.astype(jnp.int32))
        e = jnp.clip(b - 14, 0, 3)
        take = (b >= 14) & (b < 18) & (jnp.sum(jnp.abs(s['carrier_pos'][e] - s['pos'][i]), dtype=jnp.int32) <= 1) & (s['load'][e] > 0) & (jnp.sum(inv, dtype=jnp.int32) < 8)
        ri = jnp.clip(s['load'][e] - 1, 0, 3)
        s['inventory'] = s['inventory'].at[i, ri].add(take.astype(jnp.int32))
        s['load'] = s['load'].at[e].set(jnp.where(take, 0, s['load'][e]))
        s['cargo_crew'] = s['cargo_crew'].at[e].set(jnp.where(take, -1, s['cargo_crew'][e]))
        e = jnp.clip(b - 18, 0, 3)
        rev = (b >= 18) & (b < 22) & (jnp.sum(jnp.abs(s['carrier_pos'][e] - s['pos'][i]), dtype=jnp.int32) <= 1)
        s['direction'] = s['direction'].at[e].multiply(jnp.where(rev, -1, 1))
        vd = jnp.sum(jnp.abs(s['village_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        vi = jnp.argmin(jnp.where(jnp.arange(4, dtype=jnp.int32) != s['crew'][i], vd, 999))
        deliver = (b == 24) & (vd[vi] <= 1) & (vi != s['crew'][i]) & (inv[resource] > 0)
        s['inventory'] = s['inventory'].at[i, resource].add(-deliver.astype(jnp.int32))
        s['received'] = s['received'].at[vi].add(deliver.astype(jnp.int32))
        s['delivered'] = s['delivered'].at[i].set(deliver.astype(jnp.int32))
        s['reward_parts'] = s['reward_parts'].at[i, 0].add(0.6 * deliver.astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[:, 1].add(deliver.astype(jnp.float32) * ((s['crew'] == s['crew'][i]).astype(jnp.float32) * 1.4 + (s['crew'] == vi).astype(jnp.float32)))
        pd = jnp.sum(jnp.abs(s['pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        free = jnp.sum(s['inventory'], axis=1, dtype=jnp.int32) < 8
        pi = jnp.argmin(jnp.where((pd == 1) & free, jnp.arange(12, dtype=jnp.int32), 999))
        item = jnp.argmax(inv > 0)
        give = (b == 25) & (pd[pi] == 1) & free[pi] & (inv[item] > 0)
        s['inventory'] = s['inventory'].at[i, item].add(-give.astype(jnp.int32)).at[pi, item].add(give.astype(jnp.int32))
        s['transferred'] = s['transferred'].at[i].set(give.astype(jnp.int32))
        pi = jnp.argmin(jnp.where(pd == 1, jnp.arange(12, dtype=jnp.int32), 999))
        swap = (b == 26) & (pd[pi] == 1)
        ti, tp = s['tool'][i], s['tool'][pi]
        s['tool'] = s['tool'].at[i].set(jnp.where(swap, tp, ti)).at[pi].set(jnp.where(swap, ti, tp))
        spent = 3 * build.astype(jnp.int32) + repair.astype(jnp.int32) + install.astype(jnp.int32) + 3 * craft.astype(jnp.int32)
        s['reward_parts'] = s['reward_parts'].at[i, 3].add(-0.05 * spent.astype(jnp.float32))
        return s

    def advance(self, key, state, actions):
        s = dict(state)
        a = jnp.clip(actions, 0, 27)
        valid = (actions >= 0) & (actions < 28) & self.legal_actions(s)[jnp.arange(12, dtype=jnp.int32), a]
        a = jnp.where(valid, a, 0)
        s['delivered'] = jnp.zeros((12,), jnp.int32)
        s['constructed'] = jnp.zeros((4,), jnp.int32)
        s['damaged'] = jnp.zeros((4,), jnp.int32)
        s['transferred'] = jnp.zeros((12,), jnp.int32)
        s['braced'] = jnp.zeros((4,), jnp.int32)
        s['reward_parts'] = jnp.zeros((12, 5), jnp.float32)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = s['pos'] + delta[jnp.where(a < 5, a, 0)]
        inside = jnp.all((proposed >= 0) & (proposed <= jnp.array([20, 16], jnp.int32)), axis=1)
        q = jnp.clip(proposed, jnp.array([0, 0], jnp.int32), jnp.array([20, 16], jnp.int32))
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(12, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1, dtype=jnp.int32) > 1
        move = (a > 0) & (a < 5) & inside & ~occupied & ~contested & (s['terrain'][q[:, 1], q[:, 0]] != 1)
        s['pos'] = jnp.where(move[:, None], q, s['pos'])
        effort = move.astype(jnp.int32) + ((a >= 5) & (a < 27)).astype(jnp.int32)
        s['energy'] = jnp.clip(s['energy'] - effort + 2 * (a == 0).astype(jnp.int32) + 4 * (a == 27).astype(jnp.int32), 0, 12)
        s['reward_parts'] = s['reward_parts'].at[:, 3].add(-0.02 * effort.astype(jnp.float32))
        s = jax.lax.fori_loop(0, 12, lambda i, carry: self.interact(i, carry, a), s)
        s['wind'] = ((s['t'] // 16) % 2).astype(jnp.int32)
        loaded = (s['load'] > 0).astype(jnp.int32)
        s['tension'] = 2 * (loaded + jnp.roll(loaded, 1)) + s['wind']
        support = 2 + s['counterweight'] + s['braced']
        excess = jnp.maximum(0, s['tension'] - support)
        wear = s['built'] & (excess > 0) & ((s['t'] % 3) == 2) & (s['health'] > 0)
        s['health'] = jnp.maximum(0, s['health'] - wear.astype(jnp.int32))
        s['damaged'] = wear.astype(jnp.int32)
        f, z = s['span_from'], s['span_to']
        usable = s['built'][f] & s['built'][z] & (s['health'][f] > 0) & (s['health'][z] > 0) & (excess[f] == 0) & (excess[z] == 0)
        s['progress'] = jnp.clip(s['progress'] + s['direction'] * (usable & (s['load'] > 0)).astype(jnp.int32), 0, s['span_length'])
        unit = jnp.sign(s['anchor_pos'][z] - s['anchor_pos'][f])
        s['carrier_pos'] = s['anchor_pos'][f] + unit * s['progress'][:, None]
        s['terrain'] = self.cable_tiles(s)
        s['reward_parts'] = s['reward_parts'].at[:, 2].add(0.25 * jnp.sum(s['constructed'], dtype=jnp.int32).astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[:, 4].add(-0.1 * jnp.sum(s['damaged'], dtype=jnp.int32).astype(jnp.float32))
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.array(False, jnp.bool_)
        s['truncated'] = (s['t'] >= 512) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        return s, jnp.sum(s['reward_parts'], axis=1, dtype=jnp.float32)

    def raw_step(self, key, state, actions):
        s, rewards = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros((12,), jnp.float32)), lambda _: self.advance(key, state, actions), operand=None)
        return self.observe(s), s, rewards, jnp.full((12,), s['done'], jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
