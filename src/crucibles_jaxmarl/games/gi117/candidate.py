"""Hidden-demand exchange yard: original unexecuted spatial rules."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 150
    num_actions = 15
    max_steps = 160

    def refresh(self, s):
        n = dict(s)
        held = (s['goods_owner'] >= 0) & (s['goods_owner'] < 3)
        n['goods_pos'] = jnp.where(held[:, None], s['agent_pos'][jnp.clip(s['goods_owner'], 0, 2)], s['goods_pos'])
        has_item = s['goods_owner'][None, :] == jnp.arange(3)[:, None]
        item = jnp.argmax(has_item, axis=1).astype(jnp.int32)
        n['cargo_code'] = jnp.where(jnp.any(has_item, axis=1), s['goods_type'][item], -1).astype(jnp.int32)
        n['agent_active'] = s['delivered'] < 2
        n['agent_visible'] = jnp.broadcast_to(n['agent_active'][None, :], (3, 3))
        n['goods_active'] = s['goods_owner'] != 3
        n['goods_visible'] = jnp.broadcast_to(n['goods_active'][None, :], (3, 6))
        n['order_active'] = n['agent_active']
        n['order_visible'] = jnp.eye(3, dtype=jnp.bool_) & n['order_active'][None, :]
        n['remaining'] = 2 - s['delivered']
        n['offer_pos'] = s['agent_pos']
        n['offer_active'] = (s['offer_ttl'] > 0) & n['agent_active']
        n['offer_visible'] = jnp.broadcast_to(n['offer_active'][None, :], (3, 3))
        return n

    def reset(self, key):
        x = jnp.arange(9)[None, :]
        y = jnp.arange(7)[:, None]
        walls = (x == 0) | (x == 8) | (y == 0) | (y == 6) | (((y == 2) | (y == 4)) & (x != 2) & (x != 4) & (x != 6))
        tiles = jnp.where(walls, 1, 0).astype(jnp.int32)
        tiles = tiles.at[3, 3:6].set(2).at[1, 1].set(3).at[1, 7].set(3).at[5, 4].set(3)
        tiles = tiles.at[3, 1].set(4).at[3, 7].set(4).at[1, 4].set(4)
        pos = jnp.array([[1, 1], [7, 1], [4, 5]], jnp.int32)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False),
             'terminated': jnp.array(False), 'truncated': jnp.array(False), 'tiles': tiles,
             'agent_pos': pos, 'agent_active': jnp.ones(3, jnp.bool_), 'agent_visible': jnp.ones((3, 3), jnp.bool_),
             'delivered': jnp.zeros(3, jnp.int32), 'cargo_code': jnp.arange(3, dtype=jnp.int32),
             'signal_type': jnp.full(3, -1, jnp.int32), 'signal_ttl': jnp.zeros(3, jnp.int32),
             'goods_pos': jnp.concatenate((pos, jnp.array([[1, 3], [7, 3], [4, 1]], jnp.int32)), axis=0),
             'goods_type': jnp.array([0, 1, 2, 0, 1, 2], jnp.int32),
             'goods_owner': jnp.array([0, 1, 2, -1, -1, -1], jnp.int32),
             'goods_active': jnp.ones(6, jnp.bool_), 'goods_visible': jnp.ones((3, 6), jnp.bool_),
             'dock_pos': pos, 'dock_active': jnp.ones(3, jnp.bool_), 'dock_visible': jnp.ones((3, 3), jnp.bool_),
             'order_pos': pos, 'order_active': jnp.ones(3, jnp.bool_), 'order_visible': jnp.eye(3, dtype=jnp.bool_),
             'demand': jax.random.permutation(key, jnp.arange(3, dtype=jnp.int32)), 'remaining': jnp.full(3, 2, jnp.int32),
             'bay_pos': jnp.array([[3, 3], [4, 3], [5, 3]], jnp.int32),
             'bay_active': jnp.ones(3, jnp.bool_), 'bay_visible': jnp.ones((3, 3), jnp.bool_),
             'bay_owner': jnp.full(3, -1, jnp.int32), 'bay_ttl': jnp.zeros(3, jnp.int32),
             'offer_pos': pos, 'offer_active': jnp.zeros(3, jnp.bool_), 'offer_visible': jnp.zeros((3, 3), jnp.bool_),
             'offer_give': jnp.full(3, -1, jnp.int32), 'offer_request': jnp.full(3, -1, jnp.int32),
             'offer_item': jnp.full(3, -1, jnp.int32), 'offer_ttl': jnp.zeros(3, jnp.int32),
             'trade_count': jnp.array(0, jnp.int32), 'delivery_count': jnp.array(0, jnp.int32)}
        return self.observe(s), s

    def observe(self, s):
        agents = jnp.concatenate((s['agent_pos'] / 8.0, (s['cargo_code'][:, None] + 1) / 3.0,
            s['delivered'][:, None] / 2.0, (s['signal_type'][:, None] + 1) / 3.0, s['signal_ttl'][:, None] / 5.0), axis=1)
        goods = jnp.concatenate((s['goods_pos'] / 8.0, s['goods_type'][:, None] / 2.0,
            (s['goods_owner'][:, None] + 1) / 4.0), axis=1)
        bays = jnp.concatenate((s['bay_pos'] / 8.0, (s['bay_owner'][:, None] + 1) / 3.0, s['bay_ttl'][:, None] / 6.0), axis=1)
        offers = jnp.concatenate((s['offer_pos'] / 8.0, (s['offer_give'][:, None] + 1) / 3.0,
            (s['offer_request'][:, None] + 1) / 3.0, s['offer_ttl'][:, None] / 4.0), axis=1)
        docks = jnp.concatenate((s['dock_pos'] / 8.0, s['delivered'][:, None] / 2.0), axis=1)
        public = jnp.concatenate(((s['tiles'] / 4.0).reshape(63), agents.reshape(18), goods.reshape(24),
            bays.reshape(12), offers.reshape(15), docks.reshape(9),
            jnp.array([s['t'] / 160.0, s['trade_count'] / 160.0, s['delivery_count'] / 6.0])))
        # Only the row owner's demand enters its flat observation. No others' order vector.
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jax.nn.one_hot(s['demand'], 3),
            jnp.broadcast_to(public, (3, 144))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        # Public availability only; masks do not reveal another agent's demand.
        return (jnp.arange(15)[None, :] == 0) | ((~s['done'] & s['agent_active'])[:, None])

    def raw_step(self, key, s, actions):
        a = jnp.asarray(actions, jnp.int32)
        a = jnp.where((a >= 0) & (a < 15) & s['agent_active'] & ~s['done'], a, 0)
        n = dict(s)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        target = jnp.clip(s['agent_pos'] + delta[a], jnp.array([0, 0]), jnp.array([8, 6]))
        committed = s['offer_ttl'] > 0
        for offset in range(3):
            i = (s['t'] + offset) % 3
            at_bay = jnp.all(s['bay_pos'] == target[i], axis=1)
            reserved = jnp.any(at_bay & (s['bay_ttl'] > 0) & (s['bay_owner'] != i))
            occupied = jnp.any(n['agent_active'] & (jnp.arange(3) != i) & jnp.all(n['agent_pos'] == target[i], axis=1))
            move = s['agent_active'][i] & ~committed[i] & ~reserved & ~occupied & (s['tiles'][target[i, 1], target[i, 0]] != 1)
            n['agent_pos'] = n['agent_pos'].at[i].set(jnp.where(move, target[i], n['agent_pos'][i]))
        n['bay_ttl'] = jnp.maximum(0, s['bay_ttl'] - 1)
        n['bay_owner'] = jnp.where(n['bay_ttl'] > 0, s['bay_owner'], -1)
        n['offer_ttl'] = jnp.maximum(0, s['offer_ttl'] - 1)
        n['signal_ttl'] = jnp.maximum(0, s['signal_ttl'] - 1)
        n['signal_type'] = jnp.where(n['signal_ttl'] > 0, s['signal_type'], -1)
        n = self.refresh(n)
        delivered_now = jnp.zeros(3, jnp.bool_)
        # Inventory changes settle one object at a time in rotating physical priority.
        for offset in range(3):
            i = (s['t'] + offset) % 3
            free = (n['goods_owner'] == -1) & (jnp.sum(jnp.abs(n['goods_pos'] - n['agent_pos'][i]), axis=1) <= 1)
            gi = jnp.argmax(free).astype(jnp.int32)
            pickup = (a[i] == 5) & ~committed[i] & (n['cargo_code'][i] < 0) & jnp.any(free)
            n['goods_owner'] = n['goods_owner'].at[gi].set(jnp.where(pickup, i, n['goods_owner'][gi]))
            held = n['goods_owner'] == i
            gi = jnp.argmax(held).astype(jnp.int32)
            drop = (a[i] == 6) & ~committed[i] & jnp.any(held)
            deliver = (a[i] == 7) & ~committed[i] & jnp.any(held) & (
                n['goods_type'][gi] == s['demand'][i]) & jnp.all(n['agent_pos'][i] == s['dock_pos'][i])
            n['goods_pos'] = n['goods_pos'].at[gi].set(jnp.where(drop | deliver, n['agent_pos'][i], n['goods_pos'][gi]))
            n['goods_owner'] = n['goods_owner'].at[gi].set(jnp.where(deliver, 3, jnp.where(drop, -1, n['goods_owner'][gi])))
            delivered_now = delivered_now.at[i].set(deliver)
            n['delivered'] = n['delivered'].at[i].add(deliver.astype(jnp.int32))
            n = self.refresh(n)
        for offset in range(3):
            i = (s['t'] + offset) % 3
            distance = jnp.sum(jnp.abs(n['bay_pos'] - n['agent_pos'][i]), axis=1)
            occupied_by_other = jnp.any(jnp.all(n['agent_pos'][:, None, :] == n['bay_pos'][None, :, :], axis=2) &
                (n['agent_active'] & (jnp.arange(3) != i))[:, None], axis=0)
            owns = n['bay_owner'] == i
            eligible = (distance <= 1) & ~occupied_by_other & jnp.where(jnp.any(owns), owns, n['bay_owner'] < 0)
            bi = jnp.argmin(jnp.where(eligible, distance * 4 + jnp.arange(3), 99)).astype(jnp.int32)
            reserve = (a[i] == 8) & n['agent_active'][i] & jnp.any(eligible)
            n['bay_owner'] = n['bay_owner'].at[bi].set(jnp.where(reserve, i, n['bay_owner'][bi]))
            n['bay_ttl'] = n['bay_ttl'].at[bi].set(jnp.where(reserve, 6, n['bay_ttl'][bi]))
            release = (a[i] == 9) | ~n['agent_active'][i]
            n['bay_ttl'] = jnp.where(release & (n['bay_owner'] == i), 0, n['bay_ttl'])
            n['bay_owner'] = jnp.where(n['bay_ttl'] > 0, n['bay_owner'], -1)
            cancel = (a[i] == 14) | ~n['agent_active'][i]
            n['offer_ttl'] = n['offer_ttl'].at[i].set(jnp.where(cancel, 0, n['offer_ttl'][i]))
            held = n['goods_owner'] == i
            gi = jnp.argmax(held).astype(jnp.int32)
            at_bay = jnp.any(jnp.all(n['bay_pos'] == n['agent_pos'][i], axis=1))
            post = (a[i] >= 10) & (a[i] <= 12) & (n['offer_ttl'][i] == 0) & at_bay & jnp.any(held) & n['agent_active'][i]
            n['offer_item'] = n['offer_item'].at[i].set(jnp.where(post, gi, n['offer_item'][i]))
            n['offer_give'] = n['offer_give'].at[i].set(jnp.where(post, n['goods_type'][gi], n['offer_give'][i]))
            n['offer_request'] = n['offer_request'].at[i].set(jnp.where(post, a[i] - 10, n['offer_request'][i]))
            n['offer_ttl'] = n['offer_ttl'].at[i].set(jnp.where(post, 4, n['offer_ttl'][i]))
            signal = (a[i] == 13) & n['agent_active'][i]
            n['signal_type'] = n['signal_type'].at[i].set(jnp.where(signal, s['demand'][i], n['signal_type'][i]))
            n['signal_ttl'] = n['signal_ttl'].at[i].set(jnp.where(signal, 5, n['signal_ttl'][i]))
        valid = (n['offer_ttl'] > 0) & (n['offer_item'] >= 0) & (
            n['goods_owner'][jnp.maximum(n['offer_item'], 0)] == jnp.arange(3))
        settled = jnp.zeros(3, jnp.bool_)
        pair_done = jnp.array(False)
        # Atomic bilateral clearing: neither transfer is observable without the other.
        for offset in range(3):
            i = (s['t'] + offset) % 3
            match = valid & valid[i] & (jnp.arange(3) != i) & (n['offer_request'] == n['offer_give'][i]) & (
                n['offer_give'] == n['offer_request'][i])
            rank = (jnp.arange(3) - s['t']) % 3
            j = jnp.argmin(jnp.where(match, rank, 9)).astype(jnp.int32)
            take = ~pair_done & jnp.any(match)
            gi = jnp.maximum(n['offer_item'][i], 0)
            gj = jnp.maximum(n['offer_item'][j], 0)
            n['goods_owner'] = n['goods_owner'].at[gi].set(jnp.where(take, j, n['goods_owner'][gi]))
            n['goods_owner'] = n['goods_owner'].at[gj].set(jnp.where(take, i, n['goods_owner'][gj]))
            settled = settled | (take & ((jnp.arange(3) == i) | (jnp.arange(3) == j)))
            pair_done = pair_done | take
        # If no pair clears, three simultaneous compatible commitments can clear as a cycle.
        forward = jnp.all(valid & (n['offer_request'] == jnp.roll(n['offer_give'], -1)))
        reverse = jnp.all(valid & (n['offer_request'] == jnp.roll(n['offer_give'], 1)))
        cycle = ~pair_done & (forward | reverse)
        for i in range(3):
            gi = jnp.maximum(n['offer_item'][i], 0)
            receiver = jnp.where(forward, (i + 2) % 3, (i + 1) % 3)
            n['goods_owner'] = n['goods_owner'].at[gi].set(jnp.where(cycle, receiver, n['goods_owner'][gi]))
        settled = settled | cycle
        n['offer_ttl'] = jnp.where(settled, 0, n['offer_ttl'])
        n['offer_item'] = jnp.where(n['offer_ttl'] > 0, n['offer_item'], -1)
        n['offer_give'] = jnp.where(n['offer_ttl'] > 0, n['offer_give'], -1)
        n['offer_request'] = jnp.where(n['offer_ttl'] > 0, n['offer_request'], -1)
        n['trade_count'] = s['trade_count'] + (pair_done | cycle).astype(jnp.int32)
        n['delivery_count'] = jnp.sum(n['delivered'])
        n = self.refresh(n)
        n['t'] = s['t'] + jnp.array(1, jnp.int32)
        n['terminated'] = jnp.all(n['delivered'] == 2)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        reward = (4.0 * delivered_now + 0.3 * jnp.sum(delivered_now.astype(jnp.float32)) -
            0.01 * s['agent_active'] - 0.03 * ((a == 14) & committed)).astype(jnp.float32)
        n = {k: jnp.where(s['done'], s[k], n[k]) for k in s}
        reward = jnp.where(s['done'], jnp.zeros(3, jnp.float32), reward)
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), {
            'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
