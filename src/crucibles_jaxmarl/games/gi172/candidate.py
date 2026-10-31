"""Signal Bargain: original prospective source; never executed by author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 108
    num_actions = 44
    max_steps = 60

    def refresh(self, s):
        n = dict(s)
        n['agent_active'] = (s['t'] < s['deadline']) & (s['deliveries'] < 2)
        n['agent_visible'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=-1) <= 2
        n['source_visible'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['source_pos'][None, :, :]), axis=-1) <= 2
        n['source_active'] = s['supply'] > 0
        n['offer_active'] = s['offer_ttl'] > 0
        n['offer_visible'] = (jnp.eye(3, dtype=jnp.bool_) | (jnp.arange(3)[:, None] == s['offer_target'][None, :])) & n['offer_active'][None, :]
        n['signal_active'] = s['signal_ttl'] > 0
        n['signal_visible'] = s['signal_heard'] & n['signal_active'][None, :]
        return n

    def reset(self, key):
        priority = jax.random.randint(key, (3,), 0, 3, dtype=jnp.int32)
        recipe = 2 * jax.nn.one_hot(priority, 3, dtype=jnp.int32) + jax.nn.one_hot((priority + 1) % 3, 3, dtype=jnp.int32)
        s = {
            't': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'agent_pos': jnp.array([[3, 3], [4, 3], [4, 4]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_),
            'agent_visible': jnp.ones((3, 3), jnp.bool_),
            'inventory': 2 * jnp.eye(3, dtype=jnp.int32),
            'recipe': recipe, 'escrow': jnp.zeros((3,), jnp.int32),
            'deliveries': jnp.zeros((3,), jnp.int32),
            'deadline': jnp.array([42, 48, 54], jnp.int32),
            'signals_left': jnp.full((3,), 3, jnp.int32),
            'source_pos': jnp.array([[1, 1], [7, 1], [4, 5]], jnp.int32),
            'source_active': jnp.ones((3,), jnp.bool_),
            'source_visible': jnp.ones((3, 3), jnp.bool_),
            'supply': jnp.full((3,), 6, jnp.int32),
            'source_type': jnp.arange(3, dtype=jnp.int32),
            'home_pos': jnp.array([[0, 3], [8, 3], [4, 6]], jnp.int32),
            'home_active': jnp.ones((3,), jnp.bool_),
            'home_visible': jnp.ones((3, 3), jnp.bool_),
            'market_pos': jnp.array([[3, 3], [5, 3]], jnp.int32),
            'market_active': jnp.ones((2,), jnp.bool_),
            'market_visible': jnp.ones((3, 2), jnp.bool_),
            'offer_pos': jnp.array([[3, 3], [4, 3], [4, 4]], jnp.int32),
            'offer_active': jnp.zeros((3,), jnp.bool_),
            'offer_visible': jnp.zeros((3, 3), jnp.bool_),
            'offer_target': jnp.full((3,), -1, jnp.int32),
            'offer_give': jnp.zeros((3,), jnp.int32),
            'offer_ask': jnp.ones((3,), jnp.int32),
            'offer_ttl': jnp.zeros((3,), jnp.int32),
            'signal_pos': jnp.array([[3, 3], [4, 3], [4, 4]], jnp.int32),
            'signal_active': jnp.zeros((3,), jnp.bool_),
            'signal_visible': jnp.zeros((3, 3), jnp.bool_),
            'signal_heard': jnp.zeros((3, 3), jnp.bool_),
            'signal_code': jnp.full((3,), -1, jnp.int32),
            'signal_ttl': jnp.zeros((3,), jnp.int32),
            'consumed': jnp.zeros((3,), jnp.int32),
            'settled': jnp.zeros((3,), jnp.bool_),
            'cancelled': jnp.zeros((3,), jnp.bool_),
            'harvested': jnp.zeros((3,), jnp.int32),
            'delivered_event': jnp.zeros((3,), jnp.bool_),
            'signalled': jnp.zeros((3,), jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        p = s['agent_pos']
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = p[:, None, :] + dirs[None, :, :]
        move = (q[:, :, 0] >= 0) & (q[:, :, 0] <= 8) & (q[:, :, 1] >= 0) & (q[:, :, 1] <= 6)
        at_source = jnp.all(p[:, None, :] == s['source_pos'][None, :, :], axis=-1)
        harvest = jnp.any(at_source & (s['supply'] > 0)[None, :], axis=1) & (jnp.sum(s['inventory'], axis=1) + s['escrow'] < 4)
        deliver = jnp.all(p == s['home_pos'], axis=1) & jnp.all(s['inventory'] >= s['recipe'], axis=1)
        signal = jnp.broadcast_to((s['signals_left'] > 0)[:, None], (3, 3))
        pending = s['offer_ttl'] > 0
        addressed = (jnp.arange(3)[:, None] == s['offer_target'][None, :]) & pending[None, :]
        at_offer = jnp.sum(jnp.abs(p[:, None, :] - s['offer_pos'][None, :, :]), axis=-1) <= 1
        proposer_here = jnp.sum(jnp.abs(p - s['offer_pos']), axis=1) <= 1
        adjacent = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=-1) <= 1
        accept = addressed & at_offer & proposer_here[None, :] & adjacent & (s['inventory'][:, s['offer_ask']] > 0) & s['agent_active'][None, :]
        near_market = jnp.sum(jnp.abs(p[:, None, :] - s['market_pos'][None, :, :]), axis=-1) <= 1
        together = jnp.any(near_market[:, None, :] & near_market[None, :, :], axis=-1) & adjacent
        code = jnp.arange(27)
        target, give, ask = code // 9, (code // 3) % 3, code % 3
        offers = (s['inventory'][:, give] > 0) & ~pending[:, None] & together[:, target] & s['agent_active'][target][None, :] & (jnp.arange(3)[:, None] != target[None, :]) & (give[None, :] != ask[None, :])
        mask = jnp.concatenate([jnp.ones((3, 1), jnp.bool_), move, harvest[:, None], deliver[:, None], signal, pending[:, None], accept, addressed, offers], axis=1)
        available = s['agent_active'][:, None] & ~s['episode_done']
        return mask & (available | (jnp.arange(44)[None, :] == 0))

    def observe(self, s):
        own = jnp.concatenate([jnp.eye(3, dtype=jnp.int32), s['agent_pos'], s['inventory'], s['recipe'], s['escrow'][:, None], s['signals_left'][:, None], s['deliveries'][:, None], s['deadline'][:, None], s['agent_active'][:, None].astype(jnp.int32)], axis=1)
        public = jnp.concatenate([s['t'][None], s['home_pos'].reshape(-1), s['market_pos'].reshape(-1), s['deliveries'], s['deadline'], s['agent_active'].astype(jnp.int32), s['settled'].astype(jnp.int32), s['cancelled'].astype(jnp.int32)])
        adata = jnp.concatenate([s['agent_pos'], s['inventory']], axis=1)
        agent_obs = jnp.concatenate([s['agent_visible'][:, :, None].astype(jnp.int32), jnp.where(s['agent_visible'][:, :, None], adata[None, :, :], -1)], axis=2).reshape(3, 18)
        source_obs = jnp.concatenate([jnp.broadcast_to(s['source_pos'].reshape(1, 6), (3, 6)), jnp.stack([s['source_visible'].astype(jnp.int32), jnp.where(s['source_visible'], s['supply'][None, :], -1)], axis=-1).reshape(3, 6)], axis=1)
        odata = jnp.concatenate([s['offer_pos'], s['offer_target'][:, None], s['offer_give'][:, None], s['offer_ask'][:, None], s['offer_ttl'][:, None]], axis=1)
        offers = jnp.concatenate([s['offer_visible'][:, :, None].astype(jnp.int32), jnp.where(s['offer_visible'][:, :, None], odata[None, :, :], -1)], axis=2).reshape(3, 21)
        sigdata = jnp.concatenate([s['signal_pos'], s['signal_code'][:, None], s['signal_ttl'][:, None]], axis=1)
        signals = jnp.concatenate([s['signal_visible'][:, :, None].astype(jnp.int32), jnp.where(s['signal_visible'][:, :, None], sigdata[None, :, :], -1)], axis=2).reshape(3, 15)
        return jnp.concatenate([own, jnp.broadcast_to(public[None, :], (3, 26)), agent_obs, source_obs, offers, signals], axis=1).astype(jnp.float32)

    def advance(self, s, actions):
        safe = jnp.clip(actions, 0, 43)
        a = jnp.where((safe == actions) & self.legal_actions(s)[jnp.arange(3), safe], safe, 0)
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['agent_pos'] + dirs[jnp.where((a >= 1) & (a <= 4), a, 0)]
        inv, escrow = s['inventory'], s['escrow']
        ttl = jnp.maximum(s['offer_ttl'] - 1, 0)
        target, give, ask = s['offer_target'], s['offer_give'], s['offer_ask']
        cancelled = (s['offer_ttl'] > 0) & ((a == 10) | (s['offer_ttl'] == 1) | (a[jnp.maximum(target, 0)] == 14 + jnp.arange(3)))
        inv = inv + jax.nn.one_hot(give, 3, dtype=jnp.int32) * (escrow * cancelled.astype(jnp.int32))[:, None]
        escrow = jnp.where(cancelled, 0, escrow)
        ttl = jnp.where(cancelled, 0, ttl)
        settled = jnp.zeros((3,), jnp.bool_)
        # Old contracts: expiry/rejection/cancellation precede attempted acceptance.
        for i in range(3):
            r = jnp.maximum(target[i], 0)
            present = (jnp.sum(jnp.abs(p[r] - s['offer_pos'][i])) <= 1) & (jnp.sum(jnp.abs(p[i] - s['offer_pos'][i])) <= 1) & (jnp.sum(jnp.abs(p[i] - p[r])) <= 1)
            trade = (ttl[i] > 0) & (a[r] == 11 + i) & present & (inv[r, ask[i]] > 0) & (escrow[i] == 1)
            unit = trade.astype(jnp.int32)
            inv = inv.at[r, ask[i]].add(-unit).at[r, give[i]].add(unit).at[i, ask[i]].add(unit)
            escrow = escrow.at[i].add(-unit)
            ttl = ttl.at[i].set(jnp.where(trade, 0, ttl[i]))
            settled = settled.at[i].set(trade)
        code = jnp.clip(a - 17, 0, 26)
        new = a >= 17
        new_target, new_give, new_ask = code // 9, (code // 3) % 3, code % 3
        # Both parties must still share a market after simultaneous movement.
        near_market = jnp.sum(jnp.abs(p[:, None, :] - s['market_pos'][None, :, :]), axis=-1) <= 1
        shared_market = jnp.any(near_market & near_market[new_target], axis=1)
        new = new & shared_market & (jnp.sum(jnp.abs(p - p[new_target]), axis=1) <= 1) & (inv[jnp.arange(3), new_give] > 0)
        inv = inv - jax.nn.one_hot(new_give, 3, dtype=jnp.int32) * new[:, None].astype(jnp.int32)
        escrow = jnp.where(new, 1, escrow)
        ttl = jnp.where(new, 5, ttl)
        target = jnp.where(new, new_target, target)
        give = jnp.where(new, new_give, give)
        ask = jnp.where(new, new_ask, ask)
        offer_pos = jnp.where(new[:, None], p, s['offer_pos'])
        supply = s['supply']
        harvested = jnp.zeros((3,), jnp.int32)
        for offset in range(3):
            i = (s['t'] + offset) % 3
            at_source = jnp.all(p[i][None, :] == s['source_pos'], axis=1)
            g = jnp.argmax(at_source)
            harvest = (a[i] == 5) & jnp.any(at_source) & (supply[g] > 0) & (jnp.sum(inv[i]) + escrow[i] < 4)
            unit = harvest.astype(jnp.int32)
            inv = inv.at[i, g].add(unit)
            supply = supply.at[g].add(-unit)
            harvested = harvested.at[i].set(unit)
        deliver = (a == 6) & jnp.all(p == s['home_pos'], axis=1) & jnp.all(inv >= s['recipe'], axis=1)
        used = s['recipe'] * deliver[:, None].astype(jnp.int32)
        inv = inv - used
        deliveries = s['deliveries'] + deliver.astype(jnp.int32)
        t = s['t'] + 1
        active = (t < s['deadline']) & (deliveries < 2)
        closing = (ttl > 0) & (~active | ~active[jnp.maximum(target, 0)])
        inv = inv + jax.nn.one_hot(give, 3, dtype=jnp.int32) * (escrow * closing.astype(jnp.int32))[:, None]
        escrow = jnp.where(closing, 0, escrow)
        ttl = jnp.where(closing, 0, ttl)
        signal = (a >= 7) & (a <= 9)
        heard_now = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=-1) <= 3
        n = dict(s)
        n['t'], n['agent_pos'], n['inventory'], n['escrow'] = t, p, inv, escrow
        n['deliveries'], n['supply'] = deliveries, supply
        n['consumed'] = s['consumed'] + jnp.sum(used, axis=0, dtype=jnp.int32)
        n['offer_pos'], n['offer_target'], n['offer_give'], n['offer_ask'], n['offer_ttl'] = offer_pos, target, give, ask, ttl
        n['signal_pos'] = jnp.where(signal[:, None], p, s['signal_pos'])
        n['signal_code'] = jnp.where(signal, a - 7, s['signal_code'])
        n['signal_ttl'] = jnp.where(signal, 4, jnp.maximum(s['signal_ttl'] - 1, 0))
        n['signal_heard'] = jnp.where(signal[None, :], heard_now, s['signal_heard'])
        n['signals_left'] = s['signals_left'] - signal.astype(jnp.int32)
        n['settled'], n['cancelled'], n['harvested'], n['delivered_event'], n['signalled'] = settled, cancelled | closing, harvested, deliver, signal
        n['terminated'] = ~jnp.any(active)
        n['truncated'] = (t >= self.max_steps) & ~n['terminated']
        n['episode_done'] = n['terminated'] | n['truncated']
        missed = (t == s['deadline']) & (deliveries < 2)
        reward = 18 * deliver.astype(jnp.int32) - signal.astype(jnp.int32) - 3 * missed.astype(jnp.int32)
        return self.refresh(n), reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['episode_done'], lambda _: (state, jnp.zeros((3,), jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['episode_done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((3,), n['episode_done'], jnp.bool_), info
