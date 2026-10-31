"""Border Loom. Original local-information road and trading game; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 252
    num_actions = 14
    max_steps = 112

    def visibility(self, s):
        s['agent_visibility'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=2) <= 2
        s['road_visibility'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['road_pos'][None, :, :]), axis=2) <= 2
        s['town_visibility'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['town_pos'][None, :, :]), axis=2) <= 2
        s['town_visibility'] = s['town_visibility'] | jnp.eye(3, dtype=jnp.bool_)
        return s

    def reset(self, key):
        cells = jnp.arange(49, dtype=jnp.int32)
        xy = jnp.stack([cells % 7, cells // 7], axis=1)
        towns = jnp.array([[1, 3], [5, 1], [5, 5]], dtype=jnp.int32)
        terrain = jnp.zeros((7, 7), dtype=jnp.int32).at[2, 3].set(1).at[4, 3].set(1)
        roads = ((xy[:, 1] == 3) & ((xy[:, 0] <= 2) | (xy[:, 0] >= 4)))
        roads = roads | ((xy[:, 0] == 5) & (xy[:, 1] >= 1) & (xy[:, 1] <= 5))
        roads = roads | (((xy[:, 1] == 1) | (xy[:, 1] == 5)) & (xy[:, 0] >= 4))
        s = {
            't': jnp.array(0, dtype=jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False), 'terrain': terrain,
            'agent_pos': towns, 'agent_active': jnp.ones((3,), dtype=jnp.bool_),
            'agent_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'facing': jnp.array([2, 4, 4], dtype=jnp.int32),
            'timber': jnp.full((3,), 8, dtype=jnp.int32), 'coins': jnp.full((3,), 5, dtype=jnp.int32),
            'cargo': jnp.zeros((3,), dtype=jnp.int32), 'delivered': jnp.zeros((3,), dtype=jnp.int32),
            'score': jnp.zeros((3,), dtype=jnp.int32),
            'sale_value': jax.random.randint(key, (3,), 2, 4, dtype=jnp.int32),
            'offer_target': jnp.full((3,), -1, dtype=jnp.int32), 'offer_ttl': jnp.zeros((3,), dtype=jnp.int32),
            'permits': jnp.zeros((3, 3), dtype=jnp.int32),
            'road_pos': xy, 'road_active': roads, 'road_visibility': jnp.zeros((3, 49), dtype=jnp.bool_),
            'road_owner': jnp.full((49,), -1, dtype=jnp.int32), 'road_toll': jnp.zeros((49,), dtype=jnp.int32),
            'town_pos': towns, 'town_active': jnp.ones((3,), dtype=jnp.bool_),
            'town_visibility': jnp.zeros((3, 3), dtype=jnp.bool_),
            'stock': jnp.full((3,), 4, dtype=jnp.int32), 'received': jnp.zeros((3,), dtype=jnp.int32),
            'e_built': jnp.zeros((49,), dtype=jnp.bool_), 'e_claimed': jnp.zeros((49,), dtype=jnp.bool_),
            'e_toll': jnp.zeros((3,), dtype=jnp.int32), 'e_trade': jnp.zeros((3,), dtype=jnp.bool_),
            'e_treaty': jnp.zeros((3, 3), dtype=jnp.bool_), 'e_move': jnp.zeros((3,), dtype=jnp.bool_),
            'e_discard': jnp.zeros((3,), dtype=jnp.bool_),
        }
        s = self.visibility(s)
        return self.observe(s), s

    def destination(self, s, actions):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        move = (actions >= 1) & (actions <= 4)
        p = s['agent_pos'] + delta[jnp.where(move, actions, 0)]
        bounded = jnp.all((p >= 0) & (p <= 6), axis=1)
        q = jnp.clip(p, 0, 6)
        cell = q[:, 1] * 7 + q[:, 0]
        owner = s['road_owner'][cell]
        oi = jnp.clip(owner, 0, 2)
        charged = move & (s['cargo'] > 0) & (owner >= 0) & (owner != jnp.arange(3))
        charged = charged & (s['road_toll'][cell] == 1) & (s['permits'][oi, jnp.arange(3)] == 0)
        fee = charged.astype(jnp.int32)
        valid = bounded & (s['terrain'][q[:, 1], q[:, 0]] == 0)
        valid = valid & ((s['cargo'] == 0) | s['road_active'][cell]) & (s['coins'] >= fee)
        return jnp.where(valid[:, None], p, s['agent_pos']), valid, fee * valid.astype(jnp.int32), oi

    def ahead(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        p = s['agent_pos'] + delta[s['facing']]
        valid = jnp.all((p >= 0) & (p <= 6), axis=1)
        q = jnp.clip(p, 0, 6)
        valid = valid & (s['terrain'][q[:, 1], q[:, 0]] == 0)
        return q[:, 1] * 7 + q[:, 0], valid

    def legal_actions(self, s):
        me = jnp.arange(3)
        cell = s['agent_pos'][:, 1] * 7 + s['agent_pos'][:, 0]
        ahead, valid_ahead = self.ahead(s)
        home = jnp.all(s['agent_pos'] == s['town_pos'], axis=1)
        dest = (me + 1) % 3
        market = jnp.all(s['agent_pos'] == s['town_pos'][dest], axis=1)
        on_town = jnp.any(jnp.all(s['agent_pos'][:, None, :] == s['town_pos'][None, :, :], axis=2), axis=1)
        near = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=2) <= 1
        offers = (s['offer_target'][None, :] == me[:, None]) & (s['offer_ttl'][None, :] > 0) & near
        m = jnp.zeros((3, 14), dtype=jnp.bool_).at[:, 0].set(True)
        for d in range(1, 5):
            m = m.at[:, d].set(self.destination(s, jnp.full((3,), d, dtype=jnp.int32))[1])
        m = m.at[:, 5].set(valid_ahead & ~s['road_active'][ahead] & (s['timber'] > 0))
        m = m.at[:, 6].set(s['road_active'][cell] & (s['road_owner'][cell] < 0) & ~on_town & (s['timber'] >= 2))
        m = m.at[:, 7].set(s['road_owner'][cell] == me)
        m = m.at[:, 8].set(s['coins'] > 0).at[:, 9].set(s['coins'] > 0)
        m = m.at[:, 10].set(jnp.any(offers, axis=1))
        m = m.at[:, 11].set(home & (s['stock'] > 0) & (s['cargo'] == 0))
        m = m.at[:, 12].set(market & (s['cargo'] > 0) & (s['received'][dest] < 4))
        m = m.at[:, 13].set(s['cargo'] > 0)
        return m & (~s['done'] | (jnp.arange(14)[None, :] == 0))

    def raw_step(self, key, s, actions):
        safe = jnp.clip(actions, 0, 13)
        a = jnp.where((actions >= 0) & (actions < 14) & self.legal_actions(s)[jnp.arange(3), safe], safe, 0)
        pos, valid, fees, owners = self.destination(s, a)
        n = dict(s)
        n['t'] = s['t'] + jnp.array(1, dtype=jnp.int32)
        n['agent_pos'] = pos
        n['facing'] = jnp.where((a >= 1) & (a <= 4), a, s['facing'])
        n['e_move'] = jnp.any(pos != s['agent_pos'], axis=1)
        n['e_toll'] = fees
        n['coins'] = s['coins'] - fees + jnp.sum(jax.nn.one_hot(owners, 3, dtype=jnp.int32) * fees[:, None], axis=0, dtype=jnp.int32)
        n['permits'] = jnp.maximum(s['permits'] - 1, 0)
        n['offer_ttl'] = jnp.maximum(s['offer_ttl'] - 1, 0)
        n['offer_target'] = jnp.where(n['offer_ttl'] > 0, s['offer_target'], -1)
        n['e_built'] = jnp.zeros((49,), dtype=jnp.bool_)
        n['e_claimed'] = jnp.zeros((49,), dtype=jnp.bool_)
        n['e_trade'] = jnp.zeros((3,), dtype=jnp.bool_)
        n['e_treaty'] = jnp.zeros((3, 3), dtype=jnp.bool_)
        n['e_discard'] = a == 13
        payoff = jnp.zeros((3,), dtype=jnp.int32)
        ahead, ahead_valid = self.ahead(s)
        # Construction claims have a rotating, deterministic initiative order.
        for k in range(3):
            i = (s['t'] + k) % 3
            c = ahead[i]
            build = (a[i] == 5) & ~n['road_active'][c] & ahead_valid[i] & (n['timber'][i] > 0)
            n['road_active'] = n['road_active'].at[c].set(n['road_active'][c] | build)
            n['road_owner'] = n['road_owner'].at[c].set(jnp.where(build, i, n['road_owner'][c]))
            n['timber'] = n['timber'].at[i].add(-build.astype(jnp.int32))
            n['e_built'] = n['e_built'].at[c].set(n['e_built'][c] | build)
            cell = pos[i, 1] * 7 + pos[i, 0]
            claim = (a[i] == 6) & (n['road_owner'][cell] < 0) & (n['timber'][i] >= 2)
            n['road_owner'] = n['road_owner'].at[cell].set(jnp.where(claim, i, n['road_owner'][cell]))
            n['road_toll'] = n['road_toll'].at[cell].set(jnp.where(claim, 1, n['road_toll'][cell]))
            n['timber'] = n['timber'].at[i].add(-2 * claim.astype(jnp.int32))
            n['e_claimed'] = n['e_claimed'].at[cell].set(n['e_claimed'][cell] | claim)
            toggle = (a[i] == 7) & (n['road_owner'][cell] == i)
            n['road_toll'] = n['road_toll'].at[cell].set(jnp.where(toggle, 1 - n['road_toll'][cell], n['road_toll'][cell]))
            offer = (a[i] == 8) | (a[i] == 9)
            recipient = jnp.where(a[i] == 8, (i + 1) % 3, (i + 2) % 3)
            n['coins'] = n['coins'].at[i].add(-offer.astype(jnp.int32))
            n['offer_target'] = n['offer_target'].at[i].set(jnp.where(offer, recipient, n['offer_target'][i]))
            n['offer_ttl'] = n['offer_ttl'].at[i].set(jnp.where(offer, 4, n['offer_ttl'][i]))
            # Only an offer visible in the incoming state can be accepted.
            near = jnp.sum(jnp.abs(s['agent_pos'] - s['agent_pos'][i]), axis=1) <= 1
            eligible = (s['offer_target'] == i) & (s['offer_ttl'] > 0) & near
            partner = jnp.argmax(eligible)
            accept = (a[i] == 10) & jnp.any(eligible)
            n['permits'] = n['permits'].at[i, partner].set(jnp.where(accept, 8, n['permits'][i, partner]))
            n['permits'] = n['permits'].at[partner, i].set(jnp.where(accept, 8, n['permits'][partner, i]))
            n['e_treaty'] = n['e_treaty'].at[i, partner].set(n['e_treaty'][i, partner] | accept)
            n['e_treaty'] = n['e_treaty'].at[partner, i].set(n['e_treaty'][partner, i] | accept)
            n['offer_ttl'] = n['offer_ttl'].at[partner].set(jnp.where(accept, 0, n['offer_ttl'][partner]))
            n['offer_target'] = n['offer_target'].at[partner].set(jnp.where(accept, -1, n['offer_target'][partner]))
            load = (a[i] == 11).astype(jnp.int32)
            n['stock'] = n['stock'].at[i].add(-load)
            n['cargo'] = n['cargo'].at[i].add(load)
            target = (i + 1) % 3
            trade = (a[i] == 12) & (n['received'][target] < 4)
            n['cargo'] = n['cargo'].at[i].add(-trade.astype(jnp.int32))
            n['received'] = n['received'].at[target].add(trade.astype(jnp.int32))
            n['delivered'] = n['delivered'].at[i].add(trade.astype(jnp.int32))
            payoff = payoff.at[i].add(trade.astype(jnp.int32) * s['sale_value'][i])
            payoff = payoff.at[target].add(trade.astype(jnp.int32))
            n['e_trade'] = n['e_trade'].at[i].set(trade)
            n['cargo'] = n['cargo'].at[i].set(jnp.where(a[i] == 13, 0, n['cargo'][i]))
        n['score'] = s['score'] + payoff
        n = self.visibility(n)
        n['terminated'] = jnp.all(n['stock'] == 0) & jnp.all(n['cargo'] == 0)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = jax.tree.map(lambda fresh, old: jnp.where(s['done'], old, fresh), n, s)
        rewards = jnp.where(s['done'], 0, payoff).astype(jnp.float32)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, rewards, jnp.full((3,), n['done'], dtype=jnp.bool_), info

    def observe(self, s):
        self_fields = jnp.concatenate([jnp.eye(3), jnp.full((3, 1), s['t'] / 112.0),
            s['sale_value'][:, None] / 3.0, s['agent_pos'] / 6.0, s['timber'][:, None] / 8.0,
            s['coins'][:, None] / 16.0, s['cargo'][:, None]], axis=1)
        agents = jnp.concatenate([s['agent_pos'] / 6.0, s['coins'][:, None] / 16.0,
            s['timber'][:, None] / 8.0, s['cargo'][:, None], (s['offer_target'][:, None] + 1) / 3.0,
            s['offer_ttl'][:, None] / 4.0, s['facing'][:, None] / 4.0], axis=1)
        av = s['agent_visibility']
        local_agents = jnp.concatenate([av[:, :, None], jnp.where(av[:, :, None], agents[None, :, :], -1.0)], axis=2).reshape(3, 27)
        roads = jnp.stack([s['road_active'].astype(jnp.float32), (s['road_owner'] + 1) / 3.0, s['road_toll']], axis=1)
        rv = s['road_visibility']
        local_roads = jnp.concatenate([rv[:, :, None], jnp.where(rv[:, :, None], roads[None, :, :], -1.0)], axis=2).reshape(3, 196)
        towns = jnp.stack([s['stock'] / 4.0, s['received'] / 4.0], axis=1)
        tv = s['town_visibility']
        local_towns = jnp.concatenate([tv[:, :, None], jnp.where(tv[:, :, None], towns[None, :, :], -1.0)], axis=2).reshape(3, 9)
        return jnp.concatenate([self_fields, local_agents, local_roads, local_towns,
            jnp.broadcast_to(s['permits'].reshape(1, 9) / 8.0, (3, 9)), s['delivered'][:, None] / 4.0], axis=1).astype(jnp.float32)
