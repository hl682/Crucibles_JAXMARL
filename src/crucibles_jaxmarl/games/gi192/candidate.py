"""Orchard Commons. Original seasonal pollination and shared-tool rules, unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 50
    num_actions = 14
    max_steps = 40

    def reset(self, key):
        plots = jnp.array([[1, 2], [4, 2], [7, 2]], dtype=jnp.int32)
        agents = jnp.array([[1, 3], [4, 3], [7, 3]], dtype=jnp.int32)
        terrain = jnp.zeros((7, 9), dtype=jnp.int32).at[1, 1:8].set(1).at[5, 4].set(2)
        s = {
            't': jnp.array(0, dtype=jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'season': jnp.array(0, dtype=jnp.int32), 'terrain': terrain,
            'agent_pos': agents, 'agent_active': jnp.ones((3,), dtype=jnp.bool_),
            'agent_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'basket': jnp.zeros((3,), dtype=jnp.int32), 'basket_type': jnp.full((3,), -1, dtype=jnp.int32),
            'compost': jnp.full((3,), 2, dtype=jnp.int32), 'score': jnp.zeros((3,), dtype=jnp.int32),
            'favorite': jax.random.randint(key, (3,), 0, 2, dtype=jnp.int32),
            'plot_pos': plots, 'plot_active': jnp.ones((3,), dtype=jnp.bool_),
            'plot_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'stage': jnp.zeros((3,), dtype=jnp.int32), 'crop': jnp.full((3,), -1, dtype=jnp.int32),
            'moisture': jnp.zeros((3,), dtype=jnp.int32), 'soil': jnp.full((3,), 2, dtype=jnp.int32),
            'pollen': jnp.zeros((3,), dtype=jnp.bool_), 'fruit': jnp.zeros((3,), dtype=jnp.int32),
            'ripe_age': jnp.zeros((3,), dtype=jnp.int32),
            'tool_pos': jnp.array([[4, 3]], dtype=jnp.int32), 'tool_active': jnp.ones((1,), dtype=jnp.bool_),
            'tool_visibility': jnp.ones((3, 1), dtype=jnp.bool_), 'tool_holder': jnp.array([-1], dtype=jnp.int32),
            'habitat_pos': jnp.array([[4, 0]], dtype=jnp.int32), 'habitat_active': jnp.ones((1,), dtype=jnp.bool_),
            'habitat_visibility': jnp.ones((3, 1), dtype=jnp.bool_), 'bees': jnp.array([1], dtype=jnp.int32),
            'pond_pos': jnp.array([[4, 5]], dtype=jnp.int32), 'pond_active': jnp.ones((1,), dtype=jnp.bool_),
            'pond_visibility': jnp.ones((3, 1), dtype=jnp.bool_), 'water': jnp.array([3], dtype=jnp.int32),
            'market_pos': jnp.array([[4, 6]], dtype=jnp.int32), 'market_active': jnp.ones((1,), dtype=jnp.bool_),
            'market_visibility': jnp.ones((3, 1), dtype=jnp.bool_),
            'e_water': jnp.zeros((3,), dtype=jnp.int32), 'e_tend': jnp.zeros((3,), dtype=jnp.bool_),
            'e_harvest': jnp.zeros((3,), dtype=jnp.int32), 'e_sale': jnp.zeros((3,), dtype=jnp.int32),
            'e_handoff': jnp.zeros((1,), dtype=jnp.bool_), 'e_blossom': jnp.zeros((3,), dtype=jnp.bool_),
            'e_spoil': jnp.zeros((3,), dtype=jnp.int32), 'e_move': jnp.zeros((3,), dtype=jnp.bool_),
        }
        return self.observe(s), s

    def destination(self, s, actions):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        move = (actions >= 1) & (actions <= 4)
        p = s['agent_pos'] + delta[jnp.where(move, actions, 0)]
        valid = jnp.all((p >= 0) & (p <= jnp.array([8, 6])), axis=1)
        q = jnp.clip(p, jnp.array([0, 0]), jnp.array([8, 6]))
        valid = valid & (s['terrain'][q[:, 1], q[:, 0]] != 2)
        return jnp.where(valid[:, None], p, s['agent_pos']), valid

    def legal_actions(self, s):
        distance = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['plot_pos'][None, :, :]), axis=2)
        target = jnp.argmin(distance, axis=1)
        near_plot = jnp.min(distance, axis=1) <= 1
        own = jnp.sum(jnp.abs(s['agent_pos'] - s['plot_pos']), axis=1) <= 1
        habitat = jnp.sum(jnp.abs(s['agent_pos'] - s['habitat_pos'][0]), axis=1) <= 1
        tool = jnp.sum(jnp.abs(s['agent_pos'] - s['tool_pos'][0]), axis=1) <= 1
        market = jnp.all(s['agent_pos'] == s['market_pos'][0], axis=1)
        holding = s['tool_holder'][0] == jnp.arange(3)
        m = jnp.zeros((3, 14), dtype=jnp.bool_).at[:, 0].set(True)
        for d in range(1, 5):
            m = m.at[:, d].set(self.destination(s, jnp.full((3,), d, dtype=jnp.int32))[1])
        plant = own & (s['stage'] == 0) & (s['season'] < 5)
        m = m.at[:, 5].set(plant).at[:, 6].set(plant)
        m = m.at[:, 7].set(near_plot & (s['stage'][target] >= 1) & (s['stage'][target] <= 2) &
            (s['moisture'][target] < 3) & (s['water'][0] > 0) & (s['season'] < 6))
        m = m.at[:, 8].set(habitat & (s['bees'][0] < 4) & (s['season'] < 6))
        m = m.at[:, 9].set(tool & (s['tool_holder'][0] < 0))
        m = m.at[:, 10].set(holding)
        m = m.at[:, 11].set(own & holding & (s['stage'] == 3) & (s['fruit'] > 0) &
            (s['basket'] < 3) & ((s['basket'] == 0) | (s['basket_type'] == s['crop'])))
        m = m.at[:, 12].set(market & (s['basket'] > 0))
        m = m.at[:, 13].set(near_plot & (s['soil'][target] < 3) & (s['compost'] > 0))
        return m & (~s['done'] | (jnp.arange(14)[None, :] == 0))

    def raw_step(self, key, s, actions):
        safe = jnp.clip(actions, 0, 13)
        a = jnp.where((actions >= 0) & (actions < 14) & self.legal_actions(s)[jnp.arange(3), safe], safe, 0)
        pos, valid = self.destination(s, a)
        n = dict(s)
        n['t'] = s['t'] + jnp.array(1, dtype=jnp.int32)
        n['agent_pos'] = pos
        n['e_move'] = jnp.any(pos != s['agent_pos'], axis=1)
        n['e_water'] = jnp.zeros((3,), dtype=jnp.int32)
        n['e_tend'] = jnp.zeros((3,), dtype=jnp.bool_)
        n['e_harvest'] = jnp.zeros((3,), dtype=jnp.int32)
        n['e_sale'] = jnp.zeros((3,), dtype=jnp.int32)
        n['e_handoff'] = jnp.zeros((1,), dtype=jnp.bool_)
        payoff = jnp.zeros((3,), dtype=jnp.int32)
        # Finite water and the single tool are resolved by rotating initiative.
        for k in range(3):
            i = (s['t'] + k) % 3
            plant = (a[i] == 5) | (a[i] == 6)
            n['stage'] = n['stage'].at[i].set(jnp.where(plant, 1, n['stage'][i]))
            n['crop'] = n['crop'].at[i].set(jnp.where(plant, a[i] - 5, n['crop'][i]))
            n['pollen'] = n['pollen'].at[i].set(jnp.where(plant, False, n['pollen'][i]))
            n['ripe_age'] = n['ripe_age'].at[i].set(jnp.where(plant, 0, n['ripe_age'][i]))
            distance = jnp.sum(jnp.abs(s['plot_pos'] - pos[i]), axis=1)
            p = jnp.argmin(distance)
            water = (a[i] == 7) & (n['water'][0] > 0) & (n['moisture'][p] < 3)
            n['water'] = n['water'].at[0].add(-water.astype(jnp.int32))
            n['moisture'] = n['moisture'].at[p].set(jnp.where(water, jnp.minimum(3, n['moisture'][p] + 2), n['moisture'][p]))
            n['e_water'] = n['e_water'].at[p].add(water.astype(jnp.int32))
            tend = (a[i] == 8) & (n['bees'][0] < 4)
            n['bees'] = n['bees'].at[0].set(jnp.where(tend, jnp.minimum(4, n['bees'][0] + 2), n['bees'][0]))
            n['e_tend'] = n['e_tend'].at[i].set(tend)
            pick = (a[i] == 9) & (n['tool_holder'][0] < 0)
            n['tool_holder'] = n['tool_holder'].at[0].set(jnp.where(pick, i, n['tool_holder'][0]))
            near = (jnp.sum(jnp.abs(pos - pos[i]), axis=1) <= 1) & (jnp.arange(3) != i)
            recipient = jnp.argmax(near).astype(jnp.int32)
            give = (a[i] == 10) & (n['tool_holder'][0] == i)
            next_holder = jnp.where(jnp.any(near), recipient, -1)
            n['tool_holder'] = n['tool_holder'].at[0].set(jnp.where(give, next_holder, n['tool_holder'][0]))
            n['tool_pos'] = n['tool_pos'].at[0].set(jnp.where(give, pos[i], n['tool_pos'][0]))
            n['e_handoff'] = n['e_handoff'].at[0].set(n['e_handoff'][0] | give | pick)
            harvest = (a[i] == 11) & (n['tool_holder'][0] == i)
            amount = jnp.where(harvest, jnp.minimum(n['fruit'][i], 3 - n['basket'][i]), 0)
            n['basket_type'] = n['basket_type'].at[i].set(jnp.where(amount > 0, n['crop'][i], n['basket_type'][i]))
            n['basket'] = n['basket'].at[i].add(amount)
            n['fruit'] = n['fruit'].at[i].add(-amount)
            emptied = (amount > 0) & (n['fruit'][i] == 0)
            n['stage'] = n['stage'].at[i].set(jnp.where(emptied, 0, n['stage'][i]))
            n['soil'] = n['soil'].at[i].set(jnp.where(emptied, jnp.maximum(0, n['soil'][i] - 1), n['soil'][i]))
            n['e_harvest'] = n['e_harvest'].at[i].set(amount)
            sold = jnp.where(a[i] == 12, n['basket'][i], 0)
            price = 1 + (s['favorite'][i] == n['basket_type'][i]).astype(jnp.int32)
            price = price + (s['season'] % 2 == n['basket_type'][i]).astype(jnp.int32)
            payoff = payoff.at[i].set(sold * price)
            n['basket'] = n['basket'].at[i].set(jnp.where(a[i] == 12, 0, n['basket'][i]))
            n['basket_type'] = n['basket_type'].at[i].set(jnp.where(a[i] == 12, -1, n['basket_type'][i]))
            n['e_sale'] = n['e_sale'].at[i].set(sold)
            compost = (a[i] == 13) & (n['soil'][p] < 3) & (n['compost'][i] > 0)
            n['soil'] = n['soil'].at[p].add(compost.astype(jnp.int32))
            n['compost'] = n['compost'].at[i].add(-compost.astype(jnp.int32))
        holder = n['tool_holder'][0]
        n['tool_pos'] = n['tool_pos'].at[0].set(jnp.where(holder >= 0, pos[jnp.clip(holder, 0, 2)], n['tool_pos'][0]))
        # Growth snapshots avoid a seedling flowering and ripening in one tick.
        tick = n['t'] % 4 == 0
        grow = tick & (s['season'] < 6)
        old_stage = n['stage']
        moist = n['moisture'] > 0
        blossom = grow & (old_stage == 1) & moist
        ripen = grow & (old_stage == 2) & moist
        n['pollen'] = jnp.where(blossom, n['bees'][0] >= 2, n['pollen'])
        n['fruit'] = jnp.where(ripen, 1 + n['pollen'].astype(jnp.int32) + (n['soil'] >= 2).astype(jnp.int32), n['fruit'])
        n['stage'] = jnp.where(blossom, 2, jnp.where(ripen, 3, old_stage))
        n['e_blossom'] = blossom
        n['ripe_age'] = jnp.where(ripen, 0, jnp.where(tick & (old_stage == 3), n['ripe_age'] + 1, n['ripe_age']))
        spoil = tick & (old_stage == 3) & (n['ripe_age'] >= 2)
        n['e_spoil'] = jnp.where(spoil, n['fruit'], 0)
        n['fruit'] = jnp.where(spoil, 0, n['fruit'])
        n['stage'] = jnp.where(spoil, 0, n['stage'])
        n['ripe_age'] = jnp.where(spoil, 0, n['ripe_age'])
        n['moisture'] = jnp.where(grow & ((old_stage == 1) | (old_stage == 2)), jnp.maximum(0, n['moisture'] - 1), n['moisture'])
        n['bees'] = jnp.where(grow, jnp.maximum(0, n['bees'] - 1), n['bees'])
        n['water'] = jnp.where(grow, jnp.minimum(4, n['water'] + 2), n['water'])
        n['season'] = s['season'] + grow.astype(jnp.int32)
        closed_unripe = (n['season'] >= 6) & (n['stage'] < 3)
        n['stage'] = jnp.where(closed_unripe, 0, n['stage'])
        n['score'] = s['score'] + payoff
        n['terminated'] = (n['season'] >= 6) & ~jnp.any(n['stage'] == 3) & ~jnp.any(n['basket'] > 0)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = jax.tree.map(lambda fresh, old: jnp.where(s['done'], old, fresh), n, s)
        rewards = jnp.where(s['done'], 0, payoff).astype(jnp.float32)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, rewards, jnp.full((3,), n['done'], dtype=jnp.bool_), info

    def observe(self, s):
        public = jnp.concatenate([
            jnp.stack([s['t'] / 40.0, s['season'] / 6.0, s['water'][0] / 4.0, s['bees'][0] / 4.0]),
            (s['agent_pos'] / jnp.array([8.0, 6.0])).reshape(-1), s['basket'] / 3.0,
            (s['basket_type'] + 1) / 2.0, s['compost'] / 2.0, (s['tool_holder'] + 1) / 3.0,
            (s['tool_pos'] / jnp.array([8.0, 6.0])).reshape(-1), s['stage'] / 3.0,
            (s['crop'] + 1) / 2.0, s['moisture'] / 3.0, s['soil'] / 3.0,
            s['pollen'].astype(jnp.float32), s['fruit'] / 3.0, s['ripe_age'] / 2.0, s['score'] / 128.0,
        ])
        return jnp.concatenate([jnp.broadcast_to(public, (3, 46)), jnp.eye(3), s['favorite'][:, None].astype(jnp.float32)], axis=1).astype(jnp.float32)
