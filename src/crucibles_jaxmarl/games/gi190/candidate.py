"""River Charter. Original prospective rules; never executed by the author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 49
    num_actions = 14
    max_steps = 88

    def reset(self, key):
        rows = jnp.array([1, 4, 7], dtype=jnp.int32)
        towns = jnp.stack([jnp.full((3,), 3, dtype=jnp.int32), rows], axis=1)
        docks = jnp.stack([jnp.full((3,), 4, dtype=jnp.int32), rows], axis=1)
        bridges = jnp.stack([jnp.full((3,), 5, dtype=jnp.int32), rows], axis=1)
        terrain = jnp.repeat((jnp.arange(9, dtype=jnp.int32) // 3 + 1)[:, None], 11, axis=1)
        terrain = terrain.at[:, 5].set(4)
        s = {
            't': jnp.array(0, dtype=jnp.int32),
            'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'season': jnp.array(0, dtype=jnp.int32), 'source_water': jnp.array(4, dtype=jnp.int32),
            'terrain': terrain,
            'agent_pos': towns, 'agent_active': jnp.ones((3,), dtype=jnp.bool_),
            'agent_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'grain': jnp.full((3,), 6, dtype=jnp.int32), 'cargo': jnp.zeros((3,), dtype=jnp.int32),
            'taste': jax.random.randint(key, (3,), 1, 3, dtype=jnp.int32),
            'score': jnp.zeros((3,), dtype=jnp.int32),
            'town_pos': towns, 'town_active': jnp.ones((3,), dtype=jnp.bool_),
            'town_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'dock_pos': docks, 'dock_active': jnp.ones((3,), dtype=jnp.bool_),
            'dock_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'divert': jnp.ones((3,), dtype=jnp.int32), 'pledge': jnp.zeros((3,), dtype=jnp.int32),
            'bridge_pos': bridges, 'bridge_active': jnp.ones((3,), dtype=jnp.bool_),
            'bridge_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'built': jnp.zeros((3,), dtype=jnp.bool_), 'toll': jnp.ones((3,), dtype=jnp.int32),
            'boat_pos': bridges, 'boat_active': jnp.zeros((3,), dtype=jnp.bool_),
            'boat_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'boat_load': jnp.zeros((3,), dtype=jnp.int32),
            'market_pos': jnp.array([[8, 4]], dtype=jnp.int32),
            'market_active': jnp.ones((1,), dtype=jnp.bool_),
            'market_visibility': jnp.ones((3, 1), dtype=jnp.bool_),
            'e_move': jnp.zeros((3,), dtype=jnp.bool_), 'e_cross': jnp.zeros((3,), dtype=jnp.bool_),
            'e_flow': jnp.zeros((3,), dtype=jnp.int32), 'e_arrival': jnp.zeros((3,), dtype=jnp.int32),
            'e_pledge': jnp.zeros((3,), dtype=jnp.bool_), 'e_payoff': jnp.zeros((3,), dtype=jnp.int32),
        }
        return self.observe(s), s

    def destination(self, s, actions):
        # A bridge is a real traversable river tile; tolls apply only on entry.
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        move = (actions >= 1) & (actions <= 4)
        p = s['agent_pos'] + delta[jnp.where(move, actions, 0)]
        owner = jnp.argmin(jnp.abs(p[:, 1, None] - s['bridge_pos'][None, :, 1]), axis=1)
        row = p[:, 1] == s['bridge_pos'][owner, 1]
        river = p[:, 0] == 5
        crossing = move & river & (s['agent_pos'][:, 0] != 5)
        fee = (crossing & (owner != jnp.arange(3)) & (s['toll'][owner] == 1)).astype(jnp.int32)
        valid = jnp.all((p >= jnp.array([0, 0])) & (p <= jnp.array([10, 8])), axis=1)
        valid = valid & (~river | (row & s['built'][owner])) & (s['grain'] >= fee)
        return jnp.where(valid[:, None], p, s['agent_pos']), valid, fee * valid.astype(jnp.int32), owner

    def legal_actions(self, s):
        home = jnp.all(s['agent_pos'] == s['town_pos'], axis=1)
        dock = jnp.all(s['agent_pos'] == s['dock_pos'], axis=1)
        any_town = jnp.any(jnp.all(s['agent_pos'][:, None, :] == s['town_pos'][None, :, :], axis=2), axis=1)
        market = jnp.all(s['agent_pos'] == s['market_pos'][0], axis=1)
        m = jnp.zeros((3, 14), dtype=jnp.bool_).at[:, 0].set(True)
        for d in range(1, 5):
            m = m.at[:, d].set(self.destination(s, jnp.full((3,), d, dtype=jnp.int32))[1])
        m = m.at[:, 5].set(dock & ~s['built'] & (s['grain'] >= 2))
        m = m.at[:, 6].set(dock & (s['pledge'] == 0))
        m = m.at[:, 7].set(dock & (s['pledge'] == 0) & (s['grain'] >= 1) & (s['season'] < 8))
        m = m.at[:, 8].set(home & (s['grain'] > 0) & (s['cargo'] < 2))
        m = m.at[:, 9].set(dock & (s['cargo'] > 0) & ~s['boat_active'])
        m = m.at[:, 10].set(any_town & (s['cargo'] > 0))
        m = m.at[:, 11].set(dock & s['built'])
        m = m.at[:, 12].set(home & (s['grain'] > 0))
        m = m.at[:, 13].set(market & (s['cargo'] > 0))
        return m & (~s['done'] | (jnp.arange(14)[None, :] == 0))

    def raw_step(self, key, s, actions):
        # Illegal/out-of-range input becomes wait. Co-location is permitted.
        safe = jnp.clip(actions, 0, 13)
        a = jnp.where((actions >= 0) & (actions < 14) & self.legal_actions(s)[jnp.arange(3), safe], safe, 0)
        pos, valid, fees, owners = self.destination(s, a)
        n = dict(s)
        n['t'] = s['t'] + jnp.array(1, dtype=jnp.int32)
        n['agent_pos'] = pos
        n['e_move'] = jnp.any(pos != s['agent_pos'], axis=1)
        n['e_cross'] = n['e_move'] & (pos[:, 0] == 5) & (s['agent_pos'][:, 0] != 5)
        receipts = jnp.sum(jax.nn.one_hot(owners, 3, dtype=jnp.int32) * fees[:, None], axis=0, dtype=jnp.int32)
        n['grain'] = s['grain'] - fees + receipts
        build = a == 5
        n['built'] = s['built'] | build
        n['grain'] = n['grain'] - 2 * build.astype(jnp.int32)
        n['divert'] = jnp.where(a == 6, 1 - s['divert'], s['divert'])
        pledge = a == 7
        n['grain'] = n['grain'] - pledge.astype(jnp.int32)
        n['pledge'] = jnp.where(pledge, 2, s['pledge'])
        n['divert'] = jnp.where(n['pledge'] > 0, 0, n['divert'])
        n['e_pledge'] = pledge
        n['toll'] = jnp.where(a == 11, 1 - s['toll'], s['toll'])
        load = (a == 8).astype(jnp.int32)
        n['grain'] = n['grain'] - load
        n['cargo'] = s['cargo'] + load
        launch = a == 9
        n['boat_active'] = s['boat_active'] | launch
        n['boat_pos'] = jnp.where(launch[:, None], s['bridge_pos'], s['boat_pos'])
        n['boat_load'] = jnp.where(launch, n['cargo'], s['boat_load'])
        n['cargo'] = jnp.where(launch, 0, n['cargo'])
        # Land delivery is a real transfer, with no fictitious gift reward.
        target = jnp.argmax(jnp.all(pos[:, None, :] == s['town_pos'][None, :, :], axis=2), axis=1)
        gift = jnp.where(a == 10, n['cargo'], 0)
        n['grain'] = n['grain'] + jnp.sum(jax.nn.one_hot(target, 3, dtype=jnp.int32) * gift[:, None], axis=0, dtype=jnp.int32)
        n['cargo'] = jnp.where(a == 10, 0, n['cargo'])
        eat = (a == 12).astype(jnp.int32)
        n['grain'] = n['grain'] - eat
        sales = jnp.where(a == 13, n['cargo'], 0)
        n['cargo'] = jnp.where(a == 13, 0, n['cargo'])
        payoff = eat * s['taste'] + 3 * sales
        # Barges move one actual river cell this step, including a launch step.
        ends = jnp.array([4, 7, 8], dtype=jnp.int32)
        new_y = jnp.minimum(n['boat_pos'][:, 1] + n['boat_active'].astype(jnp.int32), ends)
        n['boat_pos'] = n['boat_pos'].at[:, 1].set(new_y)
        arrival = n['boat_active'] & (new_y == ends)
        amounts = jnp.where(arrival, n['boat_load'], 0)
        n['e_arrival'] = amounts
        n['grain'] = n['grain'].at[1].add(amounts[0]).at[2].add(amounts[1])
        payoff = payoff.at[2].add(2 * amounts[2])
        n['boat_active'] = n['boat_active'] & ~arrival
        n['boat_load'] = jnp.where(arrival, 0, n['boat_load'])
        # Flood order is upstream to downstream after investment and shipping.
        flood = (n['t'] % 8 == 0) & (s['season'] < 8)
        water = s['source_water']
        flow = jnp.zeros((3,), dtype=jnp.int32)
        for i in range(3):
            intake = jnp.where(flood & (n['divert'][i] == 1), jnp.minimum(water, 2), 0)
            flow = flow.at[i].set(intake)
            water = water - intake
        n['grain'] = n['grain'] + flow
        n['e_flow'] = flow
        n['season'] = s['season'] + flood.astype(jnp.int32)
        n['pledge'] = jnp.maximum(n['pledge'] - flood.astype(jnp.int32), 0)
        n['score'] = s['score'] + payoff
        n['e_payoff'] = payoff
        n['terminated'] = (n['season'] >= 8) & ~jnp.any(n['boat_active'])
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        # The addendum requires terminal states, causes and clocks to absorb.
        n = jax.tree.map(lambda fresh, old: jnp.where(s['done'], old, fresh), n, s)
        rewards = jnp.where(s['done'], 0, payoff).astype(jnp.float32)
        dones = jnp.full((3,), n['done'], dtype=jnp.bool_)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, rewards, dones, info

    def observe(self, s):
        public = jnp.concatenate([
            jnp.stack([s['t'] / 88.0, s['season'] / 8.0, s['source_water'] / 4.0]),
            (s['agent_pos'] / jnp.array([10.0, 8.0])).reshape(-1),
            s['grain'] / 64.0, s['cargo'] / 2.0, s['divert'].astype(jnp.float32), s['pledge'] / 2.0,
            s['built'].astype(jnp.float32), s['toll'].astype(jnp.float32),
            (s['boat_pos'] / jnp.array([10.0, 8.0])).reshape(-1), s['boat_active'].astype(jnp.float32),
            s['boat_load'] / 2.0, s['e_flow'] / 2.0, s['score'] / 256.0,
        ])
        return jnp.concatenate([jnp.broadcast_to(public, (3, 45)), jnp.eye(3), s['taste'][:, None] / 2.0], axis=1).astype(jnp.float32)
