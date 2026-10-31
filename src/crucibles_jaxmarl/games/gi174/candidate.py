"""Floodgate Commons: original, source-authored, never executed by its author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 59
    num_actions = 9
    max_steps = 96

    def reset(self, key):
        state = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False, jnp.bool_),
            'terminated': jnp.array(False, jnp.bool_),
            'truncated': jnp.array(False, jnp.bool_),
            'map': jnp.zeros((7, 9), jnp.int32).at[2, 1:8].set(1),
            'agent_pos': jnp.array([[3, 3], [5, 3], [7, 3]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_),
            'agent_visibility': jnp.ones((3, 3), jnp.bool_),
            'kits': jnp.full((3,), 2, jnp.int32),
            'private_value': jax.random.randint(key, (3,), 1, 4, dtype=jnp.int32),
            'gate_pos': jnp.array([[3, 2], [5, 2], [7, 2]], jnp.int32),
            'gate_active': jnp.ones((3,), jnp.bool_),
            'gate_visibility': jnp.ones((3, 3), jnp.bool_),
            'gate_open': jnp.array([False, False, True], jnp.bool_),
            'basin_pos': jnp.array([[2, 2], [4, 2], [6, 2]], jnp.int32),
            'basin_active': jnp.ones((3,), jnp.bool_),
            'basin_visibility': jnp.ones((3, 3), jnp.bool_),
            'water': jnp.array([6, 1, 1], jnp.int32),
            'bank': jnp.zeros((3,), jnp.int32),
            'flow': jnp.zeros((3,), jnp.int32),
            'plot_pos': jnp.array([[2, 4], [4, 4], [6, 4]], jnp.int32),
            'plot_active': jnp.ones((3,), jnp.bool_),
            'plot_visibility': jnp.ones((3, 3), jnp.bool_),
            'integrity': jnp.full((3,), 4, jnp.int32),
            'growth': jnp.zeros((3,), jnp.int32),
            'harvests': jnp.zeros((3,), jnp.int32),
            'flood': jnp.zeros((3,), jnp.int32),
            'toggled': jnp.zeros((3,), jnp.bool_),
            'reinforced': jnp.zeros((3,), jnp.bool_),
            'shared': jnp.zeros((3,), jnp.bool_),
            'irrigated': jnp.zeros((3,), jnp.bool_),
            'harvest_event': jnp.zeros((3,), jnp.bool_),
            'spill': jnp.zeros((3,), jnp.int32),
        }
        return self.observe(state), state

    def distances(self, first, second):
        return jnp.sum(jnp.abs(first[:, None, :] - second[None, :, :]), axis=-1)

    def legal_actions(self, state):
        pos = state['agent_pos']
        directions = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = pos[:, None, :] + directions[None, :, :]
        inside = jnp.all((proposed >= 0) & (proposed <= jnp.array([8, 6])), axis=-1)
        clipped = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([8, 6]))
        movement = inside & (state['map'][clipped[:, :, 1], clipped[:, :, 0]] == 0)
        near_gate = self.distances(pos, state['gate_pos']) <= 1
        near_plot = self.distances(pos, state['plot_pos']) <= 1
        can_repair = (state['bank'] < 2) | (state['integrity'] < 4)
        peers = self.distances(pos, pos) <= 1
        peers = peers & ~jnp.eye(3, dtype=jnp.bool_) & (state['kits'][None, :] < 4)
        own_near = jnp.sum(jnp.abs(pos - state['plot_pos']), axis=-1) <= 1
        specials = jnp.stack([
            jnp.any(near_gate, axis=1),
            (state['kits'] > 0) & jnp.any(near_plot & can_repair[None, :], axis=1),
            (state['kits'] > 0) & jnp.any(peers, axis=1),
            own_near & (state['water'] > 0) & (state['integrity'] > 0) & (state['harvests'] < 2),
        ], axis=1)
        mask = jnp.concatenate([jnp.ones((3, 1), jnp.bool_), movement, specials], axis=1)
        return jnp.where(state['done'], jnp.arange(9)[None, :] == 0, mask)

    def advance(self, key, state, actions):
        s = dict(state)
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        pos = state['agent_pos'] + directions[jnp.minimum(actions, 4)] * (actions <= 4)[:, None]
        s['agent_pos'] = pos
        s['t'] = state['t'] + jnp.array(1, jnp.int32)

        # Every operator targets the lowest-index reachable gate. Odd parity toggles.
        gate_distance = self.distances(pos, state['gate_pos'])
        gate_choice = jnp.argmin(jnp.where(gate_distance <= 1, jnp.arange(3)[None, :], 9), axis=1)
        operate = (actions == 5) & jnp.any(gate_distance <= 1, axis=1)
        toggle_count = jnp.sum(operate[:, None] & (gate_choice[:, None] == jnp.arange(3)[None, :]), axis=0)
        s['toggled'] = (toggle_count % 2) == 1
        s['gate_open'] = state['gate_open'] ^ s['toggled']

        # Reinforcements/repairs spend only accepted kits, in ascending keeper order.
        kits = state['kits']
        bank = state['bank']
        integrity = state['integrity']
        reinforced = jnp.zeros((3,), jnp.bool_)
        plot_distance = self.distances(pos, state['plot_pos'])
        for i in range(3):
            eligible = (plot_distance[i] <= 1) & ((bank < 2) | (integrity < 4))
            target = jnp.argmin(jnp.where(eligible, jnp.arange(3), 9))
            accepted = (actions[i] == 6) & (kits[i] > 0) & jnp.any(eligible)
            raise_bank = accepted & (bank[target] < 2)
            patch_plot = accepted & ~raise_bank
            bank = bank.at[target].add(raise_bank.astype(jnp.int32))
            integrity = integrity.at[target].add(patch_plot.astype(jnp.int32))
            kits = kits.at[i].add(-accepted.astype(jnp.int32))
            reinforced = reinforced.at[target].set(reinforced[target] | accepted)
        s['bank'] = bank
        s['reinforced'] = reinforced

        # At most one gift per sender and receiver, selected from this pre-gift inventory.
        peer_distance = self.distances(pos, pos)
        recipients = (peer_distance <= 1) & ~jnp.eye(3, dtype=jnp.bool_) & (kits[None, :] < 4)
        target = jnp.argmin(jnp.where(recipients, peer_distance * 4 + jnp.arange(3)[None, :], 99), axis=1)
        offer = (actions == 7) & (kits > 0) & jnp.any(recipients, axis=1)
        requests = offer[:, None] & (target[:, None] == jnp.arange(3)[None, :])
        winner = jnp.argmin(jnp.where(requests, jnp.arange(3)[:, None], 9), axis=0)
        accepted = requests & (winner[None, :] == jnp.arange(3)[:, None])
        sent = jnp.any(accepted, axis=1)
        received = jnp.any(accepted, axis=0)
        kits = kits - sent.astype(jnp.int32) + received.astype(jnp.int32)
        s['kits'] = kits
        s['shared'] = sent

        # Simultaneous edge flux uses pre-flux pools: no same-tick cascade.
        inflow = jnp.where((s['t'] % 4) == 0, 2, 0).astype(jnp.int32)
        available = state['water'].at[0].add(inflow)
        flow = jnp.where(s['gate_open'], jnp.minimum(available, 2), 0)
        arrived = jnp.concatenate([jnp.zeros((1,), jnp.int32), flow[:2]])
        pool = available - flow + arrived
        spill = jnp.maximum(pool - (6 + bank), 0)
        water = jnp.minimum(pool, 6 + bank)
        s['flow'] = flow
        s['spill'] = spill
        s['flood'] = jnp.where(spill > 0, 3, jnp.maximum(state['flood'] - 1, 0))
        integrity = jnp.maximum(integrity - (spill > 0).astype(jnp.int32), 0)
        s['integrity'] = integrity

        # A keeper irrigates only its own plot; upstream releases arrive before irrigation.
        own_near = jnp.sum(jnp.abs(pos - state['plot_pos']), axis=-1) <= 1
        irrigated = (actions == 8) & own_near & (water > 0) & (integrity > 0) & (state['harvests'] < 2)
        water = water - irrigated.astype(jnp.int32)
        growth = state['growth'] + irrigated.astype(jnp.int32)
        harvest_event = growth >= 3
        s['water'] = water
        s['growth'] = jnp.where(harvest_event, 0, growth)
        s['harvests'] = state['harvests'] + harvest_event.astype(jnp.int32)
        s['irrigated'] = irrigated
        s['harvest_event'] = harvest_event
        success = jnp.all(s['harvests'] >= 2)
        failure = jnp.any(integrity == 0)
        s['terminated'] = success | failure
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        rewards = harvest_event.astype(jnp.float32) * state['private_value'].astype(jnp.float32)
        rewards = rewards - (spill > 0).astype(jnp.float32) + success.astype(jnp.float32) - failure.astype(jnp.float32)
        rewards = rewards - jnp.where(actions == 6, (state['kits'] - kits).clip(0, 1), 0).astype(jnp.float32) * 0.05
        return s, rewards.astype(jnp.float32)

    def observe(self, state):
        public = jnp.concatenate([
            state['t'][None] / 96.0,
            (state['agent_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
            state['kits'] / 4.0, state['water'] / 8.0,
            state['gate_open'].astype(jnp.float32), state['bank'] / 2.0,
            state['integrity'] / 4.0, state['growth'] / 3.0,
            state['harvests'] / 2.0, state['flood'] / 3.0, state['flow'] / 2.0,
            (state['plot_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
            (state['gate_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
            (state['basin_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
            jnp.stack([state['done'], state['terminated'], state['truncated']]).astype(jnp.float32),
        ])
        return jnp.concatenate([
            jnp.eye(3, dtype=jnp.float32), state['private_value'][:, None] / 3.0,
            jnp.broadcast_to(public[None, :], (3, 55)),
        ], axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions.astype(jnp.int32), 0, self.num_actions - 1)
        valid = (actions >= 0) & (actions < self.num_actions)
        valid = valid & self.legal_actions(state)[jnp.arange(3), clipped]
        clean = jnp.where(valid, clipped, 0)
        proposed, rewards = self.advance(key, state, clean)
        next_state = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), proposed, state)
        rewards = jnp.where(state['done'], jnp.zeros((3,), jnp.float32), rewards)
        dones = jnp.full((3,), next_state['done'], jnp.bool_)
        info = {'done_all': next_state['done'], 'terminated': next_state['terminated'], 'truncated': next_state['truncated']}
        return self.observe(next_state), next_state, rewards, dones, info
