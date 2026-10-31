"""Thermal Loom: original spatial three-role, shared-grid production game."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 122
    num_actions = 10
    max_steps = 128

    def refresh(self, s):
        n = dict(s)
        held = s['owner'][None, :] == jnp.arange(3)[:, None]
        n['carrying'] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        n['item_active'] = (s['stage'] != 4) & (s['stage'] != 6)
        n['item_vis'] = jnp.broadcast_to(n['item_active'][None, :], (3, 4))
        return n

    def reset(self, key):
        grid = jnp.zeros((7, 9), jnp.int32)
        grid = grid.at[jnp.array([0, 1, 3, 5, 6]), 4].set(1)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'grid': grid, 'pos': jnp.array([[2, 2], [5, 1], [6, 4]], jnp.int32),
            'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
            'carrying': jnp.full(3, -1, jnp.int32),
            'item_pos': jnp.array([[1, 1], [2, 1], [1, 2], [2, 0]], jnp.int32),
            'item_active': jnp.ones(4, jnp.bool_), 'item_vis': jnp.ones((3, 4), jnp.bool_),
            'stage': jnp.zeros(4, jnp.int32), 'owner': jnp.full(4, -1, jnp.int32),
            'heat': jnp.zeros(4, jnp.int32),
            'station_pos': jnp.array([[3, 2], [5, 2], [5, 4], [7, 4], [1, 5]], jnp.int32),
            'station_active': jnp.ones(5, jnp.bool_), 'station_vis': jnp.ones((3, 5), jnp.bool_),
            'timer': jnp.zeros(5, jnp.int32), 'loaded': jnp.full(5, -1, jnp.int32),
            'charge': jnp.array(12, jnp.int32), 'delivered': jnp.array(0, jnp.int32),
            'spoiled': jnp.array(0, jnp.int32),
            'ev_load': jnp.zeros(4, jnp.bool_), 'ev_complete': jnp.zeros(4, jnp.bool_),
            'ev_spoil': jnp.zeros(4, jnp.bool_), 'ev_deliver': jnp.zeros(4, jnp.bool_),
            'ev_recycle': jnp.zeros(4, jnp.bool_), 'ev_transfer': jnp.zeros(4, jnp.bool_),
            'ev_power': jnp.zeros(5, jnp.bool_), 'ev_crank': jnp.zeros(3, jnp.bool_),
        }
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = s['pos'][:, None, :] + delta[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 9) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 7)
        tile = s['grid'][jnp.clip(q[:, :, 1], 0, 6), jnp.clip(q[:, :, 0], 0, 8)]
        mask = jnp.concatenate([inside & (tile != 1), jnp.ones((3, 5), jnp.bool_)], axis=1)
        return (mask & ~s['done']) | (jnp.arange(10)[None, :] == 0)

    def observe(self, s):
        own = jnp.concatenate([jnp.eye(3, dtype=jnp.float32), s['pos'], s['carrying'][:, None]], axis=1)
        public = jnp.array([s['t'], s['charge'], s['delivered'], s['spoiled']], jnp.float32)
        af = jnp.concatenate([s['pos'], s['carrying'][:, None]], axis=1)
        items = jnp.concatenate([s['item_pos'], s['stage'][:, None], s['owner'][:, None], s['heat'][:, None]], axis=1)
        stations = jnp.concatenate([s['station_pos'], s['timer'][:, None], s['loaded'][:, None]], axis=1)
        world = jnp.concatenate([public, af.reshape(9), items.reshape(20), stations.reshape(20), s['grid'].reshape(63)])
        return jnp.concatenate([own, jnp.broadcast_to(world, (3, 116))], axis=1).astype(jnp.float32)

    def advance(self, s, actions):
        n = dict(s)
        safe = jnp.clip(actions, 0, 9)
        legal = self.legal_actions(s)[jnp.arange(3), safe]
        a = jnp.where((actions >= 0) & (actions < 10) & legal, safe, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        q = s['pos'] + delta[a]
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=-1) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=-1), axis=1) > 1
        n['pos'] = jnp.where((~occupied & ~conflict)[:, None], q, s['pos'])
        n['t'] = s['t'] + 1
        n['item_pos'] = jnp.where((s['owner'] >= 0)[:, None], n['pos'][jnp.maximum(s['owner'], 0)], s['item_pos'])
        n['ev_load'] = jnp.zeros(4, jnp.bool_)
        n['ev_complete'] = jnp.zeros(4, jnp.bool_)
        n['ev_deliver'] = jnp.zeros(4, jnp.bool_)
        n['ev_recycle'] = jnp.zeros(4, jnp.bool_)
        n['ev_transfer'] = jnp.zeros(4, jnp.bool_)
        crank = (a == 8) & (jnp.sum(jnp.abs(n['pos'] - s['station_pos'][4]), axis=1) <= 1)
        n['charge'] = jnp.minimum(24, s['charge'] + 4 * jnp.sum(crank.astype(jnp.int32)))
        n['ev_crank'] = crank
        # Pickup/drop, loading and shipping are deterministic role-order transactions.
        for i in range(3):
            held = n['owner'] == i
            has = jnp.any(held)
            h = jnp.argmax(held)
            candidates = (n['owner'] == -1) & (n['stage'] != 4) & (n['stage'] != 6) & (jnp.sum(jnp.abs(n['item_pos'] - n['pos'][i]), axis=1) <= 1)
            k = jnp.argmax(candidates)
            pickup = (a[i] == 5) & ~has & jnp.any(candidates)
            ground_here = jnp.any((n['owner'] == -1) & (n['stage'] != 4) & (n['stage'] != 6) & jnp.all(n['item_pos'] == n['pos'][i], axis=1))
            machine_here = jnp.any((n['loaded'] >= 0) & jnp.all(n['station_pos'] == n['pos'][i], axis=1))
            drop = (a[i] == 5) & has & ~ground_here & ~machine_here
            n['owner'] = n['owner'].at[k].set(jnp.where(pickup, i, n['owner'][k]))
            n['owner'] = n['owner'].at[h].set(jnp.where(drop, -1, n['owner'][h]))
            n['item_pos'] = n['item_pos'].at[k].set(jnp.where(pickup, n['pos'][i], n['item_pos'][k]))
            n['item_pos'] = n['item_pos'].at[h].set(jnp.where(drop, n['pos'][i], n['item_pos'][h]))
            close = jnp.sum(jnp.abs(n['pos'][i] - s['station_pos'][i])) <= 1
            output_blocked = jnp.any((n['owner'] == -1) & (n['stage'] != 4) & (n['stage'] != 6) & jnp.all(n['item_pos'] == s['station_pos'][i], axis=1))
            load = (a[i] == 6) & has & close & (n['stage'][h] == i) & (n['loaded'][i] < 0) & ~output_blocked
            n['loaded'] = n['loaded'].at[i].set(jnp.where(load, h, n['loaded'][i]))
            duration = jnp.array([3, 4, 2], jnp.int32)[i]
            n['timer'] = n['timer'].at[i].set(jnp.where(load, duration, n['timer'][i]))
            n['owner'] = n['owner'].at[h].set(jnp.where(load, -2 - i, n['owner'][h]))
            n['item_pos'] = n['item_pos'].at[h].set(jnp.where(load, s['station_pos'][i], n['item_pos'][h]))
            n['ev_load'] = n['ev_load'].at[h].set(n['ev_load'][h] | load)
            dock = jnp.sum(jnp.abs(n['pos'][i] - s['station_pos'][3])) <= 1
            generator = jnp.sum(jnp.abs(n['pos'][i] - s['station_pos'][4])) <= 1
            ship = (a[i] == 7) & has & (n['stage'][h] == 3) & dock
            recycle = (a[i] == 7) & has & (n['stage'][h] == 5) & generator
            n['stage'] = n['stage'].at[h].set(jnp.where(ship, 4, jnp.where(recycle, 6, n['stage'][h])))
            n['owner'] = n['owner'].at[h].set(jnp.where(ship | recycle, -1, n['owner'][h]))
            n['ev_deliver'] = n['ev_deliver'].at[h].set(n['ev_deliver'][h] | ship)
            n['ev_recycle'] = n['ev_recycle'].at[h].set(n['ev_recycle'][h] | recycle)
            n['charge'] = jnp.minimum(24, n['charge'] + 3 * recycle.astype(jnp.int32))
        before = n['owner']
        for k in range(4):
            holder = jnp.maximum(before[k], 0)
            free = ~jnp.any(before[None, :] == jnp.arange(3)[:, None], axis=1)
            near = jnp.sum(jnp.abs(n['pos'] - n['pos'][holder]), axis=1) == 1
            willing = free & near & (a == 9)
            receiver = jnp.argmax(willing)
            handoff = (before[k] >= 0) & (a[holder] == 9) & jnp.any(willing) & ~jnp.any(n['owner'] == receiver)
            n['owner'] = n['owner'].at[k].set(jnp.where(handoff, receiver, n['owner'][k]))
            n['item_pos'] = n['item_pos'].at[k].set(jnp.where(handoff, n['pos'][receiver], n['item_pos'][k]))
            n['ev_transfer'] = n['ev_transfer'].at[k].set(handoff)
        # A two-channel grid serves up to two machines per tick, rotating priority.
        budget = jnp.array(2, jnp.int32)
        n['ev_power'] = jnp.zeros(5, jnp.bool_)
        for offset in range(3):
            machine = (s['t'] + offset) % 3
            k = jnp.maximum(n['loaded'][machine], 0)
            run = (n['loaded'][machine] >= 0) & (n['timer'][machine] > 0) & (n['charge'] > 0) & (budget > 0)
            n['timer'] = n['timer'].at[machine].add(-run.astype(jnp.int32))
            n['charge'] = n['charge'] - run.astype(jnp.int32)
            budget = budget - run.astype(jnp.int32)
            complete = run & (n['timer'][machine] == 0)
            n['stage'] = n['stage'].at[k].set(jnp.where(complete, machine + 1, n['stage'][k]))
            n['owner'] = n['owner'].at[k].set(jnp.where(complete, -1, n['owner'][k]))
            n['heat'] = n['heat'].at[k].set(jnp.where(complete & (machine == 1), 18, n['heat'][k]))
            n['ev_complete'] = n['ev_complete'].at[k].set(n['ev_complete'][k] | complete)
            n['ev_power'] = n['ev_power'].at[machine].set(run)
            n['loaded'] = n['loaded'].at[machine].set(jnp.where(complete, -1, n['loaded'][machine]))
        decay = (s['stage'] == 2) & (n['stage'] == 2)
        n['heat'] = jnp.maximum(0, n['heat'] - decay.astype(jnp.int32))
        spoil = (n['stage'] == 2) & (n['heat'] == 0)
        n['stage'] = jnp.where(spoil, 5, n['stage'])
        for machine in range(3):
            k = jnp.maximum(n['loaded'][machine], 0)
            failed = (n['loaded'][machine] >= 0) & spoil[k]
            n['owner'] = n['owner'].at[k].set(jnp.where(failed, -1, n['owner'][k]))
            n['timer'] = n['timer'].at[machine].set(jnp.where(failed, 0, n['timer'][machine]))
            n['loaded'] = n['loaded'].at[machine].set(jnp.where(failed, -1, n['loaded'][machine]))
        n['ev_spoil'] = spoil
        n['spoiled'] = s['spoiled'] + jnp.sum(spoil.astype(jnp.int32))
        n['delivered'] = s['delivered'] + jnp.sum(n['ev_deliver'].astype(jnp.int32))
        possible = n['delivered'] + jnp.sum((n['stage'] < 4).astype(jnp.int32))
        n['terminated'] = (n['delivered'] >= 3) | (possible < 3)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.refresh(n)
        reward = jnp.sum(n['ev_deliver'].astype(jnp.float32)) - jnp.sum(spoil.astype(jnp.float32))
        return n, jnp.full(3, reward, jnp.float32)

    def raw_step(self, key, state, actions):
        n, rewards = self.advance(state, actions)
        n = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), n, state)
        rewards = jnp.where(state['done'], jnp.zeros(3, jnp.float32), rewards)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, rewards, jnp.full(3, n['done'], jnp.bool_), info
