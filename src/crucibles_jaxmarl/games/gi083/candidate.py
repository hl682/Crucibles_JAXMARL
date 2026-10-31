"""Relay Bastions: original four-player energy-tag objective tactics."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 112
    num_actions = 10
    max_steps = 96

    def reset(self, key):
        grid = jnp.zeros((7, 9), jnp.int32)
        grid = grid.at[jnp.array([2, 4]), 4].set(1)
        grid = grid.at[jnp.array([1, 5, 1, 5]), jnp.array([3, 3, 5, 5])].set(2)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'grid': grid, 'pos': jnp.array([[1, 2], [1, 4], [7, 2], [7, 4]], jnp.int32),
            'agent_active': jnp.ones(4, jnp.bool_), 'agent_vis': jnp.ones((4, 4), jnp.bool_),
            'team': jnp.array([0, 0, 1, 1], jnp.int32),
            'hp': jnp.full(4, 3, jnp.int32), 'stun': jnp.zeros(4, jnp.int32),
            'shield': jnp.zeros(4, jnp.int32),
            'node_pos': jnp.array([[4, 1], [4, 3], [4, 5]], jnp.int32),
            'node_active': jnp.ones(3, jnp.bool_), 'node_vis': jnp.ones((4, 3), jnp.bool_),
            'meter': jnp.zeros(3, jnp.int32), 'owner': jnp.full(3, -1, jnp.int32),
            'battery_pos': jnp.array([[0, 3], [8, 3]], jnp.int32),
            'battery_active': jnp.ones(2, jnp.bool_), 'battery_vis': jnp.ones((4, 2), jnp.bool_),
            'energy': jnp.full(2, 6, jnp.int32), 'score': jnp.zeros(2, jnp.int32),
            'ev_pulse': jnp.zeros(4, jnp.bool_), 'ev_tag': jnp.zeros(4, jnp.bool_),
            'ev_support': jnp.zeros(4, jnp.bool_), 'ev_capture': jnp.zeros(3, jnp.bool_),
            'ev_points': jnp.zeros(2, jnp.int32),
        }
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = s['pos'][:, None, :] + delta[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 9) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 7)
        tile = s['grid'][jnp.clip(q[:, :, 1], 0, 6), jnp.clip(q[:, :, 0], 0, 8)]
        moves = inside & (tile != 1)
        mask = jnp.concatenate([moves, jnp.ones((4, 5), jnp.bool_)], axis=1)
        return (mask & (s['stun'][:, None] == 0) & ~s['done']) | (jnp.arange(10)[None, :] == 0)

    def observe(self, s):
        own = jnp.concatenate([jax.nn.one_hot(s['team'], 2), jax.nn.one_hot(jnp.arange(4) % 2, 2)], axis=1)
        public = jnp.concatenate([s['t'].reshape(1), s['score'], s['energy']])
        af = jnp.concatenate([s['pos'], s['hp'][:, None], s['stun'][:, None], s['shield'][:, None], s['team'][:, None]], axis=1)
        nf = jnp.concatenate([s['node_pos'], s['meter'][:, None], s['owner'][:, None]], axis=1)
        all_world = jnp.concatenate([public, af.reshape(24), nf.reshape(12), s['battery_pos'].reshape(4), s['grid'].reshape(63)])
        return jnp.concatenate([own, jnp.broadcast_to(all_world, (4, 108))], axis=1).astype(jnp.float32)

    def advance(self, s, actions):
        n = dict(s)
        safe = jnp.clip(actions, 0, 9)
        legal = self.legal_actions(s)[jnp.arange(4), safe]
        a = jnp.where((actions >= 0) & (actions < 10) & legal, safe, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        q = s['pos'] + delta[a]
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=-1) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=-1), axis=1) > 1
        n['pos'] = jnp.where((~occupied & ~conflict)[:, None], q, s['pos'])
        n['t'] = s['t'] + 1
        n['stun'] = jnp.maximum(0, s['stun'] - 1)
        n['shield'] = jnp.maximum(0, s['shield'] - 1)
        ready = s['stun'] == 0
        recover = (a == 9) & ready & (jnp.sum(jnp.abs(n['pos'] - s['battery_pos'][s['team']]), axis=1) <= 1)
        gain = jnp.sum(recover[:, None].astype(jnp.int32) * jax.nn.one_hot(s['team'], 2, dtype=jnp.int32), axis=0) * 2
        n['energy'] = jnp.minimum(8, s['energy'] + gain)
        n['ev_pulse'] = jnp.zeros(4, jnp.bool_)
        n['ev_support'] = jnp.zeros(4, jnp.bool_)
        # Energy is shared. Within-team first slot has priority if only one cell remains.
        for i in range(4):
            teammate = jnp.array([1, 0, 3, 2], jnp.int32)[i]
            close = jnp.sum(jnp.abs(n['pos'][i] - n['pos'][teammate])) <= 2
            support = (a[i] == 8) & ready[i] & close & (n['energy'][s['team'][i]] > 0)
            n['shield'] = n['shield'].at[teammate].set(jnp.where(support, 2, n['shield'][teammate]))
            n['energy'] = n['energy'].at[s['team'][i]].add(-support.astype(jnp.int32))
            pulse = (a[i] == 6) & ready[i] & (n['energy'][s['team'][i]] > 0)
            n['energy'] = n['energy'].at[s['team'][i]].add(-pulse.astype(jnp.int32))
            n['ev_pulse'] = n['ev_pulse'].at[i].set(pulse)
            n['ev_support'] = n['ev_support'].at[i].set(support)
        hits = jnp.zeros(4, jnp.int32)
        for i in range(4):
            diff = n['pos'] - n['pos'][i]
            distance = jnp.sum(jnp.abs(diff), axis=1)
            cardinal = (diff[:, 0] == 0) | (diff[:, 1] == 0)
            ray = n['pos'][i][None, None, :] + jnp.sign(diff)[:, None, :] * jnp.arange(1, 4)[None, :, None]
            blocks = s['grid'][jnp.clip(ray[:, :, 1], 0, 6), jnp.clip(ray[:, :, 0], 0, 8)] == 1
            blocked = jnp.any(blocks & (jnp.arange(1, 4)[None, :] < distance[:, None]), axis=1)
            target_ok = (s['team'] != s['team'][i]) & (s['stun'] == 0) & cardinal & (distance <= 3) & ~blocked
            target = jnp.argmin(jnp.where(target_ok, distance, 99))
            fired = n['ev_pulse'][i] & jnp.any(target_ok)
            hits = hits.at[target].add(fired.astype(jnp.int32))
        cover = (s['grid'][n['pos'][:, 1], n['pos'][:, 0]] == 2).astype(jnp.int32)
        protection = cover + ((a == 7) & ready).astype(jnp.int32) + (n['shield'] > 0).astype(jnp.int32)
        damage = jnp.maximum(0, hits - protection)
        tagged = ready & (s['hp'] - damage <= 0)
        n['hp'] = jnp.where(tagged, 3, jnp.maximum(1, s['hp'] - damage))
        n['stun'] = jnp.where(tagged, 3, n['stun'])
        n['ev_tag'] = tagged
        near = jnp.sum(jnp.abs(n['pos'][:, None, :] - s['node_pos'][None, :, :]), axis=-1) <= 1
        effective = ready & ~tagged
        present0 = jnp.any(near & (effective & (s['team'] == 0))[:, None], axis=0)
        present1 = jnp.any(near & (effective & (s['team'] == 1))[:, None], axis=0)
        chan0 = jnp.sum((near & (effective & (s['team'] == 0) & (a == 5))[:, None]).astype(jnp.int32), axis=0)
        chan1 = jnp.sum((near & (effective & (s['team'] == 1) & (a == 5))[:, None]).astype(jnp.int32), axis=0)
        change = jnp.where(~present1, chan0, 0) - jnp.where(~present0, chan1, 0)
        n['meter'] = jnp.clip(s['meter'] + change, -3, 3)
        n['owner'] = jnp.where(n['meter'] == 3, 0, jnp.where(n['meter'] == -3, 1, -1)).astype(jnp.int32)
        n['ev_capture'] = (n['owner'] >= 0) & (n['owner'] != s['owner'])
        weights = jnp.array([1, 2, 1], jnp.int32)
        tick = n['t'] % 4 == 0
        points0 = jnp.sum(jnp.where((n['owner'] == 0) & ~present1 & tick, weights, 0))
        points1 = jnp.sum(jnp.where((n['owner'] == 1) & ~present0 & tick, weights, 0))
        n['ev_points'] = jnp.array([points0, points1], jnp.int32)
        n['score'] = s['score'] + n['ev_points']
        n['terminated'] = jnp.any(n['score'] >= 12)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        reward0 = (points0 - points1).astype(jnp.float32)
        reward = jnp.where(s['team'] == 0, reward0, -reward0)
        return n, reward

    def raw_step(self, key, state, actions):
        n, rewards = self.advance(state, actions)
        n = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), n, state)
        rewards = jnp.where(state['done'], jnp.zeros(4, jnp.float32), rewards)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, rewards, jnp.full(4, n['done'], jnp.bool_), info
