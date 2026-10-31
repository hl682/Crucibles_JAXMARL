"""Cipher Causeway: original asymmetric pursuit, rescue and extraction."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 154
    num_actions = 10
    max_steps = 96

    def refresh(self, s):
        n = dict(s)
        radii = jnp.array([2, 3, 3, 3], jnp.int32)
        teams = jnp.array([0, 0, 1, 1], jnp.int32)
        distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=-1)
        relay_range = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=-1) <= 3
        relay = jnp.any((teams[:, None, None] == teams[None, :, None]) & (s['flare_ttl'][None, :, None] > 0) & relay_range[None, :, :], axis=1)
        device_range = jnp.sum(jnp.abs(s['device_pos'][:, None, :] - s['pos'][None, :, :]), axis=-1)
        hidden = jnp.any((device_range <= 1) & (s['device_ttl'][:, None] > 0) & (teams[:, None] == 0), axis=0)
        scanned = jnp.any((device_range <= 4) & (s['device_ttl'][:, None] > 0) & (teams[:, None] == 1), axis=0)
        visible = (distance <= radii[:, None]) | relay | (s['flare_ttl'][None, :] > 0)
        visible = visible | ((teams[:, None] == 1) & scanned[None, :])
        obscure = (teams[:, None] == 1) & (teams[None, :] == 0) & hidden[None, :] & ~scanned[None, :] & (distance > 1) & (s['flare_ttl'][None, :] == 0)
        n['agent_vis'] = visible & ~obscure
        gd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['gate_pos'][None, :, :]), axis=-1)
        n['gate_vis'] = gd <= radii[:, None]
        occupied = jnp.any(jnp.all(s['pos'][:, None, :] == s['gate_pos'][None, :, :], axis=-1), axis=0)
        n['gate_open'] = (s['gate_timer'] > 0) | occupied
        n['device_active'] = s['device_ttl'] > 0
        n['device_vis'] = jnp.broadcast_to(n['device_active'][None, :], (4, 4))
        n['flare_active'] = s['flare_ttl'] > 0
        n['flare_pos'] = jnp.where(n['flare_active'][:, None], s['pos'], s['flare_pos'])
        n['flare_vis'] = jnp.broadcast_to(n['flare_active'][None, :], (4, 4))
        c = jnp.maximum(s['cipher_carrier'][0], 0)
        n['carrying'] = (jnp.arange(4) == s['cipher_carrier'][0]).astype(jnp.int32)
        cd = jnp.sum(jnp.abs(s['pos'] - s['cipher_pos'][0]), axis=1)
        cv = jnp.where(s['cipher_carrier'][0] >= 0, n['agent_vis'][:, c], cd <= radii)
        n['cipher_active'] = ~s['extracted'].reshape(1)
        n['cipher_vis'] = cv[:, None] & n['cipher_active'][None, :]
        cache_d = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['cache_pos'][None, :, :]), axis=-1)
        n['cache_vis'] = (cache_d <= radii[:, None]) & s['cache_active'][None, :]
        return n

    def reset(self, key):
        grid = jnp.zeros((7, 9), jnp.int32)
        grid = grid.at[:, 4].set(1)
        grid = grid.at[jnp.array([2, 4]), 4].set(2)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'grid': grid, 'pos': jnp.array([[6, 1], [6, 5], [2, 1], [2, 5]], jnp.int32),
            'agent_active': jnp.ones(4, jnp.bool_), 'agent_vis': jnp.ones((4, 4), jnp.bool_),
            'captured': jnp.zeros(4, jnp.bool_), 'carrying': jnp.zeros(4, jnp.int32),
            'energy': jnp.full(2, 4, jnp.int32),
            'cipher_pos': jnp.array([[7, 3]], jnp.int32), 'cipher_carrier': jnp.full(1, -1, jnp.int32),
            'cipher_active': jnp.ones(1, jnp.bool_), 'cipher_vis': jnp.zeros((4, 1), jnp.bool_),
            'extracted': jnp.array(False),
            'gate_pos': jnp.array([[4, 2], [4, 4]], jnp.int32),
            'gate_timer': jnp.full(2, 3, jnp.int32), 'gate_open': jnp.ones(2, jnp.bool_),
            'gate_active': jnp.ones(2, jnp.bool_), 'gate_vis': jnp.zeros((4, 2), jnp.bool_),
            'device_pos': jnp.array([[6, 1], [6, 5], [2, 1], [2, 5]], jnp.int32),
            'device_ttl': jnp.zeros(4, jnp.int32), 'device_type': jnp.array([0, 0, 1, 1], jnp.int32),
            'device_active': jnp.zeros(4, jnp.bool_), 'device_vis': jnp.zeros((4, 4), jnp.bool_),
            'flare_pos': jnp.array([[6, 1], [6, 5], [2, 1], [2, 5]], jnp.int32),
            'flare_ttl': jnp.zeros(4, jnp.int32), 'flare_active': jnp.zeros(4, jnp.bool_),
            'flare_vis': jnp.zeros((4, 4), jnp.bool_),
            'cache_pos': jnp.array([[7, 0], [1, 6]], jnp.int32), 'cache_value': jnp.full(2, 2, jnp.int32),
            'cache_active': jnp.ones(2, jnp.bool_), 'cache_vis': jnp.zeros((4, 2), jnp.bool_),
            'exit_pos': jnp.array([[1, 3]], jnp.int32), 'exit_active': jnp.ones(1, jnp.bool_),
            'exit_vis': jnp.ones((4, 1), jnp.bool_),
            'ev_capture': jnp.zeros(4, jnp.bool_), 'ev_free': jnp.zeros(4, jnp.bool_),
            'ev_gate': jnp.zeros(2, jnp.bool_), 'ev_pickup': jnp.array(False),
            'ev_extract': jnp.array(False), 'ev_cache': jnp.zeros(2, jnp.bool_),
            'ev_device': jnp.zeros(4, jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = s['pos'][:, None, :] + delta[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 9) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 7)
        tile = s['grid'][jnp.clip(q[:, :, 1], 0, 6), jnp.clip(q[:, :, 0], 0, 8)]
        gate_match = jnp.all(q[:, :, None, :] == s['gate_pos'][None, None, :, :], axis=-1)
        blocked_gate = jnp.any(gate_match & ~s['gate_open'][None, None, :], axis=-1)
        moves = inside & (tile != 1) & ~blocked_gate
        mask = jnp.concatenate([moves, jnp.ones((4, 5), jnp.bool_)], axis=1)
        return (mask & ~s['captured'][:, None] & ~s['done']) | (jnp.arange(10)[None, :] == 0)

    def observe(self, s):
        teams = jnp.array([0, 0, 1, 1], jnp.int32)
        own = jnp.concatenate([jnp.eye(4, dtype=jnp.float32), s['pos'], s['captured'][:, None], s['carrying'][:, None], s['energy'][teams, None]], axis=1)
        public = jnp.array([s['t'], s['extracted'], s['done']], jnp.float32)
        af = jnp.concatenate([s['pos'], s['captured'][:, None], s['carrying'][:, None], s['flare_ttl'][:, None]], axis=1)
        ao = jnp.concatenate([jnp.where(s['agent_vis'][:, :, None], af[None, :, :], -1).reshape(4, 20), s['agent_vis']], axis=1)
        cf = jnp.concatenate([s['cipher_pos'], s['cipher_carrier'][:, None], s['cipher_active'][:, None]], axis=1)
        co = jnp.concatenate([jnp.where(s['cipher_vis'][:, :, None], cf[None, :, :], -1).reshape(4, 4), s['cipher_vis']], axis=1)
        gf = jnp.concatenate([s['gate_pos'], s['gate_open'][:, None], s['gate_timer'][:, None]], axis=1)
        go = jnp.concatenate([jnp.where(s['gate_vis'][:, :, None], gf[None, :, :], -1).reshape(4, 8), s['gate_vis']], axis=1)
        df = jnp.concatenate([s['device_pos'], s['device_ttl'][:, None], s['device_type'][:, None]], axis=1)
        device = jnp.where(s['device_active'][:, None], df, -1).reshape(16)
        ff = jnp.concatenate([s['flare_pos'], s['flare_ttl'][:, None]], axis=1)
        flare = jnp.where(s['flare_active'][:, None], ff, -1).reshape(12)
        kf = jnp.concatenate([s['cache_pos'], s['cache_value'][:, None], s['cache_active'][:, None]], axis=1)
        ko = jnp.concatenate([jnp.where(s['cache_vis'][:, :, None], kf[None, :, :], -1).reshape(4, 8), s['cache_vis']], axis=1)
        final = jnp.concatenate([s['exit_pos'].reshape(2), s['grid'].reshape(63)])
        return jnp.concatenate([own, jnp.broadcast_to(public, (4, 3)), ao, co, go, jnp.broadcast_to(device, (4, 16)), jnp.broadcast_to(flare, (4, 12)), ko, jnp.broadcast_to(final, (4, 65))], axis=1).astype(jnp.float32)

    def advance(self, s, actions):
        n = dict(s)
        safe = jnp.clip(actions, 0, 9)
        legal = self.legal_actions(s)[jnp.arange(4), safe]
        a = jnp.where((actions >= 0) & (actions < 10) & legal, safe, 0)
        teams = jnp.array([0, 0, 1, 1], jnp.int32)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        q = s['pos'] + delta[a]
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=-1) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=-1), axis=1) > 1
        n['pos'] = jnp.where((~occupied & ~conflict)[:, None], q, s['pos'])
        n['t'] = s['t'] + 1
        n['gate_timer'] = jnp.maximum(0, s['gate_timer'] - 1)
        n['device_ttl'] = jnp.maximum(0, s['device_ttl'] - 1)
        n['flare_ttl'] = jnp.where(a == 7, 3, jnp.maximum(0, s['flare_ttl'] - 1))
        n['ev_device'] = jnp.zeros(4, jnp.bool_)
        n['ev_cache'] = jnp.zeros(2, jnp.bool_)
        n['ev_free'] = jnp.zeros(4, jnp.bool_)
        n['ev_capture'] = jnp.zeros(4, jnp.bool_)
        n['ev_pickup'] = jnp.array(False)
        # Opening/locking and device placement compete for each team's battery pool.
        for i in range(4):
            dist = jnp.sum(jnp.abs(n['pos'][i] - s['gate_pos']), axis=1)
            g = jnp.argmin(dist)
            cost = jnp.array([2, 1, 1, 1], jnp.int32)[i]
            control = (a[i] == 6) & (dist[g] <= 1) & (n['energy'][teams[i]] >= cost)
            value = jnp.where(teams[i] == 0, 5, 0)
            n['gate_timer'] = n['gate_timer'].at[g].set(jnp.where(control, value, n['gate_timer'][g]))
            n['energy'] = n['energy'].at[teams[i]].add(-cost * control.astype(jnp.int32))
            deploy = (a[i] == 8) & (n['energy'][teams[i]] > 0)
            n['device_pos'] = n['device_pos'].at[i].set(jnp.where(deploy, n['pos'][i], n['device_pos'][i]))
            n['device_ttl'] = n['device_ttl'].at[i].set(jnp.where(deploy, 4, n['device_ttl'][i]))
            n['energy'] = n['energy'].at[teams[i]].add(-deploy.astype(jnp.int32))
            n['ev_device'] = n['ev_device'].at[i].set(deploy)
        for i in range(2):
            near = jnp.sum(jnp.abs(n['pos'][i] - s['cipher_pos'][0])) <= 1
            pick = (a[i] == 5) & near & (n['cipher_carrier'][0] < 0) & ~s['captured'][i]
            n['cipher_carrier'] = n['cipher_carrier'].at[0].set(jnp.where(pick, i, n['cipher_carrier'][0]))
            n['ev_pickup'] = n['ev_pickup'] | pick
        c = jnp.maximum(n['cipher_carrier'][0], 0)
        n['cipher_pos'] = jnp.where(n['cipher_carrier'][0] >= 0, n['pos'][c][None, :], s['cipher_pos'])
        for i in range(2, 4):
            dist = jnp.sum(jnp.abs(n['pos'][:2] - n['pos'][i]), axis=1)
            eligible = (dist <= 1) & ~n['captured'][:2]
            priority = jnp.where(eligible, dist * 2 - (jnp.arange(2) == n['cipher_carrier'][0]).astype(jnp.int32), 99)
            target = jnp.argmin(priority)
            tag = (a[i] == 5) & jnp.any(eligible)
            n['captured'] = n['captured'].at[target].set(n['captured'][target] | tag)
            n['ev_capture'] = n['ev_capture'].at[target].set(n['ev_capture'][target] | tag)
        dropped = (n['cipher_carrier'][0] >= 0) & n['captured'][c]
        n['cipher_carrier'] = jnp.where(dropped, jnp.full(1, -1, jnp.int32), n['cipher_carrier'])
        for i in range(4):
            mate = jnp.array([1, 0, 3, 2], jnp.int32)[i]
            adjacent = jnp.sum(jnp.abs(n['pos'][i] - n['pos'][mate])) == 1
            free = (a[i] == 9) & (teams[i] == 0) & ~n['captured'][i] & n['captured'][mate] & adjacent
            n['captured'] = n['captured'].at[mate].set(n['captured'][mate] & ~free)
            n['ev_free'] = n['ev_free'].at[mate].set(n['ev_free'][mate] | free)
            near_cache = (jnp.sum(jnp.abs(s['cache_pos'] - n['pos'][i]), axis=1) <= 1) & n['cache_active']
            k = jnp.argmax(near_cache)
            take = (a[i] == 9) & ~free & ~n['captured'][i] & jnp.any(near_cache)
            n['cache_active'] = n['cache_active'].at[k].set(n['cache_active'][k] & ~take)
            n['energy'] = n['energy'].at[teams[i]].add(jnp.where(take, s['cache_value'][k], 0))
            n['ev_cache'] = n['ev_cache'].at[k].set(n['ev_cache'][k] | take)
        extraction = (n['cipher_carrier'][0] >= 0) & jnp.all(n['cipher_pos'][0] == s['exit_pos'][0])
        n['extracted'] = extraction
        n['ev_extract'] = extraction
        failed = jnp.all(n['captured'][:2])
        n['terminated'] = extraction | failed
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.refresh(n)
        n['ev_gate'] = n['gate_open'] != s['gate_open']
        outcome = extraction.astype(jnp.float32) - failed.astype(jnp.float32)
        return n, jnp.where(teams == 0, outcome, -outcome)

    def raw_step(self, key, state, actions):
        n, rewards = self.advance(state, actions)
        n = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), n, state)
        rewards = jnp.where(state['done'], jnp.zeros(4, jnp.float32), rewards)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, rewards, jnp.full(4, n['done'], jnp.bool_), info
