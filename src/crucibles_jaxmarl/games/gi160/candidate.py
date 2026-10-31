"""Railguard: two teams, shared moving cover and simultaneous ray fire."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    num_actions = 16
    observation_size = 218
    max_steps = 128

    def reset(self, key):
        tile = jnp.zeros((7, 11), jnp.int32)
        tile = tile.at[0, :].set(1).at[6, :].set(1).at[:, 0].set(1).at[:, 10].set(1)
        tile = tile.at[1, 5].set(1).at[5, 5].set(1)
        s = {'t': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False),
             'terminated': jnp.array(False), 'truncated': jnp.array(False), 'tile': tile,
             'pos': jnp.array([[3, 3], [4, 4], [8, 3], [8, 5]], jnp.int32),
             'agent_active': jnp.ones(4, jnp.bool_), 'agent_vis': jnp.ones((4, 4), jnp.bool_),
             'team': jnp.array([0, 0, 1, 1], jnp.int32), 'health': jnp.full(4, 2, jnp.int32),
             'energy': jnp.full(4, 6, jnp.int32), 'ammo': jnp.full(4, 2, jnp.int32),
             'cooldown': jnp.zeros(4, jnp.int32), 'last_action': jnp.zeros(4, jnp.int32),
             'hit': jnp.zeros(4, jnp.int32), 'revived': jnp.zeros(4, jnp.bool_),
             'cover_pos': jnp.array([[4, 3], [7, 5]], jnp.int32),
             'cover_active': jnp.ones(2, jnp.bool_), 'cover_vis': jnp.ones((4, 2), jnp.bool_),
             'cover_moved': jnp.zeros(2, jnp.bool_), 'cover_votes': jnp.zeros((2, 4), jnp.int32),
             'base_pos': jnp.array([[1, 3], [9, 3]], jnp.int32), 'base_active': jnp.ones(2, jnp.bool_),
             'base_vis': jnp.ones((4, 2), jnp.bool_), 'stock': jnp.full(2, 6, jnp.int32),
             'score': jnp.zeros(2, jnp.int32), 'winner': jnp.array(-1, jnp.int32),
             'trace_pos': jnp.zeros((24, 2), jnp.int32), 'trace_active': jnp.zeros(24, jnp.bool_),
             'trace_vis': jnp.zeros((4, 24), jnp.bool_), 'trace_owner': jnp.repeat(jnp.arange(4, dtype=jnp.int32), 6)}
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(4)
        near = jnp.any(jnp.sum(jnp.abs(s['pos'][:, None, :] - s['cover_pos'][None, :, :]), axis=2) == 1, axis=1)
        m = jnp.ones((4, 16), jnp.bool_)
        m = m.at[:, 1:5].set(jnp.broadcast_to((s['energy'] >= 1)[:, None], (4, 4)))
        m = m.at[:, 5:9].set(jnp.broadcast_to((near & (s['energy'] >= 1))[:, None], (4, 4)))
        fire = (s['energy'] >= 1) & (s['ammo'] > 0) & (s['cooldown'] == 0)
        m = m.at[:, 9:13].set(jnp.broadcast_to(fire[:, None], (4, 4)))
        at_base = jnp.sum(jnp.abs(s['pos'] - s['base_pos'][s['team']]), axis=1) <= 1
        m = m.at[:, 13].set(at_base & (s['stock'][s['team']] > 0) & (s['ammo'] < 3))
        m = m.at[:, 14].set((s['ammo'] > 0) & (s['energy'] >= 1))
        m = m.at[:, 15].set((s['ammo'] > 0) & (s['energy'] >= 2))
        return (m & (s['health'] > 0)[:, None] & ~s['episode_done']).at[:, 0].set(True)

    def raw_step(self, key, state, actions):
        s = dict(state)
        ids = jnp.arange(4)
        ac = jnp.clip(actions, 0, 15)
        ac = jnp.where((actions >= 0) & (actions < 16) & self.legal_actions(state)[ids, ac], ac, 0)
        s['last_action'] = ac
        s['hit'] = jnp.zeros(4, jnp.int32)
        s['revived'] = jnp.zeros(4, jnp.bool_)
        s['cover_votes'] = jnp.zeros((2, 4), jnp.int32)
        s['cooldown'] = jnp.maximum(0, s['cooldown'] - 1)
        s['energy'] = jnp.minimum(8, s['energy'] + 2 * ((ac == 0) & (state['health'] > 0)).astype(jnp.int32))
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['cover_pos'][None, :, :]), axis=2)
        selected = jnp.argmin(distance, axis=1)
        pushing = (ac >= 5) & (ac <= 8)
        s['energy'] = s['energy'] - pushing.astype(jnp.int32)
        proposed = s['cover_pos']
        can_shift = jnp.zeros(2, jnp.bool_)
        for c in range(2):
            participants = pushing & (selected == c) & (distance[:, c] == 1)
            votes = jnp.sum((participants[:, None] & (ac[:, None] == jnp.arange(5, 9)[None, :])).astype(jnp.int32), axis=0)
            d = jnp.argmax(votes)
            consensus = (votes[d] >= 2) & (jnp.sum((votes == votes[d]).astype(jnp.int32)) == 1)
            q = jnp.clip(s['cover_pos'][c] + directions[d + 1], jnp.array([0, 0]), jnp.array([10, 6]))
            clear = (s['tile'][q[1], q[0]] == 0) & ~jnp.any(jnp.all(s['pos'] == q, axis=1))
            clear = clear & ~jnp.any(jnp.all(s['cover_pos'] == q, axis=1))
            proposed = proposed.at[c].set(jnp.where(consensus & clear, q, s['cover_pos'][c]))
            can_shift = can_shift.at[c].set(consensus & clear)
            s['cover_votes'] = s['cover_votes'].at[c].set(votes)
        conflict = jnp.all(proposed[0] == proposed[1])
        s['cover_moved'] = can_shift & ~conflict
        s['cover_pos'] = jnp.where(s['cover_moved'][:, None], proposed, s['cover_pos'])
        q = jnp.clip(s['pos'] + directions[jnp.where(ac <= 4, ac, 0)], jnp.array([0, 0]), jnp.array([10, 6]))
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        collide = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        cover = jnp.any(jnp.all(q[:, None, :] == s['cover_pos'][None, :, :], axis=2), axis=1)
        moving = (ac >= 1) & (ac <= 4) & ~occupied & ~collide & ~cover & (s['tile'][q[:, 1], q[:, 0]] == 0)
        s['pos'] = jnp.where(moving[:, None], q, s['pos'])
        s['energy'] = s['energy'] - moving.astype(jnp.int32)
        for k in range(4):
            i = (state['t'] + k) % 4
            mate = i ^ 1
            reload = (ac[i] == 13) & (s['stock'][s['team'][i]] > 0) & (s['ammo'][i] < 3)
            s['stock'] = s['stock'].at[s['team'][i]].add(-reload.astype(jnp.int32))
            s['ammo'] = s['ammo'].at[i].add(reload.astype(jnp.int32))
            give = (ac[i] == 14) & (jnp.sum(jnp.abs(s['pos'][i] - s['pos'][mate])) == 1) & (s['ammo'][mate] < 3) & (s['health'][mate] > 0)
            s['ammo'] = s['ammo'].at[i].add(-give.astype(jnp.int32)).at[mate].add(give.astype(jnp.int32))
            s['energy'] = s['energy'].at[i].add(-give.astype(jnp.int32))
        firing = (ac >= 9) & (ac <= 12)
        s['ammo'] = s['ammo'] - firing.astype(jnp.int32)
        s['energy'] = s['energy'] - firing.astype(jnp.int32)
        s['cooldown'] = jnp.where(firing, 2, s['cooldown'])
        for i in range(4):
            ray = firing[i]
            direction = directions[jnp.clip(ac[i] - 8, 1, 4)]
            for r in range(6):
                cell = jnp.clip(s['pos'][i] + (r + 1) * direction, jnp.array([0, 0]), jnp.array([10, 6]))
                wall = s['tile'][cell[1], cell[0]] == 1
                shield = jnp.any(jnp.all(s['cover_pos'] == cell, axis=1))
                bodies = jnp.all(s['pos'] == cell, axis=1) & (ids != i)
                victim = bodies & (s['team'] != s['team'][i]) & (state['health'] > 0) & ray & ~wall & ~shield
                s['hit'] = s['hit'] + victim.astype(jnp.int32)
                s['trace_pos'] = s['trace_pos'].at[i * 6 + r].set(cell)
                s['trace_active'] = s['trace_active'].at[i * 6 + r].set(ray)
                ray = ray & ~wall & ~shield & ~jnp.any(bodies)
        s['health'] = jnp.maximum(0, s['health'] - s['hit'])
        down = (state['health'] > 0) & (s['health'] == 0)
        for i in range(4):
            mate = i ^ 1
            revive = (ac[i] == 15) & (s['health'][i] > 0) & (state['health'][mate] == 0) & (jnp.sum(jnp.abs(s['pos'][i] - s['pos'][mate])) == 1)
            s['health'] = s['health'].at[mate].set(jnp.where(revive, 1, s['health'][mate]))
            s['ammo'] = s['ammo'].at[i].add(-revive.astype(jnp.int32))
            s['energy'] = s['energy'].at[i].add(-2 * revive.astype(jnp.int32))
            s['revived'] = s['revived'].at[mate].set(revive)
        lost = jnp.array([jnp.sum(down[:2].astype(jnp.int32)), jnp.sum(down[2:].astype(jnp.int32))], jnp.int32)
        s['score'] = s['score'] + lost[::-1]
        eliminated = jnp.array([jnp.all(s['health'][:2] == 0), jnp.all(s['health'][2:] == 0)])
        s['winner'] = jnp.where(eliminated[1] & ~eliminated[0], 0, jnp.where(eliminated[0] & ~eliminated[1], 1, -1)).astype(jnp.int32)
        team_reward = 0.5 * (lost[::-1] - lost).astype(jnp.float32)
        team_reward = team_reward + jnp.where(s['winner'] >= 0, jnp.where(jnp.arange(2) == s['winner'], 1.0, -1.0), 0.0)
        reward = team_reward[s['team']].astype(jnp.float32)
        s['trace_vis'] = jnp.broadcast_to(s['trace_active'][None, :], (4, 24))
        s['t'] = state['t'] + 1
        s['terminated'] = jnp.any(eliminated)
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['episode_done'] = s['terminated'] | s['truncated']
        s = jax.tree.map(lambda new, old: jnp.where(state['episode_done'], old, new), s, state)
        reward = jnp.where(state['episode_done'], jnp.zeros(4, jnp.float32), reward)
        return self.observe(s), s, reward, jnp.full(4, s['episode_done']), {'done_all': s['episode_done'], 'terminated': s['terminated'], 'truncated': s['truncated']}

    def observe(self, s):
        common = jnp.concatenate((s['tile'].reshape(-1), s['pos'].reshape(-1), s['team'], s['health'], s['energy'],
            s['ammo'], s['cooldown'], s['last_action'], s['hit'], s['revived'], s['cover_pos'].reshape(-1),
            s['cover_moved'], s['cover_votes'].reshape(-1), s['base_pos'].reshape(-1), s['stock'], s['score'],
            s['trace_pos'].reshape(-1), s['trace_active'], jnp.array([s['t'], s['episode_done'], s['winner']], jnp.int32))).astype(jnp.float32)
        return jnp.concatenate((jnp.eye(4, dtype=jnp.float32), jnp.broadcast_to(common, (4, common.shape[0]))), axis=1)
