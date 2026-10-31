"""Shared Firebreak: original prospective source; never executed by author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 291
    num_actions = 15
    max_steps = 72

    def refresh(self, s):
        distance = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['cell_pos'][None, :, :]), axis=-1)
        visible = distance <= 3
        agents = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=-1) <= 3
        reservoir_cell = jnp.all(s['cell_pos'] == s['reservoir_pos'][0], axis=1)
        tile = jnp.where(s['fuel'] > 0, 1, 0)
        tile = jnp.where(s['cleared'], 3, tile)
        tile = jnp.where(s['fire'], 2, tile)
        tile = jnp.where(reservoir_cell, 4, tile)
        n = dict(s)
        n['cell_visible'] = visible
        n['map_visible'] = visible.reshape(3, 7, 9)
        n['agent_visible'] = agents
        n['tiles'] = tile.reshape(7, 9).astype(jnp.int32)
        n['zone_active'] = s['zone_health'] > 0
        return n

    def reset(self, key):
        cell = jnp.stack([jnp.arange(63, dtype=jnp.int32) % 9, jnp.arange(63, dtype=jnp.int32) // 9], axis=1)
        fuel = jnp.full((63,), 3, jnp.int32).at[58].set(0)
        fire = jnp.zeros((63,), jnp.bool_).at[jnp.array([31, 10])].set(True)
        s = {
            't': jnp.array(0, jnp.int32),
            'episode_done': jnp.array(False),
            'terminated': jnp.array(False),
            'truncated': jnp.array(False),
            'agent_pos': jnp.array([[3, 3], [5, 3], [4, 5]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_),
            'agent_visible': jnp.ones((3, 3), jnp.bool_),
            'water': jnp.full((3,), 2, jnp.int32),
            'labor': jnp.full((3,), 6, jnp.int32),
            'priority': jax.random.randint(key, (3,), 0, 3, dtype=jnp.int32),
            'cell_pos': cell,
            'cell_active': jnp.ones((63,), jnp.bool_),
            'cell_visible': jnp.ones((3, 63), jnp.bool_),
            'fuel': fuel,
            'fire': fire,
            'cleared': jnp.zeros((63,), jnp.bool_),
            'tiles': jnp.zeros((7, 9), jnp.int32),
            'map_visible': jnp.ones((3, 7, 9), jnp.bool_),
            'reservoir_pos': jnp.array([[4, 6]], jnp.int32),
            'reservoir_active': jnp.ones((1,), jnp.bool_),
            'reservoir_visible': jnp.ones((3, 1), jnp.bool_),
            'tank': jnp.array([18], jnp.int32),
            'zone_pos': jnp.array([[6, 3], [6, 1], [2, 5]], jnp.int32),
            'zone_active': jnp.ones((3,), jnp.bool_),
            'zone_visible': jnp.ones((3, 3), jnp.bool_),
            'zone_health': jnp.full((3,), 4, jnp.int32),
            'wind': jnp.array(0, jnp.int32),
            'extinguished': jnp.zeros((63,), jnp.bool_),
            'ignited': jnp.zeros((63,), jnp.bool_),
            'cleared_event': jnp.zeros((63,), jnp.bool_),
            'zone_damage': jnp.zeros((3,), jnp.bool_),
            'transferred': jnp.zeros((3, 3), jnp.int32),
            'refilled': jnp.zeros((3,), jnp.int32),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['agent_pos']
        q = p[:, None, :] + dirs[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] <= 8) & (q[:, :, 1] >= 0) & (q[:, :, 1] <= 6)
        index = jnp.clip(q[:, :, 1], 0, 6) * 9 + jnp.clip(q[:, :, 0], 0, 8)
        move = inside & (s['labor'][:, None] > 0)
        spray = inside & s['fire'][index] & (s['water'][:, None] > 0) & (s['labor'][:, None] > 0)
        own = p[:, 1] * 9 + p[:, 0]
        clear = (s['fuel'][own] > 0) & ~s['fire'][own] & (s['labor'] >= 2)
        refill = (jnp.sum(jnp.abs(p - s['reservoir_pos'][0]), axis=1) <= 1) & (s['tank'][0] > 0) & (s['water'] < 4) & (s['labor'] > 0)
        nearby = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=-1) <= 1
        give = nearby & ~jnp.eye(3, dtype=jnp.bool_) & (s['water'][:, None] > 0) & (s['water'][None, :] < 4) & (s['labor'][:, None] > 0)
        self_spray = s['fire'][own] & (s['water'] > 0) & (s['labor'] > 0)
        mask = jnp.concatenate([jnp.ones((3, 1), jnp.bool_), move, spray, clear[:, None], refill[:, None], give, self_spray[:, None]], axis=1)
        return mask & (~s['episode_done'] | (jnp.arange(15)[None, :] == 0))

    def observe(self, s):
        vis = s['agent_visible']
        local_agents = jnp.concatenate([
            jnp.where(vis[:, :, None], jnp.broadcast_to(s['agent_pos'][None, :, :], (3, 3, 2)), -1).reshape(3, 6),
            jnp.where(vis, s['water'][None, :], -1),
            jnp.where(vis, s['labor'][None, :], -1), vis.astype(jnp.int32),
        ], axis=1)
        own = jnp.concatenate([jnp.eye(3, dtype=jnp.int32), s['agent_pos'], s['water'][:, None], s['labor'][:, None], jax.nn.one_hot(s['priority'], 3, dtype=jnp.int32)], axis=1)
        public = jnp.concatenate([s['t'][None], s['wind'][None], s['tank'], s['reservoir_pos'].reshape(-1), s['zone_pos'].reshape(-1), s['zone_health']])
        cells = jnp.stack([s['cell_visible'].astype(jnp.int32),
            jnp.where(s['cell_visible'], s['fuel'][None, :], -1),
            jnp.where(s['cell_visible'], s['fire'][None, :].astype(jnp.int32), -1),
            jnp.where(s['cell_visible'], s['cleared'][None, :].astype(jnp.int32), -1)], axis=-1).reshape(3, 252)
        return jnp.concatenate([own, jnp.broadcast_to(public[None, :], (3, 14)), local_agents, cells], axis=1).astype(jnp.float32)

    def advance(self, s, actions):
        safe = jnp.clip(actions, 0, 14)
        valid = (safe == actions) & self.legal_actions(s)[jnp.arange(3), safe]
        a = jnp.where(valid, safe, 0)
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        moving = (a >= 1) & (a <= 4)
        p = s['agent_pos'] + jnp.where(moving[:, None], dirs[a], 0)
        spray = ((a >= 5) & (a <= 8)) | (a == 14)
        spray_pos = p + dirs[a]
        hit = jnp.all(spray_pos[:, None, :] == s['cell_pos'][None, :, :], axis=-1) & spray[:, None]
        extinguished = s['fire'] & jnp.any(hit, axis=0)
        clear_hit = jnp.all(p[:, None, :] == s['cell_pos'][None, :, :], axis=-1) & (a == 9)[:, None]
        cleared_event = jnp.any(clear_hit, axis=0)
        fuel = jnp.where(cleared_event, 0, s['fuel'])
        fire = s['fire'] & ~extinguished
        water = s['water'] - spray.astype(jnp.int32)
        tank = s['tank'][0]
        refilled = jnp.zeros((3,), jnp.int32)
        transferred = jnp.zeros((3, 3), jnp.int32)
        # Competition for a shared tank and transfers settle in public rotating role order.
        for offset in range(3):
            i = (s['t'] + offset) % 3
            refill = (a[i] == 10) & (jnp.sum(jnp.abs(p[i] - s['reservoir_pos'][0])) <= 1)
            amount = jnp.where(refill, jnp.minimum(2, jnp.minimum(4 - water[i], tank)), 0)
            water = water.at[i].add(amount)
            tank = tank - amount
            refilled = refilled.at[i].set(amount)
            target = jnp.clip(a[i] - 11, 0, 2)
            adjacent = jnp.sum(jnp.abs(p[i] - p[target])) <= 1
            gift = (a[i] >= 11) & (a[i] <= 13) & adjacent & (target != i) & (water[i] > 0) & (water[target] < 4)
            unit = gift.astype(jnp.int32)
            water = water.at[i].add(-unit).at[target].add(unit)
            transferred = transferred.at[i, target].set(unit)
        t = s['t'] + 1
        burn_tick = (t % 3) == 0
        fuel = jnp.maximum(fuel - (fire & burn_tick).astype(jnp.int32), 0)
        survivors = fire & (fuel > 0)
        wind_dirs = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], jnp.int32)
        wind = s['wind']
        source = s['cell_pos'] - wind_dirs[wind][None, :]
        inside = (source[:, 0] >= 0) & (source[:, 0] <= 8) & (source[:, 1] >= 0) & (source[:, 1] <= 6)
        index = jnp.clip(source[:, 1], 0, 6) * 9 + jnp.clip(source[:, 0], 0, 8)
        ignited = burn_tick & inside & survivors[index] & (fuel > 0) & ~survivors
        fire = survivors | ignited
        zi = s['zone_pos'][:, 1] * 9 + s['zone_pos'][:, 0]
        damage = fire[zi] & (s['zone_health'] > 0)
        health = s['zone_health'] - damage.astype(jnp.int32)
        quiet = ~jnp.any(fire)
        terminal = quiet | jnp.all(health == 0)
        weights = 1 + 3 * jax.nn.one_hot(s['priority'], 3, dtype=jnp.int32)
        reward = -jnp.sum(weights * damage[None, :].astype(jnp.int32), axis=1) - (a != 0).astype(jnp.int32)
        reward = reward + jnp.where(quiet, 2 * jnp.sum(weights * (health > 0)[None, :].astype(jnp.int32), axis=1), 0)
        n = dict(s)
        n['t'] = t
        n['agent_pos'] = p
        n['water'] = water
        n['labor'] = jnp.clip(s['labor'] - (a != 0).astype(jnp.int32) - (a == 9).astype(jnp.int32) + 2 * (a == 0).astype(jnp.int32), 0, 8)
        n['tank'] = jnp.minimum(tank + 2 * ((t % 6) == 0).astype(jnp.int32), 24)[None]
        n['fuel'] = fuel
        n['fire'] = fire
        n['cleared'] = s['cleared'] | cleared_event
        n['zone_health'] = health
        n['wind'] = (t // 6) % 4
        n['extinguished'] = extinguished
        n['ignited'] = ignited
        n['cleared_event'] = cleared_event
        n['zone_damage'] = damage
        n['transferred'] = transferred
        n['refilled'] = refilled
        n['terminated'] = terminal
        n['truncated'] = (t >= self.max_steps) & ~terminal
        n['episode_done'] = terminal | n['truncated']
        return self.refresh(n), reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['episode_done'], lambda _: (state, jnp.zeros((3,), jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['episode_done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((3,), n['episode_done'], jnp.bool_), info
