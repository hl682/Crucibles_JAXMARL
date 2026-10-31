"""Glacier Foundry Settlements. Original blind source; no execution claimed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 384
    num_actions = 16
    observation_size = 4144
    max_steps = 896

    def refresh(self, s):
        p = s['pos']
        av = (jnp.max(jnp.abs(p[:, None] - p[None, :]), axis=2) <= 3)
        av = av & s['alive'][:, None] & s['alive'][None, :]
        cv = (jnp.max(jnp.abs(p[:, None] - s['crane_pos'][None, :]), axis=2) <= 3) & s['alive'][:, None]
        bv = (jnp.max(jnp.abs(p[:, None] - s['brake_pos'][None, :]), axis=2) <= 3) & s['alive'][:, None]
        lv = (jnp.max(jnp.abs(p[:, None] - s['load_pos'][None, :]), axis=2) <= 3) & s['alive'][:, None]
        x = jnp.arange(48, dtype=jnp.int32)[None, None, :]
        y = jnp.arange(32, dtype=jnp.int32)[None, :, None]
        mv = (jnp.abs(x - p[:, 0, None, None]) <= 3) & (jnp.abs(y - p[:, 1, None, None]) <= 3) & s['alive'][:, None, None]
        tiles = jnp.where(s['support'] > 0, 4, s['terrain'])
        tiles = jnp.where(s['terrain'] == 5, 5, tiles)
        tiles = jnp.where((s['water'] >= 3.0) & (s['support'] == 0), 7, tiles)
        tiles = jnp.where(s['marks'], 8, tiles).astype(jnp.int32)
        return dict(s, agent_vis=av, crane_vis=cv, brake_vis=bv, load_vis=lv, map_vis=mv, tiles=tiles)

    def reset(self, key):
        a = jnp.arange(384, dtype=jnp.int32)
        c = jnp.arange(24, dtype=jnp.int32)
        base = jnp.stack(((c % 6) * 8, (c // 6) * 8), axis=1)
        pos = base[a // 16] + jnp.stack((1 + a % 4, 1 + (a % 16) // 4), axis=1)
        pos = pos.at[c * 16].set(base + jnp.array([3, 4], jnp.int32))
        pos = pos.at[c * 16 + 2].set(base + jnp.array([5, 4], jnp.int32))
        cp = base + jnp.array([5, 4], jnp.int32)
        bp = base + jnp.array([3, 4], jnp.int32)
        terrain = jnp.zeros((32, 48), jnp.int32)
        terrain = jnp.where((jnp.arange(32)[:, None] % 8) == 5, 3, terrain).astype(jnp.int32)
        terrain = terrain.at[base[:, 1] + 6, base[:, 0] + 1].set(1)
        terrain = terrain.at[base[:, 1] + 6, base[:, 0] + 2].set(2)
        z = jnp.zeros((384,), jnp.bool_)
        s = {'t': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
             'pos': pos, 'alive': jnp.ones((384,), jnp.bool_), 'team': a // 16, 'tool': a % 4,
             'stone': (a % 4 == 2).astype(jnp.int32), 'fuel': (a % 4 == 1).astype(jnp.int32),
             'health': jnp.full((384,), 100, jnp.int32), 'marker': a % 4 == 3,
             'crane_pos': cp, 'brake_pos': bp, 'load_pos': cp + jnp.array([0, 1], jnp.int32),
             'site_active': jnp.ones((24,), jnp.bool_), 'phase': jnp.zeros((24,), jnp.int32),
             'progress': jnp.zeros((24,), jnp.int32), 'brake': jnp.zeros((24,), jnp.bool_), 'workshop': jnp.zeros((24,), jnp.int32),
             'drop': jnp.zeros((24,), jnp.bool_), 'delivery': jnp.zeros((24,), jnp.bool_),
             'terrain': terrain, 'water': jnp.zeros((32, 48), jnp.float32), 'heat': jnp.zeros((32, 48), jnp.float32),
             'support': jnp.zeros((32, 48), jnp.int32), 'marks': jnp.zeros((32, 48), jnp.bool_),
             'water_owner': jnp.full((32, 48), -1, jnp.int32),
             'weather_phase': jax.random.randint(key, (), 0, 32, dtype=jnp.int32),
             'built': z, 'heated': z, 'injury': z,
             'agent_vis': jnp.zeros((384, 384), jnp.bool_), 'crane_vis': jnp.zeros((384, 24), jnp.bool_),
             'brake_vis': jnp.zeros((384, 24), jnp.bool_), 'load_vis': jnp.zeros((384, 24), jnp.bool_),
             'map_vis': jnp.zeros((384, 32, 48), jnp.bool_), 'tiles': terrain}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        p = s['pos']
        d = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = p[:, None] + d[None, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 48) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 32)
        x = jnp.clip(q[:, :, 0], 0, 47)
        y = jnp.clip(q[:, :, 1], 0, 31)
        walk = inside & (s['terrain'][y, x] != 1) & (s['terrain'][y, x] != 5) & ((s['water'][y, x] < 3) | (s['support'][y, x] > 0))
        cell = s['terrain'][p[:, 1], p[:, 0]]
        nearstone = jnp.any((s['terrain'][y, x] == 1) & inside, axis=1)
        nearfuel = jnp.any((s['terrain'][y, x] == 2) & inside, axis=1)
        nearcrane = jnp.any(jnp.all(p[:, None] == s['crane_pos'][None, :], axis=2) & (s['phase'][None, :] == 0), axis=1)
        atbrake = jnp.any(jnp.all(p[:, None] == s['brake_pos'][None, :], axis=2), axis=1)
        nearrubble = jnp.any((s['terrain'][y, x] == 5) & inside, axis=1)
        rest = jnp.stack((jnp.ones((384,), jnp.bool_), nearstone & (s['tool'] == 0) & (s['stone'] < 3),
             nearfuel & (s['tool'] == 1) & (s['fuel'] < 3), (s['stone'] > 0) & (s['tool'] == 2),
             atbrake, nearcrane & (s['stone'] > 0) & (s['tool'] == 2), (s['fuel'] > 0) & (s['tool'] == 1),
             (s['stone'] > 0) & (s['tool'] == 2), (s['tool'] == 3) & (s['marker'] | s['marks'][p[:, 1], p[:, 0]]),
             nearrubble & (s['tool'] == 0), (s['stone'] > 0) | (s['fuel'] > 0)), axis=1)
        mask = jnp.concatenate((walk, rest), axis=1) & s['alive'][:, None] & ~s['episode_done']
        return mask.at[:, 0].set(True)

    def winner(self, target, wanted, count, t):
        rank = (jnp.arange(384, dtype=jnp.int32) - t) % 384
        costs = jnp.where(wanted[:, None] & (target[:, None] == jnp.arange(count)[None, :]), rank[:, None], 385)
        best = jnp.min(costs, axis=0)
        return wanted & (rank == best[target])

    def advance(self, s, actions):
        idx = jnp.arange(384, dtype=jnp.int32)
        clipped = jnp.clip(actions, 0, 15)
        valid = (actions == clipped) & self.legal_actions(s)[idx, clipped]
        act = jnp.where(valid, clipped, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['pos'] + delta[jnp.minimum(act, 4)] * (act <= 4)[:, None]
        cell = p[:, 1] * 48 + p[:, 0]
        neighbors = p[:, None] + delta[None, :]
        nx = jnp.clip(neighbors[:, :, 0], 0, 47)
        ny = jnp.clip(neighbors[:, :, 1], 0, 31)
        tile = s['terrain'][ny, nx]
        want = jnp.where(act == 6, 1, jnp.where(act == 7, 2, 5))
        choice = jnp.argmax(tile == want[:, None], axis=1)
        target = ny[idx, choice] * 48 + nx[idx, choice]
        gather = self.winner(target, (act == 6) | (act == 7) | (act == 14), 1536, s['t'])
        terrain = s['terrain'].reshape(-1).at[target].add(-jnp.where(gather, want, 0))
        terrain = terrain.reshape((32, 48))
        stone = s['stone'] + (gather & (act == 6)).astype(jnp.int32)
        fuel = s['fuel'] + (gather & (act == 7)).astype(jnp.int32)
        build = self.winner(cell, (act == 8) | (act == 12), 1536, s['t'])
        stone = stone - build.astype(jnp.int32)
        supports = s['support'].reshape(-1).at[cell].add(jnp.where(build & (act == 8), 12, 0)).reshape((32, 48))
        shelter = jnp.zeros((1536,), jnp.int32).at[cell].add((build & (act == 12)).astype(jnp.int32)).reshape((32, 48))
        terrain = jnp.where(shelter > 0, 6, terrain)
        heated = self.winner(cell, act == 11, 1536, s['t'])
        fuel = fuel - heated.astype(jnp.int32)
        heatadd = jnp.zeros((1536,), jnp.float32).at[cell].add(heated.astype(jnp.float32) * 4).reshape((32, 48))
        heat = jnp.clip(s['heat'] * 0.82 + heatadd, 0, 8)
        ownadd = jnp.zeros((1536,), jnp.int32).at[cell].add(jnp.where(heated, idx + 1, 0)).reshape((32, 48))
        owner = jnp.where(ownadd > 0, ownadd - 1, s['water_owner'])
        melt = (terrain == 3) & (heat > 0.7)
        incoming = jnp.concatenate((jnp.zeros((1, 48), jnp.float32), s['water'][:-1] * 0.35), axis=0)
        upstream = jnp.concatenate((jnp.full((1, 48), -1, jnp.int32), s['water_owner'][:-1]), axis=0)
        owner = jnp.where((incoming > s['water'] * 0.45) & (heatadd == 0) & (incoming > 0.01), upstream, owner)
        water = jnp.clip(s['water'] * 0.55 + incoming + melt.astype(jnp.float32) * heat * 0.7, 0, 6)
        supports = jnp.clip(supports - (water >= 2).astype(jnp.int32), 0, 24)
        markswitch = self.winner(cell, act == 13, 1536, s['t'])
        marked = s['marks'][p[:, 1], p[:, 0]]
        mdelta = jnp.where(markswitch, jnp.where(s['marker'], 1, -1), 0)
        marks = (s['marks'].astype(jnp.int32).reshape(-1).at[cell].add(mdelta)).reshape((32, 48)) > 0
        marker = jnp.where(markswitch, ~s['marker'] & marked, s['marker'])
        cdist = jnp.sum(jnp.abs(p[:, None] - s['crane_pos'][None, :]), axis=2)
        ci = jnp.argmin(cdist, axis=1)
        attach = self.winner(ci, act == 10, 24, s['t'])
        attached = jnp.zeros((24,), jnp.int32).at[ci].add(attach.astype(jnp.int32)) > 0
        stone = stone - attach.astype(jnp.int32)
        brake = jnp.any((act[:, None] == 9) & jnp.all(p[:, None] == s['brake_pos'][None, :], axis=2), axis=0)
        suspended = (s['phase'] == 1) | attached
        drop = suspended & ~brake
        progress = jnp.where(suspended & brake, s['progress'] + 1, s['progress'])
        delivery = suspended & brake & (progress >= 3)
        phase = jnp.where(drop, 3, jnp.where(delivery, 2, jnp.where(attached, 1, s['phase']))).astype(jnp.int32)
        lp = s['crane_pos'] + jnp.stack((jnp.minimum(progress, 2), jnp.ones((24,), jnp.int32)), axis=1)
        rubble = jnp.zeros((32, 48), jnp.int32).at[lp[:, 1], lp[:, 0]].add(drop.astype(jnp.int32))
        terrain = jnp.where(rubble > 0, 5, terrain)
        bridgeadd = jnp.zeros((32, 48), jnp.int32).at[lp[:, 1], lp[:, 0]].add(delivery.astype(jnp.int32) * 24)
        supports = jnp.minimum(supports + bridgeadd, 24)
        workshop = jnp.minimum(s['workshop'] + ((phase == 2) & (heat[s['crane_pos'][:, 1], s['crane_pos'][:, 0]] > 0.7)).astype(jnp.int32), 8)
        warmed = (workshop == 8) & (s['workshop'] < 8)
        warmtiles = jnp.zeros((32, 48), jnp.int32).at[s['crane_pos'][:, 1], s['crane_pos'][:, 0]].add(warmed.astype(jnp.int32))
        terrain = jnp.where(warmtiles > 0, 6, terrain)
        dist = jnp.max(jnp.abs(p[:, None] - p[None, :]), axis=2)
        can = (act[:, None] == 15) & (act[None, :] == 0) & s['alive'][None, :] & (dist <= 1) & (idx[:, None] != idx[None, :])
        can = can & (s['stone'][None, :] < 3) & (s['fuel'][None, :] < 3)
        dest = jnp.argmin(jnp.where(can, idx[None, :], 385), axis=1)
        offer = self.winner(dest, jnp.any(can, axis=1), 384, s['t'])
        give_s = offer & (stone > 0)
        give_f = offer & ~give_s & (fuel > 0)
        stone = stone - give_s.astype(jnp.int32) + jnp.zeros((384,), jnp.int32).at[dest].add(give_s.astype(jnp.int32))
        fuel = fuel - give_f.astype(jnp.int32) + jnp.zeros((384,), jnp.int32).at[dest].add(give_f.astype(jnp.int32))
        flood = (water[p[:, 1], p[:, 0]] >= 3) & (supports[p[:, 1], p[:, 0]] == 0)
        cold = ((s['t'] + s['weather_phase']) % 32 == 31) & (terrain[p[:, 1], p[:, 0]] != 6) & (heat[p[:, 1], p[:, 0]] < 0.3)
        damage = s['alive'] & (flood | cold)
        health = jnp.maximum(s['health'] - damage.astype(jnp.int32) * 2, 0)
        health = jnp.minimum(health + ((act == 5) & (terrain[p[:, 1], p[:, 0]] == 6) & s['alive']).astype(jnp.int32), 100)
        alive = health > 0
        blamed = owner[p[:, 1], p[:, 0]]
        penalties = jnp.zeros((384,), jnp.float32).at[jnp.maximum(blamed, 0)].add((flood & s['alive'] & (blamed >= 0)).astype(jnp.float32) * 0.001)
        reward = 0.5 * delivery[s['team']] + 0.2 * warmed[s['team']] + 0.025 * build - 0.003 * (act >= 6) - 0.001 * (stone + fuel) - 0.04 * damage - penalties
        reward = jnp.clip(reward + 0.01 * jnp.mean(delivery.astype(jnp.float32)), -1, 2).astype(jnp.float32)
        t = s['t'] + jnp.array(1, jnp.int32)
        terminated = ~jnp.any(alive) | jnp.all(workshop == 8)
        truncated = (t >= 896) & ~terminated
        n = dict(s, t=t, episode_done=terminated | truncated, terminated=terminated, truncated=truncated,
                 pos=p, alive=alive, stone=stone, fuel=fuel, health=health, marker=marker,
                 terrain=terrain, heat=heat, water=water, support=supports, marks=marks, water_owner=owner,
                 phase=phase, progress=progress, load_pos=lp, brake=brake, drop=drop, delivery=delivery, workshop=workshop,
                 built=build, heated=heated, injury=damage)
        return self.refresh(n), reward

    def raw_step(self, key, state, actions):
        n, reward = self.advance(state, actions)
        n = jax.tree.map(lambda old, new: jnp.where(state['episode_done'], old, new), state, n)
        reward = jnp.where(state['episode_done'], jnp.zeros((384,), jnp.float32), reward)
        dones = jnp.full((384,), n['episode_done'], jnp.bool_)
        info = {'done_all': n['episode_done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, dones, info

    def observe(self, s):
        p = s['pos']
        own = jnp.stack((jnp.full((384,), s['t'] / 896), jnp.arange(384) / 383, s['team'] / 23, s['tool'] / 3,
                         p[:, 0] / 47, p[:, 1] / 31, s['stone'] / 3, s['fuel'] / 3, s['health'] / 100,
                         s['marker'].astype(jnp.float32), s['alive'].astype(jnp.float32)), axis=1)
        actors = jnp.stack((p[:, 0] / 47, p[:, 1] / 31, s['team'] / 23, s['tool'] / 3, s['stone'] / 3,
                            s['fuel'] / 3, s['health'] / 100, s['marker'].astype(jnp.float32), s['alive'].astype(jnp.float32)), axis=1)
        av = (actors[None, :, :] * s['agent_vis'][:, :, None]).reshape((384, 3456))
        cranes = jnp.stack((s['crane_pos'][:, 0] / 47, s['crane_pos'][:, 1] / 31, s['phase'] / 3, s['progress'] / 3, s['brake'].astype(jnp.float32), s['workshop'] / 8, jnp.ones((24,))), axis=1)
        brakes = jnp.stack((s['brake_pos'][:, 0] / 47, s['brake_pos'][:, 1] / 31, s['brake'].astype(jnp.float32), jnp.ones((24,))), axis=1)
        loads = jnp.stack((s['load_pos'][:, 0] / 47, s['load_pos'][:, 1] / 31, s['phase'] / 3, s['progress'] / 3, s['drop'].astype(jnp.float32), s['delivery'].astype(jnp.float32), jnp.ones((24,))), axis=1)
        off = jnp.arange(49, dtype=jnp.int32)
        q = p[:, None, :] + jnp.stack((off % 7 - 3, off // 7 - 3), axis=1)[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 48) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 32) & s['alive'][:, None]
        x = jnp.clip(q[:, :, 0], 0, 47)
        y = jnp.clip(q[:, :, 1], 0, 31)
        patch = jnp.stack((s['tiles'][y, x] / 8, s['water'][y, x] / 6, s['heat'][y, x] / 8, s['support'][y, x] / 24, inside.astype(jnp.float32)), axis=2) * inside[:, :, None]
        return jnp.concatenate((own, av, (cranes[None] * s['crane_vis'][:, :, None]).reshape((384, 168)),
                    (brakes[None] * s['brake_vis'][:, :, None]).reshape((384, 96)),
                    (loads[None] * s['load_vis'][:, :, None]).reshape((384, 168)), patch.reshape((384, 245))), axis=1).astype(jnp.float32)
