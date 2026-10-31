"""Lantern rescue and moving shadows: original unexecuted spatial rules."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 194
    num_actions = 10
    max_steps = 128

    def sight(self, sources, targets, tiles):
        # Finite 17-sample rounded cell ray, excluding the two endpoint cells.
        delta = targets[None, :, None, :] - sources[:, None, None, :]
        ray = jnp.round(sources[:, None, None, :] + delta *
                        jnp.linspace(0.0, 1.0, 17)[None, None, :, None]).astype(jnp.int32)
        interior = jnp.any(ray != sources[:, None, None, :], axis=-1) & jnp.any(
            ray != targets[None, :, None, :], axis=-1)
        blocked = (tiles[ray[..., 1], ray[..., 0]] == 1) & interior
        return ~jnp.any(blocked, axis=-1)

    def lighting(self, s):
        cells = jnp.stack((jnp.arange(63) % 9, jnp.arange(63) // 9), axis=1)
        lp = s['lamp_pos']
        bp = s['beacon_pos']
        lamp_rays = self.sight(lp, cells, s['tiles'])
        beacon_rays = self.sight(bp, cells, s['tiles'])
        lamp_near = jnp.sum(jnp.abs(lp[:, None, :] - cells[None, :, :]), axis=-1) <= 3
        beacon_near = jnp.sum(jnp.abs(bp[:, None, :] - cells[None, :, :]), axis=-1) <= 3
        lit = jnp.any(lamp_rays & lamp_near & (s['lamp_on'] & (s['lamp_charge'] > 0))[:, None], axis=0)
        lit = lit | jnp.any(beacon_rays & beacon_near & (s['beacon_ttl'] > 0)[:, None], axis=0)
        lit = lit | (jnp.sum(jnp.abs(cells - jnp.array([1, 3], jnp.int32)), axis=1) <= 1)
        return lit.reshape(7, 9) & (s['tiles'] != 1)

    def refresh(self, s):
        n = dict(s)
        n['lamp_pos'] = jnp.where((s['lamp_owner'] >= 0)[:, None],
                                  s['agent_pos'][jnp.maximum(s['lamp_owner'], 0)], s['lamp_pos'])
        n['evac_pos'] = jnp.where((s['evac_carrier'] >= 0)[:, None],
                                  s['agent_pos'][jnp.maximum(s['evac_carrier'], 0)], s['evac_pos'])
        n['agent_lamp'] = jnp.max(jnp.where(s['lamp_owner'][None, :] == jnp.arange(3)[:, None],
                                           jnp.arange(2)[None, :], -1), axis=1).astype(jnp.int32)
        n['agent_evac'] = jnp.max(jnp.where(s['evac_carrier'][None, :] == jnp.arange(3)[:, None],
                                           jnp.arange(2)[None, :], -1), axis=1).astype(jnp.int32)
        n['agent_active'] = s['health'] > 0
        n['evac_active'] = ~s['evac_rescued']
        n['illumination'] = self.lighting(n)
        n['render_tiles'] = jnp.where((s['tiles'] == 0) & n['illumination'], 4, s['tiles']).astype(jnp.int32)
        cells = jnp.stack((jnp.arange(63) % 9, jnp.arange(63) // 9), axis=1)
        distance = jnp.sum(jnp.abs(n['agent_pos'][:, None, :] - cells[None, :, :]), axis=-1)
        visible = self.sight(n['agent_pos'], cells, n['tiles']) & (
            (distance <= 1) | ((distance <= 4) & n['illumination'].reshape(1, 63)))
        visible = visible & n['agent_active'][:, None]
        n['map_visible'] = visible.reshape(3, 7, 9)
        n['agent_visible'] = visible[:, n['agent_pos'][:, 1] * 9 + n['agent_pos'][:, 0]] & n['agent_active'][None, :]
        n['lamp_visible'] = visible[:, n['lamp_pos'][:, 1] * 9 + n['lamp_pos'][:, 0]]
        n['evac_visible'] = visible[:, n['evac_pos'][:, 1] * 9 + n['evac_pos'][:, 0]] & n['evac_active'][None, :]
        n['beacon_visible'] = visible[:, n['beacon_pos'][:, 1] * 9 + n['beacon_pos'][:, 0]]
        n['depot_visible'] = visible[:, n['depot_pos'][:, 1] * 9 + n['depot_pos'][:, 0]]
        n['hazard_visible'] = visible[:, n['hazard_pos'][:, 1] * 9 + n['hazard_pos'][:, 0]] & n['illumination'][
            n['hazard_pos'][:, 1], n['hazard_pos'][:, 0]][None, :]
        return n

    def reset(self, key):
        x = jnp.arange(9)[None, :]
        y = jnp.arange(7)[:, None]
        walls = (x == 0) | (x == 8) | (y == 0) | (y == 6) | ((x == 4) & ((y == 2) | (y == 4)))
        tiles = jnp.where(walls, 1, 0).astype(jnp.int32).at[3, 1].set(2).at[5, 1].set(3)
        phase = jax.random.randint(key, (), 0, 6, dtype=jnp.int32)
        path = jnp.array([[5, 2], [5, 3], [5, 4], [6, 4], [6, 3], [6, 2]], jnp.int32)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False),
             'terminated': jnp.array(False), 'truncated': jnp.array(False), 'tiles': tiles,
             'phase': phase, 'agent_pos': jnp.array([[2, 3], [1, 3], [1, 4]], jnp.int32),
             'agent_active': jnp.ones(3, jnp.bool_), 'health': jnp.full(3, 3, jnp.int32),
             'agent_lamp': jnp.array([0, -1, -1], jnp.int32), 'agent_evac': jnp.full(3, -1, jnp.int32),
             'agent_visible': jnp.zeros((3, 3), jnp.bool_),
             'lamp_pos': jnp.array([[2, 3], [3, 5]], jnp.int32), 'lamp_active': jnp.ones(2, jnp.bool_),
             'lamp_owner': jnp.array([0, -1], jnp.int32), 'lamp_charge': jnp.array([48, 30], jnp.int32),
             'lamp_on': jnp.ones(2, jnp.bool_), 'lamp_visible': jnp.zeros((3, 2), jnp.bool_),
             'evac_pos': jnp.array([[7, 2], [7, 4]], jnp.int32), 'evac_active': jnp.ones(2, jnp.bool_),
             'evac_carrier': jnp.full(2, -1, jnp.int32), 'evac_rescued': jnp.zeros(2, jnp.bool_),
             'evac_visible': jnp.zeros((3, 2), jnp.bool_),
             'beacon_pos': jnp.array([[3, 1], [5, 5]], jnp.int32), 'beacon_active': jnp.ones(2, jnp.bool_),
             'beacon_ttl': jnp.zeros(2, jnp.int32), 'beacon_visible': jnp.zeros((3, 2), jnp.bool_),
             'depot_pos': jnp.array([[1, 5]], jnp.int32), 'depot_active': jnp.ones(1, jnp.bool_),
             'depot_stock': jnp.array([16], jnp.int32), 'depot_visible': jnp.zeros((3, 1), jnp.bool_),
             'hazard_pos': path[(phase + jnp.array([0, 3], jnp.int32)) % 6],
             'hazard_active': jnp.ones(2, jnp.bool_), 'hazard_visible': jnp.zeros((3, 2), jnp.bool_),
             'map_visible': jnp.zeros((3, 7, 9), jnp.bool_), 'illumination': jnp.zeros((7, 9), jnp.bool_),
             'render_tiles': tiles,
             'rescued_count': jnp.array(0, jnp.int32), 'rescue_flash': jnp.array(0, jnp.int32)}
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        self_fields = jnp.concatenate((jnp.eye(3, dtype=jnp.float32), s['agent_pos'] / 8.0,
            s['health'][:, None] / 3.0, (s['agent_lamp'][:, None] + 1) / 2.0,
            (s['agent_evac'][:, None] + 1) / 2.0,
            jnp.full((3, 1), s['t'] / 128.0), jnp.full((3, 1), s['rescued_count'] / 2.0)), axis=1)
        tile = jnp.where(s['map_visible'], s['tiles'][None, :, :] / 3.0, -1.0).reshape(3, 63)
        light = jnp.where(s['map_visible'], s['illumination'][None, :, :], -1.0).reshape(3, 63)
        af = jnp.concatenate((s['agent_pos'] / 8.0, s['health'][:, None] / 3.0,
            (s['agent_lamp'][:, None] + 1) / 2.0, (s['agent_evac'][:, None] + 1) / 2.0,
            s['agent_active'][:, None]), axis=1)
        lf = jnp.concatenate((s['lamp_pos'] / 8.0, s['lamp_charge'][:, None] / 64.0,
            (s['lamp_owner'][:, None] + 1) / 3.0, s['lamp_on'][:, None], jnp.ones((2, 1))), axis=1)
        ef = jnp.concatenate((s['evac_pos'] / 8.0, s['evac_rescued'][:, None],
            (s['evac_carrier'][:, None] + 1) / 3.0, jnp.ones((2, 1))), axis=1)
        bf = jnp.concatenate((s['beacon_pos'] / 8.0, s['beacon_ttl'][:, None] / 12.0, jnp.ones((2, 1))), axis=1)
        df = jnp.concatenate((s['depot_pos'] / 8.0, s['depot_stock'][:, None] / 16.0, jnp.ones((1, 1))), axis=1)
        hf = jnp.concatenate((s['hazard_pos'] / 8.0, jnp.ones((2, 1))), axis=1)
        return jnp.concatenate((self_fields, tile, light,
            jnp.where(s['agent_visible'][:, :, None], af[None, :, :], -1.0).reshape(3, 18),
            jnp.where(s['lamp_visible'][:, :, None], lf[None, :, :], -1.0).reshape(3, 12),
            jnp.where(s['evac_visible'][:, :, None], ef[None, :, :], -1.0).reshape(3, 10),
            jnp.where(s['beacon_visible'][:, :, None], bf[None, :, :], -1.0).reshape(3, 8),
            jnp.where(s['depot_visible'][:, :, None], df[None, :, :], -1.0).reshape(3, 4),
            jnp.where(s['hazard_visible'][:, :, None], hf[None, :, :], -1.0).reshape(3, 6)), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        # Commands are attempts: failed physical prerequisites produce no effect.
        return (jnp.arange(10)[None, :] == 0) | ((~s['done'] & s['agent_active'])[:, None])

    def raw_step(self, key, s, actions):
        a = jnp.asarray(actions, jnp.int32)
        a = jnp.where((a >= 0) & (a < 10) & s['agent_active'] & ~s['done'], a, 0)
        n = dict(s)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        target = jnp.clip(s['agent_pos'] + delta[a], jnp.array([0, 0]), jnp.array([8, 6]))
        can_walk = (s['tiles'][target[:, 1], target[:, 0]] != 1) & ((s['agent_evac'] < 0) |
            s['illumination'][target[:, 1], target[:, 0]]) & s['agent_active']
        n['agent_pos'] = jnp.where(can_walk[:, None], target, s['agent_pos'])
        n = self.refresh(n)
        # Rotating agent order resolves competition for the same physical object.
        for offset in range(3):
            i = (s['t'] + offset) % 3
            lp = jnp.sum(jnp.abs(n['lamp_pos'] - n['agent_pos'][i]), axis=1)
            free_lamp = (lp <= 1) & (n['lamp_owner'] < 0)
            li = jnp.argmax(free_lamp).astype(jnp.int32)
            held_lamp = n['agent_lamp'][i]
            drop_lamp = (a[i] == 5) & (held_lamp >= 0)
            pick_lamp = (a[i] == 5) & (i != 1) & (held_lamp < 0) & (n['agent_evac'][i] < 0) & jnp.any(free_lamp)
            n['lamp_owner'] = jnp.where(drop_lamp & (jnp.arange(2) == held_lamp), -1, n['lamp_owner'])
            n['lamp_owner'] = jnp.where(pick_lamp & (jnp.arange(2) == li), i, n['lamp_owner'])
            ep = jnp.sum(jnp.abs(n['evac_pos'] - n['agent_pos'][i]), axis=1)
            free_evac = (ep <= 1) & (n['evac_carrier'] < 0) & ~n['evac_rescued']
            ei = jnp.argmax(free_evac).astype(jnp.int32)
            held_evac = n['agent_evac'][i]
            drop_evac = (a[i] == 6) & (held_evac >= 0)
            pick_evac = (a[i] == 6) & (i != 0) & (held_evac < 0) & (held_lamp < 0) & jnp.any(free_evac)
            n['evac_carrier'] = jnp.where(drop_evac & (jnp.arange(2) == held_evac), -1, n['evac_carrier'])
            n['evac_carrier'] = jnp.where(pick_evac & (jnp.arange(2) == ei), i, n['evac_carrier'])
            toggle = (a[i] == 7) & (held_lamp >= 0)
            n['lamp_on'] = n['lamp_on'] ^ (toggle & (jnp.arange(2) == held_lamp))
            safe_li = jnp.maximum(held_lamp, 0)
            near_depot = jnp.sum(jnp.abs(n['agent_pos'][i] - n['depot_pos'][0])) <= 1
            amount = jnp.where((a[i] == 8) & (held_lamp >= 0) & near_depot,
                               jnp.minimum(8, jnp.minimum(n['depot_stock'][0], 64 - n['lamp_charge'][safe_li])), 0)
            n['lamp_charge'] = n['lamp_charge'].at[safe_li].add(amount)
            n['depot_stock'] = n['depot_stock'] - amount
            near_beacon = jnp.sum(jnp.abs(n['beacon_pos'] - n['agent_pos'][i]), axis=1) <= 1
            bi = jnp.argmax(near_beacon).astype(jnp.int32)
            pulse = (a[i] == 9) & (held_lamp >= 0) & n['lamp_on'][safe_li] & (n['lamp_charge'][safe_li] >= 4) & jnp.any(near_beacon)
            n['beacon_ttl'] = n['beacon_ttl'].at[bi].set(jnp.where(pulse, 13, n['beacon_ttl'][bi]))
            n['lamp_charge'] = n['lamp_charge'].at[safe_li].add(jnp.where(pulse, -4, 0))
            n = self.refresh(n)
        n['t'] = s['t'] + jnp.array(1, jnp.int32)
        n['lamp_charge'] = jnp.maximum(0, n['lamp_charge'] - n['lamp_on'].astype(jnp.int32))
        n['beacon_ttl'] = jnp.maximum(0, n['beacon_ttl'] - 1)
        path = jnp.array([[5, 2], [5, 3], [5, 4], [6, 4], [6, 3], [6, 2]], jnp.int32)
        n['hazard_pos'] = path[(n['phase'] + n['t'] // 3 + jnp.array([0, 3], jnp.int32)) % 6]
        n = self.refresh(n)
        touched = jnp.any(jnp.all(n['agent_pos'][:, None, :] == n['hazard_pos'][None, :, :], axis=2), axis=1)
        hit = touched & ~n['illumination'][n['agent_pos'][:, 1], n['agent_pos'][:, 0]] & n['agent_active']
        n['health'] = jnp.maximum(0, n['health'] - hit.astype(jnp.int32))
        n['lamp_owner'] = jnp.where((n['lamp_owner'] >= 0) & (n['health'][jnp.maximum(n['lamp_owner'], 0)] == 0), -1, n['lamp_owner'])
        n['evac_carrier'] = jnp.where((n['evac_carrier'] >= 0) & (n['health'][jnp.maximum(n['evac_carrier'], 0)] == 0), -1, n['evac_carrier'])
        rescued_now = jnp.all(n['evac_pos'] == jnp.array([1, 3], jnp.int32), axis=1) & ~n['evac_rescued']
        n['evac_rescued'] = n['evac_rescued'] | rescued_now
        n['evac_carrier'] = jnp.where(n['evac_rescued'], -1, n['evac_carrier'])
        n['rescue_flash'] = jnp.sum(rescued_now.astype(jnp.int32))
        n['rescued_count'] = jnp.sum(n['evac_rescued'].astype(jnp.int32))
        n = self.refresh(n)
        n['terminated'] = jnp.all(n['evac_rescued']) | ~jnp.any(n['agent_active'])
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        reward = jnp.full(3, 5.0 * n['rescue_flash'] - 0.25 * jnp.sum(hit.astype(jnp.float32)) - 0.01, jnp.float32)
        # Absorbing raw terminal; never reset or advance an already finished clock.
        n = {k: jnp.where(s['done'], s[k], n[k]) for k in s}
        reward = jnp.where(s['done'], jnp.zeros(3, jnp.float32), reward)
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), {
            'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
