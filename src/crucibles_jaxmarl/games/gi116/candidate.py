"""Bridge builders with unequal tool kits: original unexecuted spatial rules."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 102
    num_actions = 12
    max_steps = 144

    def refresh(self, s):
        n = dict(s)
        held = (s['part_owner'] >= 0) & (s['part_owner'] < 3)
        deployed = s['part_owner'] >= 3
        n['part_pos'] = jnp.where(held[:, None], s['agent_pos'][jnp.clip(s['part_owner'], 0, 2)], s['part_pos'])
        n['part_pos'] = jnp.where(deployed[:, None], s['bridge_pos'][jnp.clip(s['part_owner'] - 3, 0, 3)], n['part_pos'])
        n['load'] = jnp.sum((s['part_owner'][None, :] == jnp.arange(3)[:, None]).astype(jnp.int32), axis=1)
        n['built'] = jnp.any(s['part_owner'][:, None] == (3 + jnp.arange(4))[None, :], axis=0)
        n['render_tiles'] = s['tiles'].at[s['bridge_pos'][:, 1], s['bridge_pos'][:, 0]].set(jnp.where(n['built'], 4, 2))
        n['part_active'] = ~deployed
        n['part_visible'] = jnp.broadcast_to(n['part_active'][None, :], (3, 4))
        n['capacity'] = jnp.where(s['support'] > 0, 2, 1).astype(jnp.int32)
        n['material_total'] = jnp.array([jnp.sum((s['part_owner'] >= -1).astype(jnp.int32))], jnp.int32)
        return n

    def reset(self, key):
        x = jnp.arange(9)[None, :]
        y = jnp.arange(5)[:, None]
        land = (y >= 1) & (y <= 3) & ((x <= 2) | (x == 4) | (x >= 6))
        tiles = jnp.where(land, jnp.where(x >= 6, 3, 0), 1).astype(jnp.int32)
        tiles = tiles.at[1, 3].set(2).at[1, 5].set(2).at[3, 3].set(2).at[3, 5].set(2)
        parts = jnp.array([[2, 1], [2, 3], [1, 2], [4, 1]], jnp.int32)
        parts = parts.at[3, 1].set(1 + 2 * jax.random.randint(key, (), 0, 2, dtype=jnp.int32))
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False),
             'terminated': jnp.array(False), 'truncated': jnp.array(False), 'tiles': tiles, 'render_tiles': tiles,
             'agent_pos': jnp.array([[2, 1], [1, 1], [1, 3]], jnp.int32),
             'agent_active': jnp.ones(3, jnp.bool_), 'agent_visible': jnp.ones((3, 3), jnp.bool_),
             'role': jnp.arange(3, dtype=jnp.int32), 'load': jnp.zeros(3, jnp.int32),
             'arrived': jnp.zeros(3, jnp.bool_),
             'part_pos': parts, 'part_owner': jnp.full(4, -1, jnp.int32),
             'part_active': jnp.ones(4, jnp.bool_), 'part_visible': jnp.ones((3, 4), jnp.bool_),
             'bridge_pos': jnp.array([[3, 1], [5, 1], [3, 3], [5, 3]], jnp.int32),
             'bridge_active': jnp.ones(4, jnp.bool_), 'bridge_visible': jnp.ones((3, 4), jnp.bool_),
             'built': jnp.zeros(4, jnp.bool_), 'life': jnp.zeros(4, jnp.int32),
             'support': jnp.zeros(4, jnp.int32), 'capacity': jnp.ones(4, jnp.int32),
             'material_total': jnp.array([4], jnp.int32),
             'collapse_count': jnp.array(0, jnp.int32), 'fall_count': jnp.array(0, jnp.int32)}
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        agents = jnp.concatenate((s['agent_pos'] / 8.0, s['load'][:, None] / 2.0,
            s['role'][:, None] / 2.0, s['arrived'][:, None]), axis=1)
        parts = jnp.concatenate((s['part_pos'] / 8.0, (s['part_owner'][:, None] + 1) / 7.0), axis=1)
        bridges = jnp.concatenate((s['bridge_pos'] / 8.0, s['built'][:, None], s['life'][:, None] / 12.0,
            s['support'][:, None] / 10.0, s['capacity'][:, None] / 2.0), axis=1)
        public = jnp.concatenate((jnp.array([s['t'] / 144.0]), (s['tiles'] / 3.0).reshape(45),
            agents.reshape(15), parts.reshape(12), bridges.reshape(24),
            jnp.array([s['collapse_count'] / 576.0, s['fall_count'] / 432.0])))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, 99))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        return (jnp.arange(12)[None, :] == 0) | jnp.broadcast_to(~s['done'], (3, 12))

    def raw_step(self, key, s, actions):
        a = jnp.asarray(actions, jnp.int32)
        a = jnp.where((a >= 0) & (a < 12) & ~s['done'], a, 0)
        n = dict(s)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        target = jnp.clip(s['agent_pos'] + delta[a], jnp.array([0, 0]), jnp.array([8, 4]))
        # Land permits co-location. Each bridge tile has a genuine total mass limit.
        for offset in range(3):
            i = (s['t'] + offset) % 3
            at_bridge = jnp.all(s['bridge_pos'] == target[i], axis=1)
            bi = jnp.argmax(at_bridge).astype(jnp.int32)
            occupants = jnp.all(n['agent_pos'] == target[i], axis=1) & (jnp.arange(3) != i)
            mass = jnp.sum(jnp.where(occupants, 1 + s['load'], 0)) + 1 + s['load'][i]
            tile = s['tiles'][target[i, 1], target[i, 0]]
            allowed = (tile == 0) | (tile == 3) | (jnp.any(at_bridge) & s['built'][bi] & (mass <= s['capacity'][bi]))
            n['agent_pos'] = n['agent_pos'].at[i].set(jnp.where(allowed, target[i], n['agent_pos'][i]))
        n = self.refresh(n)
        carry_limit = jnp.array([0, 1, 2], jnp.int32)
        for offset in range(3):
            i = (s['t'] + offset) % 3
            on_land = n['tiles'][n['agent_pos'][i, 1], n['agent_pos'][i, 0]] != 2
            nearby = jnp.sum(jnp.abs(n['part_pos'] - n['agent_pos'][i]), axis=1) <= 1
            available = nearby & (n['part_owner'] == -1)
            pi = jnp.argmax(available).astype(jnp.int32)
            pickup = (a[i] == 5) & on_land & (n['load'][i] < carry_limit[i]) & jnp.any(available)
            n['part_owner'] = n['part_owner'].at[pi].set(jnp.where(pickup, i, n['part_owner'][pi]))
            held = n['part_owner'] == i
            pi = jnp.argmax(held).astype(jnp.int32)
            drop = (a[i] == 11) & on_land & jnp.any(held)
            n['part_pos'] = n['part_pos'].at[pi].set(jnp.where(drop, n['agent_pos'][i], n['part_pos'][pi]))
            n['part_owner'] = n['part_owner'].at[pi].set(jnp.where(drop, -1, n['part_owner'][pi]))
            n = self.refresh(n)
        # Consent-based physical handoff: both give and receive actions are required.
        for offset in range(3):
            i = (s['t'] + offset) % 3
            receivers = (a == 9) & (jnp.arange(3) != i) & (n['load'] < carry_limit) & (
                jnp.sum(jnp.abs(n['agent_pos'] - n['agent_pos'][i]), axis=1) <= 1)
            rank = (jnp.arange(3) - s['t']) % 3
            receiver = jnp.argmin(jnp.where(receivers, rank, 9)).astype(jnp.int32)
            held = n['part_owner'] == i
            pi = jnp.argmax(held).astype(jnp.int32)
            give = (a[i] == 8) & jnp.any(held) & jnp.any(receivers)
            n['part_owner'] = n['part_owner'].at[pi].set(jnp.where(give, receiver, n['part_owner'][pi]))
            n = self.refresh(n)
        n['support'] = jnp.maximum(0, s['support'] - 1)
        brace_near = jnp.sum(jnp.abs(n['bridge_pos'] - n['agent_pos'][0]), axis=1) <= 1
        bi = jnp.argmax(brace_near).astype(jnp.int32)
        brace = (a[0] == 7) & jnp.any(brace_near)
        n['support'] = n['support'].at[bi].set(jnp.where(brace, 10, n['support'][bi]))
        builder_near = jnp.sum(jnp.abs(n['bridge_pos'] - n['agent_pos'][1]), axis=1) <= 1
        empty_site = builder_near & ~n['built']
        bi = jnp.argmax(empty_site).astype(jnp.int32)
        held = n['part_owner'] == 1
        pi = jnp.argmax(held).astype(jnp.int32)
        build = (a[1] == 6) & jnp.any(empty_site) & jnp.any(held)
        n['part_owner'] = n['part_owner'].at[pi].set(jnp.where(build, 3 + bi, n['part_owner'][pi]))
        n['life'] = n['life'].at[bi].set(jnp.where(build, 12, n['life'][bi]))
        n = self.refresh(n)
        occupied_sites = jnp.any(jnp.all(n['agent_pos'][:, None, :] == n['bridge_pos'][None, :, :], axis=2), axis=0)
        removable = builder_near & n['built'] & ~occupied_sites
        bi = jnp.argmax(removable).astype(jnp.int32)
        pi = jnp.argmax(n['part_owner'] == 3 + bi).astype(jnp.int32)
        remove = (a[1] == 10) & (n['load'][1] < 1) & jnp.any(removable)
        n['part_owner'] = n['part_owner'].at[pi].set(jnp.where(remove, 1, n['part_owner'][pi]))
        n = self.refresh(n)
        n['life'] = jnp.where(n['built'], jnp.where(n['support'] > 0,
            jnp.maximum(n['life'], 6), jnp.maximum(n['life'] - 1, 0)), 0).astype(jnp.int32)
        at_site = jnp.all(n['agent_pos'][:, None, :] == n['bridge_pos'][None, :, :], axis=2)
        mass = jnp.sum(at_site.astype(jnp.int32) * (1 + n['load'])[:, None], axis=0)
        collapsed = n['built'] & ((n['life'] <= 0) | (mass > n['capacity']))
        fallen = jnp.any(at_site & collapsed[None, :], axis=1)
        bank = n['bridge_pos'] - jnp.array([1, 0], jnp.int32)
        agent_site = jnp.argmax(at_site, axis=1).astype(jnp.int32)
        n['agent_pos'] = jnp.where(fallen[:, None], bank[agent_site], n['agent_pos'])
        deployed = n['part_owner'] >= 3
        part_site = jnp.clip(n['part_owner'] - 3, 0, 3)
        lost_span = deployed & collapsed[part_site]
        held = (n['part_owner'] >= 0) & (n['part_owner'] < 3)
        carrier = jnp.clip(n['part_owner'], 0, 2)
        dropped = held & fallen[carrier]
        n['part_pos'] = jnp.where(lost_span[:, None], bank[part_site], n['part_pos'])
        n['part_pos'] = jnp.where(dropped[:, None], n['agent_pos'][carrier], n['part_pos'])
        n['part_owner'] = jnp.where(lost_span | dropped, -1, n['part_owner'])
        n['life'] = jnp.where(collapsed, 0, n['life'])
        n = self.refresh(n)
        n['collapse_count'] = s['collapse_count'] + jnp.sum(collapsed.astype(jnp.int32))
        n['fall_count'] = s['fall_count'] + jnp.sum(fallen.astype(jnp.int32))
        first_arrival = (n['agent_pos'][:, 0] >= 6) & ~s['arrived']
        n['arrived'] = s['arrived'] | first_arrival
        n['t'] = s['t'] + jnp.array(1, jnp.int32)
        n['terminated'] = jnp.all(n['agent_pos'][:, 0] >= 6)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        reward = jnp.full(3, 2.0 * jnp.sum(first_arrival.astype(jnp.float32)) + 5.0 * n['terminated'] -
                          0.5 * jnp.sum(fallen.astype(jnp.float32)) - 0.01, jnp.float32)
        n = {k: jnp.where(s['done'], s[k], n[k]) for k in s}
        reward = jnp.where(s['done'], jnp.zeros(3, jnp.float32), reward)
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), {
            'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
