"""Counterweight Crew: original prospective source; never executed by author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 119
    num_actions = 7
    max_steps = 80

    def reset(self, key):
        tiles = jnp.zeros((7, 9), dtype=jnp.int32)
        tiles = tiles.at[jnp.array([0, 2, 4, 6]), 4].set(1)
        s = {
            't': jnp.array(0, jnp.int32),
            'episode_done': jnp.array(False),
            'terminated': jnp.array(False),
            'truncated': jnp.array(False),
            'tiles': tiles,
            'agent_pos': jnp.array([[3, 3], [5, 3], [6, 4]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_),
            'agent_visible': jnp.ones((3, 3), jnp.bool_),
            'stamina': jnp.full((3,), 8, jnp.int32),
            'attachment': jnp.full((3,), -1, jnp.int32),
            'cargo_pos': jnp.array([[4, 3], [2, 1], [6, 5]], jnp.int32),
            'cargo_active': jnp.ones((3,), jnp.bool_),
            'cargo_visible': jnp.ones((3, 3), jnp.bool_),
            'cargo_weight': jnp.array([2, 1, 1], jnp.int32),
            'cargo_delivered': jnp.zeros((3,), jnp.bool_),
            'force': jnp.zeros((3, 2), jnp.int32),
            'effort': jnp.zeros((3,), jnp.int32),
            'moved': jnp.zeros((3,), jnp.bool_),
            'delivered_event': jnp.zeros((3,), jnp.bool_),
            'depot_pos': jnp.array([[0, 3], [8, 1], [8, 5]], jnp.int32),
            'depot_active': jnp.ones((3,), jnp.bool_),
            'depot_visible': jnp.ones((3, 3), jnp.bool_),
            'values': jnp.array([[6, 1, 2], [1, 6, 2], [2, 1, 6]], jnp.int32),
        }
        return self.observe(s), s

    def floor_ok(self, s, p):
        inside = (p[..., 0] >= 0) & (p[..., 0] <= 8) & (p[..., 1] >= 0) & (p[..., 1] <= 6)
        tile = s['tiles'][jnp.clip(p[..., 1], 0, 6), jnp.clip(p[..., 0], 0, 8)]
        return inside & (tile == 0)

    def legal_actions(self, s):
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        target = s['agent_pos'][:, None, :] + dirs[None, :, :]
        cargo_hit = jnp.any(jnp.all(target[:, :, None, :] == s['cargo_pos'][None, None, :, :], axis=-1) & s['cargo_active'][None, None, :], axis=-1)
        free = self.floor_ok(s, target) & ~cargo_hit
        moving = (s['stamina'][:, None] > 0) & ((s['attachment'][:, None] >= 0) | free)
        near = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['cargo_pos'][None, :, :]), axis=-1) <= 1
        attach = (s['attachment'] < 0) & jnp.any(near & s['cargo_active'][None, :], axis=1)
        mask = jnp.concatenate([jnp.ones((3, 1), jnp.bool_), moving, attach[:, None], (s['attachment'] >= 0)[:, None]], axis=1)
        return mask & (~s['episode_done'] | (jnp.arange(7)[None, :] == 0))

    def observe(self, s):
        public = jnp.concatenate([
            s['t'][None], s['agent_pos'].reshape(-1), s['stamina'], s['attachment'],
            s['cargo_pos'].reshape(-1), s['cargo_active'].astype(jnp.int32), s['cargo_weight'],
            s['force'].reshape(-1), s['depot_pos'].reshape(-1), s['values'].reshape(-1),
            s['tiles'].reshape(-1), s['episode_done'][None].astype(jnp.int32),
            s['effort'], s['moved'].astype(jnp.int32),
        ]).astype(jnp.float32)
        return jnp.concatenate([jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public[None, :], (3, 116))], axis=1)

    def advance(self, s, actions):
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0]], jnp.int32)
        clipped = jnp.clip(actions, 0, 6)
        valid = (actions == clipped) & self.legal_actions(s)[jnp.arange(3), clipped]
        a = jnp.where(valid, clipped, 0)
        distance = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['cargo_pos'][None, :, :]), axis=-1)
        nearest = jnp.argmin(jnp.where(s['cargo_active'][None, :] & (distance <= 1), distance, 99), axis=1).astype(jnp.int32)
        attachment = jnp.where(a == 5, nearest, jnp.where(a == 6, -1, s['attachment']))
        effort = ((a >= 1) & (a <= 4)).astype(jnp.int32)
        pos = s['agent_pos'] + jnp.where((attachment < 0)[:, None], dirs[a], 0)
        attached = attachment[:, None] == jnp.arange(3)[None, :]
        force = jnp.sum(attached[:, :, None].astype(jnp.int32) * dirs[a][:, None, :], axis=0, dtype=jnp.int32)
        cargo = s['cargo_pos']
        moved = jnp.zeros((3,), jnp.bool_)
        # All workers choose simultaneously; cargo occupancy resolves in stable cargo-ID order.
        for k in range(3):
            fx, fy = force[k, 0], force[k, 1]
            xaxis = jnp.abs(fx) >= jnp.abs(fy)
            delta = jnp.array([jnp.where(xaxis, jnp.sign(fx), 0), jnp.where(xaxis, 0, jnp.sign(fy))], jnp.int32)
            enough = jnp.maximum(jnp.abs(fx), jnp.abs(fy)) >= s['cargo_weight'][k]
            target = cargo[k] + delta
            conflict = jnp.any(jnp.all(cargo == target[None, :], axis=1) & s['cargo_active'] & (jnp.arange(3) != k))
            tether = attachment == k
            followers = pos + delta[None, :]
            follower_block = jnp.any(jnp.all(followers[:, None, :] == cargo[None, :, :], axis=-1) & s['cargo_active'][None, :] & (jnp.arange(3)[None, :] != k), axis=1)
            followers_ok = jnp.all(~tether | (self.floor_ok(s, followers) & ~follower_block))
            worker_block = jnp.any(jnp.all(pos == target[None, :], axis=1) & ~tether)
            go = s['cargo_active'][k] & enough & self.floor_ok(s, target) & ~conflict & ~worker_block & followers_ok
            cargo = cargo.at[k].set(jnp.where(go, target, cargo[k]))
            pos = pos + jnp.where((go & tether)[:, None], delta[None, :], 0)
            moved = moved.at[k].set(go)
        at_depot = jnp.all(cargo[:, None, :] == s['depot_pos'][None, :, :], axis=-1)
        delivered = s['cargo_active'] & jnp.any(at_depot, axis=1)
        destination = jnp.argmax(at_depot, axis=1)
        payoff = jnp.sum(s['values'][:, destination] * (delivered.astype(jnp.int32) * s['cargo_weight'])[None, :], axis=1)
        active = s['cargo_active'] & ~delivered
        attachment = jnp.where((attachment >= 0) & delivered[jnp.maximum(attachment, 0)], -1, attachment)
        n = dict(s)
        n['t'] = s['t'] + 1
        n['agent_pos'] = pos
        n['stamina'] = jnp.clip(s['stamina'] - effort + 2 * (a == 0).astype(jnp.int32), 0, 10)
        n['attachment'] = attachment
        n['cargo_pos'] = cargo
        n['cargo_active'] = active
        n['cargo_delivered'] = s['cargo_delivered'] | delivered
        n['force'] = force
        n['effort'] = effort
        n['moved'] = moved
        n['delivered_event'] = delivered
        n['terminated'] = ~jnp.any(active)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['episode_done'] = n['terminated'] | n['truncated']
        return n, (payoff - effort).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        next_state, reward = jax.lax.cond(
            state['episode_done'],
            lambda _: (state, jnp.zeros((3,), jnp.float32)),
            lambda _: self.advance(state, actions),
            operand=None,
        )
        info = {'done_all': next_state['episode_done'], 'terminated': next_state['terminated'], 'truncated': next_state['truncated']}
        dones = jnp.full((3,), next_state['episode_done'], jnp.bool_)
        return self.observe(next_state), next_state, reward, dones, info
