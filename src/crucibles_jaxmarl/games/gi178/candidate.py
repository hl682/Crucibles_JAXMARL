"""Orbit Salvage: original bounded integer tug game; authored, never executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 44
    num_actions = 8
    max_steps = 96

    def refresh(self, s):
        open_dock = jnp.full((2,), jnp.mod(s['t'], 6) < 3, jnp.bool_)
        delta = s['cargo_pos'][0] - s['pos']
        midpoint = (s['pos'] + s['cargo_pos'][0]).astype(jnp.float32) / 2.0
        return dict(s, dock_open=open_dock, tether_pos=midpoint,
                    tether_vector=jnp.where(s['attached'][:, None], delta, 0))

    def reset(self, key):
        preference = jax.random.randint(key, (3,), 0, 2, dtype=jnp.int32)
        s = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False), 'terminated': jnp.array(False),
            'truncated': jnp.array(False),
            'pos': jnp.array([[5, 5], [6, 4], [7, 5]], jnp.int32),
            'vel': jnp.zeros((3, 2), jnp.int32),
            'fuel': jnp.full((3,), 18, jnp.int32),
            'attached': jnp.zeros((3,), jnp.bool_),
            'offset': jnp.zeros((3, 2), jnp.int32),
            'preference': preference,
            'dock_values': 1 + 3 * (preference[:, None] == jnp.arange(2)[None, :]).astype(jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_),
            'agent_visible': jnp.ones((3, 3), jnp.bool_),
            'private_visible': jnp.eye(3, dtype=jnp.bool_),
            'cargo_pos': jnp.array([[6, 5]], jnp.int32),
            'cargo_vel': jnp.zeros((1, 2), jnp.int32),
            'cargo_health': jnp.array([6], jnp.int32),
            'cargo_delivered': jnp.array([False]),
            'cargo_active': jnp.ones((1,), jnp.bool_),
            'cargo_visible': jnp.ones((3, 1), jnp.bool_),
            'dock_pos': jnp.array([[2, 5], [10, 5]], jnp.int32),
            'dock_active': jnp.ones((2,), jnp.bool_),
            'dock_visible': jnp.ones((3, 2), jnp.bool_),
            'dock_open': jnp.ones((2,), jnp.bool_),
            'dock_occupied': jnp.zeros((2,), jnp.bool_),
            'tether_pos': jnp.zeros((3, 2), jnp.float32),
            'tether_vector': jnp.zeros((3, 2), jnp.int32),
            'spent': jnp.zeros((3,), jnp.int32),
            'fuel_given': jnp.zeros((3,), jnp.int32),
            'burned': jnp.array(0, jnp.int32),
            'collision': jnp.zeros((3,), jnp.bool_),
            'cargo_hit': jnp.zeros((1,), jnp.bool_),
            'docking': jnp.zeros((2,), jnp.bool_)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        near = jnp.sum(jnp.abs(s['pos'] - s['cargo_pos'][0]), axis=1) == 1
        neighbors = (jnp.sum(jnp.abs(s['pos'][:, None] - s['pos'][None, :]), axis=2) <= 1)
        neighbors = neighbors & ~jnp.eye(3, dtype=jnp.bool_)
        has_fuel = s['fuel'] > 0
        mask = jnp.stack([jnp.ones((3,), jnp.bool_), has_fuel, has_fuel,
                          has_fuel, has_fuel, has_fuel,
                          s['attached'] | near, has_fuel & jnp.any(neighbors, axis=1)], axis=1)
        return mask & ((~s['done']) | (jnp.arange(8)[None, :] == 0))

    def advance(self, s, actions):
        # Invalid actions become coast. Release/latch happens before force aggregation.
        safe = jnp.clip(actions, 0, 7)
        valid = (actions >= 0) & (actions < 8) & self.legal_actions(s)[jnp.arange(3), safe]
        a = jnp.where(valid, safe, 0)
        attached = jnp.where(a == 6, ~s['attached'], s['attached'])
        offset = jnp.where(((a == 6) & attached)[:, None], s['pos'] - s['cargo_pos'][0], s['offset'])
        thrust_table = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1],
                                  [0, 1], [0, 0], [0, 0], [0, 0]], jnp.int32)
        spent = ((a >= 1) & (a <= 5)).astype(jnp.int32)
        force = jnp.where((a == 5)[:, None], -jnp.sign(s['cargo_vel'][0]), thrust_table[a])
        total_force = jnp.sum(jnp.where(attached[:, None], force, 0), axis=0)
        acceleration = jnp.sign(total_force) * (jnp.abs(total_force) // 2)
        cv = jnp.clip(s['cargo_vel'][0] + acceleration, -1, 1)
        cp = s['cargo_pos'][0] + cv
        ap = cp + offset
        upper = jnp.array([12, 10], jnp.int32)
        group_wall = jnp.any((cp < 0) | (cp > upper)) | jnp.any(
            attached[:, None] & ((ap < 0) | (ap > upper)))
        cargo_free = jnp.any((~attached) & jnp.all(s['pos'] == cp, axis=1))
        arm_free = jnp.any(attached[:, None] & (~attached)[None, :] &
                           jnp.all(ap[:, None, :] == s['pos'][None, :, :], axis=2))
        group_blocked = group_wall | cargo_free | arm_free
        cargo_hit = group_blocked & jnp.any(cv != 0)
        cp = jnp.where(group_blocked, s['cargo_pos'][0], cp)
        cv = jnp.where(group_blocked, 0, cv)
        ap = cp + offset
        # Heavy assembly settles first; free tugs cannot enter any old free-tug cell.
        free_v = jnp.where((a == 5)[:, None], 0, jnp.clip(s['vel'] + thrust_table[a], -1, 1))
        target = s['pos'] + free_v
        wall = jnp.any((target < 0) | (target > upper), axis=1)
        cargo_block = jnp.all(target == cp, axis=1)
        arm_block = jnp.any(attached[None, :] &
                            jnp.all(target[:, None, :] == ap[None, :, :], axis=2), axis=1)
        other = ~jnp.eye(3, dtype=jnp.bool_)
        old_block = jnp.any(other & (~attached)[None, :] &
                            jnp.all(target[:, None, :] == s['pos'][None, :, :], axis=2), axis=1)
        contention = jnp.any(other & (~attached)[None, :] &
                             jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1)
        blocked = wall | cargo_block | arm_block | old_block | contention
        collision = (~attached) & blocked & jnp.any(free_v != 0, axis=1)
        pos = jnp.where(attached[:, None], ap, jnp.where(blocked[:, None], s['pos'], target))
        vel = jnp.where(attached[:, None], cv, jnp.where(blocked[:, None], 0, free_v))
        # One unit is handed to the poorest old adjacent peer; all transfers are simultaneous.
        neighbors = (jnp.sum(jnp.abs(s['pos'][:, None] - s['pos'][None, :]), axis=2) <= 1) & other
        target_id = jnp.argmin(jnp.where(neighbors, s['fuel'][None, :] * 4 + jnp.arange(3)[None, :], 999), axis=1)
        given = (a == 7).astype(jnp.int32)
        received = jnp.sum(jax.nn.one_hot(target_id, 3, dtype=jnp.int32) * given[:, None], axis=0)
        fuel = s['fuel'] - spent - given + received
        t = s['t'] + 1
        health = jnp.maximum(s['cargo_health'] - cargo_hit.astype(jnp.int32), 0)
        # Exact-position zero-speed docking needs two attached tugs and a live load.
        docking = (jnp.all(s['dock_pos'] == cp, axis=1) & jnp.all(cv == 0) &
                   (jnp.sum(attached) >= 2) & (jnp.mod(t, 6) < 3) & (health[0] > 0))
        success = jnp.any(docking)
        terminated = success | (health[0] == 0)
        truncated = (t >= self.max_steps) & ~terminated
        done = terminated | truncated
        reward = jnp.sum(s['dock_values'] * docking[None, :], axis=1).astype(jnp.float32)
        reward = reward - 0.02 * spent.astype(jnp.float32) - 0.15 * collision.astype(jnp.float32)
        reward = reward - 0.1 * cargo_hit.astype(jnp.float32)
        n = dict(s, t=t, done=done, terminated=terminated, truncated=truncated,
                 pos=pos, vel=vel, fuel=fuel, attached=attached, offset=offset,
                 cargo_pos=cp[None, :], cargo_vel=cv[None, :], cargo_health=health,
                 cargo_delivered=jnp.array([success]), dock_occupied=docking,
                 spent=spent, fuel_given=given, burned=s['burned'] + jnp.sum(spent),
                 collision=collision, cargo_hit=jnp.array([cargo_hit]), docking=docking)
        return self.refresh(n), reward

    def observe(self, s):
        own = jnp.concatenate([s['pos'] / jnp.array([12.0, 10.0]), s['vel'],
                               s['fuel'][:, None] / 54.0, s['attached'][:, None],
                               jax.nn.one_hot(s['preference'], 2)], axis=1)
        public = jnp.concatenate([jnp.array([s['t'] / 96.0]),
                                  (s['pos'] / jnp.array([12.0, 10.0])).reshape(-1),
                                  s['vel'].reshape(-1), s['fuel'] / 54.0, s['attached'],
                                  (s['cargo_pos'] / jnp.array([12.0, 10.0])).reshape(-1),
                                  s['cargo_vel'].reshape(-1), s['cargo_health'] / 6.0,
                                  (s['dock_pos'] / jnp.array([12.0, 10.0])).reshape(-1),
                                  s['dock_open'], s['dock_occupied'], jnp.array([s['done']])])
        return jnp.concatenate([own, jnp.broadcast_to(public, (3, 33)), jnp.eye(3)], axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'],
                                 lambda _: (state, jnp.zeros((3,), jnp.float32)),
                                 lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((3,), n['done'], jnp.bool_), info
