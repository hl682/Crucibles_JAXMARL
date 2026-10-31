"""Bridge Pact: original complementary-material crossings; authored, never executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 137
    num_actions = 11
    max_steps = 120

    def refresh(self, s):
        occupancy = jnp.sum(jnp.all(s['pos'][:, None, :] == s['bridge_pos'][None, :, :], axis=2), axis=0).astype(jnp.int32)
        return dict(s, bridge_occupancy=occupancy)

    def reset(self, key):
        x = jnp.arange(9, dtype=jnp.int32)[None, :]
        y = jnp.arange(7, dtype=jnp.int32)[:, None]
        water = (x >= 3) & (x <= 5)
        tiles = jnp.where(water, jnp.where((y == 1) | (y == 5), 2, 1), 0).astype(jnp.int32)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'map': tiles,
            'pos': jnp.array([[2, 1], [2, 2], [6, 1], [6, 2]], jnp.int32),
            'agent_active': jnp.ones((4,), jnp.bool_),
            'agent_visible': jnp.ones((4, 4), jnp.bool_),
            'private_visible': jnp.eye(4, dtype=jnp.bool_),
            'materials': jnp.array([[4, 0], [0, 4], [0, 4], [4, 0]], jnp.int32),
            'values': jax.random.randint(key, (4, 2), 2, 6, dtype=jnp.int32),
            'delivered': jnp.zeros((4,), jnp.bool_),
            'bridge_pos': jnp.array([[3, 1], [4, 1], [5, 1], [3, 5], [4, 5], [5, 5]], jnp.int32),
            'bridge_active': jnp.ones((6,), jnp.bool_),
            'bridge_visible': jnp.ones((4, 6), jnp.bool_),
            'wood': jnp.zeros((6,), jnp.int32),
            'bolts': jnp.zeros((6,), jnp.int32),
            'health': jnp.zeros((6,), jnp.int32),
            'bridge_occupancy': jnp.zeros((6,), jnp.int32),
            'terminal_pos': jnp.array([[1, 1], [1, 5], [7, 1], [7, 5]], jnp.int32),
            'terminal_active': jnp.ones((4,), jnp.bool_),
            'terminal_visible': jnp.ones((4, 4), jnp.bool_),
            'repair_used': jnp.zeros((2,), jnp.int32),
            'contributed': jnp.zeros((4, 2), jnp.int32),
            'shared': jnp.zeros((4, 2), jnp.int32),
            'repaired': jnp.zeros((4,), jnp.bool_),
            'completed_now': jnp.zeros((6,), jnp.bool_),
            'worn_out': jnp.zeros((6,), jnp.bool_),
            'delivered_now': jnp.zeros((4,), jnp.bool_)
        }
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(4, dtype=jnp.int32)
        directions = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
        target = s['pos'][:, None, :] + directions[None, :, :]
        clipped = jnp.clip(target, jnp.array([0, 0]), jnp.array([8, 6]))
        tile = s['map'][clipped[:, :, 1], clipped[:, :, 0]]
        match = jnp.all(target[:, :, None, :] == s['bridge_pos'][None, None, :, :], axis=3)
        live = (s['wood'] == 1) & (s['bolts'] == 1) & (s['health'] > 0)
        bridge_ok = jnp.any(match & live[None, None, :], axis=2)
        occupied = jnp.any(jnp.all(target[:, :, None, :] == s['pos'][None, None, :, :], axis=3), axis=2)
        move = jnp.all(target == clipped, axis=2) & ~occupied & ((tile == 0) | bridge_ok)
        reach = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['bridge_pos'][None, :, :]), axis=2) <= 1
        wood = (s['materials'][:, 0] > 0) & jnp.any(reach & (s['wood'][None, :] == 0), axis=1)
        bolts = (s['materials'][:, 1] > 0) & jnp.any(reach & (s['bolts'][None, :] == 0), axis=1)
        neighbor = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2) == 1
        give = jnp.any(neighbor, axis=1)
        repairable = (s['wood'] == 1) & (s['bolts'] == 1) & (s['health'] < 12)
        repair = jnp.all(s['materials'] > 0, axis=1) & jnp.any(reach & repairable[None, :], axis=1)
        goal_x = jnp.where(ids < 2, 7, 1)
        deliver = (~s['delivered']) & (s['pos'][:, 0] == goal_x) & ((s['pos'][:, 1] == 1) | (s['pos'][:, 1] == 5))
        mask = jnp.concatenate([jnp.ones((4, 1), jnp.bool_), move, wood[:, None], bolts[:, None],
                                (give & (s['materials'][:, 0] > 0))[:, None],
                                (give & (s['materials'][:, 1] > 0))[:, None],
                                repair[:, None], deliver[:, None]], axis=1)
        return mask & ((~s['done']) | (jnp.arange(11)[None, :] == 0))

    def advance(self, s, actions):
        ids = jnp.arange(4, dtype=jnp.int32)
        safe = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(s)[ids, safe]
        a = jnp.where(valid, safe, 0)
        directions = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        target = s['pos'] + directions[a]
        clash = jnp.all(target[:, None, :] == target[None, :, :], axis=2)
        loses = jnp.any(clash & (ids[None, :] < ids[:, None]), axis=1)
        pos = jnp.where(loses[:, None], s['pos'], target)
        moved = jnp.any(pos != s['pos'], axis=1)
        reach = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['bridge_pos'][None, :, :]), axis=2) <= 1
        wood_eligible = reach & (s['wood'][None, :] == 0)
        bolt_eligible = reach & (s['bolts'][None, :] == 0)
        wood_target = jnp.argmin(jnp.where(wood_eligible, jnp.arange(6)[None, :], 99), axis=1)
        bolt_target = jnp.argmin(jnp.where(bolt_eligible, jnp.arange(6)[None, :], 99), axis=1)
        wants_wood = a == 5
        wants_bolt = a == 6
        lower = ids[None, :] < ids[:, None]
        wood_lost = jnp.any((wood_target[:, None] == wood_target[None, :]) & wants_wood[None, :] & lower, axis=1)
        bolt_lost = jnp.any((bolt_target[:, None] == bolt_target[None, :]) & wants_bolt[None, :] & lower, axis=1)
        put_wood = wants_wood & ~wood_lost
        put_bolt = wants_bolt & ~bolt_lost
        contributed = jnp.stack([put_wood, put_bolt], axis=1).astype(jnp.int32)
        wood = s['wood'] + jnp.sum(jax.nn.one_hot(wood_target, 6, dtype=jnp.int32) * put_wood[:, None], axis=0)
        bolts = s['bolts'] + jnp.sum(jax.nn.one_hot(bolt_target, 6, dtype=jnp.int32) * put_bolt[:, None], axis=0)
        complete = (wood == 1) & (bolts == 1)
        completed_now = complete & ~((s['wood'] == 1) & (s['bolts'] == 1))
        health = jnp.where(completed_now, 12, s['health'])
        # Repairs consume both materials. Concurrent repairs may waste capacity, never material.
        repairable = reach & (s['wood'][None, :] == 1) & (s['bolts'][None, :] == 1) & (s['health'][None, :] < 12)
        repair_target = jnp.argmin(jnp.where(repairable, s['health'][None, :] * 7 + jnp.arange(6)[None, :], 999), axis=1)
        repaired = a == 9
        repair_count = jnp.sum(jax.nn.one_hot(repair_target, 6, dtype=jnp.int32) * repaired[:, None], axis=0)
        health = jnp.minimum(12, health + 6 * repair_count)
        # Material handoffs use old adjacency; the poorest peer in that material wins ties by id.
        neighbor = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2) == 1
        materials = s['materials'] - contributed - repaired[:, None].astype(jnp.int32)
        shared = jnp.stack([a == 7, a == 8], axis=1).astype(jnp.int32)
        for resource in range(2):
            recipient = jnp.argmin(jnp.where(neighbor, s['materials'][None, :, resource] * 5 + ids[None, :], 999), axis=1)
            received = jnp.sum(jax.nn.one_hot(recipient, 4, dtype=jnp.int32) * shared[:, resource, None], axis=0)
            materials = materials.at[:, resource].add(received - shared[:, resource])
        # Old live cells admit movement. Entry wear and four-tick decay apply afterward.
        entry = jnp.all(pos[:, None, :] == s['bridge_pos'][None, :, :], axis=2) & moved[:, None]
        wear = jnp.sum(entry, axis=0).astype(jnp.int32)
        t = s['t'] + 1
        decay = ((jnp.mod(t, 4) == 0) & complete).astype(jnp.int32)
        health = jnp.maximum(health - wear - decay, 0)
        worn_out = (s['health'] > 0) & (health == 0)
        # A zero-health occupied cell retains its occupant; only new entry is prohibited.
        delivered_now = a == 10
        destination = (s['pos'][:, 1] == 5).astype(jnp.int32)
        value = s['values'][ids, destination].astype(jnp.float32) * delivered_now
        reward = value + 0.2 * jnp.sum(delivered_now) - 0.02 * jnp.sum(contributed, axis=1)
        reward = reward - 0.04 * repaired - 0.005 * moved
        delivered = s['delivered'] | delivered_now
        terminated = jnp.all(delivered)
        truncated = (t >= self.max_steps) & ~terminated
        used = jnp.full((2,), jnp.sum(repaired), jnp.int32)
        n = dict(s, t=t, done=terminated | truncated, terminated=terminated, truncated=truncated,
                 pos=pos, materials=materials, wood=wood, bolts=bolts, health=health,
                 delivered=delivered, repair_used=s['repair_used'] + used,
                 contributed=contributed, shared=shared, repaired=repaired,
                 completed_now=completed_now, worn_out=worn_out, delivered_now=delivered_now)
        return self.refresh(n), reward.astype(jnp.float32)

    def observe(self, s):
        public = jnp.concatenate([jnp.array([s['t'] / 120.0]),
                                  (s['pos'] / jnp.array([8.0, 6.0])).reshape(-1),
                                  s['materials'].reshape(-1) / 8.0, s['delivered'],
                                  (s['bridge_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
                                  s['wood'], s['bolts'], s['health'] / 12.0, s['bridge_occupancy'],
                                  (s['terminal_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
                                  s['map'].reshape(-1) / 2.0, s['repair_used'] / 8.0,
                                  jnp.array([s['done']])])
        return jnp.concatenate([s['values'] / 5.0, jnp.broadcast_to(public, (4, 131)),
                                 jnp.eye(4)], axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'],
                                 lambda _: (state, jnp.zeros((4,), jnp.float32)),
                                 lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((4,), n['done'], jnp.bool_), info
