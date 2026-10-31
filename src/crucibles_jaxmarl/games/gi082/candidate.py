"""Lantern Flood: original three-role rescue kernel; author has not executed it."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 121
    num_actions = 9
    max_steps = 96

    def refresh(self, s):
        n = dict(s)
        radii = jnp.array([4, 2, 2], jnp.int32)
        ad = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=-1)
        vd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['victim_pos'][None, :, :]), axis=-1)
        pd = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pack_pos'][None, :, :]), axis=-1)
        ba = jnp.sum(jnp.abs(s['beacon_pos'][:, None, :] - s['pos'][None, :, :]), axis=-1)
        bv = jnp.sum(jnp.abs(s['beacon_pos'][:, None, :] - s['victim_pos'][None, :, :]), axis=-1)
        bp = jnp.sum(jnp.abs(s['beacon_pos'][:, None, :] - s['pack_pos'][None, :, :]), axis=-1)
        n['agent_vis'] = (ad <= radii[:, None]) | jnp.any((ba <= 2) & (s['beacon_ttl'][:, None] > 0), axis=0)[None, :]
        n['victim_active'] = s['victim_status'] == 0
        n['victim_vis'] = ((vd <= radii[:, None]) | jnp.any((bv <= 2) & (s['beacon_ttl'][:, None] > 0), axis=0)[None, :]) & n['victim_active'][None, :]
        n['pack_vis'] = ((pd <= radii[:, None]) | jnp.any((bp <= 2) & (s['beacon_ttl'][:, None] > 0), axis=0)[None, :]) & s['pack_active'][None, :]
        n['beacon_active'] = s['beacon_ttl'] > 0
        n['beacon_vis'] = jnp.broadcast_to(n['beacon_active'][None, :], (3, 3))
        held = (s['carrier'][None, :] == jnp.arange(3)[:, None]) & n['victim_active'][None, :]
        n['carrying'] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        return n

    def reset(self, key):
        grid = jnp.zeros((7, 9), jnp.int32)
        grid = grid.at[jnp.array([0, 1, 2, 4, 5]), 4].set(1)
        grid = grid.at[3, 4].set(2)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'grid': grid, 'pos': jnp.array([[1, 1], [1, 3], [1, 5]], jnp.int32),
            'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
            'carrying': jnp.full(3, -1, jnp.int32), 'supply': jnp.array(3, jnp.int32),
            'victim_pos': jnp.array([[7, 1], [7, 5]], jnp.int32),
            'victim_active': jnp.ones(2, jnp.bool_), 'victim_vis': jnp.zeros((3, 2), jnp.bool_),
            'life': jnp.array([54, 60], jnp.int32), 'stable': jnp.zeros(2, jnp.bool_),
            'carrier': jnp.full(2, -1, jnp.int32), 'victim_status': jnp.zeros(2, jnp.int32),
            'pack_pos': jnp.array([[2, 3], [6, 6]], jnp.int32),
            'pack_active': jnp.ones(2, jnp.bool_), 'pack_vis': jnp.zeros((3, 2), jnp.bool_),
            'pack_value': jnp.full(2, 2, jnp.int32),
            'gate_pos': jnp.array([[4, 3]], jnp.int32), 'gate_active': jnp.ones(1, jnp.bool_),
            'gate_vis': jnp.ones((3, 1), jnp.bool_), 'gate_open': jnp.zeros(1, jnp.bool_),
            'depot_pos': jnp.array([[0, 3]], jnp.int32), 'depot_active': jnp.ones(1, jnp.bool_),
            'depot_vis': jnp.ones((3, 1), jnp.bool_),
            'beacon_pos': jnp.array([[1, 1], [1, 3], [1, 5]], jnp.int32),
            'beacon_ttl': jnp.zeros(3, jnp.int32), 'beacon_active': jnp.zeros(3, jnp.bool_),
            'beacon_vis': jnp.zeros((3, 3), jnp.bool_),
            'rescued': jnp.array(0, jnp.int32), 'lost': jnp.array(0, jnp.int32),
            'ev_rescue': jnp.zeros(2, jnp.bool_), 'ev_loss': jnp.zeros(2, jnp.bool_),
            'ev_handoff': jnp.zeros(2, jnp.bool_), 'ev_stabilize': jnp.zeros(2, jnp.bool_),
            'ev_gate': jnp.array(False), 'ev_pack': jnp.zeros(2, jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = s['pos'][:, None, :] + delta[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 9) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 7)
        tile = s['grid'][jnp.clip(q[:, :, 1], 0, 6), jnp.clip(q[:, :, 0], 0, 8)]
        moves = inside & (tile != 1) & ((tile != 2) | s['gate_open'][0])
        mask = jnp.concatenate([moves, jnp.ones((3, 4), jnp.bool_)], axis=1)
        return (mask & ~s['done']) | (jnp.arange(9)[None, :] == 0)

    def observe(self, s):
        own = jnp.concatenate([jnp.eye(3, dtype=jnp.float32), s['pos'], s['carrying'][:, None]], axis=1)
        public = jnp.array([s['t'], s['supply'], s['gate_open'][0], s['rescued'], s['lost']], jnp.float32)
        af = jnp.concatenate([s['pos'], s['carrying'][:, None]], axis=1)
        vf = jnp.concatenate([s['victim_pos'], s['life'][:, None], s['stable'][:, None], s['carrier'][:, None], s['victim_status'][:, None]], axis=1)
        pf = jnp.concatenate([s['pack_pos'], s['pack_value'][:, None]], axis=1)
        ao = jnp.concatenate([jnp.where(s['agent_vis'][:, :, None], af[None, :, :], -1).reshape(3, 9), s['agent_vis']], axis=1)
        vo = jnp.concatenate([jnp.where(s['victim_vis'][:, :, None], vf[None, :, :], -1).reshape(3, 12), s['victim_vis']], axis=1)
        po = jnp.concatenate([jnp.where(s['pack_vis'][:, :, None], pf[None, :, :], -1).reshape(3, 6), s['pack_vis']], axis=1)
        bf = jnp.concatenate([s['beacon_pos'], s['beacon_ttl'][:, None]], axis=1)
        bo = jnp.where(s['beacon_active'][:, None], bf, -1).reshape(9)
        tail = jnp.concatenate([bo, s['grid'].reshape(63), s['gate_pos'].reshape(2), s['depot_pos'].reshape(2)])
        return jnp.concatenate([own, jnp.broadcast_to(public, (3, 5)), ao, vo, po, jnp.broadcast_to(tail, (3, 76))], axis=1).astype(jnp.float32)

    def advance(self, s, actions):
        n = dict(s)
        mask = self.legal_actions(s)
        safe = jnp.clip(actions, 0, 8)
        valid = (actions >= 0) & (actions < 9) & mask[jnp.arange(3), safe]
        a = jnp.where(valid, safe, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        q = s['pos'] + delta[a]
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=-1) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=-1), axis=1) > 1
        n['pos'] = jnp.where((~occupied & ~conflict)[:, None], q, s['pos'])
        n['victim_pos'] = jnp.where((s['carrier'] >= 0)[:, None], n['pos'][jnp.maximum(s['carrier'], 0)], s['victim_pos'])
        n['t'] = s['t'] + 1
        n['beacon_ttl'] = jnp.where(a == 8, 6, jnp.maximum(0, s['beacon_ttl'] - 1))
        n['beacon_pos'] = jnp.where((a == 8)[:, None], n['pos'], s['beacon_pos'])
        near_gate = jnp.sum(jnp.abs(n['pos'][1] - s['gate_pos'][0])) <= 1
        opened = (a[1] == 6) & near_gate & ~s['gate_open'][0] & (s['supply'] >= 2)
        n['gate_open'] = s['gate_open'] | opened
        n['supply'] = s['supply'] - 2 * opened.astype(jnp.int32)
        n['ev_gate'] = opened
        n['ev_pack'] = jnp.zeros(2, jnp.bool_)
        n['ev_stabilize'] = jnp.zeros(2, jnp.bool_)
        # Fixed role priority applies only to contended pickup/resource requests.
        for i in range(3):
            near_v = (jnp.sum(jnp.abs(n['victim_pos'] - n['pos'][i]), axis=1) <= 1) & (n['victim_status'] == 0)
            pickable = near_v & (n['carrier'] < 0)
            v = jnp.argmax(pickable)
            empty = ~jnp.any(n['carrier'] == i)
            take = (a[i] == 5) & jnp.any(pickable) & empty
            n['carrier'] = n['carrier'].at[v].set(jnp.where(take, i, n['carrier'][v]))
            near_p = (jnp.sum(jnp.abs(s['pack_pos'] - n['pos'][i]), axis=1) <= 1) & n['pack_active']
            p = jnp.argmax(near_p)
            collect = (a[i] == 5) & ~take & jnp.any(near_p)
            n['pack_active'] = n['pack_active'].at[p].set(n['pack_active'][p] & ~collect)
            n['supply'] = n['supply'] + jnp.where(collect, n['pack_value'][p], 0)
            n['ev_pack'] = n['ev_pack'].at[p].set(n['ev_pack'][p] | collect)
        need = (jnp.sum(jnp.abs(n['victim_pos'] - n['pos'][2]), axis=1) <= 1) & (n['victim_status'] == 0) & ~n['stable']
        v = jnp.argmax(need)
        stabilize = (a[2] == 6) & jnp.any(need) & (n['supply'] > 0)
        n['stable'] = n['stable'].at[v].set(n['stable'][v] | stabilize)
        n['ev_stabilize'] = n['ev_stabilize'].at[v].set(stabilize)
        n['supply'] = n['supply'] - stabilize.astype(jnp.int32)
        # At most one handoff per victim, using carrier occupancy before handoffs.
        before = n['carrier']
        n['ev_handoff'] = jnp.zeros(2, jnp.bool_)
        for v in range(2):
            c = jnp.maximum(before[v], 0)
            free = ~jnp.any(before[None, :] == jnp.arange(3)[:, None], axis=1)
            nearby = jnp.sum(jnp.abs(n['pos'] - n['pos'][c]), axis=1) == 1
            willing = free & nearby & (a == 7)
            receiver = jnp.argmax(willing)
            used = jnp.any(n['carrier'] == receiver)
            handoff = (before[v] >= 0) & (a[c] == 7) & jnp.any(willing) & ~used
            n['carrier'] = n['carrier'].at[v].set(jnp.where(handoff, receiver, n['carrier'][v]))
            n['ev_handoff'] = n['ev_handoff'].at[v].set(handoff)
        n['victim_pos'] = jnp.where((n['carrier'] >= 0)[:, None], n['pos'][jnp.maximum(n['carrier'], 0)], s['victim_pos'])
        delivery = (n['carrier'] >= 0) & (jnp.sum(jnp.abs(n['victim_pos'] - s['depot_pos'][0]), axis=1) <= 1) & (s['victim_status'] == 0)
        n['life'] = jnp.maximum(0, s['life'] - ((s['victim_status'] == 0) & ~n['stable'] & ~delivery).astype(jnp.int32))
        expired = (n['life'] == 0) & (s['victim_status'] == 0) & ~delivery
        n['victim_status'] = jnp.where(delivery, 1, jnp.where(expired, 2, s['victim_status']))
        n['carrier'] = jnp.where(n['victim_status'] == 0, n['carrier'], -1)
        n['ev_rescue'] = delivery
        n['ev_loss'] = expired
        n['rescued'] = s['rescued'] + jnp.sum(delivery.astype(jnp.int32))
        n['lost'] = s['lost'] + jnp.sum(expired.astype(jnp.int32))
        n['terminated'] = jnp.all(n['victim_status'] != 0)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = self.refresh(n)
        reward = jnp.sum(delivery.astype(jnp.float32)) - jnp.sum(expired.astype(jnp.float32))
        return n, jnp.full(3, reward, jnp.float32)

    def raw_step(self, key, state, actions):
        n, rewards = self.advance(state, actions)
        n = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), n, state)
        rewards = jnp.where(state['done'], jnp.zeros(3, jnp.float32), rewards)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, rewards, jnp.full(3, n['done'], jnp.bool_), info
