"""Sealed Routes: courier claims, collateral and contingent delivery fees."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 15
    observation_size = 171
    max_steps = 128

    def refresh(self, s):
        s = dict(s)
        held = s['carrier'] >= 0
        s['parcel_pos'] = jnp.where(held[:, None], s['pos'][jnp.maximum(s['carrier'], 0)], s['parcel_pos'])
        s['parcel_active'] = s['status'] != 2
        s['parcel_vis'] = jnp.broadcast_to(s['parcel_active'][None, :], (3, 3))
        s['known'] = s['known'] | (s['carrier'][None, :] == jnp.arange(3)[:, None])
        s['label_vis'] = s['known'] & s['parcel_active'][None, :]
        s['carrying'] = jnp.sum((s['carrier'][None, :] == jnp.arange(3)[:, None]).astype(jnp.int32), axis=1)
        s['message_active'] = s['message_ttl'] > 0
        s['message_vis'] = jnp.broadcast_to(s['message_active'][None, :], (3, 3))
        return s

    def reset(self, key):
        keys = jax.random.split(key, 2)
        tile = jnp.zeros((7, 11), jnp.int32)
        tile = tile.at[0, :].set(1).at[6, :].set(1).at[:, 0].set(1).at[:, 10].set(1)
        tile = tile.at[1:6, 5].set(1).at[3, 5].set(0)
        s = {'t': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False),
             'terminated': jnp.array(False), 'truncated': jnp.array(False), 'tile': tile,
             'pos': jnp.array([[3, 3], [4, 3], [7, 3]], jnp.int32),
             'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
             'credits': jnp.full(3, 4, jnp.int32), 'carrying': jnp.ones(3, jnp.int32),
             'last_action': jnp.zeros(3, jnp.int32), 'message': jnp.full(3, -1, jnp.int32),
             'message_ttl': jnp.zeros(3, jnp.int32), 'message_active': jnp.zeros(3, jnp.bool_),
             'message_vis': jnp.zeros((3, 3), jnp.bool_),
             'parcel_pos': jnp.array([[3, 3], [4, 3], [7, 3]], jnp.int32),
             'parcel_active': jnp.ones(3, jnp.bool_), 'parcel_vis': jnp.ones((3, 3), jnp.bool_),
             'origin': jnp.arange(3, dtype=jnp.int32), 'carrier': jnp.arange(3, dtype=jnp.int32),
             'status': jnp.ones(3, jnp.int32), 'destination': jax.random.randint(keys[0], (3,), 0, 3, dtype=jnp.int32),
             'value': jax.random.randint(keys[1], (3,), 4, 9, dtype=jnp.int32),
             'known': jnp.eye(3, dtype=jnp.bool_), 'label_vis': jnp.eye(3, dtype=jnp.bool_),
             'offer_by': jnp.full(3, -1, jnp.int32), 'fee': jnp.zeros(3, jnp.int32),
             'escrow': jnp.zeros(3, jnp.int32), 'offer_expiry': jnp.zeros(3, jnp.int32),
             'deadline': jnp.zeros(3, jnp.int32), 'contractor': jnp.full(3, -1, jnp.int32),
             'bond': jnp.zeros(3, jnp.int32), 'claim': jnp.full(3, -1, jnp.int32),
             'accepted': jnp.zeros(3, jnp.bool_), 'delivered': jnp.zeros(3, jnp.bool_),
             'failed': jnp.zeros(3, jnp.bool_), 'expired': jnp.zeros(3, jnp.bool_),
             'inspected': jnp.zeros(3, jnp.bool_), 'fees_burned': jnp.array(0, jnp.int32),
             'minted': jnp.array(0, jnp.int32), 'depot_pos': jnp.array([[1, 1], [9, 1], [9, 5]], jnp.int32),
             'depot_active': jnp.ones(3, jnp.bool_), 'depot_vis': jnp.ones((3, 3), jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(3)
        own_held = (s['carrier'] == ids) & (s['status'] == 1)
        available = own_held & (s['offer_by'] < 0) & (s['contractor'] < 0)
        m = jnp.ones((3, 15), jnp.bool_)
        tied = jnp.any((s['carrier'][None, :] == ids[:, None]) & ((s['offer_by'] >= 0) | (s['contractor'] >= 0))[None, :], axis=1)
        m = m.at[:, 5].set(~tied).at[:, 6].set(s['carrying'] > 0)
        m = m.at[:, 7].set(available & (s['credits'] >= 1)).at[:, 8].set(available & (s['credits'] >= 3))
        m = m.at[:, 9].set((s['carrying'] == 0) & (s['credits'] >= 1))
        m = m.at[:, 13].set(s['offer_by'] == ids).at[:, 14].set(s['credits'] > 0)
        return (m & ~s['episode_done']).at[:, 0].set(True)

    def raw_step(self, key, state, actions):
        s = dict(state)
        ids = jnp.arange(3)
        ac = jnp.clip(actions, 0, 14)
        ac = jnp.where((actions >= 0) & (actions < 15) & self.legal_actions(state)[ids, ac], ac, 0)
        s['last_action'] = ac
        s['accepted'] = jnp.zeros(3, jnp.bool_)
        s['delivered'] = jnp.zeros(3, jnp.bool_)
        s['failed'] = jnp.zeros(3, jnp.bool_)
        s['expired'] = jnp.zeros(3, jnp.bool_)
        s['inspected'] = jnp.zeros(3, jnp.bool_)
        s['message_ttl'] = jnp.maximum(0, s['message_ttl'] - 1)
        speaking = (ac >= 10) & (ac <= 12)
        s['message'] = jnp.where(speaking, ac - 10, jnp.where(s['message_ttl'] > 0, s['message'], -1))
        s['message_ttl'] = jnp.where(speaking, 5, s['message_ttl'])
        reward = jnp.zeros(3, jnp.float32)
        tick = state['t'] + 1
        for p in range(3):
            owner = s['origin'][p]
            pending = s['offer_by'][p] >= 0
            cancel = pending & ((tick > s['offer_expiry'][p]) | (ac[owner] == 13))
            late = (s['contractor'][p] >= 0) & (tick > s['deadline'][p])
            worker = jnp.maximum(s['contractor'][p], 0)
            refund = jnp.where(cancel, s['escrow'][p] - 1, 0) + jnp.where(late, s['escrow'][p] + s['bond'][p], 0)
            s['credits'] = s['credits'].at[owner].add(refund)
            reward = reward.at[owner].add(-0.1 * cancel.astype(jnp.float32) + 0.1 * late.astype(jnp.float32))
            reward = reward.at[worker].add(-0.1 * late.astype(jnp.float32))
            s['fees_burned'] = s['fees_burned'] + cancel.astype(jnp.int32)
            s['offer_by'] = s['offer_by'].at[p].set(jnp.where(cancel, -1, s['offer_by'][p]))
            s['contractor'] = s['contractor'].at[p].set(jnp.where(late, -1, s['contractor'][p]))
            s['escrow'] = s['escrow'].at[p].set(jnp.where(cancel | late, 0, s['escrow'][p]))
            s['bond'] = s['bond'].at[p].set(jnp.where(late, 0, s['bond'][p]))
            s['expired'] = s['expired'].at[p].set(cancel)
            s['failed'] = s['failed'].at[p].set(late)
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = jnp.clip(s['pos'] + directions[jnp.where(ac <= 4, ac, 0)], jnp.array([0, 0]), jnp.array([10, 6]))
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        move = (ac >= 1) & (ac <= 4) & (s['tile'][q[:, 1], q[:, 0]] == 0) & ~occupied & ~conflict
        s['pos'] = jnp.where(move[:, None], q, s['pos'])
        s = self.refresh(s)
        for k in range(3):
            i = (state['t'] + k) % 3
            held = s['carrier'] == i
            holding = jnp.any(held)
            p = jnp.argmax(held.astype(jnp.int32))
            floor = (s['status'] == 0) & (jnp.sum(jnp.abs(s['parcel_pos'] - s['pos'][i]), axis=1) <= 1)
            f = jnp.argmax(floor.astype(jnp.int32))
            drop = (ac[i] == 5) & holding & (s['contractor'][p] < 0) & (s['offer_by'][p] < 0)
            pickup = (ac[i] == 5) & ~holding & jnp.any(floor)
            s['carrier'] = s['carrier'].at[p].set(jnp.where(drop, -1, s['carrier'][p]))
            s['status'] = s['status'].at[p].set(jnp.where(drop, 0, s['status'][p]))
            s['parcel_pos'] = s['parcel_pos'].at[p].set(jnp.where(drop, s['pos'][i], s['parcel_pos'][p]))
            s['carrier'] = s['carrier'].at[f].set(jnp.where(pickup, i, s['carrier'][f]))
            s['status'] = s['status'].at[f].set(jnp.where(pickup, 1, s['status'][f]))
            post = ((ac[i] == 7) | (ac[i] == 8)) & (s['carrier'][i] == i) & (s['offer_by'][i] < 0) & (s['contractor'][i] < 0)
            quote = jnp.where(ac[i] == 8, 3, 1)
            post = post & (s['credits'][i] >= quote)
            s['credits'] = s['credits'].at[i].add(-quote * post.astype(jnp.int32))
            s['offer_by'] = s['offer_by'].at[i].set(jnp.where(post, i, s['offer_by'][i]))
            s['fee'] = s['fee'].at[i].set(jnp.where(post, quote, s['fee'][i]))
            s['escrow'] = s['escrow'].at[i].set(jnp.where(post, quote, s['escrow'][i]))
            s['offer_expiry'] = s['offer_expiry'].at[i].set(jnp.where(post, tick + 5, s['offer_expiry'][i]))
            s['deadline'] = s['deadline'].at[i].set(jnp.where(post, tick + 18, s['deadline'][i]))
            s['claim'] = s['claim'].at[i].set(jnp.where(post, s['message'][i], s['claim'][i]))
            neighbor = jnp.sum(jnp.abs(s['parcel_pos'] - s['pos'][i]), axis=1) <= 1
            offers = neighbor & (s['offer_by'] >= 0) & (state['offer_by'] >= 0) & (s['origin'] != i)
            p = jnp.argmin(jnp.where(offers, (ids - state['t']) % 3, 99))
            accept = (ac[i] == 9) & ~jnp.any(s['carrier'] == i) & jnp.any(offers) & (s['credits'][i] >= 1)
            s['credits'] = s['credits'].at[i].add(-accept.astype(jnp.int32))
            s['carrier'] = s['carrier'].at[p].set(jnp.where(accept, i, s['carrier'][p]))
            s['contractor'] = s['contractor'].at[p].set(jnp.where(accept, i, s['contractor'][p]))
            s['bond'] = s['bond'].at[p].set(jnp.where(accept, 1, s['bond'][p]))
            s['offer_by'] = s['offer_by'].at[p].set(jnp.where(accept, -1, s['offer_by'][p]))
            s['accepted'] = s['accepted'].at[p].set(s['accepted'][p] | accept)
            inspectable = neighbor & (s['status'] != 2) & ~s['known'][i]
            p = jnp.argmin(jnp.where(inspectable, (ids - state['t']) % 3, 99))
            inspect = (ac[i] == 14) & jnp.any(inspectable) & (s['credits'][i] > 0)
            s['known'] = s['known'].at[i, p].set(s['known'][i, p] | inspect)
            s['credits'] = s['credits'].at[i].add(-inspect.astype(jnp.int32))
            s['fees_burned'] = s['fees_burned'] + inspect.astype(jnp.int32)
            reward = reward.at[i].add(-0.1 * inspect.astype(jnp.float32))
            s['inspected'] = s['inspected'].at[i].set(inspect)
            held = s['carrier'] == i
            p = jnp.argmax(held.astype(jnp.int32))
            at_depot = jnp.all(s['pos'][i] == s['depot_pos'][s['destination'][p]])
            deliver = (ac[i] == 6) & jnp.any(held) & at_depot
            contracted = s['contractor'][p] == i
            payout = jnp.where(contracted, s['escrow'][p], 0) * deliver.astype(jnp.int32)
            returned_bond = jnp.where(contracted, s['bond'][p], 0) * deliver.astype(jnp.int32)
            owner = s['origin'][p]
            value = s['value'][p] * deliver.astype(jnp.int32)
            refund_offer = jnp.where(s['offer_by'][p] >= 0, s['escrow'][p], 0) * deliver.astype(jnp.int32)
            s['credits'] = s['credits'].at[owner].add(value + refund_offer).at[i].add(payout + returned_bond)
            s['minted'] = s['minted'] + value
            reward = reward.at[owner].add(0.1 * (value - payout).astype(jnp.float32)).at[i].add(0.1 * payout.astype(jnp.float32))
            s['status'] = s['status'].at[p].set(jnp.where(deliver, 2, s['status'][p]))
            s['carrier'] = s['carrier'].at[p].set(jnp.where(deliver, -1, s['carrier'][p]))
            s['contractor'] = s['contractor'].at[p].set(jnp.where(deliver, -1, s['contractor'][p]))
            s['offer_by'] = s['offer_by'].at[p].set(jnp.where(deliver, -1, s['offer_by'][p]))
            s['escrow'] = s['escrow'].at[p].set(jnp.where(deliver, 0, s['escrow'][p]))
            s['bond'] = s['bond'].at[p].set(jnp.where(deliver, 0, s['bond'][p]))
            s['parcel_pos'] = s['parcel_pos'].at[p].set(jnp.where(deliver, s['pos'][i], s['parcel_pos'][p]))
            s['delivered'] = s['delivered'].at[p].set(s['delivered'][p] | deliver)
        s['t'] = tick
        s['terminated'] = jnp.all(s['status'] == 2)
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['episode_done'] = s['terminated'] | s['truncated']
        s = self.refresh(s)
        s = jax.tree.map(lambda new, old: jnp.where(state['episode_done'], old, new), s, state)
        reward = jnp.where(state['episode_done'], jnp.zeros(3, jnp.float32), reward)
        return self.observe(s), s, reward, jnp.full(3, s['episode_done']), {'done_all': s['episode_done'], 'terminated': s['terminated'], 'truncated': s['truncated']}

    def observe(self, s):
        common = jnp.concatenate((s['tile'].reshape(-1), s['pos'].reshape(-1), s['credits'], s['last_action'],
            s['message'], s['message_ttl'], s['parcel_pos'].reshape(-1), s['origin'], s['carrier'], s['status'],
            s['offer_by'], s['fee'], s['escrow'], s['offer_expiry'], s['deadline'], s['contractor'], s['bond'], s['claim'],
            s['accepted'], s['delivered'], s['failed'], s['expired'], s['inspected'], s['depot_pos'].reshape(-1),
            jnp.array([s['t'], s['episode_done'], s['fees_burned'], s['minted']], jnp.int32))).astype(jnp.float32)
        secret = jnp.stack((s['destination'], s['value']), axis=1)
        private = jnp.where(s['known'][:, :, None], secret[None, :, :], -1).reshape(3, 6)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(common, (3, common.shape[0])), s['known'].astype(jnp.float32), private.astype(jnp.float32)), axis=1)
