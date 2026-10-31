"""Original courier board: bounded escrow, one bridge slot, private orders."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 115
    num_actions = 13
    max_steps = 96

    def moves(self):
        return jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def inventory(self, carriers):
        return jnp.max(jnp.where(carriers[None, :] == jnp.arange(3)[:, None], jnp.arange(3)[None, :], -1), axis=1).astype(jnp.int32)

    def auction_winner(self, s):
        best = jnp.max(s['escrow'])
        priority = jnp.mod(jnp.arange(3) - jnp.mod(s['t'] // 2, 3), 3)
        winner = jnp.argmin(jnp.where(s['escrow'] == best, priority, 99)).astype(jnp.int32)
        return jnp.where(best > 0, winner, -1).astype(jnp.int32)

    def refresh(self, s):
        s['carry'] = self.inventory(s['parcel_carrier'])
        p = s['agent_pos']
        s['agent_vis'] = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=-1) <= 3
        s['parcel_active'] = ~s['delivered']
        s['parcel_vis'] = (jnp.sum(jnp.abs(p[:, None, :] - s['parcel_pos'][None, :, :]), axis=-1) <= 3) & s['parcel_active'][None, :]
        prepared = jnp.where(jnp.mod(s['t'], 2) == 1, self.auction_winner(s), -1)
        s['permit'] = jnp.reshape(prepared, (1,)).astype(jnp.int32)
        return s

    def reset(self, key):
        xx = jnp.arange(9)[None, :]
        yy = jnp.arange(7)[:, None]
        walls = (xx == 0) | (xx == 8) | (yy == 0) | (yy == 6) | ((xx == 4) & (yy != 3))
        tiles = walls.astype(jnp.int32).at[1, 1].set(2).at[1, 7].set(2).at[5, 7].set(2)
        tiles = tiles.at[2, 3].set(3).at[4, 2].set(3).at[3, 6].set(3).at[3, 4].set(4)
        s = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'map': tiles,
            'agent_pos': jnp.array([[3, 3], [2, 3], [5, 3]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_), 'agent_vis': jnp.zeros((3, 3), jnp.bool_),
            'carry': jnp.array([2, 0, 1], jnp.int32),
            'parcel_pos': jnp.array([[2, 3], [5, 3], [3, 3]], jnp.int32),
            'parcel_active': jnp.ones((3,), jnp.bool_), 'parcel_vis': jnp.zeros((3, 3), jnp.bool_),
            'parcel_carrier': jnp.array([1, 2, 0], jnp.int32), 'parcel_owner': jnp.arange(3, dtype=jnp.int32),
            'need': jax.random.permutation(key, 3).astype(jnp.int32),
            'announced': jnp.full((3,), -1, jnp.int32), 'delivered': jnp.zeros((3,), jnp.bool_),
            'credits': jnp.full((3,), 6, jnp.int32), 'escrow': jnp.zeros((3,), jnp.int32),
            'spent': jnp.zeros((3,), jnp.int32), 'paid_step': jnp.zeros((3,), jnp.int32),
            'settled': jnp.array(False), 'delivered_step': jnp.zeros((3,), jnp.bool_),
            'board_pos': jnp.array([[3, 2], [2, 4], [6, 3]], jnp.int32),
            'board_active': jnp.ones((3,), jnp.bool_), 'board_vis': jnp.ones((3, 3), jnp.bool_),
            'depot_pos': jnp.array([[1, 1], [7, 1], [7, 5]], jnp.int32),
            'depot_active': jnp.ones((3,), jnp.bool_), 'depot_vis': jnp.ones((3, 3), jnp.bool_),
            'depot_id': jnp.arange(3, dtype=jnp.int32),
            'bridge_pos': jnp.array([[4, 3]], jnp.int32),
            'bridge_active': jnp.ones((1,), jnp.bool_), 'bridge_vis': jnp.ones((3, 1), jnp.bool_),
            'permit': jnp.array([-1], jnp.int32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def move_options(self, s):
        q = s['agent_pos'][:, None, :] + self.moves()[None, :, :]
        safe = jnp.clip(q, jnp.array([0, 0]), jnp.array([8, 6]))
        terrain = (s['map'][safe[:, :, 1], safe[:, :, 0]] != 1) & jnp.all(q == safe, axis=-1)
        occupied = jnp.any(jnp.all(q[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=-1)
                           & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=-1)
        entering = jnp.all(q == s['bridge_pos'][0], axis=-1) & jnp.any(s['agent_pos'] != s['bridge_pos'][0], axis=-1)[:, None]
        permit = (s['permit'][0] == jnp.arange(3)) & (jnp.mod(s['t'], 2) == 1)
        return q, terrain & ~occupied & (~entering | permit[:, None])

    def legal_actions(self, s):
        q, clear = self.move_options(s)
        holding = s['carry'] >= 0
        distance = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['parcel_pos'][None, :, :]), axis=-1)
        floor = (s['parcel_carrier'] < 0) & ~s['delivered']
        pickup = jnp.any((distance <= 1) & floor[None, :], axis=1)
        drop = ~jnp.any((distance == 0) & floor[None, :], axis=1)
        adjacent = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=-1) == 1
        queue = (s['agent_pos'][:, 1] == 3) & ((s['agent_pos'][:, 0] == 3) | (s['agent_pos'][:, 0] == 5))
        bid_phase = jnp.mod(s['t'], 2) == 0
        at_depot = jnp.any(jnp.all(s['agent_pos'][:, None, :] == s['depot_pos'][None, :, :], axis=-1), axis=1)
        at_board = jnp.sum(jnp.abs(s['agent_pos'] - s['board_pos']), axis=1) <= 1
        mask = jnp.zeros((3, 13), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(clear[:, 1:5])
        mask = mask.at[:, 5].set(queue & bid_phase & (s['credits'] >= 1))
        mask = mask.at[:, 6].set(queue & bid_phase & (s['credits'] >= 2))
        mask = mask.at[:, 7].set(jnp.where(holding, drop, pickup))
        mask = mask.at[:, 8].set(holding & jnp.any(adjacent & ~holding[None, :], axis=1))
        mask = mask.at[:, 9].set(~holding & jnp.any(adjacent & holding[None, :], axis=1))
        # Delivery legality depends only on being at any depot, never on a hidden order.
        mask = mask.at[:, 10].set(holding & at_depot)
        mask = mask.at[:, 11].set(holding & jnp.any(adjacent & holding[None, :], axis=1))
        mask = mask.at[:, 12].set(at_board & (s['announced'] < 0))
        return jnp.where(s['done'], jnp.broadcast_to(jnp.arange(13)[None, :] == 0, (3, 13)), mask)

    def observe(self, s):
        av = s['agent_vis']
        agents = jnp.concatenate((s['agent_pos'].astype(jnp.float32) / 8,
                                  ((s['carry'] + 1).astype(jnp.float32) / 4)[:, None]), axis=1)
        agent_fields = jnp.concatenate((av[:, :, None].astype(jnp.float32),
                                        jnp.where(av[:, :, None], agents[None, :, :], 0)), axis=-1).reshape(3, 12)
        pv = s['parcel_vis']
        parcels = jnp.concatenate((s['parcel_pos'].astype(jnp.float32) / 8,
                                   (s['parcel_owner'].astype(jnp.float32) / 4)[:, None],
                                   ((s['parcel_carrier'] + 1).astype(jnp.float32) / 4)[:, None],
                                   s['delivered'].astype(jnp.float32)[:, None]), axis=1)
        parcel_fields = jnp.concatenate((pv[:, :, None].astype(jnp.float32),
                                         jnp.where(pv[:, :, None], parcels[None, :, :], 0)), axis=-1).reshape(3, 18)
        accounts = jnp.stack((s['credits'].astype(jnp.float32) / 8, s['escrow'].astype(jnp.float32) / 2,
                              s['spent'].astype(jnp.float32) / 8), axis=1).reshape(-1)
        public = jnp.concatenate((accounts, (s['announced'] + 1).astype(jnp.float32) / 4,
                                  (s['permit'] + 1).astype(jnp.float32) / 4, s['delivered'].astype(jnp.float32)))
        clock = jnp.array([s['t'].astype(jnp.float32) / 128, jnp.mod(s['t'], 2).astype(jnp.float32)], jnp.float32)
        world = jnp.broadcast_to(s['map'].astype(jnp.float32).reshape(1, 63) / 4, (3, 63))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(clock, (3, 2)),
                                s['need'].astype(jnp.float32)[:, None] / 4, jnp.broadcast_to(public, (3, 16)),
                                agent_fields, parcel_fields, world), axis=1)

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions, 0, 12)
        mask = self.legal_actions(state)
        a = jnp.where((actions >= 0) & (actions < 13) & mask[jnp.arange(3), clipped], clipped, 0)
        s = dict(state)
        settlement = jnp.mod(s['t'], 2) == 1
        winner = s['permit'][0]
        bids = jnp.where(a == 5, 1, jnp.where(a == 6, 2, 0)).astype(jnp.int32)
        paid = jnp.where(settlement & (jnp.arange(3) == winner), s['escrow'], 0).astype(jnp.int32)
        refund = jnp.where(settlement, s['escrow'] - paid, 0)
        s['credits'] = s['credits'] - bids + refund
        s['spent'] = s['spent'] + paid
        s['escrow'] = jnp.where(settlement, jnp.zeros((3,), jnp.int32), bids)
        s['paid_step'] = paid
        s['settled'] = settlement
        # Payment is enforced even if the winner elects not to use the physical slot.
        q = s['agent_pos'] + self.moves()[jnp.clip(a, 0, 4)]
        moving = (a >= 1) & (a <= 4)
        q = jnp.where(moving[:, None], q, s['agent_pos'])
        clash = jnp.any(jnp.all(q[:, None, :] == q[None, :, :], axis=-1) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        pos = jnp.where((moving & ~clash)[:, None], q, s['agent_pos'])
        carrier = s['parcel_carrier']
        parcels = jnp.where((carrier >= 0)[:, None], pos[jnp.maximum(carrier, 0)], s['parcel_pos'])
        priority = jnp.mod(jnp.arange(3) - jnp.mod(s['t'], 3), 3)
        for k in range(3):
            i = jnp.mod(k + s['t'], 3)
            carry = self.inventory(carrier)
            holding = carry[i] >= 0
            p = jnp.maximum(carry[i], 0)
            floor = (carrier < 0) & ~s['delivered']
            empty_floor = ~jnp.any(floor & jnp.all(parcels == pos[i], axis=1))
            drop = (a[i] == 7) & holding & empty_floor
            carrier = carrier.at[p].set(jnp.where(drop, -1, carrier[p]))
            parcels = parcels.at[p].set(jnp.where(drop, pos[i], parcels[p]))
            available = (carrier < 0) & ~s['delivered'] & (jnp.sum(jnp.abs(parcels - pos[i]), axis=1) <= 1)
            picked = jnp.argmax(available).astype(jnp.int32)
            take = (a[i] == 7) & ~holding & jnp.any(available)
            carrier = carrier.at[picked].set(jnp.where(take, i, carrier[picked]))
        # Hand-off requires two complementary actions, with a single recipient slot.
        for k in range(3):
            i = jnp.mod(k + s['t'], 3)
            carry = self.inventory(carrier)
            recipients = (a == 9) & (carry < 0) & (jnp.sum(jnp.abs(pos - pos[i]), axis=1) == 1)
            j = jnp.argmin(jnp.where(recipients, priority, 99)).astype(jnp.int32)
            p = jnp.maximum(carry[i], 0)
            transfer = (a[i] == 8) & (carry[i] >= 0) & jnp.any(recipients)
            carrier = carrier.at[p].set(jnp.where(transfer, j, carrier[p]))
        # Bilateral swap is atomic; each courier participates in at most one pair.
        used = jnp.zeros((3,), jnp.bool_)
        for k in range(3):
            i = jnp.mod(k + s['t'], 3)
            carry = self.inventory(carrier)
            partners = (a == 11) & (carry >= 0) & ~used & (jnp.sum(jnp.abs(pos - pos[i]), axis=1) == 1)
            j = jnp.argmin(jnp.where(partners, priority, 99)).astype(jnp.int32)
            swap = (a[i] == 11) & (carry[i] >= 0) & ~used[i] & jnp.any(partners)
            p = jnp.maximum(carry[i], 0)
            qid = jnp.maximum(carry[j], 0)
            carrier = carrier.at[p].set(jnp.where(swap, j, carrier[p]))
            carrier = carrier.at[qid].set(jnp.where(swap, i, carrier[qid]))
            used = used.at[i].set(used[i] | swap).at[j].set(used[j] | swap)
        parcels = jnp.where((carrier >= 0)[:, None], pos[jnp.maximum(carrier, 0)], parcels)
        s['announced'] = jnp.where(a == 12, s['need'], s['announced']).astype(jnp.int32)
        carry = self.inventory(carrier)
        delivered = s['delivered']
        reward = -(a != 0).astype(jnp.float32) / 128 - paid.astype(jnp.float32) / 16
        # Unknown destinations affect success, but do not leak through the action mask.
        for i in range(3):
            p = jnp.maximum(carry[i], 0)
            correct_place = jnp.all(pos[i] == s['depot_pos'][s['need'][p]])
            success = (a[i] == 10) & (carry[i] >= 0) & correct_place & ~delivered[p]
            reward = reward.at[p].add(success.astype(jnp.float32))
            reward = reward.at[i].add(success.astype(jnp.float32) / 4)
            delivered = delivered.at[p].set(delivered[p] | success)
            carrier = carrier.at[p].set(jnp.where(success, -1, carrier[p]))
            parcels = parcels.at[p].set(jnp.where(success, pos[i], parcels[p]))
        s['agent_pos'] = pos
        s['parcel_pos'] = parcels
        s['parcel_carrier'] = carrier
        s['delivered_step'] = delivered & ~s['delivered']
        s['delivered'] = delivered
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.all(delivered)
        s['truncated'] = (s['t'] >= 96) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        # Terminal cancellation returns deposits for slots that can no longer be used.
        s['credits'] = s['credits'] + jnp.where(s['done'], s['escrow'], 0)
        s['escrow'] = jnp.where(s['done'], jnp.zeros((3,), jnp.int32), s['escrow'])
        s = self.refresh(s)
        s = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), s, state)
        reward = jnp.where(state['done'], jnp.zeros((3,), jnp.float32), reward)
        info = {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
        return self.observe(s), s, reward, jnp.full((3,), s['done'], jnp.bool_), info
