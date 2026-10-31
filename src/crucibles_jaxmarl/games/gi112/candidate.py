"""Original three-specialist workshop with conserved goods and shared heat."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 112
    num_actions = 11
    max_steps = 96

    def moves(self):
        return jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def inventory(self, carriers):
        return jnp.max(jnp.where(carriers[None, :] == jnp.arange(3)[:, None], jnp.arange(2)[None, :], -1), axis=1).astype(jnp.int32)

    def refresh(self, s):
        s['carry'] = self.inventory(s['piece_carrier'])
        p = s['agent_pos']
        s['agent_vis'] = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=-1) <= 3
        s['piece_active'] = ~s['piece_delivered']
        s['piece_vis'] = (jnp.sum(jnp.abs(p[:, None, :] - s['piece_pos'][None, :, :]), axis=-1) <= 3) & s['piece_active'][None, :]
        s['station_vis'] = jnp.sum(jnp.abs(p[:, None, :] - s['station_pos'][None, :, :]), axis=-1) <= 3
        s['tank_vis'] = jnp.sum(jnp.abs(p[:, None, :] - s['tank_pos'][None, :, :]), axis=-1) <= 3
        return s

    def reset(self, key):
        xx = jnp.arange(11)[None, :]
        yy = jnp.arange(7)[:, None]
        walls = (xx == 0) | (xx == 10) | (yy == 0) | (yy == 6) | (((xx == 4) | (xx == 7)) & (yy != 3))
        tiles = walls.astype(jnp.int32).at[2, 2].set(2).at[2, 5].set(3).at[2, 8].set(4)
        tiles = tiles.at[4, 5].set(5).at[4, 8].set(6)
        s = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'map': tiles,
            'agent_pos': jnp.array([[2, 3], [5, 3], [8, 3]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_), 'agent_vis': jnp.zeros((3, 3), jnp.bool_),
            'carry': jnp.full((3,), -1, jnp.int32), 'role': jnp.arange(3, dtype=jnp.int32),
            'piece_pos': jnp.array([[2, 4], [3, 4]], jnp.int32),
            'piece_active': jnp.ones((2,), jnp.bool_), 'piece_vis': jnp.zeros((3, 2), jnp.bool_),
            'piece_carrier': jnp.full((2,), -1, jnp.int32), 'piece_stage': jnp.zeros((2,), jnp.int32),
            'piece_delivered': jnp.zeros((2,), jnp.bool_),
            'station_pos': jnp.array([[2, 2], [5, 2], [8, 2]], jnp.int32),
            'station_active': jnp.ones((3,), jnp.bool_), 'station_vis': jnp.zeros((3, 3), jnp.bool_),
            'station_heat': jnp.zeros((3,), jnp.int32), 'station_role': jnp.arange(3, dtype=jnp.int32),
            'heat_vented': jnp.zeros((3,), jnp.int32),
            'tank_pos': jnp.array([[5, 4]], jnp.int32),
            'tank_active': jnp.ones((1,), jnp.bool_), 'tank_vis': jnp.zeros((3, 1), jnp.bool_),
            'coolant': jnp.array([6], jnp.int32), 'coolant_spent': jnp.array([0], jnp.int32),
            'ship_pos': jnp.array([[8, 4]], jnp.int32),
            'ship_active': jnp.ones((1,), jnp.bool_), 'ship_vis': jnp.ones((3, 1), jnp.bool_),
            'shipped': jnp.array([0], jnp.int32), 'shipped_step': jnp.array(0, jnp.int32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def move_options(self, s):
        q = s['agent_pos'][:, None, :] + self.moves()[None, :, :]
        safe = jnp.clip(q, jnp.array([0, 0]), jnp.array([10, 6]))
        tile = s['map'][safe[:, :, 1], safe[:, :, 0]]
        walkable = (tile == 0) | (tile == 6)
        occupied = jnp.any(jnp.all(q[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=-1)
                           & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=-1)
        return q, walkable & jnp.all(q == safe, axis=-1) & ~occupied

    def legal_actions(self, s):
        q, clear = self.move_options(s)
        holding = s['carry'] >= 0
        item = jnp.maximum(s['carry'], 0)
        distance = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['piece_pos'][None, :, :]), axis=-1)
        floor_item = (s['piece_carrier'] < 0) & ~s['piece_delivered']
        pickup = jnp.any((distance <= 1) & floor_item[None, :], axis=1)
        drop = ~jnp.any((distance == 0) & floor_item[None, :], axis=1)
        near_station = jnp.sum(jnp.abs(s['agent_pos'] - s['station_pos']), axis=1) <= 1
        process = holding & (s['piece_stage'][item] == s['role']) & near_station & (s['station_heat'] <= jnp.array([6, 5, 7], jnp.int32))
        near_tank = jnp.sum(jnp.abs(s['agent_pos'] - s['tank_pos'][0]), axis=1) <= 1
        adjacent = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=-1) == 1
        give = holding & jnp.any(adjacent & ~holding[None, :], axis=1)
        receive = ~holding & jnp.any(adjacent & holding[None, :], axis=1)
        dispatch = holding & (s['piece_stage'][item] == 3) & (jnp.sum(jnp.abs(s['agent_pos'] - s['ship_pos'][0]), axis=1) <= 1)
        mask = jnp.zeros((3, 11), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(clear[:, 1:5])
        mask = mask.at[:, 5].set(jnp.where(holding, drop, pickup)).at[:, 6].set(process)
        mask = mask.at[:, 7].set(near_tank & (s['coolant'][0] > 0))
        mask = mask.at[:, 8].set(give).at[:, 9].set(receive).at[:, 10].set(dispatch)
        return jnp.where(s['done'], jnp.broadcast_to(jnp.arange(11)[None, :] == 0, (3, 11)), mask)

    def observe(self, s):
        av = s['agent_vis']
        agents = jnp.concatenate((s['agent_pos'].astype(jnp.float32) / 16,
                                  ((s['carry'] + 1).astype(jnp.float32) / 4)[:, None]), axis=1)
        agent_fields = jnp.concatenate((av[:, :, None].astype(jnp.float32),
                                        jnp.where(av[:, :, None], agents[None, :, :], 0)), axis=-1).reshape(3, 12)
        pv = s['piece_vis']
        pieces = jnp.concatenate((s['piece_pos'].astype(jnp.float32) / 16,
                                  ((s['piece_carrier'] + 1).astype(jnp.float32) / 4)[:, None],
                                  (s['piece_stage'].astype(jnp.float32) / 4)[:, None]), axis=1)
        piece_fields = jnp.concatenate((pv[:, :, None].astype(jnp.float32),
                                        jnp.where(pv[:, :, None], pieces[None, :, :], 0)), axis=-1).reshape(3, 10)
        sv = s['station_vis']
        station_fields = jnp.stack((sv.astype(jnp.float32), jnp.where(sv, s['station_heat'][None, :].astype(jnp.float32) / 16, 0)), axis=-1).reshape(3, 6)
        tv = s['tank_vis']
        tank_fields = jnp.concatenate((tv.astype(jnp.float32), jnp.where(tv, s['coolant'][None, :].astype(jnp.float32) / 8, 0)), axis=1)
        public = jnp.array([s['t'].astype(jnp.float32) / 128, s['shipped'][0].astype(jnp.float32) / 2], jnp.float32)
        world = jnp.broadcast_to(s['map'].astype(jnp.float32).reshape(1, 77) / 8, (3, 77))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, 2)),
                                agent_fields, piece_fields, station_fields, tank_fields, world), axis=1)

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions, 0, 10)
        mask = self.legal_actions(state)
        a = jnp.where((actions >= 0) & (actions < 11) & mask[jnp.arange(3), clipped], clipped, 0)
        s = dict(state)
        q = s['agent_pos'] + self.moves()[jnp.clip(a, 0, 4)]
        moving = (a >= 1) & (a <= 4)
        q = jnp.where(moving[:, None], q, s['agent_pos'])
        clash = jnp.any(jnp.all(q[:, None, :] == q[None, :, :], axis=-1) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        pos = jnp.where((moving & ~clash)[:, None], q, s['agent_pos'])
        carrier = s['piece_carrier']
        pieces = jnp.where((carrier >= 0)[:, None], pos[jnp.maximum(carrier, 0)], s['piece_pos'])
        # Pick/drop is an atomic ownership change; processing never creates another piece.
        for k in range(3):
            i = jnp.mod(k + s['t'], 3)
            carry = self.inventory(carrier)
            holding = carry[i] >= 0
            p = jnp.maximum(carry[i], 0)
            floor = (carrier < 0) & ~s['piece_delivered']
            empty_floor = ~jnp.any(floor & jnp.all(pieces == pos[i], axis=1))
            drop = (a[i] == 5) & holding & empty_floor
            carrier = carrier.at[p].set(jnp.where(drop, -1, carrier[p]))
            pieces = pieces.at[p].set(jnp.where(drop, pos[i], pieces[p]))
            available = (carrier < 0) & ~s['piece_delivered'] & (jnp.sum(jnp.abs(pieces - pos[i]), axis=1) <= 1)
            picked = jnp.argmax(available).astype(jnp.int32)
            take = (a[i] == 5) & ~holding & jnp.any(available)
            carrier = carrier.at[picked].set(jnp.where(take, i, carrier[picked]))
        # A transfer requires adjacent donor OFFER and empty recipient ACCEPT together.
        priority = jnp.mod(jnp.arange(3) - jnp.mod(s['t'], 3), 3)
        for k in range(3):
            i = jnp.mod(k + s['t'], 3)
            carry = self.inventory(carrier)
            recipients = (a == 9) & (carry < 0) & (jnp.sum(jnp.abs(pos - pos[i]), axis=1) == 1)
            j = jnp.argmin(jnp.where(recipients, priority, 99)).astype(jnp.int32)
            p = jnp.maximum(carry[i], 0)
            transfer = (a[i] == 8) & (carry[i] >= 0) & jnp.any(recipients)
            carrier = carrier.at[p].set(jnp.where(transfer, j, carrier[p]))
        pieces = jnp.where((carrier >= 0)[:, None], pos[jnp.maximum(carrier, 0)], pieces)
        # One cooling valve serves the whole manifold and consumes one real unit.
        heat = jnp.maximum(s['station_heat'] - 1, 0)
        coolers = (a == 7) & (jnp.sum(jnp.abs(pos - s['tank_pos'][0]), axis=1) <= 1)
        cool = jnp.any(coolers) & (s['coolant'][0] > 0)
        heat = jnp.maximum(heat - 3 * cool.astype(jnp.int32), 0)
        s['coolant'] = s['coolant'] - cool.astype(jnp.int32)
        s['coolant_spent'] = s['coolant_spent'] + cool.astype(jnp.int32)
        stages = s['piece_stage']
        vented = jnp.zeros((3,), jnp.int32)
        carry = self.inventory(carrier)
        # Shared heat means one specialist can close another specialist's work window.
        for k in range(3):
            i = jnp.mod(k + s['t'], 3)
            p = jnp.maximum(carry[i], 0)
            near = jnp.sum(jnp.abs(pos[i] - s['station_pos'][i])) <= 1
            threshold = jnp.array([6, 5, 7], jnp.int32)[i]
            work = (a[i] == 6) & (carry[i] >= 0) & (stages[p] == i) & near & (heat[i] <= threshold)
            stages = stages.at[p].set(stages[p] + work.astype(jnp.int32))
            addition = jnp.where(jnp.arange(3) == i, 4, 2) * work.astype(jnp.int32)
            vented = vented + jnp.maximum(heat + addition - 12, 0)
            heat = jnp.minimum(heat + addition, 12).astype(jnp.int32)
        delivered = s['piece_delivered']
        for i in range(3):
            p = jnp.maximum(carry[i], 0)
            dispatch = (a[i] == 10) & (carry[i] >= 0) & (stages[p] == 3)
            dispatch = dispatch & (jnp.sum(jnp.abs(pos[i] - s['ship_pos'][0])) <= 1)
            delivered = delivered.at[p].set(delivered[p] | dispatch)
            carrier = carrier.at[p].set(jnp.where(dispatch, -1, carrier[p]))
            pieces = pieces.at[p].set(jnp.where(dispatch, s['ship_pos'][0], pieces[p]))
        shipped_step = jnp.sum(delivered & ~s['piece_delivered'], dtype=jnp.int32)
        s['agent_pos'] = pos
        s['piece_pos'] = pieces
        s['piece_carrier'] = carrier
        s['piece_stage'] = stages
        s['piece_delivered'] = delivered
        s['station_heat'] = heat
        s['heat_vented'] = vented
        s['shipped'] = jnp.reshape(jnp.sum(delivered, dtype=jnp.int32), (1,))
        s['shipped_step'] = shipped_step
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.all(delivered)
        s['truncated'] = (s['t'] >= 96) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        s = self.refresh(s)
        reward = shipped_step.astype(jnp.float32) - (a != 0).astype(jnp.float32) / 128
        s = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), s, state)
        reward = jnp.where(state['done'], jnp.zeros((3,), jnp.float32), reward)
        info = {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
        return self.observe(s), s, reward, jnp.full((3,), s['done'], jnp.bool_), info
