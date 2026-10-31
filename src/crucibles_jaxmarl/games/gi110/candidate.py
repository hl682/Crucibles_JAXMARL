"""Original four-tug convoy. Authored source; no execution evidence."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 142
    num_actions = 12
    max_steps = 64

    def moves(self):
        return jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def reset(self, key):
        xx = jnp.arange(11)[None, :]
        yy = jnp.arange(9)[:, None]
        walls = (xx == 0) | (xx == 10) | (yy == 0) | (yy == 8) | ((xx == 5) & (yy == 4))
        tiles = walls.astype(jnp.int32)
        tiles = tiles.at[2, 4].set(2).at[6, 4].set(2)
        tiles = tiles.at[2, 8].set(3).at[6, 8].set(3)
        tiles = tiles.at[2, 5].set(4).at[6, 5].set(4)
        s = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'map': tiles,
            'agent_pos': jnp.array([[3, 1], [3, 3], [3, 5], [3, 7]], jnp.int32),
            'agent_active': jnp.ones((4,), jnp.bool_), 'agent_vis': jnp.ones((4, 4), jnp.bool_),
            'attached': jnp.full((4,), -1, jnp.int32),
            'barge_pos': jnp.array([[3, 2], [3, 6]], jnp.int32),
            'barge_active': jnp.ones((2,), jnp.bool_), 'barge_vis': jnp.ones((4, 2), jnp.bool_),
            'delivered': jnp.zeros((2,), jnp.bool_),
            'gate_pos': jnp.array([[5, 2], [5, 6]], jnp.int32),
            'gate_active': jnp.ones((2,), jnp.bool_), 'gate_vis': jnp.ones((4, 2), jnp.bool_),
            'gate_timer': jnp.zeros((2,), jnp.int32),
            'gate_flow': jnp.array([[1, 0], [-1, 0]], jnp.int32),
            'control_pos': jnp.array([[4, 2], [4, 6]], jnp.int32),
            'control_active': jnp.ones((2,), jnp.bool_), 'control_vis': jnp.ones((4, 2), jnp.bool_),
            'dock_pos': jnp.array([[8, 2], [8, 6]], jnp.int32),
            'dock_active': jnp.ones((2,), jnp.bool_), 'dock_vis': jnp.ones((4, 2), jnp.bool_),
            'tow_moved': jnp.zeros((2,), jnp.bool_), 'tow_blocked': jnp.zeros((2,), jnp.bool_),
            'delivered_step': jnp.zeros((2,), jnp.bool_), 'gate_changed': jnp.zeros((2,), jnp.bool_)
        }
        return self.observe(s), s

    def free_targets(self, s):
        p = s['agent_pos'][:, None, :] + self.moves()[None, :, :]
        safe = jnp.clip(p, jnp.array([0, 0]), jnp.array([10, 8]))
        terrain = s['map'][safe[:, :, 1], safe[:, :, 0]] != 1
        inside = jnp.all(p == safe, axis=-1)
        other_agent = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=-1)
                              & ~jnp.eye(4, dtype=jnp.bool_)[:, None, :], axis=-1)
        cargo = jnp.any(jnp.all(p[:, :, None, :] == s['barge_pos'][None, None, :, :], axis=-1), axis=-1)
        return p, terrain & inside & ~other_agent & ~cargo

    def legal_actions(self, s):
        p, free = self.free_targets(s)
        distance = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['barge_pos'][None, :, :]), axis=-1)
        can_attach = jnp.any((distance == 1) & ~s['delivered'][None, :], axis=1) & (s['attached'] < 0)
        controls = jnp.any(jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['control_pos'][None, :, :]), axis=-1) <= 1, axis=1)
        mask = jnp.zeros((4, 12), jnp.bool_).at[:, 0].set(True)
        mask = mask.at[:, 1:5].set(free[:, 1:5] & (s['attached'] < 0)[:, None])
        mask = mask.at[:, 5].set(can_attach).at[:, 6].set(s['attached'] >= 0)
        mask = mask.at[:, 7:11].set(jnp.broadcast_to((s['attached'] >= 0)[:, None], (4, 4)))
        mask = mask.at[:, 11].set(controls)
        return jnp.where(s['done'], jnp.broadcast_to(jnp.arange(12)[None, :] == 0, (4, 12)), mask)

    def observe(self, s):
        agents = jnp.concatenate((s['agent_pos'].astype(jnp.float32) / 16,
                                  ((s['attached'] + 1).astype(jnp.float32) / 2)[:, None],
                                  s['agent_active'].astype(jnp.float32)[:, None]), axis=1).reshape(-1)
        cargo = jnp.concatenate((s['barge_pos'].astype(jnp.float32) / 16,
                                 s['delivered'].astype(jnp.float32)[:, None]), axis=1).reshape(-1)
        gates = jnp.concatenate((s['gate_pos'].astype(jnp.float32) / 16,
                                 (s['gate_timer'].astype(jnp.float32) / 4)[:, None],
                                 s['gate_flow'][:, 0].astype(jnp.float32)[:, None]), axis=1).reshape(-1)
        common = jnp.concatenate((agents, cargo, gates, jnp.reshape(s['t'].astype(jnp.float32) / 64, (1,)),
                                  s['map'].astype(jnp.float32).reshape(-1) / 4,
                                  s['tow_moved'].astype(jnp.float32), s['tow_blocked'].astype(jnp.float32),
                                  s['delivered_step'].astype(jnp.float32), s['gate_changed'].astype(jnp.float32)))
        return jnp.concatenate((jnp.eye(4, dtype=jnp.float32), jnp.broadcast_to(common, (4, 138))), axis=1)

    def raw_step(self, key, state, actions):
        mask = self.legal_actions(state)
        clipped = jnp.clip(actions, 0, 11)
        a = jnp.where((actions >= 0) & (actions < 12) & mask[jnp.arange(4), clipped], clipped, 0)
        s = dict(state)
        # Attachments are physical adjacent anchors; no agent can attach to two barges.
        distance = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['barge_pos'][None, :, :]), axis=-1)
        choice = jnp.argmax((distance == 1) & ~s['delivered'][None, :], axis=1).astype(jnp.int32)
        attached = jnp.where(a == 6, -1, jnp.where(a == 5, choice, s['attached'])).astype(jnp.int32)
        targets, free = self.free_targets(s)
        delta = self.moves()[jnp.clip(a, 0, 4)]
        proposed = s['agent_pos'] + delta
        moving = (a >= 1) & (a <= 4) & (attached < 0)
        proposed = jnp.where(moving[:, None], proposed, s['agent_pos'])
        clash = jnp.any(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=-1)
                         & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        pos = jnp.where((moving & ~clash)[:, None], proposed, s['agent_pos'])
        # One hydraulic selector: a crank closes the other channel. Priority rotates.
        control_dist = jnp.sum(jnp.abs(pos[:, None, :] - s['control_pos'][None, :, :]), axis=-1)
        control_choice = jnp.argmin(control_dist, axis=1).astype(jnp.int32)
        priority = jnp.mod(jnp.arange(4) - jnp.mod(s['t'], 4), 4)
        cranker = jnp.argmin(jnp.where(a == 11, priority, 99)).astype(jnp.int32)
        has_crank = jnp.any(a == 11)
        selected = control_choice[cranker]
        timers = jnp.maximum(s['gate_timer'] - 1, 0)
        timers = jnp.where(has_crank, jnp.where(jnp.arange(2) == selected, 4, 0), timers).astype(jnp.int32)
        flow_x = s['gate_flow'][:, 0]
        next_flow_x = jnp.where(jnp.mod((s['t'] + 1) // 8, 2) == 0, 1, -1) * jnp.array([1, -1], jnp.int32)
        flow = jnp.stack((next_flow_x, jnp.zeros((2,), jnp.int32)), axis=1)
        barge = s['barge_pos']
        moved = jnp.zeros((2,), jnp.bool_)
        blocked = jnp.zeros((2,), jnp.bool_)
        # Resolve two possible convoy moves against actual updated footprints.
        for k in range(2):
            b = jnp.mod(k + s['t'], 2)
            members = attached == b
            pull = (a >= 7) & (a <= 10) & members
            vectors = self.moves()[jnp.clip(a - 6, 0, 4)] * pull[:, None].astype(jnp.int32)
            force = jnp.sum(vectors, axis=0, dtype=jnp.int32)
            horizontal = jnp.abs(force[0]) > jnp.abs(force[1])
            vertical = jnp.abs(force[1]) > jnp.abs(force[0])
            d = jnp.array([jnp.where(horizontal, jnp.sign(force[0]), 0),
                           jnp.where(vertical, jnp.sign(force[1]), 0)], jnp.int32)
            magnitude = jnp.max(jnp.abs(force))
            offsets = pos - barge[b]
            supporting = pull & jnp.all(vectors == d[None, :], axis=1)
            perpendicular = jnp.sum(offsets * d[None, :], axis=1) == 0
            opposite = jnp.all(offsets[:, None, :] + offsets[None, :, :] == 0, axis=-1)
            paired = jnp.any(opposite & supporting[:, None] & supporting[None, :]
                              & perpendicular[:, None] & ~jnp.eye(4, dtype=jnp.bool_))
            bp = barge[b] + d
            ap = pos + members[:, None].astype(jnp.int32) * d[None, :]
            bs = jnp.clip(bp, jnp.array([0, 0]), jnp.array([10, 8]))
            aps = jnp.clip(ap, jnp.array([0, 0]), jnp.array([10, 8]))
            land_clear = jnp.all(bp == bs) & (s['map'][bs[1], bs[0]] != 1)
            land_clear = land_clear & jnp.all(~members | (jnp.all(ap == aps, axis=1) & (s['map'][aps[:, 1], aps[:, 0]] != 1)))
            hull_clear = ~jnp.any(jnp.all(bp[None, :] == barge, axis=1) & (jnp.arange(2) != b))
            hull_clear = hull_clear & ~jnp.any(jnp.all(bp[None, :] == pos, axis=1) & ~members)
            tug_clear = ~jnp.any(jnp.all(ap[:, None, :] == pos[None, :, :], axis=-1) & members[:, None] & ~members[None, :])
            tug_clear = tug_clear & ~jnp.any(jnp.all(ap[:, None, :] == barge[None, :, :], axis=-1)
                                            & members[:, None] & (jnp.arange(2) != b)[None, :])
            channel = jnp.argmin(jnp.abs(s['gate_pos'][:, 1] - bp[1])).astype(jnp.int32)
            entering = bp[0] == 5
            in_current = ((bp[0] == 5) | (barge[b, 0] == 5)) & (d[0] != 0)
            against = in_current & (d[0] != flow_x[channel])
            required = jnp.where(against, 3, 2)
            gate_clear = ~entering | (timers[channel] > 0)
            ok = paired & (magnitude >= required) & land_clear & hull_clear & tug_clear & gate_clear & ~s['delivered'][b]
            pos = jnp.where((members & ok)[:, None], ap, pos)
            barge = barge.at[b].set(jnp.where(ok, bp, barge[b]))
            moved = moved.at[b].set(ok)
            blocked = blocked.at[b].set(jnp.any(pull) & ~ok & ~s['delivered'][b])
        delivered = s['delivered'] | jnp.all(barge == s['dock_pos'], axis=1)
        delivered_step = delivered & ~s['delivered']
        attached = jnp.where((attached >= 0) & delivered[jnp.maximum(attached, 0)], -1, attached).astype(jnp.int32)
        s['agent_pos'] = pos
        s['attached'] = attached
        s['barge_pos'] = barge
        s['delivered'] = delivered
        s['gate_timer'] = timers
        s['gate_flow'] = flow
        s['tow_moved'] = moved
        s['tow_blocked'] = blocked
        s['delivered_step'] = delivered_step
        s['gate_changed'] = timers != state['gate_timer']
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.all(delivered)
        s['truncated'] = (s['t'] >= 64) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        reward = jnp.sum(delivered_step, dtype=jnp.int32).astype(jnp.float32) - (a != 0).astype(jnp.float32) / 64
        # Absorbing raw terminal: state, causes and clock are preserved exactly.
        s = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), s, state)
        reward = jnp.where(state['done'], jnp.zeros((4,), jnp.float32), reward)
        info = {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
        return self.observe(s), s, reward, jnp.full((4,), s['done'], jnp.bool_), info
