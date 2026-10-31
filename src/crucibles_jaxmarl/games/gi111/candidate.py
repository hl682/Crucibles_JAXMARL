"""Original local-information relay contest; authored, never executed here."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 146
    num_actions = 10
    max_steps = 80

    def moves(self):
        return jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def refresh(self, s):
        p = s['agent_pos']
        s['agent_vis'] = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=-1) <= 3
        native = s['team'][:, None] == jnp.arange(2)[None, :]
        s['relay_vis'] = native | (jnp.sum(jnp.abs(p[:, None, :] - s['relay_pos'][None, :, :]), axis=-1) <= 3)
        s['signal_active'] = s['signal_ttl'] > 0
        s['signal_vis'] = (native | (jnp.sum(jnp.abs(p[:, None, :] - s['signal_pos'][None, :, :]), axis=-1) <= 3)) & s['signal_active'][None, :]
        s['plot_vis'] = jnp.sum(jnp.abs(p[:, None, :] - s['plot_pos'][None, :, :]), axis=-1) <= 3
        carries = jnp.where(s['relay_carrier'][None, :] == jnp.arange(4)[:, None], jnp.arange(2)[None, :], -1)
        s['agent_carry'] = jnp.max(carries, axis=1).astype(jnp.int32)
        return s

    def reset(self, key):
        xx = jnp.arange(9)[None, :]
        yy = jnp.arange(9)[:, None]
        walls = (xx == 0) | (xx == 8) | (yy == 0) | (yy == 8) | ((xx == 4) & (jnp.mod(yy, 2) == 1))
        tiles = walls.astype(jnp.int32).at[2, 4].set(2).at[4, 4].set(2).at[6, 4].set(2)
        tiles = tiles.at[4, 1].set(3).at[4, 7].set(3)
        s = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'map': tiles,
            'agent_pos': jnp.array([[2, 3], [2, 5], [6, 3], [6, 5]], jnp.int32),
            'agent_active': jnp.ones((4,), jnp.bool_), 'agent_vis': jnp.ones((4, 4), jnp.bool_),
            'team': jnp.array([0, 0, 1, 1], jnp.int32),
            'agent_carry': jnp.full((4,), -1, jnp.int32), 'guard': jnp.zeros((4,), jnp.bool_),
            'relay_pos': jnp.array([[2, 4], [6, 4]], jnp.int32),
            'relay_active': jnp.ones((2,), jnp.bool_), 'relay_vis': jnp.zeros((4, 2), jnp.bool_),
            'relay_carrier': jnp.full((2,), -1, jnp.int32), 'relay_charge': jnp.full((2,), 4, jnp.int32),
            'relay_team': jnp.array([0, 1], jnp.int32),
            'signal_pos': jnp.array([[2, 4], [6, 4]], jnp.int32),
            'signal_active': jnp.zeros((2,), jnp.bool_), 'signal_vis': jnp.zeros((4, 2), jnp.bool_),
            'signal_ttl': jnp.zeros((2,), jnp.int32), 'signal_radius': jnp.full((2,), 3, jnp.int32),
            'plot_pos': jnp.array([[4, 2], [4, 4], [4, 6]], jnp.int32),
            'plot_active': jnp.ones((3,), jnp.bool_), 'plot_vis': jnp.zeros((4, 3), jnp.bool_),
            'plot_control': jnp.zeros((3,), jnp.int32), 'plot_owner': jnp.full((3,), -1, jnp.int32),
            'plot_award': jnp.zeros((3,), jnp.int32),
            'base_pos': jnp.array([[1, 4], [7, 4]], jnp.int32),
            'base_active': jnp.ones((2,), jnp.bool_), 'base_vis': jnp.ones((4, 2), jnp.bool_),
            'score': jnp.zeros((2,), jnp.int32), 'score_step': jnp.zeros((2,), jnp.int32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def move_options(self, s):
        q = s['agent_pos'][:, None, :] + self.moves()[None, :, :]
        safe = jnp.clip(q, 0, 8)
        terrain = (s['map'][safe[:, :, 1], safe[:, :, 0]] != 1) & jnp.all(q == safe, axis=-1)
        occupied = jnp.any(jnp.all(q[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=-1)
                           & ~jnp.eye(4, dtype=jnp.bool_)[:, None, :], axis=-1)
        return q, terrain & ~occupied

    def legal_actions(self, s):
        q, clear = self.move_options(s)
        team = s['team']
        near = jnp.sum(jnp.abs(s['agent_pos'] - s['relay_pos'][team]), axis=1) <= 1
        own_holder = s['relay_carrier'][team]
        usable = near & ((own_holder < 0) | (own_holder == jnp.arange(4)))
        enemy_near = jnp.sum(jnp.abs(s['agent_pos'] - s['relay_pos'][1 - team]), axis=1) <= 2
        at_base = jnp.sum(jnp.abs(s['agent_pos'] - s['base_pos'][team]), axis=1) <= 1
        relay_at_base = jnp.sum(jnp.abs(s['relay_pos'] - s['base_pos']), axis=1) <= 1
        mask = jnp.zeros((4, 10), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(clear[:, 1:5])
        mask = mask.at[:, 5].set(usable & (s['relay_charge'][team] > 0))
        mask = mask.at[:, 6].set(usable)
        mask = mask.at[:, 7].set(enemy_near)
        mask = mask.at[:, 8].set(near)
        mask = mask.at[:, 9].set(at_base & relay_at_base[team] & (s['relay_charge'][team] < 4))
        return jnp.where(s['done'], jnp.broadcast_to(jnp.arange(10)[None, :] == 0, (4, 10)), mask)

    def observe(self, s):
        av = s['agent_vis']
        agents = jnp.concatenate((s['agent_pos'].astype(jnp.float32) / 8,
                                  s['team'].astype(jnp.float32)[:, None],
                                  ((s['agent_carry'] + 1).astype(jnp.float32) / 2)[:, None],
                                  s['guard'].astype(jnp.float32)[:, None]), axis=1)
        agent_fields = jnp.concatenate((av[:, :, None].astype(jnp.float32),
                                        jnp.where(av[:, :, None], agents[None, :, :], 0)), axis=-1).reshape(4, 24)
        rv = s['relay_vis']
        relays = jnp.concatenate((s['relay_pos'].astype(jnp.float32) / 8,
                                  s['relay_team'].astype(jnp.float32)[:, None],
                                  (s['relay_charge'].astype(jnp.float32) / 4)[:, None],
                                  ((s['relay_carrier'] + 1).astype(jnp.float32) / 4)[:, None]), axis=1)
        relay_fields = jnp.concatenate((rv[:, :, None].astype(jnp.float32),
                                        jnp.where(rv[:, :, None], relays[None, :, :], 0)), axis=-1).reshape(4, 12)
        sv = s['signal_vis']
        signals = jnp.concatenate((s['signal_pos'].astype(jnp.float32) / 8,
                                   (s['signal_ttl'].astype(jnp.float32) / 4)[:, None],
                                   (s['signal_radius'].astype(jnp.float32) / 4)[:, None]), axis=1)
        signal_fields = jnp.concatenate((sv[:, :, None].astype(jnp.float32),
                                         jnp.where(sv[:, :, None], signals[None, :, :], 0)), axis=-1).reshape(4, 10)
        pv = s['plot_vis']
        plots = jnp.stack(((s['plot_owner'] + 1).astype(jnp.float32) / 2,
                           s['plot_control'].astype(jnp.float32) / 4,
                           s['plot_award'].astype(jnp.float32) / 2), axis=1)
        plot_fields = jnp.concatenate((pv[:, :, None].astype(jnp.float32),
                                       jnp.where(pv[:, :, None], plots[None, :, :], 0)), axis=-1).reshape(4, 12)
        public = jnp.concatenate((jnp.reshape(s['t'].astype(jnp.float32) / 128, (1,)), s['score'].astype(jnp.float32) / 256))
        world = jnp.broadcast_to(s['map'].astype(jnp.float32).reshape(1, 81) / 4, (4, 81))
        return jnp.concatenate((jnp.eye(4, dtype=jnp.float32), jnp.broadcast_to(public, (4, 3)),
                                agent_fields, relay_fields, signal_fields, plot_fields, world), axis=1)

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions, 0, 9)
        mask = self.legal_actions(state)
        a = jnp.where((actions >= 0) & (actions < 10) & mask[jnp.arange(4), clipped], clipped, 0)
        s = dict(state)
        q = s['agent_pos'] + self.moves()[jnp.clip(a, 0, 4)]
        moving = (a >= 1) & (a <= 4)
        q = jnp.where(moving[:, None], q, s['agent_pos'])
        clash = jnp.any(jnp.all(q[:, None, :] == q[None, :, :], axis=-1) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        pos = jnp.where((moving & ~clash)[:, None], q, s['agent_pos'])
        priority = jnp.mod(jnp.arange(4) - jnp.mod(s['t'], 4), 4)
        carriers = s['relay_carrier']
        relay = s['relay_pos']
        # One action prevents a holder from pulsing and guarding/jamming together.
        for r in range(2):
            previous = carriers[r]
            dropping = (previous >= 0) & (a[jnp.maximum(previous, 0)] == 6)
            holder = jnp.where(dropping, -1, previous)
            candidate = (s['team'] == r) & (a == 6) & (previous < 0)
            candidate = candidate & (jnp.sum(jnp.abs(pos - relay[r]), axis=1) <= 1)
            picker = jnp.argmin(jnp.where(candidate, priority, 99)).astype(jnp.int32)
            holder = jnp.where(jnp.any(candidate), picker, holder).astype(jnp.int32)
            relay = relay.at[r].set(jnp.where(previous >= 0, pos[jnp.maximum(previous, 0)], relay[r]))
            relay = relay.at[r].set(jnp.where(holder >= 0, pos[jnp.maximum(holder, 0)], relay[r]))
            carriers = carriers.at[r].set(holder)
        near = jnp.sum(jnp.abs(pos - relay[s['team']]), axis=1) <= 1
        guarding = (a == 8) & near
        charge = s['relay_charge']
        ttl = jnp.maximum(s['signal_ttl'] - 1, 0)
        signal_pos = s['signal_pos']
        radius = s['signal_radius']
        # Guard prevents simultaneous enemy jamming and strengthens a new local pulse.
        for r in range(2):
            protected = jnp.any(guarding & (s['team'] == r))
            jammer = (a == 7) & (s['team'] != r) & (jnp.sum(jnp.abs(pos - relay[r]), axis=1) <= 2)
            hits = jnp.where(protected, 0, jnp.sum(jammer, dtype=jnp.int32))
            remaining = jnp.maximum(charge[r] - hits, 0)
            signal_life = jnp.maximum(ttl[r] - hits, 0)
            sender = (a == 5) & (s['team'] == r) & (jnp.sum(jnp.abs(pos - relay[r]), axis=1) <= 1)
            sender = sender & ((carriers[r] < 0) | (carriers[r] == jnp.arange(4)))
            pulse = jnp.any(sender) & (remaining > 0)
            remaining = remaining - pulse.astype(jnp.int32)
            charger = (a == 9) & (s['team'] == r) & (jnp.sum(jnp.abs(pos - s['base_pos'][r]), axis=1) <= 1)
            recharge = jnp.any(charger) & (jnp.sum(jnp.abs(relay[r] - s['base_pos'][r])) <= 1)
            charge = charge.at[r].set(jnp.minimum(4, remaining + recharge.astype(jnp.int32)))
            ttl = ttl.at[r].set(jnp.where(pulse, 3, signal_life))
            signal_pos = signal_pos.at[r].set(jnp.where(pulse, relay[r], signal_pos[r]))
            radius = radius.at[r].set(jnp.where(pulse, jnp.where(protected, 4, 3), radius[r]))
        # Pulses remain where emitted even when the transmitter is carried elsewhere.
        cover = (jnp.sum(jnp.abs(signal_pos[:, None, :] - s['plot_pos'][None, :, :]), axis=-1) <= radius[:, None]) & (ttl > 0)[:, None]
        pressure = cover[0].astype(jnp.int32) - cover[1].astype(jnp.int32)
        control = jnp.clip(s['plot_control'] + pressure, -3, 3).astype(jnp.int32)
        owner = jnp.where(control >= 2, 0, jnp.where(control <= -2, 1, s['plot_owner'])).astype(jnp.int32)
        award0 = (owner == 0) & cover[0] & ~cover[1]
        award1 = (owner == 1) & cover[1] & ~cover[0]
        score_step = jnp.array([jnp.sum(award0, dtype=jnp.int32), jnp.sum(award1, dtype=jnp.int32)], jnp.int32)
        s['agent_pos'] = pos
        s['guard'] = guarding
        s['relay_pos'] = relay
        s['relay_carrier'] = carriers
        s['relay_charge'] = charge
        s['signal_pos'] = signal_pos
        s['signal_ttl'] = ttl
        s['signal_radius'] = radius
        s['plot_control'] = control
        s['plot_owner'] = owner
        s['plot_award'] = award0.astype(jnp.int32) + 2 * award1.astype(jnp.int32)
        s['score_step'] = score_step
        s['score'] = s['score'] + score_step
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.any(s['score'] >= 24)
        s['truncated'] = (s['t'] >= 80) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        s = self.refresh(s)
        advantage = (score_step[s['team']] - score_step[1 - s['team']]).astype(jnp.float32) / 4
        reward = advantage - (a != 0).astype(jnp.float32) / 128
        s = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), s, state)
        reward = jnp.where(state['done'], jnp.zeros((4,), jnp.float32), reward)
        info = {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
        return self.observe(s), s, reward, jnp.full((4,), s['done'], jnp.bool_), info
