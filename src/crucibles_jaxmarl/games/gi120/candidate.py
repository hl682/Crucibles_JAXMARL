"""Original simultaneous team patrol game; source, not an executed demonstration."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 239
    num_actions = 14
    max_steps = 96

    def decorate(self, state):
        s = dict(state)
        active_decoy = s['decoy_ttl'] > 0
        s['decoy_active'] = active_decoy
        distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2)
        decoy_distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['decoy_pos'][None, :, :]), axis=2)
        hostile_decoy = s['team'][:, None] != s['team'][None, :]
        jammed = jnp.any(hostile_decoy & active_decoy[None, :] & (decoy_distance <= 2), axis=1)
        radius = jnp.where(s['scout'], 5, jnp.where(jammed, 1, 3))
        s['agent_visible'] = distance <= radius[:, None]
        s['decoy_visible'] = ((decoy_distance <= radius[:, None]) | jnp.eye(4, dtype=jnp.bool_)) & active_decoy[None, :]
        carried = s['objective_mode'] == 1
        s['objective_pos'] = jnp.where(carried[:, None], s['pos'][jnp.clip(s['objective_carrier'], 0, 3)], s['objective_pos'])
        s['objective_active'] = s['objective_mode'] != 2
        object_distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['objective_pos'][None, :, :]), axis=2)
        s['objective_visible'] = (object_distance <= radius[:, None]) & s['objective_active'][None, :]
        own = carried[:, None] & (s['objective_carrier'][:, None] == jnp.arange(4)[None, :])
        s['carrying'] = jnp.where(jnp.any(own, axis=0), jnp.argmax(own.astype(jnp.int32), axis=0), -1).astype(jnp.int32)
        s['signal_pos'] = s['pos']
        s['signal_active'] = s['scout']
        s['signal_visible'] = ((s['team'][:, None] == s['team'][None, :]) | (distance <= 5)) & s['scout'][None, :]
        s['jammed'] = jammed
        return s

    def reset(self, key):
        tiles = jnp.zeros((9, 13), jnp.int32)
        tiles = tiles.at[0, :].set(1)
        tiles = tiles.at[8, :].set(1)
        tiles = tiles.at[:, 0].set(1)
        tiles = tiles.at[:, 12].set(1)
        tiles = tiles.at[1:8, 6].set(1)
        tiles = tiles.at[2, 6].set(0)
        tiles = tiles.at[6, 6].set(0)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False), 'map': tiles,
            'pos': jnp.array([[2, 2], [2, 6], [10, 2], [10, 6]], jnp.int32),
            'spawn_pos': jnp.array([[2, 2], [2, 6], [10, 2], [10, 6]], jnp.int32),
            'agent_active': jnp.ones((4,), jnp.bool_), 'agent_visible': jnp.ones((4, 4), jnp.bool_),
            'team': jnp.array([0, 0, 1, 1], jnp.int32), 'stun': jnp.zeros((4,), jnp.int32),
            'guard': jnp.zeros((4,), jnp.bool_), 'scout': jnp.zeros((4,), jnp.bool_),
            'charges': jnp.full((4,), 2, jnp.int32), 'cooldown': jnp.zeros((4,), jnp.int32),
            'carrying': jnp.full((4,), -1, jnp.int32), 'jammed': jnp.zeros((4,), jnp.bool_),
            'decoy_pos': jnp.array([[2, 2], [2, 6], [10, 2], [10, 6]], jnp.int32),
            'decoy_heading': jnp.zeros((4, 2), jnp.int32), 'decoy_owner': jnp.arange(4, dtype=jnp.int32),
            'decoy_ttl': jnp.zeros((4,), jnp.int32), 'decoy_active': jnp.zeros((4,), jnp.bool_),
            'decoy_visible': jnp.zeros((4, 4), jnp.bool_),
            'objective_pos': jnp.array([[5, 2], [7, 2], [5, 6], [7, 6]], jnp.int32),
            'objective_mode': jnp.zeros((4,), jnp.int32), 'objective_carrier': jnp.full((4,), -1, jnp.int32),
            'objective_bank': jnp.full((4,), -1, jnp.int32),
            'objective_active': jnp.ones((4,), jnp.bool_), 'objective_visible': jnp.ones((4, 4), jnp.bool_),
            'home_pos': jnp.array([[1, 4], [11, 4]], jnp.int32), 'home_team': jnp.array([0, 1], jnp.int32),
            'home_active': jnp.ones((2,), jnp.bool_), 'home_visible': jnp.ones((4, 2), jnp.bool_),
            'signal_pos': jnp.array([[2, 2], [2, 6], [10, 2], [10, 6]], jnp.int32),
            'signal_active': jnp.zeros((4,), jnp.bool_), 'signal_visible': jnp.zeros((4, 4), jnp.bool_),
            'score': jnp.zeros((2,), jnp.int32),
            'capture_event': jnp.zeros((4,), jnp.bool_), 'distraction_event': jnp.zeros((4,), jnp.bool_),
            'deploy_event': jnp.zeros((4,), jnp.bool_), 'handoff_event': jnp.zeros((4,), jnp.bool_),
            'delivery_event': jnp.zeros((4,), jnp.bool_),
        }
        s = self.decorate(s)
        return self.observe(s), s

    def observe(self, s):
        own = jnp.concatenate((s['pos'] / 12.0, s['team'][:, None], s['stun'][:, None] / 2.0,
            s['carrying'][:, None] / 3.0, s['charges'][:, None] / 2.0, s['cooldown'][:, None] / 3.0,
            jnp.eye(4)), axis=1)
        peer = jnp.concatenate((s['pos'] / 12.0, s['team'][:, None], s['stun'][:, None] / 2.0,
            s['guard'][:, None], s['scout'][:, None], s['carrying'][:, None] / 3.0, s['charges'][:, None] / 2.0), axis=1)
        peers = jnp.concatenate((s['agent_visible'][:, :, None],
            jnp.where(s['agent_visible'][:, :, None], peer[None, :, :], 0.0)), axis=2).reshape((4, 36))
        decoy = jnp.concatenate((s['decoy_pos'] / 12.0, s['decoy_heading'], s['decoy_owner'][:, None] / 3.0,
            s['decoy_ttl'][:, None] / 6.0), axis=1)
        decoys = jnp.concatenate((s['decoy_visible'][:, :, None],
            jnp.where(s['decoy_visible'][:, :, None], decoy[None, :, :], 0.0)), axis=2).reshape((4, 28))
        obj = jnp.concatenate((s['objective_pos'] / 12.0, s['objective_mode'][:, None] / 2.0,
            s['objective_carrier'][:, None] / 3.0, s['objective_bank'][:, None]), axis=1)
        objects = jnp.concatenate((s['objective_visible'][:, :, None],
            jnp.where(s['objective_visible'][:, :, None], obj[None, :, :], 0.0)), axis=2).reshape((4, 24))
        signal = jnp.concatenate((s['signal_pos'] / 12.0, s['team'][:, None]), axis=1)
        signals = jnp.concatenate((s['signal_visible'][:, :, None],
            jnp.where(s['signal_visible'][:, :, None], signal[None, :, :], 0.0)), axis=2).reshape((4, 16))
        public = jnp.concatenate((s['map'].reshape((117,)), s['home_pos'].reshape((4,)) / 12.0,
            s['score'] / 4.0, jnp.array([s['t'] / self.max_steps])))
        return jnp.concatenate((own, peers, decoys, objects, signals, jnp.broadcast_to(public, (4, 124))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        return (((s['stun'] == 0) & ~s['done'])[:, None] | (jnp.arange(14)[None, :] == 0)).astype(jnp.bool_)

    def advance(self, key, state, actions):
        s = dict(state)
        valid = (actions >= 0) & (actions < 14)
        a = jnp.clip(actions, 0, 13)
        a = jnp.where(valid & self.legal_actions(s)[jnp.arange(4), a], a, 0)
        awake = s['stun'] == 0
        s['stun'] = jnp.maximum(s['stun'] - 1, 0)
        direction = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        trial = s['pos'] + direction[a]
        floor = s['map'][jnp.clip(trial[:, 1], 0, 8), jnp.clip(trial[:, 0], 0, 12)] == 0
        enemy_refuge = ((s['team'] == 0) & (trial[:, 0] >= 10)) | ((s['team'] == 1) & (trial[:, 0] <= 2))
        occupied = jnp.any(jnp.all(trial[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        moving = (a >= 1) & (a <= 4) & floor & ~enemy_refuge & ~occupied
        proposed = jnp.where(moving[:, None], trial, s['pos'])
        clash = jnp.any(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        s['pos'] = jnp.where((moving & ~clash)[:, None], trial, s['pos'])
        decoy_trial = s['decoy_pos'] + s['decoy_heading']
        decoy_floor = s['map'][jnp.clip(decoy_trial[:, 1], 0, 8), jnp.clip(decoy_trial[:, 0], 0, 12)] == 0
        decoy_live = s['decoy_ttl'] > 0
        s['decoy_pos'] = jnp.where((decoy_live & decoy_floor)[:, None], decoy_trial, s['decoy_pos'])
        s['decoy_heading'] = jnp.where((decoy_live & ~decoy_floor)[:, None], -s['decoy_heading'], s['decoy_heading'])
        s['decoy_ttl'] = jnp.maximum(s['decoy_ttl'] - 1, 0)
        deploy = (a >= 9) & (a <= 12) & (s['charges'] > 0) & (s['cooldown'] == 0) & awake
        heading_table = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]], jnp.int32)
        heading = heading_table[jnp.clip(a - 9, 0, 3)]
        launch = s['pos'] + heading
        launch_floor = s['map'][jnp.clip(launch[:, 1], 0, 8), jnp.clip(launch[:, 0], 0, 12)] == 0
        launch = jnp.where(launch_floor[:, None], launch, s['pos'])
        s['decoy_pos'] = jnp.where(deploy[:, None], launch, s['decoy_pos'])
        s['decoy_heading'] = jnp.where(deploy[:, None], heading, s['decoy_heading'])
        s['decoy_ttl'] = jnp.where(deploy, 6, s['decoy_ttl'])
        s['cooldown'] = jnp.where(deploy, 3, jnp.maximum(s['cooldown'] - 1, 0))
        at_home = jnp.sum(jnp.abs(s['pos'] - s['home_pos'][s['team']]), axis=1) <= 1
        replenish = (a == 0) & awake & at_home
        s['charges'] = jnp.clip(s['charges'] - deploy.astype(jnp.int32) + replenish.astype(jnp.int32), 0, 2)
        guard = (a == 7) & awake
        scout = (a == 8) & awake
        decoy_distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['decoy_pos'][None, :, :]), axis=2)
        decoy_targets = (s['team'][:, None] != s['team'][None, :]) & (s['decoy_ttl'][None, :] > 0) & (decoy_distance <= 2)
        distraction = guard & jnp.any(decoy_targets, axis=1)
        target = jnp.argmin(jnp.where(decoy_targets, decoy_distance * 4 + jnp.arange(4)[None, :], 1000), axis=1)
        destroyed = jnp.any(distraction[:, None] & (target[:, None] == jnp.arange(4)[None, :]), axis=0)
        s['decoy_ttl'] = jnp.where(destroyed, 0, s['decoy_ttl'])
        distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2)
        refuge = ((s['team'] == 0) & (s['pos'][:, 0] <= 2)) | ((s['team'] == 1) & (s['pos'][:, 0] >= 10))
        tagged = (guard & ~distraction)[:, None] & (s['team'][:, None] != s['team'][None, :]) & (distance <= 1)
        captured = jnp.any(tagged, axis=0) & awake & ~refuge
        old_carrier = jnp.clip(s['objective_carrier'], 0, 3)
        carried = s['objective_mode'] == 1
        s['objective_pos'] = jnp.where(carried[:, None], s['pos'][old_carrier], s['objective_pos'])
        dropped = carried & captured[old_carrier]
        s['objective_mode'] = jnp.where(dropped, 0, s['objective_mode'])
        s['objective_carrier'] = jnp.where(dropped, -1, s['objective_carrier'])
        s['pos'] = jnp.where(captured[:, None], s['spawn_pos'], s['pos'])
        s['stun'] = jnp.where(captured, 2, s['stun'])
        s['guard'] = guard & ~captured
        s['scout'] = scout & ~captured
        s['handoff_event'] = jnp.zeros((4,), jnp.bool_)
        s['delivery_event'] = jnp.zeros((4,), jnp.bool_)
        old_score = s['score']
        for offset in range(4):
            i = (s['t'] + offset) % 4
            own = (s['objective_mode'] == 1) & (s['objective_carrier'] == i)
            has_object = jnp.any(own)
            held = jnp.argmax(own.astype(jnp.int32))
            usable = awake[i] & ~captured[i]
            home = jnp.sum(jnp.abs(s['pos'][i] - s['home_pos'][s['team'][i]])) <= 1
            bank = (a[i] == 5) & usable & has_object & home
            s['objective_mode'] = s['objective_mode'].at[held].set(jnp.where(bank, 2, s['objective_mode'][held]))
            s['objective_carrier'] = s['objective_carrier'].at[held].set(jnp.where(bank, -1, s['objective_carrier'][held]))
            s['objective_bank'] = s['objective_bank'].at[held].set(jnp.where(bank, s['team'][i], s['objective_bank'][held]))
            s['score'] = s['score'].at[s['team'][i]].add(bank.astype(jnp.int32))
            s['delivery_event'] = s['delivery_event'].at[i].set(bank)
            close = jnp.sum(jnp.abs(s['objective_pos'] - s['pos'][i]), axis=1) <= 1
            available = (s['objective_mode'] == 0) & close
            picked = jnp.argmax(available.astype(jnp.int32))
            take = (a[i] == 5) & usable & ~has_object & jnp.any(available)
            s['objective_mode'] = s['objective_mode'].at[picked].set(jnp.where(take, 1, s['objective_mode'][picked]))
            s['objective_carrier'] = s['objective_carrier'].at[picked].set(jnp.where(take, i, s['objective_carrier'][picked]))
            loaded = jnp.any((s['objective_mode'][:, None] == 1) & (s['objective_carrier'][:, None] == jnp.arange(4)[None, :]), axis=0)
            neighbor = (jnp.sum(jnp.abs(s['pos'] - s['pos'][i]), axis=1) == 1) & (s['team'] == s['team'][i]) & ~loaded & (s['stun'] == 0)
            receiver = jnp.argmax(neighbor.astype(jnp.int32))
            handoff = (a[i] == 6) & usable & has_object & jnp.any(neighbor)
            s['objective_carrier'] = s['objective_carrier'].at[held].set(jnp.where(handoff, receiver, s['objective_carrier'][held]))
            s['handoff_event'] = s['handoff_event'].at[i].set(handoff)
            drop = (a[i] == 13) & usable & has_object
            s['objective_mode'] = s['objective_mode'].at[held].set(jnp.where(drop, 0, s['objective_mode'][held]))
            s['objective_carrier'] = s['objective_carrier'].at[held].set(jnp.where(drop, -1, s['objective_carrier'][held]))
            s['objective_pos'] = s['objective_pos'].at[held].set(jnp.where(drop, s['pos'][i], s['objective_pos'][held]))
        delta_score = s['score'] - old_score
        captured_left = jnp.sum(captured[:2].astype(jnp.float32))
        captured_right = jnp.sum(captured[2:].astype(jnp.float32))
        left_reward = (delta_score[0] - delta_score[1]).astype(jnp.float32) + 0.1 * (captured_right - captured_left)
        rewards = jnp.where(s['team'] == 0, left_reward, -left_reward).astype(jnp.float32)
        s['capture_event'] = captured
        s['distraction_event'] = distraction
        s['deploy_event'] = deploy
        s = self.decorate(s)
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.sum(s['score']) == 4
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        info = {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
        return self.observe(s), s, rewards, jnp.full((4,), s['done'], jnp.bool_), info

    def absorb(self, state):
        info = {'done_all': state['done'], 'terminated': state['terminated'], 'truncated': state['truncated']}
        return self.observe(state), state, jnp.zeros((4,), jnp.float32), jnp.ones((4,), jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state['done'], lambda _: self.absorb(state),
            lambda _: self.advance(key, state, actions), operand=None)
