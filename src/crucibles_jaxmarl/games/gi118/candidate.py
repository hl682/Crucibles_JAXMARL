"""Original three-boat game. Source authoring only; no execution receipt."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 152
    num_actions = 9
    max_steps = 80

    def shadows(self, pos, sail, arrived, wind):
        relative = pos[:, None, :] - pos[None, :, :]
        along = jnp.sum(relative * wind[0], axis=2)
        across = jnp.abs(relative[:, :, 0] * wind[0, 1] - relative[:, :, 1] * wind[0, 0])
        casting = sail[None, :] & ~arrived[None, :]
        return jnp.any((along >= 1) & (along <= 2) & (across <= 1) & casting, axis=1)

    def decorate(self, state):
        s = dict(state)
        distance = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2)
        s['boat_visible'] = distance <= 3
        s['shadow'] = self.shadows(s['pos'], s['sail'], s['arrived'], s['wind'])
        safe = jnp.clip(s['tow'], 0, 2)
        s['link_active'] = s['tow'] >= 0
        s['link_pos'] = (s['pos'] + s['pos'][safe]).astype(jnp.float32) * 0.5
        s['link_direction'] = jnp.where(s['link_active'][:, None], s['pos'][safe] - s['pos'], 0)
        s['link_visible'] = s['boat_visible'] & s['boat_visible'][:, safe] & s['link_active'][None, :]
        s['goal_pos'] = jnp.where((s['progress'] == 0)[:, None], s['buoy_pos'][s['buoy_route']], s['dock_pos'])
        return s

    def reset(self, key):
        tiles = jnp.zeros((9, 11), dtype=jnp.int32)
        tiles = tiles.at[2:4, 5].set(1)
        tiles = tiles.at[5:7, 5].set(1)
        s = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'map': tiles,
            'pos': jnp.array([[2, 2], [3, 2], [2, 6]], jnp.int32),
            'boat_active': jnp.ones((3,), jnp.bool_),
            'boat_visible': jnp.ones((3, 3), jnp.bool_),
            'sail': jnp.ones((3,), jnp.bool_), 'stamina': jnp.full((3,), 4, jnp.int32),
            'progress': jnp.zeros((3,), jnp.int32), 'arrived': jnp.zeros((3,), jnp.bool_),
            'tow': jnp.full((3,), -1, jnp.int32), 'shadow': jnp.zeros((3,), jnp.bool_),
            'wind': jnp.array([[1, 0]], jnp.int32),
            'wind_strength': jax.random.randint(key, (1,), 1, 3, dtype=jnp.int32),
            'wind_pos': jnp.array([[5, 0]], jnp.int32),
            'wind_active': jnp.ones((1,), jnp.bool_), 'wind_visible': jnp.ones((3, 1), jnp.bool_),
            'buoy_pos': jnp.array([[7, 2], [7, 6]], jnp.int32),
            'buoy_active': jnp.ones((2,), jnp.bool_), 'buoy_visible': jnp.ones((3, 2), jnp.bool_),
            'buoy_route': jnp.array([0, 1, 0], jnp.int32),
            'dock_pos': jnp.array([[9, 1], [9, 7], [1, 4]], jnp.int32),
            'dock_active': jnp.ones((3,), jnp.bool_), 'dock_visible': jnp.ones((3, 3), jnp.bool_),
            'dock_owner': jnp.arange(3, dtype=jnp.int32),
            'goal_pos': jnp.zeros((3, 2), jnp.int32),
            'link_pos': jnp.zeros((3, 2), jnp.float32), 'link_direction': jnp.zeros((3, 2), jnp.int32),
            'link_active': jnp.zeros((3,), jnp.bool_), 'link_visible': jnp.zeros((3, 3), jnp.bool_),
            'link_helper': jnp.arange(3, dtype=jnp.int32),
            'arrival_event': jnp.zeros((3,), jnp.bool_), 'buoy_event': jnp.zeros((3,), jnp.bool_),
            'tow_event': jnp.zeros((3,), jnp.bool_), 'stall_event': jnp.zeros((3,), jnp.bool_),
        }
        s = self.decorate(s)
        return self.observe(s), s

    def observe(self, s):
        own = jnp.concatenate((s['pos'] / 10.0, s['stamina'][:, None] / 4.0,
            s['sail'][:, None], s['progress'][:, None] / 2.0, s['arrived'][:, None],
            s['tow'][:, None] / 2.0, s['goal_pos'] / 10.0, jnp.eye(3)), axis=1)
        peer = jnp.concatenate((s['pos'] / 10.0, s['sail'][:, None], s['stamina'][:, None] / 4.0,
            s['progress'][:, None] / 2.0, s['arrived'][:, None], s['tow'][:, None] / 2.0,
            s['shadow'][:, None]), axis=1)
        local = jnp.concatenate((s['boat_visible'][:, :, None],
            jnp.where(s['boat_visible'][:, :, None], peer[None, :, :], 0.0)), axis=2).reshape((3, 27))
        public = jnp.concatenate((s['map'].reshape((99,)), s['buoy_pos'].reshape((4,)) / 10.0,
            s['dock_pos'].reshape((6,)) / 10.0, s['wind'].reshape((2,)),
            s['wind_strength'] / 2.0, jnp.array([s['t'] / self.max_steps])))
        return jnp.concatenate((own, local, jnp.broadcast_to(public, (3, 113))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        return ((~s['arrived'] & ~s['done'])[:, None] | (jnp.arange(9)[None, :] == 0)).astype(jnp.bool_)

    def move_once(self, pos, delta, moving, tiles):
        trial = pos + delta
        inside = (trial[:, 0] >= 0) & (trial[:, 0] <= 10) & (trial[:, 1] >= 0) & (trial[:, 1] <= 8)
        terrain = tiles[jnp.clip(trial[:, 1], 0, 8), jnp.clip(trial[:, 0], 0, 10)] == 0
        occupied = jnp.any(jnp.all(trial[:, None, :] == pos[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        tentative = moving & inside & terrain & ~occupied
        proposed = jnp.where(tentative[:, None], trial, pos)
        duplicate = jnp.any(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        accepted = tentative & ~duplicate
        return jnp.where(accepted[:, None], trial, pos), accepted

    def advance(self, key, state, actions):
        s = dict(state)
        valid = (actions >= 0) & (actions < 9)
        a = jnp.clip(actions, 0, 8)
        a = jnp.where(valid & self.legal_actions(s)[jnp.arange(3), a], a, 0)
        sail = jnp.where(a == 5, True, jnp.where(a == 6, False, s['sail']))
        sail = jnp.where(s['arrived'], False, sail)
        tow = s['tow']
        safe = jnp.clip(tow, 0, 2)
        adjacent = jnp.sum(jnp.abs(s['pos'] - s['pos'][safe]), axis=1) <= 1
        keep = (tow >= 0) & adjacent & sail & ~sail[safe] & ~s['arrived'] & ~s['arrived'][safe]
        keep = keep & (s['stamina'] > 0) & (a != 8) & (a[safe] != 8)
        tow = jnp.where(keep, tow, -1)
        for offset in range(3):
            i = (s['t'] + offset) % 3
            in_use = jnp.any(tow[:, None] == jnp.arange(3)[None, :], axis=0)
            distance = jnp.sum(jnp.abs(s['pos'] - s['pos'][i]), axis=1)
            candidates = (distance == 1) & ~sail & ~s['arrived'] & ~in_use & (a != 8)
            target = jnp.argmax(candidates.astype(jnp.int32))
            attach = (a[i] == 7) & sail[i] & (s['stamina'][i] > 0) & jnp.any(candidates)
            tow = tow.at[i].set(jnp.where(attach, target, tow[i]))
        incoming = jnp.any(tow[:, None] == jnp.arange(3)[None, :], axis=0)
        helper = jnp.argmax((tow[:, None] == jnp.arange(3)[None, :]).astype(jnp.int32), axis=0)
        direction = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        delta = direction[a]
        moving = (a >= 1) & (a <= 4) & ~s['arrived']
        shadow = self.shadows(s['pos'], sail, s['arrived'], s['wind'])
        downwind = jnp.sum(delta * s['wind'][0], axis=1) > 0
        sailing_speed = jnp.where(shadow, 0, 1 + (downwind & (s['wind_strength'][0] == 2)).astype(jnp.int32))
        speed = jnp.where(sail, sailing_speed, (s['stamina'] > 0).astype(jnp.int32))
        speed = jnp.where(tow >= 0, jnp.minimum(speed, 1), speed)
        pos1, leader_moved = self.move_once(s['pos'], delta, moving & (speed >= 1) & ~incoming, s['map'])
        pulled = incoming & leader_moved[helper]
        pos1 = jnp.where(pulled[:, None], s['pos'][helper], pos1)
        pos2, bonus_moved = self.move_once(pos1, delta, moving & leader_moved & (speed == 2) & ~incoming, s['map'])
        helper_moved = (tow >= 0) & leader_moved
        rowing_cost = (~sail & leader_moved).astype(jnp.int32)
        towing_cost = helper_moved.astype(jnp.int32)
        recovery = ((a == 0) & ~s['arrived'] & ~helper_moved).astype(jnp.int32)
        stamina = jnp.clip(s['stamina'] - rowing_cost - towing_cost + recovery, 0, 4)
        buoy_event = (s['progress'] == 0) & jnp.all(pos2 == s['buoy_pos'][s['buoy_route']], axis=1)
        progress = s['progress'] + buoy_event.astype(jnp.int32)
        arrival = (progress == 1) & jnp.all(pos2 == s['dock_pos'], axis=1) & ~s['arrived']
        progress = progress + arrival.astype(jnp.int32)
        arrived = s['arrived'] | arrival
        first = ~jnp.any(s['arrived'])
        bonus = 0.5 * first.astype(jnp.float32) / jnp.maximum(jnp.sum(arrival.astype(jnp.float32)), 1.0)
        rewards = -0.01 * (~s['arrived']).astype(jnp.float32) + 0.25 * buoy_event.astype(jnp.float32)
        rewards = rewards + arrival.astype(jnp.float32) * (1.0 + bonus)
        keys = jax.random.split(key, 2)
        wind_table = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], jnp.int32)
        turn = ((s['t'] + 1) % 6) == 0
        wind = wind_table[jax.random.randint(keys[0], (), 0, 4, dtype=jnp.int32)][None, :]
        strength = jax.random.randint(keys[1], (1,), 1, 3, dtype=jnp.int32)
        s['wind'] = jnp.where(turn, wind, s['wind'])
        s['wind_strength'] = jnp.where(turn, strength, s['wind_strength'])
        s['pos'] = pos2
        s['sail'] = sail & ~arrived
        s['stamina'] = stamina
        s['progress'] = progress
        s['arrived'] = arrived
        s['tow'] = jnp.where((tow >= 0) & ~arrived & ~arrived[jnp.clip(tow, 0, 2)], tow, -1)
        s['arrival_event'] = arrival
        s['buoy_event'] = buoy_event
        s['tow_event'] = helper_moved
        s['stall_event'] = moving & sail & shadow & ~incoming
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.all(arrived)
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        s = self.decorate(s)
        info = {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
        return self.observe(s), s, rewards.astype(jnp.float32), jnp.full((3,), s['done'], jnp.bool_), info

    def absorb(self, state):
        info = {'done_all': state['done'], 'terminated': state['terminated'], 'truncated': state['truncated']}
        return self.observe(state), state, jnp.zeros((3,), jnp.float32), jnp.ones((3,), jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state['done'], lambda _: self.absorb(state),
            lambda _: self.advance(key, state, actions), operand=None)
