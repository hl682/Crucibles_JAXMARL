"""Tag Exchange: original four-controlled-player capture, freeze, rescue and handoff."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 135
    num_actions = 11
    max_steps = 96

    def distances(self, first, second):
        return jnp.sum(jnp.abs(first[:, None, :] - second[None, :, :]), axis=-1)

    def reset(self, key):
        state = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False, jnp.bool_),
            'terminated': jnp.array(False, jnp.bool_),
            'truncated': jnp.array(False, jnp.bool_),
            'map': jnp.zeros((7, 11), jnp.int32).at[1:3, 5].set(1).at[4:6, 5].set(1),
            'agent_pos': jnp.array([[3, 2], [3, 3], [7, 3], [7, 4]], jnp.int32),
            'agent_active': jnp.ones((4,), jnp.bool_),
            'agent_visibility': jnp.ones((4, 4), jnp.bool_),
            'team': jnp.array([0, 0, 1, 1], jnp.int32),
            'energy': jnp.full((4,), 3, jnp.int32),
            'frozen': jnp.zeros((4,), jnp.int32),
            'flag_pos': jnp.array([[1, 3], [9, 3]], jnp.int32),
            'flag_active': jnp.ones((2,), jnp.bool_),
            'flag_visibility': jnp.ones((4, 2), jnp.bool_),
            'flag_team': jnp.array([0, 1], jnp.int32),
            'holder': jnp.full((2,), -1, jnp.int32),
            'base_pos': jnp.array([[1, 3], [9, 3]], jnp.int32),
            'base_active': jnp.ones((2,), jnp.bool_),
            'base_visibility': jnp.ones((4, 2), jnp.bool_),
            'recovery_stock': jnp.full((2,), 4, jnp.int32),
            'scores': jnp.zeros((2,), jnp.int32),
            'tagged': jnp.zeros((4,), jnp.bool_),
            'tag_target': jnp.full((4,), -1, jnp.int32),
            'rescued': jnp.zeros((4,), jnp.bool_),
            'passed': jnp.zeros((4,), jnp.bool_),
            'picked': jnp.zeros((4,), jnp.bool_),
            'returned': jnp.zeros((4,), jnp.bool_),
            'recharged': jnp.zeros((4,), jnp.bool_),
            'self_thawed': jnp.zeros((4,), jnp.bool_),
            'goal_event': jnp.zeros((2,), jnp.bool_),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        pos = state['agent_pos']
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = pos[:, None, :] + delta[None, :, :]
        inside = jnp.all((proposed >= 0) & (proposed <= jnp.array([10, 6])), axis=-1)
        clipped = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([10, 6]))
        movement = inside & (state['map'][clipped[:, :, 1], clipped[:, :, 0]] == 0)
        free = state['frozen'] == 0
        carrying = jnp.any(state['holder'][None, :] == jnp.arange(4)[:, None], axis=1)
        near_base = jnp.sum(jnp.abs(pos - state['base_pos'][state['team']]), axis=1) <= 1
        specials = jnp.stack([
            free & (state['energy'] > 0), free & (state['energy'] > 0), free,
            free & carrying,
            free & near_base & (state['energy'] < 3) & (state['recovery_stock'][state['team']] > 0),
            ~free & (state['energy'] >= 2),
        ], axis=1)
        mask = jnp.concatenate([jnp.ones((4, 1), jnp.bool_), movement & free[:, None], specials], axis=1)
        return jnp.where(state['done'], jnp.arange(11)[None, :] == 0, mask)

    def advance(self, key, state, actions):
        s = dict(state)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        pos = state['agent_pos'] + delta[jnp.minimum(actions, 4)] * (actions <= 4)[:, None]
        s['agent_pos'] = pos
        s['t'] = state['t'] + jnp.array(1, jnp.int32)
        distance = self.distances(pos, pos)
        free = state['frozen'] == 0
        enemy = state['team'][:, None] != state['team'][None, :]

        # Tags use post-movement positions and starting freedom; mutual tags both land.
        targets = (distance <= 1) & enemy & free[None, :]
        target = jnp.argmin(jnp.where(targets, distance * 5 + jnp.arange(4)[None, :], 99), axis=1)
        attack = (actions == 5) & free & (state['energy'] > 0) & jnp.any(targets, axis=1)
        hits = attack[:, None] & (target[:, None] == jnp.arange(4)[None, :])
        tagged = jnp.any(hits, axis=0)
        frozen = jnp.where(tagged, 3, jnp.maximum(state['frozen'] - 1, 0))
        energy = state['energy'] - attack.astype(jnp.int32)
        s['tagged'] = tagged
        s['tag_target'] = jnp.where(attack, target, -1)

        # A frozen player may spend two energy to thaw; rescue costs the free teammate one.
        self_thaw = (actions == 10) & ~free & (energy >= 2)
        frozen = jnp.where(self_thaw, 0, frozen)
        energy = energy - 2 * self_thaw.astype(jnp.int32)
        mate = jnp.array([1, 0, 3, 2], jnp.int32)
        rescue = (actions == 6) & free & ~tagged & (energy > 0)
        rescue = rescue & (distance[jnp.arange(4), mate] <= 1) & (frozen[mate] > 0) & ~tagged[mate]
        rescued = rescue[mate]
        frozen = jnp.where(rescued, 0, frozen)
        energy = energy - rescue.astype(jnp.int32)
        s['rescued'] = rescued
        s['self_thawed'] = self_thaw

        # Recharge draws from a finite team depot, lower actor index wins its last unit.
        stock = state['recovery_stock']
        recharged = jnp.zeros((4,), jnp.bool_)
        for i in range(4):
            team = state['team'][i]
            near_base = jnp.sum(jnp.abs(pos[i] - state['base_pos'][team])) <= 1
            charge = (actions[i] == 9) & free[i] & ~tagged[i] & near_base & (energy[i] < 3) & (stock[team] > 0)
            energy = energy.at[i].add(charge.astype(jnp.int32))
            stock = stock.at[team].add(-charge.astype(jnp.int32))
            recharged = recharged.at[i].set(charge)
        s['energy'] = energy
        s['frozen'] = frozen
        s['recovery_stock'] = stock
        s['recharged'] = recharged

        # Tags drop carried flags at the real contact cell before object interactions.
        holder = state['holder']
        flag_pos = jnp.where((holder >= 0)[:, None], pos[jnp.maximum(holder, 0)], state['flag_pos'])
        dropped = (holder >= 0) & tagged[jnp.maximum(holder, 0)]
        holder = jnp.where(dropped, -1, holder)
        picked = jnp.zeros((4,), jnp.bool_)
        returned = jnp.zeros((4,), jnp.bool_)
        for i in range(4):
            own_flag = state['team'][i]
            enemy_flag = 1 - own_flag
            interact = (actions[i] == 7) & free[i] & ~tagged[i]
            return_home = interact & (holder[own_flag] < 0) & jnp.all(flag_pos[own_flag] == pos[i])
            return_home = return_home & ~jnp.all(flag_pos[own_flag] == state['base_pos'][own_flag])
            flag_pos = flag_pos.at[own_flag].set(jnp.where(return_home, state['base_pos'][own_flag], flag_pos[own_flag]))
            returned = returned.at[i].set(return_home)
            pickup = interact & ~jnp.any(holder == i) & (holder[enemy_flag] < 0) & jnp.all(flag_pos[enemy_flag] == pos[i])
            holder = holder.at[enemy_flag].set(jnp.where(pickup, i, holder[enemy_flag]))
            picked = picked.at[i].set(pickup)

        # There are exactly two players on each team; each flag has at most one holder.
        old_holder = holder
        safe_holder = jnp.maximum(old_holder, 0)
        receiver = mate[safe_holder]
        pass_flag = (old_holder >= 0) & (actions[safe_holder] == 8) & free[safe_holder] & ~tagged[safe_holder]
        pass_flag = pass_flag & (distance[safe_holder, receiver] <= 1) & (frozen[receiver] == 0) & ~tagged[receiver]
        occupied = jnp.any(old_holder[None, :] == receiver[:, None], axis=1)
        pass_flag = pass_flag & ~occupied
        holder = jnp.where(pass_flag, receiver, old_holder)
        passed = jnp.any(pass_flag[None, :] & (old_holder[None, :] == jnp.arange(4)[:, None]), axis=1)
        flag_pos = jnp.where((holder >= 0)[:, None], pos[jnp.maximum(holder, 0)], flag_pos)
        s['picked'] = picked
        s['returned'] = returned
        s['passed'] = passed

        # Return an enemy flag to your own base while your own flag is home to score.
        home = (holder < 0) & jnp.all(flag_pos == state['base_pos'], axis=1)
        enemy_holder = holder[jnp.array([1, 0], jnp.int32)]
        scorer = jnp.maximum(enemy_holder, 0)
        goal = (enemy_holder >= 0) & home & (state['team'][scorer] == jnp.arange(2)) & (frozen[scorer] == 0)
        goal = goal & jnp.all(pos[scorer] == state['base_pos'], axis=1)
        scored_flag = goal[jnp.array([1, 0], jnp.int32)]
        holder = jnp.where(scored_flag, -1, holder)
        flag_pos = jnp.where(scored_flag[:, None], state['base_pos'], flag_pos)
        s['holder'] = holder
        s['flag_pos'] = flag_pos
        s['goal_event'] = goal
        s['scores'] = state['scores'] + goal.astype(jnp.int32)
        s['terminated'] = jnp.any(s['scores'] >= 2)
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        rewards = goal[state['team']].astype(jnp.float32) - goal[1 - state['team']].astype(jnp.float32)
        return s, rewards.astype(jnp.float32)

    def observe(self, state):
        public = jnp.concatenate([
            state['t'][None] / 96.0, state['map'].reshape(-1).astype(jnp.float32),
            (state['agent_pos'] / jnp.array([10.0, 6.0])).reshape(-1), state['energy'] / 3.0,
            state['frozen'] / 3.0, state['team'].astype(jnp.float32),
            (state['flag_pos'] / jnp.array([10.0, 6.0])).reshape(-1), (state['holder'] + 1) / 4.0,
            (state['base_pos'] / jnp.array([10.0, 6.0])).reshape(-1), state['recovery_stock'] / 4.0,
            state['scores'] / 2.0, state['tagged'].astype(jnp.float32), state['rescued'].astype(jnp.float32),
            state['passed'].astype(jnp.float32), state['goal_event'].astype(jnp.float32),
            jnp.stack([state['done'], state['terminated'], state['truncated']]).astype(jnp.float32),
        ])
        return jnp.concatenate([jnp.eye(4, dtype=jnp.float32), jax.nn.one_hot(state['team'], 2),
            jnp.broadcast_to(public[None, :], (4, 129))], axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions.astype(jnp.int32), 0, self.num_actions - 1)
        valid = (actions >= 0) & (actions < self.num_actions)
        valid = valid & self.legal_actions(state)[jnp.arange(4), clipped]
        proposed, rewards = self.advance(key, state, jnp.where(valid, clipped, 0))
        next_state = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), proposed, state)
        rewards = jnp.where(state['done'], jnp.zeros((4,), jnp.float32), rewards)
        dones = jnp.full((4,), next_state['done'], jnp.bool_)
        info = {'done_all': next_state['done'], 'terminated': next_state['terminated'], 'truncated': next_state['truncated']}
        return self.observe(next_state), next_state, rewards, dones, info
