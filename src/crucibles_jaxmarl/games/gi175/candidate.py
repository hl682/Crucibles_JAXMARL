"""Lantern Convoy: original discrete light, fragile cargo and timestamped reports."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 184
    num_actions = 11
    max_steps = 96

    def distances(self, first, second):
        return jnp.sum(jnp.abs(first[:, None, :] - second[None, :, :]), axis=-1)

    def reset(self, key):
        priorities = jax.random.randint(key, (3, 3), 1, 3, dtype=jnp.int32)
        priorities = jnp.where(jnp.eye(3, dtype=jnp.bool_), 3, priorities)
        state = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False, jnp.bool_),
            'terminated': jnp.array(False, jnp.bool_),
            'truncated': jnp.array(False, jnp.bool_),
            'map': jnp.zeros((7, 11), jnp.int32).at[1:6, 5].set(1).at[2, 5].set(0).at[4, 5].set(0),
            'agent_pos': jnp.array([[1, 2], [2, 2], [1, 4]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_),
            'agent_visibility': jnp.ones((3, 3), jnp.bool_),
            'lantern': jnp.array([True, False, True], jnp.bool_),
            'battery': jnp.array([8, 0, 8], jnp.int32),
            'lamp_on': jnp.zeros((3,), jnp.bool_),
            'glow': jnp.zeros((3,), jnp.bool_),
            'private_priority': priorities,
            'cargo_pos': jnp.array([[1, 2], [1, 3], [1, 4]], jnp.int32),
            'cargo_active': jnp.ones((3,), jnp.bool_),
            'cargo_visibility': jnp.ones((3, 3), jnp.bool_),
            'holder': jnp.full((3,), -1, jnp.int32),
            'integrity': jnp.full((3,), 3, jnp.int32),
            'delivered': jnp.zeros((3,), jnp.bool_),
            'goal_pos': jnp.array([[9, 2], [9, 3], [9, 4]], jnp.int32),
            'goal_active': jnp.ones((3,), jnp.bool_),
            'goal_visibility': jnp.ones((3, 3), jnp.bool_),
            'hazard_pos': jnp.array([[4, 2], [6, 4], [8, 3], [4, 4]], jnp.int32),
            'hazard_active': jnp.ones((4,), jnp.bool_),
            'hazard_visibility': jnp.zeros((3, 4), jnp.bool_),
            'hazard_phase': jnp.array([0, 1, 2, 3], jnp.int32),
            'armed': jnp.array([True, True, False, False], jnp.bool_),
            'memory_stamp': jnp.full((3, 4), -1, jnp.int32),
            'memory_armed': jnp.zeros((3, 4), jnp.bool_),
            'relay_received': jnp.zeros((3,), jnp.bool_),
            'lamp_passed': jnp.zeros((3,), jnp.bool_),
            'cargo_passed': jnp.zeros((3,), jnp.bool_),
            'contact': jnp.zeros((3,), jnp.bool_),
            'delivered_event': jnp.zeros((3,), jnp.bool_),
            'broken_event': jnp.zeros((3,), jnp.bool_),
        }
        state = self.perceive(state, jnp.zeros((3,), jnp.bool_))
        return self.observe(state), state

    def perceive(self, state, relay):
        s = dict(state)
        distance = self.distances(state['agent_pos'], state['hazard_pos'])
        lit = jnp.any((distance <= 3) & state['glow'][:, None], axis=0)
        visible = (distance <= 1) | ((distance <= 4) & lit[None, :])
        s['hazard_visibility'] = visible
        stamp = jnp.where(visible, state['t'], state['memory_stamp'])
        remembered = jnp.where(visible, state['armed'][None, :], state['memory_armed'])
        adjacent = self.distances(state['agent_pos'], state['agent_pos']) <= 1
        # Receiver x sender x hazard; one-hop relay copies capture time unchanged.
        receive_from = adjacent & relay[None, :]
        receive_from = receive_from | jnp.eye(3, dtype=jnp.bool_)
        offers = jnp.where(receive_from[:, :, None], stamp[None, :, :], -1)
        best_sender = jnp.argmax(offers, axis=1)
        best_stamp = jnp.max(offers, axis=1)
        report = remembered[best_sender, jnp.arange(4)[None, :]]
        newer = best_stamp > stamp
        s['memory_stamp'] = jnp.where(newer, best_stamp, stamp)
        s['memory_armed'] = jnp.where(newer, report, remembered)
        s['relay_received'] = jnp.any(newer, axis=1)
        return s

    def legal_actions(self, state):
        pos = state['agent_pos']
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = pos[:, None, :] + delta[None, :, :]
        inside = jnp.all((proposed >= 0) & (proposed <= jnp.array([10, 6])), axis=-1)
        clipped = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([10, 6]))
        movement = inside & (state['map'][clipped[:, :, 1], clipped[:, :, 0]] == 0)
        holds = state['holder'][None, :] == jnp.arange(3)[:, None]
        carrying = jnp.any(holds, axis=1)
        adjacent = (self.distances(pos, pos) <= 1) & ~jnp.eye(3, dtype=jnp.bool_)
        ground_here = (self.distances(pos, state['cargo_pos']) == 0) & (state['holder'][None, :] < 0) & state['cargo_active'][None, :]
        destination = self.distances(pos, state['goal_pos']) == 0
        specials = jnp.stack([
            state['lantern'] & ((state['battery'] > 0) | state['lamp_on']),
            state['lantern'] & jnp.any(adjacent & ~state['lantern'][None, :], axis=1),
            carrying | jnp.any(ground_here, axis=1),
            carrying & jnp.any(adjacent & ~carrying[None, :], axis=1),
            jnp.any(holds & destination, axis=1),
            jnp.any(adjacent, axis=1) & jnp.any(state['memory_stamp'] >= 0, axis=1),
        ], axis=1)
        mask = jnp.concatenate([jnp.ones((3, 1), jnp.bool_), movement, specials], axis=1)
        return jnp.where(state['done'], jnp.arange(11)[None, :] == 0, mask)

    def matching(self, pos, sender, recipient):
        distance = self.distances(pos, pos)
        available = (distance <= 1) & ~jnp.eye(3, dtype=jnp.bool_) & recipient[None, :]
        target = jnp.argmin(jnp.where(available, distance * 4 + jnp.arange(3)[None, :], 99), axis=1)
        offer = sender & jnp.any(available, axis=1)
        requests = offer[:, None] & (target[:, None] == jnp.arange(3)[None, :])
        winner = jnp.argmin(jnp.where(requests, jnp.arange(3)[:, None], 9), axis=0)
        return requests & (winner[None, :] == jnp.arange(3)[:, None])

    def advance(self, key, state, actions):
        s = dict(state)
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        pos = state['agent_pos'] + directions[jnp.minimum(actions, 4)] * (actions <= 4)[:, None]
        s['agent_pos'] = pos
        s['t'] = state['t'] + jnp.array(1, jnp.int32)

        # Lantern, remaining charge and switch setting travel together.
        lamp_on = state['lamp_on'] ^ ((actions == 5) & state['lantern'])
        matching = self.matching(pos, (actions == 6) & state['lantern'], ~state['lantern'])
        sent = jnp.any(matching, axis=1)
        received = jnp.any(matching, axis=0)
        incoming_battery = jnp.sum(matching.astype(jnp.int32) * state['battery'][:, None], axis=0)
        incoming_on = jnp.any(matching & lamp_on[:, None], axis=0)
        battery = jnp.where(sent, 0, jnp.where(received, incoming_battery, state['battery']))
        lantern = (state['lantern'] & ~sent) | received
        lamp_on = jnp.where(sent, False, jnp.where(received, incoming_on, lamp_on))
        glow = lantern & lamp_on & (battery > 0)
        s['battery'] = battery - glow.astype(jnp.int32)
        s['lantern'] = lantern
        s['lamp_on'] = lamp_on & (s['battery'] > 0)
        s['glow'] = glow
        s['lamp_passed'] = sent

        # Pickup/drop uses exact cells and ascending actor order for contested pickup.
        holder = state['holder']
        cargo_pos = state['cargo_pos']
        for i in range(3):
            own = holder == i
            carrying = jnp.any(own)
            carried_id = jnp.argmax(own)
            ground = (holder < 0) & state['cargo_active'] & jnp.all(cargo_pos == pos[i], axis=1)
            picked_id = jnp.argmin(jnp.where(ground, jnp.arange(3), 9))
            drop = (actions[i] == 7) & carrying
            pick = (actions[i] == 7) & ~carrying & jnp.any(ground)
            holder = holder.at[carried_id].set(jnp.where(drop, -1, holder[carried_id]))
            cargo_pos = cargo_pos.at[carried_id].set(jnp.where(drop, pos[i], cargo_pos[carried_id]))
            holder = holder.at[picked_id].set(jnp.where(pick, i, holder[picked_id]))

        holds = holder[None, :] == jnp.arange(3)[:, None]
        carrying = jnp.any(holds, axis=1)
        passing = self.matching(pos, (actions == 8) & carrying, ~carrying)
        sent_cargo = jnp.any(passing, axis=1)
        destination = jnp.argmax(passing, axis=1)
        old_holder = holder
        safe_holder = jnp.maximum(old_holder, 0)
        transfers = (old_holder >= 0) & sent_cargo[safe_holder]
        holder = jnp.where(transfers, destination[safe_holder], old_holder)
        s['cargo_passed'] = sent_cargo
        cargo_pos = jnp.where((holder >= 0)[:, None], pos[jnp.maximum(holder, 0)], cargo_pos)

        # Hazards pulse by the new clock, then damage any carried cargo on contact.
        armed = ((s['t'] + state['hazard_phase']) % 4) < 2
        contact = jnp.any((self.distances(pos, state['hazard_pos']) == 0) & armed[None, :], axis=1)
        damaged = (holder >= 0) & contact[jnp.maximum(holder, 0)] & state['cargo_active']
        integrity = jnp.maximum(state['integrity'] - damaged.astype(jnp.int32), 0)
        broken_event = (state['integrity'] > 0) & (integrity == 0)
        holder = jnp.where(integrity == 0, -1, holder)
        delivery = (holder >= 0) & (actions[jnp.maximum(holder, 0)] == 9)
        delivery = delivery & jnp.all(cargo_pos == state['goal_pos'], axis=1) & (integrity > 0) & ~state['delivered']
        delivered = state['delivered'] | delivery
        holder = jnp.where(delivered, -1, holder)
        s['cargo_pos'] = cargo_pos
        s['holder'] = holder
        s['integrity'] = integrity
        s['delivered'] = delivered
        s['cargo_active'] = ~delivered & (integrity > 0)
        s['armed'] = armed
        s['contact'] = contact
        s['delivered_event'] = delivery
        s['broken_event'] = broken_event
        s = self.perceive(s, actions == 10)
        s['terminated'] = jnp.all(delivered | (integrity == 0))
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        outcomes = delivery.astype(jnp.float32) - broken_event.astype(jnp.float32)
        rewards = jnp.sum(state['private_priority'].astype(jnp.float32) * outcomes[None, :], axis=1)
        rewards = rewards - glow.astype(jnp.float32) * 0.02
        return s, rewards.astype(jnp.float32)

    def observe(self, state):
        held = state['holder'][None, :] == jnp.arange(3)[:, None]
        own = jnp.concatenate([
            jnp.eye(3, dtype=jnp.float32), state['private_priority'] / 3.0,
            held.astype(jnp.float32), state['lantern'][:, None].astype(jnp.float32),
            state['battery'][:, None] / 8.0, state['lamp_on'][:, None].astype(jnp.float32),
            state['glow'][:, None].astype(jnp.float32),
        ], axis=1)
        public = jnp.concatenate([
            state['t'][None] / 96.0, state['map'].reshape(-1).astype(jnp.float32),
            state['delivered'].astype(jnp.float32), (state['integrity'] == 0).astype(jnp.float32),
            (state['goal_pos'] / jnp.array([10.0, 6.0])).reshape(-1),
            jnp.stack([state['done'], state['terminated'], state['truncated']]).astype(jnp.float32),
        ])
        carried_code = jnp.max(jnp.where(held, jnp.arange(3)[None, :] + 1, 0), axis=1) / 3.0
        agent_data = jnp.concatenate([
            jnp.ones((3, 1), jnp.float32), state['agent_pos'] / jnp.array([10.0, 6.0]),
            state['lantern'][:, None].astype(jnp.float32), state['battery'][:, None] / 8.0,
            state['lamp_on'][:, None].astype(jnp.float32), state['glow'][:, None].astype(jnp.float32),
            carried_code[:, None],
        ], axis=1).reshape(-1)
        cargo_data = jnp.concatenate([
            jnp.ones((3, 1), jnp.float32), state['cargo_pos'] / jnp.array([10.0, 6.0]),
            state['integrity'][:, None] / 3.0, (state['holder'][:, None] + 1) / 3.0,
            state['cargo_active'][:, None].astype(jnp.float32),
        ], axis=1).reshape(-1)
        visible = state['hazard_visibility']
        direct_values = jnp.concatenate([
            state['hazard_pos'] / jnp.array([10.0, 6.0]), state['armed'][:, None].astype(jnp.float32),
        ], axis=1)
        direct = jnp.concatenate([visible[:, :, None].astype(jnp.float32),
            jnp.where(visible[:, :, None], direct_values[None, :, :], 0.0)], axis=2).reshape(3, -1)
        known = state['memory_stamp'] >= 0
        age = jnp.where(known, state['t'] - state['memory_stamp'], 0) / 96.0
        memories = jnp.concatenate([
            known[:, :, None].astype(jnp.float32),
            jnp.where(known[:, :, None], (state['hazard_pos'] / jnp.array([10.0, 6.0]))[None, :, :], 0.0),
            jnp.where(known, state['memory_armed'], False)[:, :, None].astype(jnp.float32), age[:, :, None],
        ], axis=2).reshape(3, -1)
        return jnp.concatenate([own, jnp.broadcast_to(public[None, :], (3, 93)),
            jnp.broadcast_to(agent_data[None, :], (3, 24)), jnp.broadcast_to(cargo_data[None, :], (3, 18)),
            direct, memories], axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions.astype(jnp.int32), 0, self.num_actions - 1)
        valid = (actions >= 0) & (actions < self.num_actions)
        valid = valid & self.legal_actions(state)[jnp.arange(3), clipped]
        proposed, rewards = self.advance(key, state, jnp.where(valid, clipped, 0))
        next_state = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), proposed, state)
        rewards = jnp.where(state['done'], jnp.zeros((3,), jnp.float32), rewards)
        dones = jnp.full((3,), next_state['done'], jnp.bool_)
        info = {'done_all': next_state['done'], 'terminated': next_state['terminated'], 'truncated': next_state['truncated']}
        return self.observe(next_state), next_state, rewards, dones, info
