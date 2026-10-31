import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        state = {
            't': jnp.array(0, dtype=jnp.int32),
            'position': jnp.array(0.0, dtype=jnp.float32),
            'velocity': jnp.array(0.0, dtype=jnp.float32),
            'gear': jnp.array([1, -1], dtype=jnp.int32),
            'clutch': jnp.zeros((2,), dtype=jnp.bool_),
            'heat': jnp.zeros((2,), dtype=jnp.float32),
            'energy': jnp.full((2,), 0.75, dtype=jnp.float32),
            'goal_index': jax.random.randint(key, (2,), 0, 4, dtype=jnp.int32),
            'delivered': jnp.zeros((2,), dtype=jnp.float32),
            'last_action': jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        goal = -0.60 + 0.40 * state['goal_index'].astype(jnp.float32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set((state['position'] + 1.0) / 2.0)
        obs = obs.at[:, 2].set(state['velocity'] / 0.22)
        obs = obs.at[:, 3].set((goal + 1.0) / 2.0)
        obs = obs.at[:, 4].set(state['gear'].astype(jnp.float32))
        obs = obs.at[:, 5].set(state['clutch'].astype(jnp.float32))
        obs = obs.at[:, 6].set(state['heat'])
        obs = obs.at[:, 7].set(state['energy'])
        obs = obs.at[:, 8].set(state['gear'][::-1].astype(jnp.float32))
        obs = obs.at[:, 9].set(state['clutch'][::-1].astype(jnp.float32))
        obs = obs.at[:, 10].set(state['heat'][::-1])
        obs = obs.at[:, 11].set(state['energy'][::-1])
        obs = obs.at[:, 12].set(state['delivered'])
        obs = obs.at[:, 13].set(state['delivered'][::-1])
        obs = obs.at[:, 14].set(state['last_action'].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 15].set(state['last_action'][::-1].astype(jnp.float32) / 4.0)
        return obs

    def legal_actions(self, state):
        mask = jnp.ones((2, 5), dtype=jnp.bool_)
        mask = mask.at[:, 1].set(~state['clutch'])
        mask = mask.at[:, 2].set(state['clutch'])
        mask = mask.at[:, 3].set(~state['clutch'])
        mask = mask.at[:, 4].set(state['clutch'] & (state['energy'] >= 0.16) & (state['heat'] < 0.95))
        return mask

    def raw_step(self, key, state, actions):
        coast = actions == 0
        engage = actions == 1
        disengage = actions == 2
        shift = (actions == 3) & (~state['clutch'])
        pulse = (actions == 4) & state['clutch'] & (state['energy'] >= 0.16) & (state['heat'] < 0.95)
        clutch = (state['clutch'] | engage) & (~disengage)
        gear = jnp.where(shift, -state['gear'], state['gear']).astype(jnp.int32)
        opposed = clutch[0] & clutch[1] & (gear[0] != gear[1])
        torque = 0.09 * pulse.astype(jnp.float32) * gear.astype(jnp.float32) * (1.0 - 0.70 * state['heat'])
        drag = 0.84 - 0.12 * jnp.sum(clutch.astype(jnp.float32))
        velocity = jnp.clip(drag * state['velocity'] + jnp.sum(torque) * jnp.where(opposed, 0.45, 1.0), -0.22, 0.22)
        proposed_position = state['position'] + velocity
        collision = jnp.abs(proposed_position) > 0.95
        position = jnp.clip(proposed_position, -0.95, 0.95)
        velocity = jnp.where(collision, -0.30 * velocity, velocity)
        grinding = opposed & jnp.any(pulse)
        heat = jnp.clip(state['heat'] + 0.15 * pulse.astype(jnp.float32) + 0.10 * grinding.astype(jnp.float32) + 0.07 * collision.astype(jnp.float32) * clutch.astype(jnp.float32) - 0.08 * coast.astype(jnp.float32) - 0.025 * (~clutch).astype(jnp.float32), 0.0, 1.0)
        energy = jnp.clip(state['energy'] + 0.12 * coast.astype(jnp.float32) - 0.16 * pulse.astype(jnp.float32) - 0.015 * clutch.astype(jnp.float32), 0.0, 1.0)
        goal = -0.60 + 0.40 * state['goal_index'].astype(jnp.float32)
        before_distance = jnp.abs(state['position'] - goal)
        after_distance = jnp.abs(position - goal)
        delivered = (after_distance < 0.08) & (jnp.abs(velocity) < 0.045)
        offset = jax.random.randint(key, (2,), 1, 4, dtype=jnp.int32)
        replacement_goal = jnp.mod(state['goal_index'] + offset, 4)
        goal_index = jnp.where(delivered, replacement_goal, state['goal_index'])
        utility = 0.65 * delivered.astype(jnp.float32) + 0.30 * (before_distance - after_distance) - 0.055 * pulse.astype(jnp.float32) - 0.02 * jnp.maximum(heat - 0.85, 0.0) - 0.04 * collision.astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'position': position.astype(jnp.float32),
            'velocity': velocity.astype(jnp.float32),
            'gear': gear,
            'clutch': clutch.astype(jnp.bool_),
            'heat': heat.astype(jnp.float32),
            'energy': energy.astype(jnp.float32),
            'goal_index': goal_index.astype(jnp.int32),
            'delivered': delivered.astype(jnp.float32),
            'last_action': actions.astype(jnp.int32),
        }
        rewards = jnp.tanh(2.0 * utility).astype(jnp.float32)
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
