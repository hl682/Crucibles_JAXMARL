import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        keys = jax.random.split(key, 2)
        state = {
            't': jnp.array(0, dtype=jnp.int32),
            'position': jax.random.uniform(keys[0], (2,), dtype=jnp.float32, minval=-0.6, maxval=0.6),
            'velocity': jnp.zeros((2,), dtype=jnp.float32),
            'phase': jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.0, maxval=6.283185307),
            'coupler': jnp.ones((2,), dtype=jnp.float32),
            'energy': jnp.ones((2,), dtype=jnp.float32),
            'seen_position': jnp.zeros((2,), dtype=jnp.float32),
            'seen_velocity': jnp.zeros((2,), dtype=jnp.float32),
            'seen_age': jnp.ones((2,), dtype=jnp.float32),
            'last_impulse': jnp.zeros((2,), dtype=jnp.float32),
            'last_reward': jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        time = state['t'].astype(jnp.float32)
        target = 0.55 * jnp.sin(0.17 * time + state['phase'])
        preview = 0.55 * jnp.sin(0.17 * (time + 1.0) + state['phase'])
        strain = jnp.abs(state['position'][0] - state['position'][1]) / 3.0
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(time / 32.0)
        obs = obs.at[:, 1].set(state['position'] / 1.5)
        obs = obs.at[:, 2].set(state['velocity'] / 0.6)
        obs = obs.at[:, 3].set(target)
        obs = obs.at[:, 4].set(state['energy'])
        obs = obs.at[:, 5].set(state['coupler'])
        obs = obs.at[:, 6].set(state['coupler'][::-1])
        obs = obs.at[:, 7].set(state['seen_position'] / 1.5)
        obs = obs.at[:, 8].set(state['seen_velocity'] / 0.6)
        obs = obs.at[:, 9].set(state['seen_age'])
        obs = obs.at[:, 10].set(strain)
        obs = obs.at[:, 11].set(state['last_impulse'])
        obs = obs.at[:, 12].set((state['position'] - target) / 2.05)
        obs = obs.at[:, 13].set(state['last_reward'])
        obs = obs.at[:, 14].set(preview)
        obs = obs.at[:, 15].set(state['energy'][::-1])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        pulse_available = (state['energy'] >= 0.12).astype(jnp.float32)
        impulse = ((actions == 1).astype(jnp.float32) - (actions == 2).astype(jnp.float32)) * pulse_available
        switch = jnp.logical_and(actions == 3, state['energy'] >= 0.08)
        probe = jnp.logical_and(actions == 4, state['energy'] >= 0.06)
        switched = switch.astype(jnp.float32)
        probed = probe.astype(jnp.float32)
        coupling_force = 0.08 * state['coupler'] * (state['position'][::-1] - state['position'])
        velocity = 0.86 * state['velocity'] - 0.1 * state['position'] + coupling_force + 0.19 * impulse
        velocity = jnp.clip(jnp.where(probe, 0.5 * velocity, velocity), -0.6, 0.6)
        position = jnp.clip(state['position'] + velocity, -1.5, 1.5)
        coupler = jnp.where(switch, -state['coupler'], state['coupler'])
        energy = jnp.clip(state['energy'] + 0.045 + 0.05 * (actions == 0).astype(jnp.float32) - 0.12 * jnp.abs(impulse) - 0.08 * switched - 0.06 * probed, 0.0, 1.0)
        next_t = state['t'] + jnp.array(1, dtype=jnp.int32)
        target = 0.55 * jnp.sin(0.17 * next_t.astype(jnp.float32) + state['phase'])
        rewards = jnp.tanh(0.65 - 1.3 * jnp.square(position - target) - 0.22 * jnp.square(velocity) - 0.07 * jnp.abs(impulse) - 0.025 * switched - 0.02 * probed)
        next_state = {
            't': next_t,
            'position': position,
            'velocity': velocity,
            'phase': state['phase'],
            'coupler': coupler,
            'energy': energy,
            'seen_position': jnp.where(probe, position[::-1], state['seen_position']),
            'seen_velocity': jnp.where(probe, velocity[::-1], state['seen_velocity']),
            'seen_age': jnp.where(probe, 0.0, jnp.minimum(1.0, state['seen_age'] + 1.0 / 32.0)),
            'last_impulse': impulse,
            'last_reward': rewards,
        }
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards.astype(jnp.float32), dones, info
