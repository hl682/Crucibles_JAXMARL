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
            'reservoir': jnp.array(0.7, dtype=jnp.float32),
            'soil': jax.random.uniform(key, (2,), minval=0.2, maxval=0.5),
            'pipe': jnp.zeros((2,), dtype=jnp.float32),
            'crop': jnp.zeros((2,), dtype=jnp.float32),
            'gate': jnp.ones((2,), dtype=jnp.float32) * 0.5,
            'draw': jnp.zeros((2,), dtype=jnp.float32),
            'harvest': jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'] / 32.0)
        obs = obs.at[:, 1].set(state['reservoir'])
        obs = obs.at[:, 2].set(state['soil'])
        obs = obs.at[:, 3].set(state['pipe'])
        obs = obs.at[:, 4].set(state['crop'])
        obs = obs.at[:, 5].set(state['gate'])
        obs = obs.at[:, 6].set(state['gate'][::-1])
        obs = obs.at[:, 7].set(state['draw'])
        obs = obs.at[:, 8].set(state['draw'][::-1])
        obs = obs.at[:, 9].set(state['harvest'])
        obs = obs.at[:, 10].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 11].set(1.0 - state['soil'])
        obs = obs.at[:, 12].set(state['pipe'][::-1])
        obs = obs.at[:, 13].set(state['crop'][::-1])
        obs = obs.at[:, 14].set(0.1)
        obs = obs.at[:, 15].set(1.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        gate = jnp.clip(state['gate'] + 0.25 * (actions == 1) - 0.25 * (actions == 2), 0.0, 1.0)
        demand = 0.18 * gate + 0.15 * (actions == 3)
        supply = jnp.minimum(1.0, state['reservoir'] + 0.1)
        draw = demand * jnp.minimum(1.0, supply / jnp.maximum(jnp.sum(demand), 0.001))
        moisture = jnp.clip(0.82 * state['soil'] + state['pipe'], 0.0, 1.0)
        growth = 0.15 * jnp.maximum(0.0, 1.0 - jnp.abs(moisture - 0.55) / 0.4)
        ripe = jnp.minimum(1.0, state['crop'] + growth)
        harvest = ripe * (actions == 4)
        crop = ripe * (actions != 4)
        reservoir = jnp.clip(supply - jnp.sum(draw) + 0.03 * jnp.sum(actions == 2), 0.0, 1.0)
        rewards = (harvest - 0.04 * (actions == 3) - 0.02 * (actions == 4)).astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'reservoir': reservoir, 'soil': moisture, 'pipe': draw,
            'crop': crop, 'gate': gate, 'draw': draw, 'harvest': harvest,
        }
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        return self.observe(next_state), next_state, rewards, dones, {
            'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
