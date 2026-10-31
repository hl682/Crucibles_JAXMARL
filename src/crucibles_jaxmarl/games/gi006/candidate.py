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
            'pollution': jax.random.uniform(key, (2,), minval=0.1, maxval=0.3),
            'filter': jnp.ones((2,), dtype=jnp.float32) * 0.2,
            'stock': jnp.ones((2,), dtype=jnp.float32) * 0.5,
            'plume': jnp.zeros((2,), dtype=jnp.float32),
            'gate': jnp.ones((2,), dtype=jnp.float32),
            'output': jnp.zeros((2,), dtype=jnp.float32),
            'cleanup': jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'] / 32.0)
        obs = obs.at[:, 1].set(state['pollution'])
        obs = obs.at[:, 2].set(state['pollution'][::-1])
        obs = obs.at[:, 3].set(state['filter'])
        obs = obs.at[:, 4].set(state['stock'])
        obs = obs.at[:, 5].set(state['plume'])
        obs = obs.at[:, 6].set(state['plume'][::-1])
        obs = obs.at[:, 7].set(state['gate'])
        obs = obs.at[:, 8].set(state['gate'][::-1])
        obs = obs.at[:, 9].set(state['output'])
        obs = obs.at[:, 10].set(state['output'][::-1])
        obs = obs.at[:, 11].set(state['cleanup'])
        obs = obs.at[:, 12].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 13].set(1.0 - state['filter'])
        obs = obs.at[:, 14].set(1.0 - state['pollution'])
        obs = obs.at[:, 15].set(1.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        gate = jnp.where(actions == 3, 0.0, jnp.where(actions == 4, 1.0, state['gate']))
        available = jnp.minimum(1.0, state['stock'] + 0.08)
        effort = jnp.minimum(available, 0.2)
        upkeep = effort * (actions == 2)
        cleanup = effort * (actions == 1)
        filtration = jnp.clip(0.94 * state['filter'] + upkeep, 0.0, 1.0)
        output = (actions == 0) * (1.0 - state['pollution']) * (0.4 + 0.6 * gate)
        emissions = 0.24 * output * (1.0 - filtration)
        export = 0.3 * gate * state['pollution']
        pollution = jnp.clip(0.9 * state['pollution'] - export + state['plume'] * gate + emissions - cleanup, 0.0, 1.0)
        plume = export[::-1]
        stock = jnp.clip(available - upkeep - cleanup + 0.12 * output, 0.0, 1.0)
        rewards = (0.45 * output - 0.5 * pollution - 0.1 * (upkeep + cleanup)).astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'pollution': pollution, 'filter': filtration, 'stock': stock,
            'plume': plume, 'gate': gate, 'output': output, 'cleanup': cleanup,
        }
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        return self.observe(next_state), next_state, rewards, dones, {
            'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
