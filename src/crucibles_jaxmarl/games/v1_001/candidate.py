import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        state = {'t': jnp.array(0, jnp.int32), 'stock': jnp.array(0.8, jnp.float32),
                 'soil': jnp.array(0.6, jnp.float32), 'queue': jnp.zeros((2,), jnp.float32),
                 'filters': jnp.zeros((2,), jnp.float32), 'yield': jnp.zeros((2,), jnp.float32)}
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'] / 32.0)
        obs = obs.at[:, 1].set(state['stock'])
        obs = obs.at[:, 2].set(state['soil'])
        obs = obs.at[:, 3].set(state['queue'][0])
        obs = obs.at[:, 4].set(state['queue'][1])
        obs = obs.at[:, 5].set(state['filters'])
        obs = obs.at[:, 6].set(jnp.roll(state['filters'], 1))
        obs = obs.at[:, 7].set(state['yield'])
        obs = obs.at[:, 8].set(jnp.roll(state['yield'], 1))
        obs = obs.at[:, 9].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        # 0 rest; 1 gentle draw; 2 deep draw; 3 restore soil; 4 install filter.
        demand = 0.18 * (actions == 1) + 0.5 * (actions == 2)
        allocation = demand * jnp.minimum(1.0, state['stock'] / jnp.maximum(0.01, jnp.sum(demand)))
        tending = (actions == 3).astype(jnp.float32)
        installing = (actions == 4).astype(jnp.float32)
        runoff = jnp.sum(allocation * (actions == 2) * (1.0 - state['filters']))
        soil = jnp.clip(state['soil'] + 0.08 * jnp.sum(tending) - 0.35 * state['queue'][0] - 0.01, 0.0, 1.0)
        renewal = 0.04 + 0.20 * soil * (1.0 - state['stock'])
        stock = jnp.clip(state['stock'] - jnp.sum(allocation) + renewal, 0.0, 1.0)
        rewards = jnp.tanh(2.0 * allocation - 0.12 * tending - 0.15 * installing
                           - 0.25 * state['queue'][0]).astype(jnp.float32)
        next_state = {'t': state['t'] + jnp.array(1, jnp.int32), 'stock': stock, 'soil': soil,
                      'queue': jnp.stack((state['queue'][1], runoff)).astype(jnp.float32),
                      'filters': jnp.clip(0.92 * state['filters'] + 0.3 * installing, 0.0, 1.0),
                      'yield': allocation.astype(jnp.float32)}
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
