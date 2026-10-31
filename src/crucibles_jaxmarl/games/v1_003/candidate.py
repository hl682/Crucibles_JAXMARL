import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        state = {'t': jnp.array(0, jnp.int32), 'mode': jax.random.randint(key, (), 0, 2, dtype=jnp.int32),
                 'signal': jnp.full((2,), -1.0, jnp.float32), 'age': jnp.ones((2,), jnp.float32),
                 'project': jnp.full((2,), -1, jnp.int32), 'timer': jnp.zeros((2,), jnp.int32),
                 'pressure': jnp.array(0.0, jnp.float32), 'outcome': jnp.zeros((2,), jnp.float32)}
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'] / 32.0)
        obs = obs.at[:, 1].set(state['signal'])
        obs = obs.at[:, 2].set(state['age'])
        obs = obs.at[:, 3].set(state['project'].astype(jnp.float32))
        obs = obs.at[:, 4].set(state['timer'] / 2.0)
        obs = obs.at[:, 5].set(jnp.roll(state['timer'], 1) / 2.0)
        obs = obs.at[:, 6].set(state['pressure'])
        obs = obs.at[:, 7].set(state['outcome'])
        obs = obs.at[:, 8].set(jnp.roll(state['outcome'], 1))
        obs = obs.at[:, 9].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        # 0 wait; 1 privately sense; 2 start mode-0 project; 3 start mode-1 project; 4 push mode toward own index.
        sensing = actions == 1
        pushing = (actions == 4).astype(jnp.float32)
        pressure = jnp.clip(0.65 * state['pressure'] + 0.6 * (pushing[1] - pushing[0]), -1.0, 1.0)
        drift = jax.random.bernoulli(key, 0.08)
        mode = jnp.where(pressure > 0.5, 1, jnp.where(pressure < -0.5, 0, jnp.where(drift, 1 - state['mode'], state['mode']))).astype(jnp.int32)
        mature = state['timer'] == 1
        outcome = jnp.where(mature, jnp.where(state['project'] == mode, 0.8, -0.6), 0.0).astype(jnp.float32)
        starts = (state['timer'] == 0) & ((actions == 2) | (actions == 3))
        timer = jnp.where(starts, 2, jnp.maximum(0, state['timer'] - 1)).astype(jnp.int32)
        project = jnp.where(starts, actions - 2, jnp.where(timer > 0, state['project'], -1)).astype(jnp.int32)
        rewards = jnp.tanh(outcome - 0.12 * sensing - 0.18 * pushing - 0.1 * starts).astype(jnp.float32)
        next_state = {'t': state['t'] + jnp.array(1, jnp.int32), 'mode': mode,
                      'signal': jnp.where(sensing, mode.astype(jnp.float32), state['signal']),
                      'age': jnp.where(sensing, 0.0, jnp.minimum(1.0, state['age'] + 0.1)),
                      'project': project, 'timer': timer, 'pressure': pressure, 'outcome': outcome}
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
