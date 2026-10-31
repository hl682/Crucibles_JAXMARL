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
            'target': jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32),
            'dial': jnp.zeros((2,), dtype=jnp.int32),
            'inbox': jnp.full((2,), -1, dtype=jnp.int32),
            'wire': jnp.full((2,), -1, dtype=jnp.int32),
            'charge': jnp.zeros((2,), dtype=jnp.float32),
            'opened': jnp.array(0.0, dtype=jnp.float32),
            'jam': jnp.array(0.0, dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'] / 32.0)
        obs = obs.at[:, 1].set(state['target'])
        obs = obs.at[:, 2].set(state['dial'])
        obs = obs.at[:, 3].set(state['inbox'])
        obs = obs.at[:, 4].set(state['inbox'] >= 0)
        obs = obs.at[:, 5].set(state['charge'])
        obs = obs.at[:, 6].set(state['charge'][::-1])
        obs = obs.at[:, 7].set(state['opened'])
        obs = obs.at[:, 8].set(state['jam'])
        obs = obs.at[:, 9].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 10].set(state['wire'][::-1])
        obs = obs.at[:, 11].set(state['wire'][::-1] >= 0)
        obs = obs.at[:, 12].set(1.0 - state['charge'])
        obs = obs.at[:, 13].set(jnp.mod(state['t'], 2))
        obs = obs.at[:, 14].set(1.0 - state['jam'])
        obs = obs.at[:, 15].set(1.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        dial = jnp.where(actions < 2, actions, state['dial']).astype(jnp.int32)
        message = jnp.where((actions == 2) | (actions == 3), actions - 2, -1).astype(jnp.int32)
        wire = message[::-1]
        inbox = jnp.where(state['wire'] >= 0, state['wire'], state['inbox'])
        charge = jnp.clip(state['charge'] + 0.4 * (actions == 4) - 0.15 * (actions != 4), 0.0, 1.0)
        ready = jnp.all(charge >= 0.8)
        correct = jnp.all(dial == state['target'][::-1])
        opened = ready & correct & (state['jam'] <= 0.0)
        failed = ready & jnp.logical_not(correct) & (state['jam'] <= 0.0)
        new_target = jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32)
        target = jnp.where(opened, new_target, state['target'])
        jam = jnp.where(failed, 1.0, jnp.maximum(0.0, state['jam'] - 0.25))
        charge = jnp.where(opened | failed, jnp.zeros((2,), dtype=jnp.float32), charge)
        rewards = (0.8 * jnp.ones((2,), dtype=jnp.float32) * opened - 0.2 * failed - 0.015 * ((actions == 2) | (actions == 3))).astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'target': target, 'dial': dial,
            'inbox': jnp.where(opened, jnp.full((2,), -1, dtype=jnp.int32), inbox),
            'wire': jnp.where(opened, jnp.full((2,), -1, dtype=jnp.int32), wire),
            'charge': charge, 'opened': opened.astype(jnp.float32), 'jam': jam,
        }
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        return self.observe(next_state), next_state, rewards, dones, {
            'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
