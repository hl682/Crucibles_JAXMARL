import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        state = {'t': jnp.array(0, jnp.int32), 'due': jnp.zeros((2,), jnp.int32),
                 'work': jnp.zeros((2,), jnp.float32), 'trust': jnp.full((2,), 0.5, jnp.float32),
                 'escrow': jnp.zeros((2,), jnp.float32), 'last': jnp.zeros((2,), jnp.float32)}
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'] / 32.0)
        obs = obs.at[:, 1].set(state['due'] / 3.0)
        obs = obs.at[:, 2].set(jnp.roll(state['due'], 1) / 3.0)
        obs = obs.at[:, 3].set(state['work'] / 2.0)
        obs = obs.at[:, 4].set(jnp.roll(state['work'], 1) / 2.0)
        obs = obs.at[:, 5].set(state['trust'])
        obs = obs.at[:, 6].set(jnp.roll(state['trust'], 1))
        obs = obs.at[:, 7].set(state['escrow'])
        obs = obs.at[:, 8].set(jnp.roll(state['escrow'], 1))
        obs = obs.at[:, 9].set(state['last'])
        obs = obs.at[:, 10].set(jnp.roll(state['last'], 1))
        obs = obs.at[:, 11].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        # 0 wait; 1 open promise; 2 work; 3 endorse partner; 4 withdraw promise.
        active = state['due'] > 0
        opened = (actions == 1) & (~active)
        effort = (actions == 2) & active
        withdrawn = (actions == 4) & active
        endorsement = (actions == 3).astype(jnp.float32)
        work = jnp.minimum(2.0, state['work'] + effort.astype(jnp.float32))
        mature = (state['due'] == 1) & (~withdrawn)
        fulfilled = mature & (work >= 2.0)
        broken = (mature & (~fulfilled)) | withdrawn
        strength = 0.3 + 0.5 * state['trust'] + 0.2 * state['escrow']
        received = jnp.roll(fulfilled.astype(jnp.float32) * strength, 1)
        damage = jnp.roll(broken.astype(jnp.float32) * (0.2 + state['escrow']), 1)
        rewards = jnp.tanh(received - damage + 0.25 * fulfilled.astype(jnp.float32)
                           - 0.15 * effort.astype(jnp.float32) - 0.1 * endorsement
                           - 0.25 * broken.astype(jnp.float32)).astype(jnp.float32)
        due = jnp.where(opened, 3, jnp.maximum(0, state['due'] - 1))
        due = jnp.where(withdrawn, 0, due).astype(jnp.int32)
        next_state = {'t': state['t'] + jnp.array(1, jnp.int32), 'due': due,
                      'work': jnp.where(due > 0, work, 0.0).astype(jnp.float32),
                      'trust': jnp.clip(state['trust'] + 0.15 * fulfilled - 0.25 * broken, 0.0, 1.0),
                      'escrow': jnp.where(due > 0, jnp.minimum(1.0, state['escrow'] + jnp.roll(endorsement, 1) * 0.5), 0.0),
                      'last': fulfilled.astype(jnp.float32) - broken.astype(jnp.float32)}
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
