import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        keys = jax.random.split(key, 3)
        state = {
            't': jnp.array(0, dtype=jnp.int32),
            'biomass': jax.random.uniform(keys[0], (2,), dtype=jnp.float32, minval=0.15, maxval=0.25),
            'moisture': jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.4, maxval=0.7),
            'flower': jnp.zeros((2,), dtype=jnp.float32),
            'pollen': jnp.zeros((2,), dtype=jnp.float32),
            'spore': jnp.zeros((2,), dtype=jnp.float32),
            'fruit': jnp.zeros((2,), dtype=jnp.float32),
            'soil': jax.random.uniform(keys[2], (2,), dtype=jnp.float32, minval=0.6, maxval=1.0),
            'seed_bank': jnp.full((2,), 0.7, dtype=jnp.float32),
            'blight': jnp.zeros((2,), dtype=jnp.float32),
            'last_yield': jnp.zeros((2,), dtype=jnp.float32),
            'last_action': jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['biomass'])
        obs = obs.at[:, 2].set(state['moisture'])
        obs = obs.at[:, 3].set(state['flower'])
        obs = obs.at[:, 4].set(state['pollen'])
        obs = obs.at[:, 5].set(state['fruit'])
        obs = obs.at[:, 6].set(state['soil'])
        obs = obs.at[:, 7].set(state['seed_bank'])
        obs = obs.at[:, 8].set(state['blight'])
        obs = obs.at[:, 9].set(jnp.roll(state['biomass'], 1))
        obs = obs.at[:, 10].set(jnp.roll(state['flower'], 1))
        obs = obs.at[:, 11].set(jnp.roll(state['fruit'], 1))
        obs = obs.at[:, 12].set(jnp.roll(state['blight'], 1))
        obs = obs.at[:, 13].set(state['last_yield'])
        obs = obs.at[:, 14].set(jnp.roll(state['last_yield'], 1))
        obs = obs.at[:, 15].set(jnp.roll(state['last_action'], 1).astype(jnp.float32) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        rest = (actions == 0).astype(jnp.float32)
        seed_spent = jnp.minimum(state['seed_bank'], 0.25) * (actions == 1).astype(jnp.float32)
        irrigation = 0.35 * (actions == 2).astype(jnp.float32)
        emission = jnp.minimum(state['biomass'], 0.25) * (actions == 3).astype(jnp.float32)
        harvest = jnp.minimum(state['fruit'], 0.6) * (actions == 4).astype(jnp.float32)
        fertilized = jnp.minimum(state['flower'], state['pollen']) * (0.5 + 0.5 * state['soil']) * (1.0 - state['blight'])
        shade = jnp.maximum(jnp.sum(state['biomass']) - 1.2, 0.0)
        growth = 0.07 * state['soil'] * state['moisture'] * (1.0 - state['biomass']) * (1.0 - 0.6 * state['blight'])
        rain = jax.random.uniform(key, (), dtype=jnp.float32, minval=0.0, maxval=0.06)
        wetted = state['moisture'] + irrigation + rain
        soaked = (wetted > 1.0).astype(jnp.float32)
        usable_yield = harvest * (1.0 - 0.4 * state['blight'])
        rewards = jnp.clip(
            0.80 * usable_yield + 0.12 * jnp.roll(usable_yield, 1)
            - 0.04 * irrigation - 0.02 * seed_spent - 0.02 * emission
            - 0.08 * state['blight'],
            -1.0, 1.0,
        ).astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'biomass': jnp.clip(state['biomass'] + growth + 0.70 * seed_spent - 0.35 * emission - 0.02 * harvest - 0.03 * shade, 0.0, 1.0),
            'moisture': jnp.clip(0.86 * state['moisture'] + irrigation + rain - 0.08 * state['biomass'] - 0.05 * emission + 0.04 * rest, 0.0, 1.0),
            'flower': jnp.clip(0.65 * state['flower'] + emission - fertilized, 0.0, 1.0),
            'pollen': jnp.roll(emission, 1),
            'spore': jnp.roll(emission * state['blight'], 1),
            'fruit': jnp.clip(0.95 * state['fruit'] + 1.40 * fertilized - harvest - 0.12 * state['blight'] * state['fruit'], 0.0, 1.0),
            'soil': jnp.clip(state['soil'] + 0.02 * rest - 0.015 * irrigation - 0.03 * seed_spent, 0.25, 1.0),
            'seed_bank': jnp.clip(state['seed_bank'] + 0.035 * rest + 0.15 * harvest - seed_spent, 0.0, 1.0),
            'blight': jnp.clip(0.85 * state['blight'] + 0.18 * soaked + 0.50 * state['spore'] - 0.18 * rest - 0.05 * harvest, 0.0, 1.0),
            'last_yield': usable_yield,
            'last_action': actions.astype(jnp.int32),
        }
        done_all = next_state['t'] >= 32
        dones = jnp.stack([done_all, done_all]).astype(jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
