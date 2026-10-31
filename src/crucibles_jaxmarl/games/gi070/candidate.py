import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        draws = jax.random.uniform(key, (6,), dtype=jnp.float32)
        state = {
            't': jnp.array(0, dtype=jnp.int32),
            'crop': jnp.full((2,), -1, dtype=jnp.int32),
            'growth': jnp.zeros((2,), dtype=jnp.float32),
            'health': jnp.zeros((2,), dtype=jnp.float32),
            'nutrients': 0.50 + 0.30 * draws[:2],
            'aquifer': 0.65 + 0.20 * draws[2],
            'weather': draws[3],
            'preference': jnp.floor(2.0 * draws[4:]).astype(jnp.int32),
            'pollen': jnp.full((2,), -1, dtype=jnp.int32),
            'harvest': jnp.zeros((2,), dtype=jnp.float32),
            'last_action': jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['nutrients'])
        obs = obs.at[:, 2].set(state['nutrients'][::-1])
        obs = obs.at[:, 3].set(state['aquifer'])
        obs = obs.at[:, 4].set(state['weather'])
        obs = obs.at[:, 5].set((state['crop'].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 6].set(state['growth'])
        obs = obs.at[:, 7].set(state['health'])
        obs = obs.at[:, 8].set(state['preference'].astype(jnp.float32))
        obs = obs.at[:, 9].set((state['pollen'].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 10].set(state['growth'][::-1])
        obs = obs.at[:, 11].set((state['crop'][::-1] >= 0).astype(jnp.float32))
        obs = obs.at[:, 12].set(state['harvest'])
        obs = obs.at[:, 13].set(state['harvest'][::-1])
        obs = obs.at[:, 14].set(state['last_action'].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 15].set(state['last_action'][::-1].astype(jnp.float32) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        active = state['crop'] >= 0
        plant = ((actions == 1) | (actions == 2)) & (~active) & (state['nutrients'] >= 0.20)
        variety = (actions - 1).astype(jnp.int32)
        irrigate = (actions == 3) & active
        release = (actions == 4) & active
        requests = jnp.sum(irrigate.astype(jnp.float32))
        ration = jnp.minimum(0.16, state['aquifer'] / jnp.maximum(requests, 1.0))
        water = irrigate.astype(jnp.float32) * ration
        incoming = (state['pollen'] >= 0) & active
        compatible = incoming & (state['pollen'] != state['crop'])
        incompatible = incoming & (state['pollen'] == state['crop'])
        drought = active & (state['weather'] < 0.25) & (water < 0.01)
        health = jnp.clip(state['health'] + 0.22 * compatible.astype(jnp.float32) - 0.14 * incompatible.astype(jnp.float32) - 0.02 * drought.astype(jnp.float32), 0.0, 1.0)
        growth = jnp.clip(state['growth'] + active.astype(jnp.float32) * (0.09 + 0.03 * state['weather']) + 0.18 * water / 0.16 - 0.07 * release.astype(jnp.float32), 0.0, 1.20)
        harvested = active & (growth >= 1.0)
        preferred = state['crop'] == state['preference']
        harvest_value = harvested.astype(jnp.float32) * (0.65 * health + 0.15 * preferred.astype(jnp.float32))
        crop = jnp.where(plant, variety, jnp.where(harvested, -1, state['crop'])).astype(jnp.int32)
        next_growth = jnp.where(plant, 0.05, jnp.where(harvested, 0.0, growth))
        next_health = jnp.where(plant, 0.45 + 0.20 * state['nutrients'], jnp.where(harvested, 0.0, health))
        nutrients = jnp.clip(state['nutrients'] + 0.07 * (actions == 0).astype(jnp.float32) - 0.20 * plant.astype(jnp.float32) - 0.15 * harvested.astype(jnp.float32), 0.0, 1.0)
        aquifer = jnp.clip(state['aquifer'] - jnp.sum(water) + 0.09 + 0.06 * state['weather'], 0.0, 1.0)
        pollen = jnp.where(release, state['crop'], -1)[::-1].astype(jnp.int32)
        utility = harvest_value - 0.12 * plant.astype(jnp.float32) - 0.035 * water / 0.16 - 0.04 * release.astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'crop': crop,
            'growth': next_growth.astype(jnp.float32),
            'health': next_health.astype(jnp.float32),
            'nutrients': nutrients.astype(jnp.float32),
            'aquifer': aquifer.astype(jnp.float32),
            'weather': jax.random.uniform(key, (), dtype=jnp.float32),
            'preference': state['preference'],
            'pollen': pollen,
            'harvest': harvest_value.astype(jnp.float32),
            'last_action': actions.astype(jnp.int32),
        }
        rewards = jnp.tanh(2.0 * utility).astype(jnp.float32)
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
