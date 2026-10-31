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
            'adults': jax.random.uniform(keys[0], (2, 2), dtype=jnp.float32, minval=0.08, maxval=0.25),
            'seed_bank': jnp.zeros((2, 2), dtype=jnp.float32),
            'seedlings': jnp.zeros((2, 2), dtype=jnp.float32),
            'nutrient': jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.6, maxval=1.0),
            'water': jax.random.uniform(keys[2], (), dtype=jnp.float32, minval=0.6, maxval=1.6),
            'gate': jnp.zeros((2,), dtype=jnp.float32),
            'intake': jnp.zeros((2,), dtype=jnp.float32),
            'seen_mass': jnp.zeros((2,), dtype=jnp.float32),
            'seen_mix': jnp.zeros((2,), dtype=jnp.float32),
            'seen_age': jnp.ones((2,), dtype=jnp.float32),
            'last_yield': jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1:3].set(state['adults'])
        obs = obs.at[:, 3:5].set(state['seed_bank'])
        obs = obs.at[:, 5:7].set(state['seedlings'])
        obs = obs.at[:, 7].set(state['nutrient'])
        obs = obs.at[:, 8].set(state['gate'])
        obs = obs.at[:, 9].set(state['gate'][::-1])
        obs = obs.at[:, 10].set(state['water'] / 1.8)
        obs = obs.at[:, 11].set(state['intake'])
        obs = obs.at[:, 12].set(state['seen_mass'])
        obs = obs.at[:, 13].set(state['seen_mix'])
        obs = obs.at[:, 14].set(state['seen_age'])
        obs = obs.at[:, 15].set(state['last_yield'])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        can_sow = (state['nutrient'] >= 0.08).astype(jnp.float32)
        sow = jnp.stack(((actions == 1).astype(jnp.float32), (actions == 2).astype(jnp.float32)), axis=1) * can_sow[:, None]
        sow_count = jnp.sum(sow, axis=1)
        harvest = (actions == 3).astype(jnp.float32)
        inspect = actions == 4
        toggled = inspect.astype(jnp.float32)
        demand = 0.08 + 0.12 * state['adults'][:, 0] + 0.3 * state['adults'][:, 1] + 0.12 * state['gate']
        request = demand * (0.4 + 0.6 * state['gate'])
        recharge = 0.2 + 0.07 * jnp.cos(0.3 * state['t'].astype(jnp.float32))
        available_water = jnp.minimum(1.8, state['water'] + recharge)
        allocation = jnp.minimum(1.0, available_water / (jnp.sum(request) + 0.000001))
        intake = allocation * request
        hydration = jnp.clip(intake / (demand + 0.000001), 0.0, 1.0)
        harvested = 0.7 * harvest[:, None] * state['adults']
        yield_value = jnp.sum(harvested * jnp.array([0.45, 1.0], dtype=jnp.float32)[None, :], axis=1)
        residual = state['adults'] - harvested
        survival = jnp.stack((0.94 + 0.03 * hydration, 0.78 + 0.18 * hydration), axis=1)
        growth = 0.32 * state['seedlings'] * hydration[:, None] * state['nutrient'][:, None] * jnp.array([0.7, 1.2], dtype=jnp.float32)[None, :]
        raw_adults = residual * survival + growth
        adults = raw_adults / jnp.maximum(1.0, jnp.sum(raw_adults, axis=1))[:, None]
        space = jnp.maximum(0.0, 1.0 - jnp.sum(adults, axis=1))
        seedlings = jnp.clip(0.45 * state['seedlings'] + 0.35 * state['seed_bank'] * space[:, None], 0.0, 1.0)
        connected = state['gate'] * state['gate'][::-1]
        incoming = 0.06 * connected[:, None] * state['adults'][::-1, :]
        seed_bank = jnp.clip(0.65 * state['seed_bank'] + 0.12 * state['adults'] + incoming + 0.35 * sow, 0.0, 1.0)
        nutrient = jnp.clip(state['nutrient'] + 0.035 + 0.09 * jnp.sum(harvested, axis=1) + 0.03 * state['adults'][:, 0] - 0.09 * jnp.sum(growth, axis=1) - 0.045 * state['adults'][:, 1] - 0.08 * sow_count, 0.05, 1.0)
        drought = jnp.maximum(0.0, 0.4 - hydration) * state['adults'][:, 1]
        rewards = jnp.tanh(1.5 * yield_value - 0.05 * sow_count - 0.025 * toggled - 0.1 * drought)
        partner_mass = jnp.sum(adults[::-1, :], axis=1)
        partner_mix = adults[::-1, 1] / (partner_mass + 0.000001)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'adults': adults,
            'seed_bank': seed_bank,
            'seedlings': seedlings,
            'nutrient': nutrient,
            'water': jnp.clip(available_water - jnp.sum(intake), 0.0, 1.8),
            'gate': jnp.where(inspect, 1.0 - state['gate'], state['gate']),
            'intake': intake,
            'seen_mass': jnp.where(inspect, partner_mass, state['seen_mass']),
            'seen_mix': jnp.where(inspect, partner_mix, state['seen_mix']),
            'seen_age': jnp.where(inspect, 0.0, jnp.minimum(1.0, state['seen_age'] + 1.0 / 32.0)),
            'last_yield': yield_value,
        }
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards.astype(jnp.float32), dones, info
