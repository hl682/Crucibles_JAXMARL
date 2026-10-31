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
            'volume': jax.random.uniform(keys[0], (2,), dtype=jnp.float32, minval=0.2, maxval=0.7),
            'cistern': jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.7, maxval=1.2),
            'pipe': jnp.zeros((2, 2), dtype=jnp.float32),
            'wear': jnp.zeros((2,), dtype=jnp.float32),
            'need': jax.random.uniform(keys[2], (2,), dtype=jnp.float32, minval=0.4, maxval=0.85),
            'silt': jnp.array(0.0, dtype=jnp.float32),
            'last_sale': jnp.zeros((2,), dtype=jnp.float32),
            'last_spill': jnp.zeros((2,), dtype=jnp.float32),
            'last_action': jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['volume'] / 2.0)
        obs = obs.at[:, 2].set(state['cistern'] / 1.5)
        obs = obs.at[:, 3].set(state['pipe'][:, 0] / 0.6)
        obs = obs.at[:, 4].set(state['pipe'][:, 1] / 0.6)
        obs = obs.at[:, 5].set(state['wear'])
        obs = obs.at[:, 6].set(state['need'])
        obs = obs.at[:, 7].set(jnp.roll(state['volume'], 1) / 2.0)
        obs = obs.at[:, 8].set(jnp.roll(state['wear'], 1))
        obs = obs.at[:, 9].set(state['last_sale'])
        obs = obs.at[:, 10].set(jnp.roll(state['last_sale'], 1))
        obs = obs.at[:, 11].set(state['last_spill'])
        obs = obs.at[:, 12].set(jnp.roll(state['last_spill'], 1))
        obs = obs.at[:, 13].set(state['silt'])
        obs = obs.at[:, 14].set(state['last_action'].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 15].set(jnp.roll(state['last_action'], 1).astype(jnp.float32) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        rest = (actions == 0).astype(jnp.float32)
        service = (actions == 4).astype(jnp.float32)
        pump = jnp.minimum(state['cistern'], 0.5) * (actions == 1).astype(jnp.float32)
        requested_send = jnp.minimum(state['volume'], 0.6) * (1.0 - 0.5 * state['wear']) * (actions == 2).astype(jnp.float32)
        shared_fraction = jnp.minimum(1.0, 0.75 / (jnp.sum(requested_send) + 0.000001))
        sent = requested_send * shared_fraction
        sale = jnp.minimum(state['volume'], 0.7) * (actions == 3).astype(jnp.float32)
        vent = jnp.minimum(state['volume'], 0.3) * service
        arrival = state['pipe'][:, 0]
        uncapped_volume = state['volume'] + arrival + pump - sent - sale - vent
        spill = jnp.maximum(uncapped_volume - 2.0, 0.0)
        quality = (1.0 - 0.5 * state['wear']) * (1.0 - 0.5 * state['silt'])
        fit = jnp.maximum(0.0, 1.0 - jnp.abs(sale - state['need']))
        fresh_need = jax.random.uniform(key, (2,), dtype=jnp.float32, minval=0.4, maxval=0.85)
        rewards = jnp.clip(
            0.90 * sale * fit * quality + 0.12 * jnp.roll(sale, 1)
            + 0.04 * sent - 0.07 * pump - 0.04 * service
            - 0.20 * spill - 0.15 * jnp.roll(spill, 1),
            -1.0, 1.0,
        ).astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'volume': jnp.clip(uncapped_volume, 0.0, 2.0),
            'cistern': jnp.clip(state['cistern'] + 0.04 + 0.16 * rest - pump, 0.0, 1.5),
            'pipe': jnp.stack([state['pipe'][:, 1], jnp.roll(sent, 1)], axis=1),
            'wear': jnp.clip(0.98 * state['wear'] + 0.12 * pump + 0.10 * arrival + 0.06 * sale - 0.32 * service, 0.0, 1.0),
            'need': jnp.where(sale > 0.0, fresh_need, state['need']),
            'silt': jnp.clip(0.85 * state['silt'] + 0.10 * jnp.sum(pump) + 0.10 * jnp.sum(spill) - 0.08 * jnp.sum(service), 0.0, 1.0),
            'last_sale': sale,
            'last_spill': spill,
            'last_action': actions.astype(jnp.int32),
        }
        done_all = next_state['t'] >= 32
        dones = jnp.stack([done_all, done_all]).astype(jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
