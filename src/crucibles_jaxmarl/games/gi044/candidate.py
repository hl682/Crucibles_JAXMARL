import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        keys = jax.random.split(key, 2)
        state = {
            't': jnp.array(0, dtype=jnp.int32),
            'cargo': jnp.zeros((2,), dtype=jnp.float32),
            'quality': jax.random.uniform(keys[0], (2,), dtype=jnp.float32, minval=0.2, maxval=1.0),
            'certificate': jnp.zeros((2,), dtype=jnp.float32),
            'energy': jnp.ones((2,), dtype=jnp.float32),
            'transit': jnp.zeros((2,), dtype=jnp.float32),
            'commission': jnp.zeros((2,), dtype=jnp.float32),
            'seen_quality': jnp.zeros((2,), dtype=jnp.float32),
            'seen_age': jnp.ones((2,), dtype=jnp.float32),
            'congestion': jnp.array(0.0, dtype=jnp.float32),
            'demand': jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.6, maxval=1.0),
            'last_reward': jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['cargo'])
        obs = obs.at[:, 2].set(state['cargo'][::-1])
        obs = obs.at[:, 3].set(state['quality'])
        obs = obs.at[:, 4].set(state['certificate'])
        obs = obs.at[:, 5].set(state['certificate'][::-1])
        obs = obs.at[:, 6].set(state['energy'])
        obs = obs.at[:, 7].set(state['energy'][::-1])
        obs = obs.at[:, 8].set(state['transit'])
        obs = obs.at[:, 9].set(state['commission'])
        obs = obs.at[:, 10].set(state['seen_quality'])
        obs = obs.at[:, 11].set(state['seen_age'])
        obs = obs.at[:, 12].set(state['congestion'])
        obs = obs.at[:, 13].set(state['demand'])
        obs = obs.at[:, 14].set(state['demand'][::-1])
        obs = obs.at[:, 15].set(state['last_reward'])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        prepare = jnp.logical_and(actions == 1, state['energy'] >= 0.28)
        inspect = jnp.logical_and(actions == 2, state['energy'] >= 0.12)
        ship = jnp.logical_and(jnp.logical_and(actions == 3, state['energy'] >= 0.08), state['cargo'] > 0.05)
        recharge = (actions == 4).astype(jnp.float32)
        prepared = prepare.astype(jnp.float32)
        inspected = inspect.astype(jnp.float32)
        shipped = ship.astype(jnp.float32)
        congestion = jnp.clip(0.65 * state['congestion'] + 0.2 * jnp.sum(shipped), 0.0, 1.0)
        payload = shipped * state['cargo'] * state['certificate'] * state['demand']
        transit = 0.72 * payload / (1.0 + congestion)
        commission = 0.16 * payload[::-1] / (1.0 + congestion)
        rewards = jnp.clip(state['transit'] + state['commission'] - 0.06 * prepared - 0.025 * inspected - 0.035 * shipped, -1.0, 1.0)
        sampled_quality = jax.random.uniform(key, (2,), dtype=jnp.float32, minval=0.2, maxval=1.0)
        quality = jnp.where(prepare, sampled_quality, state['quality'])
        cargo = jnp.where(prepare, 1.0, jnp.where(ship, 0.0, state['cargo']))
        valid_inspection = jnp.logical_and(inspect[::-1], state['cargo'] > 0.05)
        certificate = jnp.where(valid_inspection, state['quality'], 0.94 * state['certificate'])
        certificate = jnp.where(jnp.logical_or(prepare, ship), 0.0, certificate)
        energy = jnp.clip(state['energy'] + 0.03 + 0.28 * recharge - 0.28 * prepared - 0.12 * inspected - 0.08 * shipped, 0.0, 1.0)
        seen_quality = jnp.where(inspect, state['quality'][::-1] * state['cargo'][::-1], state['seen_quality'])
        seen_age = jnp.where(inspect, 0.0, jnp.minimum(1.0, state['seen_age'] + 1.0 / 32.0))
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'cargo': cargo,
            'quality': quality,
            'certificate': certificate,
            'energy': energy,
            'transit': transit,
            'commission': commission,
            'seen_quality': seen_quality,
            'seen_age': seen_age,
            'congestion': congestion,
            'demand': state['demand'],
            'last_reward': rewards,
        }
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards.astype(jnp.float32), dones, info
