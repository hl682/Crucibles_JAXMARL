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
        target = jax.random.randint(keys[0], (2,), 0, 2, dtype=jnp.int32)
        state = {
            't': jnp.array(0, dtype=jnp.int32),
            'target': target,
            'survey': jnp.roll(target, 1),
            'survey_age': jnp.zeros((2,), dtype=jnp.int32),
            'message': jnp.full((2,), -1, dtype=jnp.int32),
            'message_age': jnp.full((2,), 8, dtype=jnp.int32),
            'charge': jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.65, maxval=1.0),
            'strain': jnp.zeros((2,), dtype=jnp.float32),
            'pending': jnp.zeros((2,), dtype=jnp.int32),
            'busy': jnp.array(0, dtype=jnp.int32),
            'last_work': jnp.zeros((2,), dtype=jnp.float32),
            'last_success': jnp.zeros((2,), dtype=jnp.float32),
            'last_return': jnp.zeros((2,), dtype=jnp.float32),
            'last_action': jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['survey'].astype(jnp.float32))
        obs = obs.at[:, 2].set(state['survey_age'].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 3].set((state['message'].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 4].set(state['message_age'].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 5].set(state['charge'])
        obs = obs.at[:, 6].set(state['strain'])
        obs = obs.at[:, 7].set(jnp.roll(state['strain'], 1))
        obs = obs.at[:, 8].set(state['busy'].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 9].set(state['last_work'])
        obs = obs.at[:, 10].set(jnp.roll(state['last_work'], 1))
        obs = obs.at[:, 11].set(state['last_success'])
        obs = obs.at[:, 12].set(jnp.roll(state['last_success'], 1))
        obs = obs.at[:, 13].set(state['last_return'])
        obs = obs.at[:, 14].set(state['last_action'].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 15].set(jnp.roll(state['last_action'], 1).astype(jnp.float32) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        rest = (actions == 0).astype(jnp.float32)
        work = ((actions == 1) | (actions == 2)) & (state['charge'] >= 0.22)
        probe = (actions == 3) & (state['charge'] >= 0.06)
        send = (actions == 4) & (state['charge'] >= 0.04)
        success = work & ((actions - 1) == state['target'])
        work_float = work.astype(jnp.float32)
        success_float = success.astype(jnp.float32)
        one_sender = jnp.sum(send.astype(jnp.int32)) == 1
        delivered = send & one_sender & (state['busy'] == 0)
        received = jnp.roll(delivered, 1)
        collision = jnp.all(send)
        burst = state['strain'] >= 0.72
        random_flip = jax.random.bernoulli(key, p=0.02 + 0.08 * state['strain'], shape=(2,))
        target = jnp.mod(state['target'] + state['pending'] + burst.astype(jnp.int32) + random_flip.astype(jnp.int32), 2).astype(jnp.int32)
        rewards = jnp.clip(
            work_float * jnp.where(success, 0.65, -0.45)
            + 0.15 * jnp.roll(success_float, 1)
            - 0.015 * probe.astype(jnp.float32)
            - 0.010 * send.astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'target': target,
            'survey': jnp.where(probe, jnp.roll(state['target'], 1), state['survey']).astype(jnp.int32),
            'survey_age': jnp.where(probe, 1, jnp.minimum(state['survey_age'] + 1, 8)).astype(jnp.int32),
            'message': jnp.where(received, jnp.roll(state['survey'], 1), state['message']).astype(jnp.int32),
            'message_age': jnp.where(received, jnp.minimum(jnp.roll(state['survey_age'], 1) + 1, 8), jnp.minimum(state['message_age'] + 1, 8)).astype(jnp.int32),
            'charge': jnp.clip(state['charge'] + 0.02 + 0.18 * rest - 0.22 * work_float - 0.06 * probe.astype(jnp.float32) - 0.04 * send.astype(jnp.float32), 0.0, 1.0),
            'strain': jnp.clip(0.86 * state['strain'] + 0.16 * jnp.roll(work_float, 1) + 0.08 * work_float - 0.12 * rest - 0.35 * burst.astype(jnp.float32), 0.0, 1.0),
            'pending': jnp.roll(work.astype(jnp.int32), 1),
            'busy': jnp.where(collision, 2, jnp.where(jnp.any(delivered), 1, jnp.maximum(state['busy'] - 1, 0))).astype(jnp.int32),
            'last_work': work_float,
            'last_success': success_float,
            'last_return': rewards,
            'last_action': actions.astype(jnp.int32),
        }
        done_all = next_state['t'] >= 32
        dones = jnp.stack([done_all, done_all]).astype(jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
