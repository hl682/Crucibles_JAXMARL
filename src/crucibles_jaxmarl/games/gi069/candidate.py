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
            'pipe': jnp.zeros((2, 2), dtype=jnp.float32),
            'shield': jnp.zeros((2, 2), dtype=jnp.bool_),
            'inbox': jnp.zeros((2,), dtype=jnp.float32),
            'ack': jnp.zeros((2,), dtype=jnp.float32),
            'credits': jnp.full((2,), 2, dtype=jnp.int32),
            'cooldown': jnp.zeros((2,), dtype=jnp.int32),
            'load': jnp.array(0.0, dtype=jnp.float32),
            'demand': 0.40 + 0.60 * jax.random.uniform(key, (2,), dtype=jnp.float32),
            'dropped': jnp.zeros((2,), dtype=jnp.float32),
            'last_action': jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['credits'].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 2].set(state['pipe'][:, 0])
        obs = obs.at[:, 3].set(state['pipe'][:, 1])
        obs = obs.at[:, 4].set(state['inbox'])
        obs = obs.at[:, 5].set(state['ack'])
        obs = obs.at[:, 6].set(state['cooldown'].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 7].set(state['load'])
        obs = obs.at[:, 8].set(state['demand'])
        obs = obs.at[:, 9].set((jnp.sum(state['pipe'], axis=1)[::-1] > 0.0).astype(jnp.float32))
        obs = obs.at[:, 10].set(state['cooldown'][::-1].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 11].set(state['dropped'])
        obs = obs.at[:, 12].set(jnp.mean((state['pipe'] > 0.0).astype(jnp.float32)))
        obs = obs.at[:, 13].set(state['last_action'].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 14].set(state['last_action'][::-1].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 15].set((state['inbox'][::-1] > 0.0).astype(jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        timeout = actions == 4
        send = ((actions == 1) | (actions == 2)) & (state['credits'] > 0) & (state['cooldown'] == 0)
        protected = send & (actions == 2)
        processed = (actions == 3) & (state['inbox'] > 0.0)
        room = (state['inbox'] <= 0.0) | processed
        arriving = state['pipe'][:, 1]
        survives = (arriving > 0.0) & (state['shield'][:, 1] | (state['load'] < 0.65)) & (~timeout)
        accepted = survives & room[::-1]
        arriving_receiver = jnp.where(accepted, arriving, 0.0)[::-1]
        inbox = jnp.where(arriving_receiver > 0.0, arriving_receiver, jnp.where(processed, 0.0, state['inbox']))
        dropped = (arriving > 0.0) & (~accepted)
        sent_value = jnp.where(send, state['demand'], 0.0)
        pipe = jnp.stack((sent_value, jnp.where(timeout, 0.0, state['pipe'][:, 0])), axis=1)
        shield = jnp.stack((protected, state['shield'][:, 0] & (~timeout)), axis=1)
        ack = (processed.astype(jnp.float32) * state['inbox'])[::-1]
        credits = jnp.where(timeout, 2, jnp.clip(state['credits'] + (state['ack'] > 0.0).astype(jnp.int32) - send.astype(jnp.int32), 0, 2))
        cooldown = jnp.where(timeout, 2, jnp.maximum(state['cooldown'] - 1, 0))
        load = jnp.clip(0.72 * state['load'] + 0.18 * jnp.sum(send.astype(jnp.float32)) + 0.12 * jnp.sum(dropped.astype(jnp.float32)) - 0.12 * jnp.sum(processed.astype(jnp.float32)), 0.0, 1.0)
        demand_draw = 0.40 + 0.60 * jax.random.uniform(key, (2,), dtype=jnp.float32)
        demand = jnp.where(send, demand_draw, state['demand'])
        send_cost = 0.035 * send.astype(jnp.float32) + 0.045 * protected.astype(jnp.float32)
        utility = 0.50 * state['ack'] + 0.20 * processed.astype(jnp.float32) * state['inbox'] - send_cost - 0.06 * timeout.astype(jnp.float32) - 0.08 * dropped.astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'pipe': pipe.astype(jnp.float32),
            'shield': shield.astype(jnp.bool_),
            'inbox': inbox.astype(jnp.float32),
            'ack': ack.astype(jnp.float32),
            'credits': credits.astype(jnp.int32),
            'cooldown': cooldown.astype(jnp.int32),
            'load': load.astype(jnp.float32),
            'demand': demand.astype(jnp.float32),
            'dropped': dropped.astype(jnp.float32),
            'last_action': actions.astype(jnp.int32),
        }
        rewards = jnp.tanh(1.4 * utility).astype(jnp.float32)
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
