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
            'packets': jnp.zeros((2, 4), dtype=jnp.float32),
            'switches': jax.random.bernoulli(key, 0.5, (2,)).astype(jnp.float32),
            'energy': jnp.ones((2,), dtype=jnp.float32),
            'last_delivery': jnp.zeros((2,), dtype=jnp.float32),
            'last_loss': jnp.array(0.0, dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['energy'])
        obs = obs.at[:, 2].set(state['energy'][::-1])
        obs = obs.at[:, 3].set(state['switches'])
        obs = obs.at[:, 4].set(state['switches'][::-1])
        obs = obs.at[:, 5:9].set(state['packets'])
        obs = obs.at[:, 9:13].set(state['packets'][::-1, :])
        obs = obs.at[:, 13].set(state['last_delivery'])
        obs = obs.at[:, 14].set(state['last_delivery'][::-1])
        obs = obs.at[:, 15].set(state['last_loss'])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        inject = jnp.logical_and(actions == 1, state['energy'] >= 0.18).astype(jnp.float32)
        switch = jnp.logical_and(actions == 2, state['energy'] >= 0.08).astype(jnp.float32)
        receive = jnp.logical_and(actions == 3, state['energy'] >= 0.07).astype(jnp.float32)
        flush = jnp.logical_and(actions == 4, state['energy'] >= 0.05).astype(jnp.float32)
        delivered = jnp.stack((receive[1] * state['packets'][0, 2], receive[0] * state['packets'][1, 0]))
        working = state['packets'].at[0, 2].set(state['packets'][0, 2] - delivered[0])
        working = working.at[1, 0].set(state['packets'][1, 0] - delivered[1])
        flush_mask = jnp.stack((flush[0], jnp.array(0.0, dtype=jnp.float32), flush[1], jnp.array(0.0, dtype=jnp.float32)))
        working = working * (1.0 - flush_mask[None, :])
        injected = 0.65 * inject
        working = working.at[0, 0].add(injected[0])
        working = working.at[1, 2].add(injected[1])
        route0 = state['switches'][0]
        route1 = state['switches'][1]
        routed = jnp.stack((
            working[:, 3],
            working[:, 0] * (1.0 - route0) + working[:, 2] * route1,
            working[:, 1],
            working[:, 0] * route0 + working[:, 2] * (1.0 - route1),
        ), axis=1)
        load = jnp.sum(routed, axis=0)
        packets = routed * (0.96 / jnp.maximum(1.0, load))[None, :]
        lost = jnp.maximum(0.0, jnp.sum(state['packets'], axis=1) + injected - delivered - jnp.sum(packets, axis=1))
        rewards = jnp.tanh(1.35 * delivered + 0.35 * delivered[::-1] - 0.09 * inject - 0.03 * switch - 0.02 * receive - 0.04 * flush - 0.18 * lost)
        energy = jnp.clip(state['energy'] + 0.035 + 0.045 * (actions == 0).astype(jnp.float32) - 0.18 * inject - 0.08 * switch - 0.07 * receive - 0.05 * flush, 0.0, 1.0)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'packets': packets,
            'switches': jnp.where(switch > 0.0, 1.0 - state['switches'], state['switches']),
            'energy': energy,
            'last_delivery': delivered,
            'last_loss': jnp.minimum(1.0, jnp.sum(lost) / 4.0),
        }
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards.astype(jnp.float32), dones, info
