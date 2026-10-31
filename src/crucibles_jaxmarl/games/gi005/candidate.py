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
            'stock': jax.random.uniform(key, (2,), minval=0.4, maxval=0.8),
            'escrow': jnp.zeros((2,), dtype=jnp.float32),
            'bond': jnp.zeros((2,), dtype=jnp.float32),
            'reputation': jnp.ones((2,), dtype=jnp.float32) * 0.5,
            'settled': jnp.zeros((2,), dtype=jnp.float32),
            'withdrawn': jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'] / 32.0)
        obs = obs.at[:, 1].set(jnp.mod(state['t'], 4) / 3.0)
        obs = obs.at[:, 2].set(state['stock'])
        obs = obs.at[:, 3].set(state['escrow'])
        obs = obs.at[:, 4].set(state['escrow'][::-1])
        obs = obs.at[:, 5].set(state['bond'])
        obs = obs.at[:, 6].set(state['bond'][::-1])
        obs = obs.at[:, 7].set(state['reputation'])
        obs = obs.at[:, 8].set(state['reputation'][::-1])
        obs = obs.at[:, 9].set(state['settled'])
        obs = obs.at[:, 10].set(state['withdrawn'])
        obs = obs.at[:, 11].set(state['withdrawn'][::-1])
        obs = obs.at[:, 12].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 13].set(1.0 - state['stock'])
        obs = obs.at[:, 14].set(state['escrow'] + state['bond'])
        obs = obs.at[:, 15].set(1.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        stocked = jnp.minimum(1.0, state['stock'] + 0.18 * (actions == 0))
        offer = jnp.minimum(stocked, 0.25) * (actions == 1)
        remaining = stocked - offer
        pledge = jnp.minimum(remaining, 0.2) * (actions == 2)
        escrow = jnp.minimum(1.0, state['escrow'] + offer)
        bond = jnp.minimum(1.0, state['bond'] + pledge)
        withdrawn = escrow * (actions == 3)
        forfeited = bond * (actions == 3)
        escrow = escrow - withdrawn
        bond = bond - forfeited
        settlement = jnp.mod(state['t'] + 1, 4) == 0
        matched = jnp.minimum(escrow[0], escrow[1])
        delivered = jnp.ones((2,), dtype=jnp.float32) * matched * settlement
        credit = 0.5 + 0.5 * state['reputation'][::-1]
        consumed = jnp.minimum(remaining - pledge, 0.2) * (actions == 4)
        rewards = (0.65 * delivered * credit + 0.3 * forfeited[::-1] + 0.4 * consumed - 0.35 * forfeited).astype(jnp.float32)
        reputation = jnp.clip(state['reputation'] + 0.15 * delivered - 0.25 * (actions == 3), 0.0, 1.0)
        stock = jnp.clip(remaining - pledge - consumed + withdrawn + bond * settlement, 0.0, 1.0)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'stock': stock, 'escrow': escrow - delivered,
            'bond': bond * (1.0 - settlement), 'reputation': reputation,
            'settled': delivered, 'withdrawn': withdrawn,
        }
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        return self.observe(next_state), next_state, rewards, dones, {
            'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
