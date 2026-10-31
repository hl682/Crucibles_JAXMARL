import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        state = {'t': jnp.array(0, jnp.int32), 'goods': jnp.zeros((2,), jnp.float32),
                 'offers': jnp.zeros((2,), jnp.float32), 'tools': jnp.full((2,), 0.25, jnp.float32),
                 'orders': jnp.zeros((2,), jnp.float32), 'sales': jnp.zeros((2,), jnp.float32)}
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'] / 32.0)
        obs = obs.at[:, 1].set(state['goods'] / 3.0)
        obs = obs.at[:, 2].set(jnp.roll(state['goods'], 1) / 3.0)
        obs = obs.at[:, 3].set(state['offers'])
        obs = obs.at[:, 4].set(jnp.roll(state['offers'], 1))
        obs = obs.at[:, 5].set(state['tools'])
        obs = obs.at[:, 6].set(jnp.roll(state['tools'], 1))
        obs = obs.at[:, 7].set(state['orders'])
        obs = obs.at[:, 8].set(jnp.roll(state['orders'], 1))
        obs = obs.at[:, 9].set(state['sales'])
        obs = obs.at[:, 10].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        # 0 wait; 1 manufacture; 2 post offer; 3 accept partner offer; 4 order a custom good.
        buy = (actions == 3) & (jnp.roll(state['offers'], 1) > 0) & (jnp.roll(state['goods'], 1) >= 1.0)
        sale = jnp.roll(buy, 1).astype(jnp.float32)
        manufacture = (actions == 1).astype(jnp.float32)
        custom = (actions == 4).astype(jnp.float32)
        bonus = buy.astype(jnp.float32) * state['orders']
        goods = jnp.clip(state['goods'] - sale + manufacture * (0.5 + state['tools']), 0.0, 3.0)
        offers = jnp.where((actions == 2) & (goods >= 1.0), 1.0, jnp.maximum(0.0, state['offers'] - 0.34 - sale))
        rewards = jnp.tanh(0.35 * buy + 0.2 * sale + 0.5 * bonus - 0.16 * manufacture - 0.08 * custom).astype(jnp.float32)
        next_state = {'t': state['t'] + jnp.array(1, jnp.int32), 'goods': goods,
                      'offers': offers.astype(jnp.float32),
                      'tools': jnp.clip(0.96 * state['tools'] + 0.18 * buy + 0.05 * jnp.roll(custom, 1), 0.0, 1.0),
                      'orders': jnp.clip(0.9 * state['orders'] * (1.0 - buy) + custom, 0.0, 1.0),
                      'sales': sale}
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
