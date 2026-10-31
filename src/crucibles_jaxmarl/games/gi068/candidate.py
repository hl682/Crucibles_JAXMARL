import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        draws = jax.random.uniform(key, (4,), dtype=jnp.float32)
        state = {
            't': jnp.array(0, dtype=jnp.int32),
            'cash': 0.45 + 0.20 * draws[:2],
            'timer': jnp.zeros((2,), dtype=jnp.int32),
            'quality': jnp.zeros((2,), dtype=jnp.float32),
            'quote': 0.10 + 0.90 * draws[2:],
            'cover': jnp.zeros((2,), dtype=jnp.float32),
            'audit': jnp.full((2,), -1.0, dtype=jnp.float32),
            'event': jnp.zeros((2,), dtype=jnp.float32),
            'last_action': jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['cash'])
        obs = obs.at[:, 2].set(state['cash'][::-1])
        obs = obs.at[:, 3].set(state['timer'].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 4].set(state['timer'][::-1].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 5].set(state['quality'])
        obs = obs.at[:, 6].set(state['quote'])
        obs = obs.at[:, 7].set(state['cover'] / 0.15)
        obs = obs.at[:, 8].set(state['cover'][::-1] / 0.15)
        obs = obs.at[:, 9].set(state['audit'])
        obs = obs.at[:, 10].set(state['event'])
        obs = obs.at[:, 11].set(state['event'][::-1])
        obs = obs.at[:, 12].set(state['last_action'].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 13].set(state['last_action'][::-1].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 14].set((state['audit'] >= 0.0).astype(jnp.float32))
        obs = obs.at[:, 15].set((state['timer'] > 0).astype(jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        active = state['timer'] > 0
        start = (actions == 1) & (~active) & (state['cash'] >= 0.25)
        issue = (actions == 2) & active[::-1] & (state['cover'][::-1] < 0.01) & (state['cash'] >= 0.15)
        liquidate = (actions == 3) & active
        audit = (actions == 4) & active[::-1] & (state['cash'] >= 0.025)
        mature = (state['timer'] == 1) & (~liquidate)
        ended = mature | liquidate
        funded_cover = state['cover'] + 0.15 * issue[::-1].astype(jnp.float32)
        success = mature & (state['quality'] >= 0.35)
        failure = mature & (~success)
        proceeds = 0.75 * state['quality'] * success.astype(jnp.float32)
        indemnity = funded_cover * failure.astype(jnp.float32)
        liquidation_cash = 0.12 * liquidate.astype(jnp.float32)
        cover_return = (funded_cover * (success | liquidate).astype(jnp.float32))[::-1]
        underwriting_fee = (0.04 * (success & (funded_cover > 0.0)).astype(jnp.float32))[::-1]
        investment = 0.25 * start.astype(jnp.float32)
        cover_cost = 0.15 * issue.astype(jnp.float32)
        audit_cost = 0.025 * audit.astype(jnp.float32)
        saving = 0.08 * (actions == 0).astype(jnp.float32)
        net_receipt = proceeds + indemnity + liquidation_cash + cover_return + underwriting_fee - investment - cover_cost - audit_cost
        cash = jnp.clip(state['cash'] + saving + net_receipt, 0.0, 1.0)
        timer = jnp.where(start, 3, jnp.where(ended, 0, jnp.maximum(state['timer'] - 1, 0))).astype(jnp.int32)
        quality = jnp.where(start, state['quote'], jnp.where(ended, 0.0, state['quality']))
        quote_draw = 0.10 + 0.90 * jax.random.uniform(key, (2,), dtype=jnp.float32)
        quote = jnp.where(start, quote_draw, state['quote'])
        cover = jnp.where(ended, 0.0, funded_cover)
        stale_audit = ended[::-1] | start[::-1]
        remembered = jnp.where(stale_audit, -1.0, state['audit'])
        audit_value = jnp.where(audit & (~ended[::-1]), state['quality'][::-1], remembered)
        event = success.astype(jnp.float32) - failure.astype(jnp.float32) - 0.5 * liquidate.astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'cash': cash.astype(jnp.float32),
            'timer': timer,
            'quality': quality.astype(jnp.float32),
            'quote': quote.astype(jnp.float32),
            'cover': cover.astype(jnp.float32),
            'audit': audit_value.astype(jnp.float32),
            'event': event.astype(jnp.float32),
            'last_action': actions.astype(jnp.int32),
        }
        rewards = jnp.tanh(2.0 * net_receipt).astype(jnp.float32)
        done_all = next_state['t'] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
