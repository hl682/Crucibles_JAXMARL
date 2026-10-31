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
            'wallet': jax.random.uniform(keys[0], (2,), dtype=jnp.float32, minval=1.0, maxval=1.6),
            'parcel': jnp.zeros((2,), dtype=jnp.float32),
            'age': jnp.zeros((2,), dtype=jnp.int32),
            'seal': jnp.zeros((2,), dtype=jnp.bool_),
            'quote': jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.55, maxval=0.95),
            'demand': jax.random.uniform(keys[2], (2,), dtype=jnp.float32, minval=0.4, maxval=1.0),
            'reputation': jnp.zeros((2,), dtype=jnp.float32),
            'last_delivery': jnp.zeros((2,), dtype=jnp.float32),
            'last_action': jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state['t'].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state['wallet'] / 2.0)
        obs = obs.at[:, 2].set(state['parcel'])
        obs = obs.at[:, 3].set(state['age'].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 4].set(state['seal'].astype(jnp.float32))
        obs = obs.at[:, 5].set(state['quote'])
        obs = obs.at[:, 6].set(state['demand'])
        obs = obs.at[:, 7].set(jnp.roll(state['parcel'], 1))
        obs = obs.at[:, 8].set(jnp.roll(state['age'], 1).astype(jnp.float32) / 8.0)
        obs = obs.at[:, 9].set(jnp.roll(state['seal'], 1).astype(jnp.float32))
        obs = obs.at[:, 10].set(state['reputation'])
        obs = obs.at[:, 11].set(jnp.roll(state['reputation'], 1))
        obs = obs.at[:, 12].set(state['last_delivery'])
        obs = obs.at[:, 13].set(jnp.roll(state['last_delivery'], 1))
        obs = obs.at[:, 14].set(state['last_action'].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 15].set(jnp.roll(state['last_action'], 1).astype(jnp.float32) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        occupied = state['parcel'] > 0.0
        place = (actions == 1) & (~occupied) & (state['wallet'] >= 0.25)
        amount = jnp.where(place, jnp.minimum(state['wallet'], 0.8), 0.0)
        peer_parcel = jnp.roll(state['parcel'], 1)
        witness = (actions == 2) & (peer_parcel > 0.0) & (jnp.roll(state['age'], 1) >= 1) & (~jnp.roll(state['seal'], 1))
        release = (actions == 3) & occupied & state['seal'] & (state['age'] >= 2)
        recall = (actions == 4) & occupied
        expire = occupied & (state['age'] >= 7) & (~release) & (~recall)
        clear = release | recall | expire
        freshness = jnp.maximum(0.2, 1.0 - 0.08 * state['age'].astype(jnp.float32))
        delivery = jnp.where(release, state['parcel'] * (0.5 + 0.5 * state['quote']) * freshness, 0.0)
        incoming = jnp.roll(delivery, 1)
        recall_amount = jnp.where(recall, state['parcel'], 0.0)
        expired_amount = jnp.where(expire, state['parcel'], 0.0)
        refund = recall_amount * jnp.maximum(0.2, 0.7 - 0.05 * state['age'].astype(jnp.float32))
        parcel = jnp.where(place, amount, jnp.where(clear, 0.0, state['parcel']))
        age = jnp.where(place, 0, jnp.where(parcel > 0.0, jnp.minimum(state['age'] + 1, 8), 0)).astype(jnp.int32)
        seal = (parcel > 0.0) & (~place) & (state['seal'] | jnp.roll(witness, 1))
        fresh_quote = jax.random.uniform(key, (2,), dtype=jnp.float32, minval=0.55, maxval=0.95)
        rewards = jnp.clip(
            0.65 * delivery
            + 0.25 * incoming * (0.5 + 0.5 * state['demand'])
            + 0.04 * witness.astype(jnp.float32) * peer_parcel
            - 0.04 * amount - 0.06 * recall_amount - 0.10 * expired_amount,
            -1.0, 1.0,
        ).astype(jnp.float32)
        next_state = {
            't': state['t'] + jnp.array(1, dtype=jnp.int32),
            'wallet': jnp.clip(state['wallet'] + 0.20 * (actions == 0).astype(jnp.float32) - amount + 0.55 * delivery + refund + 0.10 * expired_amount, 0.0, 2.0),
            'parcel': parcel,
            'age': age,
            'seal': seal,
            'quote': jnp.where(clear, fresh_quote, state['quote']),
            'demand': jnp.clip(state['demand'] + 0.03 - 0.25 * incoming, 0.2, 1.0),
            'reputation': jnp.clip(0.96 * state['reputation'] + 0.10 * delivery + 0.06 * witness.astype(jnp.float32) - 0.05 * expire.astype(jnp.float32), 0.0, 1.0),
            'last_delivery': delivery,
            'last_action': actions.astype(jnp.int32),
        }
        done_all = next_state['t'] >= 32
        dones = jnp.stack([done_all, done_all]).astype(jnp.bool_)
        info = {'done_all': done_all, 'legal_mask': self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
