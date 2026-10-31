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
            "t": jnp.array(0, dtype=jnp.int32),
            "task": jax.random.randint(keys[0], (2,), 0, 2, dtype=jnp.int32),
            "decoder": jax.random.randint(keys[1], (2,), 0, 2, dtype=jnp.int32),
            "fluency": jnp.full((2,), 0.5, dtype=jnp.float32),
            "symbol": jnp.zeros((2,), dtype=jnp.int32),
            "payload": jnp.zeros((2,), dtype=jnp.int32),
            "pending": jnp.zeros((2,), dtype=jnp.bool_),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "last_received": jnp.zeros((2,), dtype=jnp.int32),
            "last_outgoing": jnp.zeros((2,), dtype=jnp.int32),
            "last_decoded": jnp.full((2,), -1, dtype=jnp.int32),
            "last_revealed": jnp.full((2,), -1, dtype=jnp.int32),
            "sent_count": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        visible_symbol = jnp.where(state["pending"], state["symbol"], -1)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(jnp.array([-1.0, 1.0], dtype=jnp.float32))
        obs = obs.at[:, 2].set(state["task"].astype(jnp.float32))
        obs = obs.at[:, 3].set(state["decoder"].astype(jnp.float32))
        obs = obs.at[:, 4].set(state["fluency"])
        obs = obs.at[:, 5].set(visible_symbol.astype(jnp.float32))
        obs = obs.at[:, 6].set(state["pending"].astype(jnp.float32))
        obs = obs.at[:, 7].set(state["age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 8].set(jnp.roll(state["pending"], 1).astype(jnp.float32))
        obs = obs.at[:, 9].set(jnp.roll(state["age"], 1).astype(jnp.float32) / 32.0)
        obs = obs.at[:, 10].set(state["last_received"].astype(jnp.float32))
        obs = obs.at[:, 11].set(state["last_outgoing"].astype(jnp.float32))
        obs = obs.at[:, 12].set(state["last_decoded"].astype(jnp.float32))
        obs = obs.at[:, 13].set(state["last_revealed"].astype(jnp.float32))
        obs = obs.at[:, 14].set(jnp.roll(visible_symbol, 1).astype(jnp.float32))
        obs = obs.at[:, 15].set(state["sent_count"].astype(jnp.float32) / 32.0)
        return obs

    def legal_actions(self, state):
        available = jnp.ones((2,), dtype=jnp.bool_)
        can_accept = state["pending"] & (state["age"] >= 1)
        return jnp.stack((available, available, available, available, can_accept), axis=1)

    def raw_step(self, key, state, actions):
        send = (actions == 1) | (actions == 2)
        toggle = actions == 3
        accepted = (actions == 4) & state["pending"] & (state["age"] >= 1)
        decoded = jnp.mod(state["symbol"] + state["decoder"], 2).astype(jnp.int32)
        correct = accepted & (decoded == state["payload"])
        wrong = accepted & (~correct)
        quality = (
            jnp.clip(1.0 - 0.02 * state["age"].astype(jnp.float32), 0.3, 1.0)
            * (0.5 + 0.5 * state["fluency"])
        )
        receiver_payoff = jnp.where(correct, 0.65 * quality, jnp.where(wrong, -0.45, 0.0))
        sender_payoff = jnp.roll(jnp.where(correct, 0.35 * quality, jnp.where(wrong, -0.25, 0.0)), 1)
        incoming_send = jnp.roll(send, 1)
        replacement = incoming_send & state["pending"] & (~accepted)
        rewards = jnp.clip(
            receiver_payoff + sender_payoff
            - 0.05 * send.astype(jnp.float32)
            - 0.04 * toggle.astype(jnp.float32)
            - 0.08 * jnp.roll(replacement, 1).astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)
        pending = incoming_send | (state["pending"] & (~accepted))
        symbol = jnp.where(incoming_send, jnp.roll((actions == 2).astype(jnp.int32), 1), state["symbol"])
        payload = jnp.where(incoming_send, jnp.roll(state["task"], 1), state["payload"])
        age = jnp.where(
            incoming_send, 0,
            jnp.where(pending, jnp.minimum(state["age"] + 1, 32), 0),
        ).astype(jnp.int32)
        decoder = jnp.mod(state["decoder"] + toggle.astype(jnp.int32), 2).astype(jnp.int32)
        fluency = jnp.clip(
            state["fluency"] + 0.08 * correct.astype(jnp.float32)
            - 0.12 * wrong.astype(jnp.float32)
            - 0.07 * toggle.astype(jnp.float32),
            0.1, 1.0,
        )
        next_t = state["t"] + jnp.array(1, dtype=jnp.int32)
        refresh = jnp.roll(accepted, 1) | (jnp.mod(next_t, 4) == 0)
        fresh_task = jax.random.bernoulli(key, 0.5, (2,)).astype(jnp.int32)
        task = jnp.where(refresh, fresh_task, state["task"]).astype(jnp.int32)
        feedback = jnp.where(accepted, jnp.where(correct, 1, -1), 0).astype(jnp.int32)
        next_state = {
            "t": next_t,
            "task": task,
            "decoder": decoder,
            "fluency": fluency.astype(jnp.float32),
            "symbol": symbol.astype(jnp.int32),
            "payload": payload.astype(jnp.int32),
            "pending": pending,
            "age": age,
            "last_received": feedback,
            "last_outgoing": jnp.roll(feedback, 1),
            "last_decoded": jnp.where(accepted, decoded, -1).astype(jnp.int32),
            "last_revealed": jnp.where(accepted, state["payload"], -1).astype(jnp.int32),
            "sent_count": state["sent_count"] + send.astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
