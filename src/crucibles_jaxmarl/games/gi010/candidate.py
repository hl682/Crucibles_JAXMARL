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
            "balance": jax.random.uniform(keys[0], (2,), minval=0.6, maxval=1.4, dtype=jnp.float32),
            "debt": jnp.zeros((2,), dtype=jnp.float32),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "request": jnp.zeros((2,), dtype=jnp.int32),
            "frozen": jnp.zeros((2,), dtype=jnp.int32),
            "skill": jax.random.uniform(keys[1], (2,), minval=0.03, maxval=0.07, dtype=jnp.float32),
            "last_income": jnp.zeros((2,), dtype=jnp.float32),
            "last_transfer": jnp.zeros((2,), dtype=jnp.float32),
            "last_enforce": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["balance"] / 3.0)
        obs = obs.at[:, 2].set(jnp.roll(state["balance"], 1) / 3.0)
        obs = obs.at[:, 3].set(state["debt"] / 0.61)
        obs = obs.at[:, 4].set(jnp.roll(state["debt"], 1) / 0.61)
        obs = obs.at[:, 5].set(state["age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 6].set(jnp.roll(state["age"], 1).astype(jnp.float32) / 32.0)
        obs = obs.at[:, 7].set(state["request"].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 8].set(jnp.roll(state["request"], 1).astype(jnp.float32) / 3.0)
        obs = obs.at[:, 9].set(state["frozen"].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 10].set(jnp.roll(state["frozen"], 1).astype(jnp.float32) / 3.0)
        obs = obs.at[:, 11].set(state["skill"] / 0.07)
        obs = obs.at[:, 12].set(state["last_income"] / 0.14)
        obs = obs.at[:, 13].set(state["last_transfer"] / 0.61)
        obs = obs.at[:, 14].set(jnp.roll(state["last_enforce"], 1))
        obs = obs.at[:, 15].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        repaid = jnp.where(
            actions == 3,
            jnp.minimum(state["balance"], state["debt"]),
            0.0,
        )
        lendable = (
            (actions == 2)
            & (jnp.roll(state["request"], 1) > 0)
            & (jnp.roll(state["debt"], 1) <= 0.000001)
        )
        lent = jnp.where(
            lendable,
            jnp.minimum(state["balance"] - repaid, 0.55),
            0.0,
        )
        borrowed = jnp.roll(lent, 1)
        returned = jnp.roll(repaid, 1)
        transfer = returned - repaid + borrowed - lent
        working_balance = jnp.clip(state["balance"] + transfer, 0.0, 3.0)
        debt = jnp.maximum(state["debt"] - repaid, 0.0) + 1.1 * borrowed
        age = jnp.where(
            debt > 0.000001,
            jnp.where(borrowed > 0.0, 0, state["age"] + 1),
            0,
        ).astype(jnp.int32)
        enforced = (
            (actions == 4)
            & (jnp.roll(debt, 1) > 0.000001)
            & (jnp.roll(state["age"], 1) >= 4)
        )
        frozen = jnp.maximum(state["frozen"] - 1, 0)
        frozen = jnp.where(jnp.roll(enforced, 1), 3, frozen)
        frozen = jnp.where(debt > 0.000001, frozen, 0).astype(jnp.int32)
        income = (
            state["skill"] * jnp.sqrt(working_balance)
            * (frozen == 0).astype(jnp.float32)
        )
        balance = jnp.clip(working_balance + 0.5 * income, 0.0, 3.0)
        request = jnp.where(
            actions == 1,
            3,
            jnp.maximum(state["request"] - 1, 0),
        )
        request = jnp.where(borrowed > 0.0, 0, request).astype(jnp.int32)
        rewards = (
            income / 0.14
            - 0.05 * enforced.astype(jnp.float32)
            - 0.01 * (actions == 1).astype(jnp.float32)
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "balance": balance.astype(jnp.float32),
            "debt": debt.astype(jnp.float32),
            "age": age,
            "request": request,
            "frozen": frozen,
            "skill": state["skill"],
            "last_income": income.astype(jnp.float32),
            "last_transfer": transfer.astype(jnp.float32),
            "last_enforce": enforced.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info