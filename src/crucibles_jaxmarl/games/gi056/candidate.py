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
            "stock": jax.random.uniform(keys[0], (2,), minval=1.0, maxval=2.0, dtype=jnp.float32),
            "credit": jnp.full((2,), 1.5, dtype=jnp.float32),
            "outbox": jnp.zeros((2,), dtype=jnp.float32),
            "active": jnp.zeros((2,), dtype=jnp.float32),
            "progress": jnp.zeros((2,), dtype=jnp.float32),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "corridor": jnp.array(0.7, dtype=jnp.float32),
            "demand": jax.random.uniform(keys[1], (2,), minval=0.3, maxval=1.0, dtype=jnp.float32),
            "last_delivered": jnp.zeros((2,), dtype=jnp.float32),
            "last_lost": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        other = jnp.array([1, 0], dtype=jnp.int32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["stock"] / 3.0)
        obs = obs.at[:, 2].set(state["credit"] / 2.0)
        obs = obs.at[:, 3].set(state["outbox"])
        obs = obs.at[:, 4].set(state["active"])
        obs = obs.at[:, 5].set(state["progress"])
        obs = obs.at[:, 6].set(state["age"].astype(jnp.float32) / 6.0)
        obs = obs.at[:, 7].set(state["active"][other])
        obs = obs.at[:, 8].set(state["progress"][other])
        obs = obs.at[:, 9].set(state["age"][other].astype(jnp.float32) / 6.0)
        obs = obs.at[:, 10].set(state["corridor"])
        obs = obs.at[:, 11].set(state["demand"])
        obs = obs.at[:, 12].set(state["last_delivered"])
        obs = obs.at[:, 13].set(state["last_delivered"][other])
        obs = obs.at[:, 14].set(state["last_lost"])
        obs = obs.at[:, 15].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        other = jnp.array([1, 0], dtype=jnp.int32)
        active = state["active"] > 0.5
        pack = ((actions == 1) & (state["stock"] >= 1.0) & (state["outbox"] < 0.5)).astype(jnp.float32)
        launch = ((actions == 2) & (state["outbox"] > 0.5) & (state["credit"] >= 0.75) & (~active)).astype(jnp.float32)
        maintenance = (actions == 4).astype(jnp.float32)
        moved = jnp.minimum(1.0, state["progress"] + 0.15 + 0.45 * state["corridor"])
        aged = state["age"] + active.astype(jnp.int32)
        received = (active & (moved >= 1.0) & (actions[other] == 3)).astype(jnp.float32)
        lost = (active & (aged >= 6) & (received < 0.5)).astype(jnp.float32)
        retained = state["active"] - received - lost
        next_active = retained + launch
        incoming = received[other]
        new_demand = jax.random.uniform(key, (2,), minval=0.3, maxval=1.0, dtype=jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "stock": jnp.clip(state["stock"] - pack + 0.12, 0.0, 3.0),
            "credit": jnp.clip(state["credit"] - 0.75 * launch + 0.75 * received + 0.025, 0.0, 2.0),
            "outbox": state["outbox"] + pack - launch,
            "active": next_active,
            "progress": jnp.where(retained > 0.5, moved, 0.0).astype(jnp.float32),
            "age": jnp.where(retained > 0.5, aged, 0).astype(jnp.int32),
            "corridor": jnp.clip(state["corridor"] + 0.15 * jnp.sum(maintenance) - 0.06 * jnp.sum(launch) - 0.025, 0.0, 1.0),
            "demand": jnp.where(incoming > 0.5, new_demand, state["demand"]),
            "last_delivered": received,
            "last_lost": lost,
        }
        utility = 0.6 * received + 0.45 * state["demand"] * incoming - 0.5 * lost - 0.025 * pack - 0.02 * launch - 0.04 * maintenance
        rewards = jnp.tanh(utility).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
