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
            "queue": 0.6 * jnp.eye(2, dtype=jnp.float32),
            "near": jnp.zeros((2, 2), dtype=jnp.float32),
            "far": jnp.zeros((2, 2), dtype=jnp.float32),
            "energy": jax.random.uniform(keys[0], (2,), minval=0.65, maxval=1.0, dtype=jnp.float32),
            "condition": jax.random.uniform(keys[1], (2,), minval=0.5, maxval=0.9, dtype=jnp.float32),
            "last_reward": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        ids = jnp.arange(2, dtype=jnp.int32)
        other = 1 - ids
        near_load = jnp.sum(state["near"], axis=1)
        far_load = jnp.sum(state["far"], axis=1)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(jnp.array([-1.0, 1.0], dtype=jnp.float32))
        obs = obs.at[:, 2].set(state["queue"][ids, ids] / 4.0)
        obs = obs.at[:, 3].set(state["queue"][ids, other] / 4.0)
        obs = obs.at[:, 4].set(state["queue"][other, ids] / 4.0)
        obs = obs.at[:, 5].set(state["queue"][other, other] / 4.0)
        obs = obs.at[:, 6].set(near_load)
        obs = obs.at[:, 7].set(jnp.roll(near_load, 1))
        obs = obs.at[:, 8].set(far_load)
        obs = obs.at[:, 9].set(jnp.roll(far_load, 1))
        obs = obs.at[:, 10].set(state["energy"])
        obs = obs.at[:, 11].set(jnp.roll(state["energy"], 1))
        obs = obs.at[:, 12].set(state["condition"])
        obs = obs.at[:, 13].set(jnp.roll(state["condition"], 1))
        obs = obs.at[:, 14].set(state["last_reward"])
        obs = obs.at[:, 15].set(jnp.roll(state["last_reward"], 1))
        return obs

    def legal_actions(self, state):
        loaded = jnp.sum(state["queue"], axis=1) > 0.00001
        can_inject = state["energy"] >= 0.25
        can_process = loaded & (state["energy"] >= 0.22)
        can_route = loaded & (state["energy"] >= 0.16)
        can_maintain = state["energy"] >= 0.12
        return jnp.stack((jnp.ones((2,), dtype=jnp.bool_), can_inject, can_process, can_route, can_maintain), axis=1)

    def raw_step(self, key, state, actions):
        legal = self.legal_actions(state)
        inject = (actions == 1) & legal[:, 1]
        process = (actions == 2) & legal[:, 2]
        route = (actions == 3) & legal[:, 3]
        maintain = (actions == 4) & legal[:, 4]
        load = jnp.sum(state["queue"], axis=1)
        safe_load = jnp.maximum(load, 0.000001)
        processing_amount = jnp.where(process, jnp.minimum(load, 0.35 + 0.65 * state["condition"]), 0.0)
        routing_amount = jnp.where(route, jnp.minimum(load, 1.0), 0.0)
        processed = state["queue"] * (processing_amount / safe_load)[:, None]
        sent = state["queue"] * (routing_amount / safe_load)[:, None]
        remainder = jnp.maximum(state["queue"] - processed - sent, 0.0)
        injected = jnp.eye(2, dtype=jnp.float32) * inject.astype(jnp.float32)[:, None]
        before_storage = remainder + injected + state["near"]
        surviving = 0.985 * before_storage
        storage_load = jnp.sum(surviving, axis=1)
        storage_factor = jnp.minimum(1.0, 4.0 / jnp.maximum(storage_load, 0.000001))
        queue = surviving * storage_factor[:, None]
        near = 0.985 * state["far"]
        far = 0.985 * jnp.roll(sent, 1, axis=0)
        storage_loss = jnp.sum(before_storage - queue, axis=0)
        transport_loss = 0.015 * jnp.sum(state["far"] + sent, axis=0)
        owner_dividend = 0.6 * jnp.sum(processed, axis=0)
        station_fee = 0.2 * jnp.sum(processed, axis=1)
        utility = (
            owner_dividend + station_fee
            - 0.08 * inject.astype(jnp.float32)
            - 0.035 * route.astype(jnp.float32)
            - 0.03 * maintain.astype(jnp.float32)
            - 0.08 * (storage_loss + transport_loss)
        )
        rewards = jnp.tanh(utility).astype(jnp.float32)
        energy = jnp.clip(
            state["energy"] + 0.08 + 0.12 * (actions == 0).astype(jnp.float32)
            - 0.25 * inject.astype(jnp.float32)
            - 0.22 * process.astype(jnp.float32)
            - 0.16 * route.astype(jnp.float32)
            - 0.12 * maintain.astype(jnp.float32),
            0.0, 1.0,
        )
        condition = jnp.clip(
            state["condition"] - 0.004
            - 0.10 * process.astype(jnp.float32)
            - 0.03 * route.astype(jnp.float32)
            + 0.20 * maintain.astype(jnp.float32),
            0.25, 1.0,
        )
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "queue": queue.astype(jnp.float32),
            "near": near.astype(jnp.float32),
            "far": far.astype(jnp.float32),
            "energy": energy.astype(jnp.float32),
            "condition": condition.astype(jnp.float32),
            "last_reward": rewards,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
