"""Perishable two-party escrow with asynchronous consent and delayed delivery."""

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
            "t": jnp.array(0, dtype=jnp.int32),
            "inventory": 0.45 + 0.35 * draws[:2],
            "grade": 0.35 + 0.60 * draws[2:],
            "escrow": jnp.zeros((2,), dtype=jnp.float32),
            "age": jnp.zeros((2,), dtype=jnp.float32),
            "consent": jnp.zeros((2,), dtype=jnp.int32),
            "timer": jnp.array(0, dtype=jnp.int32),
            "shipment_value": jnp.array(0.0, dtype=jnp.float32),
            "inspected": jnp.zeros((2,), dtype=jnp.bool_),
            "last_reward": jnp.zeros((2,), dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        other_escrow = jnp.roll(state["escrow"], 1, axis=0)
        other_age = jnp.roll(state["age"], 1, axis=0)
        other_consent = jnp.roll(state["consent"], 1, axis=0)
        known_grade = jnp.where(
            state["inspected"], jnp.roll(state["grade"], 1, axis=0), 0.0
        )
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["inventory"])
        obs = obs.at[:, 2].set(state["grade"])
        obs = obs.at[:, 3].set(state["escrow"])
        obs = obs.at[:, 4].set(other_escrow)
        obs = obs.at[:, 5].set(state["age"] / 32.0)
        obs = obs.at[:, 6].set(other_age / 32.0)
        obs = obs.at[:, 7].set(state["consent"] / 3.0)
        obs = obs.at[:, 8].set(other_consent / 3.0)
        obs = obs.at[:, 9].set(state["timer"] / 3.0)
        obs = obs.at[:, 10].set(state["shipment_value"])
        obs = obs.at[:, 11].set(known_grade)
        obs = obs.at[:, 12].set(state["inspected"].astype(jnp.float32))
        obs = obs.at[:, 13].set(state["last_reward"])
        obs = obs.at[:, 14].set(state["last_action"] / 4.0)
        obs = obs.at[:, 15].set(jnp.roll(state["last_action"], 1, axis=0) / 4.0)
        return obs.astype(jnp.float32)

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        produce = actions == 0
        pledge = actions == 1
        authorize = actions == 2
        sell = actions == 3
        inspect = actions == 4
        deposited = jnp.where(
            pledge,
            jnp.minimum(
                jnp.minimum(state["inventory"], 0.5),
                jnp.maximum(1.0 - state["escrow"], 0.0),
            ),
            0.0,
        )
        private_sale = jnp.where(
            sell, 0.10 * state["inventory"] + 0.025 * state["escrow"], 0.0
        )
        inventory = jnp.clip(
            jnp.where(sell, 0.0, state["inventory"] - deposited)
            + 0.35 * produce.astype(jnp.float32),
            0.0,
            1.0,
        )
        pre_spoil = jnp.where(sell, 0.0, state["escrow"] + deposited)
        escrow = 0.985 * pre_spoil
        age = jnp.where(
            pre_spoil > 0.0,
            jnp.where(state["escrow"] > 0.0, state["age"] + 1.0, 0.0),
            0.0,
        )
        consent = jnp.where(
            authorize, 3, jnp.maximum(state["consent"] - 1, 0)
        ).astype(jnp.int32)
        consent = jnp.where(sell, 0, consent).astype(jnp.int32)
        launch = (
            (state["timer"] == 0)
            & jnp.all(escrow >= 0.35)
            & jnp.all(consent > 0)
        )
        completed = state["timer"] == 1
        batch_value = (
            0.82
            * jnp.minimum(jnp.min(escrow) / 0.5, 1.0)
            * (0.4 + 0.6 * jnp.mean(state["grade"]))
            * jnp.exp(-0.035 * jnp.mean(age))
        )
        timer = jnp.where(
            launch, 3, jnp.maximum(state["timer"] - 1, 0)
        ).astype(jnp.int32)
        shipment_value = jnp.where(
            launch,
            batch_value,
            jnp.where(completed, 0.0, state["shipment_value"]),
        ).astype(jnp.float32)
        costs = (
            0.025 * produce.astype(jnp.float32)
            + 0.012 * pledge.astype(jnp.float32)
            + 0.008 * authorize.astype(jnp.float32)
            + 0.015 * inspect.astype(jnp.float32)
        )
        rewards = jnp.clip(
            private_sale
            + completed.astype(jnp.float32) * state["shipment_value"]
            - costs,
            -1.0,
            1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "inventory": inventory.astype(jnp.float32),
            "grade": state["grade"],
            "escrow": jnp.where(launch, 0.0, escrow).astype(jnp.float32),
            "age": jnp.where(launch, 0.0, age).astype(jnp.float32),
            "consent": jnp.where(launch, 0, consent).astype(jnp.int32),
            "timer": timer,
            "shipment_value": shipment_value,
            "inspected": state["inspected"] | inspect,
            "last_reward": rewards,
            "last_action": actions.astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
