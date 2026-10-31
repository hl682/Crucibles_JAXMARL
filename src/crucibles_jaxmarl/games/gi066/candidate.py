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
            "t": jnp.array(0, dtype=jnp.int32),
            "pressure": jax.random.uniform(keys[0], (2,), minval=0.2, maxval=0.7, dtype=jnp.float32),
            "heat": jax.random.uniform(keys[1], (2,), minval=0.0, maxval=0.25, dtype=jnp.float32),
            "gate": jax.random.randint(keys[2], (), 0, 2, dtype=jnp.int32),
            "transit": jnp.zeros((2,), dtype=jnp.float32),
            "stress": jnp.array(0.1, dtype=jnp.float32),
            "cooldown": jnp.array(0, dtype=jnp.int32),
            "last_draw": jnp.zeros((2,), dtype=jnp.float32),
            "last_overflow": jnp.zeros((2,), dtype=jnp.float32),
            "total_draw": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["pressure"])
        obs = obs.at[:, 1].set(jnp.roll(state["pressure"], 1))
        obs = obs.at[:, 2].set(state["transit"] / 0.16)
        obs = obs.at[:, 3].set(jnp.roll(state["transit"], 1) / 0.16)
        obs = obs.at[:, 4].set(state["heat"])
        obs = obs.at[:, 5].set(jnp.roll(state["heat"], 1))
        obs = obs.at[:, 6].set((jnp.arange(2, dtype=jnp.int32) == state["gate"]).astype(jnp.float32))
        obs = obs.at[:, 7].set(state["stress"])
        obs = obs.at[:, 8].set(state["cooldown"].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 9].set(state["last_draw"] / 0.25)
        obs = obs.at[:, 10].set(jnp.roll(state["last_draw"], 1) / 0.25)
        obs = obs.at[:, 11].set(state["last_overflow"])
        obs = obs.at[:, 12].set(jnp.roll(state["last_overflow"], 1))
        obs = obs.at[:, 13].set(state["total_draw"] / 8.0)
        obs = obs.at[:, 14].set(jnp.roll(state["total_draw"], 1) / 8.0)
        obs = obs.at[:, 15].set(state["t"].astype(jnp.float32) / 32.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        coast = (actions == 0).astype(jnp.float32)
        pump = (actions == 1).astype(jnp.float32)
        collect = (actions == 2).astype(jnp.float32)
        switch = (actions == 3).astype(jnp.float32)
        reinforce = (actions == 4).astype(jnp.float32)
        arrived = jnp.roll(state["transit"], 1)
        filled = state["pressure"] + arrived + 0.12 * pump
        overflow = jnp.maximum(filled - 1.0, 0.0)
        available = jnp.clip(filled, 0.0, 1.0)
        draw = 0.25 * available * collect * (1.0 - 0.5 * state["heat"])
        after_draw = available - draw
        donor = (jnp.arange(2, dtype=jnp.int32) == state["gate"]).astype(jnp.float32)
        head = jnp.sum(donor * (after_draw - jnp.roll(after_draw, 1)))
        flow = 0.16 * jnp.maximum(head, 0.0) * (state["cooldown"] == 0).astype(jnp.float32)
        dispatched = donor * flow
        stress_load = (
            0.86 * state["stress"] + 0.22 * jnp.sum(switch)
            + 0.18 * jnp.maximum(-head, 0.0) - 0.16 * jnp.sum(reinforce)
        )
        burst = (stress_load > 0.85) & (state["cooldown"] == 0)
        next_transit = jnp.where(burst, jnp.zeros((2,), dtype=jnp.float32), dispatched)
        next_pressure = jnp.clip(0.99 * (after_draw - dispatched), 0.0, 1.0)
        next_heat = jnp.clip(0.84 * state["heat"] + 0.7 * draw - 0.04 * coast - 0.03 * reinforce, 0.0, 1.0)
        next_stress = jnp.where(burst, 0.15, jnp.clip(stress_load, 0.0, 1.0))
        next_cooldown = jnp.where(burst, 3, jnp.maximum(state["cooldown"] - 1, 0)).astype(jnp.int32)
        switch_count = jnp.sum((actions == 3).astype(jnp.int32))
        next_gate = jnp.mod(state["gate"] + switch_count, 2).astype(jnp.int32)
        rewards = jnp.tanh(
            1.7 * draw - 0.04 * pump - 0.03 * switch - 0.035 * reinforce
            - 0.08 * burst.astype(jnp.float32) - 0.05 * overflow
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "pressure": next_pressure.astype(jnp.float32),
            "heat": next_heat.astype(jnp.float32),
            "gate": next_gate,
            "transit": next_transit.astype(jnp.float32),
            "stress": next_stress.astype(jnp.float32),
            "cooldown": next_cooldown,
            "last_draw": draw.astype(jnp.float32),
            "last_overflow": overflow.astype(jnp.float32),
            "total_draw": (state["total_draw"] + draw).astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
