import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        backlog_key, heat_key, phase_key = jax.random.split(key, 3)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "backlog": jax.random.uniform(backlog_key, (2,), minval=0.5, maxval=1.5, dtype=jnp.float32),
            "heat": jax.random.uniform(heat_key, (2, 2), minval=0.1, maxval=0.4, dtype=jnp.float32),
            "pulse": jnp.zeros((2, 2), dtype=jnp.float32),
            "shield": jnp.zeros((2,), dtype=jnp.float32),
            "phase": jax.random.randint(phase_key, (2,), 0, 3, dtype=jnp.int32),
            "last_service": jnp.zeros((2,), dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        other = jnp.array([1, 0], dtype=jnp.int32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["backlog"] / 5.0)
        obs = obs.at[:, 2].set(state["heat"][:, 0] / 3.0)
        obs = obs.at[:, 3].set(state["heat"][:, 1] / 3.0)
        obs = obs.at[:, 4].set(state["pulse"][:, 0])
        obs = obs.at[:, 5].set(state["pulse"][:, 1])
        obs = obs.at[:, 6].set(state["shield"])
        obs = obs.at[:, 7].set(state["phase"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 8].set(state["heat"][other, 0] / 3.0)
        obs = obs.at[:, 9].set(state["heat"][other, 1] / 3.0)
        obs = obs.at[:, 10].set(state["shield"][other])
        obs = obs.at[:, 11].set(state["backlog"][other] / 5.0)
        obs = obs.at[:, 12].set(state["last_service"])
        obs = obs.at[:, 13].set(state["last_service"][other])
        obs = obs.at[:, 14].set(state["last_action"].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 15].set(state["last_action"][other].astype(jnp.float32) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        absorbed = state["pulse"] * state["shield"][:, None]
        effective_heat = jnp.clip(state["heat"] + state["pulse"] - absorbed, 0.0, 3.0)
        route_mask = jax.nn.one_hot(actions - 1, 2, dtype=jnp.float32)
        routed = (actions == 1) | (actions == 2)
        chosen_heat = jnp.sum(effective_heat * route_mask, axis=1)
        service = jnp.minimum(state["backlog"], 1.0) * (routed & (chosen_heat < 1.25)).astype(jnp.float32)
        work_heat = route_mask * (0.65 * service[:, None])
        hottest = jnp.argmax(effective_heat, axis=1)
        vent_mask = jax.nn.one_hot(hottest, 2, dtype=jnp.float32)
        vent_heat = vent_mask * jnp.minimum(effective_heat, 0.7) * (actions == 4).astype(jnp.float32)[:, None]
        local_spread = (1.0 - route_mask) * (0.12 * service[:, None])
        cooling = 0.18 + 0.22 * (actions == 0).astype(jnp.float32)
        next_heat = jnp.clip(effective_heat + work_heat + local_spread - vent_heat - cooling[:, None], 0.0, 3.0)
        next_pulse = 0.32 * jnp.roll(work_heat, 1, axis=0) + 0.8 * jnp.roll(vent_heat, (1, 1), axis=(0, 1))
        next_shield = jnp.clip(state["shield"] - 0.5 * jnp.sum(absorbed, axis=1) + 0.45 * (actions == 3).astype(jnp.float32), 0.0, 1.0)
        arrival = 0.15 + 0.65 * (jnp.mod(state["t"] + state["phase"], 3) == 0).astype(jnp.float32)
        unbounded_backlog = state["backlog"] - service + arrival
        overflow = jnp.maximum(unbounded_backlog - 5.0, 0.0)
        next_backlog = jnp.clip(unbounded_backlog, 0.0, 5.0)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "backlog": next_backlog,
            "heat": next_heat,
            "pulse": next_pulse,
            "shield": next_shield,
            "phase": state["phase"],
            "last_service": service,
            "last_action": actions.astype(jnp.int32),
        }
        hot_cost = 0.08 * jnp.maximum(jnp.max(next_heat, axis=1) - 1.7, 0.0)
        rewards = jnp.clip(0.55 * service - 0.035 * next_backlog - hot_cost - 0.03 * (actions == 3).astype(jnp.float32) - 0.02 * (actions == 4).astype(jnp.float32) - 0.12 * overflow, -1.0, 1.0).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info