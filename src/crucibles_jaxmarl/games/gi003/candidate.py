import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "position": jax.random.randint(key, (2,), 0, 8, dtype=jnp.int32),
            "gate": jnp.zeros((2,), dtype=jnp.bool_),
            "target": jnp.array([4, 0], dtype=jnp.int32),
            "last": jnp.zeros((2,), dtype=jnp.int32),
            "delivered": jnp.zeros((2,), dtype=jnp.float32),
            "blocked": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["position"] / 7.0)
        obs = obs.at[:, 1].set(jnp.roll(state["position"], 1) / 7.0)
        obs = obs.at[:, 2].set(state["target"] / 7.0)
        obs = obs.at[:, 3].set(jnp.roll(state["target"], 1) / 7.0)
        obs = obs.at[:, 4].set(state["gate"])
        obs = obs.at[:, 5].set(jnp.roll(state["gate"], 1))
        obs = obs.at[:, 6].set(state["t"] / 32.0)
        obs = obs.at[:, 7].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 8:13].set(jax.nn.one_hot(state["last"], 5))
        obs = obs.at[:, 13].set(state["delivered"])
        obs = obs.at[:, 14].set(state["blocked"])
        obs = obs.at[:, 15].set(state["position"] == state["target"])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        # 0 wait, 1 clockwise, 2 anticlockwise, 3 toggle owned gate, 4 deliver.
        direction = (actions == 1).astype(jnp.int32) - (actions == 2).astype(jnp.int32)
        proposed = jnp.mod(state["position"] + direction, 8)
        # Gates are edges 1<->2 and 5<->6. Old gate states govern this move.
        cross_first = ((state["position"] == 1) & (proposed == 2)) | ((state["position"] == 2) & (proposed == 1))
        cross_second = ((state["position"] == 5) & (proposed == 6)) | ((state["position"] == 6) & (proposed == 5))
        blocked = (cross_first & ~state["gate"][0]) | (cross_second & ~state["gate"][1])
        position = jnp.where(blocked, state["position"], proposed)
        gate = jnp.where(actions == 3, ~state["gate"], state["gate"])
        delivered = ((actions == 4) & (position == state["target"])).astype(jnp.float32)
        target = jnp.where(delivered > 0.0, jnp.mod(state["target"] + 4, 8), state["target"])
        # Sharing a location dilutes delivery service, but never prevents movement.
        crowd = (position[0] == position[1]).astype(jnp.float32)
        rewards = delivered * (0.8 - 0.3 * crowd) - 0.03 * (actions == 3) - 0.02 * blocked
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "position": position.astype(jnp.int32),
            "gate": gate.astype(jnp.bool_),
            "target": target.astype(jnp.int32),
            "last": actions.astype(jnp.int32),
            "delivered": delivered.astype(jnp.float32),
            "blocked": blocked.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards.astype(jnp.float32), dones, info
