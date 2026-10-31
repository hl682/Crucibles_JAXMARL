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
            "position": jax.random.uniform(keys[0], (), minval=-0.25, maxval=0.25, dtype=jnp.float32),
            "velocity": jnp.array(0.0, dtype=jnp.float32),
            "goal": jax.random.uniform(keys[1], (2,), minval=-0.8, maxval=0.8, dtype=jnp.float32),
            "width": jax.random.uniform(keys[2], (2,), minval=0.2, maxval=0.4, dtype=jnp.float32),
            "charge": jnp.ones((2,), dtype=jnp.float32),
            "gear": jnp.ones((2,), dtype=jnp.float32),
            "pending": jnp.zeros((2, 2), dtype=jnp.float32),
            "dwell": jnp.zeros((2,), dtype=jnp.int32),
            "last_effort": jnp.zeros((2,), dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["position"])
        obs = obs.at[:, 2].set(state["velocity"] / 0.35)
        obs = obs.at[:, 3].set(state["goal"])
        obs = obs.at[:, 4].set(state["width"] / 0.4)
        obs = obs.at[:, 5].set(state["charge"])
        obs = obs.at[:, 6].set(jnp.roll(state["charge"], 1))
        obs = obs.at[:, 7].set(state["gear"])
        obs = obs.at[:, 8].set(jnp.roll(state["gear"], 1))
        obs = obs.at[:, 9].set(state["pending"][:, 0] / 0.45)
        obs = obs.at[:, 10].set(state["pending"][:, 1] / 0.45)
        obs = obs.at[:, 11].set(jnp.roll(state["pending"][:, 0], 1) / 0.45)
        obs = obs.at[:, 12].set(jnp.roll(state["pending"][:, 1], 1) / 0.45)
        obs = obs.at[:, 13].set(state["dwell"] / 4.0)
        obs = obs.at[:, 14].set(jnp.roll(state["last_action"], 1) / 4.0)
        obs = obs.at[:, 15].set(state["last_effort"] / 0.45)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        direction = jnp.where(actions == 1, 1.0, jnp.where(actions == 2, -1.0, 0.0)).astype(jnp.float32)
        pulse = direction * jnp.minimum(state["charge"], 0.45)
        brake = jnp.where(actions == 3, jnp.minimum(state["charge"], 0.25), 0.0)
        flip = (actions == 4) & (state["charge"] >= 0.08)
        gear = jnp.where(flip, -state["gear"], state["gear"])
        fired = state["pending"][:, 0] * gear
        damping = 0.84 - 0.8 * jnp.sum(brake)
        velocity = jnp.clip(damping * state["velocity"] + 0.35 * jnp.sum(fired), -0.35, 0.35)
        unconstrained_position = state["position"] + velocity
        collision = jnp.abs(unconstrained_position) > 1.0
        position = jnp.clip(unconstrained_position, -1.0, 1.0)
        velocity = jnp.where(collision, -0.2 * velocity, velocity)
        quality = jnp.exp(-jnp.square((position - state["goal"]) / state["width"]))
        dwell = jnp.where(quality >= 0.5, jnp.minimum(state["dwell"] + 1, 4), 0).astype(jnp.int32)
        effort = jnp.abs(pulse) + brake + 0.08 * flip.astype(jnp.float32)
        charge = jnp.clip(state["charge"] - effort + 0.06 + 0.12 * (actions == 0).astype(jnp.float32), 0.0, 1.0)
        pending = jnp.stack((state["pending"][:, 1], pulse), axis=1)
        rewards = 0.40 * quality + 0.35 * (dwell >= 3).astype(jnp.float32)
        rewards = rewards - 0.10 * jnp.abs(velocity) / 0.35 - 0.12 * effort - 0.05 * collision.astype(jnp.float32)
        rewards = jnp.clip(rewards, -1.0, 1.0).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "position": position.astype(jnp.float32),
            "velocity": velocity.astype(jnp.float32),
            "goal": state["goal"],
            "width": state["width"],
            "charge": charge.astype(jnp.float32),
            "gear": gear.astype(jnp.float32),
            "pending": pending.astype(jnp.float32),
            "dwell": dwell,
            "last_effort": effort.astype(jnp.float32),
            "last_action": actions.astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
