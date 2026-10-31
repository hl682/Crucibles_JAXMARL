import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        symbols = jax.random.randint(key, (2, 2), 0, 3, dtype=jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "emitter": symbols[0],
            "decoder": jnp.zeros((2,), dtype=jnp.int32),
            "target": symbols[1],
            "charge": jnp.full((2,), 0.75, dtype=jnp.float32),
            "flight_glyph": jnp.zeros((2,), dtype=jnp.int32),
            "flight_ttl": jnp.zeros((2,), dtype=jnp.int32),
            "mail_glyph": jnp.zeros((2,), dtype=jnp.int32),
            "mail_valid": jnp.zeros((2,), dtype=jnp.bool_),
            "congestion": jnp.array(0.0, dtype=jnp.float32),
            "last_delivery": jnp.zeros((2,), dtype=jnp.bool_),
            "last_success": jnp.zeros((2,), dtype=jnp.bool_),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["charge"])
        obs = obs.at[:, 2].set(state["emitter"] / 2.0)
        obs = obs.at[:, 3].set(state["decoder"] / 2.0)
        obs = obs.at[:, 4].set(state["target"] / 2.0)
        obs = obs.at[:, 5].set(jnp.where(state["mail_valid"], state["mail_glyph"] / 2.0, -1.0))
        obs = obs.at[:, 6].set(state["mail_valid"].astype(jnp.float32))
        obs = obs.at[:, 7].set(state["flight_glyph"] / 2.0)
        obs = obs.at[:, 8].set(state["flight_ttl"] / 4.0)
        obs = obs.at[:, 9].set(jnp.roll(state["flight_ttl"], 1) / 4.0)
        obs = obs.at[:, 10].set(jnp.roll(state["emitter"], 1) / 2.0)
        obs = obs.at[:, 11].set(state["congestion"])
        obs = obs.at[:, 12].set(state["last_delivery"].astype(jnp.float32))
        obs = obs.at[:, 13].set(state["last_success"].astype(jnp.float32))
        obs = obs.at[:, 14].set(state["last_action"] / 4.0)
        obs = obs.at[:, 15].set(jnp.roll(state["last_action"], 1) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        charge_start = jnp.minimum(state["charge"] + 0.08, 1.0)
        sending = (actions == 2) & (state["flight_ttl"] == 0) & (charge_start >= 0.25)
        collision = jnp.all(sending)
        launch_ttl = (
            jnp.array(2, dtype=jnp.int32)
            + collision.astype(jnp.int32)
            + (state["congestion"] > 0.60).astype(jnp.int32)
        )
        arrivals = jnp.roll(state["flight_ttl"] == 1, 1)
        arriving_glyph = jnp.roll(state["flight_glyph"], 1)
        accepting = (actions == 4) & state["mail_valid"]
        decoded = jnp.mod(state["mail_glyph"] + state["decoder"], 3)
        success = accepting & (decoded == state["target"])
        failure = accepting & ~success
        overflow = arrivals & state["mail_valid"] & ~accepting
        target_increment = jax.random.randint(key, (2,), 1, 3, dtype=jnp.int32)
        rotating = (actions == 1) | (actions == 3)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "emitter": jnp.mod(state["emitter"] + (actions == 1).astype(jnp.int32), 3),
            "decoder": jnp.mod(state["decoder"] + (actions == 3).astype(jnp.int32), 3),
            "target": jnp.where(accepting, jnp.mod(state["target"] + target_increment, 3), state["target"]),
            "charge": jnp.clip(charge_start - 0.25 * sending - 0.03 * rotating, 0.0, 1.0),
            "flight_glyph": jnp.where(sending, state["emitter"], state["flight_glyph"]),
            "flight_ttl": jnp.where(sending, launch_ttl, jnp.maximum(state["flight_ttl"] - 1, 0)).astype(jnp.int32),
            "mail_glyph": jnp.where(arrivals, arriving_glyph, state["mail_glyph"]),
            "mail_valid": arrivals | (state["mail_valid"] & ~accepting),
            "congestion": 0.70 * state["congestion"] + 0.30 * collision.astype(jnp.float32),
            "last_delivery": arrivals,
            "last_success": success,
            "last_action": actions.astype(jnp.int32),
        }
        rewards = (
            0.70 * success.astype(jnp.float32)
            - 0.25 * failure.astype(jnp.float32)
            - 0.04 * sending.astype(jnp.float32)
            - 0.08 * overflow.astype(jnp.float32)
            + 0.10 * jnp.roll(success.astype(jnp.float32), 1)
        )
        rewards = jnp.clip(rewards, -1.0, 1.0).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.broadcast_to(done_all, (2,))
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
