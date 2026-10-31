"""Cross-wired binary registers with delayed pulses and stability audits."""

import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        bits = jax.random.randint(key, (2, 2, 3), 0, 2, dtype=jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "register": bits[0],
            "target": bits[1],
            "pulse": jnp.zeros((2, 3), dtype=jnp.int32),
            "shield_charge": jnp.full((2,), 0.70, dtype=jnp.float32),
            "audit_timer": jnp.zeros((2,), dtype=jnp.int32),
            "snapshot_match": jnp.zeros((2,), dtype=jnp.bool_),
            "audit_dirty": jnp.zeros((2,), dtype=jnp.bool_),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1:4].set(state["register"].astype(jnp.float32))
        obs = obs.at[:, 4:7].set(state["target"].astype(jnp.float32))
        obs = obs.at[:, 7:10].set(jnp.roll(state["register"], 1, axis=0).astype(jnp.float32))
        obs = obs.at[:, 10:13].set(state["pulse"].astype(jnp.float32))
        obs = obs.at[:, 13].set(state["shield_charge"])
        obs = obs.at[:, 14].set(state["audit_timer"] / 2.0)
        obs = obs.at[:, 15].set(jnp.roll(state["audit_timer"], 1, axis=0) / 2.0)
        return obs.astype(jnp.float32)

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        flip_first = actions == 0
        flip_second = actions == 1
        rotate = actions == 2
        shield = (
            (actions == 3)
            & (state["shield_charge"] >= 0.35)
            & jnp.any(state["pulse"] != 0, axis=1)
        )
        incoming = jnp.where(shield[:, None], 0, state["pulse"])
        local_flip = jnp.stack(
            (
                flip_first.astype(jnp.int32),
                flip_second.astype(jnp.int32),
                jnp.zeros((2,), dtype=jnp.int32),
            ),
            axis=1,
        )
        register = jnp.mod(state["register"] + incoming + local_flip, 2)
        register = jnp.where(
            rotate[:, None], jnp.roll(register, 1, axis=1), register
        ).astype(jnp.int32)
        outgoing = jnp.stack(
            (
                jnp.zeros((2,), dtype=jnp.int32),
                flip_first.astype(jnp.int32),
                flip_second.astype(jnp.int32),
            ),
            axis=1,
        )
        pulse = jnp.roll(outgoing, 1, axis=0).astype(jnp.int32)
        changed = jnp.any(register != state["register"], axis=1)
        dirty = state["audit_dirty"] | changed
        completed = state["audit_timer"] == 1
        success = completed & state["snapshot_match"] & jnp.logical_not(dirty)
        start = (actions == 4) & (state["audit_timer"] == 0)
        audit_timer = jnp.where(
            start, 2, jnp.maximum(state["audit_timer"] - 1, 0)
        ).astype(jnp.int32)
        snapshot_match = jnp.where(
            start,
            jnp.all(register == state["target"], axis=1),
            state["snapshot_match"],
        )
        audit_dirty = jnp.where(
            start,
            False,
            jnp.where(state["audit_timer"] > 0, dirty, False),
        ).astype(jnp.bool_)
        target = jnp.where(
            success[:, None], 1 - state["target"], state["target"]
        ).astype(jnp.int32)
        shield_charge = jnp.clip(
            state["shield_charge"] + 0.08 - 0.35 * shield.astype(jnp.float32),
            0.0,
            1.0,
        )
        costs = (
            0.020 * (flip_first | flip_second).astype(jnp.float32)
            + 0.010 * rotate.astype(jnp.float32)
            + 0.015 * shield.astype(jnp.float32)
            + 0.015 * start.astype(jnp.float32)
        )
        rewards = jnp.clip(
            jnp.where(completed, jnp.where(success, 0.70, -0.18), 0.0) - costs,
            -1.0,
            1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "register": register,
            "target": target,
            "pulse": pulse,
            "shield_charge": shield_charge.astype(jnp.float32),
            "audit_timer": audit_timer,
            "snapshot_match": snapshot_match.astype(jnp.bool_),
            "audit_dirty": audit_dirty,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
