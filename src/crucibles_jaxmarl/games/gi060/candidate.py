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
            "inventory": jax.random.uniform(keys[0], (2,), minval=0.6, maxval=1.0, dtype=jnp.float32),
            "escrow": jnp.zeros((2,), dtype=jnp.float32),
            "signature": jnp.zeros((2,), dtype=jnp.int32),
            "pipeline": jnp.zeros((2, 2), dtype=jnp.float32),
            "value": jax.random.uniform(keys[1], (2,), minval=0.55, maxval=0.95, dtype=jnp.float32),
            "inspection": jnp.zeros((2,), dtype=jnp.int32),
            "last_transfer": jnp.zeros((2,), dtype=jnp.float32),
            "last_withdrawal": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        inventory = state["inventory"]
        escrow = state["escrow"]
        signature = state["signature"].astype(jnp.float32) / 3.0
        pipeline = state["pipeline"]
        visible = state["inspection"] > 0
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(inventory)
        obs = obs.at[:, 2].set(inventory[::-1])
        obs = obs.at[:, 3].set(escrow)
        obs = obs.at[:, 4].set(escrow[::-1])
        obs = obs.at[:, 5].set(signature)
        obs = obs.at[:, 6].set(signature[::-1])
        obs = obs.at[:, 7].set(pipeline[:, 0])
        obs = obs.at[:, 8].set(pipeline[:, 1])
        obs = obs.at[:, 9].set(pipeline[::-1, 0])
        obs = obs.at[:, 10].set(pipeline[::-1, 1])
        obs = obs.at[:, 11].set(state["value"])
        obs = obs.at[:, 12].set(jnp.where(visible, state["value"][::-1], 0.0))
        obs = obs.at[:, 13].set(state["inspection"].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 14].set(state["last_transfer"])
        obs = obs.at[:, 15].set(state["last_withdrawal"][::-1])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        deposit = jnp.where(
            actions == 1,
            jnp.minimum(jnp.minimum(state["inventory"], 0.3), 1.0 - state["escrow"]),
            0.0,
        )
        offered = state["escrow"] + deposit
        withdrawal = jnp.where(actions == 3, offered, 0.0)
        available = offered - withdrawal
        signatures = jnp.where(actions == 2, 3, jnp.maximum(state["signature"] - 1, 0)).astype(jnp.int32)
        release = jnp.all(available >= 0.2) & jnp.all(signatures > 0)
        outgoing = jnp.where(release, available, 0.0)
        delivered = state["pipeline"][:, 0]
        pipeline = jnp.stack((state["pipeline"][:, 1], outgoing[::-1]), axis=1)
        inventory = jnp.clip(
            state["inventory"] - deposit + withdrawal
            + 0.04 * (actions == 0).astype(jnp.float32) + 0.6 * delivered,
            0.0, 1.0,
        )
        inspection = jnp.where(actions == 4, 3, jnp.maximum(state["inspection"] - 1, 0)).astype(jnp.int32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "inventory": inventory.astype(jnp.float32),
            "escrow": (available - outgoing).astype(jnp.float32),
            "signature": jnp.where(release, 0, signatures).astype(jnp.int32),
            "pipeline": pipeline.astype(jnp.float32),
            "value": state["value"],
            "inspection": inspection,
            "last_transfer": outgoing.astype(jnp.float32),
            "last_withdrawal": withdrawal.astype(jnp.float32),
        }
        rewards = jnp.clip(
            delivered * state["value"] - 0.15 * deposit
            - 0.025 * (actions == 2).astype(jnp.float32)
            - 0.04 * (actions == 4).astype(jnp.float32) - 0.08 * withdrawal,
            -1.0, 1.0,
        ).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
