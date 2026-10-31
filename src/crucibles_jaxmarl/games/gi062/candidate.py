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
            "register": jax.random.randint(keys[0], (2,), 0, 3, dtype=jnp.int32),
            "target": jax.random.randint(keys[1], (2,), 0, 3, dtype=jnp.int32),
            "epoch": jnp.zeros((2,), dtype=jnp.int32),
            "packet_value": jnp.zeros((2, 2), dtype=jnp.int32),
            "packet_epoch": jnp.zeros((2, 2), dtype=jnp.int32),
            "packet_valid": jnp.zeros((2, 2), dtype=jnp.bool_),
            "inbox_value": jnp.zeros((2,), dtype=jnp.int32),
            "inbox_epoch": jnp.zeros((2,), dtype=jnp.int32),
            "inbox_valid": jnp.zeros((2,), dtype=jnp.bool_),
            "last_success": jnp.zeros((2,), dtype=jnp.bool_),
        }
        return self.observe(state), state

    def observe(self, state):
        registers = state["register"].astype(jnp.float32) / 2.0
        epochs = state["epoch"].astype(jnp.float32) / 32.0
        valid = state["inbox_valid"]
        packets = state["packet_value"].astype(jnp.float32) / 2.0
        packet_valid = state["packet_valid"]
        stale = valid & (state["inbox_epoch"] != state["epoch"][::-1])
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(registers)
        obs = obs.at[:, 2].set(registers[::-1])
        obs = obs.at[:, 3].set(state["target"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 4].set(epochs)
        obs = obs.at[:, 5].set(epochs[::-1])
        obs = obs.at[:, 6].set(jnp.where(valid, state["inbox_value"].astype(jnp.float32) / 2.0, 0.0))
        obs = obs.at[:, 7].set(valid.astype(jnp.float32))
        obs = obs.at[:, 8].set(state["inbox_epoch"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 9].set(jnp.where(packet_valid[:, 0], packets[:, 0], 0.0))
        obs = obs.at[:, 10].set(packet_valid[:, 0].astype(jnp.float32))
        obs = obs.at[:, 11].set(jnp.where(packet_valid[:, 1], packets[:, 1], 0.0))
        obs = obs.at[:, 12].set(packet_valid[:, 1].astype(jnp.float32))
        obs = obs.at[:, 13].set(state["last_success"].astype(jnp.float32))
        obs = obs.at[:, 14].set(state["last_success"][::-1].astype(jnp.float32))
        obs = obs.at[:, 15].set(stale.astype(jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        transmit = actions == 3
        execute = actions == 4
        success = execute & (state["register"] == state["target"][::-1])
        failure = execute & jnp.logical_not(success)
        registers = jnp.mod(
            state["register"] + (actions == 1).astype(jnp.int32) - (actions == 2).astype(jnp.int32),
            3,
        ).astype(jnp.int32)
        delivery = state["packet_valid"][::-1, 0]
        inbox_value = jnp.where(delivery, state["packet_value"][::-1, 0], state["inbox_value"])
        inbox_epoch = jnp.where(delivery, state["packet_epoch"][::-1, 0], state["inbox_epoch"])
        redraw = jax.random.randint(key, (2,), 0, 3, dtype=jnp.int32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "register": registers,
            "target": jnp.where(execute[::-1], redraw, state["target"]).astype(jnp.int32),
            "epoch": state["epoch"] + execute[::-1].astype(jnp.int32),
            "packet_value": jnp.stack((state["packet_value"][:, 1], state["target"]), axis=1),
            "packet_epoch": jnp.stack((state["packet_epoch"][:, 1], state["epoch"]), axis=1),
            "packet_valid": jnp.stack((state["packet_valid"][:, 1], transmit), axis=1),
            "inbox_value": inbox_value.astype(jnp.int32),
            "inbox_epoch": inbox_epoch.astype(jnp.int32),
            "inbox_valid": state["inbox_valid"] | delivery,
            "last_success": success,
        }
        rewards = jnp.clip(
            0.6 * success.astype(jnp.float32) + 0.2 * success[::-1].astype(jnp.float32)
            - 0.25 * failure.astype(jnp.float32) - 0.03 * transmit.astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
