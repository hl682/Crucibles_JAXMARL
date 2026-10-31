"""Windlift Quarry. Original prospective source; never executed by its author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 12
    observation_size = 186
    max_steps = 160
    obs_keys = ("t", "done", "terminated", "truncated", "tiles", "agent_pos",
                "agent_active", "carrying", "block_pos", "block_active", "block_kind",
                "block_owner", "block_mass", "block_status", "lift_pos", "lift_active",
                "lift_target", "rope", "countermass", "rig_pos", "rig_active", "rig_kind",
                "depot_pos", "depot_active", "delivered", "hoisted", "delivered_now",
                "repaired", "sale_now")

    def refresh(self, s):
        s = dict(s)
        status = s["block_status"]
        held = status[None, :] == jnp.arange(3)[:, None]
        s["carrying"] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        carrier = jnp.clip(status, 0, 2)
        s["block_pos"] = jnp.where((status >= 0)[:, None], s["agent_pos"][carrier], s["block_pos"])
        s["block_active"] = status >= -2
        s["block_visible"] = jnp.broadcast_to(s["block_active"][None, :], (3, 10))
        s["countermass"] = jnp.array([jnp.sum(jnp.where(status == -2, s["block_mass"], 0))], dtype=jnp.int32)
        return s

    def reset(self, key):
        tiles = jnp.zeros((7, 9), dtype=jnp.int32)
        tiles = tiles.at[3, 1:8].set(1)
        tiles = tiles.at[2:5, 4].set(2)
        s = {"t": jnp.array(0, jnp.int32), "done": jnp.array(False),
             "terminated": jnp.array(False), "truncated": jnp.array(False), "tiles": tiles,
             "agent_pos": jnp.array([[3, 5], [5, 5], [4, 5]], jnp.int32),
             "agent_active": jnp.ones(3, jnp.bool_), "agent_visible": jnp.ones((3, 3), jnp.bool_),
             "carrying": jnp.full(3, -1, jnp.int32),
             "block_pos": jnp.array([[4, 5], [2, 5], [6, 5], [1, 6], [7, 6], [2, 6], [5, 5], [7, 5], [3, 6], [6, 6]], jnp.int32),
             "block_active": jnp.ones(10, jnp.bool_), "block_visible": jnp.ones((3, 10), jnp.bool_),
             "block_kind": jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], jnp.int32),
             "block_owner": jnp.array([0, 1, 2, 0, 1, 2, 1, 0, -1, -1], jnp.int32),
             "block_mass": jnp.array([1, 1, 1, 1, 1, 1, 2, 2, 1, 1], jnp.int32),
             "block_status": jnp.full(10, -1, jnp.int32),
             "lift_pos": jnp.array([[4, 5]], jnp.int32), "lift_active": jnp.ones(1, jnp.bool_),
             "lift_visible": jnp.ones((3, 1), jnp.bool_), "lift_target": jnp.array([5], jnp.int32),
             "rope": jnp.array([8], jnp.int32), "countermass": jnp.zeros(1, jnp.int32),
             "rig_pos": jnp.array([[3, 5], [5, 5]], jnp.int32), "rig_kind": jnp.array([0, 1], jnp.int32),
             "rig_active": jnp.ones(2, jnp.bool_), "rig_visible": jnp.ones((3, 2), jnp.bool_),
             "depot_pos": jnp.array([[2, 0], [4, 0], [6, 0]], jnp.int32),
             "depot_active": jnp.ones(3, jnp.bool_), "depot_visible": jnp.ones((3, 3), jnp.bool_),
             "delivered": jnp.zeros(3, jnp.int32), "hoisted": jnp.array(False),
             "delivered_now": jnp.zeros(3, jnp.int32), "repaired": jnp.array(False),
             "sale_now": jnp.zeros(3, jnp.int32)}
        s = self.refresh(s)
        return self.observe(s), s

    def movement(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = s["agent_pos"][:, None, :] + delta[None, :, :]
        bounds = jnp.all((q >= 0) & (q <= jnp.array([8, 6])), axis=2)
        tile = s["tiles"][jnp.clip(q[:, :, 1], 0, 6), jnp.clip(q[:, :, 0], 0, 8)]
        on_platform = jnp.all(q == s["lift_pos"][0], axis=2)
        occupied = jnp.any(jnp.all(q[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=3)
                           & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=2)
        travelling = s["lift_pos"][0, 1] != s["lift_target"][0]
        rider = jnp.all(s["agent_pos"] == s["lift_pos"][0], axis=1)
        reserved = travelling & (q[:, :, 0] == 4) & (q[:, :, 1] == s["lift_target"][0])
        ok = bounds & ((tile == 0) | on_platform) & ~occupied & ~reserved
        ok = ok & ~(travelling & rider[:, None])
        return q, ok.at[:, 0].set(True)

    def legal_actions(self, s):
        q, move = self.movement(s)
        pos = s["agent_pos"]
        carry = s["carrying"]
        held = jnp.clip(carry, 0, 9)
        free = carry < 0
        loose = jnp.any(jnp.all(pos[:, None, :] == s["block_pos"][None, :, :], axis=2)
                        & (s["block_status"][None, :] == -1), axis=1)
        crank = jnp.all(pos == s["rig_pos"][0], axis=1)
        brake = jnp.all(pos == s["rig_pos"][1], axis=1)
        near_rig = jnp.any(jnp.sum(jnp.abs(pos[:, None, :] - s["rig_pos"][None, :, :]), axis=2) <= 1, axis=1)
        owner = jnp.clip(s["block_owner"][held], 0, 2)
        depot = jnp.all(pos == s["depot_pos"][owner], axis=1)
        mask = jnp.zeros((3, 12), jnp.bool_).at[:, :5].set(move)
        mask = mask.at[:, 5].set(free & loose).at[:, 6].set(~free)
        mask = mask.at[:, 7].set(crank & free & (jnp.arange(3) == 0))
        mask = mask.at[:, 8].set(brake & free & (jnp.arange(3) == 1))
        mask = mask.at[:, 9].set(brake & ((~free & (s["block_kind"][held] == 1)) | (free & jnp.any(s["block_status"] == -2))))
        mask = mask.at[:, 10].set(~free & depot & (s["block_kind"][held] <= 1))
        mask = mask.at[:, 11].set(~free & near_rig & (s["block_kind"][held] == 2) & (s["rope"][0] < 8))
        return jnp.where(s["done"], jnp.arange(12)[None, :] == 0, mask)

    def observe(self, s):
        v = jnp.concatenate([s[k].astype(jnp.float32).reshape(-1) for k in self.obs_keys])
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(v[None, :], (3, 183))), axis=1)

    def advance(self, payload):
        key, old, actions = payload
        s = dict(old)
        raw = jnp.clip(actions, 0, 11)
        actions = jnp.where((actions >= 0) & (actions < 12) & self.legal_actions(s)[jnp.arange(3), raw], raw, 0)
        candidates, allowed = self.movement(s)
        mi = jnp.where(actions <= 4, actions, 0)
        p = candidates[jnp.arange(3), mi]
        rank = jnp.mod(jnp.arange(3) - s["t"], 3)
        clash = jnp.all(p[:, None, :] == p[None, :, :], axis=2)
        lose = jnp.any(clash & (rank[None, :] < rank[:, None]), axis=1)
        s["agent_pos"] = jnp.where(lose[:, None], s["agent_pos"], p)
        s["delivered_now"] = jnp.zeros(3, jnp.int32)
        s["sale_now"] = jnp.zeros(3, jnp.int32)
        s["repaired"] = jnp.array(False)
        for k in range(3):
            i = jnp.mod(s["t"] + k, 3)
            status = s["block_status"]
            has = jnp.any(status == i)
            held = jnp.argmax(status == i)
            loose = (status == -1) & jnp.all(s["block_pos"] == s["agent_pos"][i], axis=1)
            pick = jnp.argmax(loose)
            take = (actions[i] == 5) & ~has & jnp.any(loose)
            status = status.at[pick].set(jnp.where(take, i, status[pick]))
            drop = (actions[i] == 6) & has
            status = status.at[held].set(jnp.where(drop, -1, status[held]))
            s["block_pos"] = s["block_pos"].at[held].set(jnp.where(drop, s["agent_pos"][i], s["block_pos"][held]))
            install = (actions[i] == 9) & has & (s["block_kind"][held] == 1)
            installed = status == -2
            take_weight = (actions[i] == 9) & ~has & jnp.any(installed)
            weight = jnp.argmax(installed)
            status = status.at[weight].set(jnp.where(take_weight, i, status[weight]))
            status = status.at[held].set(jnp.where(install, -2, status[held]))
            s["block_pos"] = s["block_pos"].at[held].set(jnp.where(install, s["rig_pos"][1], s["block_pos"][held]))
            deliver = (actions[i] == 10) & has
            ore = deliver & (s["block_kind"][held] == 0)
            sale = deliver & (s["block_kind"][held] == 1)
            owner = jnp.clip(s["block_owner"][held], 0, 2)
            s["delivered_now"] = s["delivered_now"].at[owner].add(ore.astype(jnp.int32))
            s["sale_now"] = s["sale_now"].at[owner].add(sale.astype(jnp.int32))
            repair = (actions[i] == 11) & has
            status = status.at[held].set(jnp.where(deliver, -3, jnp.where(repair, -4, status[held])))
            s["rope"] = jnp.minimum(8, s["rope"] + 3 * repair.astype(jnp.int32))
            s["repaired"] = s["repaired"] | repair
            s["block_status"] = status
        s = self.refresh(s)
        platform = s["lift_pos"][0]
        rider = jnp.all(s["agent_pos"] == platform, axis=1)
        aboard = ((s["block_status"] == -1) & jnp.all(s["block_pos"] == platform, axis=1)) | ((s["block_status"] >= 0) & rider[jnp.clip(s["block_status"], 0, 2)])
        load = jnp.sum(rider.astype(jnp.int32)) + jnp.sum(jnp.where(aboard, s["block_mass"], 0))
        imbalance = jnp.abs(load - s["countermass"][0])
        destination = 6 - platform[1]
        occupied = jnp.any(jnp.all(s["agent_pos"] == jnp.array([4, destination]), axis=1) & ~rider)
        docked = (platform[1] == 1) | (platform[1] == 5)
        launch = docked & (actions[0] == 7) & (actions[1] == 8) & (s["countermass"][0] > 0) & (imbalance <= 2) & (s["rope"][0] >= 1 + imbalance) & ~occupied
        s["lift_target"] = jnp.where(launch, destination, s["lift_target"]).astype(jnp.int32)
        dy = jnp.sign(s["lift_target"][0] - platform[1]).astype(jnp.int32)
        shift = jnp.array([0, dy], jnp.int32)
        s["agent_pos"] = s["agent_pos"] + rider[:, None].astype(jnp.int32) * shift
        s["block_pos"] = s["block_pos"] + aboard[:, None].astype(jnp.int32) * shift
        s["lift_pos"] = s["lift_pos"] + shift
        s["rope"] = s["rope"] - launch.astype(jnp.int32) * (1 + imbalance)
        s["hoisted"] = dy != 0
        s["delivered"] = s["delivered"] + s["delivered_now"]
        s["t"] = s["t"] + 1
        success = jnp.sum(s["delivered"]) == 6
        s["terminated"] = success
        s["truncated"] = (s["t"] >= self.max_steps) & ~success
        s["done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = -0.01 + 0.2 * jnp.sum(s["delivered_now"]) + s["delivered_now"].astype(jnp.float32) + 0.5 * s["sale_now"].astype(jnp.float32) + 2.0 * success
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full(3, s["done"], jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def absorb(self, payload):
        key, s, actions = payload
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
