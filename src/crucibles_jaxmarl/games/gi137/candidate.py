"""Mossline Pursuit: original two-team scent and extraction game; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 288
    num_actions = 11
    max_steps = 64

    def refresh(self, s):
        n = dict(s)
        ids = jnp.arange(4)
        teams = ids // 2
        same_team = teams[:, None] == teams[None, :]
        distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        radius = jnp.array([2, 3, 3, 3], jnp.int32) + 3 * s["scan"].astype(jnp.int32)
        terrain = s["tilemap"][s["agent_pos"][:, 1], s["agent_pos"][:, 0]]
        concealed = (terrain == 2) | s["hidden"]
        detectable = (distance <= radius[:, None]) & (~concealed[None, :] | (distance <= 1) | s["scan"][:, None])
        n["agent_visibility"] = (same_team | detectable) & s["agent_active"][None, :]
        n["kit_pos"] = s["agent_pos"]
        n["kit_visibility"] = same_team & s["agent_active"][None, :]
        n["kit_active"] = s["agent_active"]
        nose = jnp.array([3, 1, 2, 2], jnp.int32)
        trail_distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["trail_pos"][None, :, :]), axis=2)
        n["trail_visibility"] = (trail_distance <= nose[:, None]) & s["agent_active"][:, None]
        n["trail_active"] = s["scent"] > 0
        return n

    def reset(self, key):
        tilemap = jnp.zeros((9, 9), jnp.int32).at[2:7, 4].set(1).at[4, 4].set(0)
        tilemap = tilemap.at[jnp.array([2, 6, 2, 6]), jnp.array([2, 2, 6, 6])].set(2)
        tilemap = tilemap.at[jnp.array([1, 7]), 8].set(3)
        positions = jnp.array([[5, 3], [5, 5], [1, 1], [1, 7]], jnp.int32)
        trail_pos = jnp.stack((jnp.arange(81) % 9, jnp.arange(81) // 9), axis=1).astype(jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tilemap": tilemap,
            "agent_pos": positions,
            "agent_active": jnp.ones((4,), jnp.bool_), "agent_visibility": jnp.zeros((4, 4), jnp.bool_),
            "role": jnp.arange(4, dtype=jnp.int32),
            "status": jnp.zeros((4,), jnp.int32),
            "hidden": jnp.zeros((4,), jnp.bool_), "scan": jnp.zeros((4,), jnp.bool_),
            "charge": jnp.array([8, 8, 6, 6], jnp.int32),
            "kit_pos": positions,
            "kit_active": jnp.ones((4,), jnp.bool_), "kit_visibility": jnp.zeros((4, 4), jnp.bool_),
            "trail_pos": trail_pos,
            "trail_active": jnp.zeros((81,), jnp.bool_), "trail_visibility": jnp.zeros((4, 81), jnp.bool_),
            "scent": jnp.zeros((81,), jnp.int32),
            "exit_pos": jnp.array([[8, 1], [8, 7]], jnp.int32),
            "exit_active": jnp.ones((2,), jnp.bool_), "exit_visibility": jnp.ones((4, 2), jnp.bool_),
            "captured_now": jnp.zeros((2,), jnp.bool_), "escaped_now": jnp.zeros((2,), jnp.bool_),
            "resolved_count": jnp.array(0, jnp.int32),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(4)
        active = s["agent_active"] & ~s["done"]
        mask = jnp.zeros((4, 11), jnp.bool_).at[:, 0].set(True)
        mask = mask.at[:, 1:5].set(jnp.broadcast_to(active[:, None], (4, 4)))
        one = active & (s["charge"] >= 1)
        mask = mask.at[:, 5:7].set(jnp.broadcast_to(one[:, None], (4, 2)))
        # Tag legal independently of hidden opponent presence; no mask information leak.
        mask = mask.at[:, 7].set(active & jnp.where(ids < 2, s["charge"] >= 1, s["charge"] >= 2))
        return mask.at[:, 8:11].set(jnp.broadcast_to((active & (ids >= 2) & (s["charge"] >= 2))[:, None], (4, 3)))

    def raw_step(self, key, s, actions):
        ids = jnp.arange(4)
        teams = ids // 2
        clipped = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(s)[ids, clipped]
        a = jnp.where(valid, clipped, 0)
        n = dict(s)
        nt = s["t"] + jnp.array(1, jnp.int32)
        n["t"] = nt
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        moving = (a >= 1) & (a <= 4) & s["agent_active"]
        proposed = s["agent_pos"] + delta[jnp.where(moving, a, 0)]
        inside = jnp.all((proposed >= 0) & (proposed <= 8), axis=1)
        safe = jnp.clip(proposed, 0, 8)
        floor = s["tilemap"][safe[:, 1], safe[:, 0]] != 1
        same_team = teams[:, None] == teams[None, :]
        occupied = jnp.any(jnp.all(proposed[:, None, :] == s["agent_pos"][None, :, :], axis=2) &
                           same_team & s["agent_active"][None, :] & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2) &
                            same_team & s["agent_active"][None, :], axis=1) > 1
        p = jnp.where((moving & inside & floor & ~occupied & ~contested)[:, None], proposed, s["agent_pos"])
        n["agent_pos"] = p
        scan = (a == 5) & (ids < 2) & s["agent_active"]
        hide = (a == 5) & (ids >= 2) & s["agent_active"]
        erase = (a == 6) & s["agent_active"]
        tag = (a == 7) & (ids < 2) & s["agent_active"]
        decoy = (a >= 7) & (a <= 10) & (ids >= 2) & s["agent_active"]
        spent = (scan | hide | erase | tag).astype(jnp.int32) + 2 * decoy.astype(jnp.int32)
        n["charge"] = s["charge"] - spent
        n["scan"], n["hidden"] = scan, hide
        # Decay, then all erasures, then fresh genuine scent and false scent.
        scent = jnp.maximum(s["scent"] - 1, 0)
        distances = jnp.sum(jnp.abs(p[:, None, :] - s["trail_pos"][None, :, :]), axis=2)
        scrubbed = jnp.any(erase[:, None] & (distances <= 1), axis=0)
        scent = jnp.where(scrubbed, 0, scent)
        for i in range(2, 4):
            emit = s["agent_active"][i] & ~hide[i] & ~erase[i]
            old_cell = 9 * s["agent_pos"][i, 1] + s["agent_pos"][i, 0]
            scent = scent.at[old_cell].set(jnp.maximum(scent[old_cell], jnp.where(emit, 4, 0)))
            direction = jnp.clip(a[i] - 6, 1, 4)
            target = jnp.clip(p[i] + 2 * delta[direction], 0, 8)
            legal_target = s["tilemap"][target[1], target[0]] != 1
            target = jnp.where(legal_target, target, p[i])
            cell = 9 * target[1] + target[0]
            scent = scent.at[cell].set(jnp.maximum(scent[cell], jnp.where(decoy[i], 6, 0)))
        n["scent"] = scent.astype(jnp.int32)
        # Blind adjacent sweep can catch a hidden runner. Capture precedes extraction.
        distance = jnp.sum(jnp.abs(p[:2, None, :] - p[None, 2:, :]), axis=2)
        captured = jnp.any(tag[:2, None] & (distance <= 1), axis=0) & s["agent_active"][2:]
        at_exit = jnp.any(jnp.all(p[2:, None, :] == s["exit_pos"][None, :, :], axis=2), axis=1)
        escaped = at_exit & s["agent_active"][2:] & ~captured
        status = s["status"].at[2:].set(jnp.where(captured, 1, jnp.where(escaped, 2, s["status"][2:])))
        n["status"] = status.astype(jnp.int32)
        n["agent_active"] = (ids < 2) | (status == 0)
        n["captured_now"], n["escaped_now"] = captured, escaped
        n["resolved_count"] = jnp.sum(status[2:] != 0, dtype=jnp.int32)
        balance = jnp.sum(captured).astype(jnp.float32) - jnp.sum(escaped).astype(jnp.float32)
        reward = jnp.array([balance, balance, -balance, -balance], jnp.float32)
        natural = jnp.all(status[2:] != 0)
        n["terminated"] = natural
        n["truncated"] = (nt >= self.max_steps) & ~natural
        n["done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        n = jax.tree.map(lambda old, new: jnp.where(s["done"], old, new), s, n)
        reward = jnp.where(s["done"], jnp.zeros((4,), jnp.float32), reward)
        dones = jnp.full((4,), n["done"], jnp.bool_)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, reward, dones, info

    def observe(self, s):
        rows = []
        for i in range(4):
            seen = s["agent_visibility"][i]
            own_team = (jnp.arange(4) // 2) == (i // 2)
            trail_seen = s["trail_visibility"][i]
            row = jnp.concatenate((
                s["t"][None] / 64.0, jnp.eye(4, dtype=jnp.float32)[i],
                s["tilemap"].reshape(-1) / 3.0, seen.astype(jnp.float32),
                jnp.where(seen[:, None], s["agent_pos"] / 8.0, 0.0).reshape(-1),
                jnp.where(seen, s["role"] / 3.0, 0.0),
                (seen & s["hidden"]).astype(jnp.float32), (seen & s["scan"]).astype(jnp.float32),
                jnp.where(own_team, s["charge"] / 8.0, 0.0), s["status"] / 2.0,
                jnp.where(trail_seen, s["scent"] / 6.0, 0.0), trail_seen.astype(jnp.float32),
                jnp.zeros((8,), jnp.float32),
            ))
            rows = rows + [row]
        return jnp.stack(rows, axis=0).astype(jnp.float32)
