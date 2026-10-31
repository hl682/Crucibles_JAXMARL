"""Turnhouse Compact: original rotating-room logistics game; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 168
    num_actions = 10
    max_steps = 96

    def rotate_points(self, points, room_pos, turns):
        membership = (jnp.abs(points[:, None, 0] - room_pos[None, :, 0]) <= 1) & (jnp.abs(points[:, None, 1] - 3) <= 1)
        room = jnp.argmax(membership, axis=1)
        centre = room_pos[room]
        relative = points - centre
        cw = jnp.stack((-relative[:, 1], relative[:, 0]), axis=1)
        ccw = jnp.stack((relative[:, 1], -relative[:, 0]), axis=1)
        turn = turns[room]
        rotated = centre + jnp.where((turn == 1)[:, None], cw, ccw)
        use = jnp.any(membership, axis=1) & (turn != 0)
        return jnp.where(use[:, None], rotated, points).astype(jnp.int32)

    def refresh(self, s):
        n = dict(s)
        # Each room has two adjacent ports. Orientation 0 is North+East.
        ports = ((jnp.arange(4)[None, :] - s["orientation"][:, None]) % 4) < 2
        desired = jnp.array([ports[0, 0], ports[0, 2], ports[0, 1] & ports[1, 3],
                             ports[1, 0], ports[1, 2], ports[1, 1] & ports[2, 3],
                             ports[2, 0], ports[2, 2]], jnp.bool_)
        agent_here = jnp.any(jnp.all(s["gate_pos"][:, None, :] == s["agent_pos"][None, :, :], axis=2) & s["agent_active"][None, :], axis=1)
        cargo_here = jnp.any(jnp.all(s["gate_pos"][:, None, :] == s["cargo_pos"][None, :, :], axis=2) & ~s["delivered"][None, :], axis=1)
        opened = desired | agent_here | cargo_here
        n["gate_open"] = opened
        n["jammed"] = ~desired & (agent_here | cargo_here)
        tilemap = jnp.ones((7, 13), jnp.int32)
        for r in range(3):
            tilemap = tilemap.at[2:5, 1 + 4 * r:4 + 4 * r].set(0)
        tilemap = tilemap.at[s["gate_pos"][:, 1], s["gate_pos"][:, 0]].set(jnp.where(opened, 0, 1))
        n["tilemap"] = tilemap
        held = s["carrier"][None, :] == jnp.arange(3)[:, None]
        n["carrying"] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        n["cargo_active"] = ~s["delivered"]
        return n

    def reset(self, key):
        s = {
            "t": jnp.array(0, jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tilemap": jnp.ones((7, 13), jnp.int32),
            "agent_pos": jnp.array([[2, 3], [6, 3], [10, 3]], jnp.int32),
            "agent_active": jnp.ones((3,), jnp.bool_), "agent_visibility": jnp.ones((3, 3), jnp.bool_),
            "energy": jnp.full((3,), 4, jnp.int32),
            "carrying": jnp.full((3,), -1, jnp.int32),
            "escaped": jnp.zeros((3,), jnp.bool_),
            "command": jnp.zeros((3,), jnp.int32),
            "room_pos": jnp.array([[2, 3], [6, 3], [10, 3]], jnp.int32),
            "room_active": jnp.ones((3,), jnp.bool_), "room_visibility": jnp.ones((3, 3), jnp.bool_),
            "orientation": jnp.array([0, 2, 2], jnp.int32),
            "cargo_pos": jnp.array([[1, 2], [5, 4], [11, 2]], jnp.int32),
            "cargo_active": jnp.ones((3,), jnp.bool_), "cargo_visibility": jnp.ones((3, 3), jnp.bool_),
            "carrier": jnp.full((3,), -1, jnp.int32),
            "owner": jnp.arange(3, dtype=jnp.int32),
            "delivered": jnp.zeros((3,), jnp.bool_),
            "goal_pos": jnp.array([[6, 5], [10, 1], [2, 1]], jnp.int32),
            "goal_active": jnp.ones((3,), jnp.bool_), "goal_visibility": jnp.ones((3, 3), jnp.bool_),
            "gate_pos": jnp.array([[2, 1], [2, 5], [4, 3], [6, 1], [6, 5], [8, 3], [10, 1], [10, 5]], jnp.int32),
            "gate_active": jnp.ones((8,), jnp.bool_), "gate_visibility": jnp.ones((3, 8), jnp.bool_),
            "gate_open": jnp.zeros((8,), jnp.bool_), "jammed": jnp.zeros((8,), jnp.bool_),
            "rotated": jnp.zeros((3,), jnp.bool_),
            "picked": jnp.zeros((3,), jnp.bool_),
            "delivered_now": jnp.zeros((3,), jnp.bool_),
            "escaped_now": jnp.zeros((3,), jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        active = s["agent_active"] & ~s["done"]
        mask = jnp.zeros((3, 10), jnp.bool_).at[:, 0].set(True)
        mask = mask.at[:, 1:5].set(jnp.broadcast_to(active[:, None], (3, 4)))
        near = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["room_pos"][None, :, :]), axis=2) <= 1
        control = jnp.any(near, axis=1) & (s["energy"] > 0) & active
        mask = mask.at[:, 5:7].set(jnp.broadcast_to(control[:, None], (3, 2)))
        same = jnp.all(s["agent_pos"][:, None, :] == s["cargo_pos"][None, :, :], axis=2)
        can_pick = jnp.any(same & (s["carrier"][None, :] < 0) & ~s["delivered"][None, :], axis=1)
        h = jnp.maximum(s["carrying"], 0)
        goal = jnp.all(s["agent_pos"] == s["goal_pos"][h], axis=1)
        mask = mask.at[:, 7].set(active & jnp.where(s["carrying"] >= 0, goal, can_pick))
        mask = mask.at[:, 8].set(active & (s["carrying"] >= 0))
        bay = (s["agent_pos"][:, 1] == 1) | (s["agent_pos"][:, 1] == 5)
        return mask.at[:, 9].set(active & bay & (s["carrying"] < 0))

    def raw_step(self, key, s, actions):
        ids = jnp.arange(3)
        clipped = jnp.clip(actions, 0, 9)
        valid = (actions >= 0) & (actions < 10) & self.legal_actions(s)[ids, clipped]
        a = jnp.where(valid, clipped, 0)
        n = dict(s)
        nt = s["t"] + jnp.array(1, jnp.int32)
        n["t"] = nt
        command = jnp.where(a == 5, 1, jnp.where(a == 6, -1, 0)).astype(jnp.int32)
        near = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["room_pos"][None, :, :]), axis=2) <= 1
        clockwise = jnp.any(near & (command == 1)[:, None], axis=0)
        counter = jnp.any(near & (command == -1)[:, None], axis=0)
        turns = jnp.where(clockwise & ~counter, 1, jnp.where(counter & ~clockwise, -1, 0)).astype(jnp.int32)
        paid = (command != 0) & jnp.any(near & (turns != 0)[None, :], axis=1)
        n["energy"] = s["energy"] - paid.astype(jnp.int32)
        n["command"] = command
        n["orientation"] = (s["orientation"] + turns) % 4
        n["rotated"] = turns != 0
        rotated_agents = self.rotate_points(s["agent_pos"], s["room_pos"], turns)
        n["agent_pos"] = jnp.where(s["agent_active"][:, None], rotated_agents, s["agent_pos"])
        rotated_cargo = self.rotate_points(s["cargo_pos"], s["room_pos"], turns)
        n["cargo_pos"] = jnp.where((~s["delivered"])[:, None], rotated_cargo, s["cargo_pos"])
        n = self.refresh(n)
        base = n["agent_pos"]
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        moving = (a >= 1) & (a <= 4) & s["agent_active"]
        proposed = base + delta[jnp.where(moving, a, 0)]
        inside = jnp.all((proposed >= jnp.array([0, 0])) & (proposed <= jnp.array([12, 6])), axis=1)
        safe = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([12, 6]))
        floor = n["tilemap"][safe[:, 1], safe[:, 0]] == 0
        occupied = jnp.any(jnp.all(proposed[:, None, :] == base[None, :, :], axis=2) &
                           s["agent_active"][None, :] & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2) &
                            s["agent_active"][None, :], axis=1) > 1
        p = jnp.where((moving & inside & floor & ~occupied & ~contested)[:, None], proposed, base)
        cargo = jnp.where((s["carrier"] >= 0)[:, None], p[jnp.maximum(s["carrier"], 0)], n["cargo_pos"])
        carrier, delivered = s["carrier"], s["delivered"]
        picked, delivered_now = jnp.zeros((3,), jnp.bool_), jnp.zeros((3,), jnp.bool_)
        reward = jnp.zeros((3,), jnp.float32)
        for i in range(3):
            held = carrier == i
            h = jnp.argmax(held)
            holding = jnp.any(held)
            deliver = (a[i] == 7) & holding & jnp.all(p[i] == s["goal_pos"][h])
            drop = (a[i] == 8) & holding
            delivered = delivered.at[h].set(delivered[h] | deliver)
            delivered_now = delivered_now.at[h].set(delivered_now[h] | deliver)
            carrier = carrier.at[h].set(jnp.where(deliver | drop, -1, carrier[h]))
            reward = reward + deliver.astype(jnp.float32)
            reward = reward.at[i].add(deliver.astype(jnp.float32))
            reward = reward.at[s["owner"][h]].add(3.0 * deliver.astype(jnp.float32))
            available = jnp.all(cargo == p[i], axis=1) & (carrier < 0) & ~delivered
            q = jnp.argmax(available)
            pickup = (a[i] == 7) & ~holding & jnp.any(available)
            carrier = carrier.at[q].set(jnp.where(pickup, i, carrier[q]))
            picked = picked.at[q].set(picked[q] | pickup)
        n["agent_pos"], n["cargo_pos"], n["carrier"] = p, cargo, carrier
        n["delivered"], n["picked"], n["delivered_now"] = delivered, picked, delivered_now
        escape = (a == 9) & s["agent_active"]
        n["escaped_now"] = escape
        n["escaped"] = s["escaped"] | escape
        n["agent_active"] = ~n["escaped"]
        natural = jnp.all(n["escaped"])
        reward = reward + 4.0 * (natural & jnp.all(delivered)).astype(jnp.float32)
        n["terminated"] = natural
        n["truncated"] = (nt >= self.max_steps) & ~natural
        n["done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        n = jax.tree.map(lambda old, new: jnp.where(s["done"], old, new), s, n)
        reward = jnp.where(s["done"], jnp.zeros((3,), jnp.float32), reward)
        dones = jnp.full((3,), n["done"], jnp.bool_)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, reward, dones, info

    def observe(self, s):
        common = jnp.concatenate((
            s["t"][None] / 96.0, s["agent_pos"].reshape(-1) / 12.0,
            s["agent_active"].astype(jnp.float32), s["energy"] / 4.0, (s["carrying"] + 1) / 3.0,
            s["orientation"] / 3.0, (s["command"] + 1) / 2.0,
            s["cargo_pos"].reshape(-1) / 12.0, (s["carrier"] + 1) / 3.0,
            s["delivered"].astype(jnp.float32), s["goal_pos"].reshape(-1) / 12.0,
            s["gate_open"].astype(jnp.float32), s["jammed"].astype(jnp.float32),
            s["rotated"].astype(jnp.float32), s["picked"].astype(jnp.float32),
            s["delivered_now"].astype(jnp.float32), s["escaped_now"].astype(jnp.float32),
            s["tilemap"].reshape(-1).astype(jnp.float32),
        ))
        return jnp.concatenate((jnp.broadcast_to(common, (3, 159)), jnp.eye(3, dtype=jnp.float32),
                                jnp.zeros((3, 6))), axis=1).astype(jnp.float32)
