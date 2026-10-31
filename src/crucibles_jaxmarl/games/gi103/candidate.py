"""Original Switchback Repair Crew kernel; no author-side numerical execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 110
    num_actions = 10
    max_steps = 96

    def distances(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=-1)

    def refresh(self, s):
        n = dict(s)
        owner = s["tool_carrier"]
        n["tool_pos"] = jnp.where((owner >= 0)[:, None],
                                  s["agent_pos"][jnp.clip(owner, 0, 2)], s["tool_pos"])
        n["agent_holding"] = jnp.max(jnp.where(
            owner[None, :] == jnp.arange(3)[:, None], jnp.arange(2)[None, :], -1), axis=1)
        n["device_done"] = s["device_progress"] >= s["device_need"]
        n["map"] = s["map"].at[3, 4].set(jnp.where(n["device_done"][0], 0, 2))
        return n

    def reset(self, key):
        # A fixed public map, fixed roles, and fixed initial tool locations.
        grid = jnp.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 2, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]], jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False), "map": grid,
            "agent_pos": jnp.array([[1, 2], [1, 4], [2, 3]], jnp.int32),
            "agent_active": jnp.ones(3, jnp.bool_), "agent_vis": jnp.ones((3, 3), jnp.bool_),
            "agent_role": jnp.array([0, 1, 2], jnp.int32), "agent_holding": jnp.full(3, -1, jnp.int32),
            "tool_pos": jnp.array([[2, 2], [2, 4]], jnp.int32),
            "tool_active": jnp.ones(2, jnp.bool_), "tool_vis": jnp.ones((3, 2), jnp.bool_),
            "tool_kind": jnp.array([0, 1], jnp.int32), "tool_carrier": jnp.full(2, -1, jnp.int32),
            "device_pos": jnp.array([[3, 3], [6, 2], [6, 4]], jnp.int32),
            "device_active": jnp.ones(3, jnp.bool_), "device_vis": jnp.ones((3, 3), jnp.bool_),
            "device_progress": jnp.zeros(3, jnp.int32), "device_need": jnp.array([3, 2, 3], jnp.int32),
            "device_done": jnp.zeros(3, jnp.bool_),
            "event_repair": jnp.zeros(3, jnp.bool_), "event_handoff": jnp.zeros(2, jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + delta[None, :, :]
        tile = s["map"][jnp.clip(p[:, :, 1], 0, 6), jnp.clip(p[:, :, 0], 0, 8)]
        moves = (tile == 0) & jnp.all((p >= jnp.array([0, 0])) & (p <= jnp.array([8, 6])), axis=-1)
        one = jnp.ones((3, 1), jnp.bool_)
        mask = jnp.concatenate((one, moves, one, one, one,
                                 (s["agent_holding"] >= 0)[:, None], one), axis=1)
        return mask & (~s["episode_done"] | (jnp.arange(10)[None, :] == 0))

    def move(self, s, a):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)]
        equal_dest = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=-1)
        old_occupied = jnp.all(proposed[:, None, :] == s["agent_pos"][None, :, :], axis=-1)
        blocked = jnp.any((equal_dest | old_occupied) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        return jnp.where(blocked[:, None], s["agent_pos"], proposed)

    def advance(self, s, a):
        n = dict(s)
        n["t"] = s["t"] + jnp.array(1, jnp.int32)
        n["agent_pos"] = self.move(s, a)
        n = self.refresh(n)
        # Toggle: carried tools drop; empty hands pick the nearest free tool within one tile.
        for i in range(3):
            held = n["tool_carrier"] == i
            has_tool = jnp.any(held)
            distance = jnp.sum(jnp.abs(n["tool_pos"] - n["agent_pos"][i]), axis=1)
            reachable = (n["tool_carrier"] < 0) & (distance <= 1)
            target = jnp.argmin(jnp.where(reachable, distance, 30))
            pickup = (a[i] == 5) & ~has_tool & jnp.any(reachable)
            drop = (a[i] == 5) & has_tool
            n["tool_carrier"] = jnp.where(drop & held, -1, n["tool_carrier"])
            n["tool_carrier"] = n["tool_carrier"].at[target].set(jnp.where(
                pickup, i, n["tool_carrier"][target]))
        n = self.refresh(n)
        n["event_handoff"] = jnp.zeros(2, jnp.bool_)
        # An offer requires a distinct adjacent empty-handed receiver choosing action 9.
        for k in range(2):
            owner = n["tool_carrier"][k]
            owner_index = jnp.clip(owner, 0, 2)
            distance = jnp.sum(jnp.abs(n["agent_pos"] - n["agent_pos"][owner_index]), axis=1)
            empty = ~jnp.any(n["tool_carrier"][None, :] == jnp.arange(3)[:, None], axis=1)
            ready = (a == 9) & empty & (distance <= 1) & (jnp.arange(3) != owner)
            receiver = jnp.argmin(jnp.where(ready, jnp.arange(3), 9))
            transfer = (owner >= 0) & (a[owner_index] == 8) & jnp.any(ready)
            n["tool_carrier"] = n["tool_carrier"].at[k].set(jnp.where(transfer, receiver, owner))
            n["event_handoff"] = n["event_handoff"].at[k].set(transfer)
        n = self.refresh(n)

        # Each worker addresses the nearest unfinished device within Manhattan distance one.
        d = self.distances(n["agent_pos"], n["device_pos"])
        can = (d <= 1) & ~s["device_done"][None, :]
        target = jnp.argmin(jnp.where(can, d, 30), axis=1)
        nearby = jnp.any(can, axis=1)
        works = (a == 6) & nearby
        assists = (a == 7) & nearby
        engineer = (n["agent_role"] == 0) & (n["agent_holding"] == 0)
        electrician = (n["agent_role"] == 1) & (n["agent_holding"] == 1)
        bridge_work = jnp.any(works & engineer & (target == 0))
        bridge_assist = jnp.any(assists & (target == 0))
        bridge = jnp.minimum(3, s["device_progress"][0]
                            + bridge_work.astype(jnp.int32) * (1 + bridge_assist.astype(jnp.int32)))
        pump_work = (jnp.any(works & electrician & (target == 1))
                     & jnp.any(assists & (target == 1)))
        # An incomplete pump loses one priming tick whenever the pair breaks timing.
        pump = jnp.where(s["device_done"][1], 2, jnp.where(pump_work,
            jnp.minimum(2, s["device_progress"][1] + 1), jnp.maximum(0, s["device_progress"][1] - 1)))
        radio_work = (s["device_done"][1] & jnp.any(works & engineer & (target == 2))
                      & jnp.any(works & electrician & (target == 2)))
        radio = jnp.minimum(3, s["device_progress"][2] + radio_work.astype(jnp.int32))
        n["device_progress"] = jnp.stack((bridge, pump, radio)).astype(jnp.int32)
        n = self.refresh(n)
        n["event_repair"] = n["device_done"] & ~s["device_done"]
        n["terminated"] = jnp.all(n["device_done"])
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["episode_done"] = n["terminated"] | n["truncated"]
        reward = jnp.full((3,), jnp.sum(n["event_repair"].astype(jnp.int32)), jnp.float32)
        return n, reward

    def observe(self, s):
        clock = jnp.array([s["t"], s["episode_done"], s["terminated"], s["truncated"]], jnp.int32)
        agents = jnp.concatenate((s["agent_pos"], s["agent_role"][:, None],
                                  s["agent_holding"][:, None]), axis=1).reshape(-1)
        tools = jnp.concatenate((s["tool_pos"], s["tool_kind"][:, None],
                                 s["tool_carrier"][:, None]), axis=1).reshape(-1)
        devices = jnp.concatenate((s["device_pos"], s["device_progress"][:, None],
            s["device_need"][:, None], s["device_done"][:, None].astype(jnp.int32)), axis=1).reshape(-1)
        shared = jnp.concatenate((clock, agents, tools, devices, s["map"].reshape(-1),
            s["event_repair"], s["event_handoff"])).astype(jnp.int32)
        obs = jnp.concatenate((jnp.eye(3, dtype=jnp.int32), jnp.broadcast_to(shared, (3, 107))), axis=1)
        return obs.astype(jnp.float32) / 128.0

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions, 0, self.num_actions - 1)
        allowed = self.legal_actions(state)[jnp.arange(3), clipped]
        valid = (actions >= 0) & (actions < self.num_actions) & allowed
        a = jnp.where(valid, clipped, 0).astype(jnp.int32)
        proposed, reward = self.advance(state, a)
        n = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), proposed, state)
        reward = jnp.where(state["episode_done"], jnp.zeros(3, jnp.float32), reward)
        dones = jnp.full((3,), n["episode_done"], jnp.bool_)
        info = {"done_all": n["episode_done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, reward, dones, info
