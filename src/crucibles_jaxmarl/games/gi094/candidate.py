"""Original Three-Hand Kiln kitchen. Authored, never imported or executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 143
    num_actions = 10
    max_steps = 96

    def refresh(self, s):
        s = dict(s)
        held = (s["item_holder"][None, :] == jnp.arange(3)[:, None]) & s["item_active"][None, :]
        s["carry"] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        s["item_pos"] = jnp.where((s["item_holder"] >= 0)[:, None],
                                  s["agent_pos"][jnp.maximum(s["item_holder"], 0)], s["item_pos"])
        s["tool_pos"] = jnp.where((s["tool_carrier"] >= 0)[:, None],
                                  s["agent_pos"][jnp.maximum(s["tool_carrier"], 0)], s["tool_pos"])
        s["item_visible"] = jnp.broadcast_to(s["item_active"], (3, 4))
        return s

    def reset(self, key):
        terrain = jnp.array([[1,1,1,1,1,1,1,1,1],
                             [1,0,0,0,1,0,0,0,1],
                             [1,0,0,0,1,0,0,0,1],
                             [1,0,0,0,0,0,0,0,1],
                             [1,0,0,0,1,0,0,0,1],
                             [1,0,0,0,1,0,0,0,1],
                             [1,1,1,1,1,1,1,1,1]], dtype=jnp.int32)
        s = {"t": jnp.array(0, dtype=jnp.int32),
             "episode_done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
             "map": terrain, "agent_pos": jnp.array([[2,2],[3,3],[7,4]], dtype=jnp.int32),
             "agent_active": jnp.ones(3, dtype=jnp.bool_), "agent_visible": jnp.ones((3,3), dtype=jnp.bool_),
             "carry": jnp.full(3, -1, dtype=jnp.int32),
             "item_pos": jnp.array([[1,1],[1,2],[1,3],[2,1]], dtype=jnp.int32),
             "item_active": jnp.ones(4, dtype=jnp.bool_), "item_visible": jnp.ones((3,4), dtype=jnp.bool_),
             "item_holder": jnp.full(4, -1, dtype=jnp.int32), "item_stage": jnp.zeros(4, dtype=jnp.int32),
             "item_flavor": jnp.array([0,1,1,0], dtype=jnp.int32), "item_cut": jnp.zeros(4, dtype=jnp.int32),
             "item_heat": jnp.zeros(4, dtype=jnp.int32), "item_age": jnp.zeros(4, dtype=jnp.int32),
             "item_started": jnp.zeros(4, dtype=jnp.bool_),
             "station_pos": jnp.array([[2,2],[6,2],[7,5]], dtype=jnp.int32),
             "station_kind": jnp.array([0,1,2], dtype=jnp.int32),
             "station_active": jnp.ones(3, dtype=jnp.bool_), "station_visible": jnp.ones((3,3), dtype=jnp.bool_),
             "tool_pos": jnp.array([[2,2]], dtype=jnp.int32), "tool_carrier": jnp.full(1,-1,dtype=jnp.int32),
             "tool_active": jnp.ones(1,dtype=jnp.bool_), "tool_visible": jnp.ones((3,1),dtype=jnp.bool_),
             "orders": jnp.array([0,1,0,1], dtype=jnp.int32), "order_index": jnp.array(0,dtype=jnp.int32),
             "spoiled": jnp.array(0,dtype=jnp.int32), "delivered_now": jnp.array(False),
             "spoiled_now": jnp.array(False), "handed_now": jnp.array(False),
             "blocked": jnp.zeros(3,dtype=jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        flat = jnp.concatenate([s["agent_pos"].reshape(-1), s["carry"], s["tool_pos"].reshape(-1),
                                s["tool_carrier"], s["item_pos"].reshape(-1), s["item_active"],
                                s["item_holder"], s["item_stage"], s["item_flavor"], s["item_cut"],
                                s["item_heat"], s["item_age"], s["item_started"],
                                s["station_pos"].reshape(-1), s["station_kind"], s["map"].reshape(-1),
                                s["orders"], jnp.stack([s["order_index"], s["t"], s["spoiled"],
                                s["delivered_now"], s["spoiled_now"], s["handed_now"]]),
                                s["blocked"], jnp.stack([s["episode_done"],s["terminated"],s["truncated"]])]).astype(jnp.float32)
        return jnp.concatenate([jnp.broadcast_to(flat, (3,140)), jnp.eye(3,dtype=jnp.float32)], axis=1)

    def move_agents(self, state, actions):
        offsets = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        step = offsets[jnp.where((actions >= 1) & (actions <= 4), actions, 0)]
        p = state["agent_pos"]
        q = jnp.clip(p + step, jnp.array([0, 0], dtype=jnp.int32), jnp.array([8, 6], dtype=jnp.int32))
        terrain_ok = state["map"][q[:, 1], q[:, 0]] == 0
        occupied = jnp.any(jnp.all(q[:, None, :] == p[None, :, :], axis=2) &
                           ~jnp.eye(self.num_agents, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        moving = jnp.any(step != 0, axis=1)
        blocked = moving & (~terrain_ok | occupied | contested)
        return jnp.where(blocked[:, None], p, q), blocked

    def advance(self, key, state, actions):
        a = jnp.where((actions >= 0) & (actions < 10), actions, 0).astype(jnp.int32)
        s = dict(state)
        p, blocked = self.move_agents(state, a)
        s["agent_pos"] = p
        s = self.refresh(s)
        holder, active, stage = s["item_holder"], s["item_active"], s["item_stage"]
        ip, cut, started = s["item_pos"], s["item_cut"], s["item_started"]
        tool, toolp = s["tool_carrier"], s["tool_pos"]
        delivered = jnp.array(0, dtype=jnp.int32)
        handoffs = jnp.array(0, dtype=jnp.int32)
        order = s["order_index"]
        touched = jnp.zeros(4, dtype=jnp.bool_)
        # Rotating priority resolves simultaneous operations on the same physical object.
        for offset in range(3):
            i = jnp.mod(s["t"] + offset, 3)
            mine = (holder == i) & active
            j = jnp.argmax(mine)
            carrying = jnp.any(mine)
            available = active & (holder < 0) & jnp.all(ip == p[i], axis=1)
            k = jnp.argmax(available)
            pickup = (a[i] == 5) & ~carrying & jnp.any(available)
            holder = holder.at[k].set(jnp.where(pickup, i, holder[k]))
            started = started.at[k].set(started[k] | pickup)
            mine = (holder == i) & active
            j = jnp.argmax(mine)
            carrying = jnp.any(mine)
            occupied = jnp.any(active & (holder < 0) & jnp.all(ip == p[i], axis=1))
            drop = (a[i] == 6) & carrying & ~occupied
            holder = holder.at[j].set(jnp.where(drop, -1, holder[j]))
            ip = ip.at[j].set(jnp.where(drop, p[i], ip[j]))
            tool_pick = (a[i] == 9) & (tool[0] < 0) & jnp.all(toolp[0] == p[i])
            tool_drop = (a[i] == 9) & (tool[0] == i)
            tool = tool.at[0].set(jnp.where(tool_pick, i, jnp.where(tool_drop, -1, tool[0])))
            toolp = toolp.at[0].set(jnp.where(tool_drop | tool_pick, p[i], toolp[0]))
            chopping = ((a[i] == 7) & carrying & (holder[j] == i) & (stage[j] == 0) &
                        (tool[0] == i) & jnp.all(p[i] == s["station_pos"][0]))
            cut = cut.at[j].set(jnp.where(chopping, jnp.minimum(cut[j] + 1, 2), cut[j]))
            stage = stage.at[j].set(jnp.where(chopping & (cut[j] == 2), 1, stage[j]))
            serving = ((a[i] == 7) & carrying & (holder[j] == i) & (stage[j] == 2) &
                       jnp.all(p[i] == s["station_pos"][2]) &
                       (s["item_flavor"][j] == s["orders"][jnp.minimum(order, 3)]))
            active = active.at[j].set(active[j] & ~serving)
            holder = holder.at[j].set(jnp.where(serving, -1, holder[j]))
            order = order + serving.astype(jnp.int32)
            delivered = delivered + serving.astype(jnp.int32)
            # Both partners must explicitly choose HANDOFF, and the receiver must be empty.
            empty = ~jnp.any((holder[None,:] == jnp.arange(3)[:,None]) & active[None,:], axis=1)
            receivers = empty & (a == 8) & (jnp.sum(jnp.abs(p - p[i]),axis=1) == 1)
            receiver = jnp.argmax(receivers)
            giving = (a[i] == 8) & carrying & active[j] & (holder[j] == i) & ~touched[j] & jnp.any(receivers)
            holder = holder.at[j].set(jnp.where(giving, receiver, holder[j]))
            ip = ip.at[j].set(jnp.where(giving, p[receiver], ip[j]))
            touched = touched.at[j].set(touched[j] | giving)
            handoffs = handoffs + giving.astype(jnp.int32)
        age = jnp.minimum(s["item_age"] + (active & started).astype(jnp.int32), 50)
        in_oven = active & (holder < 0) & jnp.all(ip == s["station_pos"][1], axis=1) & (stage >= 1)
        heat = jnp.where(in_oven, jnp.minimum(s["item_heat"] + 1, 8), s["item_heat"])
        stage = jnp.where((stage == 1) & (heat >= 4), 2, stage)
        ruined = active & ((age >= 50) | (in_oven & (heat >= 8)))
        active = active & ~ruined
        stage = jnp.where(ruined, 3, stage)
        holder = jnp.where(ruined, -1, holder)
        lost = jnp.sum(ruined.astype(jnp.int32), dtype=jnp.int32)
        s["item_holder"], s["item_active"], s["item_stage"] = holder, active, stage
        s["item_pos"], s["item_cut"], s["item_started"] = ip, cut, started
        s["tool_carrier"], s["tool_pos"] = tool, toolp
        s["item_age"], s["item_heat"] = age, heat
        s["order_index"], s["spoiled"] = order, s["spoiled"] + lost
        s["delivered_now"], s["spoiled_now"], s["handed_now"] = delivered > 0, lost > 0, handoffs > 0
        s["blocked"] = blocked
        s["t"] = s["t"] + 1
        s["terminated"] = ~jnp.any(active)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = delivered.astype(jnp.float32) - lost.astype(jnp.float32) * 0.5
        rewards = jnp.full((3,), reward, dtype=jnp.float32)
        dones = jnp.full((3,), s["episode_done"], dtype=jnp.bool_)
        info = {"done_all": s["episode_done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, rewards, dones, info

    def legal_actions(self, state):
        live = jnp.broadcast_to(~state["episode_done"], (self.num_agents, self.num_actions))
        return live.at[:, 0].set(True)

    def absorb(self, key, state, actions):
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        dones = jnp.ones((self.num_agents,), dtype=jnp.bool_)
        info = {"done_all": state["episode_done"], "terminated": state["terminated"],
                "truncated": state["truncated"]}
        return self.observe(state), state, rewards, dones, info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["episode_done"], self.absorb, self.advance, key, state, actions)
