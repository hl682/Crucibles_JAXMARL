"""Relay Foundry: original four-role spatial production game; authored, unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 153
    num_actions = 13
    max_steps = 192

    def refresh(self, s):
        s = dict(s)
        ids = jnp.arange(4, dtype=jnp.int32)
        own = s["part_carrier"][None, :] == ids[:, None]
        s["carrying"] = jnp.where(jnp.any(own, axis=1), jnp.argmax(own, axis=1), -1).astype(jnp.int32)
        s["has_clamp"] = s["tool_carrier"][0] == ids
        s["part_pos"] = jnp.where((s["part_carrier"] >= 0)[:, None],
                                  s["agent_pos"][jnp.clip(s["part_carrier"], 0, 3)], s["part_pos"])
        s["tool_pos"] = jnp.where((s["tool_carrier"] >= 0)[:, None],
                                  s["agent_pos"][jnp.clip(s["tool_carrier"], 0, 3)], s["tool_pos"])
        s["part_active"] = s["part_stage"] < 4
        s["agent_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2) <= 3
        s["part_visible"] = (jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["part_pos"][None, :, :]), axis=2) <= 3) & s["part_active"][None, :]
        s["tool_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["tool_pos"][None, :, :]), axis=2) <= 3
        s["station_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["station_pos"][None, :, :]), axis=2) <= 3
        return s

    def reset(self, key):
        board = jnp.array([
            [1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,2,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,3,0,0,4,0,5,1],
            [1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1]], dtype=jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "map": board, "agent_pos": jnp.array([[2,2],[3,3],[7,3],[8,2]], jnp.int32),
            "agent_active": jnp.ones(4, jnp.bool_), "agent_visible": jnp.ones((4,4), jnp.bool_),
            "role": jnp.arange(4, dtype=jnp.int32), "carrying": jnp.full(4, -1, jnp.int32),
            "has_clamp": jnp.zeros(4, jnp.bool_),
            "part_pos": jnp.array([[2,1],[2,4],[7,1],[8,4]], jnp.int32),
            "part_active": jnp.ones(4, jnp.bool_), "part_visible": jnp.zeros((4,4), jnp.bool_),
            "part_carrier": jnp.full(4, -1, jnp.int32), "part_stage": jnp.zeros(4, jnp.int32),
            "tool_pos": jnp.array([[4,3]], jnp.int32), "tool_active": jnp.ones(1, jnp.bool_),
            "tool_visible": jnp.zeros((4,1), jnp.bool_), "tool_carrier": jnp.full(1, -1, jnp.int32),
            "tool_heat": jnp.zeros(1, jnp.int32),
            "station_pos": jnp.array([[3,1],[4,3],[7,3],[9,3]], jnp.int32),
            "station_active": jnp.ones(4, jnp.bool_), "station_visible": jnp.zeros((4,4), jnp.bool_),
            "station_type": jnp.arange(4, dtype=jnp.int32),
            "queue_head": jnp.array(0, jnp.int32), "stage_events": jnp.array(0, jnp.int32),
            "transfer_events": jnp.array(0, jnp.int32), "delivery_event": jnp.array(False)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(4, dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        target = jnp.clip(s["agent_pos"][:,None,:] + delta[None,:,:], jnp.array([0,0],jnp.int32), jnp.array([10,6],jnp.int32))
        walk = s["map"][target[:,:,1], target[:,:,0]] != 1
        mask = jnp.ones((4,13), jnp.bool_).at[:,:5].set(walk)
        mask = mask.at[:,9:13].set(ids[:,None] != ids[None,:])
        # Object operations are legal attempts; missing local prerequisites make them no-ops.
        return mask & (~s["episode_done"] | (jnp.arange(13)[None,:] == 0))

    def move(self, s, actions):
        ids = jnp.arange(4, dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        proposed = s["agent_pos"] + delta[jnp.clip(actions, 0, 4)] * ((actions >= 1) & (actions <= 4))[:,None]
        proposed = jnp.clip(proposed, jnp.array([0,0],jnp.int32), jnp.array([10,6],jnp.int32))
        terrain = s["map"][proposed[:,1], proposed[:,0]] != 1
        occupied = jnp.any(jnp.all(proposed[:,None,:] == s["agent_pos"][None,:,:], axis=2) & (ids[:,None] != ids[None,:]), axis=1)
        possible = (actions >= 1) & (actions <= 4) & terrain & ~occupied
        rank = jnp.mod(ids - s["t"], 4)
        same = jnp.all(proposed[:,None,:] == proposed[None,:,:], axis=2)
        loses = jnp.any(same & possible[None,:] & (rank[None,:] < rank[:,None]), axis=1)
        return jnp.where((possible & ~loses)[:,None], proposed, s["agent_pos"])

    def raw_step(self, key, state, actions):
        mask = self.legal_actions(state)
        safe = jnp.clip(actions, 0, 12)
        actions = jnp.where((actions >= 0) & (actions < 13) & mask[jnp.arange(4),safe], safe, 0)
        s = dict(state)
        s["agent_pos"] = self.move(state, actions)
        s = self.refresh(s)
        s["stage_events"] = jnp.array(0, jnp.int32)
        s["transfer_events"] = jnp.array(0, jnp.int32)
        s["delivery_event"] = jnp.array(False)
        s["tool_heat"] = jnp.maximum(state["tool_heat"] - 1, 0)
        # Priority rotates with the clock. Start-of-phase ownership prevents transfer chains.
        original = s["part_carrier"]
        original_tool = s["tool_carrier"][0]
        for offset in range(4):
            i = jnp.mod(state["t"] + offset, 4)
            target = jnp.clip(actions[i] - 9, 0, 3)
            adjacent = jnp.sum(jnp.abs(s["agent_pos"][i] - s["agent_pos"][target])) == 1
            pass_ok = (actions[i] >= 9) & (actions[i] <= 12) & (target != i) & adjacent
            held = original == i
            p = jnp.argmax(held)
            empty = ~jnp.any(s["part_carrier"] == target)
            part_pass = pass_ok & jnp.any(held) & empty & (s["part_carrier"][p] == i)
            s["part_carrier"] = s["part_carrier"].at[p].set(jnp.where(part_pass, target, s["part_carrier"][p]))
            tool_pass = pass_ok & (original_tool == i) & (s["tool_carrier"][0] == i)
            s["tool_carrier"] = jnp.where(tool_pass, jnp.full(1,target,jnp.int32), s["tool_carrier"])
            s["transfer_events"] = s["transfer_events"] + part_pass.astype(jnp.int32) + tool_pass.astype(jnp.int32)
        s = self.refresh(s)
        # Pickup/drop and clamp checkout are separate from processing and passing.
        for offset in range(4):
            i = jnp.mod(state["t"] + offset, 4)
            held = s["part_carrier"] == i
            ground = jnp.all(s["part_pos"] == s["agent_pos"][i], axis=1) & (s["part_carrier"] < 0) & s["part_active"]
            p = jnp.where(jnp.any(held), jnp.argmax(held), jnp.argmax(ground))
            manipulate = (actions[i] == 5) & (jnp.any(held) | jnp.any(ground))
            owner = jnp.where(jnp.any(held), -1, i)
            s["part_carrier"] = s["part_carrier"].at[p].set(jnp.where(manipulate, owner, s["part_carrier"][p]))
            near_tool = jnp.all(s["agent_pos"][i] == s["tool_pos"][0])
            checkout = (actions[i] == 8) & ((s["tool_carrier"][0] == i) | ((s["tool_carrier"][0] < 0) & near_tool))
            new_owner = jnp.where(s["tool_carrier"][0] == i, -1, i)
            s["tool_carrier"] = jnp.where(checkout, jnp.full(1,new_owner,jnp.int32), s["tool_carrier"])
        s = self.refresh(s)
        # Irreversible complementary operations. Both powered stages need a separate assistant.
        for role in range(3):
            held = s["part_carrier"] == role
            p = jnp.argmax(held)
            at_station = jnp.all(s["agent_pos"][role] == s["station_pos"][role])
            assisting = (actions == 7) & (jnp.arange(4) != role) & (jnp.sum(jnp.abs(s["agent_pos"] - s["station_pos"][role]), axis=1) == 1)
            powered = (role == 0) | ((s["tool_carrier"][0] == role) & (state["tool_heat"][0] == 0) & jnp.any(assisting))
            work = (actions[role] == 6) & jnp.any(held) & (s["part_stage"][p] == role) & at_station & powered
            s["part_stage"] = s["part_stage"].at[p].add(work.astype(jnp.int32))
            s["stage_events"] = s["stage_events"] + work.astype(jnp.int32)
            s["tool_heat"] = jnp.where(work & (role > 0), jnp.full(1,3,jnp.int32), s["tool_heat"])
        # A single public FIFO shipping order prevents independent per-agent races.
        head = jnp.clip(s["queue_head"], 0, 3)
        holder = jnp.clip(s["part_carrier"][head], 0, 3)
        delivered = (s["queue_head"] < 4) & (s["part_stage"][head] == 3) & (s["part_carrier"][head] >= 0) & (actions[holder] == 6) & jnp.all(s["agent_pos"][holder] == s["station_pos"][3])
        s["part_stage"] = s["part_stage"].at[head].set(jnp.where(delivered, 4, s["part_stage"][head]))
        s["part_carrier"] = s["part_carrier"].at[head].set(jnp.where(delivered, -1, s["part_carrier"][head]))
        s["part_pos"] = s["part_pos"].at[head].set(jnp.where(delivered, s["station_pos"][3], s["part_pos"][head]))
        s["queue_head"] = s["queue_head"] + delivered.astype(jnp.int32)
        s["delivery_event"] = delivered
        s["t"] = state["t"] + jnp.array(1, jnp.int32)
        s["terminated"] = s["queue_head"] == 4
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = jnp.full(4, -0.002 + 0.1 * s["stage_events"] + delivered.astype(jnp.float32), jnp.float32)
        s = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, s)
        reward = jnp.where(state["episode_done"], jnp.zeros(4,jnp.float32), reward)
        return self.observe(s), s, reward, jnp.full(4,s["episode_done"],jnp.bool_), {"done_all": s["episode_done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def observe(self, s):
        core = jnp.stack([s["role"], s["agent_pos"][:,0], s["agent_pos"][:,1], s["carrying"], s["has_clamp"].astype(jnp.int32), jnp.full(4,s["t"],jnp.int32)], axis=1).astype(jnp.float32)
        public = jnp.broadcast_to(jnp.array([s["queue_head"],s["stage_events"],s["transfer_events"],s["delivery_event"],s["episode_done"]], jnp.float32), (4,5))
        av = s["agent_visible"]
        agent_data = jnp.concatenate([s["agent_pos"],s["role"][:,None],s["carrying"][:,None],s["has_clamp"][:,None]], axis=1)
        agents = jnp.concatenate([jnp.where(av[:,:,None],agent_data[None,:,:],-1),av[:,:,None]], axis=2).reshape(4,24)
        pv = s["part_visible"]
        part_data = jnp.concatenate([s["part_pos"],s["part_stage"][:,None],s["part_carrier"][:,None]], axis=1)
        parts = jnp.concatenate([jnp.where(pv[:,:,None],part_data[None,:,:],-1),pv[:,:,None]], axis=2).reshape(4,20)
        tv = s["tool_visible"]
        tool_data = jnp.concatenate([s["tool_pos"],s["tool_carrier"][:,None],s["tool_heat"][:,None]], axis=1)
        tool = jnp.concatenate([jnp.where(tv[:,:,None],tool_data[None,:,:],-1),tv[:,:,None]], axis=2).reshape(4,5)
        sv = s["station_visible"]
        station_data = jnp.concatenate([s["station_pos"],s["station_type"][:,None]], axis=1)
        stations = jnp.concatenate([jnp.where(sv[:,:,None],station_data[None,:,:],-1),sv[:,:,None]], axis=2).reshape(4,16)
        board = jnp.broadcast_to(s["map"].reshape(1,77),(4,77))
        return jnp.concatenate([core,public,agents,parts,tool,stations,board], axis=1).astype(jnp.float32)
