"""Chromatic Switchyard: original local-observation rail world, not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 557
    num_actions = 9
    max_steps = 512

    def directions(self):
        return jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)

    def visible(self, state, positions, active):
        distance = jnp.max(jnp.abs(state["agent_pos"][:, None, :] - positions[None, :, :]), axis=-1)
        return (distance <= 2) & active[None, :]

    def refresh(self, state):
        s = dict(state)
        sink_index = jnp.arange(4, dtype=jnp.int32)
        demand = s["sink_recipe"][sink_index, jnp.clip(s["sink_progress"], 0, 2)]
        s["sink_demand"] = jnp.where(s["sink_progress"] < 3, demand, -1).astype(jnp.int32)
        safe_nodes = jnp.clip(s["pulse_node"], 0, 63)
        s["pulse_pos"] = jnp.where(s["pulse_active"][:, None], s["node_pos"][safe_nodes], -1).astype(jnp.int32)
        s["blue_active"] = s["pulse_active"] & (s["pulse_colour"] == 0)
        s["amber_active"] = s["pulse_active"] & (s["pulse_colour"] == 1)
        s["source_blue"] = s["source_stock"][0:1]
        s["source_amber"] = s["source_stock"][1:2]
        s["source_timer"] = s["source_cooldown"].reshape(1)
        s["agent_visibility"] = self.visible(s, s["agent_pos"], s["agent_active"])
        s["node_visibility"] = self.visible(s, s["node_pos"], s["node_active"])
        s["source_visibility"] = self.visible(s, s["source_pos"], s["source_active"])
        s["switch_visibility"] = self.visible(s, s["switch_pos"], s["switch_active"])
        s["sink_visibility"] = self.visible(s, s["sink_pos"], s["sink_active"])
        s["pulse_visibility"] = self.visible(s, s["pulse_pos"], s["pulse_active"])
        s["blue_visibility"] = self.visible(s, s["pulse_pos"], s["blue_active"])
        s["amber_visibility"] = self.visible(s, s["pulse_pos"], s["amber_active"])
        switch_indices = s["node_switch_id"]
        modes = jnp.where(switch_indices >= 0, s["switch_mode"][jnp.clip(switch_indices, 0, 2)], 0)
        successor = s["node_next"][jnp.arange(64, dtype=jnp.int32), modes]
        flow = s["node_pos"][jnp.clip(successor, 0, 63)] - s["node_pos"]
        s["node_flow"] = jnp.where((s["node_active"] & (successor >= 0))[:, None], flow, 0).astype(jnp.int32)
        s["switch_direction"] = s["node_flow"][jnp.array([4, 12, 20], dtype=jnp.int32)]
        xs = jnp.arange(21, dtype=jnp.int32)[None, None, :]
        ys = jnp.arange(15, dtype=jnp.int32)[None, :, None]
        s["map_visibility"] = ((jnp.abs(xs - s["agent_pos"][:, 0, None, None]) <= 2)
                               & (jnp.abs(ys - s["agent_pos"][:, 1, None, None]) <= 2))
        return s

    def reset(self, key):
        keys = jax.random.split(key, 2)
        catalogue = jnp.array([[0, 0, 1], [0, 1, 0], [1, 0, 0],
                               [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=jnp.int32)
        recipe_ids = jax.random.randint(keys[0], (4,), 0, 6, dtype=jnp.int32)
        starts = jnp.array([[2, 8], [5, 7], [9, 3], [9, 11]], dtype=jnp.int32)
        agent_pos = starts[jax.random.permutation(keys[1], 4)]
        node_pos = jnp.array([[2,7],[3,7],[4,7],[5,7],[6,7],[6,6],[6,5],[6,4],[6,3],[7,3],[8,3],[9,3],[10,3],[6,8],[6,9],[6,10],[6,11],[7,11],[8,11],[9,11],[10,11],[10,2],[11,2],[12,2],[13,2],[14,2],[15,2],[16,2],[17,2],[18,2],[10,4],[10,5],[11,5],[12,5],[13,5],[14,5],[15,5],[16,5],[17,5],[18,5],[10,10],[10,9],[11,9],[12,9],[13,9],[14,9],[15,9],[16,9],[17,9],[18,9],[10,12],[11,12],[12,12],[13,12],[14,12],[15,12],[16,12],[17,12],[18,12],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]], dtype=jnp.int32)
        node_index = jnp.arange(64, dtype=jnp.int32)
        node_active = node_index < 59
        switch_nodes = jnp.array([4, 12, 20], dtype=jnp.int32)
        sink_nodes = jnp.array([29, 39, 49, 58], dtype=jnp.int32)
        node_kind = jnp.where(node_active, 0, -1).astype(jnp.int32)
        node_kind = node_kind.at[0].set(1)
        node_kind = node_kind.at[switch_nodes].set(2)
        node_kind = node_kind.at[sink_nodes].set(3)
        ordinary_next = jnp.where(node_active & (node_kind != 3), node_index + 1, -1)
        node_next = jnp.stack([ordinary_next, ordinary_next], axis=1)
        node_next = node_next.at[switch_nodes].set(jnp.array([[5, 13], [21, 30], [40, 50]], dtype=jnp.int32))
        node_switch_id = jnp.full((64,), -1, dtype=jnp.int32).at[switch_nodes].set(jnp.arange(3, dtype=jnp.int32))
        node_sink_id = jnp.full((64,), -1, dtype=jnp.int32).at[sink_nodes].set(jnp.arange(4, dtype=jnp.int32))
        xs = jnp.arange(21, dtype=jnp.int32)[None, :]
        ys = jnp.arange(15, dtype=jnp.int32)[:, None]
        wall = (xs == 0) | (xs == 20) | (ys == 0) | (ys == 14)
        tile_map = jnp.where(wall, 1, 0).astype(jnp.int32)
        tile_map = tile_map.at[node_pos[:59, 1], node_pos[:59, 0]].set(2)
        tile_map = tile_map.at[7, 2].set(3)
        tile_map = tile_map.at[node_pos[switch_nodes, 1], node_pos[switch_nodes, 0]].set(4)
        tile_map = tile_map.at[node_pos[sink_nodes, 1], node_pos[sink_nodes, 0]].set(5)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "wall": wall,
            "tile_map": tile_map,
            "node_active": node_active,
            "node_pos": node_pos,
            "node_next": node_next,
            "node_kind": node_kind,
            "node_switch_id": node_switch_id,
            "node_sink_id": node_sink_id,
            "agent_active": jnp.ones((4,), dtype=jnp.bool_),
            "agent_pos": agent_pos,
            "source_pos": jnp.array([[2, 7]], dtype=jnp.int32),
            "source_active": jnp.ones((1,), dtype=jnp.bool_),
            "switch_pos": node_pos[switch_nodes],
            "switch_active": jnp.ones((3,), dtype=jnp.bool_),
            "sink_pos": node_pos[sink_nodes],
            "sink_active": jnp.ones((4,), dtype=jnp.bool_),
            "switch_mode": jnp.zeros((3,), dtype=jnp.int32),
            "source_stock": jnp.array([8, 8], dtype=jnp.int32),
            "source_cooldown": jnp.array(0, dtype=jnp.int32),
            "sink_recipe": catalogue[recipe_ids],
            "sink_progress": jnp.zeros((4,), dtype=jnp.int32),
            "pulse_active": jnp.zeros((16,), dtype=jnp.bool_),
            "pulse_node": jnp.full((16,), -1, dtype=jnp.int32),
            "pulse_colour": jnp.full((16,), -1, dtype=jnp.int32),
            "pulse_id": jnp.full((16,), -1, dtype=jnp.int32),
            "launch_count": jnp.array(0, dtype=jnp.int32),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "done": jnp.array(False, dtype=jnp.bool_),
            "terminal_reason": jnp.array(0, dtype=jnp.int32),
            "prev_agent_pos": agent_pos,
            "prev_switch_mode": jnp.zeros((3,), dtype=jnp.int32),
            "prev_sink_progress": jnp.zeros((4,), dtype=jnp.int32),
            "event_agent_outcome": jnp.zeros((4,), dtype=jnp.int32),
            "event_invalid_action": jnp.zeros((4,), dtype=jnp.bool_),
            "event_switch_delta": jnp.zeros((3,), dtype=jnp.int32),
            "event_switch_conflict": jnp.zeros((3,), dtype=jnp.bool_),
            "event_source_conflict": jnp.array(False, dtype=jnp.bool_),
            "event_pulse_active": jnp.zeros((17,), dtype=jnp.bool_),
            "event_pulse_kind": jnp.zeros((17,), dtype=jnp.int32),
            "event_pulse_from": jnp.full((17,), -1, dtype=jnp.int32),
            "event_pulse_to": jnp.full((17,), -1, dtype=jnp.int32),
            "event_pulse_colour": jnp.full((17,), -1, dtype=jnp.int32),
            "event_pulse_id": jnp.full((17,), -1, dtype=jnp.int32),
            "event_pulse_slot": jnp.full((17,), -1, dtype=jnp.int32),
            "event_sink_outcome": jnp.zeros((4,), dtype=jnp.int32),
            "event_terminal_reason": jnp.array(0, dtype=jnp.int32),
        }
        state = self.refresh(state)
        return self.observe(state), state

    def walk_geometry(self, state):
        targets = state["agent_pos"][:, None, :] + self.directions()[None, :, :]
        x = targets[..., 0]
        y = targets[..., 1]
        inside = (x >= 0) & (x < 21) & (y >= 0) & (y < 15)
        tiles = state["tile_map"][jnp.clip(y, 0, 14), jnp.clip(x, 0, 20)]
        occupied = jnp.any(
            jnp.all(targets[:, :, None, :] == state["agent_pos"][None, None, :, :], axis=-1)
            & state["agent_active"][None, None, :], axis=-1)
        return inside & (tiles != 1) & (tiles < 3) & ~occupied

    def fixture_adjacency(self, state):
        switch_distance = jnp.sum(
            jnp.abs(state["agent_pos"][:, None, :] - state["switch_pos"][None, :, :]), axis=-1)
        source_distance = jnp.sum(jnp.abs(state["agent_pos"] - state["source_pos"][0]), axis=-1)
        return switch_distance == 1, source_distance == 1

    def observe(self, state):
        xs = jnp.broadcast_to(jnp.arange(21, dtype=jnp.int32)[None, :], (15, 21))
        ys = jnp.broadcast_to(jnp.arange(15, dtype=jnp.int32)[:, None], (15, 21))
        grid = jnp.stack([xs, ys], axis=-1)
        switch_cells = jnp.all(grid[None, :, :, :] == state["switch_pos"][:, None, None, :], axis=-1)
        sink_cells = jnp.all(grid[None, :, :, :] == state["sink_pos"][:, None, None, :], axis=-1)
        pulse_cells = (jnp.all(grid[None, :, :, :] == state["pulse_pos"][:, None, None, :], axis=-1)
                       & state["pulse_active"][:, None, None])
        agent_cells = (jnp.all(grid[None, :, :, :] == state["agent_pos"][:, None, None, :], axis=-1)
                       & state["agent_active"][:, None, None])
        source_cells = state["tile_map"] == 3
        sink_progress = jnp.sum(
            sink_cells.astype(jnp.float32) * state["sink_progress"][:, None, None].astype(jnp.float32), axis=0) / 3.0
        features = jnp.stack([
            jnp.ones((15, 21), dtype=jnp.bool_),
            state["wall"] | (state["tile_map"] >= 3),
            state["tile_map"] >= 2,
            source_cells,
            source_cells.astype(jnp.float32) * state["source_stock"][0].astype(jnp.float32) / 8.0,
            source_cells.astype(jnp.float32) * state["source_stock"][1].astype(jnp.float32) / 8.0,
            source_cells.astype(jnp.float32) * state["source_cooldown"].astype(jnp.float32) / 2.0,
            state["tile_map"] == 4,
            jnp.any(switch_cells & (state["switch_mode"][:, None, None] == 0), axis=0),
            jnp.any(switch_cells & (state["switch_mode"][:, None, None] == 1), axis=0),
            state["tile_map"] == 5,
            jnp.any(sink_cells & (state["sink_demand"][:, None, None] == 0), axis=0),
            jnp.any(sink_cells & (state["sink_demand"][:, None, None] == 1), axis=0),
            sink_progress,
            jnp.any(sink_cells & (state["sink_progress"][:, None, None] == 3), axis=0),
            jnp.any(pulse_cells & (state["pulse_colour"][:, None, None] == 0), axis=0),
            jnp.any(pulse_cells & (state["pulse_colour"][:, None, None] == 1), axis=0),
            jnp.zeros((15, 21), dtype=jnp.bool_),
            agent_cells[0], agent_cells[1], agent_cells[2], agent_cells[3]
        ], axis=-1).astype(jnp.float32)
        local_x = jnp.broadcast_to(
            state["agent_pos"][:, 0, None, None] + jnp.arange(-2, 3, dtype=jnp.int32)[None, None, :], (4, 5, 5))
        local_y = jnp.broadcast_to(
            state["agent_pos"][:, 1, None, None] + jnp.arange(-2, 3, dtype=jnp.int32)[None, :, None], (4, 5, 5))
        in_bounds = (local_x >= 0) & (local_x < 21) & (local_y >= 0) & (local_y < 15)
        patch = features[jnp.clip(local_y, 0, 14), jnp.clip(local_x, 0, 20)]
        patch = patch * in_bounds[:, :, :, None].astype(jnp.float32)
        patch = patch.at[:, 2, 2, 17].set(1.0)
        position = state["agent_pos"].astype(jnp.float32) / jnp.array([20.0, 14.0], dtype=jnp.float32)
        time = jnp.broadcast_to(state["t"].astype(jnp.float32).reshape(1, 1) / 512.0, (4, 1))
        return jnp.concatenate([
            patch.reshape(4, 550), jnp.eye(4, dtype=jnp.float32), position, time
        ], axis=1).astype(jnp.float32)

    def legal_actions(self, state):
        switch_adjacent, source_adjacent = self.fixture_adjacency(state)
        can_set = jnp.any(switch_adjacent, axis=1)
        can_launch = source_adjacent & (state["source_cooldown"] == 0)
        available = jnp.concatenate([
            jnp.ones((4, 1), dtype=jnp.bool_), self.walk_geometry(state),
            can_set[:, None], can_set[:, None],
            (can_launch & (state["source_stock"][0] > 0))[:, None],
            (can_launch & (state["source_stock"][1] > 0))[:, None]
        ], axis=1)
        absorbing = jnp.broadcast_to(jnp.arange(9, dtype=jnp.int32)[None, :] == 0, (4, 9))
        return jnp.where(state["done"], absorbing, available)

    def absorb(self, inputs):
        state, actions = inputs
        return (self.observe(state), state, jnp.zeros((4,), dtype=jnp.float32),
                jnp.ones((4,), dtype=jnp.bool_),
                {"done_all": state["done"], "terminated": state["terminated"], "truncated": state["truncated"]})

    def advance(self, inputs):
        state, supplied_actions = inputs
        invalid = (supplied_actions < 0) | (supplied_actions >= 9)
        actions = jnp.where(invalid, 0, supplied_actions).astype(jnp.int32)
        s = dict(state)
        s["prev_agent_pos"] = state["agent_pos"]
        s["prev_switch_mode"] = state["switch_mode"]
        s["prev_sink_progress"] = state["sink_progress"]
        walking = (actions >= 1) & (actions <= 4)
        walk_index = jnp.clip(actions - 1, 0, 3)
        targets = state["agent_pos"] + self.directions()[walk_index]
        walk_ok = jnp.take_along_axis(self.walk_geometry(state), walk_index[:, None], axis=1)[:, 0]
        proposals = walking & walk_ok
        same_target = jnp.all(targets[:, None, :] == targets[None, :, :], axis=-1)
        competing = jnp.sum(same_target & proposals[:, None] & proposals[None, :], axis=1)
        accepted_walk = proposals & (competing == 1)
        switch_adjacent, source_adjacent = self.fixture_adjacency(state)
        request_zero = jnp.any(switch_adjacent & (actions[:, None] == 5), axis=0)
        request_one = jnp.any(switch_adjacent & (actions[:, None] == 6), axis=0)
        switch_conflict = request_zero & request_one
        new_modes = jnp.where(request_zero & ~request_one, 0,
                              jnp.where(request_one & ~request_zero, 1, state["switch_mode"])).astype(jnp.int32)
        active = state["pulse_active"]
        old_nodes = state["pulse_node"]
        safe_nodes = jnp.clip(old_nodes, 0, 63)
        switch_id = state["node_switch_id"][safe_nodes]
        old_branch = jnp.where(switch_id >= 0, state["switch_mode"][jnp.clip(switch_id, 0, 2)], 0)
        destination = state["node_next"][safe_nodes, old_branch]
        sink_id = state["node_sink_id"][jnp.clip(destination, 0, 63)]
        safe_sink = jnp.clip(sink_id, 0, 3)
        arrival = active & (sink_id >= 0)
        old_progress = state["sink_progress"][safe_sink]
        demand = state["sink_recipe"][safe_sink, jnp.clip(old_progress, 0, 2)]
        accepted = arrival & (old_progress < 3) & (state["pulse_colour"] == demand)
        already_complete = arrival & (old_progress == 3)
        wasted = arrival & ~accepted
        sink_matches = sink_id[:, None] == jnp.arange(4, dtype=jnp.int32)[None, :]
        progress_increment = jnp.sum(sink_matches & accepted[:, None], axis=0).astype(jnp.int32)
        pulse_outcome = jnp.where(accepted, 1, jnp.where(already_complete, 3, 2)).astype(jnp.int32)
        sink_outcome = jnp.sum(
            jnp.where(sink_matches & arrival[:, None], pulse_outcome[:, None], 0), axis=0).astype(jnp.int32)
        remaining_active = active & ~arrival
        remaining_nodes = jnp.where(remaining_active, destination, -1).astype(jnp.int32)
        remaining_colour = jnp.where(remaining_active, state["pulse_colour"], -1).astype(jnp.int32)
        remaining_id = jnp.where(remaining_active, state["pulse_id"], -1).astype(jnp.int32)
        request_blue = jnp.any(source_adjacent & (actions == 7))
        request_amber = jnp.any(source_adjacent & (actions == 8))
        source_conflict = request_blue & request_amber
        selected_colour = jnp.where(request_amber, 1, 0).astype(jnp.int32)
        free = ~remaining_active
        slot = jnp.argmax(free.astype(jnp.int32)).astype(jnp.int32)
        source_empty = ~jnp.any(remaining_active & (remaining_nodes == 0))
        emit = ((request_blue | request_amber) & ~source_conflict
                & (state["source_cooldown"] == 0) & (state["source_stock"][selected_colour] > 0)
                & jnp.any(free) & source_empty)
        s["pulse_active"] = remaining_active.at[slot].set(remaining_active[slot] | emit)
        s["pulse_node"] = remaining_nodes.at[slot].set(jnp.where(emit, 0, remaining_nodes[slot]))
        s["pulse_colour"] = remaining_colour.at[slot].set(jnp.where(emit, selected_colour, remaining_colour[slot]))
        s["pulse_id"] = remaining_id.at[slot].set(jnp.where(emit, state["launch_count"], remaining_id[slot]))
        used_colour = (jnp.arange(2, dtype=jnp.int32) == selected_colour).astype(jnp.int32) * emit.astype(jnp.int32)
        s["source_stock"] = state["source_stock"] - used_colour
        s["launch_count"] = state["launch_count"] + emit.astype(jnp.int32)
        s["source_cooldown"] = jnp.where(emit, 2, jnp.maximum(state["source_cooldown"] - 1, 0)).astype(jnp.int32)
        s["sink_progress"] = state["sink_progress"] + progress_increment
        s["agent_pos"] = jnp.where(accepted_walk[:, None], targets, state["agent_pos"])
        s["switch_mode"] = new_modes
        s["event_switch_delta"] = new_modes - state["switch_mode"]
        s["event_switch_conflict"] = switch_conflict
        s["event_source_conflict"] = source_conflict
        has_switch = jnp.any(switch_adjacent, axis=1)
        own_conflict = jnp.any(switch_adjacent & switch_conflict[None, :], axis=1)
        setting = (actions == 5) | (actions == 6)
        launching = (actions == 7) | (actions == 8)
        set_outcome = jnp.where(has_switch, jnp.where(own_conflict, 4, 3), 7)
        launch_outcome = jnp.where(source_adjacent & emit, 5, 6)
        s["event_agent_outcome"] = jnp.where(
            walking, jnp.where(accepted_walk, 1, 2),
            jnp.where(setting, set_outcome, jnp.where(launching, launch_outcome, 0))).astype(jnp.int32)
        s["event_invalid_action"] = invalid
        old_event_kind = jnp.where(active,
            jnp.where(arrival, jnp.where(accepted, 2, jnp.where(already_complete, 4, 3)), 1), 0).astype(jnp.int32)
        s["event_pulse_active"] = jnp.concatenate([active, emit.reshape(1)])
        s["event_pulse_kind"] = jnp.concatenate([old_event_kind, jnp.where(emit, 5, 0).reshape(1)]).astype(jnp.int32)
        s["event_pulse_from"] = jnp.concatenate([
            jnp.where(active, old_nodes, -1), jnp.array([-1], dtype=jnp.int32)]).astype(jnp.int32)
        s["event_pulse_to"] = jnp.concatenate([
            jnp.where(active, destination, -1), jnp.where(emit, 0, -1).reshape(1)]).astype(jnp.int32)
        s["event_pulse_colour"] = jnp.concatenate([
            jnp.where(active, state["pulse_colour"], -1),
            jnp.where(emit, selected_colour, -1).reshape(1)]).astype(jnp.int32)
        s["event_pulse_id"] = jnp.concatenate([
            jnp.where(active, state["pulse_id"], -1),
            jnp.where(emit, state["launch_count"], -1).reshape(1)]).astype(jnp.int32)
        s["event_pulse_slot"] = jnp.concatenate([
            jnp.where(active, jnp.arange(16, dtype=jnp.int32), -1),
            jnp.where(emit, slot, -1).reshape(1)]).astype(jnp.int32)
        s["event_sink_outcome"] = sink_outcome
        s["t"] = state["t"] + jnp.array(1, dtype=jnp.int32)
        success = jnp.all(s["sink_progress"] == 3)
        exhausted = (jnp.sum(s["source_stock"]) == 0) & ~jnp.any(s["pulse_active"]) & ~success
        s["terminated"] = success | exhausted
        s["truncated"] = (s["t"] >= 512) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        s["terminal_reason"] = jnp.where(success, 1, jnp.where(exhausted, 2, jnp.where(s["truncated"], 3, 0))).astype(jnp.int32)
        s["event_terminal_reason"] = s["terminal_reason"]
        s = self.refresh(s)
        reward = (jnp.sum(accepted).astype(jnp.float32) / 12.0 + success.astype(jnp.float32)
                  - jnp.array(0.001, dtype=jnp.float32)
                  - jnp.array(0.02, dtype=jnp.float32) * jnp.sum(wasted).astype(jnp.float32))
        return (self.observe(s), s, jnp.full((4,), reward, dtype=jnp.float32),
                jnp.full((4,), s["done"], dtype=jnp.bool_),
                {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]})

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (state, actions))
