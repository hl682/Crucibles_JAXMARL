"""Twin-Grip Lockway: original spatial kernel, authored without execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 1876
    num_actions = 9
    max_steps = 192

    def directions(self):
        return jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)

    def refresh(self, state):
        s = dict(state)
        occupied = jnp.any(
            jnp.all(s["agent_pos"][:, None, :] == s["plate_pos"][None, :, :], axis=-1)
            & s["agent_active"][:, None], axis=0)
        associated = s["plate_gate"][None, :] == jnp.arange(2, dtype=jnp.int32)[:, None]
        requested = jnp.any(associated & occupied[None, :], axis=1)
        agent_gate = s["gate_id"][s["agent_pos"][:, 1], s["agent_pos"][:, 0]]
        pallet_gate = s["gate_id"][s["pallet_pos"][0, 1], s["pallet_pos"][0, 0]]
        gate_indices = jnp.arange(2, dtype=jnp.int32)
        aperture_occupied = jnp.any(
            (agent_gate[None, :] == gate_indices[:, None]) & s["agent_active"][None, :], axis=1)
        aperture_occupied = aperture_occupied | ((pallet_gate == gate_indices) & s["pallet_active"][0])
        s["plate_occupied"] = occupied
        s["gate_requested_open"] = requested
        s["gate_open"] = requested | aperture_occupied
        s["gate_plate_count"] = jnp.sum(associated & occupied[None, :], axis=1).astype(jnp.int32)
        tiles = jnp.where(s["wall"], 1, 0).astype(jnp.int32)
        gate_cells = s["gate_id"] >= 0
        gate_values = jnp.where(s["gate_open"][jnp.clip(s["gate_id"], 0, 1)], 3, 2)
        tiles = jnp.where(gate_cells, gate_values, tiles)
        tiles = tiles.at[s["plate_pos"][:, 1], s["plate_pos"][:, 0]].set(
            jnp.array([4, 4, 5, 5], dtype=jnp.int32))
        tiles = tiles.at[s["berth_pos"][0, 1], s["berth_pos"][0, 0]].set(6)
        s["tile_map"] = tiles
        s["pallet_on_berth"] = jnp.all(s["pallet_pos"] == s["berth_pos"], axis=1)
        s["agent_visibility"] = jnp.broadcast_to(s["agent_active"][None, :], (3, 3))
        s["pallet_visibility"] = jnp.broadcast_to(s["pallet_active"][None, :], (3, 1))
        s["plate_visibility"] = jnp.broadcast_to(s["plate_active"][None, :], (3, 4))
        s["gate_visibility"] = jnp.broadcast_to(s["gate_active"][None, :], (3, 2))
        s["berth_visibility"] = jnp.broadcast_to(s["berth_active"][None, :], (3, 1))
        return s

    def reset(self, key):
        keys = jax.random.split(key, 2)
        pallet_x = jax.random.randint(keys[0], (), 2, 5, dtype=jnp.int32)
        starts = jnp.array([[0, 4], [0, 6], [2, 2]], dtype=jnp.int32)
        starts = starts.at[0, 0].set(pallet_x)
        starts = starts.at[1, 0].set(pallet_x)
        agent_pos = starts[jax.random.permutation(keys[1], 3)]
        pallet_pos = jnp.stack([pallet_x, jnp.array(5, dtype=jnp.int32)]).reshape(1, 2)
        xs = jnp.arange(17, dtype=jnp.int32)[None, :]
        ys = jnp.arange(11, dtype=jnp.int32)[:, None]
        aperture_rows = (ys >= 4) & (ys <= 6)
        wall = (xs == 0) | (xs == 16) | (ys == 0) | (ys == 10)
        wall = wall | (((xs == 6) | (xs == 11)) & ~aperture_rows)
        gate_id = jnp.where((xs == 6) & aperture_rows, 0,
                            jnp.where((xs == 11) & aperture_rows, 1, -1)).astype(jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "wall": wall,
            "gate_id": gate_id,
            "plate_pos": jnp.array([[4, 2], [8, 2], [9, 8], [13, 8]], dtype=jnp.int32),
            "plate_gate": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
            "berth_pos": jnp.array([[14, 5]], dtype=jnp.int32),
            "gate_pos": jnp.array([[6, 5], [11, 5]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_pos": agent_pos,
            "pallet_active": jnp.ones((1,), dtype=jnp.bool_),
            "pallet_pos": pallet_pos,
            "plate_active": jnp.ones((4,), dtype=jnp.bool_),
            "gate_active": jnp.ones((2,), dtype=jnp.bool_),
            "berth_active": jnp.ones((1,), dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "done": jnp.array(False, dtype=jnp.bool_),
            "prev_agent_pos": agent_pos,
            "prev_pallet_pos": pallet_pos,
            "event_agent_outcome": jnp.zeros((3,), dtype=jnp.int32),
            "event_invalid_action": jnp.zeros((3,), dtype=jnp.bool_),
            "event_haul_agents": jnp.zeros((3,), dtype=jnp.bool_),
            "event_haul_direction": jnp.array(-1, dtype=jnp.int32),
            "event_gate_delta": jnp.zeros((2,), dtype=jnp.int32),
            "event_success": jnp.array(False, dtype=jnp.bool_),
        }
        state = self.refresh(state)
        return self.observe(state), state

    def traversable(self, state, positions):
        x = positions[..., 0]
        y = positions[..., 1]
        inside = (x >= 0) & (x < 17) & (y >= 0) & (y < 11)
        safe_x = jnp.clip(x, 0, 16)
        safe_y = jnp.clip(y, 0, 10)
        gate = state["gate_id"][safe_y, safe_x]
        passable_gate = (gate < 0) | state["gate_open"][jnp.clip(gate, 0, 1)]
        return inside & ~state["wall"][safe_y, safe_x] & passable_gate

    def walk_geometry(self, state):
        targets = state["agent_pos"][:, None, :] + self.directions()[None, :, :]
        occupied_agent = jnp.any(
            jnp.all(targets[:, :, None, :] == state["agent_pos"][None, None, :, :], axis=-1)
            & state["agent_active"][None, None, :], axis=-1)
        occupied_pallet = jnp.all(targets == state["pallet_pos"][0], axis=-1)
        return self.traversable(state, targets) & ~occupied_agent & ~occupied_pallet

    def haul_geometry(self, state):
        directions = self.directions()
        perpendicular = jnp.stack([-directions[:, 1], directions[:, 0]], axis=1)
        pallet = state["pallet_pos"][0]
        positive = jnp.all(
            state["agent_pos"][None, :, :] == (pallet[None, :] + perpendicular)[:, None, :], axis=-1)
        negative = jnp.all(
            state["agent_pos"][None, :, :] == (pallet[None, :] - perpendicular)[:, None, :], axis=-1)
        positive = positive & state["agent_active"][None, :]
        negative = negative & state["agent_active"][None, :]
        members = positive | negative
        pair = jnp.any(positive, axis=1) & jnp.any(negative, axis=1)
        cart_targets = pallet[None, :] + directions
        worker_targets = state["agent_pos"][None, :, :] + directions[:, None, :]
        worker_clear = jnp.all(~members | self.traversable(state, worker_targets), axis=1)
        other = state["agent_active"][None, :] & ~members
        cart_blocked = jnp.any(
            jnp.all(cart_targets[:, None, :] == state["agent_pos"][None, :, :], axis=-1) & other, axis=1)
        worker_blocked = jnp.any(
            jnp.all(worker_targets[:, :, None, :] == state["agent_pos"][None, None, :, :], axis=-1)
            & other[:, None, :], axis=2)
        geometry = (pair & self.traversable(state, cart_targets) & worker_clear
                    & ~cart_blocked & jnp.all(~members | ~worker_blocked, axis=1))
        return geometry, members

    def observe(self, state):
        xs = jnp.arange(17, dtype=jnp.int32)[None, :]
        ys = jnp.arange(11, dtype=jnp.int32)[:, None]
        agent_cells = ((xs[None, :, :] == state["agent_pos"][:, 0, None, None])
                       & (ys[None, :, :] == state["agent_pos"][:, 1, None, None]))
        pallet = state["pallet_pos"][0]
        cart_cells = (xs == pallet[0]) & (ys == pallet[1])
        gate_cells = state["gate_id"] >= 0
        open_cells = gate_cells & state["gate_open"][jnp.clip(state["gate_id"], 0, 1)]
        spatial = jnp.stack([
            state["wall"], open_cells, gate_cells & ~open_cells,
            state["tile_map"] == 4, state["tile_map"] == 5, cart_cells,
            state["tile_map"] == 6, agent_cells[0], agent_cells[1], agent_cells[2]
        ], axis=-1).astype(jnp.float32)
        shared = jnp.broadcast_to(spatial.reshape(1870)[None, :], (3, 1870))
        scalar = jnp.concatenate([
            state["t"].astype(jnp.float32).reshape(1) / 192.0,
            state["gate_plate_count"].astype(jnp.float32) / 2.0
        ])
        return jnp.concatenate([
            shared, jnp.eye(3, dtype=jnp.float32),
            jnp.broadcast_to(scalar[None, :], (3, 3))
        ], axis=1).astype(jnp.float32)

    def legal_actions(self, state):
        geometry, members = self.haul_geometry(state)
        available = jnp.concatenate([
            jnp.ones((3, 1), dtype=jnp.bool_),
            self.walk_geometry(state),
            jnp.transpose(members & geometry[:, None])
        ], axis=1)
        absorbing = jnp.broadcast_to(jnp.arange(9, dtype=jnp.int32)[None, :] == 0, (3, 9))
        return jnp.where(state["done"], absorbing, available)

    def absorb(self, inputs):
        state, actions = inputs
        return (self.observe(state), state, jnp.zeros((3,), dtype=jnp.float32),
                jnp.ones((3,), dtype=jnp.bool_),
                {"done_all": state["done"], "terminated": state["terminated"], "truncated": state["truncated"]})

    def advance(self, inputs):
        state, supplied_actions = inputs
        invalid = (supplied_actions < 0) | (supplied_actions >= 9)
        actions = jnp.where(invalid, 0, supplied_actions).astype(jnp.int32)
        s = dict(state)
        s["prev_agent_pos"] = state["agent_pos"]
        s["prev_pallet_pos"] = state["pallet_pos"]
        geometry, members = self.haul_geometry(state)
        agreeing = jnp.all(
            ~members | (actions[None, :] == jnp.arange(5, 9, dtype=jnp.int32)[:, None]), axis=1)
        accepted_directions = geometry & agreeing
        hauling = jnp.any(accepted_directions)
        chosen = jnp.argmax(accepted_directions.astype(jnp.int32)).astype(jnp.int32)
        haulers = members[chosen] & hauling
        direction = self.directions()[chosen]
        cart_destination = state["pallet_pos"][0] + jnp.where(hauling, direction, jnp.zeros((2,), dtype=jnp.int32))
        haul_destinations = state["agent_pos"] + direction[None, :]
        walk_index = jnp.clip(actions - 1, 0, 3)
        walk_targets = state["agent_pos"] + self.directions()[walk_index]
        walking_action = (actions >= 1) & (actions <= 4)
        walk_valid = jnp.take_along_axis(self.walk_geometry(state), walk_index[:, None], axis=1)[:, 0]
        reserved_worker = jnp.any(
            jnp.all(walk_targets[:, None, :] == haul_destinations[None, :, :], axis=-1)
            & haulers[None, :], axis=1)
        reserved_cart = jnp.all(walk_targets == cart_destination, axis=1) & hauling
        proposals = walking_action & walk_valid & ~reserved_worker & ~reserved_cart & ~haulers
        same_target = jnp.all(walk_targets[:, None, :] == walk_targets[None, :, :], axis=-1)
        competing = jnp.sum(same_target & proposals[:, None] & proposals[None, :], axis=1)
        accepted_walk = proposals & (competing == 1)
        s["agent_pos"] = jnp.where(
            haulers[:, None], haul_destinations,
            jnp.where(accepted_walk[:, None], walk_targets, state["agent_pos"]))
        s["pallet_pos"] = cart_destination.reshape(1, 2)
        s["event_agent_outcome"] = jnp.where(
            walking_action, jnp.where(accepted_walk, 1, 3),
            jnp.where(actions >= 5, jnp.where(haulers, 2, 4), 0)).astype(jnp.int32)
        s["event_invalid_action"] = invalid
        s["event_haul_agents"] = haulers
        s["event_haul_direction"] = jnp.where(hauling, chosen, -1).astype(jnp.int32)
        s = self.refresh(s)
        s["event_gate_delta"] = s["gate_open"].astype(jnp.int32) - state["gate_open"].astype(jnp.int32)
        s["t"] = state["t"] + jnp.array(1, dtype=jnp.int32)
        success = jnp.all(s["pallet_pos"] == s["berth_pos"]) & jnp.all(s["agent_pos"][:, 0] >= 12)
        s["terminated"] = success
        s["truncated"] = (s["t"] >= 192) & ~success
        s["done"] = s["terminated"] | s["truncated"]
        s["event_success"] = success
        reward = jnp.array(-0.005, dtype=jnp.float32) + success.astype(jnp.float32)
        rewards = jnp.full((3,), reward, dtype=jnp.float32)
        dones = jnp.full((3,), s["done"], dtype=jnp.bool_)
        return (self.observe(s), s, rewards, dones,
                {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]})

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (state, actions))
