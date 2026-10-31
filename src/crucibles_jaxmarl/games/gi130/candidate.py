"""Sluice Commons: conserved water, delayed gates, footbridges and cargo."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 158
    num_actions = 10
    max_steps = 160

    def refresh(self, s):
        s = dict(s)
        s["boat_pos"] = s["node_pos"][s["boat_node"]]
        s["boat_active"] = ~s["boat_delivered"]
        s["crop_active"] = jnp.ones((3,), dtype=jnp.bool_)
        s["agent_vis"] = jnp.ones((3, 3), dtype=jnp.bool_)
        s["gate_vis"] = jnp.ones((3, 3), dtype=jnp.bool_)
        s["node_vis"] = jnp.ones((3, 4), dtype=jnp.bool_)
        s["boat_vis"] = jnp.ones((3, 3), dtype=jnp.bool_)
        s["crop_vis"] = jnp.ones((3, 3), dtype=jnp.bool_)
        return s

    def reset(self, key):
        tile = jnp.zeros((7, 11), dtype=jnp.int32)
        tile = tile.at[3, :].set(1)
        tile = tile.at[3, jnp.array([3, 6, 8], dtype=jnp.int32)].set(2)
        s = {
            "t": jnp.array(0, dtype=jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False),
            "truncated": jnp.array(False), "tile": tile,
            "agent_pos": jnp.array([[2, 2], [5, 2], [8, 2]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "gate_pos": jnp.array([[3, 3], [6, 3], [8, 3]], dtype=jnp.int32),
            "gate_active": jnp.ones((3,), dtype=jnp.bool_),
            "gate_open": jnp.zeros((3,), dtype=jnp.bool_),
            "node_pos": jnp.array([[1, 3], [4, 3], [7, 3], [9, 3]], dtype=jnp.int32),
            "node_active": jnp.ones((4,), dtype=jnp.bool_),
            "water": jnp.array([12, 1, 1, 1], dtype=jnp.int32),
            "transit": jnp.zeros((3,), dtype=jnp.int32),
            "drained": jnp.array(0, dtype=jnp.int32),
            "boat_node": jnp.array([0, 1, 2], dtype=jnp.int32),
            "boat_running": jnp.zeros((3,), dtype=jnp.bool_),
            "boat_delivered": jnp.zeros((3,), dtype=jnp.bool_),
            "crop_pos": jnp.array([[4, 5], [7, 5], [9, 5]], dtype=jnp.int32),
            "crop_dose": jnp.zeros((3,), dtype=jnp.int32),
            "crop_health": jnp.full((3,), 3, dtype=jnp.int32),
            "last_flow": jnp.zeros((3,), dtype=jnp.int32),
            "last_irrigated": jnp.zeros((3,), dtype=jnp.bool_),
            "last_delivered": jnp.zeros((3,), dtype=jnp.bool_),
            "last_flood": jnp.zeros((3,), dtype=jnp.bool_),
            "last_spill": jnp.zeros((4,), dtype=jnp.int32),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate(tuple(s[k].reshape(-1).astype(jnp.float32) for k in (
            "t", "done", "terminated", "truncated", "tile", "agent_pos",
            "gate_pos", "gate_open", "node_pos", "water", "transit", "drained",
            "boat_pos", "boat_node", "boat_running", "boat_delivered",
            "crop_pos", "crop_dose", "crop_health", "last_flow", "last_irrigated",
            "last_delivered", "last_flood", "last_spill")))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32),
                                jnp.broadcast_to(public, (3, public.shape[0]))), axis=1)

    def legal_actions(self, s):
        mask = jnp.ones((3, 10), dtype=jnp.bool_) & ~s["done"]
        return mask.at[:, 0].set(True)

    def move(self, s, a):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)]
        inside = jnp.all((target >= 0) & (target <= jnp.array([10, 6])), axis=1)
        q = jnp.clip(target, jnp.array([0, 0]), jnp.array([10, 6]))
        at_gate = jnp.all(q[:, None, :] == s["gate_pos"][None, :, :], axis=2)
        closed_bridge = jnp.any(at_gate & ~s["gate_open"][None, :], axis=1)
        passable = (s["tile"][q[:, 1], q[:, 0]] == 0) | closed_bridge
        occupied = jnp.any(jnp.all(q[:, None, :] == s["agent_pos"][None, :, :], axis=2)
                           & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        valid = inside & passable & ~occupied & ~conflict
        return jnp.where(valid[:, None], q, s["agent_pos"])

    def advance(self, s, a):
        n = dict(s)
        n["t"] = s["t"] + 1
        n["agent_pos"] = self.move(s, a)
        # GATE_VOTES: all nearby votes apply together; any close vetoes opening.
        near_gate = jnp.sum(jnp.abs(n["agent_pos"][:, None, :] - s["gate_pos"][None, :, :]), axis=2) <= 1
        opening = jnp.any(near_gate & (a[:, None] == 5), axis=0)
        closing = jnp.any(near_gate & (a[:, None] == 6), axis=0)
        n["gate_open"] = (s["gate_open"] | opening) & ~closing
        # CONSERVED_FLOW: old packets arrive; departures use pre-arrival donor stock.
        arrived = s["water"].at[1:].add(s["transit"])
        flow = (n["gate_open"] & (s["water"][:-1] > 0)
                & (arrived[1:] < 6)).astype(jnp.int32)
        water = arrived.at[:-1].add(-flow)
        n["transit"] = flow
        n["last_flow"] = flow
        doses = s["crop_dose"]
        irrigated = jnp.zeros((3,), dtype=jnp.bool_)
        spill = jnp.zeros((4,), dtype=jnp.int32)
        # ALLOCATION: rotating role priority resolves scarce water and shared crops.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            distances = jnp.sum(jnp.abs(s["crop_pos"] - n["agent_pos"][i]), axis=1)
            eligible = (distances <= 1) & (doses < 2) & (s["crop_health"] > 0) & (water[1:] > 0)
            c = jnp.argmax(eligible.astype(jnp.int32))
            take = (a[i] == 7) & jnp.any(eligible)
            doses = doses.at[c].add(take.astype(jnp.int32))
            water = water.at[c + 1].add(-take.astype(jnp.int32))
            irrigated = irrigated.at[c].set(irrigated[c] | take)
            distance_node = jnp.sum(jnp.abs(s["node_pos"] - n["agent_pos"][i]), axis=1)
            can_spill = (distance_node <= 1) & (water > 0)
            b = jnp.argmax(can_spill.astype(jnp.int32))
            release = (a[i] == 9) & jnp.any(can_spill)
            water = water.at[b].add(-release.astype(jnp.int32))
            spill = spill.at[b].add(release.astype(jnp.int32))
        n["water"] = water
        n["crop_dose"] = doses
        n["drained"] = s["drained"] + jnp.sum(spill)
        n["last_spill"] = spill
        n["last_irrigated"] = irrigated
        # CARGO_PASSAGE: a launched boat advances at most one wet, open lock per turn.
        near_boat = jnp.sum(jnp.abs(n["agent_pos"][:, None, :] - s["boat_pos"][None, :, :]), axis=2) <= 1
        launched = jnp.any(near_boat & (a[:, None] == 8), axis=0)
        running = s["boat_running"] | (launched & ~s["boat_delivered"])
        b = s["boat_node"]
        edge = jnp.minimum(b, 2)
        next_b = jnp.minimum(b + 1, 3)
        passage = (running & ~s["boat_delivered"] & (b < 3) & n["gate_open"][edge]
                   & (water[b] >= 1) & (water[b] <= 4)
                   & (water[next_b] >= 1) & (water[next_b] <= 4))
        n["boat_node"] = b + passage.astype(jnp.int32)
        delivered = (n["boat_node"] == 3) & ~s["boat_delivered"]
        n["boat_delivered"] = s["boat_delivered"] | delivered
        n["boat_running"] = running & ~n["boat_delivered"]
        n["last_delivered"] = delivered
        # FLOOD_COST: unfinished crops lose persistent health at basin depth five or six.
        flood = (water[1:] >= 5) & (doses < 2) & (s["crop_health"] > 0)
        n["crop_health"] = s["crop_health"] - flood.astype(jnp.int32)
        n["last_flood"] = flood
        lost = (n["crop_health"] == 0) & (s["crop_health"] > 0)
        rewards = (0.1 * jnp.sum(irrigated) + 0.3 * jnp.sum(delivered)
                   + 0.5 * irrigated.astype(jnp.float32) + 2.0 * delivered.astype(jnp.float32)
                   - 0.2 * flood.astype(jnp.float32) - 1.0 * lost.astype(jnp.float32))
        n["terminated"] = jnp.all(n["boat_delivered"]) & jnp.all((doses >= 2) | (n["crop_health"] == 0))
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        return self.refresh(n), rewards.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions, 0, 9)
        valid = self.legal_actions(state)[jnp.arange(3), safe] & (safe == actions)
        a = jnp.where(valid, safe, 0)
        n, rewards = self.advance(state, a)
        # ABSORBING_TERMINAL: preserve all leaves, causes and clock on terminal input.
        n = {k: jnp.where(state["done"], state[k], n[k]) for k in state}
        rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        dones = jnp.full((3,), n["done"], dtype=jnp.bool_)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, dones, info
