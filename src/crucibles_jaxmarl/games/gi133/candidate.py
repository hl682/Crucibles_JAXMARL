"""Conveyor Workshop: conserved parts, limited buffers and private quality/order data."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 303
    num_actions = 13
    max_steps = 144

    def refresh(self, s):
        s = dict(s)
        s["part_active"] = s["part_mode"] < 3
        s["product_active"] = (s["product_status"] == 1) | (s["product_status"] == 2)
        roles = jnp.arange(4)[:, None]
        s["defect_vis"] = (roles == 1) | s["part_inspected"][None, :]
        s["recipe_vis"] = (roles == 2) | s["order_public"][None, :]
        s["agent_vis"] = jnp.ones((4, 4), dtype=jnp.bool_)
        s["part_vis"] = jnp.ones((4, 6), dtype=jnp.bool_)
        s["station_vis"] = jnp.ones((4, 2), dtype=jnp.bool_)
        s["order_vis"] = jnp.ones((4, 3), dtype=jnp.bool_)
        s["product_vis"] = jnp.ones((4, 3), dtype=jnp.bool_)
        s["belt_vis"] = jnp.ones((4, 16), dtype=jnp.bool_)
        s["control_vis"] = jnp.ones((4, 1), dtype=jnp.bool_)
        s["bench_vis"] = jnp.ones((4, 1), dtype=jnp.bool_)
        s["dock_vis"] = jnp.ones((4, 1), dtype=jnp.bool_)
        held_parts = s["part_carrier"][None, :] == roles
        held_products = s["product_carrier"][None, :] == roles
        s["agent_part"] = jnp.where(jnp.any(held_parts, axis=1), jnp.argmax(held_parts.astype(jnp.int32), axis=1), -1)
        s["agent_product"] = jnp.where(jnp.any(held_products, axis=1), jnp.argmax(held_products.astype(jnp.int32), axis=1), -1)
        nxt = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 12, 14, 15, 15], dtype=jnp.int32)
        s["belt_next"] = nxt.at[4].set(jnp.array([5, 10, 13], dtype=jnp.int32)[s["junction"][0]])
        return s

    def reset(self, key):
        keys = jax.random.split(key, 2)
        belt_pos = jnp.array([[1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], [10, 3], [5, 2], [6, 2], [7, 2], [5, 4], [6, 4], [7, 4]], dtype=jnp.int32)
        tile = jnp.zeros((8, 12), dtype=jnp.int32).at[belt_pos[:, 1], belt_pos[:, 0]].set(1)
        tile = tile.at[2, 7].set(2).at[4, 7].set(2).at[3, 10].set(3).at[6, 10].set(4).at[1, 9:12].set(5).at[3, 5].set(6)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tile": tile,
            "agent_pos": jnp.array([[1, 1], [4, 5], [9, 2], [10, 5]], dtype=jnp.int32),
            "agent_active": jnp.ones((4,), dtype=jnp.bool_),
            "belt_pos": belt_pos, "belt_active": jnp.ones((16,), dtype=jnp.bool_),
            "control_pos": jnp.array([[5, 3]], dtype=jnp.int32),
            "control_active": jnp.ones((1,), dtype=jnp.bool_),
            "junction": jnp.array([0], dtype=jnp.int32), "belt_on": jnp.array([True]),
            "part_pos": belt_pos[jnp.array([0, 1, 2, 3, 5, 6], dtype=jnp.int32)],
            "part_kind": jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32),
            "part_color": jnp.array([0, 0, 1, 1, 0, 1], dtype=jnp.int32),
            "part_defect": jax.random.randint(keys[0], (6,), 0, 2, dtype=jnp.int32),
            "part_inspected": jnp.zeros((6,), dtype=jnp.bool_),
            "part_mode": jnp.zeros((6,), dtype=jnp.int32),
            "part_carrier": jnp.full((6,), -1, dtype=jnp.int32),
            "part_order": jnp.full((6,), -1, dtype=jnp.int32),
            "station_pos": jnp.array([[7, 2], [7, 4]], dtype=jnp.int32),
            "station_active": jnp.ones((2,), dtype=jnp.bool_),
            "station_part": jnp.full((2,), -1, dtype=jnp.int32),
            "station_timer": jnp.zeros((2,), dtype=jnp.int32),
            "station_job": jnp.zeros((2,), dtype=jnp.int32),
            "power": jnp.array(12, dtype=jnp.int32), "spent_power": jnp.array(0, dtype=jnp.int32),
            "order_pos": jnp.array([[9, 1], [10, 1], [11, 1]], dtype=jnp.int32),
            "order_active": jnp.ones((3,), dtype=jnp.bool_),
            "order_color": jax.random.randint(keys[1], (3,), 0, 2, dtype=jnp.int32),
            "order_public": jnp.zeros((3,), dtype=jnp.bool_),
            "order_deadline": jnp.array([48, 80, 112], dtype=jnp.int32),
            "bench_pos": jnp.array([[10, 3]], dtype=jnp.int32),
            "bench_active": jnp.ones((1,), dtype=jnp.bool_),
            "dock_pos": jnp.array([[10, 6]], dtype=jnp.int32),
            "dock_active": jnp.ones((1,), dtype=jnp.bool_),
            "product_pos": jnp.array([[10, 3], [10, 3], [10, 3]], dtype=jnp.int32),
            "product_status": jnp.zeros((3,), dtype=jnp.int32),
            "product_carrier": jnp.full((3,), -1, dtype=jnp.int32),
            "last_belt": jnp.zeros((6,), dtype=jnp.bool_),
            "last_started": jnp.zeros((2,), dtype=jnp.bool_),
            "last_finished": jnp.zeros((2,), dtype=jnp.bool_),
            "last_inspected": jnp.zeros((6,), dtype=jnp.bool_),
            "last_assembled": jnp.zeros((3,), dtype=jnp.bool_),
            "last_rejected": jnp.zeros((3,), dtype=jnp.bool_),
            "last_shipped": jnp.zeros((3,), dtype=jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate(tuple(s[k].reshape(-1).astype(jnp.float32) for k in (
            "t", "done", "terminated", "truncated", "tile", "agent_pos", "agent_part", "agent_product",
            "belt_pos", "belt_next", "control_pos", "junction", "belt_on", "part_pos", "part_kind", "part_color",
            "part_inspected", "part_mode", "part_carrier", "part_order", "station_pos", "station_part",
            "station_timer", "station_job", "power", "spent_power", "order_pos", "order_public", "order_deadline",
            "bench_pos", "dock_pos", "product_pos", "product_status", "product_carrier", "last_belt", "last_started",
            "last_finished", "last_inspected", "last_assembled", "last_rejected", "last_shipped")))
        defects = jnp.where(s["defect_vis"], s["part_defect"][None, :], -1)
        recipes = jnp.where(s["recipe_vis"], s["order_color"][None, :], -1)
        return jnp.concatenate((jnp.eye(4, dtype=jnp.float32), jnp.broadcast_to(public, (4, public.shape[0])),
                                s["defect_vis"].astype(jnp.float32), defects.astype(jnp.float32),
                                s["recipe_vis"].astype(jnp.float32), recipes.astype(jnp.float32)), axis=1)

    def legal_actions(self, s):
        return (jnp.ones((4, 13), dtype=jnp.bool_) & ~s["done"]).at[:, 0].set(True)

    def move(self, s, a):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)]
        inside = jnp.all((target >= 0) & (target <= jnp.array([11, 7])), axis=1)
        q = jnp.clip(target, jnp.array([0, 0]), jnp.array([11, 7]))
        occupied = jnp.any(jnp.all(q[:, None, :] == s["agent_pos"][None, :, :], axis=2) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        return jnp.where((inside & ~occupied & ~conflict)[:, None], q, s["agent_pos"])

    def advance(self, s, a):
        n = dict(s)
        n["t"] = s["t"] + 1
        n["agent_pos"] = self.move(s, a)
        n["part_pos"] = jnp.where((s["part_carrier"] >= 0)[:, None], n["agent_pos"][jnp.clip(s["part_carrier"], 0, 3)], s["part_pos"])
        n["product_pos"] = jnp.where((s["product_carrier"] >= 0)[:, None], n["agent_pos"][jnp.clip(s["product_carrier"], 0, 3)], s["product_pos"])
        # MACHINE_CLOCKS: existing jobs finish before new orders; no instant repair or paint.
        finished = (s["station_timer"] == 1) & (s["station_part"] >= 0)
        n["station_timer"] = jnp.maximum(s["station_timer"] - 1, 0)
        for station in range(2):
            p = jnp.maximum(s["station_part"][station], 0)
            repair = finished[station] & (s["station_job"][station] == 1)
            paint = finished[station] & (s["station_job"][station] == 2)
            n["part_defect"] = n["part_defect"].at[p].set(jnp.where(repair, 0, n["part_defect"][p]))
            n["part_inspected"] = n["part_inspected"].at[p].set(n["part_inspected"][p] | repair)
            n["part_color"] = n["part_color"].at[p].set(jnp.where(paint, 1 - n["part_color"][p], n["part_color"][p]))
        n["station_job"] = jnp.where(finished, 0, s["station_job"])
        started = jnp.zeros((2,), dtype=jnp.bool_)
        inspected = jnp.zeros((6,), dtype=jnp.bool_)
        # HANDOFF_ORDER: cyclic worker priority serializes contested picks, drops and machines.
        for offset in range(4):
            i = (s["t"] + offset) % 4
            at_control = jnp.sum(jnp.abs(n["agent_pos"][i] - s["control_pos"][0])) <= 1
            switch = (a[i] == 5) & at_control
            brake = (a[i] == 12) & at_control
            n["junction"] = (n["junction"] + switch.astype(jnp.int32)) % 3
            n["belt_on"] = n["belt_on"] != brake
            holding_part = jnp.any(n["part_carrier"] == i)
            holding_product = jnp.any(n["product_carrier"] == i)
            p = jnp.argmax((n["part_carrier"] == i).astype(jnp.int32))
            occupied = jnp.any(jnp.all(n["part_pos"] == n["agent_pos"][i], axis=1) & (n["part_mode"] < 3) & (n["part_carrier"] == -1))
            drop = (a[i] == 6) & holding_part & ~occupied
            machine_here = jnp.all(s["station_pos"] == n["agent_pos"][i], axis=1)
            station = jnp.argmax(machine_here.astype(jnp.int32))
            n["part_carrier"] = n["part_carrier"].at[p].set(jnp.where(drop, -1, n["part_carrier"][p]))
            n["part_mode"] = n["part_mode"].at[p].set(jnp.where(drop, jnp.where(jnp.any(machine_here), 2, 0), n["part_mode"][p]))
            n["station_part"] = n["station_part"].at[station].set(jnp.where(drop & jnp.any(machine_here), p, n["station_part"][station]))
            busy = jnp.any((jnp.arange(6)[:, None] == n["station_part"][None, :]) & (n["station_timer"][None, :] > 0), axis=1)
            near = jnp.sum(jnp.abs(n["part_pos"] - n["agent_pos"][i]), axis=1) <= 1
            available = near & (n["part_carrier"] == -1) & (n["part_mode"] < 3) & ~busy
            p = jnp.argmax(available.astype(jnp.int32))
            take = (a[i] == 6) & ~holding_part & ~holding_product & jnp.any(available)
            n["part_carrier"] = n["part_carrier"].at[p].set(jnp.where(take, i, n["part_carrier"][p]))
            n["part_mode"] = n["part_mode"].at[p].set(jnp.where(take, 1, n["part_mode"][p]))
            n["part_pos"] = n["part_pos"].at[p].set(jnp.where(take, n["agent_pos"][i], n["part_pos"][p]))
            n["station_part"] = jnp.where((n["station_part"] == p) & take, -1, n["station_part"])
            station_d = jnp.sum(jnp.abs(s["station_pos"] - n["agent_pos"][i]), axis=1)
            station = jnp.argmin(station_d)
            cost = jnp.where(a[i] == 7, 2, 1)
            start = ((a[i] == 7) | (a[i] == 8)) & (station_d[station] <= 1) & (n["station_part"][station] >= 0) & (n["station_timer"][station] == 0) & (n["power"] >= cost)
            n["station_timer"] = n["station_timer"].at[station].set(jnp.where(start, jnp.where(a[i] == 7, 3, 2), n["station_timer"][station]))
            n["station_job"] = n["station_job"].at[station].set(jnp.where(start, a[i] - 6, n["station_job"][station]))
            n["power"] = n["power"] - jnp.where(start, cost, 0)
            n["spent_power"] = n["spent_power"] + jnp.where(start, cost, 0)
            started = started.at[station].set(started[station] | start)
            board = jnp.any(jnp.sum(jnp.abs(s["order_pos"] - n["agent_pos"][i]), axis=1) <= 1)
            publish = (a[i] == 9) & board
            n["order_public"] = n["order_public"] | publish
            candidates = near & (n["part_mode"] < 3) & ~n["part_inspected"]
            p = jnp.argmax(candidates.astype(jnp.int32))
            inspect = (a[i] == 9) & ~board & jnp.any(candidates)
            n["part_inspected"] = n["part_inspected"].at[p].set(n["part_inspected"][p] | inspect)
            inspected = inspected.at[p].set(inspected[p] | inspect)
        # ASSEMBLY: first pending order consumes one available chassis and one coil exactly once.
        near_bench = jnp.sum(jnp.abs(n["agent_pos"] - s["bench_pos"][0]), axis=1) <= 1
        workers = near_bench & (a == 10)
        priority = (jnp.arange(4) - s["t"]) % 4
        assembler = jnp.argmin(jnp.where(workers, priority, 4))
        nearby = jnp.sum(jnp.abs(n["part_pos"] - s["bench_pos"][0]), axis=1) <= 1
        available = nearby & (n["part_mode"] == 0) & (n["part_carrier"] == -1)
        chassis = available & (s["part_kind"] == 0)
        coils = available & (s["part_kind"] == 1)
        p0, p1 = jnp.argmax(chassis.astype(jnp.int32)), jnp.argmax(coils.astype(jnp.int32))
        pending = n["product_status"] == 0
        order = jnp.argmax(pending.astype(jnp.int32))
        assemble = jnp.any(workers) & jnp.any(chassis) & jnp.any(coils) & jnp.any(pending)
        correct = (n["part_defect"][p0] == 0) & (n["part_defect"][p1] == 0) & (n["part_color"][p1] == s["order_color"][order])
        good, rejected = assemble & correct, assemble & ~correct
        selected = ((jnp.arange(6) == p0) | (jnp.arange(6) == p1)) & assemble
        n["part_mode"] = jnp.where(selected, jnp.where(correct, 3, 5), n["part_mode"])
        n["part_order"] = jnp.where(selected, order, n["part_order"])
        n["product_status"] = n["product_status"].at[order].set(jnp.where(assemble, jnp.where(correct, 1, 3), n["product_status"][order]))
        n["last_assembled"] = (jnp.arange(3) == order) & good
        n["last_rejected"] = (jnp.arange(3) == order) & rejected
        # PRODUCT_CARRYING: one carried part or product per worker; delivery uses the real dock.
        for offset in range(4):
            i = (s["t"] + offset) % 4
            holding = jnp.any(n["part_carrier"] == i) | jnp.any(n["product_carrier"] == i)
            near = jnp.sum(jnp.abs(n["product_pos"] - n["agent_pos"][i]), axis=1) <= 1
            available_products = near & (n["product_status"] == 1) & (n["product_carrier"] == -1)
            product = jnp.argmax(available_products.astype(jnp.int32))
            take = (a[i] == 11) & ~holding & jnp.any(available_products)
            n["product_carrier"] = n["product_carrier"].at[product].set(jnp.where(take, i, n["product_carrier"][product]))
            n["product_pos"] = n["product_pos"].at[product].set(jnp.where(take, n["agent_pos"][i], n["product_pos"][product]))
        carriers = n["product_carrier"]
        shipped = (carriers >= 0) & (n["product_status"] == 1) & jnp.all(n["product_pos"] == s["dock_pos"][0], axis=1)
        n["product_status"] = jnp.where(shipped, 2, n["product_status"])
        n["product_carrier"] = jnp.where(shipped, -1, carriers)
        constituent_shipped = (n["part_order"] >= 0) & shipped[jnp.clip(n["part_order"], 0, 2)] & (n["part_mode"] == 3)
        n["part_mode"] = jnp.where(constituent_shipped, 4, n["part_mode"])
        # CONVEYOR_QUEUE: one-cell buffers, pre-move occupancy and cyclic part tie-breaking.
        n = self.refresh(n)
        on_node = jnp.all(n["part_pos"][:, None, :] == s["belt_pos"][None, :, :], axis=2)
        node = jnp.argmax(on_node.astype(jnp.int32), axis=1)
        destination = n["belt_next"][node]
        target = s["belt_pos"][destination]
        moving = jnp.any(on_node, axis=1) & (n["part_mode"] == 0) & (destination != node) & n["belt_on"][0] & (n["t"] % 2 == 0)
        occupied = jnp.any(jnp.all(target[:, None, :] == n["part_pos"][None, :, :], axis=2) & (n["part_mode"][None, :] < 3) & (n["part_carrier"][None, :] == -1) & ~jnp.eye(6, dtype=jnp.bool_), axis=1)
        moving = moving & ~occupied
        part_priority = (jnp.arange(6) - n["t"]) % 6
        same = jnp.all(target[:, None, :] == target[None, :, :], axis=2)
        earlier = jnp.any(same & moving[None, :] & (part_priority[None, :] < part_priority[:, None]), axis=1)
        moving = moving & ~earlier
        n["part_pos"] = jnp.where(moving[:, None], target, n["part_pos"])
        for station in range(2):
            arriving = moving & jnp.all(target == s["station_pos"][station], axis=1)
            p = jnp.argmax(arriving.astype(jnp.int32))
            n["station_part"] = n["station_part"].at[station].set(jnp.where(jnp.any(arriving), p, n["station_part"][station]))
            n["part_mode"] = jnp.where(arriving, 2, n["part_mode"])
        n["last_belt"], n["last_started"], n["last_finished"] = moving, started, finished
        n["last_inspected"], n["last_shipped"] = inspected, shipped
        delivery_credit = jnp.sum(shipped[:, None] & (carriers[:, None] == jnp.arange(4)[None, :]), axis=0)
        timely = shipped & (n["t"] <= s["order_deadline"])
        owner_credit = jnp.sum(timely[:, None] & (((jnp.arange(3)[:, None] + 1) % 4) == jnp.arange(4)[None, :]), axis=0)
        rewards = 0.15 * good - 0.4 * rejected + (0.4 * good - 0.6 * rejected) * (jnp.arange(4) == assembler) + 0.5 * jnp.sum(shipped) + delivery_credit + owner_credit
        n["terminated"] = jnp.all(n["product_status"] >= 2)
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        return self.refresh(n), rewards.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions, 0, 12)
        valid = self.legal_actions(state)[jnp.arange(4), safe] & (safe == actions)
        n, rewards = self.advance(state, jnp.where(valid, safe, 0))
        # ABSORBING_TERMINAL: terminal observations and native state remain unchanged.
        n = {k: jnp.where(state["done"], state[k], n[k]) for k in state}
        rewards = jnp.where(state["done"], jnp.zeros((4,), dtype=jnp.float32), rewards)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, jnp.full((4,), n["done"], dtype=jnp.bool_), info
