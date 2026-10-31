"""Lightkeepers: shared ray geometry, scarce charge and observable shadow crossings."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 316
    num_actions = 13
    max_steps = 128

    def rays(self, s):
        directions = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        turns = jnp.array([[1, 0, 3, 2], [3, 2, 1, 0]], dtype=jnp.int32)
        position = s["lamp_pos"]
        direction = s["lamp_direction"]
        travelling = s["lamp_on"] & (s["lamp_energy"] > 0)
        visited = jnp.zeros((2, 77), dtype=jnp.bool_)
        # RAY_GEOMETRY: twelve bounded segments per lamp; closed shutters absorb rays.
        for segment in range(12):
            position = position + directions[direction]
            inside = jnp.all((position >= 0) & (position <= jnp.array([10, 6])), axis=1)
            q = jnp.clip(position, jnp.array([0, 0]), jnp.array([10, 6]))
            blocked = jnp.any(jnp.all(q[:, None, :] == s["shutter_pos"][None, :, :], axis=2) & ~s["shutter_open"][None, :], axis=1)
            travelling = travelling & inside & ~blocked
            index = q[:, 1] * 11 + q[:, 0]
            visited = visited.at[jnp.arange(2), index].set(visited[jnp.arange(2), index] | travelling)
            hits = jnp.all(q[:, None, :] == s["mirror_pos"][None, :, :], axis=2) & (s["mirror_carrier"][None, :] == -1)
            mirror = jnp.argmax(hits.astype(jnp.int32), axis=1)
            reflected = turns[s["mirror_orientation"][mirror], direction]
            direction = jnp.where(jnp.any(hits, axis=1) & travelling, reflected, direction)
        return jnp.sum(visited.astype(jnp.int32), axis=0)

    def refresh(self, s):
        s = dict(s)
        s["shadow"] = s["beam"] == 0
        d = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        agent_lit = s["beam"][s["agent_pos"][:, 1] * 11 + s["agent_pos"][:, 0]] > 0
        s["agent_vis"] = (d <= 2) | agent_lit[None, :] | jnp.eye(3, dtype=jnp.bool_)
        mirror_d = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["mirror_pos"][None, :, :]), axis=2)
        mirror_lit = s["beam"][s["mirror_pos"][:, 1] * 11 + s["mirror_pos"][:, 0]] > 0
        s["mirror_vis"] = (mirror_d <= 2) | mirror_lit[None, :] | (s["mirror_carrier"][None, :] == jnp.arange(3)[:, None])
        cache_d = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["cache_pos"][None, :, :]), axis=2)
        cache_lit = s["beam"][s["cache_pos"][:, 1] * 11 + s["cache_pos"][:, 0]] > 0
        s["cache_vis"] = (cache_d <= 2) | cache_lit[None, :] | jnp.eye(3, dtype=jnp.bool_)
        s["cache_active"] = s["cache_status"] < 2
        s["lamp_vis"] = jnp.ones((3, 2), dtype=jnp.bool_)
        s["shutter_vis"] = jnp.ones((3, 2), dtype=jnp.bool_)
        s["cell_vis"] = jnp.ones((3, 77), dtype=jnp.bool_)
        s["socket_vis"] = jnp.ones((3, 1), dtype=jnp.bool_)
        return s

    def reset(self, key):
        tile = jnp.zeros((7, 11), dtype=jnp.int32)
        tile = tile.at[:, 4].set(1).at[:, 7].set(1).at[3, 5].set(2)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tile": tile,
            "agent_pos": jnp.array([[1, 2], [9, 4], [2, 4]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "pack": jnp.full((3,), 3, dtype=jnp.int32),
            "exposure": jnp.zeros((3,), dtype=jnp.int32),
            "stun": jnp.zeros((3,), dtype=jnp.int32),
            "lamp_pos": jnp.array([[1, 1], [9, 5]], dtype=jnp.int32),
            "lamp_active": jnp.ones((2,), dtype=jnp.bool_),
            "lamp_direction": jnp.array([1, 3], dtype=jnp.int32),
            "lamp_on": jnp.ones((2,), dtype=jnp.bool_),
            "lamp_energy": jnp.full((2,), 12, dtype=jnp.int32),
            "shutter_pos": jnp.array([[4, 1], [7, 5]], dtype=jnp.int32),
            "shutter_active": jnp.ones((2,), dtype=jnp.bool_),
            "shutter_open": jnp.ones((2,), dtype=jnp.bool_),
            "mirror_pos": jnp.array([[4, 4], [7, 2]], dtype=jnp.int32),
            "mirror_active": jnp.ones((2,), dtype=jnp.bool_),
            "mirror_orientation": jnp.array([0, 1], dtype=jnp.int32),
            "mirror_carrier": jnp.full((2,), -1, dtype=jnp.int32),
            "cache_pos": jnp.array([[9, 1], [1, 5], [9, 3]], dtype=jnp.int32),
            "cache_code": jax.random.randint(key, (3,), 0, 2, dtype=jnp.int32),
            "cache_status": jnp.zeros((3,), dtype=jnp.int32),
            "cache_carrier": jnp.full((3,), -1, dtype=jnp.int32),
            "cache_deadline": jnp.array([32, 48, 64], dtype=jnp.int32),
            "socket_pos": jnp.array([[5, 3]], dtype=jnp.int32),
            "socket_active": jnp.ones((1,), dtype=jnp.bool_),
            "socket_stock": jnp.array([18], dtype=jnp.int32),
            "spent_energy": jnp.array(0, dtype=jnp.int32),
            "cell_pos": jnp.stack((jnp.tile(jnp.arange(11), 7), jnp.repeat(jnp.arange(7), 11)), axis=1).astype(jnp.int32),
            "cell_active": jnp.ones((77,), dtype=jnp.bool_),
            "last_capture": jnp.zeros((3,), dtype=jnp.bool_),
            "last_collected": jnp.zeros((3,), dtype=jnp.bool_),
            "last_delivered": jnp.zeros((3,), dtype=jnp.bool_),
        }
        s["beam"] = self.rays(s)
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate(tuple(s[k].reshape(-1).astype(jnp.float32) for k in (
            "t", "done", "terminated", "truncated", "tile", "lamp_pos", "lamp_direction",
            "lamp_on", "lamp_energy", "shutter_pos", "shutter_open", "socket_pos", "socket_stock",
            "spent_energy", "beam", "shadow", "cache_deadline", "last_capture", "last_collected", "last_delivered")))
        agents = jnp.concatenate((s["agent_pos"], s["pack"][:, None], s["exposure"][:, None], s["stun"][:, None]), axis=1)
        mirrors = jnp.concatenate((s["mirror_pos"], s["mirror_orientation"][:, None], s["mirror_carrier"][:, None]), axis=1)
        caches = jnp.concatenate((s["cache_pos"], s["cache_code"][:, None], s["cache_status"][:, None], s["cache_carrier"][:, None]), axis=1)
        agent_obs = jnp.where(s["agent_vis"][:, :, None], agents[None, :, :], -1).reshape(3, 15)
        mirror_obs = jnp.where(s["mirror_vis"][:, :, None], mirrors[None, :, :], -1).reshape(3, 8)
        cache_obs = jnp.where(s["cache_vis"][:, :, None], caches[None, :, :], -1).reshape(3, 15)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, public.shape[0])),
                                s["agent_vis"].astype(jnp.float32), agent_obs.astype(jnp.float32),
                                s["mirror_vis"].astype(jnp.float32), mirror_obs.astype(jnp.float32),
                                s["cache_vis"].astype(jnp.float32), cache_obs.astype(jnp.float32)), axis=1)

    def legal_actions(self, s):
        live = ~s["done"] & (s["exposure"] < 3) & (s["stun"] == 0)
        return (jnp.ones((3, 13), dtype=jnp.bool_) & live[:, None]).at[:, 0].set(True)

    def move(self, s, a):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)]
        inside = jnp.all((target >= 0) & (target <= jnp.array([10, 6])), axis=1)
        q = jnp.clip(target, jnp.array([0, 0]), jnp.array([10, 6]))
        closed = jnp.any(jnp.all(q[:, None, :] == s["shutter_pos"][None, :, :], axis=2) & ~s["shutter_open"][None, :], axis=1)
        occupied = jnp.any(jnp.all(q[:, None, :] == s["agent_pos"][None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        return jnp.where((inside & ~closed & ~occupied & ~conflict)[:, None], q, s["agent_pos"])

    def advance(self, s, a):
        n = dict(s)
        n["t"] = s["t"] + 1
        n["agent_pos"] = self.move(s, a)
        n["stun"] = jnp.maximum(s["stun"] - 1, 0)
        n["mirror_pos"] = jnp.where((s["mirror_carrier"] >= 0)[:, None], n["agent_pos"][jnp.clip(s["mirror_carrier"], 0, 2)], s["mirror_pos"])
        n["cache_pos"] = jnp.where((s["cache_carrier"] >= 0)[:, None], n["agent_pos"][jnp.clip(s["cache_carrier"], 0, 2)], s["cache_pos"])
        # DEVICE_INTERACTIONS: cyclic actor priority; toggles compose and resources transfer once.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            lamp_d = jnp.sum(jnp.abs(s["lamp_pos"] - n["agent_pos"][i]), axis=1)
            l = jnp.argmin(lamp_d)
            near_lamp = lamp_d[l] <= 1
            rotate = (a[i] == 5) & near_lamp
            toggle = (a[i] == 6) & near_lamp
            n["lamp_direction"] = n["lamp_direction"].at[l].set((n["lamp_direction"][l] + rotate.astype(jnp.int32)) % 4)
            n["lamp_on"] = n["lamp_on"].at[l].set(n["lamp_on"][l] != toggle)
            shutter_d = jnp.sum(jnp.abs(s["shutter_pos"] - n["agent_pos"][i]), axis=1)
            g = jnp.argmin(shutter_d)
            flip_gate = (a[i] == 7) & (shutter_d[g] <= 1)
            n["shutter_open"] = n["shutter_open"].at[g].set(n["shutter_open"][g] != flip_gate)
            holding = jnp.any(n["mirror_carrier"] == i)
            held = jnp.argmax((n["mirror_carrier"] == i).astype(jnp.int32))
            occupied = jnp.any(jnp.all(n["mirror_pos"] == n["agent_pos"][i], axis=1) & (n["mirror_carrier"] == -1))
            drop = (a[i] == 8) & holding & ~occupied
            n["mirror_carrier"] = n["mirror_carrier"].at[held].set(jnp.where(drop, -1, n["mirror_carrier"][held]))
            mirror_d = jnp.sum(jnp.abs(n["mirror_pos"] - n["agent_pos"][i]), axis=1)
            eligible = (mirror_d <= 1) & (n["mirror_carrier"] == -1)
            m = jnp.argmax(eligible.astype(jnp.int32))
            pickup = (a[i] == 8) & ~holding & jnp.any(eligible)
            n["mirror_carrier"] = n["mirror_carrier"].at[m].set(jnp.where(pickup, i, n["mirror_carrier"][m]))
            n["mirror_pos"] = n["mirror_pos"].at[m].set(jnp.where(pickup, n["agent_pos"][i], n["mirror_pos"][m]))
            m = jnp.where(holding, held, m)
            flip_mirror = (a[i] == 9) & (holding | jnp.any(eligible))
            n["mirror_orientation"] = n["mirror_orientation"].at[m].set(jnp.where(flip_mirror, 1 - n["mirror_orientation"][m], n["mirror_orientation"][m]))
            charge = (a[i] == 10) & near_lamp & (n["pack"][i] > 0) & (n["lamp_energy"][l] < 12)
            refill = (a[i] == 10) & ~near_lamp & (jnp.sum(jnp.abs(n["agent_pos"][i] - s["socket_pos"][0])) <= 1) & (n["socket_stock"][0] > 0) & (n["pack"][i] < 6)
            n["lamp_energy"] = n["lamp_energy"].at[l].add(charge.astype(jnp.int32))
            n["pack"] = n["pack"].at[i].add(refill.astype(jnp.int32) - charge.astype(jnp.int32))
            n["socket_stock"] = n["socket_stock"].at[0].add(-refill.astype(jnp.int32))
        # EMISSION: the stored beam is the actual just-emitted light, before paying its charge.
        n["beam"] = self.rays(n)
        consumed = (n["lamp_on"] & (n["lamp_energy"] > 0)).astype(jnp.int32)
        n["lamp_energy"] = n["lamp_energy"] - consumed
        n["spent_energy"] = s["spent_energy"] + jnp.sum(consumed)
        collected = jnp.zeros((3,), dtype=jnp.bool_)
        # INFORMATION_AND_PICKUP: code entry requires real illumination and free hands.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            close = jnp.sum(jnp.abs(n["cache_pos"] - n["agent_pos"][i]), axis=1) <= 1
            lit = n["beam"][n["cache_pos"][:, 1] * 11 + n["cache_pos"][:, 0]] > 0
            eligible = close & lit & (n["cache_status"] == 0) & (s["cache_code"] == a[i] - 11)
            c = jnp.argmax(eligible.astype(jnp.int32))
            take = ((a[i] == 11) | (a[i] == 12)) & ~jnp.any(n["cache_carrier"] == i) & jnp.any(eligible)
            n["cache_carrier"] = n["cache_carrier"].at[c].set(jnp.where(take, i, n["cache_carrier"][c]))
            n["cache_status"] = n["cache_status"].at[c].set(jnp.where(take, 1, n["cache_status"][c]))
            n["cache_pos"] = n["cache_pos"].at[c].set(jnp.where(take, n["agent_pos"][i], n["cache_pos"][c]))
            collected = collected.at[c].set(collected[c] | take)
        carrying = n["cache_carrier"] >= 0
        carriers = n["cache_carrier"]
        delivered = carrying & jnp.all(n["cache_pos"] == s["socket_pos"][0], axis=1)
        n["cache_status"] = jnp.where(delivered, 2, n["cache_status"])
        n["cache_carrier"] = jnp.where(delivered, -1, carriers)
        # WATCHED_CROSSINGS: partners can darken the cell to prevent repeated exposure.
        p = n["agent_pos"]
        capture = (s["tile"][p[:, 1], p[:, 0]] == 1) & (n["beam"][p[:, 1] * 11 + p[:, 0]] > 0) & (s["exposure"] < 3)
        n["exposure"] = s["exposure"] + capture.astype(jnp.int32)
        n["stun"] = jnp.where(capture, 2, n["stun"])
        dropped = (n["cache_carrier"] >= 0) & capture[jnp.clip(n["cache_carrier"], 0, 2)]
        n["cache_status"] = jnp.where(dropped, 0, n["cache_status"])
        n["cache_carrier"] = jnp.where(dropped, -1, n["cache_carrier"])
        n["last_capture"], n["last_collected"], n["last_delivered"] = capture, collected, delivered
        carrier_credit = jnp.sum(delivered[:, None] & (carriers[:, None] == jnp.arange(3)[None, :]), axis=0)
        timely = delivered & (n["t"] <= s["cache_deadline"])
        rewards = 0.4 * jnp.sum(delivered) + 1.5 * carrier_credit + timely.astype(jnp.float32) - 0.2 * capture - 0.05 * jnp.sum(capture) - 0.01
        n["terminated"] = jnp.all(n["cache_status"] == 2) | jnp.all(n["exposure"] == 3)
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        return self.refresh(n), rewards.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions, 0, 12)
        valid = self.legal_actions(state)[jnp.arange(3), safe] & (safe == actions)
        n, rewards = self.advance(state, jnp.where(valid, safe, 0))
        # ABSORBING_TERMINAL: exact state retention, zero reward, original cause flags.
        n = {k: jnp.where(state["done"], state[k], n[k]) for k in state}
        rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, jnp.full((3,), n["done"], dtype=jnp.bool_), info
