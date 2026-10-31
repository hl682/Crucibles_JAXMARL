"""Counterweight Yard: shared ballast lifts, bolt-on bridge cells and falling cargo."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 141
    num_actions = 13
    max_steps = 160

    def solid(self, s, positions):
        safe = jnp.clip(positions, 0, 8)
        terrain = s["map"][safe[:, 1], safe[:, 0]] == 1
        anchored = jnp.any(jnp.all(positions[:, None, :] == s["block_pos"][None, :, :], axis=2)
                           & s["block_active"][None, :] & s["block_anchored"][None, :], axis=1)
        platforms = jnp.any(jnp.all(positions[:, None, :] == s["lift_pos"][None, :, :], axis=2), axis=1)
        return terrain | anchored | platforms

    def loads(self, s):
        tops = s["lift_pos"] - jnp.array([0, 1], dtype=jnp.int32)
        riders = jnp.all(s["agent_pos"][:, None, :] == tops[None, :, :], axis=2)
        cargo = jnp.all(s["block_pos"][:, None, :] == tops[None, :, :], axis=2)
        load = (jnp.sum(riders.astype(jnp.int32), axis=0, dtype=jnp.int32)
                + jnp.sum(cargo.astype(jnp.int32) * (s["block_weight"] * s["block_active"].astype(jnp.int32)
                                                   * (~s["block_anchored"]).astype(jnp.int32))[:, None],
                          axis=0, dtype=jnp.int32))
        on_pad = jnp.all(s["block_pos"][:, None, :] == s["ballast_pos"][None, :, :], axis=2)
        fixed_mass = s["block_weight"] * (s["block_active"] & (s["block_carrier"] < 0) & ~s["block_anchored"]).astype(jnp.int32)
        human_pad = jnp.all(s["agent_pos"][:, None, :] == s["ballast_pos"][None, :, :], axis=2)
        ballast = (jnp.sum(on_pad.astype(jnp.int32) * fixed_mass[:, None], axis=0, dtype=jnp.int32)
                   + 2 * jnp.sum(human_pad.astype(jnp.int32) * s["braced"].astype(jnp.int32)[:, None], axis=0, dtype=jnp.int32))
        return load, ballast[::-1]

    def refresh(self, s):
        n = dict(s)
        held = s["block_carrier"][None, :] == jnp.arange(3)[:, None]
        n["carrying"] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        n["lift_load"], n["lift_support"] = self.loads(n)
        return n

    def reset(self, key):
        tile = jnp.zeros((9, 9), dtype=jnp.int32)
        tile = tile.at[8, :].set(1)
        tile = tile.at[4, 0].set(1)
        tile = tile.at[4, 8].set(1)
        tile = tile.at[:8, 1].set(2)
        tile = tile.at[:8, 7].set(2)
        tile = tile.at[5:8, 4].set(2)
        s = {
            "t": jnp.array(0, dtype=jnp.int32),
            "done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "map": tile,
            "agent_pos": jnp.array([[1, 7], [7, 7], [4, 7]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_vis": jnp.ones((3, 3), dtype=jnp.bool_),
            "carrying": jnp.full((3,), -1, dtype=jnp.int32),
            "stun": jnp.zeros((3,), dtype=jnp.int32),
            "braced": jnp.zeros((3,), dtype=jnp.bool_),
            "block_pos": jnp.array([[1, 7], [2, 7], [6, 7], [7, 7], [0, 7], [8, 7], [4, 7], [3, 7], [5, 7]], dtype=jnp.int32),
            "block_active": jnp.ones((9,), dtype=jnp.bool_),
            "block_vis": jnp.ones((3, 9), dtype=jnp.bool_),
            "block_weight": jnp.array([1, 1, 1, 1, 1, 1, 1, 2, 2], dtype=jnp.int32),
            "block_carrier": jnp.full((9,), -1, dtype=jnp.int32),
            "block_anchored": jnp.zeros((9,), dtype=jnp.bool_),
            "lift_pos": jnp.array([[2, 8], [6, 8]], dtype=jnp.int32),
            "lift_active": jnp.ones((2,), dtype=jnp.bool_),
            "lift_vis": jnp.ones((3, 2), dtype=jnp.bool_),
            "lift_load": jnp.zeros((2,), dtype=jnp.int32),
            "lift_support": jnp.zeros((2,), dtype=jnp.int32),
            "ballast_pos": jnp.array([[3, 7], [5, 7]], dtype=jnp.int32),
            "ballast_active": jnp.ones((2,), dtype=jnp.bool_),
            "ballast_vis": jnp.ones((3, 2), dtype=jnp.bool_),
            "station_pos": jnp.array([[0, 3], [8, 3], [4, 3]], dtype=jnp.int32),
            "station_active": jnp.ones((3,), dtype=jnp.bool_),
            "station_vis": jnp.ones((3, 3), dtype=jnp.bool_),
            "station_count": jnp.zeros((3,), dtype=jnp.int32),
            "station_required": jnp.array([1, 1, 2], dtype=jnp.int32),
            "crank_pos": jnp.array([[4, 7]], dtype=jnp.int32),
            "crank_active": jnp.ones((1,), dtype=jnp.bool_),
            "crank_vis": jnp.ones((3, 1), dtype=jnp.bool_),
            "brake": jnp.zeros((1,), dtype=jnp.int32),
            "lift_up_event": jnp.zeros((2,), dtype=jnp.bool_),
            "lift_down_event": jnp.zeros((2,), dtype=jnp.bool_),
            "lift_drop_event": jnp.zeros((2,), dtype=jnp.bool_),
            "block_fall_event": jnp.zeros((9,), dtype=jnp.bool_),
            "impact_event": jnp.zeros((3,), dtype=jnp.bool_),
            "build_event": jnp.zeros((9,), dtype=jnp.bool_),
            "reclaim_event": jnp.zeros((9,), dtype=jnp.bool_),
            "delivery_event": jnp.zeros((3,), dtype=jnp.bool_),
            "brake_event": jnp.zeros((1,), dtype=jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((
            jnp.stack((s["t"], s["done"].astype(jnp.int32),
                       s["terminated"].astype(jnp.int32), s["truncated"].astype(jnp.int32))),
            s["agent_pos"].reshape(-1), s["carrying"], s["stun"],
            s["block_pos"].reshape(-1), s["block_active"].astype(jnp.int32),
            s["block_weight"], s["block_carrier"], s["block_anchored"].astype(jnp.int32),
            s["lift_pos"].reshape(-1), s["lift_load"], s["lift_support"],
            s["ballast_pos"].reshape(-1), s["station_pos"].reshape(-1),
            s["station_count"], s["station_required"], s["crank_pos"].reshape(-1), s["brake"],
            s["lift_up_event"].astype(jnp.int32), s["lift_down_event"].astype(jnp.int32), s["lift_drop_event"].astype(jnp.int32),
            s["block_fall_event"].astype(jnp.int32), s["impact_event"].astype(jnp.int32),
            s["build_event"].astype(jnp.int32), s["reclaim_event"].astype(jnp.int32),
            s["delivery_event"].astype(jnp.int32), s["braced"].astype(jnp.int32),
            s["brake_event"].astype(jnp.int32),
        ))
        return jnp.concatenate((jnp.arange(3, dtype=jnp.int32)[:, None],
                                jnp.broadcast_to(public, (3, 140))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        able = (s["stun"] == 0) & ~s["done"]
        return able[:, None] | (jnp.arange(13)[None, :] == 0)

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 12)
        valid = (actions >= 0) & (actions < 13) & self.legal_actions(s)[jnp.arange(3), clipped]
        a = jnp.where(valid, clipped, 0)
        able = s["stun"] == 0
        n["stun"] = jnp.maximum(s["stun"] - 1, 0)
        n["braced"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["brake"] = jnp.maximum(s["brake"] - 1, 0)
        n["lift_up_event"] = jnp.zeros((2,), dtype=jnp.bool_)
        n["lift_down_event"] = jnp.zeros((2,), dtype=jnp.bool_)
        n["lift_drop_event"] = jnp.zeros((2,), dtype=jnp.bool_)
        n["block_fall_event"] = jnp.zeros((9,), dtype=jnp.bool_)
        n["impact_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["build_event"] = jnp.zeros((9,), dtype=jnp.bool_)
        n["reclaim_event"] = jnp.zeros((9,), dtype=jnp.bool_)
        n["delivery_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["brake_event"] = jnp.zeros((1,), dtype=jnp.bool_)
        moves = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)

        # Workers climb ladders only when empty-handed; cargo rises on the shared lifts.
        def move(k, pos):
            i = (k + s["t"]) % 3
            target = pos[i] + moves[a[i]]
            inside = jnp.all(target >= 0) & (target[0] <= 8) & (target[1] <= 7)
            safe = jnp.minimum(jnp.maximum(target, 0), jnp.array([8, 7], dtype=jnp.int32))
            here_ladder = s["map"][pos[i, 1], pos[i, 0]] == 2
            there_ladder = s["map"][safe[1], safe[0]] == 2
            climb = (a[i] != 1) | ((s["carrying"][i] < 0) & (here_ladder | there_ladder))
            occupied = jnp.any(jnp.all(pos == target, axis=1) & (jnp.arange(3) != i))
            solid = self.solid(n, target[None, :])[0]
            ok = able[i] & inside & climb & ~occupied & ~solid
            return pos.at[i].set(jnp.where(ok, target, pos[i]))

        n["agent_pos"] = jax.lax.fori_loop(0, 3, move, s["agent_pos"])
        n["block_pos"] = jnp.where((n["block_carrier"] >= 0)[:, None],
                                   n["agent_pos"][jnp.maximum(n["block_carrier"], 0)], n["block_pos"])
        credit = jnp.zeros((3,), dtype=jnp.int32)
        bridge = jnp.array([[3, 4], [4, 4], [5, 4]], dtype=jnp.int32)

        def handle(k, carry):
            q, delivery_credit = carry
            i = (k + s["t"]) % 3
            held = (q["block_carrier"] == i) & q["block_active"]
            has = jnp.any(held)
            good = jnp.argmax(held)
            stations = jnp.all(q["station_pos"] == q["agent_pos"][i], axis=1) & (q["station_count"] < q["station_required"])
            station = jnp.argmax(stations)
            deliver = able[i] & (a[i] == 5) & has & jnp.any(stations)
            drop = able[i] & (a[i] == 5) & has & ~deliver
            q["block_active"] = q["block_active"].at[good].set(q["block_active"][good] & ~deliver)
            q["block_carrier"] = q["block_carrier"].at[good].set(jnp.where(deliver | drop, -1, q["block_carrier"][good]))
            q["station_count"] = q["station_count"].at[station].add(deliver.astype(jnp.int32))
            q["delivery_event"] = q["delivery_event"].at[station].set(q["delivery_event"][station] | deliver)
            delivery_credit = delivery_credit.at[i].add(deliver.astype(jnp.int32))
            loose = (q["block_active"] & (q["block_carrier"] < 0) & ~q["block_anchored"]
                     & jnp.all(q["block_pos"] == q["agent_pos"][i], axis=1))
            pickup = able[i] & (a[i] == 5) & ~has & jnp.any(loose)
            target = jnp.argmax(loose)
            q["block_carrier"] = q["block_carrier"].at[target].set(jnp.where(pickup, i, q["block_carrier"][target]))
            dx = jnp.abs(bridge[:, 0] - q["agent_pos"][i, 0])
            adjacent_support = self.solid(q, bridge + jnp.array([-1, 0], dtype=jnp.int32)) | self.solid(q, bridge + jnp.array([1, 0], dtype=jnp.int32))
            eligible = (dx <= 1) & (q["agent_pos"][i, 1] == 3) & ~self.solid(q, bridge) & adjacent_support
            socket = jnp.argmin(jnp.where(eligible, dx, 99))
            build = able[i] & (a[i] == 6) & has & jnp.any(eligible)
            q["block_pos"] = q["block_pos"].at[good].set(jnp.where(build, bridge[socket], q["block_pos"][good]))
            q["block_carrier"] = q["block_carrier"].at[good].set(jnp.where(build, -1, q["block_carrier"][good]))
            q["block_anchored"] = q["block_anchored"].at[good].set(q["block_anchored"][good] | build)
            q["build_event"] = q["build_event"].at[good].set(q["build_event"][good] | build)
            distance = jnp.sum(jnp.abs(q["block_pos"] - q["agent_pos"][i]), axis=1, dtype=jnp.int32)
            bolts = q["block_active"] & q["block_anchored"] & (distance <= 2)
            anchor = jnp.argmin(jnp.where(bolts, distance, 99))
            reclaim = able[i] & (a[i] == 10) & ~has & jnp.any(bolts)
            q["block_anchored"] = q["block_anchored"].at[anchor].set(q["block_anchored"][anchor] & ~reclaim)
            q["block_carrier"] = q["block_carrier"].at[anchor].set(jnp.where(reclaim, i, q["block_carrier"][anchor]))
            q["block_pos"] = q["block_pos"].at[anchor].set(jnp.where(reclaim, q["agent_pos"][i], q["block_pos"][anchor]))
            q["reclaim_event"] = q["reclaim_event"].at[anchor].set(q["reclaim_event"][anchor] | reclaim)
            return q, delivery_credit

        n, credit = jax.lax.fori_loop(0, 3, handle, (n, credit))
        at_crank = jnp.all(n["agent_pos"] == n["crank_pos"][0], axis=1)
        at_ballast = jnp.any(jnp.all(n["agent_pos"][:, None, :] == n["ballast_pos"][None, :, :], axis=2), axis=1)
        n["braced"] = able & (a == 9) & at_ballast
        braking = jnp.any(able & (a == 9) & at_crank)
        n["brake"] = jnp.where(braking, jnp.array([2], dtype=jnp.int32), n["brake"])
        n["brake_event"] = jnp.array([braking], dtype=jnp.bool_)
        load, support = self.loads(n)
        winding = jnp.stack((jnp.any(able & (a == 7) & at_crank),
                             jnp.any(able & (a == 8) & at_crank)))
        lowering = jnp.stack((jnp.any(able & (a == 11) & at_crank),
                              jnp.any(able & (a == 12) & at_crank)))
        free = n["brake"][0] == 0
        up = winding & (support >= load) & (support > 0) & (n["lift_pos"][:, 1] > 4) & free
        controlled_down = lowering & (n["lift_pos"][:, 1] < 8) & free
        down = (support < load) & (n["lift_pos"][:, 1] < 8) & free & ~controlled_down
        shift = (down | controlled_down).astype(jnp.int32) - up.astype(jnp.int32)
        old_tops = n["lift_pos"] - jnp.array([0, 1], dtype=jnp.int32)
        riders = jnp.all(n["agent_pos"][:, None, :] == old_tops[None, :, :], axis=2)
        n["agent_pos"] = n["agent_pos"].at[:, 1].add(jnp.sum(riders.astype(jnp.int32) * shift[None, :], axis=1, dtype=jnp.int32))
        cargo_riders = (jnp.all(n["block_pos"][:, None, :] == old_tops[None, :, :], axis=2)
                         & (n["block_carrier"] < 0)[:, None] & n["block_active"][:, None] & ~n["block_anchored"][:, None])
        n["block_pos"] = n["block_pos"].at[:, 1].add(jnp.sum(cargo_riders.astype(jnp.int32) * shift[None, :], axis=1, dtype=jnp.int32))
        n["lift_pos"] = n["lift_pos"].at[:, 1].add(shift)
        n["lift_up_event"] = up
        n["lift_down_event"] = controlled_down
        n["lift_drop_event"] = down
        n["impact_event"] = jnp.any(riders & down[None, :], axis=1)
        n["block_pos"] = jnp.where((n["block_carrier"] >= 0)[:, None],
                                   n["agent_pos"][jnp.maximum(n["block_carrier"], 0)], n["block_pos"])

        # Each loose block falls at most one cell in this tick; anchored blocks are bolted.
        def gravity(k, q):
            g = (k + s["t"]) % 9
            below = q["block_pos"][g] + jnp.array([0, 1], dtype=jnp.int32)
            stacked = jnp.any(jnp.all(q["block_pos"] == below, axis=1) & q["block_active"] & (q["block_carrier"] < 0))
            supported = self.solid(q, below[None, :])[0] | stacked
            fall = q["block_active"][g] & (q["block_carrier"][g] < 0) & ~q["block_anchored"][g] & (q["block_pos"][g, 1] < 7) & ~supported
            q["block_pos"] = q["block_pos"].at[g].set(jnp.where(fall, below, q["block_pos"][g]))
            q["block_fall_event"] = q["block_fall_event"].at[g].set(fall)
            hit = fall & jnp.all(q["agent_pos"] == below, axis=1)
            q["impact_event"] = q["impact_event"] | hit
            return q

        n = jax.lax.fori_loop(0, 9, gravity, n)
        below_agents = n["agent_pos"] + jnp.array([0, 1], dtype=jnp.int32)
        on_ladder = n["map"][n["agent_pos"][:, 1], n["agent_pos"][:, 0]] == 2
        fall_agent = (n["agent_pos"][:, 1] < 7) & ~on_ladder & ~self.solid(n, below_agents)
        n["agent_pos"] = n["agent_pos"].at[:, 1].add(fall_agent.astype(jnp.int32))
        n["stun"] = jnp.maximum(n["stun"], 2 * n["impact_event"].astype(jnp.int32))
        n["block_pos"] = jnp.where((n["block_carrier"] >= 0)[:, None],
                                   n["agent_pos"][jnp.maximum(n["block_carrier"], 0)], n["block_pos"])
        n = self.refresh(n)
        success = jnp.all(n["station_count"] == n["station_required"])
        n["t"] = s["t"] + 1
        n["terminated"] = success
        n["truncated"] = (n["t"] >= self.max_steps) & ~success
        n["done"] = n["terminated"] | n["truncated"]
        station_gain = n["station_count"] - s["station_count"]
        deliveries = jnp.sum(station_gain, dtype=jnp.int32)
        rewards = (-jnp.ones((3,), dtype=jnp.int32) + 4 * station_gain + deliveries + credit
                   - 2 * n["impact_event"].astype(jnp.int32) + 3 * success.astype(jnp.int32))
        return n, rewards.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        proposed, rewards = self.advance(state, actions)
        next_state = jax.tree.map(lambda old, new: jnp.where(state["done"], old, new), state, proposed)
        rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        dones = jnp.full((3,), next_state["done"], dtype=jnp.bool_)
        info = {"done_all": next_state["done"], "terminated": next_state["terminated"],
                "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, dones, info
