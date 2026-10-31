"""Winter Road Confederation. Original authored rules; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 192
    observation_size = 224
    num_actions = 23
    max_steps = 896

    def directions(self):
        return jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def context(self, s, i):
        p = s["pos"][i]
        q = jnp.clip(p + self.directions()[s["face"][i]], jnp.array([0, 0], jnp.int32), jnp.array([63, 47], jnp.int32))
        gate = jnp.argmin(jnp.sum(jnp.abs(s["gate_pos"] - p), axis=1))
        ferry = jnp.argmin(jnp.sum(jnp.abs(s["ferry_pos"] - p), axis=1))
        town = jnp.argmin(jnp.sum(jnp.abs(s["town_pos"] - p), axis=1))
        j = jnp.argmin(jnp.sum(jnp.abs(s["pos"] - q), axis=1))
        near = (j != i) & jnp.all(s["pos"][j] == q)
        return q, gate, ferry, town, j, near

    def walkable(self, s, i, xy):
        inside = jnp.all((xy >= jnp.array([0, 0], jnp.int32)) & (xy <= jnp.array([63, 47], jnp.int32)))
        q = jnp.clip(xy, jnp.array([0, 0], jnp.int32), jnp.array([63, 47], jnp.int32))
        tile = s["tiles"][q[1], q[0]]
        g = jnp.argmin(jnp.sum(jnp.abs(s["gate_pos"] - q), axis=1))
        b = jnp.argmin(jnp.sum(jnp.abs(s["bridge_pos"] - q), axis=1))
        phase = (s["t"] // 64) % 3
        bridge = s["road_grid"][s["bridge_pos"][b, 1], s["bridge_pos"][b, 0]] >= jnp.where(phase == 1, 3, 2)
        foreign = s["team"][i] != g
        gate = (s["gate_open"][g] | (s["permit"][g, s["team"][i]] > 0)) & (~foreign | (s["gold"][i] >= s["toll"][g]))
        free = ~jnp.any(jnp.all(s["pos"] == q, axis=1))
        terrain = (tile == 0) | (tile == 2) | ((tile == 3) & bridge) | ((tile == 4) & gate)
        return inside & terrain & free

    def legal_row(self, s, i):
        q, g, f, town, j, near = self.context(s, i)
        p = s["pos"][i]
        role = s["tool"][i]
        team = s["team"][i]
        mat = s["material"][i]
        tile = s["tiles"][q[1], q[0]]
        own_town = (town == team) & (jnp.sum(jnp.abs(s["town_pos"][town] - p)) <= 1)
        at_town = jnp.sum(jnp.abs(s["town_pos"][town] - p)) <= 1
        at_gate = jnp.sum(jnp.abs(s["gate_pos"][g] - p)) <= 1
        on_ferry = (p[1] == s["ferry_pos"][f, 1]) & (jnp.abs(p[0] - s["ferry_pos"][f, 0]) == 1)
        dest = jnp.stack([2 * s["ferry_pos"][f, 0] - p[0], p[1]])
        empty_dest = ~jnp.any(jnp.all(s["pos"] == dest, axis=1))
        foreign = team != f
        moves = jax.vmap(lambda xy: self.walkable(s, i, xy))(p[None, :] + self.directions()) & (s["cooldown"][i] == 0)
        cargo = s["cargo"][i] > 0
        deliver = at_town & cargo & (town != s["origin"][i]) & (s["received"][town] < 2)
        load = own_town & ~cargo & (s["provisions"][town] > 0)
        mask = jnp.concatenate([jnp.ones((1,), jnp.bool_), moves, jnp.ones((4,), jnp.bool_), jnp.stack([
            (role == 0) | (role == 4),
            own_town & (mat < 3) & (s["materials"][town] > 0),
            (role == 1) & (mat > 0) & ((tile == 0) | (tile == 2) | (tile == 3)) & (s["road_grid"][q[1], q[0]] < 3),
            deliver | load,
            on_ferry & (((s["t"] // 64) % 3) == 0) & (s["ferry_charge"][f] > 0) & empty_dest & (s["cooldown"][i] == 0) & (~foreign | (s["gold"][i] > 0)),
            on_ferry & (role == 2) & (mat > 0) & (s["ferry_charge"][f] <= 2),
            at_gate & (g == team) & (role == 5),
            at_gate & (g == team) & (role == 5),
            at_gate & (g != team) & (s["gold"][i] > 0) & (s["permit"][g, team] == 0),
            at_town & (town != team) & (role == 4) & ~cargo & (s["provisions"][town] > 0),
            (role == 4) & (s["road_grid"][q[1], q[0]] > 0),
            near & ((cargo & (s["cargo"][j] == 0)) | ((mat > 0) & (s["material"][j] < 3))),
            s["report"][i] > 0,
            near
        ])])
        return jnp.where(~s["done"], mask, jnp.arange(23) == 0)

    def legal_actions(self, s):
        return jax.vmap(lambda i: self.legal_row(s, i))(jnp.arange(192))

    def refresh(self, s):
        z = dict(s)
        p = z["pos"]
        distances = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2)
        local = distances <= 3
        other = local & ~jnp.eye(192, dtype=jnp.bool_)
        nearest = jnp.argsort(jnp.where(other, distances, 999), axis=1)[:, :4]
        detail = jnp.sum(jax.nn.one_hot(nearest, 192, dtype=jnp.int32), axis=1) > 0
        z["agent_vis"] = local & (detail | jnp.eye(192, dtype=jnp.bool_))
        z["map_vis"] = (jnp.abs(jnp.arange(64)[None, None, :] - p[:, 0, None, None]) <= 3) & (jnp.abs(jnp.arange(48)[None, :, None] - p[:, 1, None, None]) <= 3)
        z["road_level"] = z["road_grid"].reshape((3072,))
        z["road_active"] = z["road_level"] > 0
        z["bridge_strength"] = z["road_grid"][z["bridge_pos"][:, 1], z["bridge_pos"][:, 0]]
        for name in ("gate", "bridge", "ferry", "town", "road"):
            z[name + "_vis"] = (jnp.max(jnp.abs(p[:, None, :] - z[name + "_pos"][None, :, :]), axis=2) <= 3) & z[name + "_active"][None, :]
        return z

    def reset(self, key):
        g = jnp.arange(16, dtype=jnp.int32)
        base = jnp.stack([(g % 4) * 16, (g // 4) * 12], axis=1)
        off = jnp.array([[7, 4], [7, 7], [7, 9], [7, 5], [9, 6], [9, 5], [3, 4], [3, 5], [3, 6], [4, 4], [4, 5], [4, 6]], jnp.int32)
        pp = base[:, None, :] + off[None, :, :]
        pp = pp.at[:, 3, :].set(base[(g + 15) % 16] + jnp.array([7, 5], jnp.int32))
        pos = pp.reshape((192, 2))
        x = jnp.arange(64)[None, :]
        y = jnp.arange(48)[:, None]
        tiles = jnp.where((x % 16 == 8) & (y % 12 != 0), 1, 0).astype(jnp.int32)
        town, bridge, gate, ferry = (base + jnp.array(v, jnp.int32) for v in ([3, 3], [8, 7], [8, 5], [8, 9]))
        for positions, value in ((town, 2), (bridge, 3), (gate, 4), (ferry, 5)):
            tiles = tiles.at[positions[:, 1], positions[:, 0]].set(value)
        road = jnp.zeros((48, 64), jnp.int32).at[bridge[:, 1], bridge[:, 0]].set(1)
        k = jnp.arange(3072, dtype=jnp.int32)
        s = {
            "t": jnp.asarray(0, jnp.int32), "done": jnp.asarray(False), "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
            "tiles": tiles, "road_grid": road, "map_vis": jnp.zeros((192, 48, 64), jnp.bool_),
            "pos": pos, "active": jnp.ones((192,), jnp.bool_), "agent_vis": jnp.zeros((192, 192), jnp.bool_),
            "team": jnp.arange(192, dtype=jnp.int32) // 12, "tool": jnp.tile(jnp.arange(12, dtype=jnp.int32) % 6, 16),
            "face": jnp.tile(jnp.array([1, 1, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0], jnp.int32), 16),
            "material": jnp.tile(jnp.array([0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0], jnp.int32), 16),
            "cargo": jnp.tile(jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], jnp.int32), 16),
            "origin": jnp.arange(192, dtype=jnp.int32) // 12,
            "gold": jnp.full((192,), 4, jnp.int32), "cooldown": jnp.zeros((192,), jnp.int32),
            "report": jnp.zeros((192,), jnp.int32), "report_pos": pos,
            "heard": jnp.zeros((192,), jnp.int32), "heard_pos": pos, "signal": jnp.zeros((192,), jnp.bool_),
            "gate_pos": gate, "gate_active": jnp.ones((16,), jnp.bool_), "gate_vis": jnp.zeros((192, 16), jnp.bool_),
            "gate_open": jnp.zeros((16,), jnp.bool_), "toll": jnp.ones((16,), jnp.int32),
            "permit": jnp.zeros((16, 16), jnp.int32), "bank": jnp.zeros((16,), jnp.int32),
            "bridge_pos": bridge, "bridge_active": jnp.ones((16,), jnp.bool_), "bridge_vis": jnp.zeros((192, 16), jnp.bool_), "bridge_strength": jnp.ones((16,), jnp.int32),
            "ferry_pos": ferry, "ferry_active": jnp.ones((16,), jnp.bool_), "ferry_vis": jnp.zeros((192, 16), jnp.bool_), "ferry_charge": jnp.ones((16,), jnp.int32),
            "town_pos": town, "town_active": jnp.ones((16,), jnp.bool_), "town_vis": jnp.zeros((192, 16), jnp.bool_),
            "provisions": jnp.full((16,), 8, jnp.int32), "materials": jnp.full((16,), 24, jnp.int32), "received": jnp.zeros((16,), jnp.int32),
            "road_pos": jnp.stack([k % 64, k // 64], axis=1), "road_active": jnp.zeros((3072,), jnp.bool_),
            "road_vis": jnp.zeros((192, 3072), jnp.bool_), "road_level": road.reshape((3072,)),
            "blocked": jnp.zeros((192,), jnp.bool_), "built": jnp.zeros((192,), jnp.bool_),
            "delivered": jnp.zeros((192,), jnp.bool_), "crossed": jnp.zeros((192,), jnp.bool_),
            "damaged": jnp.zeros((192,), jnp.bool_), "negotiated": jnp.zeros((192,), jnp.bool_),
            "reward_parts": jnp.zeros((192, 6), jnp.float32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        p = s["pos"]
        ids = jnp.arange(192)
        phase = (s["t"] // 64) % 3
        own = jnp.stack([p[:, 0] / 63.0, p[:, 1] / 47.0, s["face"] / 3.0, s["tool"] / 5.0, s["team"] / 15.0,
             s["material"] / 3.0, s["cargo"] * 1.0, s["origin"] / 15.0, s["gold"] / 8.0, s["cooldown"] / 2.0,
             jnp.full((192,), s["t"] / 896.0), jnp.full((192,), phase / 2.0), s["report"] / 3.0,
             s["report_pos"][:, 0] / 63.0, s["report_pos"][:, 1] / 47.0, s["blocked"] * 1.0], axis=1)
        k = jnp.arange(49)
        q = p[:, None, :] + jnp.stack([k % 7 - 3, k // 7 - 3], axis=1)[None, :, :]
        inside = jnp.all((q >= jnp.array([0, 0], jnp.int32)) & (q <= jnp.array([63, 47], jnp.int32)), axis=2)
        c = jnp.clip(q, jnp.array([0, 0], jnp.int32), jnp.array([63, 47], jnp.int32))
        terrain = jnp.where(inside, s["tiles"][c[:, :, 1], c[:, :, 0]] / 5.0, -1.0)
        roads = jnp.where(inside, s["road_grid"][c[:, :, 1], c[:, :, 0]] / 3.0, -1.0)
        occ = jnp.all(q[:, :, None, :] == p[None, None, :, :], axis=3)
        actors = jnp.max(jnp.where(occ, (1 + s["cargo"][None, None, :]) / 2.0, 0.0), axis=2)
        patch = jnp.stack([terrain, roads, actors], axis=2).reshape((192, 147))
        dist = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2)
        nv = s["agent_vis"] & (ids[:, None] != ids[None, :])
        ni = jnp.argsort(jnp.where(nv, dist, 999), axis=1)[:, :4]
        valid = jnp.take_along_axis(nv, ni, axis=1)
        rel = (p[ni] - p[:, None, :]) / 3.0
        nb = jnp.stack([valid * 1.0, rel[:, :, 0], rel[:, :, 1], s["team"][ni] / 15.0,
                       s["tool"][ni] / 5.0, s["material"][ni] / 3.0, s["cargo"][ni] * 1.0, s["gold"][ni] / 8.0], axis=2)
        nb = jnp.where(valid[:, :, None], nb, 0.0).reshape((192, 32))
        gi = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["gate_pos"][None, :, :]), axis=2), axis=1)
        gv = s["gate_vis"][ids, gi]
        gr = (s["gate_pos"][gi] - p) / 3.0
        gate = jnp.stack([gv * 1.0, gr[:, 0], gr[:, 1], gi / 15.0, s["gate_open"][gi] * 1.0,
                         s["toll"][gi] / 2.0, s["permit"][gi, s["team"]] / 8.0, s["bank"][gi] / 4096.0], axis=1)
        ti = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["town_pos"][None, :, :]), axis=2), axis=1)
        tv = s["town_vis"][ids, ti]
        tr = (s["town_pos"][ti] - p) / 3.0
        town = jnp.stack([tv * 1.0, tr[:, 0], tr[:, 1], ti / 15.0, s["provisions"][ti] / 32.0,
                         s["materials"][ti] / 24.0, s["received"][ti] / 2.0], axis=1)
        fi = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["ferry_pos"][None, :, :]), axis=2), axis=1)
        fv = s["ferry_vis"][ids, fi]
        fr = (s["ferry_pos"][fi] - p) / 3.0
        ferry = jnp.stack([fv * 1.0, fr[:, 0], fr[:, 1], s["ferry_charge"][fi] / 4.0, jnp.full((192,), (phase == 0) * 1.0)], axis=1)
        bi = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["bridge_pos"][None, :, :]), axis=2), axis=1)
        bv = s["bridge_vis"][ids, bi]
        br = (s["bridge_pos"][bi] - p) / 3.0
        bridge = jnp.stack([bv * 1.0, br[:, 0], br[:, 1], s["bridge_strength"][bi] / 3.0,
                           (s["bridge_strength"][bi] >= jnp.where(phase == 1, 3, 2)) * 1.0], axis=1)
        heard = jnp.stack([(s["heard"] > 0) * 1.0, s["heard"] / 3.0, s["heard_pos"][:, 0] / 63.0, s["heard_pos"][:, 1] / 47.0], axis=1)
        return jnp.concatenate([own, patch, nb, jnp.where(gv[:, None], gate, 0.0), jnp.where(tv[:, None], town, 0.0),
              jnp.where(fv[:, None], ferry, 0.0), jnp.where(bv[:, None], bridge, 0.0), heard], axis=1).astype(jnp.float32)

    def act(self, i, s, a):
        i = jnp.asarray(i, jnp.int32)
        z = dict(s)
        mask = self.legal_row(z, i)
        valid = (a >= 0) & (a < 23) & mask[jnp.clip(a, 0, 22)]
        action = jnp.where(valid, a, 0)
        q, g, f, town, j, near = self.context(z, i)
        old_p = z["pos"][i]
        team = z["team"][i]
        cargo = z["cargo"][i]
        move = (action >= 1) & (action <= 4)
        direction = jnp.clip(action - 1, 0, 3)
        dest = jnp.clip(old_p + self.directions()[direction], jnp.array([0, 0], jnp.int32), jnp.array([63, 47], jnp.int32))
        z["face"] = z["face"].at[i].set(jnp.where(move, direction, jnp.where((action >= 5) & (action <= 8), action - 5, z["face"][i])))
        z["pos"] = z["pos"].at[i].set(jnp.where(move, dest, old_p))
        entering = jnp.argmin(jnp.sum(jnp.abs(z["gate_pos"] - dest), axis=1))
        gate_entry = move & (z["tiles"][dest[1], dest[0]] == 4)
        fee = jnp.where(gate_entry & (team != entering), z["toll"][entering], 0)
        z["gold"] = z["gold"].at[i].add(-fee)
        z["bank"] = z["bank"].at[entering].set(jnp.minimum(4096, z["bank"][entering] + fee))
        road = z["road_grid"][dest[1], dest[0]]
        z["cooldown"] = z["cooldown"].at[i].set(jnp.where(move & (road == 0), 2, z["cooldown"][i]))
        wear = move & (road > 0) & (cargo > 0)
        z["road_grid"] = z["road_grid"].at[dest[1], dest[0]].add(-wear.astype(jnp.int32))
        survey = action == 9
        spot = jnp.clip(old_p + 4 * self.directions()[s["face"][i]], jnp.array([0, 0], jnp.int32), jnp.array([63, 47], jnp.int32))
        report = 1 + (z["road_grid"][spot[1], spot[0]] > 0).astype(jnp.int32) + (z["tiles"][spot[1], spot[0]] == 1).astype(jnp.int32)
        z["report"] = z["report"].at[i].set(jnp.where(survey, report, z["report"][i]))
        z["report_pos"] = z["report_pos"].at[i].set(jnp.where(survey, spot, z["report_pos"][i]))
        gather = action == 10
        build = action == 11
        trade = action == 12
        delivery = trade & (cargo > 0)
        load = trade & (cargo == 0)
        ferry = action == 13
        operate = action == 14
        negotiate = action == 17
        raid = action == 18
        damage = action == 19
        gift_cargo = (action == 20) & (cargo > 0) & (z["cargo"][j] == 0)
        gift_mat = (action == 20) & ~gift_cargo & (z["material"][i] > 0) & (z["material"][j] < 3)
        z["material"] = z["material"].at[i].add(gather.astype(jnp.int32) - build.astype(jnp.int32) - operate.astype(jnp.int32) - gift_mat.astype(jnp.int32)).at[j].add(gift_mat.astype(jnp.int32))
        z["materials"] = z["materials"].at[town].add(-gather.astype(jnp.int32))
        z["road_grid"] = z["road_grid"].at[q[1], q[0]].add(build.astype(jnp.int32) - damage.astype(jnp.int32))
        z["cargo"] = z["cargo"].at[i].set(jnp.where(delivery | gift_cargo, 0, jnp.where(load | raid, 1, cargo)))
        z["cargo"] = z["cargo"].at[j].set(jnp.where(gift_cargo, 1, z["cargo"][j]))
        origin = z["origin"][i]
        z["origin"] = z["origin"].at[i].set(jnp.where(load | raid, town, origin))
        z["origin"] = z["origin"].at[j].set(jnp.where(gift_cargo, origin, z["origin"][j]))
        z["provisions"] = z["provisions"].at[town].add(delivery.astype(jnp.int32) - load.astype(jnp.int32) - raid.astype(jnp.int32))
        z["received"] = z["received"].at[town].add(delivery.astype(jnp.int32))
        z["gold"] = z["gold"].at[i].set(jnp.minimum(8, z["gold"][i] + 2 * delivery.astype(jnp.int32)))
        ferry_dest = jnp.stack([2 * z["ferry_pos"][f, 0] - old_p[0], old_p[1]])
        ferry_fee = ferry.astype(jnp.int32) * (team != f).astype(jnp.int32)
        z["pos"] = z["pos"].at[i].set(jnp.where(ferry, ferry_dest, z["pos"][i]))
        z["gold"] = z["gold"].at[i].add(-ferry_fee - negotiate.astype(jnp.int32))
        z["ferry_charge"] = z["ferry_charge"].at[f].add(2 * operate.astype(jnp.int32) - ferry.astype(jnp.int32))
        z["bank"] = z["bank"].at[f].set(jnp.minimum(4096, z["bank"][f] + ferry_fee))
        z["bank"] = z["bank"].at[g].set(jnp.minimum(4096, z["bank"][g] + negotiate.astype(jnp.int32)))
        z["gate_open"] = z["gate_open"].at[g].set(z["gate_open"][g] ^ (action == 15))
        z["toll"] = z["toll"].at[g].set(jnp.where(action == 16, (z["toll"][g] + 1) % 3, z["toll"][g]))
        z["permit"] = z["permit"].at[g, team].set(jnp.where(negotiate, 8, z["permit"][g, team]))
        ti, tj = z["tool"][i], z["tool"][j]
        z["tool"] = z["tool"].at[i].set(jnp.where(action == 22, tj, ti)).at[j].set(jnp.where(action == 22, ti, tj))
        z["signal"] = z["signal"].at[i].set(action == 21)
        blocked = (a >= 1) & (a <= 4) & ~valid
        z["blocked"] = z["blocked"].at[i].set(blocked)
        z["built"] = z["built"].at[i].set(build)
        z["delivered"] = z["delivered"].at[i].set(delivery)
        z["crossed"] = z["crossed"].at[i].set(gate_entry | ferry)
        z["damaged"] = z["damaged"].at[i].set(damage)
        z["negotiated"] = z["negotiated"].at[i].set(negotiate)
        rp = z["reward_parts"]
        rp = rp.at[i, 0].add(-move.astype(jnp.float32) * jnp.where(road > 0, 0.005, 0.02) - 0.02 * (build | operate | damage) - 0.005 * blocked)
        rp = rp.at[i, 1].add(0.3 * delivery + 0.1 * raid)
        rp = rp.at[:, 2].add((0.1 * delivery - 0.1 * raid) * (z["team"] == town))
        milestone = build & (s["tiles"][q[1], q[0]] == 3) & (s["road_grid"][q[1], q[0]] == 1)
        rp = rp.at[:, 3].add(0.03 * milestone * (jnp.max(jnp.abs(z["pos"] - q), axis=1) <= 3))
        rp = rp.at[i, 4].add(-0.02 * (fee + ferry_fee + negotiate.astype(jnp.int32)))
        rp = rp.at[:, 5].add(0.002 * fee * (z["team"] == entering) + 0.002 * ferry_fee * (z["team"] == f) + 0.002 * negotiate * (z["team"] == g))
        z["reward_parts"] = rp
        return z

    def advance(self, s, actions):
        z = dict(s)
        for name in ("blocked", "built", "delivered", "crossed", "damaged", "negotiated", "signal"):
            z[name] = jnp.zeros((192,), jnp.bool_)
        z["reward_parts"] = jnp.zeros((192, 6), jnp.float32)
        z = jax.lax.fori_loop(0, 192, lambda k, w: self.act((k + s["t"]) % 192, w, actions[(k + s["t"]) % 192]), z)
        thaw = ((s["t"] + 1) % 192) == 128
        z["road_grid"] = jnp.maximum(0, z["road_grid"] - thaw.astype(jnp.int32))
        z["cooldown"] = jnp.maximum(0, z["cooldown"] - 1)
        z["permit"] = jnp.maximum(0, z["permit"] - 1)
        near = (jnp.sum(jnp.abs(z["pos"][:, None, :] - z["pos"][None, :, :]), axis=2) <= 8) & z["signal"][None, :]
        sender = jnp.argmax(near, axis=1)
        heard = jnp.any(near, axis=1)
        z["heard"] = jnp.where(heard, z["report"][sender], 0)
        z["heard_pos"] = jnp.where(heard[:, None], z["report_pos"][sender], z["pos"])
        z["t"] = s["t"] + jnp.asarray(1, jnp.int32)
        z["terminated"] = jnp.all(z["received"] >= 2)
        z["truncated"] = (z["t"] >= 896) & ~z["terminated"]
        z["done"] = z["terminated"] | z["truncated"]
        return self.refresh(z)

    def raw_step(self, key, state, actions):
        ns = jax.lax.cond(state["done"], lambda _: state, lambda _: self.advance(state, actions), operand=None)
        rewards = jnp.where(state["done"], jnp.zeros((192,), jnp.float32), jnp.sum(ns["reward_parts"], axis=1))
        return self.observe(ns), ns, rewards, jnp.full((192,), ns["done"], jnp.bool_), {"done_all": ns["done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}

