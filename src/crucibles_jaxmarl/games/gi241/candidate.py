"""Original blind-authored train-only spatial kernel; not executed by the author."""
import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 64
    observation_size = 733
    num_actions = 17
    max_steps = 896
    width = 48
    height = 24


    def reset(self, key):
        ids = jnp.arange(64, dtype=jnp.int32)
        team, slot = ids // 8, ids % 8
        bases = jnp.stack(((jnp.arange(8) % 4) * 12, (jnp.arange(8) // 4) * 12), axis=1).astype(jnp.int32)
        offsets = jnp.array([[4,3],[7,3],[6,2],[2,3],[2,6],[4,6],[6,6],[8,6]], jnp.int32)
        s = self.initial(bases[team] + offsets[slot], team, slot % 4)
        s["cargo"] = jnp.where(slot == 0, 1, jnp.where(slot == 2, 3, 0)).astype(jnp.int32)
        s["load"] = (s["cargo"] > 0).astype(jnp.int32)
        s["memory"] = jnp.where(s["cargo"] > 0, 1, 0).astype(jnp.int32)
        s["supply_pos"] = bases + jnp.array([1,2])
        s["prep_pos"] = bases + jnp.array([4,2])
        s["lift_pos"] = bases + jnp.array([7,2])
        s["counter_pos"] = bases + jnp.array([7,4])
        s["terminal_pos"] = bases + jnp.array([9,2])
        s["wash_pos"] = jnp.array([[10,2],[34,2],[10,14],[34,14]], jnp.int32)
        for name in ("supply", "prep", "lift", "counter", "terminal"):
            s[name + "_active"] = jnp.ones((8,), jnp.bool_)
        s["wash_active"] = jnp.ones((4,), jnp.bool_)
        s["stock"] = jnp.full((8,),24,jnp.int32)
        for name in ("stage","age","recipe","tray","tray_tag","phase","held","served_total","lift_moved","served","spoiled"):
            s[name] = jnp.zeros((8,), jnp.int32)
        s["heat"] = jnp.ones((8,),jnp.int32)
        s["order_seed"] = jax.random.randint(key,(8,),0,2,dtype=jnp.int32)
        s["order"] = 1 + s["order_seed"]
        s["window"] = jnp.zeros((8,),jnp.int32)
        s["wash_water"] = jnp.full((4,),8,jnp.int32)
        s["steam"] = jnp.zeros((4,),jnp.int32)
        s["tiles"] = s["tiles"].at[bases[:,1]+2,bases[:,0]+8].set(2)
        s["tiles"] = s["tiles"].at[bases[:,1]+4,bases[:,0]+8].set(1)
        s = self.refresh(s)
        return self.observe(s), s

    def passable(self, terrain):
        return terrain != 1

    def legal_actions(self, s):
        si,sn = self.target(s,"supply")
        pi,pn = self.target(s,"prep")
        li,ln = self.target(s,"lift")
        ci,cn = self.target(s,"counter")
        ti,tn = self.target(s,"terminal")
        wi,wn = self.target(s,"wash")
        empty = s["load"] == 0
        mask = jnp.zeros((64,17),jnp.bool_).at[:,0].set(True)
        mask = mask.at[:,1:5].set(self.movement_mask(s))
        mask = mask.at[:,5].set(sn & empty & (s["stock"][si] > 0))
        mask = mask.at[:,6].set(pn & (s["tool"] == 1) & (s["cargo"] == 1) & (s["stage"][pi] == 0))
        mask = mask.at[:,7].set(pn & (s["tool"] == 1) & (s["stage"][pi] > 0) & (s["stage"][pi] < 4))
        mask = mask.at[:,8].set(pn & empty & (s["stage"][pi] == 4))
        mask = mask.at[:,9].set(ln & (s["cargo"] == 3) & (s["tray"][li] == 0))
        mask = mask.at[:,10].set(cn & (s["tool"] == 2))
        mask = mask.at[:,11].set(ln & empty & (s["tray"][li] > 0))
        mask = mask.at[:,12].set(tn & (s["cargo"] == 3))
        mask = mask.at[:,13].set(wn & (s["cargo"] == 4) & (s["wash_water"][wi] > 0))
        mask = mask.at[:,14].set(tn & empty)
        distance = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=2)
        adjacent = jnp.any(distance == 1,axis=1)
        mask = mask.at[:,15].set(adjacent).at[:,16].set(adjacent)
        mask = mask & ((s["energy"] > 0) & ~s["done"])[:,None]
        return mask.at[:,0].set(True)

    def advance(self, key, s, a):
        n = dict(s)
        n["pos"] = self.move(s,a)
        n["energy"] = jnp.clip(s["energy"] + jnp.where(a == 0,1,-1),0,20)
        n["tool"], swapped = self.exchange(s,a,15)
        n["cargo"], n["load"], n["memory"] = self.cargo_swap(s,a,16)
        si,sn = self.target(s,"supply")
        pi,pn = self.target(s,"prep")
        li,ln = self.target(s,"lift")
        ci,cn = self.target(s,"counter")
        ti,tn = self.target(s,"terminal")
        wi,wn = self.target(s,"wash")
        take = self.winner(si,a == 5,8)
        chop = self.winner(pi,a == 6,8)
        plate = self.winner(pi,a == 8,8)
        load = self.winner(li,a == 9,8)
        receive = self.winner(li,a == 11,8)
        serve = self.winner(ti,a == 12,8)
        wash = self.winner(wi,a == 13,4)
        chopped = self.tally(pi,chop,8)
        plated = self.tally(pi,plate,8)
        tended = self.tally(pi,a == 7,8) > 0
        washed = self.tally(wi,wash,4)
        n["stock"] = s["stock"] - self.tally(si,take,8)
        n["wash_water"] = jnp.minimum(8,s["wash_water"] - washed + jnp.where((s["t"]+1)%16 == 0,4,0))
        n["steam"] = jnp.clip(s["steam"] - 1 + 3*washed,0,4)
        close = jnp.sum(jnp.abs(s["prep_pos"][:,None,:]-s["wash_pos"][None,:,:]),axis=2) <= 6
        heat = jnp.minimum(4,1+jnp.sum(close*s["steam"][None,:],axis=1))
        n["heat"] = heat.astype(jnp.int32)
        age = jnp.where(s["stage"] > 0,jnp.minimum(12,s["age"]+1),0)
        spoiled = (s["stage"] > 0) & (heat >= 4) & (age >= 7)
        stage = jnp.where(s["stage"] > 0,jnp.minimum(4,s["stage"] + ((heat >= 2) | tended).astype(jnp.int32)),0)
        n["stage"] = jnp.where((plated > 0) | spoiled,0,jnp.where(chopped > 0,1,stage)).astype(jnp.int32)
        n["age"] = jnp.where((plated > 0) | spoiled | (chopped > 0),0,age).astype(jnp.int32)
        recipe_added = jnp.zeros((8,),jnp.int32).at[pi].add(jnp.where(chop,jnp.maximum(1,s["memory"]),0))
        n["recipe"] = jnp.where(chopped > 0,recipe_added,s["recipe"])
        n["spoiled"] = spoiled.astype(jnp.int32)
        held = self.tally(ci,a == 10,8) > 0
        received = self.tally(li,receive,8)
        loaded = self.tally(li,load,8)
        moved = held & (s["tray"] > 0) & (received == 0)
        n["held"] = held.astype(jnp.int32)
        n["phase"] = jnp.where(moved,1-s["phase"],s["phase"])
        n["tray"] = jnp.clip(s["tray"] + loaded - received,0,1)
        tag_added = jnp.zeros((8,),jnp.int32).at[li].add(jnp.where(load,s["memory"],0))
        n["tray_tag"] = jnp.where(loaded > 0,tag_added,s["tray_tag"])
        bases = jnp.stack(((jnp.arange(8)%4)*12,(jnp.arange(8)//4)*12),axis=1)
        n["lift_pos"] = bases + jnp.stack((jnp.full((8,),7,jnp.int32),2+2*n["phase"]),axis=1)
        n["lift_moved"] = moved.astype(jnp.int32)
        n["tiles"] = s["tiles"].at[bases[:,1]+2,bases[:,0]+8].set(jnp.where(n["phase"] == 0,2,1))
        n["tiles"] = n["tiles"].at[bases[:,1]+4,bases[:,0]+8].set(jnp.where(n["phase"] == 1,2,1))
        accepted = serve & (s["memory"] == s["order"][ti]) & (s["window"][ti] < 8)
        accepted_count = self.tally(ti,accepted,8)
        n["served"] = accepted_count
        n["served_total"] = s["served_total"] + accepted_count
        n["window"] = jnp.full((8,),(s["t"]+1)%12,jnp.int32)
        n["order"] = 1 + (s["order_seed"] + (s["t"]+1)//12)%2
        n["terminal_pos"] = bases + jnp.stack((jnp.full((8,),9,jnp.int32),2+2*(((s["t"]+1)//12)%2)*jnp.ones((8,),jnp.int32)),axis=1)
        remove = chop | load | wash
        n["cargo"] = jnp.where(remove,0,jnp.where(take,1,jnp.where(plate | receive,3,jnp.where(serve,4,n["cargo"]))))
        n["load"] = jnp.where(remove,0,jnp.where(take | plate | receive | serve,1,n["load"]))
        n["memory"] = jnp.where(a == 14,s["order"][ti],jnp.where(plate,s["recipe"][pi],jnp.where(receive,s["tray_tag"][li],n["memory"])))
        rewards = -0.01*(a > 0) + 1.0*accepted - 0.25*(serve & ~accepted)
        rewards = rewards + 0.05*accepted_count[s["team"]] - 0.05*spoiled[s["team"]] - 0.01*moved[s["team"]]
        natural = jnp.sum(n["served_total"]) >= 32
        return n,rewards.astype(jnp.float32),natural

    def observe(self, s):
        board = jnp.zeros((24, 48, 29), jnp.float32)
        board = board.at[:, :, 0].set(s['tiles'].astype(jnp.float32) / 10.0)
        vals = jnp.stack(((jnp.arange(self.num_agents, dtype=jnp.float32) + 1) / self.num_agents,
                          (s['role'] + 1) / 4.0, (s['team'] + 1) / 8.0,
                          s['tool'] / 4.0, s['cargo'] / 4.0, s['load'] / 4.0,
                          s['energy'] / 20.0, s['memory'] / 4.0), axis=1).astype(jnp.float32)
        board = board.at[s['pos'][:, 1], s['pos'][:, 0], 1:9].set(vals)
        values = jnp.stack(((jnp.arange(8, dtype=jnp.float32) + 1) / 8.0, s['stock'].astype(jnp.float32) / 24.0), axis=1)
        board = board.at[s['supply_pos'][:, 1], s['supply_pos'][:, 0], 9:11].set(values)
        values = jnp.stack(((jnp.arange(8, dtype=jnp.float32) + 1) / 8.0, s['stage'].astype(jnp.float32) / 4.0, s['age'].astype(jnp.float32) / 12.0, s['recipe'].astype(jnp.float32) / 2.0, s['heat'].astype(jnp.float32) / 4.0), axis=1)
        board = board.at[s['prep_pos'][:, 1], s['prep_pos'][:, 0], 11:16].set(values)
        values = jnp.stack(((jnp.arange(8, dtype=jnp.float32) + 1) / 8.0, s['tray'].astype(jnp.float32) / 1.0, s['tray_tag'].astype(jnp.float32) / 2.0, s['phase'].astype(jnp.float32) / 1.0), axis=1)
        board = board.at[s['lift_pos'][:, 1], s['lift_pos'][:, 0], 16:20].set(values)
        values = jnp.stack(((jnp.arange(8, dtype=jnp.float32) + 1) / 8.0, s['held'].astype(jnp.float32) / 1.0), axis=1)
        board = board.at[s['counter_pos'][:, 1], s['counter_pos'][:, 0], 20:22].set(values)
        values = jnp.stack(((jnp.arange(8, dtype=jnp.float32) + 1) / 8.0, s['order'].astype(jnp.float32) / 2.0, s['window'].astype(jnp.float32) / 11.0, s['served_total'].astype(jnp.float32) / 32.0), axis=1)
        board = board.at[s['terminal_pos'][:, 1], s['terminal_pos'][:, 0], 22:26].set(values)
        values = jnp.stack(((jnp.arange(4, dtype=jnp.float32) + 1) / 4.0, s['wash_water'].astype(jnp.float32) / 8.0, s['steam'].astype(jnp.float32) / 4.0), axis=1)
        board = board.at[s['wash_pos'][:, 1], s['wash_pos'][:, 0], 26:29].set(values)
        offsets = jnp.stack((jnp.tile(jnp.arange(-2, 3), 5), jnp.repeat(jnp.arange(-2, 3), 5)), axis=1)
        cells = s['pos'][:, None, :] + offsets[None, :, :]
        inside = (cells[:, :, 0] >= 0) & (cells[:, :, 0] < self.width) & (cells[:, :, 1] >= 0) & (cells[:, :, 1] < self.height)
        patch = board[jnp.clip(cells[:, :, 1], 0, self.height - 1), jnp.clip(cells[:, :, 0], 0, self.width - 1)]
        patch = jnp.where(inside[:, :, None], patch, -jnp.ones_like(patch))
        prefix = jnp.stack((jnp.full((self.num_agents,), s['t'] / self.max_steps),
                            s['pos'][:, 0] / (self.width - 1.0), s['pos'][:, 1] / (self.height - 1.0),
                            s['energy'] / 20.0, s['tool'] / 4.0, s['cargo'] / 4.0, s['load'] / 4.0, s['memory'] / 4.0), axis=1)
        return jnp.concatenate((prefix, patch.reshape((self.num_agents, -1))), axis=1).astype(jnp.float32)

    def initial(self, pos, team, role):
        return {"t": jnp.array(0, jnp.int32), "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
                "pos": pos.astype(jnp.int32), "team": team.astype(jnp.int32), "role": role.astype(jnp.int32),
                "active": jnp.ones((self.num_agents,), jnp.bool_), "tool": (role % 4 + 1).astype(jnp.int32),
                "cargo": jnp.zeros((self.num_agents,), jnp.int32), "load": jnp.zeros((self.num_agents,), jnp.int32),
                "energy": jnp.full((self.num_agents,), 20, jnp.int32), "memory": jnp.zeros((self.num_agents,), jnp.int32),
                "tiles": jnp.zeros((self.height, self.width), jnp.int32)}

    def target(self, s, name):
        distance = jnp.sum(jnp.abs(s["pos"][:, None, :] - s[name + "_pos"][None, :, :]), axis=2)
        index = jnp.argmin(distance, axis=1)
        near = jnp.min(distance, axis=1) <= 1
        return index, near

    def tally(self, index, flag, count):
        return jnp.zeros((count,), jnp.int32).at[index].add(flag.astype(jnp.int32))

    def winner(self, index, flag, count):
        claims = (index[:, None] == jnp.arange(count)[None, :]) & flag[:, None]
        first = jnp.argmax(claims, axis=0)
        return flag & (first[index] == jnp.arange(self.num_agents))

    def cargo_swap(self, s, a, action):
        distance = jnp.sum(jnp.abs(s["pos"][:, None, :] - s["pos"][None, :, :]), axis=2)
        ids = jnp.arange(self.num_agents)
        eligible = (distance == 1) & (a[None, :] == action)
        partner = jnp.argmin(jnp.where(eligible, ids[None, :], self.num_agents), axis=1)
        paired = (a == action) & jnp.any(eligible, axis=1) & (partner[partner] == ids)
        return jnp.where(paired, s["cargo"][partner], s["cargo"]), jnp.where(paired, s["load"][partner], s["load"]), jnp.where(paired, s["memory"][partner], s["memory"])

    def movement_mask(self, s):
        offsets = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]], jnp.int32)
        destination = s["pos"][:, None, :] + offsets[None, :, :]
        inside = (destination[:, :, 0] >= 0) & (destination[:, :, 0] < self.width) & (destination[:, :, 1] >= 0) & (destination[:, :, 1] < self.height)
        x = jnp.clip(destination[:, :, 0], 0, self.width - 1)
        y = jnp.clip(destination[:, :, 1], 0, self.height - 1)
        occupancy = jnp.zeros((self.height, self.width), jnp.int32).at[s["pos"][:, 1], s["pos"][:, 0]].set(1)
        terrain = s["tiles"][y, x]
        return inside & (occupancy[y, x] == 0) & self.passable(terrain)

    def move(self, s, a):
        steps = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], jnp.int32)
        moving = (a >= 1) & (a <= 4)
        destination = s["pos"] + steps[jnp.where(moving, a, 0)]
        ids = jnp.arange(self.num_agents)
        contested = jnp.all(destination[:, None, :] == destination[None, :, :], axis=2)
        lost = jnp.any(contested & (ids[None, :] < ids[:, None]) & moving[None, :], axis=1)
        return jnp.where((moving & ~lost)[:, None], destination, s["pos"])

    def exchange(self, s, a, action):
        distance = jnp.sum(jnp.abs(s["pos"][:, None, :] - s["pos"][None, :, :]), axis=2)
        ids = jnp.arange(self.num_agents)
        eligible = (distance == 1) & (a[None, :] == action)
        partner = jnp.argmin(jnp.where(eligible, ids[None, :], self.num_agents), axis=1)
        paired = (a == action) & jnp.any(eligible, axis=1) & (partner[partner] == ids)
        return jnp.where(paired, s["tool"][partner], s["tool"]), paired

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions.astype(jnp.int32), 0, self.num_actions - 1)
        legal = self.legal_actions(state)
        valid = (safe == actions) & jnp.take_along_axis(legal, safe[:, None], axis=1)[:, 0]
        a = jnp.where(valid, safe, 0)
        nxt, rewards, natural = self.advance(key, state, a)
        nxt["t"] = state["t"] + jnp.array(1, jnp.int32)
        nxt["terminated"] = natural
        nxt["truncated"] = (nxt["t"] >= self.max_steps) & ~natural
        nxt["done"] = nxt["terminated"] | nxt["truncated"]
        nxt = self.refresh(nxt)
        nxt = jax.tree.map(lambda new, old: jnp.where(state["done"], old, new), nxt, state)
        rewards = jnp.where(state["done"], jnp.zeros((self.num_agents,), jnp.float32), rewards).astype(jnp.float32)
        dones = jnp.full((self.num_agents,), nxt["done"], jnp.bool_)
        info = {"done_all": nxt["done"], "terminated": nxt["terminated"], "truncated": nxt["truncated"]}
        return self.observe(nxt), nxt, rewards, dones, info

    def refresh(self, s):
        s = dict(s)
        s["agent_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["pos"][None, :, :]) <= 2, axis=2)
        xx = jnp.arange(self.width)[None, None, :]
        yy = jnp.arange(self.height)[None, :, None]
        s["map_visibility"] = (jnp.abs(xx - s["pos"][:, 0, None, None]) <= 2) & (jnp.abs(yy - s["pos"][:, 1, None, None]) <= 2)
        s["supply_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["supply_pos"][None, :, :]) <= 2, axis=2)
        s["prep_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["prep_pos"][None, :, :]) <= 2, axis=2)
        s["lift_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["lift_pos"][None, :, :]) <= 2, axis=2)
        s["counter_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["counter_pos"][None, :, :]) <= 2, axis=2)
        s["terminal_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["terminal_pos"][None, :, :]) <= 2, axis=2)
        s["wash_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["wash_pos"][None, :, :]) <= 2, axis=2)
        return s

