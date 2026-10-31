"""Original blind-authored train-only spatial kernel; not executed by the author."""
import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 768
    observation_size = 683
    num_actions = 18
    max_steps = 1024
    width = 96
    height = 60


    def reset(self, key):
        ids = jnp.arange(768,dtype=jnp.int32)
        team,slot = ids//16,ids%16
        bases = jnp.stack(((jnp.arange(48)%8)*12,(jnp.arange(48)//8)*10),axis=1).astype(jnp.int32)
        offsets = jnp.array([[3,2],[2,3],[4,3],[5,4],[1,6],[2,6],[3,6],[4,6],[1,7],[2,7],[3,7],[4,7],[1,8],[2,8],[3,8],[4,8]],jnp.int32)
        s = self.initial(bases[team]+offsets[slot],team,slot%4)
        s["cargo"] = jnp.where(slot%4 == 0,1,0).astype(jnp.int32)
        s["load"] = jnp.where(slot%4 == 0,2,0).astype(jnp.int32)
        s["pump_pos"] = bases+jnp.array([3,3])
        s["cistern_pos"] = bases+jnp.array([5,3])
        s["stock_pos"] = bases+jnp.array([1,5])
        s["gate_pos"] = bases+jnp.array([6,4])
        s["camel_pos"] = bases+jnp.array([8,3])
        s["well_pos"] = jnp.stack((jnp.ones((6,),jnp.int32),jnp.arange(6)*10+1),axis=1)
        for name in ("pump","cistern","stock","gate","camel"):
            s[name+"_active"] = jnp.ones((48,),jnp.bool_)
        s["well_active"] = jnp.ones((6,),jnp.bool_)
        s["sand"] = jnp.ones((48,),jnp.int32)
        s["wind"] = jax.random.randint(key,(48,),0,2,dtype=jnp.int32)
        s["pressure"] = jnp.full((48,),32,jnp.int32)
        s["blocks"] = jnp.full((48,),24,jnp.int32)
        s["reserve"] = jnp.full((6,),32,jnp.int32)
        s["thirst"] = jnp.full((48,),8,jnp.int32)
        for name in ("clearing","stored","open_gate","signal","received","deliveries","pumped","delivered"):
            s[name] = jnp.zeros((48,),jnp.int32)
        s["rescued"] = jnp.zeros((768,),jnp.int32)
        for y in range(9):
            s["tiles"] = s["tiles"].at[bases[:,1]+y,bases[:,0]+6].set(3)
        s["tiles"] = s["tiles"].at[bases[:,1]+4,bases[:,0]+6].set(4)
        s["tiles"] = s["tiles"].at[bases[:,1]+3,bases[:,0]+4].set(2)
        s["tiles"] = s["tiles"].at[bases[:,1]+3,bases[:,0]+5].set(2)
        s = self.refresh(s)
        return self.observe(s),s

    def passable(self, terrain):
        return (terrain != 3) & (terrain != 4) & (terrain != 5)

    def legal_actions(self, s):
        pi,pn = self.target(s,"pump")
        ci,cn = self.target(s,"cistern")
        si,sn = self.target(s,"stock")
        gi,gn = self.target(s,"gate")
        mi,mn = self.target(s,"camel")
        empty = s["load"] == 0
        mask = jnp.zeros((768,18),jnp.bool_).at[:,0].set(True)
        mask = mask.at[:,1:5].set(self.movement_mask(s))
        mask = mask.at[:,5].set(sn & empty & (s["blocks"][si] >= 2))
        mask = mask.at[:,6].set(pn & (s["tool"] == 2))
        mask = mask.at[:,7].set(pn & (s["tool"] == 1))
        here = s["tiles"][s["pos"][:,1],s["pos"][:,0]]
        material = (s["cargo"] == 1) & (s["load"] > 0) & (s["tool"] == 1)
        mask = mask.at[:,8].set(material & (here == 0))
        target = s["pos"] + jnp.array([1,0])
        tx = jnp.minimum(target[:,0],self.width-1)
        occupied = jnp.any(jnp.all(target[:,None,:] == s["pos"][None,:,:],axis=2),axis=1)
        mask = mask.at[:,9].set(material & (target[:,0] < self.width) & ~occupied & (s["tiles"][target[:,1],tx] == 0))
        mask = mask.at[:,10].set(cn & empty & (s["stored"][ci] > 0))
        mask = mask.at[:,11].set(mn & (s["cargo"] == 2) & (s["load"] > 0))
        mask = mask.at[:,12].set(gn & (s["tool"] == 4))
        mask = mask.at[:,13].set(gn)
        distance = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=2)
        adjacent = jnp.any(distance == 1,axis=1)
        mask = mask.at[:,14].set(adjacent).at[:,15].set(adjacent)
        rescue_near = jnp.any((distance == 1) & (s["energy"][None,:] <= 5),axis=1)
        mask = mask.at[:,16].set(rescue_near & (s["cargo"] == 2) & (s["load"] > 0))
        mask = mask.at[:,17].set(gn)
        mask = mask & ((s["energy"] > 0) & ~s["done"])[:,None]
        return mask.at[:,0].set(True)

    def advance(self, key, s, a):
        n = dict(s)
        n["pos"] = self.move(s,a)
        n["energy"] = jnp.clip(s["energy"]+jnp.where(a == 0,1,-1),0,20)
        n["tool"],paired = self.exchange(s,a,14)
        n["cargo"],n["load"],n["memory"] = self.cargo_swap(s,a,15)
        pi,pn = self.target(s,"pump")
        ci,cn = self.target(s,"cistern")
        si,sn = self.target(s,"stock")
        gi,gn = self.target(s,"gate")
        mi,mn = self.target(s,"camel")
        collect = self.winner(si,a == 5,48)
        clear = self.tally(pi,a == 6,48) > 0
        operate = self.tally(pi,a == 7,48) > 0
        fill = self.winner(ci,a == 10,48)
        deliver = self.winner(mi,a == 11,48)
        staffed = self.tally(gi,a == 12,48) > 0
        signaler = self.winner(gi,a == 13,48)
        p = s["pump_pos"]
        connected = (s["tiles"][p[:,1],p[:,0]+1] == 2) & (s["tiles"][p[:,1],p[:,0]+2] == 2)
        request = operate & clear & connected & (s["sand"] < 3) & (s["stored"] < 12)
        cumulative = jnp.cumsum(request.reshape((6,8)),axis=1)
        allocation = request & (cumulative <= s["reserve"][:,None]).reshape((48,))
        withdraw = jnp.sum(allocation.reshape((6,8)),axis=1).astype(jnp.int32)
        n["reserve"] = jnp.minimum(32,s["reserve"]-withdraw+jnp.where((s["t"]+1)%48 == 0,8,0))
        n["pressure"] = n["reserve"][jnp.arange(48)//8]
        n["stored"] = s["stored"] + allocation.astype(jnp.int32)-self.tally(ci,fill,48)
        n["clearing"] = clear.astype(jnp.int32)
        n["pumped"] = allocation.astype(jnp.int32)
        storm = (s["t"]+1)%16 == 0
        side = jnp.where(s["wind"] == 0,-1,1)
        shade = (s["tiles"][p[:,1],p[:,0]+side] == 3) | (s["tiles"][p[:,1]-1,p[:,0]] == 3)
        sand = jnp.clip(s["sand"]-2*clear.astype(jnp.int32)+(storm & ~shade).astype(jnp.int32),0,4)
        n["sand"] = sand
        n["wind"] = jnp.where(storm,1-s["wind"],s["wind"])
        lay = a == 8
        wall_target = s["pos"] + jnp.array([1,0],jnp.int32)
        newly_occupied = jnp.any(jnp.all(wall_target[:,None,:] == n["pos"][None,:,:],axis=2),axis=1)
        build = (a == 9) & ~newly_occupied
        x,y = s["pos"][:,0],s["pos"][:,1]
        n["tiles"] = s["tiles"].at[y,x].set(jnp.where(lay,2,s["tiles"][y,x]))
        east = jnp.minimum(x+1,self.width-1)
        n["tiles"] = n["tiles"].at[y,east].set(jnp.where(build,3,n["tiles"][y,east]))
        middle = n["tiles"][p[:,1],p[:,0]+1]
        buried = storm & ~shade & (sand >= 3)
        middle = jnp.where(buried,5,jnp.where(clear & (middle == 5),2,middle))
        n["tiles"] = n["tiles"].at[p[:,1],p[:,0]+1].set(middle)
        n["open_gate"] = staffed.astype(jnp.int32)
        gp = s["gate_pos"]
        n["tiles"] = n["tiles"].at[gp[:,1],gp[:,0]].set(jnp.where(staffed,6,4))
        sent = jnp.zeros((48,),jnp.int32).at[gi].add(jnp.where(signaler,jnp.maximum(1,s["memory"]),0))
        n["signal"] = jnp.where(self.tally(gi,signaler,48) > 0,sent,0)
        ids = jnp.arange(48)
        upstream = jnp.maximum(0,ids-1)
        corridor_x = jnp.clip(gp[:,0,None]-12+jnp.arange(13)[None,:],0,self.width-1)
        corridor = s["tiles"][gp[:,1,None],corridor_x]
        clear_line = ~jnp.any((corridor == 3) | (corridor == 4),axis=1)
        transmitted = (ids%8 > 0) & clear_line
        n["received"] = jnp.where(transmitted,s["signal"][upstream],0)
        n["memory"] = jnp.where(a == 17,s["received"][gi],jnp.where((a == 6) | (a == 7),jnp.minimum(4,s["sand"][pi]+1),n["memory"]))
        delivered = self.tally(mi,deliver,48)
        n["delivered"] = delivered
        n["deliveries"] = s["deliveries"]+delivered
        n["thirst"] = jnp.clip(s["thirst"]+2*delivered-jnp.where((s["t"]+1)%32 == 0,1,0),0,8)
        bases = jnp.stack(((ids%8)*12,(ids//8)*10),axis=1)
        n["camel_pos"] = bases+jnp.stack((8+jnp.full((48,),((s["t"]+1)//16)%3,jnp.int32),jnp.full((48,),3,jnp.int32)),axis=1)
        distance = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=2)
        eligible = (distance == 1) & (s["energy"][None,:] <= 5)
        target = jnp.argmin(jnp.where(eligible,jnp.arange(768)[None,:],768),axis=1)
        rescue = self.winner(target,(a == 16) & jnp.any(eligible,axis=1),768)
        rescued = self.tally(target,rescue,768)
        n["rescued"] = rescued
        n["energy"] = jnp.minimum(20,n["energy"]+6*rescued)
        n["blocks"] = s["blocks"]-2*self.tally(si,collect,48)
        consumed = lay | build | deliver | rescue
        n["load"] = jnp.where(collect,2,jnp.where(fill,1,jnp.maximum(0,n["load"]-consumed.astype(jnp.int32))))
        n["cargo"] = jnp.where(collect,1,jnp.where(fill,2,jnp.where(n["load"] == 0,0,n["cargo"])))
        rewards = -0.01*(a > 0)-0.02*(lay | build)+deliver+0.25*rescue+0.1*(rescued > 0)
        rewards = rewards+0.03*delivered[s["team"]]+0.005*allocation[s["team"]]
        natural = jnp.sum(n["deliveries"]) >= 96
        return n,rewards.astype(jnp.float32),natural

    def observe(self, s):
        board = jnp.zeros((60, 96, 27), jnp.float32)
        board = board.at[:, :, 0].set(s['tiles'].astype(jnp.float32) / 10.0)
        vals = jnp.stack(((jnp.arange(self.num_agents, dtype=jnp.float32) + 1) / self.num_agents,
                          (s['role'] + 1) / 4.0, (s['team'] + 1) / 48.0,
                          s['tool'] / 4.0, s['cargo'] / 4.0, s['load'] / 4.0,
                          s['energy'] / 20.0, s['memory'] / 4.0), axis=1).astype(jnp.float32)
        board = board.at[s['pos'][:, 1], s['pos'][:, 0], 1:9].set(vals)
        values = jnp.stack(((jnp.arange(48, dtype=jnp.float32) + 1) / 48.0, s['sand'].astype(jnp.float32) / 4.0, s['clearing'].astype(jnp.float32) / 1.0, s['pressure'].astype(jnp.float32) / 32.0, s['wind'].astype(jnp.float32) / 1.0), axis=1)
        board = board.at[s['pump_pos'][:, 1], s['pump_pos'][:, 0], 9:14].set(values)
        values = jnp.stack(((jnp.arange(48, dtype=jnp.float32) + 1) / 48.0, s['stored'].astype(jnp.float32) / 12.0), axis=1)
        board = board.at[s['cistern_pos'][:, 1], s['cistern_pos'][:, 0], 14:16].set(values)
        values = jnp.stack(((jnp.arange(48, dtype=jnp.float32) + 1) / 48.0, s['blocks'].astype(jnp.float32) / 24.0), axis=1)
        board = board.at[s['stock_pos'][:, 1], s['stock_pos'][:, 0], 16:18].set(values)
        values = jnp.stack(((jnp.arange(48, dtype=jnp.float32) + 1) / 48.0, s['open_gate'].astype(jnp.float32) / 1.0, s['signal'].astype(jnp.float32) / 4.0, s['received'].astype(jnp.float32) / 4.0), axis=1)
        board = board.at[s['gate_pos'][:, 1], s['gate_pos'][:, 0], 18:22].set(values)
        values = jnp.stack(((jnp.arange(48, dtype=jnp.float32) + 1) / 48.0, s['thirst'].astype(jnp.float32) / 8.0, s['deliveries'].astype(jnp.float32) / 96.0), axis=1)
        board = board.at[s['camel_pos'][:, 1], s['camel_pos'][:, 0], 22:25].set(values)
        values = jnp.stack(((jnp.arange(6, dtype=jnp.float32) + 1) / 6.0, s['reserve'].astype(jnp.float32) / 32.0), axis=1)
        board = board.at[s['well_pos'][:, 1], s['well_pos'][:, 0], 25:27].set(values)
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
        s["pump_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["pump_pos"][None, :, :]) <= 2, axis=2)
        s["cistern_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["cistern_pos"][None, :, :]) <= 2, axis=2)
        s["stock_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["stock_pos"][None, :, :]) <= 2, axis=2)
        s["gate_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["gate_pos"][None, :, :]) <= 2, axis=2)
        s["camel_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["camel_pos"][None, :, :]) <= 2, axis=2)
        s["well_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["well_pos"][None, :, :]) <= 2, axis=2)
        return s

