"""Original blind-authored train-only spatial kernel; not executed by the author."""
import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 320
    observation_size = 833
    num_actions = 21
    max_steps = 1024
    width = 70
    height = 48


    def reset(self, key):
        ids = jnp.arange(320,dtype=jnp.int32)
        team,slot = ids//16,ids%16
        towns = jnp.arange(20,dtype=jnp.int32)
        bases = jnp.stack(((towns%5)*14,(towns//5)*12),axis=1)
        offsets = jnp.array([[3,3],[4,4],[5,3],[9,4],[1,6],[2,6],[3,6],[4,6],[1,7],[2,7],[3,7],[4,7],[1,9],[2,9],[3,9],[4,9]],jnp.int32)
        s = self.initial(bases[team]+offsets[slot],team,slot%8)
        s["cargo"] = jnp.where((slot == 0) | (slot == 1),1,0).astype(jnp.int32)
        s["load"] = jnp.where(slot == 0,2,jnp.where(slot == 1,1,0)).astype(jnp.int32)
        for name,offset in (("hall",(4,3)),("stock",(1,3)),("gate",(8,4)),("field",(5,7)),("harbor",(10,4)),("market",(11,8)),("reservoir",(7,2)),("gauge",(7,7))):
            s[name+"_pos"] = bases+jnp.array(offset,jnp.int32)
            s[name+"_active"] = jnp.ones((20,),jnp.bool_)
        for name in ("credits","window","first_donor","permit","sluice","ferry_count","market_level","sales","levee","pending","flood","granted","damage","sold"):
            s[name] = jnp.zeros((20,),jnp.int32)
        s["supplies"] = jnp.full((20,),32,jnp.int32)
        s["growth"] = jnp.full((20,),2,jnp.int32)
        s["wet"] = jnp.full((20,),2,jnp.int32)
        s["water"] = jnp.full((20,),4,jnp.int32)
        phases = jax.random.randint(key,(4,),0,8,dtype=jnp.int32)
        s["tide_phase"] = phases[towns//5]
        for x in range(5):
            s["tiles"] = s["tiles"].at[:,x*14+8].set(1)
        s["tiles"] = s["tiles"].at[bases[:,1]+4,bases[:,0]+8].set(2)
        s = self.refresh(s)
        return self.observe(s),s

    def passable(self, terrain):
        return (terrain != 1) & (terrain != 2) & (terrain != 5)

    def legal_actions(self, s):
        hi,hn = self.target(s,"hall")
        si,sn = self.target(s,"stock")
        gi,gn = self.target(s,"gate")
        fi,fn = self.target(s,"field")
        bi,bn = self.target(s,"harbor")
        mi,mn = self.target(s,"market")
        ri,rn = self.target(s,"reservoir")
        di,dn = self.target(s,"gauge")
        empty = s["load"] == 0
        supply = (s["cargo"] == 1) & (s["load"] > 0)
        mask = jnp.zeros((320,21),jnp.bool_).at[:,0].set(True)
        mask = mask.at[:,1:5].set(self.movement_mask(s))
        mask = mask.at[:,5].set(sn & empty & (s["supplies"][si] >= 2))
        distinct = (s["credits"][hi] == 0) | (s["first_donor"][hi] != jnp.arange(320)+1)
        mask = mask.at[:,6].set(hn & supply & distinct & (s["credits"][hi] < 2) & (s["permit"][hi] == 0))
        mask = mask.at[:,7].set(hn & (s["tool"] == 3))
        mask = mask.at[:,8].set(hn & empty & (s["permit"][hi] > 0))
        mask = mask.at[:,9].set(gn & (s["cargo"] == 2) & (s["load"] > 0))
        mask = mask.at[:,10].set(rn & (s["cargo"] == 2) & (s["load"] > 0))
        on_gauge = jnp.all(s["pos"] == s["gauge_pos"][di],axis=1)
        gauge_occupied = jnp.any(jnp.all(s["gauge_pos"][:,None,:] == s["pos"][None,:,:],axis=2),axis=1)
        mask = mask.at[:,11].set(dn & ~on_gauge & ~gauge_occupied[di] & supply & (s["tool"] == 1) & (s["levee"][di] < 2))
        landing = s["harbor_pos"][bi] + jnp.array([-4,0])
        landing_free = ~jnp.any(jnp.all(landing[:,None,:] == s["pos"][None,:,:],axis=2),axis=1)
        mask = mask.at[:,12].set(bn & (s["tool"] == 4) & landing_free)
        mask = mask.at[:,13].set(fn & (s["tool"] == 2) & (s["water"][fi] > 0))
        mask = mask.at[:,14].set(fn & supply & (s["growth"][fi] == 0))
        mask = mask.at[:,15].set(fn & empty & (s["growth"][fi] == 8))
        mask = mask.at[:,16].set(mn & (s["cargo"] == 3) & (s["load"] > 0))
        here = s["tiles"][s["pos"][:,1],s["pos"][:,0]]
        mask = mask.at[:,17].set(supply & (s["tool"] == 1) & (here == 0))
        mask = mask.at[:,18].set(mn & supply & (s["market_level"][mi] < 2))
        distance = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=2)
        adjacent = jnp.any(distance == 1,axis=1)
        mask = mask.at[:,19].set(adjacent).at[:,20].set(adjacent)
        mask = mask & ((s["energy"] > 0) & ~s["done"])[:,None]
        return mask.at[:,0].set(True)

    def advance(self, key, s, a):
        n = dict(s)
        n["pos"] = self.move(s,a)
        hi,hn = self.target(s,"hall")
        si,sn = self.target(s,"stock")
        gi,gn = self.target(s,"gate")
        fi,fn = self.target(s,"field")
        bi,bn = self.target(s,"harbor")
        mi,mn = self.target(s,"market")
        ri,rn = self.target(s,"reservoir")
        di,dn = self.target(s,"gauge")
        collect = self.winner(si,a == 5,20)
        donate = self.winner(hi,a == 6,20)
        pickup = self.winner(hi,a == 8,20)
        open_gate = self.winner(gi,a == 9,20)
        allocate = self.winner(ri,a == 10,20)
        levee_work = self.winner(di,a == 11,20)
        irrigation = self.winner(fi,a == 13,20)
        plant = self.winner(fi,a == 14,20)
        harvest = self.winner(fi,a == 15,20)
        sell = self.winner(mi,a == 16,20)
        invest = self.winner(mi,a == 18,20)
        landing = s["harbor_pos"][bi]+jnp.array([-4,0])
        landing_free = ~jnp.any(jnp.all(landing[:,None,:] == n["pos"][None,:,:],axis=2),axis=1)
        ferry = self.winner(bi,(a == 12) & landing_free,20)
        n["pos"] = jnp.where(ferry[:,None],landing,n["pos"])
        n["ferry_count"] = s["ferry_count"]+self.tally(bi,ferry,20)
        terrain = s["tiles"][n["pos"][:,1],n["pos"][:,0]]
        walking_road = (a >= 1) & (a <= 4) & (terrain == 4)
        n["energy"] = jnp.clip(s["energy"]+jnp.where(a == 0,1,jnp.where(walking_road,0,-1)),0,20)
        n["tool"],paired = self.exchange(s,a,19)
        n["cargo"],n["load"],n["memory"] = self.cargo_swap(s,a,20)
        donated = self.tally(hi,donate,20)
        witnessed = self.tally(hi,a == 7,20) > 0
        base_credit = jnp.where(s["window"] > 0,s["credits"],0)
        credits = jnp.minimum(2,base_credit+donated)
        start = (base_credit == 0) & (donated > 0)
        first = jnp.zeros((20,),jnp.int32).at[hi].add(jnp.where(donate,jnp.arange(320)+1,0))
        n["first_donor"] = jnp.where(start,first,s["first_donor"])
        window = jnp.where(start,4,jnp.maximum(0,s["window"]-1))
        grant = (credits == 2) & witnessed & (window > 0) & (s["permit"] == 0)
        n["credits"] = jnp.where(grant | (window == 0),0,credits)
        n["window"] = jnp.where(grant,0,window)
        n["permit"] = s["permit"]+grant.astype(jnp.int32)-self.tally(hi,pickup,20)
        n["granted"] = grant.astype(jnp.int32)
        n["supplies"] = s["supplies"]-2*self.tally(si,collect,20)
        opened = self.tally(gi,open_gate,20) > 0
        n["sluice"] = jnp.where(opened,12,jnp.maximum(0,s["sluice"]-1))
        gauge_free = ~jnp.any(jnp.all(s["gauge_pos"][:,None,:] == n["pos"][None,:,:],axis=2),axis=1)
        levee_work = levee_work & gauge_free[di]
        n["levee"] = s["levee"]+self.tally(di,levee_work,20)
        flood = jnp.clip(s["pending"]+2*(s["sluice"] > 0)-s["levee"],0,6).astype(jnp.int32)
        n["flood"] = flood
        towns = jnp.arange(20)
        up = jnp.maximum(0,towns-1)
        redirected = jnp.clip(s["pending"] + s["levee"]*(s["pending"] > 0)-1,0,4)
        pulse = jnp.where((s["t"]+1+s["tide_phase"])%24 == 0,3,0)
        n["pending"] = jnp.where(towns%5 == 0,pulse,redirected[up]).astype(jnp.int32)
        irrigated = self.tally(fi,irrigation,20)
        n["water"] = jnp.clip(s["water"]+4*self.tally(ri,allocate,20)+(s["sluice"] > 0).astype(jnp.int32)-irrigated,0,8)
        n["wet"] = jnp.clip(s["wet"]+2*irrigated-jnp.where((s["t"]+1)%4 == 0,1,0),0,4)
        damage = (flood >= 3) & (s["growth"] > 0)
        grow = (s["growth"] > 0) & (s["wet"] > 0) & ((s["t"]+1)%4 == 0)
        growth = jnp.clip(s["growth"]+grow.astype(jnp.int32)-3*damage.astype(jnp.int32),0,8)
        n["growth"] = jnp.where(self.tally(fi,harvest,20) > 0,0,jnp.where(self.tally(fi,plant,20) > 0,1,growth)).astype(jnp.int32)
        n["damage"] = damage.astype(jnp.int32)
        n["market_level"] = s["market_level"]+self.tally(mi,invest,20)
        sold = self.tally(mi,sell,20)
        n["sold"] = sold
        n["sales"] = s["sales"]+sold
        gp,dp = s["gate_pos"],s["gauge_pos"]
        x,y = s["pos"][:,0],s["pos"][:,1]
        road = a == 17
        n["tiles"] = s["tiles"].at[y,x].set(jnp.where(road,4,s["tiles"][y,x]))
        n["tiles"] = n["tiles"].at[gp[:,1],gp[:,0]].set(jnp.where(n["sluice"] > 0,3,2))
        n["tiles"] = n["tiles"].at[dp[:,1],dp[:,0]].set(jnp.where(n["levee"] > 0,5,n["tiles"][dp[:,1],dp[:,0]]))
        consumed = donate | open_gate | allocate | levee_work | plant | sell | road | invest
        n["load"] = jnp.where(collect,2,jnp.where(pickup | harvest,1,jnp.maximum(0,n["load"]-consumed.astype(jnp.int32))))
        n["cargo"] = jnp.where(collect,1,jnp.where(pickup,2,jnp.where(harvest,3,jnp.where(n["load"] == 0,0,n["cargo"]))))
        n["memory"] = jnp.where(pickup,1,jnp.where(allocate,2,jnp.where(open_gate,3,n["memory"])))
        rewards = -0.01*(a > 0)-0.02*(donate | levee_work | road | invest)
        rewards = rewards+sell*(1.0+0.25*s["market_level"][mi])+0.04*sold[s["team"]]-0.08*damage[s["team"]]
        natural = jnp.sum(n["sales"]) >= 40
        return n,rewards.astype(jnp.float32),natural

    def observe(self, s):
        board = jnp.zeros((48, 70, 33), jnp.float32)
        board = board.at[:, :, 0].set(s['tiles'].astype(jnp.float32) / 10.0)
        vals = jnp.stack(((jnp.arange(self.num_agents, dtype=jnp.float32) + 1) / self.num_agents,
                          (s['role'] + 1) / 8.0, (s['team'] + 1) / 20.0,
                          s['tool'] / 4.0, s['cargo'] / 4.0, s['load'] / 4.0,
                          s['energy'] / 20.0, s['memory'] / 4.0), axis=1).astype(jnp.float32)
        board = board.at[s['pos'][:, 1], s['pos'][:, 0], 1:9].set(vals)
        values = jnp.stack(((jnp.arange(20, dtype=jnp.float32) + 1) / 20.0, s['credits'].astype(jnp.float32) / 2.0, s['window'].astype(jnp.float32) / 4.0, s['first_donor'].astype(jnp.float32) / 320.0, s['permit'].astype(jnp.float32) / 1.0), axis=1)
        board = board.at[s['hall_pos'][:, 1], s['hall_pos'][:, 0], 9:14].set(values)
        values = jnp.stack(((jnp.arange(20, dtype=jnp.float32) + 1) / 20.0, s['supplies'].astype(jnp.float32) / 32.0), axis=1)
        board = board.at[s['stock_pos'][:, 1], s['stock_pos'][:, 0], 14:16].set(values)
        values = jnp.stack(((jnp.arange(20, dtype=jnp.float32) + 1) / 20.0, s['sluice'].astype(jnp.float32) / 12.0), axis=1)
        board = board.at[s['gate_pos'][:, 1], s['gate_pos'][:, 0], 16:18].set(values)
        values = jnp.stack(((jnp.arange(20, dtype=jnp.float32) + 1) / 20.0, s['growth'].astype(jnp.float32) / 8.0, s['wet'].astype(jnp.float32) / 4.0), axis=1)
        board = board.at[s['field_pos'][:, 1], s['field_pos'][:, 0], 18:21].set(values)
        values = jnp.stack(((jnp.arange(20, dtype=jnp.float32) + 1) / 20.0, s['ferry_count'].astype(jnp.float32) / 1024.0), axis=1)
        board = board.at[s['harbor_pos'][:, 1], s['harbor_pos'][:, 0], 21:23].set(values)
        values = jnp.stack(((jnp.arange(20, dtype=jnp.float32) + 1) / 20.0, s['market_level'].astype(jnp.float32) / 2.0, s['sales'].astype(jnp.float32) / 40.0), axis=1)
        board = board.at[s['market_pos'][:, 1], s['market_pos'][:, 0], 23:26].set(values)
        values = jnp.stack(((jnp.arange(20, dtype=jnp.float32) + 1) / 20.0, s['water'].astype(jnp.float32) / 8.0), axis=1)
        board = board.at[s['reservoir_pos'][:, 1], s['reservoir_pos'][:, 0], 26:28].set(values)
        values = jnp.stack(((jnp.arange(20, dtype=jnp.float32) + 1) / 20.0, s['levee'].astype(jnp.float32) / 2.0, s['pending'].astype(jnp.float32) / 4.0, s['flood'].astype(jnp.float32) / 6.0, s['tide_phase'].astype(jnp.float32) / 7.0), axis=1)
        board = board.at[s['gauge_pos'][:, 1], s['gauge_pos'][:, 0], 28:33].set(values)
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
        s["hall_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["hall_pos"][None, :, :]) <= 2, axis=2)
        s["stock_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["stock_pos"][None, :, :]) <= 2, axis=2)
        s["gate_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["gate_pos"][None, :, :]) <= 2, axis=2)
        s["field_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["field_pos"][None, :, :]) <= 2, axis=2)
        s["harbor_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["harbor_pos"][None, :, :]) <= 2, axis=2)
        s["market_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["market_pos"][None, :, :]) <= 2, axis=2)
        s["reservoir_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["reservoir_pos"][None, :, :]) <= 2, axis=2)
        s["gauge_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["gauge_pos"][None, :, :]) <= 2, axis=2)
        return s

