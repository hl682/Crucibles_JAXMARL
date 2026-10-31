"""Original blind-authored train-only spatial kernel; not executed by the author."""
import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 128
    observation_size = 858
    num_actions = 23
    max_steps = 896
    width = 64
    height = 40


    def reset(self, key):
        ids = jnp.arange(128,dtype=jnp.int32)
        team,slot = ids//4,ids%4
        homes = jnp.arange(32,dtype=jnp.int32)
        bases = jnp.stack(((homes%8)*8,(homes//8)*10),axis=1)
        offsets = jnp.array([[3,2],[4,4],[2,4],[5,5]],jnp.int32)
        s = self.initial(bases[team]+offsets[slot],team,slot)
        s["cargo"] = jnp.where(slot == 0,1,jnp.where(slot == 2,2,jnp.where(slot == 3,3,0))).astype(jnp.int32)
        s["load"] = jnp.where(s["cargo"] > 0,2,0).astype(jnp.int32)
        tree_ids = jnp.arange(64,dtype=jnp.int32)
        s["tree_pos"] = bases[tree_ids//2]+jnp.stack((3+2*(tree_ids%2),jnp.full((64,),3,jnp.int32)),axis=1)
        s["tree_active"] = jnp.ones((64,),jnp.bool_)
        for name,offset in (("hive",(4,5)),("bridge",(7,5)),("depot",(1,3)),("well",(1,5)),("store",(2,7)),("bee",(4,5)),("herd",(6,7))):
            s[name+"_pos"] = bases+jnp.array(offset,jnp.int32)
            s[name+"_active"] = jnp.ones((32,),jnp.bool_)
        s["species"] = tree_ids%2
        s["growth"] = jnp.full((64,),4,jnp.int32)
        s["moisture"] = jnp.full((64,),3,jnp.int32)
        s["pests"] = jax.random.randint(key,(64,),0,3,dtype=jnp.int32)
        s["tree_wind"] = jnp.zeros((64,),jnp.int32)
        for name in ("pollen","poison","fruit","pollinated","trampled","pollen_queue","pollen_next"):
            s[name] = jnp.zeros((64,),jnp.int32)
        s["health"] = jnp.full((32,),2,jnp.int32)
        s["nectar"] = jnp.full((32,),4,jnp.int32)
        s["bridge_health"] = jnp.ones((32,),jnp.int32)
        s["stock"] = jnp.full((32,),16,jnp.int32)
        s["water"] = jnp.full((32,),8,jnp.int32)
        for name in ("carrier","deliveries","bee_kind","guided","care_event","repair_event"):
            s[name] = jnp.zeros((32,),jnp.int32)
        for x in range(8):
            s["tiles"] = s["tiles"].at[:,x*8+7].set(1)
        bp = s["bridge_pos"]
        s["tiles"] = s["tiles"].at[bp[:,1],bp[:,0]].set(2)
        s = self.refresh(s)
        return self.observe(s),s

    def passable(self, terrain):
        return (terrain != 1) & (terrain != 3)

    def legal_actions(self, s):
        ti,tn = self.target(s,"tree")
        hi,hn = self.target(s,"hive")
        bi,bn = self.target(s,"bridge")
        di,dn = self.target(s,"depot")
        wi,wn = self.target(s,"well")
        si,sn = self.target(s,"store")
        ai,an = self.target(s,"herd")
        empty = s["load"] == 0
        mask = jnp.zeros((128,23),jnp.bool_).at[:,0].set(True)
        moves = self.movement_mask(s)
        destinations = s["pos"][:,None,:]+jnp.array([[1,0],[-1,0],[0,1],[0,-1]],jnp.int32)[None,:,:]
        hive_cell = jnp.any(jnp.all(destinations[:,:,None,:] == s["hive_pos"][None,None,:,:],axis=3),axis=2)
        mask = mask.at[:,1:5].set(moves & ~hive_cell)
        mask = mask.at[:,5].set(dn & empty & (s["stock"][di] > 0))
        mask = mask.at[:,6].set(tn & (s["tool"] == 1) & (s["growth"][ti] >= 2) & (s["fruit"][ti] == 0))
        mask = mask.at[:,7].set(tn & (s["tool"] == 1))
        mask = mask.at[:,8].set(tn & (s["cargo"] == 2) & (s["load"] > 0))
        mask = mask.at[:,9].set(tn & (s["cargo"] == 1) & (s["load"] > 0))
        mask = mask.at[:,10].set(tn & empty & (s["fruit"][ti] > 0))
        mask = mask.at[:,11].set(hn & (s["tool"] == 2) & (s["nectar"][hi] > 0))
        mask = mask.at[:,12].set(hn & (s["tool"] == 1))
        mask = mask.at[:,13].set(bn & (s["tool"] == 4) & (s["cargo"] == 3) & (s["load"] > 0))
        mask = mask.at[:,14].set(bn & (s["tool"] == 3))
        carrying = jnp.any(s["carrier"][None,:] == (jnp.arange(128)+1)[:,None],axis=1)
        mask = mask.at[:,15].set(hn & (s["tool"] == 2) & (s["carrier"][hi] == 0) & ~carrying)
        mask = mask.at[:,16].set(carrying)
        mask = mask.at[:,17].set(wn & empty & (s["water"][wi] > 0))
        mask = mask.at[:,18].set(an & (s["tool"] == 3))
        mask = mask.at[:,19].set(sn & (s["cargo"] == 4) & (s["load"] > 0))
        distance = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=2)
        adjacent = jnp.any(distance == 1,axis=1)
        mask = mask.at[:,20].set(adjacent).at[:,21].set(adjacent)
        mask = mask.at[:,22].set(tn & (s["cargo"] == 1) & (s["load"] > 0) & (s["growth"][ti] == 0))
        mask = mask & ((s["energy"] > 0) & ~s["done"])[:,None]
        return mask.at[:,0].set(True)

    def advance(self, key, s, a):
        n = dict(s)
        n["pos"] = self.move(s,a)
        n["energy"] = jnp.clip(s["energy"]+jnp.where(a == 0,1,-1)-((a == 6) | (a == 7)).astype(jnp.int32),0,20)
        n["tool"],paired = self.exchange(s,a,20)
        n["cargo"],n["load"],n["memory"] = self.cargo_swap(s,a,21)
        ti,tn = self.target(s,"tree")
        hi,hn = self.target(s,"hive")
        bi,bn = self.target(s,"bridge")
        di,dn = self.target(s,"depot")
        wi,wn = self.target(s,"well")
        si,sn = self.target(s,"store")
        ai,an = self.target(s,"herd")
        collect = self.winner(di,a == 5,32)
        prune = self.winner(ti,a == 6,64)
        spray = self.winner(ti,a == 7,64)
        watering = self.winner(ti,a == 8,64)
        compost = self.winner(ti,a == 9,64)
        harvest = self.winner(ti,a == 10,64)
        repair = self.winner(bi,a == 13,32)
        lift = self.winner(hi,a == 15,32)
        fill = self.winner(wi,a == 17,32)
        sell = self.winner(si,a == 19,32)
        plant = self.winner(ti,a == 22,64)
        care = self.tally(hi,a == 11,32) > 0
        shade = self.tally(hi,a == 12,32) > 0
        cared = care & shade & (s["nectar"] > 0)
        brace = self.tally(bi,a == 14,32) > 0
        repaired = (self.tally(bi,repair,32) > 0) & brace
        repair = repair & repaired[bi]
        lifted_by = jnp.zeros((32,),jnp.int32).at[hi].add(jnp.where(lift,jnp.arange(128)+1,0))
        lifted = self.tally(hi,lift,32) > 0
        previous_actor = jnp.maximum(0,s["carrier"]-1)
        dropped = (s["carrier"] > 0) & (a[previous_actor] == 16)
        carrier = jnp.where(dropped,0,jnp.where(lifted,lifted_by,s["carrier"]))
        n["carrier"] = carrier
        carrier_actor = jnp.maximum(0,carrier-1)
        n["hive_pos"] = jnp.where((carrier > 0)[:,None],n["pos"][carrier_actor],s["hive_pos"])
        sprayed = self.tally(ti,spray,64) > 0
        tree_distance = jnp.sum(jnp.abs(s["tree_pos"][:,None,:]-s["tree_pos"][None,:,:]),axis=2)
        spray_reaches = jnp.any((tree_distance <= 6) & sprayed[None,:],axis=1)
        poison = jnp.clip(s["poison"]-1+3*spray_reaches.astype(jnp.int32),0,6)
        n["poison"] = poison
        n["pests"] = jnp.clip(s["pests"]-3*sprayed.astype(jnp.int32)+jnp.where((s["t"]+1)%16 == 0,1,0),0,4)
        hive_distance = jnp.sum(jnp.abs(s["hive_pos"][:,None,:]-s["tree_pos"][None,:,:]),axis=2)
        poisoned_hive = jnp.any((hive_distance <= 5) & (poison[None,:] > 0),axis=1)
        n["health"] = jnp.clip(s["health"]+2*cared.astype(jnp.int32)-poisoned_hive.astype(jnp.int32),0,4)
        nectar_gain = ((s["t"]+1)%16 == 0) & jnp.any((hive_distance <= 5) & (s["growth"][None,:] >= 4),axis=1)
        n["nectar"] = jnp.clip(s["nectar"]-cared.astype(jnp.int32)+nectar_gain.astype(jnp.int32),0,4)
        n["care_event"] = cared.astype(jnp.int32)
        n["repair_event"] = repaired.astype(jnp.int32)
        erosion = (s["t"]+1)%32 == 0
        n["bridge_health"] = jnp.clip(s["bridge_health"]+2*repaired.astype(jnp.int32)-erosion.astype(jnp.int32),0,3)
        bp = s["bridge_pos"]
        n["tiles"] = s["tiles"].at[bp[:,1],bp[:,0]].set(jnp.where(n["bridge_health"] > 0,2,3))
        kind = jnp.full((32,),((s["t"]+1)//8)%2,jnp.int32)
        desired = s["species"][None,:] == kind[:,None]
        nearest = jnp.argmin(jnp.where(desired,hive_distance,1000),axis=1)
        goal = s["tree_pos"][nearest]
        delta = goal-s["bee_pos"]
        bee_step = jnp.stack((jnp.sign(delta[:,0]),jnp.where(delta[:,0] == 0,jnp.sign(delta[:,1]),0)),axis=1).astype(jnp.int32)
        bee_destination = s["bee_pos"]+bee_step
        bee_ids = jnp.arange(32)
        old_occupied = jnp.any(jnp.all(bee_destination[:,None,:] == s["bee_pos"][None,:,:],axis=2) & (bee_ids[:,None] != bee_ids[None,:]),axis=1)
        bee_moving = jnp.any(bee_step != 0,axis=1)
        same_target = jnp.all(bee_destination[:,None,:] == bee_destination[None,:,:],axis=2)
        bee_lost = jnp.any(same_target & (bee_ids[None,:] < bee_ids[:,None]) & bee_moving[None,:],axis=1)
        n["bee_pos"] = jnp.where((~old_occupied & ~bee_lost)[:,None],bee_destination,s["bee_pos"])
        n["bee_kind"] = kind
        bee_distance = jnp.sum(jnp.abs(s["tree_pos"][:,None,:]-s["bee_pos"][None,:,:]),axis=2)
        nearby_live_bee = jnp.any((bee_distance <= 1) & (s["health"][None,:] >= 2),axis=1)
        good_hive = jnp.any((hive_distance.T <= 5) & (s["health"][None,:] >= 2),axis=1)
        emission = (s["growth"] >= 4) & (s["growth"] <= 6) & (s["poison"] == 0) & nearby_live_bee & good_hive
        dx = s["tree_pos"][None,:,0]-s["tree_pos"][:,None,0]
        dy = jnp.abs(s["tree_pos"][None,:,1]-s["tree_pos"][:,None,1])
        compatible = s["species"][:,None] != s["species"][None,:]
        downwind = jnp.where(s["tree_wind"][:,None] == 0,dx > 0,dx < 0)
        links = compatible & downwind & (jnp.abs(dx) <= 8) & (dy <= 2) & emission[:,None]
        incoming = jnp.minimum(4,jnp.sum(links,axis=0)).astype(jnp.int32)
        received = jnp.where(poison == 0,s["pollen_queue"],0)
        n["pollen_queue"] = s["pollen_next"]
        n["pollen_next"] = incoming
        n["pollinated"] = (received > 0).astype(jnp.int32)
        n["pollen"] = jnp.clip(s["pollen"]+received-jnp.where((s["t"]+1)%16 == 0,1,0),0,4)
        n["tree_wind"] = jnp.where(erosion,1-s["tree_wind"],s["tree_wind"])
        irrigated = self.tally(ti,watering,64)
        composted = self.tally(ti,compost,64)
        rain = (s["t"]+1)%24 == 0
        n["moisture"] = jnp.clip(s["moisture"]+2*irrigated+composted+rain.astype(jnp.int32)-jnp.where((s["t"]+1)%4 == 0,1,0),0,4)
        guided = self.tally(ai,a == 18,32) > 0
        n["guided"] = guided.astype(jnp.int32)
        homes = jnp.arange(32)
        bases = jnp.stack(((homes%8)*8,(homes//8)*10),axis=1)
        roam_y = 3+((s["t"]+1)//4)%5
        roam_x = 4+((s["t"]+1)//8)%3
        n["herd_pos"] = bases+jnp.stack((jnp.where(guided,6,roam_x),jnp.where(guided,7,roam_y)),axis=1)
        animal_distance = jnp.sum(jnp.abs(s["tree_pos"][:,None,:]-s["herd_pos"][None,:,:]),axis=2)
        trampled = (s["growth"] < 4) & (s["growth"] > 0) & jnp.any((animal_distance <= 1) & ~guided[None,:],axis=1)
        n["trampled"] = trampled.astype(jnp.int32)
        grow = (s["growth"] > 0) & (s["moisture"] > 0) & ((s["t"]+1)%4 == 0)
        pests_hurt = (s["pests"] >= 3) & (s["poison"] == 0) & ((s["t"]+1)%8 == 0)
        growth = jnp.clip(s["growth"]+grow.astype(jnp.int32)+composted-trampled.astype(jnp.int32)-pests_hurt.astype(jnp.int32),0,8)
        ripe = (growth >= 6) & (s["pollen"] > 0) & (s["fruit"] < 3)
        growth = jnp.where(ripe,2,growth)
        growth = jnp.where(self.tally(ti,prune,64) > 0,4,growth)
        n["growth"] = jnp.where(self.tally(ti,plant,64) > 0,1,growth).astype(jnp.int32)
        n["fruit"] = s["fruit"]+ripe.astype(jnp.int32)-self.tally(ti,harvest,64)
        n["pollen"] = jnp.where(ripe,0,n["pollen"])
        n["stock"] = s["stock"]-self.tally(di,collect,32)
        n["water"] = jnp.minimum(8,s["water"]-self.tally(wi,fill,32)+4*rain.astype(jnp.int32))
        sold = self.tally(si,sell,32)
        n["deliveries"] = s["deliveries"]+sold
        consumed = watering | compost | repair | sell | plant
        n["load"] = jnp.where(collect | fill | harvest,1,jnp.maximum(0,n["load"]-consumed.astype(jnp.int32)))
        n["cargo"] = jnp.where(collect,jnp.where(s["tool"] == 4,3,1),jnp.where(fill,2,jnp.where(harvest,4,jnp.where(n["load"] == 0,0,n["cargo"]))))
        n["memory"] = jnp.where(harvest,jnp.maximum(1,s["pollen"][ti]),n["memory"])
        rewards = -0.01*(a > 0)-0.03*spray-0.02*(compost | repair | plant)
        rewards = rewards+sell*(1.0+0.1*s["memory"])+0.04*sold[s["team"]]+0.01*cared[s["team"]]-0.02*poisoned_hive[s["team"]]
        natural = jnp.sum(n["deliveries"]) >= 64
        return n,rewards.astype(jnp.float32),natural

    def observe(self, s):
        board = jnp.zeros((40, 64, 34), jnp.float32)
        board = board.at[:, :, 0].set(s['tiles'].astype(jnp.float32) / 10.0)
        vals = jnp.stack(((jnp.arange(self.num_agents, dtype=jnp.float32) + 1) / self.num_agents,
                          (s['role'] + 1) / 4.0, (s['team'] + 1) / 32.0,
                          s['tool'] / 4.0, s['cargo'] / 4.0, s['load'] / 4.0,
                          s['energy'] / 20.0, s['memory'] / 4.0), axis=1).astype(jnp.float32)
        board = board.at[s['pos'][:, 1], s['pos'][:, 0], 1:9].set(vals)
        values = jnp.stack(((jnp.arange(64, dtype=jnp.float32) + 1) / 64.0, s['species'].astype(jnp.float32) / 1.0, s['growth'].astype(jnp.float32) / 8.0, s['moisture'].astype(jnp.float32) / 4.0, s['pollen'].astype(jnp.float32) / 4.0, s['poison'].astype(jnp.float32) / 6.0, s['pests'].astype(jnp.float32) / 4.0, s['fruit'].astype(jnp.float32) / 3.0, s['tree_wind'].astype(jnp.float32) / 1.0), axis=1)
        board = board.at[s['tree_pos'][:, 1], s['tree_pos'][:, 0], 9:18].set(values)
        values = jnp.stack(((jnp.arange(32, dtype=jnp.float32) + 1) / 32.0, s['health'].astype(jnp.float32) / 4.0, s['nectar'].astype(jnp.float32) / 4.0, s['carrier'].astype(jnp.float32) / 128.0), axis=1)
        board = board.at[s['hive_pos'][:, 1], s['hive_pos'][:, 0], 18:22].set(values)
        values = jnp.stack(((jnp.arange(32, dtype=jnp.float32) + 1) / 32.0, s['bridge_health'].astype(jnp.float32) / 3.0), axis=1)
        board = board.at[s['bridge_pos'][:, 1], s['bridge_pos'][:, 0], 22:24].set(values)
        values = jnp.stack(((jnp.arange(32, dtype=jnp.float32) + 1) / 32.0, s['stock'].astype(jnp.float32) / 16.0), axis=1)
        board = board.at[s['depot_pos'][:, 1], s['depot_pos'][:, 0], 24:26].set(values)
        values = jnp.stack(((jnp.arange(32, dtype=jnp.float32) + 1) / 32.0, s['water'].astype(jnp.float32) / 8.0), axis=1)
        board = board.at[s['well_pos'][:, 1], s['well_pos'][:, 0], 26:28].set(values)
        values = jnp.stack(((jnp.arange(32, dtype=jnp.float32) + 1) / 32.0, s['deliveries'].astype(jnp.float32) / 64.0), axis=1)
        board = board.at[s['store_pos'][:, 1], s['store_pos'][:, 0], 28:30].set(values)
        values = jnp.stack(((jnp.arange(32, dtype=jnp.float32) + 1) / 32.0, s['bee_kind'].astype(jnp.float32) / 1.0), axis=1)
        board = board.at[s['bee_pos'][:, 1], s['bee_pos'][:, 0], 30:32].set(values)
        values = jnp.stack(((jnp.arange(32, dtype=jnp.float32) + 1) / 32.0, s['guided'].astype(jnp.float32) / 1.0), axis=1)
        board = board.at[s['herd_pos'][:, 1], s['herd_pos'][:, 0], 32:34].set(values)
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
        s["tree_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["tree_pos"][None, :, :]) <= 2, axis=2)
        s["hive_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["hive_pos"][None, :, :]) <= 2, axis=2)
        s["bridge_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["bridge_pos"][None, :, :]) <= 2, axis=2)
        s["depot_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["depot_pos"][None, :, :]) <= 2, axis=2)
        s["well_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["well_pos"][None, :, :]) <= 2, axis=2)
        s["store_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["store_pos"][None, :, :]) <= 2, axis=2)
        s["bee_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["bee_pos"][None, :, :]) <= 2, axis=2)
        s["herd_visibility"] = jnp.all(jnp.abs(s["pos"][:, None, :] - s["herd_pos"][None, :, :]) <= 2, axis=2)
        return s

