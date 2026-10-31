"""Mosaic Courtyard. Original source-only authoring; no runtime claims."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 157
    num_actions = 12
    max_steps = 128

    def ports(self,kind,orientation):
        pattern = jnp.array([[1,0,1,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]],jnp.int32)
        return pattern[kind[:,None],jnp.mod(jnp.arange(4,dtype=jnp.int32)[None,:]-orientation[:,None],4)]

    def refresh(self,state):
        s = dict(state)
        owner = s["tile_holder"]
        s["tile_pos"] = jnp.where((owner>=0)[:,None],s["pos"][jnp.maximum(owner,0)],s["tile_pos"])
        s["tile_ports"] = self.ports(s["tile_kind"],s["tile_orientation"])
        return s

    def reset(self,key):
        s = {
            "t": jnp.array(0,jnp.int32), "done": jnp.array(False), "natural": jnp.array(False),
            "map": jnp.array([[1,1,1,1,1,1,1],[1,0,0,0,0,0,1],
                [1,0,2,2,2,0,1],[1,0,2,2,2,0,1],[1,0,2,2,2,0,1],
                [1,0,0,0,0,0,1],[1,1,1,1,1,1,1]],jnp.int32),
            "pos": jnp.array([[1,2],[3,1],[5,4]],jnp.int32),
            "active": jnp.ones(3,jnp.bool_), "visible": jnp.ones((3,3),jnp.bool_),
            "facing": jnp.array([1,2,3],jnp.int32),
            "pins": jnp.full(3,3,jnp.int32), "delivered": jnp.zeros(3,jnp.bool_),
            "preference": jax.random.randint(key,(3,),0,3,dtype=jnp.int32),
            "tile_pos": jnp.array([[1,2],[3,1],[5,4],[1,3],[1,3],[1,3],[1,3],[1,3],[1,3]],jnp.int32),
            "tile_active": jnp.ones(9,jnp.bool_), "tile_visible": jnp.ones((3,9),jnp.bool_),
            "tile_status": jnp.array([1,1,1,0,0,0,0,0,0],jnp.int32),
            "tile_holder": jnp.array([0,1,2,-1,-1,-1,-1,-1,-1],jnp.int32),
            "tile_kind": jnp.array([0,1,2,0,1,1,2,3,0],jnp.int32),
            "tile_orientation": jnp.array([0,2,0,1,0,1,1,0,0],jnp.int32),
            "tile_ports": jnp.zeros((9,4),jnp.int32),
            "depot_pos": jnp.array([[1,3]],jnp.int32),
            "depot_active": jnp.ones(1,jnp.bool_), "depot_visible": jnp.ones((3,1),jnp.bool_),
            "goal_pos": jnp.array([[4,2],[4,4],[2,4]],jnp.int32),
            "goal_active": jnp.ones(3,jnp.bool_), "goal_visible": jnp.ones((3,3),jnp.bool_),
            "goal_id": jnp.array([0,1,2],jnp.int32),
            "place_event": jnp.zeros(9,jnp.bool_), "rotate_event": jnp.zeros(9,jnp.bool_),
            "claim_event": jnp.zeros(3,jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s),s

    def tile_at(self,s,points):
        hit = jnp.all(points[...,None,:]==s["tile_pos"],axis=-1)&(s["tile_status"]==2)
        return jnp.argmax(hit,axis=-1),jnp.any(hit,axis=-1)

    def fits(self,s,point,ports):
        directions = jnp.array([[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        neighbor,exists = self.tile_at(s,point[...,None,:]+directions)
        opposite = jnp.array([2,3,0,1],jnp.int32)
        matching = ports==s["tile_ports"][neighbor,opposite]
        return jnp.all(~exists|matching,axis=-1)

    def movement(self,s):
        d = jnp.array([[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        p = s["pos"][:,None,:]+d[None,:,:]
        inside = jnp.all((p>=0)&(p<=6),axis=-1)
        terrain = s["map"][jnp.clip(p[:,:,1],0,6),jnp.clip(p[:,:,0],0,6)]
        current,on_tile = self.tile_at(s,s["pos"])
        target,has_target = self.tile_at(s,p)
        outgoing = s["tile_ports"][current[:,None],jnp.arange(4,dtype=jnp.int32)[None,:]]==1
        incoming = s["tile_ports"][target,jnp.array([2,3,0,1],jnp.int32)[None,:]]==1
        enter = (terrain==0)|((terrain==2)&has_target&incoming)
        leave = ~on_tile[:,None]|outgoing
        occupied = jnp.any(jnp.all(p[:,:,None,:]==s["pos"][None,None,:,:],axis=-1),axis=-1)
        return inside&enter&leave&~occupied

    def options(self,s):
        ids = jnp.arange(3,dtype=jnp.int32)
        held = s["tile_holder"][None,:]==ids[:,None]
        has = jnp.any(held,axis=1)
        carried = jnp.argmax(held,axis=1)
        directions = jnp.array([[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        front = s["pos"]+directions[s["facing"]]
        front_tile,front_exists = self.tile_at(s,front)
        in_court = jnp.all((front>=2)&(front<=4),axis=1)
        occupied = jnp.any(jnp.all(front[:,None,:]==s["pos"][None,:,:],axis=-1),axis=1)
        fit = self.fits(s,front,s["tile_ports"][carried])
        place = has&in_court&~front_exists&~occupied&fit
        rotating = jnp.where(has,carried,front_tile)
        proposed_ports = self.ports(s["tile_kind"],jnp.mod(s["tile_orientation"]+1,4))
        fits_rotated = self.fits(s,front,proposed_ports[rotating])
        rotate = (s["pins"]>0)&(has|(front_exists&~occupied&fits_rotated))
        at_depot = jnp.all(s["pos"]==s["depot_pos"][0],axis=1)
        take = at_depot&~has&jnp.any(s["tile_status"]==0)
        putback = at_depot&has
        adjacent = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=-1)==1
        passing = adjacent&has[:,None]&~has[None,:]
        goals = jnp.all(s["pos"][:,None,:]==s["goal_pos"][None,:,:],axis=-1)
        claim = jnp.any(goals,axis=1)&~s["delivered"]
        return carried,has,front,place,rotating,rotate,take,putback,passing,goals,claim

    def legal_actions(self,s):
        carried,has,front,place,rotating,rotate,take,putback,passing,goals,claim = self.options(s)
        m = jnp.concatenate((jnp.ones((3,1),jnp.bool_),self.movement(s),
            jnp.stack((jnp.ones(3,jnp.bool_),take,place,rotate,jnp.any(passing,1),claim,putback),axis=1)),axis=1)
        return m & ((~s["done"])|(jnp.arange(12)[None,:]==0))

    def observe(self,s):
        pub = jnp.concatenate((s["map"].reshape(-1),s["pos"].reshape(-1),s["facing"],
            s["tile_pos"].reshape(-1),s["tile_status"],s["tile_holder"],s["tile_kind"],
            s["tile_orientation"],s["tile_ports"].reshape(-1),s["pins"],s["delivered"].astype(jnp.int32),
            s["t"].reshape(1))).astype(jnp.float32)
        return jnp.concatenate((jnp.broadcast_to(pub,(3,155)),s["preference"][:,None].astype(jnp.float32),
            jnp.arange(3,dtype=jnp.float32)[:,None]),axis=1)

    def raw_step(self,key,state,actions):
        legal = self.legal_actions(state)
        a = jnp.clip(actions,0,11)
        valid = (actions>=0)&(actions<12)&jnp.take_along_axis(legal,a[:,None],axis=1)[:,0]
        a = jnp.where(valid,a,0)
        s = dict(state)
        ids = jnp.arange(3,dtype=jnp.int32)
        priority = jnp.mod(ids-state["t"],3)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        proposed = state["pos"]+delta[jnp.where(a<=4,a,0)]
        same = jnp.all(proposed[:,None,:]==proposed[None,:,:],axis=-1)
        loses = jnp.any(same&(priority[None,:]<priority[:,None]),axis=1)
        s["pos"] = jnp.where(loses[:,None],state["pos"],proposed)
        s["facing"] = jnp.where((a>=1)&(a<=4),a-1,s["facing"])
        s["facing"] = jnp.where(a==5,jnp.mod(s["facing"]+1,4),s["facing"])
        s["place_event"] = jnp.zeros(9,jnp.bool_)
        s["rotate_event"] = jnp.zeros(9,jnp.bool_)
        s["claim_event"] = jnp.zeros(3,jnp.bool_)
        s = self.refresh(s)
        r = jnp.zeros(3,jnp.float32)
        # Placement/rotation conflict resolution rechecks fresh local edge constraints.
        for k in range(3):
            i = jnp.mod(state["t"]+k,3)
            carried,has,front,place,rotating,rotate,take,putback,passing,goals,claim = self.options(s)
            h = carried[i]
            free = jnp.argmax(s["tile_status"]==0)
            taking = (a[i]==6)&take[i]
            s["tile_status"] = s["tile_status"].at[free].set(jnp.where(taking,1,s["tile_status"][free]))
            s["tile_holder"] = s["tile_holder"].at[free].set(jnp.where(taking,i,s["tile_holder"][free]))
            building = (a[i]==7)&place[i]
            s["tile_pos"] = s["tile_pos"].at[h].set(jnp.where(building,front[i],s["tile_pos"][h]))
            s["tile_status"] = s["tile_status"].at[h].set(jnp.where(building,2,s["tile_status"][h]))
            s["tile_holder"] = s["tile_holder"].at[h].set(jnp.where(building,-1,s["tile_holder"][h]))
            s["place_event"] = s["place_event"].at[h].set(s["place_event"][h]|building)
            q = rotating[i]
            turning = (a[i]==8)&rotate[i]
            s["tile_orientation"] = s["tile_orientation"].at[q].set(
                jnp.where(turning,jnp.mod(s["tile_orientation"][q]+1,4),s["tile_orientation"][q]))
            s["pins"] = s["pins"].at[i].add(-turning.astype(jnp.int32))
            s["rotate_event"] = s["rotate_event"].at[q].set(s["rotate_event"][q]|turning)
            recipients = passing[i]&(a==0)
            j = jnp.argmin(jnp.where(recipients,priority,4))
            handoff = (a[i]==9)&jnp.any(recipients)
            s["tile_holder"] = s["tile_holder"].at[h].set(jnp.where(handoff,j,s["tile_holder"][h]))
            returning = (a[i]==11)&putback[i]
            s["tile_holder"] = s["tile_holder"].at[h].set(jnp.where(returning,-1,s["tile_holder"][h]))
            s["tile_status"] = s["tile_status"].at[h].set(jnp.where(returning,0,s["tile_status"][h]))
            s["tile_pos"] = s["tile_pos"].at[h].set(jnp.where(returning,s["depot_pos"][0],s["tile_pos"][h]))
            claiming = (a[i]==10)&claim[i]
            g = jnp.argmax(goals[i])
            s["delivered"] = s["delivered"].at[i].set(s["delivered"][i]|claiming)
            s["claim_event"] = s["claim_event"].at[i].set(claiming)
            r = r.at[i].add(claiming.astype(jnp.float32)*(1.0+(s["preference"][i]==g).astype(jnp.float32)))
            s = self.refresh(s)
        r = r+0.2*jnp.sum(s["claim_event"].astype(jnp.float32))
        s["t"] = state["t"]+1
        s["natural"] = jnp.all(s["delivered"])
        s["done"] = s["natural"]|(s["t"]>=self.max_steps)
        s = jax.tree.map(lambda new,old:jnp.where(state["done"],old,new),s,state)
        r = jnp.where(state["done"],jnp.zeros(3,jnp.float32),r)
        terminated = s["natural"]
        truncated = (s["t"]>=self.max_steps)&~terminated
        done = terminated|truncated
        return self.observe(s),s,r,jnp.full((3,),done,jnp.bool_),{
            "done_all":done,"terminated":terminated,"truncated":truncated}
