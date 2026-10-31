"""Ember Shelter: original unexecuted source for shared warmth and finite materials."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 96
    num_actions = 16
    max_steps = 54

    def reset(self, key):
        k1,k2 = jax.random.split(key)
        s = {
            "t":jnp.array(0,jnp.int32),"done":jnp.array(False),
            "terminated":jnp.array(False),"truncated":jnp.array(False),
            "tile":jnp.array([[1,1,1,1,1,1,1],[1,2,0,0,0,2,1],[1,0,0,0,0,0,1],[1,0,0,4,0,5,1],[1,0,0,0,0,0,1],[1,2,0,6,0,3,1],[1,1,1,1,1,1,1]],jnp.int32),
            "pos":jnp.array([[2,3],[3,4],[4,3]],jnp.int32),
            "health":jnp.full((3,),8,jnp.int32),
            "carry":jnp.array([[2,0,0],[1,0,1],[1,1,0]],jnp.int32),
            "priority":jax.random.permutation(k1,jnp.arange(3,dtype=jnp.int32)),
            "forecast":jax.random.randint(k2,(3,),1,3,dtype=jnp.int32),
            "weather":jnp.array(0,jnp.int32),
            "resource_pos":jnp.array([[1,1],[5,1],[1,5],[5,5]],jnp.int32),
            "resource_kind":jnp.array([0,0,0,1],jnp.int32),
            "stock":jnp.array([6,6,6,6],jnp.int32),
            "resource_active":jnp.ones((4,),jnp.bool_),
            "resource_visible":jnp.ones((3,4),jnp.bool_),
            "station_pos":jnp.array([[3,3],[5,3],[3,5]],jnp.int32),
            "station_active":jnp.ones((3,),jnp.bool_),
            "station_visible":jnp.ones((3,3),jnp.bool_),
            "station_status":jnp.zeros((3,3),jnp.int32),
            "agent_active":jnp.ones((3,),jnp.bool_),
            "agent_visible":jnp.ones((3,3),jnp.bool_),
            "wall":jnp.array(0,jnp.int32),"lining":jnp.array(0,jnp.int32),
            "heat":jnp.array(0,jnp.int32),"kiln_timer":jnp.array(0,jnp.int32),
            "kiln_output":jnp.array(0,jnp.int32),"smoke":jnp.array(0,jnp.int32),
            "stashed":jnp.zeros((3,3),jnp.int32),"burned_timber":jnp.array(0,jnp.int32),
            "processed_clay":jnp.array(0,jnp.int32),"lost_wall":jnp.array(0,jnp.int32),
            "events":jnp.zeros((8,),jnp.bool_)}
        return self.observe(s),s

    def legal_actions(self, s):
        near = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["station_pos"][None,:,:]),axis=2)
        resource = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["resource_pos"][None,:,:]),axis=2)<=1
        room = jnp.sum(s["carry"],axis=1)<3
        shelter,kiln,locker = near[:,0]<=2,near[:,1]<=1,near[:,2]<=1
        peer = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=2)==1
        peer = peer&room[None,:]&(s["health"][None,:]>0)
        mask = jnp.zeros((3,16),jnp.bool_).at[:,0:5].set(True)
        mask = mask.at[:,5].set(jnp.any(resource&(s["stock"][None,:]>0),axis=1)&room)
        mask = mask.at[:,6].set(shelter&(s["carry"][:,0]>0)&(s["wall"]<3))
        mask = mask.at[:,7].set(kiln&(s["carry"][:,0]>0)&(s["carry"][:,1]>0)&(s["kiln_timer"]==0)&(s["kiln_output"]==0))
        mask = mask.at[:,8].set(kiln&room&(s["kiln_output"]==1))
        mask = mask.at[:,9].set(shelter&(s["carry"][:,2]>0)&(s["lining"]<3))
        mask = mask.at[:,10].set(shelter&(s["carry"][:,0]>0)&(s["heat"]<=8))
        mask = mask.at[:,11].set(kiln&(s["smoke"]>0))
        mask = mask.at[:,12].set(jnp.any(peer,axis=1)&(jnp.sum(s["carry"],axis=1)>0))
        mask = mask.at[:,13].set(locker&(s["carry"][jnp.arange(3),s["priority"]]>0))
        mask = mask.at[:,14].set(shelter&(s["carry"][:,0]>0)&(s["wall"]<3))
        mask = mask.at[:,15].set(shelter&(jnp.mod(s["t"],18)<12)&(s["health"]<8))
        return mask&((s["health"]>0)&~s["done"])[:,None] | (jnp.arange(16)[None,:]==0)

    def observe(self, s):
        phase = jnp.mod(s["t"],18)
        current = s["weather"]
        public = jnp.concatenate((s["t"][None]/54.0,phase[None]/17.0,current[None]/2.0,
            s["tile"].reshape(-1)/6.0,s["pos"].reshape(-1)/6.0,s["health"]/8.0,
            s["carry"].reshape(-1)/3.0,s["stock"]/6.0,
            jnp.array([s["wall"]/3.0,s["lining"]/3.0,s["heat"]/12.0,s["kiln_timer"]/3.0,s["kiln_output"],s["smoke"]/8.0]),s["events"],jnp.sum(s["stashed"],axis=0)/24.0))
        return jnp.concatenate((jnp.broadcast_to(public,(3,91)),s["priority"][:,None]/2.0,s["forecast"][:,None]/2.0,jnp.eye(3)),axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        old_done = state["done"]
        s = dict(state)
        safe = jnp.clip(actions,0,15)
        a = jnp.where((safe==actions)&self.legal_actions(state)[jnp.arange(3),safe],actions,0)
        s["events"] = jnp.zeros((8,),jnp.bool_)
        reward = jnp.zeros((3,),jnp.float32)
        moving = (a>=1)&(a<=4)
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)[jnp.clip(a,0,4)]
        q = s["pos"]+jnp.where(moving[:,None],d,0)
        x,y = jnp.clip(q[:,0],0,6),jnp.clip(q[:,1],0,6)
        free = (s["tile"][y,x]==0)|(s["tile"][y,x]==6)
        occupied = jnp.any(jnp.all(q[:,None,:]==s["pos"][None,:,:],axis=2)&~jnp.eye(3,dtype=jnp.bool_),axis=1)
        rank = jnp.mod(jnp.arange(3)-s["t"],3)
        lose = jnp.any(jnp.all(q[:,None,:]==q[None,:,:],axis=2)&(rank[None,:]<rank[:,None])&moving[None,:]&free[None,:]&~occupied[None,:],axis=1)
        s["pos"] = jnp.where((moving&free&~occupied&~lose)[:,None],q,s["pos"])
        for turn in range(3):
            i = jnp.mod(s["t"]+turn,3)
            act = jnp.where(self.legal_actions(s)[i,a[i]],a[i],0)
            eligible = (jnp.sum(jnp.abs(s["resource_pos"]-s["pos"][i]),axis=1)<=1)&(s["stock"]>0)
            node = jnp.argmax(eligible)
            kind = s["resource_kind"][node]
            gather = act==5
            s["stock"] = s["stock"].at[node].add(-gather.astype(jnp.int32))
            s["carry"] = s["carry"].at[i,kind].add(gather.astype(jnp.int32))
            build = (act==6)|(act==14)
            load,take,insulate,stoke,vent = act==7,act==8,act==9,act==10,act==11
            s["carry"] = s["carry"].at[i,0].add(-build.astype(jnp.int32)-load.astype(jnp.int32)-stoke.astype(jnp.int32))
            s["carry"] = s["carry"].at[i,1].add(-load.astype(jnp.int32))
            s["carry"] = s["carry"].at[i,2].add(take.astype(jnp.int32)-insulate.astype(jnp.int32))
            s["wall"] = s["wall"]+build.astype(jnp.int32)
            s["lining"] = s["lining"]+insulate.astype(jnp.int32)
            s["heat"] = s["heat"]+4*stoke.astype(jnp.int32)
            s["kiln_timer"] = jnp.where(load,3,s["kiln_timer"])
            s["kiln_output"] = s["kiln_output"]-take.astype(jnp.int32)
            s["smoke"] = jnp.clip(s["smoke"]+2*load.astype(jnp.int32)+2*stoke.astype(jnp.int32)-3*vent.astype(jnp.int32),0,8)
            s["burned_timber"] = s["burned_timber"]+load.astype(jnp.int32)+stoke.astype(jnp.int32)
            s["processed_clay"] = s["processed_clay"]+load.astype(jnp.int32)
            peer = (jnp.sum(jnp.abs(s["pos"]-s["pos"][i]),axis=1)==1)&(jnp.sum(s["carry"],axis=1)<3)&(s["health"]>0)
            receiver = jnp.argmax(peer)
            resource = jnp.argmax(s["carry"][i]>0)
            give = act==12
            s["carry"] = s["carry"].at[i,resource].add(-give.astype(jnp.int32)).at[receiver,resource].add(give.astype(jnp.int32))
            stash = act==13
            wanted = s["priority"][i]
            s["carry"] = s["carry"].at[i,wanted].add(-stash.astype(jnp.int32))
            s["stashed"] = s["stashed"].at[i,wanted].add(stash.astype(jnp.int32))
            reward = reward.at[i].add(0.25*stash.astype(jnp.float32))
            s["health"] = s["health"].at[i].add((act==15).astype(jnp.int32))
            s["events"] = s["events"]|jnp.array([gather,build|insulate,load,take,stoke|vent,give,stash,False])
        # Kiln work, visible smoke, and weather advance after all transactions.
        finished = s["kiln_timer"]==1
        s["kiln_timer"] = jnp.maximum(s["kiln_timer"]-1,0)
        s["kiln_output"] = s["kiln_output"]+finished.astype(jnp.int32)
        s["smoke"] = jnp.maximum(s["smoke"]-(jnp.mod(s["t"]+1,3)==0).astype(jnp.int32),0)
        night = jnp.mod(s["t"],18)>=12
        cold = s["forecast"][jnp.minimum(s["t"]//18,2)]
        near_shelter = jnp.sum(jnp.abs(s["pos"]-s["station_pos"][0]),axis=1)<=2
        protected = near_shelter&(s["wall"]>=2)&(s["lining"]>=2)&(s["heat"]>=cold)
        smoke_zone = jnp.sum(jnp.abs(s["pos"]-s["station_pos"][1]),axis=1)<=1
        fire = s["smoke"]>=6
        damage = ((night&~protected).astype(jnp.int32)+(fire&smoke_zone).astype(jnp.int32))*(s["health"]>0).astype(jnp.int32)
        s["health"] = jnp.maximum(s["health"]-damage,0)
        s["heat"] = jnp.maximum(s["heat"]-night.astype(jnp.int32),0)
        wear = ((jnp.mod(s["t"],18)==17)|(fire&(jnp.mod(s["t"],4)==3)))&(s["wall"]>0)
        s["wall"] = s["wall"]-wear.astype(jnp.int32)
        s["lost_wall"] = s["lost_wall"]+wear.astype(jnp.int32)
        s["events"] = s["events"].at[7].set(jnp.any(damage>0)|wear)
        reward = reward-0.1*damage.astype(jnp.float32)
        s["t"] = s["t"]+jnp.array(1,jnp.int32)
        s["weather"] = s["forecast"][jnp.minimum(s["t"]//18,2)]*(jnp.mod(s["t"],18)>=12).astype(jnp.int32)
        sunrise = s["t"]==54
        reward = reward+sunrise.astype(jnp.float32)*(s["health"]>0).astype(jnp.float32)
        s["terminated"] = sunrise|jnp.all(s["health"]==0)
        s["truncated"] = (s["t"]>=54)&~s["terminated"]
        s["done"] = s["terminated"]|s["truncated"]
        s["resource_active"] = s["stock"]>0
        s["station_status"] = jnp.stack((jnp.array([s["wall"],s["lining"],s["heat"]]),jnp.array([s["kiln_timer"],s["kiln_output"],s["smoke"]]),jnp.sum(s["stashed"],axis=0)))
        s = jax.tree.map(lambda new,old:jnp.where(old_done,old,new),s,state)
        reward = jnp.where(old_done,jnp.zeros((3,),jnp.float32),reward).astype(jnp.float32)
        return self.observe(s),s,reward,jnp.full((3,),s["done"],jnp.bool_),{"done_all":s["done"],"terminated":s["terminated"],"truncated":s["truncated"]}
