"""Canal Kitchen: original unexecuted bowl circulation and timed-crossing game."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 118
    num_actions = 18
    max_steps = 132

    def reset(self, key):
        s = {
            "t":jnp.array(0,jnp.int32),"done":jnp.array(False),
            "terminated":jnp.array(False),"truncated":jnp.array(False),
            "tile":jnp.array([[1,1,1,1,1,1,1,1,1],[1,2,0,0,1,0,0,3,1],[1,0,4,0,1,0,5,0,1],[1,0,0,0,9,0,0,0,1],[1,0,0,0,1,0,7,0,1],[1,6,0,0,1,0,0,8,1],[1,1,1,1,1,1,1,1,1]],jnp.int32),
            "pos":jnp.array([[2,3],[5,4],[3,4]],jnp.int32),
            "carry":jnp.array([0,0,5],jnp.int32),"food_age":jnp.zeros((3,),jnp.int32),
            "stock":jnp.array([4,4],jnp.int32),
            "work":jnp.array([[0,0,0],[0,0,0],[0,1,0]],jnp.int32),
            "assembly":jnp.zeros((3,),jnp.int32),
            "order":jnp.zeros((4,),jnp.int32),"deadline":jnp.array([40,70,100,130],jnp.int32),
            "served":jnp.array(0,jnp.int32),"waste":jnp.zeros((2,),jnp.int32),
            "ferry_pos":jnp.array([[3.0,4.0]],jnp.float32),
            "ferry_cargo":jnp.array([0],jnp.int32),"ferry_age":jnp.array([0],jnp.int32),
            "ferry_timer":jnp.array([0],jnp.int32),"ferry_bank":jnp.array([0],jnp.int32),
            "ferry_active":jnp.array([True]),"ferry_visible":jnp.ones((3,1),jnp.bool_),
            "bridge_pos":jnp.array([[4,3]],jnp.int32),"bridge_open":jnp.array([True]),
            "bridge_active":jnp.array([True]),"bridge_visible":jnp.ones((3,1),jnp.bool_),
            "station_pos":jnp.array([[1,1],[7,1],[2,2],[6,2],[1,5],[6,4],[7,5]],jnp.int32),
            "station_status":jnp.array([[4,0,0],[4,0,0],[0,0,0],[0,0,0],[0,1,0],[0,0,0],[0,4,0]],jnp.int32),
            "station_active":jnp.ones((7,),jnp.bool_),"station_visible":jnp.ones((3,7),jnp.bool_),
            "agent_active":jnp.ones((3,),jnp.bool_),"agent_visible":jnp.ones((3,3),jnp.bool_),
            "events":jnp.zeros((9,),jnp.bool_)}
        return self.observe(s),s

    def legal_actions(self, s):
        near = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["station_pos"][None,:,:]),axis=2)<=1
        c = s["carry"]
        ferry = (jnp.sum(jnp.abs(s["pos"]-s["ferry_pos"][0]),axis=1)<=1)&(s["ferry_timer"][0]==0)
        load_board = near[:,2]&(c==1)&(s["work"][0,0]==0)&(s["work"][0,1]==0)
        load_pot = near[:,3]&(c==3)&(s["work"][1,0]==0)&(s["work"][1,1]==0)
        prep = near[:,5]&(((c==2)&(s["assembly"][0]==0))|((c==4)&(s["assembly"][1]==0)))
        peer = (jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=2)==1)&(c[None,:]==0)
        m = jnp.zeros((3,18),jnp.bool_).at[:,0:5].set(True)
        m = m.at[:,5].set((c==0)&jnp.any(near[:,:2]&(s["stock"][None,:]>0),axis=1))
        m = m.at[:,6].set(load_board|load_pot)
        m = m.at[:,7].set((c==0)&((near[:,2]&(s["work"][0,1]>0))|(near[:,3]&(s["work"][1,1]>0))))
        m = m.at[:,8].set(prep)
        m = m.at[:,9].set(near[:,5]&(c==5)&(s["assembly"][0]==1)&(s["assembly"][1]==1))
        m = m.at[:,10].set(near[:,6]&(c==6)&jnp.any(s["order"]==0))
        m = m.at[:,11].set(ferry&(c>0)&(s["ferry_cargo"][0]==0))
        m = m.at[:,12].set(ferry&(s["ferry_cargo"][0]>0))
        m = m.at[:,13].set(ferry&(c==0)&(s["ferry_cargo"][0]>0))
        m = m.at[:,14].set(near[:,4]&(c==0)&(s["work"][2,1]>0))
        m = m.at[:,15].set(near[:,4]&(c==7)&(s["work"][2,0]==0))
        m = m.at[:,16].set((c>0)&jnp.any(peer,axis=1))
        m = m.at[:,17].set((c==1)|(c==2)|(c==3)|(c==4)|(c==6)|(c==8))
        return m&~s["done"] | (jnp.arange(18)[None,:]==0)

    def observe(self, s):
        public = jnp.concatenate((s["t"][None]/132.0,s["bridge_open"],s["tile"].reshape(-1)/9.0,
            s["pos"].reshape(-1)/8.0,s["carry"]/8.0,s["food_age"]/12.0,s["work"].reshape(-1)/12.0,
            s["assembly"]/12.0,s["stock"]/4.0,s["order"]/2.0,s["deadline"]/132.0,
            s["ferry_pos"].reshape(-1)/8.0,s["ferry_cargo"]/8.0,s["ferry_timer"]/4.0,s["ferry_bank"],s["ferry_age"]/12.0,
            s["served"][None]/4.0,s["events"]))
        return jnp.concatenate((jnp.broadcast_to(public,(3,115)),jnp.eye(3)),axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        old_done = state["done"]
        s = dict(state)
        safe = jnp.clip(actions,0,17)
        a = jnp.where((safe==actions)&self.legal_actions(state)[jnp.arange(3),safe],actions,0)
        s["events"] = jnp.zeros((9,),jnp.bool_)
        reward = jnp.zeros((3,),jnp.float32)
        moving = (a>=1)&(a<=4)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)[jnp.clip(a,0,4)]
        q = s["pos"]+jnp.where(moving[:,None],delta,0)
        x,y = jnp.clip(q[:,0],0,8),jnp.clip(q[:,1],0,6)
        tile = s["tile"][y,x]
        free = (tile==0)|((tile==9)&s["bridge_open"][0])
        occupied = jnp.any(jnp.all(q[:,None,:]==s["pos"][None,:,:],axis=2)&~jnp.eye(3,dtype=jnp.bool_),axis=1)
        rank = jnp.mod(jnp.arange(3)-s["t"],3)
        lose = jnp.any(jnp.all(q[:,None,:]==q[None,:,:],axis=2)&(rank[None,:]<rank[:,None])&moving[None,:]&free[None,:]&~occupied[None,:],axis=1)
        s["pos"] = jnp.where((moving&free&~occupied&~lose)[:,None],q,s["pos"])
        for turn in range(3):
            i = jnp.mod(s["t"]+turn,3)
            act = jnp.where(self.legal_actions(s)[i,a[i]],a[i],0)
            near = jnp.sum(jnp.abs(s["station_pos"]-s["pos"][i]),axis=1)<=1
            c,age = s["carry"][i],s["food_age"][i]
            pick = act==5
            source = jnp.argmax(near[:2]&(s["stock"]>0))
            s["stock"] = s["stock"].at[source].add(-pick.astype(jnp.int32))
            c = jnp.where(pick,jnp.where(source==0,1,3),c)
            load = act==6
            station = jnp.where(c==1,0,1)
            timer = jnp.where(station==0,2,3)
            s["work"] = s["work"].at[station,0].set(jnp.where(load,timer,s["work"][station,0]))
            c = jnp.where(load,0,c)
            collect = act==7
            station = jnp.where(near[2]&(s["work"][0,1]>0),0,1)
            c = jnp.where(collect,s["work"][station,1],c)
            age = jnp.where(collect,s["work"][station,2],age)
            s["work"] = s["work"].at[station,1].set(jnp.where(collect,0,s["work"][station,1]))
            s["work"] = s["work"].at[station,2].set(jnp.where(collect,0,s["work"][station,2]))
            add = act==8
            which = jnp.where(c==2,0,1)
            s["assembly"] = s["assembly"].at[which].add(add.astype(jnp.int32))
            s["assembly"] = s["assembly"].at[2].set(jnp.where(add&(c==4),age,s["assembly"][2]))
            c = jnp.where(add,0,c)
            plate = act==9
            s["assembly"] = jnp.where(plate,jnp.zeros((3,),jnp.int32),s["assembly"])
            c = jnp.where(plate,6,c)
            age = jnp.where(plate,0,age)
            deliver = act==10
            order = jnp.argmax(s["order"]==0)
            s["order"] = s["order"].at[order].set(jnp.where(deliver,1,s["order"][order]))
            s["served"] = s["served"]+deliver.astype(jnp.int32)
            reward = reward+deliver.astype(jnp.float32)
            c = jnp.where(deliver,7,c)
            ship = act==11
            s["ferry_cargo"] = jnp.where(ship,c,s["ferry_cargo"])
            s["ferry_age"] = jnp.where(ship,age,s["ferry_age"])
            c = jnp.where(ship,0,c)
            launch = act==12
            s["ferry_timer"] = jnp.where(launch,4,s["ferry_timer"])
            s["ferry_bank"] = jnp.where(launch,1-s["ferry_bank"],s["ferry_bank"])
            unload = act==13
            c = jnp.where(unload,s["ferry_cargo"][0],c)
            age = jnp.where(unload,s["ferry_age"][0],age)
            s["ferry_cargo"] = jnp.where(unload,0,s["ferry_cargo"])
            s["ferry_age"] = jnp.where(unload,0,s["ferry_age"])
            bowl = act==14
            c = jnp.where(bowl,5,c)
            s["work"] = s["work"].at[2,1].add(-bowl.astype(jnp.int32))
            wash = act==15
            s["work"] = s["work"].at[2,0].set(jnp.where(wash,2,s["work"][2,0]))
            c = jnp.where(wash,0,c)
            give = act==16
            peer = (jnp.sum(jnp.abs(s["pos"]-s["pos"][i]),axis=1)==1)&(s["carry"]==0)
            receiver = jnp.argmax(peer)
            s["carry"] = s["carry"].at[receiver].set(jnp.where(give,c,s["carry"][receiver]))
            s["food_age"] = s["food_age"].at[receiver].set(jnp.where(give,age,s["food_age"][receiver]))
            c = jnp.where(give,0,c)
            discard = act==17
            s["waste"] = s["waste"]+jnp.array([discard&((c==1)|(c==2)|(c==6)),discard&((c==3)|(c==4)|(c==6))],jnp.int32)
            c = jnp.where(discard,jnp.where(c==6,7,0),c)
            age = jnp.where((c==4)|(c==6),age,0)
            s["carry"] = s["carry"].at[i].set(c)
            s["food_age"] = s["food_age"].at[i].set(age)
            s["events"] = s["events"]|jnp.array([pick,load|collect,add|plate,deliver,ship|launch|unload,bowl|wash,give,discard,False])
        # Every vessel remains in a hand, on the ferry, in washing, or in the clean stack.
        finished = s["work"][:,0]==1
        s["work"] = s["work"].at[:,0].set(jnp.maximum(s["work"][:,0]-1,0))
        s["work"] = s["work"].at[0,1].set(jnp.where(finished[0],2,s["work"][0,1]))
        s["work"] = s["work"].at[1,1].set(jnp.where(finished[1],4,s["work"][1,1]))
        s["work"] = s["work"].at[2,1].add(finished[2].astype(jnp.int32))
        pot_age = jnp.where(s["work"][1,1]==4,jnp.where(finished[1],0,s["work"][1,2]+1),0)
        burnt = (s["work"][1,1]==4)&(pot_age>=8)
        s["work"] = s["work"].at[1,1].set(jnp.where(burnt,8,s["work"][1,1])).at[1,2].set(jnp.where(burnt,0,pot_age))
        s["ferry_timer"] = jnp.maximum(s["ferry_timer"]-1,0)
        target = 3.0+2.0*s["ferry_bank"].astype(jnp.float32)
        sign = 2.0*s["ferry_bank"].astype(jnp.float32)-1.0
        fx = target-sign*s["ferry_timer"].astype(jnp.float32)*0.5
        s["ferry_pos"] = s["ferry_pos"].at[:,0].set(fx)
        # Prepared broth and plated meals age in hands, tray, and ferry alike.
        hot = (s["carry"]==4)|(s["carry"]==6)
        s["food_age"] = jnp.where(hot,s["food_age"]+1,0)
        spoiled = hot&(s["food_age"]>=12)
        meal_spoil = spoiled&(s["carry"]==6)
        s["carry"] = jnp.where(spoiled,jnp.where(s["carry"]==6,7,8),s["carry"])
        s["food_age"] = jnp.where(spoiled,0,s["food_age"])
        s["ferry_age"] = jnp.where((s["ferry_cargo"]==4)|(s["ferry_cargo"]==6),s["ferry_age"]+1,0)
        ferry_spoil = s["ferry_age"][0]>=12
        ferry_meal = ferry_spoil&(s["ferry_cargo"][0]==6)
        s["ferry_cargo"] = jnp.where(ferry_spoil,jnp.where(s["ferry_cargo"]==6,7,8),s["ferry_cargo"])
        s["ferry_age"] = jnp.where(ferry_spoil,0,s["ferry_age"])
        age = jnp.where(s["assembly"][1]>0,s["assembly"][2]+1,0)
        tray_spoil = age>=12
        s["assembly"] = s["assembly"].at[1].set(jnp.where(tray_spoil,0,s["assembly"][1])).at[2].set(jnp.where(tray_spoil,0,age))
        s["waste"] = s["waste"]+jnp.array([jnp.sum(meal_spoil)+ferry_meal.astype(jnp.int32),jnp.sum(spoiled)+ferry_spoil.astype(jnp.int32)+tray_spoil.astype(jnp.int32)+burnt.astype(jnp.int32)],jnp.int32)
        s["events"] = s["events"].at[7].set(s["events"][7]|jnp.any(spoiled)|ferry_spoil|tray_spoil|burnt)
        s["t"] = s["t"]+jnp.array(1,jnp.int32)
        missed = (s["order"]==0)&(s["t"]>=s["deadline"])
        s["order"] = jnp.where(missed,2,s["order"])
        reward = reward-0.2*jnp.sum(missed).astype(jnp.float32)
        s["events"] = s["events"].at[8].set(jnp.any(missed))
        s["bridge_open"] = jnp.array([jnp.mod(s["t"],8)<4])
        s["terminated"] = jnp.all(s["order"]!=0)
        s["truncated"] = (s["t"]>=132)&~s["terminated"]
        s["done"] = s["terminated"]|s["truncated"]
        s["station_status"] = jnp.concatenate((jnp.stack((jnp.array([s["stock"][0],0,0]),jnp.array([s["stock"][1],0,0]))),s["work"],s["assembly"][None,:],jnp.array([[s["served"],jnp.sum(s["order"]==0),jnp.sum(s["order"]==2)]],jnp.int32)),axis=0)
        s = jax.tree.map(lambda new,old:jnp.where(old_done,old,new),s,state)
        reward = jnp.where(old_done,jnp.zeros((3,),jnp.float32),reward).astype(jnp.float32)
        return self.observe(s),s,reward,jnp.full((3,),s["done"],jnp.bool_),{"done_all":s["done"],"terminated":s["terminated"],"truncated":s["truncated"]}
