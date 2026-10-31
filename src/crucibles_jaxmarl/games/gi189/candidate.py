"""Night Market Crew: original unexecuted shared boiler, washing and order game."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 141
    num_actions = 17
    max_steps = 108

    def reset(self, key):
        s = {
            "t":jnp.array(0,jnp.int32),"done":jnp.array(False),
            "terminated":jnp.array(False),"truncated":jnp.array(False),
            "tile":jnp.array([[1,1,1,1,1,1,1,1,1],[1,2,0,0,9,0,0,3,1],[1,0,5,0,8,0,6,0,1],[1,0,0,0,12,0,0,0,1],[1,0,0,0,0,0,0,0,1],[1,10,0,0,7,0,0,11,1],[1,0,0,0,4,0,0,0,1],[1,1,1,1,1,1,1,1,1]],jnp.int32),
            "pos":jnp.array([[2,3],[6,3],[3,5]],jnp.int32),
            "carry":jnp.array([1,2,3],jnp.int32),"meal_age":jnp.zeros((3,),jnp.int32),
            "tool":jnp.ones((3,),jnp.int32),
            "priority":jax.random.permutation(key,jnp.arange(3,dtype=jnp.int32)),
            "stock":jnp.full((3,),4,jnp.int32),"fuel":jnp.array(6,jnp.int32),
            "water":jnp.array(4,jnp.int32),"reserve":jnp.array(12,jnp.int32),"heat":jnp.array(4,jnp.int32),
            "cook_timer":jnp.zeros((3,),jnp.int32),"ready":jnp.zeros((3,),jnp.int32),"dish_age":jnp.zeros((3,),jnp.int32),
            "wash_timer":jnp.array(0,jnp.int32),"clean":jnp.array(0,jnp.int32),
            "order":jnp.zeros((6,),jnp.int32),"order_kind":jnp.array([0,1,2,1,0,2],jnp.int32),
            "deadline":jnp.array([30,45,60,75,90,105],jnp.int32),
            "served_kind":jnp.zeros((3,),jnp.int32),"waste":jnp.zeros((3,),jnp.int32),
            "water_used":jnp.array(0,jnp.int32),"coal_burned":jnp.array(0,jnp.int32),
            "denied":jnp.zeros((3,),jnp.bool_),
            "station_pos":jnp.array([[1,1],[7,1],[4,6],[2,2],[6,2],[4,5],[4,2],[4,1],[1,5],[7,5],[4,3]],jnp.int32),
            "station_status":jnp.array([[4,0,0],[4,0,0],[4,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[4,0,0],[4,12,0],[6,0,0],[0,6,0]],jnp.int32),
            "station_active":jnp.ones((11,),jnp.bool_),"station_visible":jnp.ones((3,11),jnp.bool_),
            "agent_active":jnp.ones((3,),jnp.bool_),"agent_visible":jnp.ones((3,3),jnp.bool_),
            "events":jnp.zeros((10,),jnp.bool_)}
        return self.observe(s),s

    def legal_actions(self, s):
        near = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["station_pos"][None,:,:]),axis=2)<=1
        c = s["carry"]
        kind = jnp.clip(c-1,0,2)
        heat_cost = jnp.array([3,2,1],jnp.int32)[kind]
        water_cost = jnp.array([0,2,1],jnp.int32)[kind]
        station = near[jnp.arange(3),kind+3]
        recipe = (c>=1)&(c<=3)&(s["tool"]==1)&station&(s["cook_timer"][kind]==0)&(s["ready"][kind]==0)&(s["heat"]>=heat_cost)&(s["water"]>=water_cost)
        dish = jnp.mod(jnp.maximum(c-4,0),3)
        matches = jnp.any((s["order"][None,:]==0)&(s["order_kind"][None,:]==dish[:,None]),axis=1)
        peers = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=2)==1
        m = jnp.zeros((3,17),jnp.bool_).at[:,0:5].set(True)
        m = m.at[:,5].set((c==0)&jnp.any(near[:,:3]&(s["stock"][None,:]>0),axis=1))
        m = m.at[:,6].set(recipe)
        m = m.at[:,7].set((c==0)&jnp.any(near[:,3:6]&(s["ready"][None,:]>0),axis=1))
        m = m.at[:,8].set(near[:,10]&(c>=4)&(c<=9)&matches)
        m = m.at[:,9].set(near[:,9]&(c==0)&(s["fuel"]>0))
        m = m.at[:,10].set(near[:,7]&(c==10)&(s["heat"]<=4))
        m = m.at[:,11].set(near[:,8]&(s["reserve"]>0)&(s["water"]<=4))
        m = m.at[:,12].set(near[:,6]&(s["tool"]==2)&(s["water"]>=1)&(s["wash_timer"]==0))
        m = m.at[:,13].set(near[:,6]&(s["tool"]==0)&(s["clean"]>0))
        m = m.at[:,14].set((c>0)&jnp.any(peers&(c[None,:]==0),axis=1))
        m = m.at[:,15].set((s["tool"]>0)&jnp.any(peers&(s["tool"][None,:]==0),axis=1))
        m = m.at[:,16].set((c>=1)&(c<=9))
        return m&~s["done"] | (jnp.arange(17)[None,:]==0)

    def observe(self, s):
        public = jnp.concatenate((s["t"][None]/108.0,s["tile"].reshape(-1)/12.0,
            s["pos"].reshape(-1)/8.0,s["carry"]/10.0,s["meal_age"]/10.0,s["tool"]/2.0,s["stock"]/4.0,
            jnp.array([s["fuel"]/6.0,s["water"]/6.0,s["reserve"]/12.0,s["heat"]/8.0]),
            s["cook_timer"]/4.0,s["ready"]/9.0,s["dish_age"]/10.0,jnp.array([s["wash_timer"]/2.0,s["clean"]/3.0]),
            s["order"]/2.0,s["deadline"]/108.0,s["order_kind"]/2.0,s["denied"],s["events"]))
        return jnp.concatenate((jnp.broadcast_to(public,(3,137)),s["priority"][:,None]/2.0,jnp.eye(3)),axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        old_done = state["done"]
        s = dict(state)
        safe = jnp.clip(actions,0,16)
        legal = (safe==actions)&self.legal_actions(state)[jnp.arange(3),safe]
        a = jnp.where(legal,actions,0)
        s["events"] = jnp.zeros((10,),jnp.bool_)
        s["denied"] = ~legal
        reward = jnp.zeros((3,),jnp.float32)
        moving = (a>=1)&(a<=4)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)[jnp.clip(a,0,4)]
        q = s["pos"]+jnp.where(moving[:,None],delta,0)
        x,y = jnp.clip(q[:,0],0,8),jnp.clip(q[:,1],0,7)
        free = s["tile"][y,x]==0
        occupied = jnp.any(jnp.all(q[:,None,:]==s["pos"][None,:,:],axis=2)&~jnp.eye(3,dtype=jnp.bool_),axis=1)
        rank = jnp.mod(jnp.arange(3)-s["t"],3)
        lose = jnp.any(jnp.all(q[:,None,:]==q[None,:,:],axis=2)&(rank[None,:]<rank[:,None])&moving[None,:]&free[None,:]&~occupied[None,:],axis=1)
        moved = moving&free&~occupied&~lose
        s["pos"] = jnp.where(moved[:,None],q,s["pos"])
        s["denied"] = s["denied"]|(moving&~moved)
        for turn in range(3):
            i = jnp.mod(s["t"]+turn,3)
            now_legal = self.legal_actions(s)[i,a[i]]
            s["denied"] = s["denied"].at[i].set(s["denied"][i]|(~now_legal&(a[i]>=5)))
            act = jnp.where(now_legal,a[i],0)
            near = jnp.sum(jnp.abs(s["station_pos"]-s["pos"][i]),axis=1)<=1
            c,age = s["carry"][i],s["meal_age"][i]
            pick = act==5
            pantry = jnp.argmax(near[:3]&(s["stock"]>0))
            s["stock"] = s["stock"].at[pantry].add(-pick.astype(jnp.int32))
            c = jnp.where(pick,pantry+1,c)
            cook = act==6
            kind = jnp.clip(c-1,0,2)
            heat_cost = jnp.array([3,2,1],jnp.int32)[kind]*cook.astype(jnp.int32)
            water_cost = jnp.array([0,2,1],jnp.int32)[kind]*cook.astype(jnp.int32)
            s["heat"] = s["heat"]-heat_cost
            s["water"] = s["water"]-water_cost
            s["water_used"] = s["water_used"]+water_cost
            duration = jnp.array([3,4,2],jnp.int32)[kind]
            s["cook_timer"] = s["cook_timer"].at[kind].set(jnp.where(cook,duration,s["cook_timer"][kind]))
            s["tool"] = s["tool"].at[i].set(jnp.where(cook,2,s["tool"][i]))
            c = jnp.where(cook,0,c)
            collect = act==7
            stall = jnp.argmax(near[3:6]&(s["ready"]>0))
            c = jnp.where(collect,s["ready"][stall],c)
            age = jnp.where(collect,s["dish_age"][stall],age)
            s["ready"] = s["ready"].at[stall].set(jnp.where(collect,0,s["ready"][stall]))
            s["dish_age"] = s["dish_age"].at[stall].set(jnp.where(collect,0,s["dish_age"][stall]))
            deliver = act==8
            dish = jnp.mod(jnp.maximum(c-4,0),3)
            queue = (s["order"]==0)&(s["order_kind"]==dish)
            order = jnp.argmax(queue)
            s["order"] = s["order"].at[order].set(jnp.where(deliver,1,s["order"][order]))
            s["served_kind"] = s["served_kind"].at[dish].add(deliver.astype(jnp.int32))
            hot = c<=6
            value = jnp.where(hot,0.5,0.15)*deliver.astype(jnp.float32)
            reward = reward+value
            reward = reward.at[i].add(jnp.where(hot,0.4,0.1)*(dish==s["priority"][i]).astype(jnp.float32)*deliver.astype(jnp.float32))
            c = jnp.where(deliver,0,c)
            coal = act==9
            s["fuel"] = s["fuel"]-coal.astype(jnp.int32)
            c = jnp.where(coal,10,c)
            stoke = act==10
            s["heat"] = s["heat"]+4*stoke.astype(jnp.int32)
            s["coal_burned"] = s["coal_burned"]+stoke.astype(jnp.int32)
            c = jnp.where(stoke,0,c)
            pump = act==11
            amount = jnp.minimum(s["reserve"],2)*pump.astype(jnp.int32)
            s["reserve"] = s["reserve"]-amount
            s["water"] = s["water"]+amount
            wash = act==12
            s["wash_timer"] = jnp.where(wash,2,s["wash_timer"])
            s["water"] = s["water"]-wash.astype(jnp.int32)
            s["water_used"] = s["water_used"]+wash.astype(jnp.int32)
            s["tool"] = s["tool"].at[i].set(jnp.where(wash,0,s["tool"][i]))
            take_tool = act==13
            s["clean"] = s["clean"]-take_tool.astype(jnp.int32)
            s["tool"] = s["tool"].at[i].set(jnp.where(take_tool,1,s["tool"][i]))
            peer = jnp.sum(jnp.abs(s["pos"]-s["pos"][i]),axis=1)==1
            give = act==14
            recipient = jnp.argmax(peer&(s["carry"]==0))
            s["carry"] = s["carry"].at[recipient].set(jnp.where(give,c,s["carry"][recipient]))
            s["meal_age"] = s["meal_age"].at[recipient].set(jnp.where(give,age,s["meal_age"][recipient]))
            c = jnp.where(give,0,c)
            lend = act==15
            recipient = jnp.argmax(peer&(s["tool"]==0))
            held_tool = s["tool"][i]
            s["tool"] = s["tool"].at[recipient].set(jnp.where(lend,held_tool,s["tool"][recipient]))
            s["tool"] = s["tool"].at[i].set(jnp.where(lend,0,s["tool"][i]))
            discard = act==16
            waste_kind = jnp.mod(jnp.maximum(c-1,0),3)
            s["waste"] = s["waste"].at[waste_kind].add(discard.astype(jnp.int32))
            c = jnp.where(discard,0,c)
            s["carry"] = s["carry"].at[i].set(c)
            s["meal_age"] = s["meal_age"].at[i].set(jnp.where((c>=4)&(c<=9),age,0))
            s["events"] = s["events"]|jnp.array([pick,cook,collect,deliver,coal|stoke,pump,wash|take_tool,give|lend,discard,False])
        finished = s["cook_timer"]==1
        s["cook_timer"] = jnp.maximum(s["cook_timer"]-1,0)
        s["ready"] = jnp.where(finished,jnp.arange(3,dtype=jnp.int32)+4,s["ready"])
        limit = jnp.array([7,6,10],jnp.int32)
        age = jnp.where(s["ready"]>0,jnp.where(finished,0,jnp.minimum(s["dish_age"]+1,limit)),0)
        cooled = (s["ready"]>=4)&(s["ready"]<=6)&(age>=limit)
        s["ready"] = s["ready"]+3*cooled.astype(jnp.int32)
        s["dish_age"] = age
        dish = jnp.mod(jnp.maximum(s["carry"]-4,0),3)
        food = (s["carry"]>=4)&(s["carry"]<=9)
        s["meal_age"] = jnp.where(food,jnp.minimum(s["meal_age"]+1,limit[dish]),0)
        carry_cold = food&(s["carry"]<=6)&(s["meal_age"]>=limit[dish])
        s["carry"] = s["carry"]+3*carry_cold.astype(jnp.int32)
        washed = s["wash_timer"]==1
        s["wash_timer"] = jnp.maximum(s["wash_timer"]-1,0)
        s["clean"] = s["clean"]+washed.astype(jnp.int32)
        s["heat"] = jnp.maximum(s["heat"]-(jnp.mod(s["t"]+1,4)==0).astype(jnp.int32),0)
        s["t"] = s["t"]+jnp.array(1,jnp.int32)
        missed = (s["order"]==0)&(s["t"]>=s["deadline"])
        s["order"] = jnp.where(missed,2,s["order"])
        reward = reward-0.2*jnp.sum(missed).astype(jnp.float32)
        s["events"] = s["events"].at[9].set(jnp.any(missed)|jnp.any(cooled)|jnp.any(carry_cold))
        s["terminated"] = jnp.all(s["order"]!=0)
        s["truncated"] = (s["t"]>=108)&~s["terminated"]
        s["done"] = s["terminated"]|s["truncated"]
        s["station_status"] = jnp.concatenate((jnp.stack((s["stock"],jnp.zeros((3,),jnp.int32),jnp.zeros((3,),jnp.int32)),axis=1),
            jnp.stack((s["cook_timer"],s["ready"],s["dish_age"]),axis=1),jnp.array([[s["wash_timer"],s["clean"],0],[s["heat"],0,0],[s["water"],s["reserve"],0],[s["fuel"],0,0],[jnp.sum(s["order"]==1),jnp.sum(s["order"]==0),jnp.sum(s["order"]==2)]],jnp.int32)),axis=0)
        s = jax.tree.map(lambda new,old:jnp.where(old_done,old,new),s,state)
        reward = jnp.where(old_done,jnp.zeros((3,),jnp.float32),reward).astype(jnp.float32)
        return self.observe(s),s,reward,jnp.full((3,),s["done"],jnp.bool_),{"done_all":s["done"],"terminated":s["terminated"],"truncated":s["truncated"]}
