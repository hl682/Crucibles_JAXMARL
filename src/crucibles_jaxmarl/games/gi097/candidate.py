"""Original Lockwake Convoys canal. Authored, never imported or executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 155
    num_actions = 9
    max_steps = 80

    def refresh(self, s):
        s = dict(s)
        held = (s["cargo_holder"][None,:] == jnp.arange(3)[:,None]) & s["cargo_active"][None,:]
        s["carry"] = jnp.where(jnp.any(held,axis=1),jnp.argmax(held,axis=1),-1).astype(jnp.int32)
        s["cargo_pos"] = jnp.where((s["cargo_holder"] >= 0)[:,None],
                                   s["agent_pos"][jnp.maximum(s["cargo_holder"],0)],s["cargo_pos"])
        s["cargo_visible"] = jnp.broadcast_to(s["cargo_active"],(3,3))
        s["goal_active"] = s["cargo_active"]
        s["goal_visible"] = jnp.broadcast_to(s["cargo_active"],(3,3))
        return s

    def reset(self, key):
        terrain = jnp.array([[1,1,1,1,1,1,1,1,1,1,1],
                             [1,0,2,0,1,1,1,0,2,0,1],
                             [1,0,0,0,1,1,1,0,0,0,1],
                             [1,0,0,0,0,0,0,0,0,0,1],
                             [1,0,0,0,1,1,1,0,0,0,1],
                             [1,0,2,0,1,1,1,0,2,0,1],
                             [1,1,1,1,1,1,1,1,1,1,1]],dtype=jnp.int32)
        s = {"t":jnp.array(0,dtype=jnp.int32),"episode_done":jnp.array(False),
             "terminated":jnp.array(False),"truncated":jnp.array(False),"map":terrain,
             "agent_pos":jnp.array([[2,1],[8,5],[2,5]],dtype=jnp.int32),
             "agent_active":jnp.ones(3,dtype=jnp.bool_),"agent_visible":jnp.ones((3,3),dtype=jnp.bool_),
             "heading":jnp.zeros(3,dtype=jnp.int32),"speed":jnp.zeros(3,dtype=jnp.int32),
             "fuel":jnp.full(3,12,dtype=jnp.int32),"tow_partner":jnp.full(3,-1,dtype=jnp.int32),
             "carry":jnp.full(3,-1,dtype=jnp.int32),"blocked":jnp.zeros(3,dtype=jnp.bool_),
             "cargo_pos":jnp.array([[2,1],[8,5],[2,5]],dtype=jnp.int32),
             "cargo_active":jnp.ones(3,dtype=jnp.bool_),"cargo_visible":jnp.ones((3,3),dtype=jnp.bool_),
             "cargo_holder":jnp.full(3,-1,dtype=jnp.int32),"cargo_owner":jnp.arange(3,dtype=jnp.int32),
             "cargo_damage":jnp.zeros(3,dtype=jnp.int32),"cargo_delivered":jnp.zeros(3,dtype=jnp.bool_),
             "cargo_lost":jnp.zeros(3,dtype=jnp.bool_),
             "goal_pos":jnp.array([[8,5],[2,1],[8,1]],dtype=jnp.int32),
             "goal_active":jnp.ones(3,dtype=jnp.bool_),"goal_visible":jnp.ones((3,3),dtype=jnp.bool_),
             "dock_pos":jnp.array([[2,1],[8,5],[2,5],[8,1]],dtype=jnp.int32),
             "dock_active":jnp.ones(4,dtype=jnp.bool_),"dock_visible":jnp.ones((3,4),dtype=jnp.bool_),
             "delivered_now":jnp.zeros(3,dtype=jnp.bool_),"lost_now":jnp.zeros(3,dtype=jnp.bool_),
             "tethered_now":jnp.zeros(3,dtype=jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        flat = jnp.concatenate([s["agent_pos"].reshape(-1),s["heading"],s["speed"],s["fuel"],
                                s["tow_partner"],s["carry"],s["blocked"],s["cargo_pos"].reshape(-1),
                                s["cargo_active"],s["cargo_holder"],s["cargo_owner"],s["cargo_damage"],
                                s["cargo_delivered"],s["cargo_lost"],s["goal_pos"].reshape(-1),
                                s["dock_pos"].reshape(-1),s["map"].reshape(-1),
                                s["delivered_now"],s["lost_now"],s["tethered_now"],
                                jnp.stack([s["t"],s["episode_done"],s["terminated"],s["truncated"]])]).astype(jnp.float32)
        return jnp.concatenate([jnp.broadcast_to(flat,(3,152)),jnp.eye(3,dtype=jnp.float32)],axis=1)

    def advance(self, key, state, actions):
        a = jnp.where((actions >= 0) & (actions < 9),actions,0).astype(jnp.int32)
        s = dict(state)
        p,holder,active = s["agent_pos"],s["cargo_holder"],s["cargo_active"]
        cargo_pos = s["cargo_pos"]
        delivered = jnp.zeros(3,dtype=jnp.bool_)
        # Cargo changes hands only while stopped; loading and delivery occupy a whole action.
        for offset in range(3):
            i = jnp.mod(s["t"]+offset,3)
            held = active & (holder == i)
            j = jnp.argmax(held)
            carrying = jnp.any(held)
            interact = (a[i] == 6) & (s["speed"][i] == 0)
            serve = interact & carrying & jnp.all(p[i] == s["goal_pos"][j])
            free_here = jnp.any(active & (holder < 0) & jnp.all(cargo_pos == p[i],axis=1))
            drop = interact & carrying & ~serve & ~free_here
            active = active.at[j].set(active[j] & ~serve)
            delivered = delivered.at[j].set(delivered[j] | serve)
            holder = holder.at[j].set(jnp.where(serve | drop,-1,holder[j]))
            cargo_pos = cargo_pos.at[j].set(jnp.where(serve | drop,p[i],cargo_pos[j]))
            available = active & (holder < 0) & jnp.all(cargo_pos == p[i],axis=1)
            k = jnp.argmax(available)
            pickup = interact & ~carrying & jnp.any(available)
            holder = holder.at[k].set(jnp.where(pickup,i,holder[k]))
        partner = s["tow_partner"]
        other = jnp.maximum(partner,0)
        release = (a == 8) | ((partner >= 0) & (a[other] == 8))
        partner = jnp.where(release,-1,partner)
        tethered = jnp.zeros(3,dtype=jnp.bool_)
        for offset in range(3):
            i = jnp.mod(s["t"]+offset,3)
            eligible = ((a == 7) & (partner < 0) & (s["speed"] == 0) &
                        (jnp.sum(jnp.abs(p-p[i]),axis=1) == 1))
            j = jnp.argmax(eligible)
            attach = (a[i] == 7) & (partner[i] < 0) & (s["speed"][i] == 0) & jnp.any(eligible)
            partner = partner.at[i].set(jnp.where(attach,j,partner[i])).at[j].set(jnp.where(attach,i,partner[j]))
            tethered = tethered.at[i].set(tethered[i] | attach).at[j].set(tethered[j] | attach)
        index = jnp.arange(3,dtype=jnp.int32)
        linked = partner >= 0
        leader = jnp.where(linked,jnp.minimum(index,partner),index)
        is_leader = index == leader
        throttle = (a >= 1) & (a <= 4) & (s["fuel"] > 0) & is_leader
        stopped = s["speed"] == 0
        same_heading = s["heading"] == a
        heading = jnp.where(throttle & stopped,a,s["heading"])
        speed = jnp.where(throttle,jnp.where(stopped,1,jnp.where(same_heading,
                          jnp.minimum(s["speed"]+1,2),jnp.maximum(s["speed"]-1,0))),s["speed"])
        docked = jnp.any(jnp.all(p[:,None,:] == s["dock_pos"][None,:,:],axis=2),axis=1)
        fuel = jnp.minimum(s["fuel"]-throttle.astype(jnp.int32)+((a == 0) & stopped & docked).astype(jnp.int32),12)
        cargo_held = jnp.any((holder[None,:] == index[:,None]) & active[None,:],axis=1)
        # An empty tug may tow a laden follower at speed two; a laden leader is capped at one.
        speed = jnp.minimum(speed,jnp.where(cargo_held,1,2))
        heading,speed = heading[leader],speed[leader]
        pair_brake = (a == 5) | (linked & (a[jnp.maximum(partner,0)] == 5))
        speed = jnp.where(pair_brake,0,speed)
        tow_fuel_ok = fuel[leader] > 0
        speed = jnp.where(linked & ~tow_fuel_ok,0,speed)
        fuel = fuel - (linked & is_leader & (speed > 0)).astype(jnp.int32)
        offsets = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],dtype=jnp.int32)
        crash = jnp.zeros(3,dtype=jnp.bool_)
        # Resolve each cell crossing; following a vacating boat is allowed, swaps are not.
        for micro in range(2):
            wants = speed > micro
            q = p + offsets[heading]*wants[:,None].astype(jnp.int32)
            safe = jnp.clip(q,jnp.array([0,0],dtype=jnp.int32),jnp.array([10,6],dtype=jnp.int32))
            land = s["map"][safe[:,1],safe[:,0]] == 1
            same_target = jnp.sum(jnp.all(q[:,None,:] == q[None,:,:],axis=2),axis=1) > 1
            to_old = jnp.all(q[:,None,:] == p[None,:,:],axis=2)
            swap = jnp.any(to_old & jnp.transpose(to_old) & ~jnp.eye(3,dtype=jnp.bool_) &
                           wants[:,None] & wants[None,:],axis=1)
            blocked = wants & (land | same_target | swap)
            for propagation in range(3):
                blocked = blocked | (linked & blocked[jnp.maximum(partner,0)] & wants)
                blocked = blocked | (wants & jnp.any(to_old & (blocked | ~wants)[None,:] &
                           ~jnp.eye(3,dtype=jnp.bool_),axis=1))
            p = jnp.where((wants & ~blocked)[:,None],q,p)
            crash = crash | blocked
            speed = jnp.where(blocked,0,speed)
        cargo_pos = jnp.where((holder >= 0)[:,None],p[jnp.maximum(holder,0)],cargo_pos)
        damage = jnp.minimum(s["cargo_damage"] + (active & (holder >= 0) &
                            crash[jnp.maximum(holder,0)]).astype(jnp.int32),3)
        lost = active & (damage >= 3)
        active = active & ~lost
        holder = jnp.where(lost,-1,holder)
        s["agent_pos"],s["heading"],s["speed"],s["fuel"],s["tow_partner"] = p,heading,speed,fuel,partner
        s["blocked"] = crash
        s["cargo_pos"],s["cargo_holder"],s["cargo_active"],s["cargo_damage"] = cargo_pos,holder,active,damage
        s["cargo_delivered"],s["cargo_lost"] = s["cargo_delivered"] | delivered,s["cargo_lost"] | lost
        s["delivered_now"],s["lost_now"],s["tethered_now"] = delivered,lost,tethered
        s["t"] = s["t"]+1
        s["terminated"] = ~jnp.any(active)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        outcomes = delivered.astype(jnp.float32)-lost.astype(jnp.float32)
        private = jnp.sum(jnp.where(s["cargo_owner"][None,:] == index[:,None],outcomes[None,:],0.0),axis=1)
        rewards = private + jnp.sum(outcomes)*0.25 - crash.astype(jnp.float32)*0.125
        dones = jnp.full((3,),s["episode_done"],dtype=jnp.bool_)
        info = {"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}
        return self.observe(s), s, rewards, dones, info

    def legal_actions(self, state):
        live = jnp.broadcast_to(~state["episode_done"], (self.num_agents, self.num_actions))
        return live.at[:, 0].set(True)

    def absorb(self, key, state, actions):
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        dones = jnp.ones((self.num_agents,), dtype=jnp.bool_)
        info = {"done_all": state["episode_done"], "terminated": state["terminated"],
                "truncated": state["truncated"]}
        return self.observe(state), state, rewards, dones, info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["episode_done"], self.absorb, self.advance, key, state, actions)
