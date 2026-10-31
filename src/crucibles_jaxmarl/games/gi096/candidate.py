"""Original Flank-and-Hold meadow. Authored, never imported or executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 133
    num_actions = 8
    max_steps = 64

    def reset(self, key):
        terrain = jnp.array([[1,1,1,1,1,1,1,1,1],
                             [1,0,0,0,3,1,0,0,1],
                             [1,0,0,0,0,1,0,4,1],
                             [1,0,0,0,0,2,0,0,1],
                             [1,0,0,0,0,1,0,4,1],
                             [1,0,0,0,3,1,0,0,1],
                             [1,1,1,1,1,1,1,1,1]],dtype=jnp.int32)
        s = {"t":jnp.array(0,dtype=jnp.int32),"episode_done":jnp.array(False),
             "terminated":jnp.array(False),"truncated":jnp.array(False),"map":terrain,
             "agent_pos":jnp.array([[1,2],[1,4],[6,3]],dtype=jnp.int32),
             "agent_active":jnp.ones(3,dtype=jnp.bool_),"agent_visible":jnp.ones((3,3),dtype=jnp.bool_),
             "focus":jnp.full(3,4,dtype=jnp.int32),"signal":jnp.zeros(3,dtype=jnp.int32),
             "blocked":jnp.zeros(3,dtype=jnp.bool_),
             "animal_pos":jnp.array([[3,2],[3,3],[3,4],[2,3]],dtype=jnp.int32),
             "animal_active":jnp.ones(4,dtype=jnp.bool_),"animal_visible":jnp.ones((3,4),dtype=jnp.bool_),
             "panic":jnp.zeros(4,dtype=jnp.int32),"saved":jnp.zeros(4,dtype=jnp.bool_),
             "lost":jnp.zeros(4,dtype=jnp.bool_),
             "gate_pos":jnp.array([[5,3]],dtype=jnp.int32),"gate_active":jnp.ones(1,dtype=jnp.bool_),
             "gate_visible":jnp.ones((3,1),dtype=jnp.bool_),"gate_open":jnp.zeros(1,dtype=jnp.bool_),
             "gate_timer":jnp.zeros(1,dtype=jnp.int32),
             "pen_pos":jnp.array([[7,2],[7,4]],dtype=jnp.int32),"pen_active":jnp.ones(2,dtype=jnp.bool_),
             "pen_visible":jnp.ones((3,2),dtype=jnp.bool_),
             "hazard_pos":jnp.array([[4,1],[4,5]],dtype=jnp.int32),"hazard_active":jnp.ones(2,dtype=jnp.bool_),
             "hazard_visible":jnp.ones((3,2),dtype=jnp.bool_),
             "saved_total":jnp.array(0,dtype=jnp.int32),"lost_total":jnp.array(0,dtype=jnp.int32),
             "saved_now":jnp.zeros(4,dtype=jnp.bool_),"lost_now":jnp.zeros(4,dtype=jnp.bool_),
             "gate_opened":jnp.array(False)}
        return self.observe(s), s

    def observe(self, s):
        flat = jnp.concatenate([s["agent_pos"].reshape(-1),s["focus"],s["signal"],s["blocked"],
                                s["animal_pos"].reshape(-1),s["animal_active"],s["panic"],s["saved"],s["lost"],
                                s["gate_pos"].reshape(-1),s["gate_active"],s["gate_open"],s["gate_timer"],
                                s["pen_pos"].reshape(-1),s["hazard_pos"].reshape(-1),s["map"].reshape(-1),
                                jnp.stack([s["saved_total"],s["lost_total"],s["t"]]),
                                s["saved_now"],s["lost_now"],
                                jnp.stack([s["gate_opened"],s["episode_done"],s["terminated"],s["truncated"]])]).astype(jnp.float32)
        return jnp.concatenate([jnp.broadcast_to(flat,(3,130)),jnp.eye(3,dtype=jnp.float32)],axis=1)

    def move_agents(self, s, actions, gate_open):
        offsets = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],dtype=jnp.int32)
        step = offsets[jnp.where((actions >= 1) & (actions <= 4),actions,0)]
        p = s["agent_pos"]
        q = jnp.clip(p+step,jnp.array([0,0],dtype=jnp.int32),jnp.array([8,6],dtype=jnp.int32))
        tile = s["map"][q[:,1],q[:,0]]
        terrain = (tile != 1) & (tile != 3) & ((tile != 2) | gate_open)
        occupied = jnp.any(jnp.all(q[:,None,:] == p[None,:,:],axis=2) &
                           ~jnp.eye(3,dtype=jnp.bool_),axis=1)
        herd = jnp.any(jnp.all(q[:,None,:] == s["animal_pos"][None,:,:],axis=2) & s["animal_active"][None,:],axis=1)
        contest = jnp.sum(jnp.all(q[:,None,:] == q[None,:,:],axis=2),axis=1) > 1
        blocked = jnp.any(step != 0,axis=1) & (~terrain | occupied | herd | contest)
        return jnp.where(blocked[:,None],p,q),blocked

    def advance(self, key, state, actions):
        a = jnp.where((actions >= 0) & (actions < 8),actions,0).astype(jnp.int32)
        s = dict(state)
        at_gate = jnp.sum(jnp.abs(s["agent_pos"]-s["gate_pos"][0]),axis=1) <= 1
        hold = jnp.any((a == 7) & at_gate)
        timer = jnp.where(hold,3,jnp.maximum(s["gate_timer"][0]-1,0)).astype(jnp.int32)
        gate_occupied = jnp.any(jnp.all(s["agent_pos"] == s["gate_pos"][0],axis=1)) | jnp.any(
            jnp.all(s["animal_pos"] == s["gate_pos"][0],axis=1) & s["animal_active"])
        opened = (timer > 0) | gate_occupied
        p,blocked = self.move_agents(s,a,opened)
        whistle = (a == 5) & (s["focus"] > 0)
        calm = a == 6
        quiet = calm | ((a == 7) & at_gate)
        focus = jnp.clip(s["focus"]-whistle.astype(jnp.int32)+calm.astype(jnp.int32),0,4)
        signal = jnp.where(whistle,1,jnp.where(calm,2,jnp.where((a == 7) & at_gate,3,0))).astype(jnp.int32)
        pos,active,panic = s["animal_pos"],s["animal_active"],s["panic"]
        distances = jnp.sum(jnp.abs(pos[:,None,:]-p[None,:,:]),axis=2)
        weight = jnp.where((distances <= jnp.where(whistle[None,:],3,2)) & ~quiet[None,:],
                           1+whistle[None,:].astype(jnp.int32),0)
        pressure = jnp.sum(weight,axis=1,dtype=jnp.int32)
        soothing = jnp.sum((calm[None,:] & (distances <= 2)).astype(jnp.int32),axis=1,dtype=jnp.int32)
        panic_delta = (pressure >= 3).astype(jnp.int32) - soothing - (pressure == 0).astype(jnp.int32)
        panic = jnp.where(active,jnp.clip(panic+panic_delta,0,4),panic)
        saved_now = jnp.zeros(4,dtype=jnp.bool_)
        lost_now = jnp.zeros(4,dtype=jnp.bool_)
        # Animals react sequentially in a rotating order, over one or two physical microsteps.
        for micro in range(2):
            for offset in range(4):
                i = jnp.mod(s["t"]+offset,4)
                d = pos[i]-p
                dist = jnp.sum(jnp.abs(d),axis=1)
                w = jnp.where((dist <= jnp.where(whistle,3,2)) & ~quiet,
                              1+whistle.astype(jnp.int32),0)
                repulse = jnp.sum(jnp.sign(d)*w[:,None],axis=0,dtype=jnp.int32)
                peers = active & (jnp.arange(4) != i) & (jnp.sum(jnp.abs(pos-pos[i]),axis=1) <= 3)
                cohesion = jnp.sum(jnp.sign(pos-pos[i])*peers[:,None].astype(jnp.int32),axis=0,dtype=jnp.int32)
                force = repulse + jnp.where((jnp.sum(w) == 0) & (panic[i] < 3),cohesion,jnp.zeros(2,dtype=jnp.int32))
                nearest_pen = jnp.argmin(jnp.sum(jnp.abs(s["pen_pos"]-pos[i]),axis=1))
                target = s["pen_pos"][nearest_pen]
                pen_vector = jnp.sign(target-pos[i])
                # In the safe-side paddock, calm animals follow the nearby marked pen.
                pen_route = jnp.where(pos[i,0] != target[0],
                                      jnp.array([1,0],dtype=jnp.int32),
                                      jnp.array([0,1],dtype=jnp.int32)*pen_vector)
                force = jnp.where((pos[i,0] >= 6) & (panic[i] < 3),pen_route,force)
                horizontal = jnp.abs(force[0]) >= jnp.abs(force[1])
                step = jnp.stack([jnp.where(horizontal,jnp.sign(force[0]),0),
                                  jnp.where(horizontal,0,jnp.sign(force[1]))]).astype(jnp.int32)
                q = jnp.clip(pos[i]+step,jnp.array([0,0],dtype=jnp.int32),jnp.array([8,6],dtype=jnp.int32))
                tile = s["map"][q[1],q[0]]
                terrain = (tile != 1) & ((tile != 2) | opened) & ((tile != 3) | (panic[i] >= 3))
                agent_block = jnp.any(jnp.all(p == q,axis=1))
                herd_block = jnp.any(active & (jnp.arange(4) != i) & jnp.all(pos == q,axis=1))
                moving = active[i] & ((micro == 0) | (panic[i] >= 3)) & terrain & ~agent_block & ~herd_block
                pos = pos.at[i].set(jnp.where(moving,q,pos[i]))
                saved_i = active[i] & jnp.any(jnp.all(s["pen_pos"] == pos[i],axis=1))
                lost_i = active[i] & (s["map"][pos[i,1],pos[i,0]] == 3) & (panic[i] >= 3)
                saved_now = saved_now.at[i].set(saved_now[i] | saved_i)
                lost_now = lost_now.at[i].set(lost_now[i] | lost_i)
                active = active.at[i].set(active[i] & ~saved_i & ~lost_i)
        s["agent_pos"],s["focus"],s["signal"],s["blocked"] = p,focus,signal,blocked
        s["animal_pos"],s["animal_active"],s["panic"] = pos,active,panic
        s["animal_visible"] = jnp.broadcast_to(active,(3,4))
        s["saved"],s["lost"] = s["saved"] | saved_now,s["lost"] | lost_now
        s["saved_now"],s["lost_now"] = saved_now,lost_now
        gained = jnp.sum(saved_now.astype(jnp.int32),dtype=jnp.int32)
        lost = jnp.sum(lost_now.astype(jnp.int32),dtype=jnp.int32)
        s["saved_total"],s["lost_total"] = s["saved_total"]+gained,s["lost_total"]+lost
        s["gate_opened"] = opened & ~s["gate_open"][0]
        s["gate_open"],s["gate_timer"] = jnp.reshape(opened,(1,)),jnp.reshape(timer,(1,))
        s["t"] = s["t"]+1
        s["terminated"] = ~jnp.any(active)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        rewards = jnp.full((3,),(gained-lost).astype(jnp.float32),dtype=jnp.float32)
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
