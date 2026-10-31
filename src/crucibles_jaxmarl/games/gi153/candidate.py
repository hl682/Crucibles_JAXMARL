"""Bonded Bridges: original four-carrier reciprocal-access game; authored, unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 182
    num_actions = 14
    max_steps = 160

    def refresh(self, s):
        s = dict(s)
        ids = jnp.arange(4,dtype=jnp.int32)
        held = s["cargo_carrier"][None,:] == ids[:,None]
        s["carrying"] = jnp.where(jnp.any(held,axis=1),jnp.argmax(held,axis=1),-1).astype(jnp.int32)
        s["cargo_pos"] = jnp.where((s["cargo_carrier"] >= 0)[:,None],s["agent_pos"][jnp.clip(s["cargo_carrier"],0,3)],s["cargo_pos"])
        s["cargo_active"] = ~s["delivered"]
        s["target_active"] = ~s["delivered"]
        s["agent_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["agent_pos"][None,:,:]),axis=2) <= 3
        s["cargo_visible"] = (jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["cargo_pos"][None,:,:]),axis=2) <= 3) & s["cargo_active"][None,:]
        s["target_visible"] = ((ids[:,None] == ids[None,:]) | (ids[:,None] == s["recipient"][None,:]) | held) & s["target_active"][None,:]
        return s

    def reset(self, key):
        board = jnp.array([
            [1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,1,0,0,0,0,1],
            [1,0,3,0,0,2,0,0,3,0,1],
            [1,0,0,0,0,1,0,0,0,0,1],
            [1,0,3,0,0,2,0,0,3,0,1],
            [1,0,0,0,0,1,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1]],jnp.int32)
        target_y = jnp.where(jax.random.bernoulli(key,0.5,(4,)),2,4).astype(jnp.int32)
        positions = jnp.array([[4,2],[6,4],[3,4],[7,2]],jnp.int32)
        s = {
            "t": jnp.array(0,jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "map": board, "agent_pos": positions, "agent_active": jnp.ones(4,jnp.bool_),
            "agent_visible": jnp.ones((4,4),jnp.bool_), "carrying": jnp.arange(4,dtype=jnp.int32),
            "wallet": jnp.full(4,2,jnp.int32), "in_transit": jnp.full(4,-1,jnp.int32),
            "entry_side": jnp.array([0,1,0,1],jnp.int32),
            "cargo_pos": positions, "cargo_active": jnp.ones(4,jnp.bool_),
            "cargo_visible": jnp.ones((4,4),jnp.bool_), "cargo_carrier": jnp.arange(4,dtype=jnp.int32),
            "cargo_type": jnp.arange(4,dtype=jnp.int32), "recipient": jnp.array([1,2,3,0],jnp.int32),
            "delivered": jnp.zeros(4,jnp.bool_), "received": jnp.zeros(4,jnp.int32),
            "target_pos": jnp.stack([jnp.array([8,2,8,2],jnp.int32),target_y],axis=1),
            "target_active": jnp.ones(4,jnp.bool_), "target_visible": jnp.zeros((4,4),jnp.bool_),
            "gate_pos": jnp.array([[5,2],[5,4]],jnp.int32), "gate_active": jnp.ones(2,jnp.bool_),
            "gate_visible": jnp.ones((4,2),jnp.bool_), "gate_owner": jnp.array([0,1],jnp.int32),
            "toll": jnp.ones(2,jnp.int32), "beneficiary": jnp.full(2,-1,jnp.int32),
            "permit_ttl": jnp.zeros(2,jnp.int32), "permit_price": jnp.zeros(2,jnp.int32),
            "bond": jnp.zeros(2,jnp.int32),
            "request_gate": jnp.full(4,-1,jnp.int32), "request_ttl": jnp.zeros(4,jnp.int32),
            "request_price": jnp.zeros(4,jnp.int32),
            "deliver_event": jnp.zeros(4,jnp.bool_), "cross_event": jnp.zeros(4,jnp.bool_),
            "grant_event": jnp.zeros(2,jnp.bool_), "expire_event": jnp.zeros(2,jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s),s

    def legal_actions(self, s):
        ids = jnp.arange(4,dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        target = jnp.clip(s["agent_pos"][:,None,:]+delta[None,:,:],jnp.array([0,0],jnp.int32),jnp.array([10,6],jnp.int32))
        tile = s["map"][target[:,:,1],target[:,:,0]]
        gate = jnp.where(target[:,:,1] == 2,0,1)
        entering = (target[:,:,0] == 5) & (s["agent_pos"][:,0] != 5)[:,None]
        permit = (s["beneficiary"][gate] == ids[:,None]) & (s["permit_ttl"][gate] > 0) & (s["wallet"][:,None] >= s["permit_price"][gate])
        walk = (tile != 1) & (~entering | permit)
        mask = jnp.ones((4,14),jnp.bool_).at[:,:5].set(walk).at[:,0].set(True)
        approach = (jnp.abs(s["agent_pos"][:,0]-5) == 1)[:,None] & (s["agent_pos"][:,1,None] == s["gate_pos"][None,:,1])
        mask = mask.at[:,5:7].set(approach & (ids[:,None] != jnp.arange(2)[None,:]))
        owner = jnp.clip(ids,0,1)
        controls = jnp.array([[4,2],[6,4]],jnp.int32)
        at_controls = jnp.sum(jnp.abs(s["agent_pos"]-controls[owner]),axis=1) <= 1
        # Grant is an attempt; the public board may change before its phase executes.
        mask = mask.at[:,7].set((ids < 2) & at_controls)
        mask = mask.at[:,8].set(ids < 2)
        mask = mask.at[:,9:13].set(ids[:,None] != ids[None,:])
        return mask & (~s["episode_done"] | (jnp.arange(14)[None,:] == 0))

    def move(self, s, actions):
        ids = jnp.arange(4,dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        proposed = s["agent_pos"] + delta[jnp.clip(actions,0,4)] * ((actions >= 1) & (actions <= 4))[:,None]
        proposed = jnp.clip(proposed,jnp.array([0,0],jnp.int32),jnp.array([10,6],jnp.int32))
        occupied = jnp.any(jnp.all(proposed[:,None,:] == s["agent_pos"][None,:,:],axis=2) & (ids[:,None] != ids[None,:]),axis=1)
        possible = (actions >= 1) & (actions <= 4) & (s["map"][proposed[:,1],proposed[:,0]] != 1) & ~occupied
        gate = jnp.where(proposed[:,1] == 2,0,1)
        entering = (proposed[:,0] == 5) & (s["agent_pos"][:,0] != 5)
        possible = possible & (~entering | ((s["beneficiary"][gate] == ids) & (s["permit_ttl"][gate] > 0) & (s["wallet"] >= s["permit_price"][gate])))
        rank = jnp.mod(ids-s["t"],4)
        same = jnp.all(proposed[:,None,:] == proposed[None,:,:],axis=2)
        loses = jnp.any(same & possible[None,:] & (rank[None,:] < rank[:,None]),axis=1)
        return jnp.where((possible & ~loses)[:,None],proposed,s["agent_pos"])

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions,0,13)
        mask = self.legal_actions(state)
        actions = jnp.where((actions >= 0) & (actions < 14) & mask[jnp.arange(4),safe],safe,0)
        s = dict(state)
        s["agent_pos"] = self.move(state,actions)
        s["deliver_event"] = jnp.zeros(4,jnp.bool_)
        s["grant_event"] = jnp.zeros(2,jnp.bool_)
        s["expire_event"] = jnp.zeros(2,jnp.bool_)
        entered = (state["agent_pos"][:,0] != 5) & (s["agent_pos"][:,0] == 5)
        exited = (state["agent_pos"][:,0] == 5) & (s["agent_pos"][:,0] != 5)
        side = (s["agent_pos"][:,0] > 5).astype(jnp.int32)
        s["cross_event"] = exited & (side != state["entry_side"])
        s["entry_side"] = jnp.where(entered,(state["agent_pos"][:,0] > 5).astype(jnp.int32),state["entry_side"])
        s["in_transit"] = jnp.where(entered,jnp.where(s["agent_pos"][:,1] == 2,0,1),jnp.where(exited,-1,state["in_transit"])).astype(jnp.int32)
        reward = jnp.full(4,-0.001,jnp.float32)
        # Entering consumes one permit and transfers its quoted toll; escrow returns to issuer.
        for i in range(4):
            g = jnp.where(s["agent_pos"][i,1] == 2,0,1)
            price = jnp.where(entered[i],state["permit_price"][g],0).astype(jnp.int32)
            refund = jnp.where(entered[i],s["bond"][g],0).astype(jnp.int32)
            s["wallet"] = s["wallet"].at[i].add(-price).at[g].add(price+refund)
            reward = reward.at[i].add(-0.05*price).at[g].add(0.05*price)
            s["bond"] = s["bond"].at[g].set(jnp.where(entered[i],0,s["bond"][g]))
            s["beneficiary"] = s["beneficiary"].at[g].set(jnp.where(entered[i],-1,s["beneficiary"][g]))
            s["permit_ttl"] = s["permit_ttl"].at[g].set(jnp.where(entered[i],0,s["permit_ttl"][g]))
        # Last-tick permits were usable above. Unused expiring bonds are now refunded.
        s["permit_ttl"] = jnp.maximum(s["permit_ttl"]-1,0)
        expired = (s["permit_ttl"] == 0) & (s["bond"] > 0)
        s["wallet"] = s["wallet"].at[:2].add(jnp.where(expired,s["bond"],0))
        s["bond"] = jnp.where(expired,0,s["bond"]).astype(jnp.int32)
        s["beneficiary"] = jnp.where(expired,-1,s["beneficiary"]).astype(jnp.int32)
        s["expire_event"] = expired
        s["request_ttl"] = jnp.maximum(state["request_ttl"]-1,0)
        s["request_gate"] = jnp.where(s["request_ttl"] > 0,state["request_gate"],-1).astype(jnp.int32)
        # Grants read the previous public requests, require a real control-point visit and bond.
        rank = jnp.mod(jnp.arange(4)-state["t"],4)
        controls = jnp.array([[4,2],[6,4]],jnp.int32)
        for g in range(2):
            candidates = (state["request_gate"] == g) & (state["request_ttl"] > 0) & (jnp.arange(4) != g) & (s["wallet"] >= state["request_price"])
            who = jnp.argmin(jnp.where(candidates,rank,99))
            grant = (actions[g] == 7) & (jnp.sum(jnp.abs(s["agent_pos"][g]-controls[g])) <= 1) & (s["wallet"][g] >= 1) & (s["permit_ttl"][g] == 0) & jnp.any(candidates)
            s["wallet"] = s["wallet"].at[g].add(-grant.astype(jnp.int32))
            s["bond"] = s["bond"].at[g].set(jnp.where(grant,1,s["bond"][g]))
            s["beneficiary"] = s["beneficiary"].at[g].set(jnp.where(grant,who,s["beneficiary"][g]))
            s["permit_ttl"] = s["permit_ttl"].at[g].set(jnp.where(grant,6,s["permit_ttl"][g]))
            s["permit_price"] = s["permit_price"].at[g].set(jnp.where(grant,state["request_price"][who],s["permit_price"][g]))
            s["request_ttl"] = s["request_ttl"].at[who].set(jnp.where(grant,0,s["request_ttl"][who]))
            s["request_gate"] = s["request_gate"].at[who].set(jnp.where(grant,-1,s["request_gate"][who]))
            s["grant_event"] = s["grant_event"].at[g].set(grant)
            s["toll"] = s["toll"].at[g].set(jnp.where(actions[g] == 8,1-state["toll"][g],state["toll"][g]))
        # Requests posted now are actionable next tick and lock the old advertised price.
        for i in range(4):
            g = jnp.clip(actions[i]-5,0,1)
            request = ((actions[i] == 5) | (actions[i] == 6)) & (i != g) & (jnp.abs(s["agent_pos"][i,0]-5) == 1) & (s["agent_pos"][i,1] == s["gate_pos"][g,1])
            s["request_gate"] = s["request_gate"].at[i].set(jnp.where(request,g,s["request_gate"][i]))
            s["request_ttl"] = s["request_ttl"].at[i].set(jnp.where(request,4,s["request_ttl"][i]))
            s["request_price"] = s["request_price"].at[i].set(jnp.where(request,state["toll"][g],s["request_price"][i]))
        s = self.refresh(s)
        original = s["cargo_carrier"]
        # Hand-offs preserve parcel identity, recipient and private delivery address.
        for offset in range(4):
            i = jnp.mod(state["t"]+offset,4)
            target = jnp.clip(actions[i]-9,0,3)
            held = original == i
            p = jnp.argmax(held)
            handoff = (actions[i] >= 9) & (actions[i] <= 12) & (target != i) & jnp.any(held) & (s["cargo_carrier"][p] == i) & ~jnp.any(s["cargo_carrier"] == target) & (jnp.sum(jnp.abs(s["agent_pos"][i]-s["agent_pos"][target])) == 1)
            s["cargo_carrier"] = s["cargo_carrier"].at[p].set(jnp.where(handoff,target,s["cargo_carrier"][p]))
        s = self.refresh(s)
        for p in range(4):
            carrier = s["cargo_carrier"][p]
            i = jnp.clip(carrier,0,3)
            recipient = s["recipient"][p]
            delivery = ~s["delivered"][p] & (carrier >= 0) & (actions[i] == 13) & jnp.all(s["agent_pos"][i] == s["target_pos"][p])
            s["delivered"] = s["delivered"].at[p].set(s["delivered"][p] | delivery)
            s["deliver_event"] = s["deliver_event"].at[p].set(delivery)
            s["received"] = s["received"].at[recipient].add(delivery.astype(jnp.int32))
            s["cargo_carrier"] = s["cargo_carrier"].at[p].set(jnp.where(delivery,-1,carrier))
            s["cargo_pos"] = s["cargo_pos"].at[p].set(jnp.where(delivery,s["target_pos"][p],s["cargo_pos"][p]))
            amount = delivery.astype(jnp.float32)
            reward = reward.at[i].add(0.5*amount).at[p].add(0.25*amount).at[recipient].add(amount)
        s["t"] = state["t"] + jnp.array(1,jnp.int32)
        s["terminated"] = jnp.all(s["delivered"])
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        s = jax.tree.map(lambda old,new: jnp.where(state["episode_done"],old,new),state,s)
        reward = jnp.where(state["episode_done"],jnp.zeros(4,jnp.float32),reward).astype(jnp.float32)
        return self.observe(s),s,reward,jnp.full(4,s["episode_done"],jnp.bool_),{"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}

    def observe(self, s):
        core = jnp.stack([s["agent_pos"][:,0],s["agent_pos"][:,1],jnp.arange(4),s["carrying"],s["received"],jnp.full(4,s["t"]),jnp.full(4,s["episode_done"])],axis=1)
        request = jnp.stack([s["request_gate"],s["request_ttl"],s["request_price"]],axis=1).reshape(12)
        gates = jnp.concatenate([s["gate_pos"],s["beneficiary"][:,None],s["permit_ttl"][:,None],s["toll"][:,None],s["permit_price"][:,None],s["bond"][:,None]],axis=1).reshape(14)
        public = jnp.concatenate([s["wallet"],request,gates,s["deliver_event"],s["cross_event"],s["grant_event"],s["expire_event"],s["received"]])
        av = s["agent_visible"]
        agent_data = jnp.concatenate([s["agent_pos"],s["carrying"][:,None],s["in_transit"][:,None]],axis=1)
        agents = jnp.concatenate([jnp.where(av[:,:,None],agent_data[None,:,:],-1),av[:,:,None]],axis=2).reshape(4,20)
        cv = s["cargo_visible"]
        cargo_data = jnp.concatenate([s["cargo_pos"],s["cargo_carrier"][:,None],s["cargo_active"][:,None]],axis=1)
        cargo = jnp.concatenate([jnp.where(cv[:,:,None],cargo_data[None,:,:],-1),cv[:,:,None]],axis=2).reshape(4,20)
        tv = s["target_visible"]
        targets = jnp.concatenate([jnp.where(tv[:,:,None],s["target_pos"][None,:,:],-1),tv[:,:,None]],axis=2).reshape(4,12)
        return jnp.concatenate([core,jnp.broadcast_to(public[None,:],(4,46)),agents,cargo,targets,jnp.broadcast_to(s["map"].reshape(1,77),(4,77))],axis=1).astype(jnp.float32)
