"""Rhythm Foundry. Original source-only authoring; no runtime claims."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 78
    num_actions = 11
    max_steps = 96

    def reset(self, key):
        s = {
            "t": jnp.array(0, jnp.int32),
            "done": jnp.array(False), "natural": jnp.array(False),
            "map": jnp.array([[1,1,1,1,1,1,1],[1,0,0,0,0,0,1],
                             [1,0,0,0,0,0,1],[1,0,0,1,0,0,1],
                             [1,0,0,0,0,0,1],[1,0,0,0,0,0,1],
                             [1,1,1,1,1,1,1]], jnp.int32),
            "pos": jnp.array([[2,1],[3,2],[2,5]], jnp.int32),
            "active": jnp.ones(3, jnp.bool_),
            "visible": jnp.ones((3,3), jnp.bool_),
            "carry": jnp.array([1,2,0], jnp.int32),
            "order": jax.random.randint(key, (3,), 0, 2, dtype=jnp.int32),
            "bin_pos": jnp.array([[1,1],[1,5]], jnp.int32),
            "bin_active": jnp.ones(2, jnp.bool_),
            "bin_visible": jnp.ones((3,2), jnp.bool_),
            "stock": jnp.array([2,2], jnp.int32),
            "furnace_pos": jnp.array([[3,1],[3,5]], jnp.int32),
            "furnace_active": jnp.ones(2, jnp.bool_),
            "furnace_visible": jnp.ones((3,2), jnp.bool_),
            "ore": jnp.zeros(2, jnp.int32), "flux": jnp.zeros(2, jnp.int32),
            "stage": jnp.zeros(2, jnp.int32), "timer": jnp.zeros(2, jnp.int32),
            "water": jnp.array([2,2], jnp.int32),
            "dock_pos": jnp.array([[5,1],[5,5]], jnp.int32),
            "dock_active": jnp.ones(2, jnp.bool_),
            "dock_visible": jnp.ones((3,2), jnp.bool_),
            "dock_kind": jnp.array([0,1], jnp.int32),
            "delivered": jnp.zeros(3, jnp.int32),
            "waste": jnp.zeros(2, jnp.int32),
            "sale_event": jnp.zeros(3, jnp.bool_),
            "stage_event": jnp.zeros(2, jnp.bool_),
        }
        return self.observe(s), s

    def movement(self, s):
        d = jnp.array([[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        p = s["pos"][:,None,:] + d[None,:,:]
        inside = jnp.all((p >= 0) & (p <= 6), axis=-1)
        tile = s["map"][jnp.clip(p[:,:,1],0,6),jnp.clip(p[:,:,0],0,6)]
        occupied = jnp.any(jnp.all(p[:,:,None,:] == s["pos"][None,None,:,:], axis=-1), axis=-1)
        return inside & (tile == 0) & ~occupied

    def options(self, s):
        c = s["carry"]
        near = jnp.sum(jnp.abs(s["pos"][:,None,:] - s["furnace_pos"][None,:,:]), axis=-1) <= 1
        atbin = jnp.all(s["pos"][:,None,:] == s["bin_pos"][None,:,:], axis=-1)
        pick = jnp.concatenate((atbin & (s["stock"][None,:] > 0),
                                near & (s["stage"][None,:] >= 3)), axis=1) & (c[:,None] == 0)
        deposit = near & (s["stage"][None,:] == 0) & (
            ((c[:,None] == 1) & (s["ore"][None,:] == 0)) |
            ((c[:,None] == 2) & (s["flux"][None,:] == 0)))
        fire = near & (s["stage"][None,:] == 0) & (s["ore"][None,:] == 1)
        quench = near & (s["stage"][None,:] == 2) & (s["water"][None,:] > 0)
        adj = jnp.sum(jnp.abs(s["pos"][:,None,:] - s["pos"][None,:,:]), axis=-1) == 1
        passing = adj & (c[:,None] > 0) & (c[None,:] == 0)
        docks = jnp.sum(jnp.abs(s["pos"][:,None,:] - s["dock_pos"][None,:,:]), axis=-1) <= 1
        deliver = ((c == 3) | (c == 4)) & docks[:,0]
        discard = (c > 0) & docks[:,1]
        return pick, deposit, fire, quench, passing, deliver, discard

    def legal_actions(self, s):
        pick, deposit, fire, quench, passing, deliver, discard = self.options(s)
        m = jnp.concatenate((jnp.ones((3,1), jnp.bool_), self.movement(s),
            jnp.stack((jnp.any(pick,1), jnp.any(deposit,1), jnp.any(fire,1),
                       jnp.any(quench,1), jnp.any(passing,1), deliver | discard), axis=1)), axis=1)
        return m & ((~s["done"]) | (jnp.arange(11)[None,:] == 0))

    def observe(self, s):
        pub = jnp.concatenate((s["map"].reshape(-1), s["pos"].reshape(-1), s["carry"],
             s["ore"], s["flux"], s["stage"], s["timer"], s["water"], s["stock"],
             s["delivered"], s["waste"], s["t"].reshape(1))).astype(jnp.float32)
        return jnp.concatenate((jnp.broadcast_to(pub, (3,76)),
            s["order"][:,None].astype(jnp.float32),
            jnp.arange(3, dtype=jnp.float32)[:,None]), axis=1)

    def raw_step(self, key, state, actions):
        legal = self.legal_actions(state)
        a = jnp.clip(actions,0,10)
        ok = (actions >= 0) & (actions < 11) & jnp.take_along_axis(legal,a[:,None],axis=1)[:,0]
        a = jnp.where(ok,a,0)
        s = dict(state)
        ids = jnp.arange(3, dtype=jnp.int32)
        priority = jnp.mod(ids - state["t"], 3)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        proposed = state["pos"] + delta[jnp.where(a <= 4,a,0)]
        same = jnp.all(proposed[:,None,:] == proposed[None,:,:], axis=-1)
        loses = jnp.any(same & (priority[None,:] < priority[:,None]), axis=1)
        s["pos"] = jnp.where(loses[:,None], state["pos"], proposed)
        s["sale_event"] = jnp.zeros(3, jnp.bool_)
        r = jnp.zeros(3, jnp.float32)
        # Cyclic actor order resolves finite bins, slots, goods and handoffs.
        for k in range(3):
            i = jnp.mod(state["t"] + k, 3)
            pick, deposit, fire, quench, passing, deliver, discard = self.options(s)
            c = s["carry"][i]
            target = jnp.argmax(pick[i])
            b = jnp.minimum(target,1)
            f = jnp.maximum(target-2,0)
            taking = (a[i] == 5) & jnp.any(pick[i])
            frombin = taking & (target < 2)
            fromfurnace = taking & (target >= 2)
            good = s["stage"][f] == 3
            material = jnp.where(target < 2, target+1,
                jnp.where(good,3+f,jnp.where(s["flux"][f] == 1,6,5)))
            s["carry"] = s["carry"].at[i].set(jnp.where(taking,material,c))
            s["stock"] = s["stock"].at[b].add(-frombin.astype(jnp.int32))
            s["ore"] = s["ore"].at[f].set(jnp.where(fromfurnace,0,s["ore"][f]))
            s["flux"] = s["flux"].at[f].set(jnp.where(fromfurnace,0,s["flux"][f]))
            s["stage"] = s["stage"].at[f].set(jnp.where(fromfurnace,0,s["stage"][f]))
            s["timer"] = s["timer"].at[f].set(jnp.where(fromfurnace,0,s["timer"][f]))
            f = jnp.argmax(deposit[i])
            loading = (a[i] == 6) & jnp.any(deposit[i])
            s["ore"] = s["ore"].at[f].add((loading & (c == 1)).astype(jnp.int32))
            s["flux"] = s["flux"].at[f].add((loading & (c == 2)).astype(jnp.int32))
            s["carry"] = s["carry"].at[i].set(jnp.where(loading,0,s["carry"][i]))
            f = jnp.argmax(fire[i])
            firing = (a[i] == 7) & jnp.any(fire[i])
            s["stage"] = s["stage"].at[f].set(jnp.where(firing,1,s["stage"][f]))
            s["timer"] = s["timer"].at[f].set(jnp.where(firing,2,s["timer"][f]))
            f = jnp.argmax(quench[i])
            cooling = (a[i] == 8) & jnp.any(quench[i])
            s["stage"] = s["stage"].at[f].set(jnp.where(cooling,3,s["stage"][f]))
            s["timer"] = s["timer"].at[f].set(jnp.where(cooling,0,s["timer"][f]))
            s["water"] = s["water"].at[f].add(-cooling.astype(jnp.int32))
            recipients = passing[i] & (a == 0)
            j = jnp.argmin(jnp.where(recipients,priority,4))
            handoff = (a[i] == 9) & jnp.any(recipients)
            s["carry"] = s["carry"].at[j].set(jnp.where(handoff,c,s["carry"][j]))
            s["carry"] = s["carry"].at[i].set(jnp.where(handoff,0,s["carry"][i]))
            selling = (a[i] == 10) & deliver[i]
            dumping = (a[i] == 10) & discard[i] & ~selling
            ore_units = jnp.array([0,1,0,1,1,1,1], jnp.int32)[c]
            flux_units = jnp.array([0,0,1,1,1,0,1], jnp.int32)[c]
            s["waste"] = s["waste"] + dumping.astype(jnp.int32) * jnp.stack((ore_units,flux_units))
            s["carry"] = s["carry"].at[i].set(jnp.where(selling | dumping,0,s["carry"][i]))
            s["delivered"] = s["delivered"].at[i].add(selling.astype(jnp.int32))
            s["sale_event"] = s["sale_event"].at[i].set(selling)
            r = r.at[i].add(selling.astype(jnp.float32) * (1.0 + (state["order"][i] == c-3).astype(jnp.float32)))
        # A newly fired charge does not age; quenching precedes expiry.
        heating = (state["stage"] == 1) & (s["stage"] == 1)
        cooling = (state["stage"] == 2) & (s["stage"] == 2)
        s["timer"] = jnp.where(heating | cooling,jnp.maximum(s["timer"]-1,0),s["timer"])
        hot_end = heating & (s["timer"] == 0)
        cool_end = cooling & (s["timer"] == 0)
        s["stage"] = jnp.where(hot_end,jnp.where(s["flux"] == 1,2,4),s["stage"])
        s["timer"] = jnp.where(hot_end & (s["flux"] == 1),2,s["timer"])
        s["stage"] = jnp.where(cool_end,4,s["stage"])
        s["stage_event"] = s["stage"] != state["stage"]
        r = r + 0.25*jnp.sum(s["sale_event"].astype(jnp.float32))
        s["t"] = state["t"] + 1
        s["natural"] = (jnp.sum(s["stock"]) == 0) & jnp.all(s["carry"] == 0) & (
            jnp.sum(s["ore"] + s["flux"]) == 0)
        s["done"] = s["natural"] | (s["t"] >= self.max_steps)
        s = jax.tree.map(lambda new, old: jnp.where(state["done"],old,new),s,state)
        r = jnp.where(state["done"],jnp.zeros(3,jnp.float32),r)
        terminated = s["natural"]
        truncated = (s["t"] >= self.max_steps) & ~terminated
        done = terminated | truncated
        return self.observe(s), s, r, jnp.full((3,),done,jnp.bool_), {
            "done_all": done, "terminated": terminated, "truncated": truncated}
