"""Lifeline Grid: original unexecuted author kernel."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 141
    num_actions = 10
    max_steps = 72

    def reset(self, key):
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "map": jnp.zeros((7, 9), jnp.int32),
            "pos": jnp.array([[3, 3], [5, 3], [1, 1], [7, 5]], jnp.int32),
            "supply": jnp.full(4, 2, jnp.int32), "value": jax.random.randint(key, (4,), 1, 4, dtype=jnp.int32),
            "agent_active": jnp.ones(4, jnp.bool_), "agent_vis": jnp.ones((4, 4), jnp.bool_),
            "node_pos": jnp.array([[4, 3], [2, 3], [6, 3], [1, 1], [7, 1], [7, 5], [1, 5]], jnp.int32),
            "node_active": jnp.ones(7, jnp.bool_), "node_vis": jnp.ones((4, 7), jnp.bool_),
            "storage": jnp.array([12, 0, 0, 0, 0, 0, 0], jnp.int32),
            "stock": jnp.array([4, 0, 0, 0, 0, 0, 0], jnp.int32),
            "link_pos": jnp.array([[3, 3], [5, 3], [1, 2], [6, 2], [6, 4], [1, 4], [4, 3]], jnp.int32),
            "ends": jnp.array([[0, 1], [0, 2], [1, 3], [2, 4], [2, 5], [1, 6], [1, 2]], jnp.int32),
            "link_active": jnp.ones(7, jnp.bool_), "link_vis": jnp.ones((4, 7), jnp.bool_),
            "closed": jnp.array([True, True, True, True, True, True, False]),
            "health": jnp.full(7, 3, jnp.int32), "stress": jnp.zeros(7, jnp.int32),
            "flow": jnp.zeros(7, jnp.int32),
            "service_pos": jnp.array([[1, 1], [7, 1], [7, 5], [1, 5]], jnp.int32),
            "service_active": jnp.ones(4, jnp.bool_), "service_vis": jnp.ones((4, 4), jnp.bool_),
            "enabled": jnp.ones(4, jnp.bool_), "delivered": jnp.zeros(4, jnp.int32),
            "served": jnp.zeros(4, jnp.int32),
            "blackout": jnp.array(0, jnp.int32), "outcome": jnp.array(0, jnp.int32),
            "boost": jnp.array(False), "tripped": jnp.zeros(7, jnp.bool_),
            "repaired": jnp.zeros(7, jnp.bool_), "toggled": jnp.zeros(7, jnp.bool_)
        }
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        mask = jnp.zeros((4, 10), jnp.bool_).at[:, 0].set(True)
        for i in range(4):
            live = ~s["done"]
            for a in range(1, 5):
                p = s["pos"][i] + delta[a]
                mask = mask.at[i, a].set(live & jnp.all((p >= 0) & (p <= jnp.array([8, 6]))))
            dist = jnp.sum(jnp.abs(s["link_pos"] - s["pos"][i]), axis=1)
            e = jnp.argmin(dist)
            near = dist[e] <= 1
            mask = mask.at[i, 5].set(live & near)
            mask = mask.at[i, 6].set(live & near & (s["supply"][i] > 0) & ((s["health"][e] < 3) | (s["stress"][e] > 0)))
            generator = jnp.sum(jnp.abs(s["pos"][i] - s["node_pos"][0])) <= 1
            mask = mask.at[i, 7].set(live & generator & (s["stock"][0] > 0) & (s["supply"][i] < 3))
            mask = mask.at[i, 8].set(live & generator & (s["storage"][0] > 0))
            sd = jnp.sum(jnp.abs(s["service_pos"] - s["pos"][i]), axis=1)
            mask = mask.at[i, 9].set(live & (jnp.min(sd) <= 1))
        return mask

    def observe(self, s):
        own = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["supply"], s["value"]), axis=1)
        public = jnp.concatenate((s["pos"].reshape(-1), s["supply"], s["node_pos"].reshape(-1), s["storage"], s["stock"], s["link_pos"].reshape(-1), s["closed"].astype(jnp.int32), s["health"], s["stress"], s["flow"], s["ends"].reshape(-1), s["enabled"].astype(jnp.int32), s["delivered"], s["served"], jnp.stack((s["t"], s["blackout"], s["outcome"], s["boost"].astype(jnp.int32))), s["tripped"].astype(jnp.int32), s["repaired"].astype(jnp.int32), s["toggled"].astype(jnp.int32)))
        return jnp.concatenate((own, jnp.eye(4, dtype=jnp.int32), jnp.broadcast_to(public[None, :], (4, 133))), axis=1).astype(jnp.float32)

    def power(self, s):
        # ROUTING: shortest conducting paths, edge-index tie break, no policy oracle.
        live = s["closed"] & (s["health"] > 0)
        ends = s["ends"]
        distance = jnp.full(7, 99, jnp.int32).at[0].set(0)
        for hop in range(7):
            old = distance
            for e in range(7):
                u, v = ends[e, 0], ends[e, 1]
                distance = distance.at[v].set(jnp.minimum(distance[v], jnp.where(live[e], old[u] + 1, 99)))
                distance = distance.at[u].set(jnp.minimum(distance[u], jnp.where(live[e], old[v] + 1, 99)))
        need = jnp.where(s["enabled"] & (s["delivered"] < 24), jnp.minimum(2, 24 - s["delivered"]), 0)
        served = jnp.where(distance[3:] < 99, need, 0)
        flow = jnp.zeros(7, jnp.int32)
        for k in range(4):
            node = jnp.array(k + 3, jnp.int32)
            for hop in range(7):
                adjacent = (ends[:, 0] == node) | (ends[:, 1] == node)
                other = jnp.where(ends[:, 0] == node, ends[:, 1], ends[:, 0])
                parent = adjacent & live & (distance[other] < distance[node])
                e = jnp.argmin(jnp.where(parent, jnp.arange(7), 99))
                walking = (node != 0) & (distance[node] < 99) & jnp.any(parent)
                flow = flow.at[e].add(jnp.where(walking, served[k], 0))
                node = jnp.where(walking, other[e], 0)
        return served, flow

    def raw_step(self, key, s, actions):
        ac = jnp.clip(actions, 0, 9)
        a = jnp.where((actions >= 0) & (actions < 10) & self.legal_actions(s)[jnp.arange(4), ac], ac, 0)
        n = dict(s)
        n["t"] = s["t"] + jnp.array(1, jnp.int32)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        n["pos"] = s["pos"] + delta[a]
        n["repaired"] = jnp.zeros(7, jnp.bool_)
        n["toggled"] = jnp.zeros(7, jnp.bool_)
        n["boost"] = jnp.array(False)
        costs = jnp.zeros(4, jnp.int32)
        # MAINTENANCE: finite shared stock and reserve, rotating request priority.
        for k in range(4):
            i = (s["t"] + k) % 4
            d = jnp.sum(jnp.abs(n["link_pos"] - n["pos"][i]), axis=1)
            e = jnp.argmin(d)
            toggle = (a[i] == 5) & (d[e] <= 1)
            n["closed"] = n["closed"].at[e].set(n["closed"][e] ^ toggle)
            n["toggled"] = n["toggled"].at[e].set(n["toggled"][e] | toggle)
            repair = (a[i] == 6) & (d[e] <= 1) & (n["supply"][i] > 0) & ((n["health"][e] < 3) | (n["stress"][e] > 0))
            n["health"] = n["health"].at[e].set(jnp.where(repair, 3, n["health"][e]))
            n["stress"] = n["stress"].at[e].set(jnp.where(repair, 0, n["stress"][e]))
            n["supply"] = n["supply"].at[i].add(-repair.astype(jnp.int32))
            n["repaired"] = n["repaired"].at[e].set(n["repaired"][e] | repair)
            generator = jnp.sum(jnp.abs(n["pos"][i] - n["node_pos"][0])) <= 1
            refill = (a[i] == 7) & generator & (n["stock"][0] > 0) & (n["supply"][i] < 3)
            n["stock"] = n["stock"].at[0].add(-refill.astype(jnp.int32))
            n["supply"] = n["supply"].at[i].add(refill.astype(jnp.int32))
            boost = (a[i] == 8) & generator & (n["storage"][0] > 0) & ~n["boost"]
            n["storage"] = n["storage"].at[0].add(-boost.astype(jnp.int32))
            n["boost"] = n["boost"] | boost
            sd = jnp.sum(jnp.abs(n["service_pos"] - n["pos"][i]), axis=1)
            site = jnp.argmin(sd)
            shed = (a[i] == 9) & (sd[site] <= 1)
            n["enabled"] = n["enabled"].at[site].set(n["enabled"][site] ^ shed)
            costs = costs.at[i].set(repair.astype(jnp.int32) + boost.astype(jnp.int32))
        n["served"], n["flow"] = self.power(n)
        # OVERLOAD: transmit this tick, then each third consecutive excess-load tick costs one health.
        capacity = jnp.array([3, 3, 2, 2, 2, 2, 3], jnp.int32) + n["boost"].astype(jnp.int32)
        excess = (n["flow"] > capacity) & n["closed"] & (n["health"] > 0)
        stress = jnp.where(excess, n["stress"] + 1, 0)
        n["tripped"] = stress >= 3
        n["health"] = jnp.maximum(n["health"] - n["tripped"].astype(jnp.int32), 0)
        n["stress"] = jnp.where(n["tripped"], 0, stress)
        n["delivered"] = jnp.minimum(s["delivered"] + n["served"], 24)
        completed = (s["delivered"] < 24) & (n["delivered"] == 24)
        n["blackout"] = jnp.where(jnp.sum(n["served"]) == 0, s["blackout"] + 1, 0)
        success = jnp.all(n["delivered"] == 24)
        failure = (n["blackout"] >= 6) & ~success
        n["outcome"] = jnp.where(success, 1, jnp.where(failure, 2, 0)).astype(jnp.int32)
        reward = n["served"] * s["value"] + 8 * completed.astype(jnp.int32) - costs
        n["terminated"] = success | failure
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        n = {k: jnp.where(s["done"], s[k], n[k]) for k in s}
        reward = jnp.where(s["done"], 0, reward).astype(jnp.float32)
        return self.observe(n), n, reward, jnp.full(4, n["done"], jnp.bool_), {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
