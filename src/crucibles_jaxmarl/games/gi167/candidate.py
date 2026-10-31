"""Echo Pursuit: original unexecuted author kernel."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 40
    num_actions = 8
    max_steps = 48

    def refresh(self, s):
        s = dict(s)
        distance = jnp.sum(jnp.abs(s["pos"][:, None, :] - s["pos"][None, :, :]), axis=2)
        radius = jnp.where((jnp.arange(3) == 2) & (s["cloak"] > 0), 1, 2)
        s["agent_vis"] = (distance <= radius[None, :]) | (s["pulse"][None, :] > 0) | jnp.eye(3, dtype=jnp.bool_)
        s["decoy_active"] = s["decoy_ttl"] > 0
        dd = jnp.sum(jnp.abs(s["pos"][:, None, :] - s["decoy_pos"][None, :, :]), axis=2)
        s["decoy_vis"] = ((dd <= 2) | (jnp.arange(3)[:, None] == 2)) & s["decoy_active"][None, :]
        s["signal_active"] = s["signal_ttl"] > 0
        s["signal_vis"] = (jnp.arange(3)[:, None] == jnp.arange(2)[None, :]) & s["signal_active"][None, :]
        s["gate_open"] = jnp.array([s["t"] >= 8], jnp.bool_)
        return s

    def reset(self, key):
        x = jnp.arange(9)[None, :]
        y = jnp.arange(9)[:, None]
        wall = (x == 4) & (((y >= 1) & (y <= 3)) | ((y >= 5) & (y <= 7)))
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "map": wall.astype(jnp.int32), "pos": jnp.array([[2, 4], [2, 6], [6, 4]], jnp.int32),
            "energy": jnp.full(3, 8, jnp.int32), "budget": jnp.array([0, 0, 2], jnp.int32),
            "cloak": jnp.zeros(3, jnp.int32), "pulse": jnp.zeros(3, jnp.int32),
            "role": jnp.array([0, 0, 1], jnp.int32),
            "agent_active": jnp.ones(3, jnp.bool_), "agent_vis": jnp.eye(3, dtype=jnp.bool_),
            "decoy_pos": jnp.array([[6, 4]], jnp.int32), "decoy_ttl": jnp.zeros(1, jnp.int32),
            "decoy_active": jnp.zeros(1, jnp.bool_), "decoy_vis": jnp.zeros((3, 1), jnp.bool_),
            "signal_pos": jnp.zeros((2, 2), jnp.int32), "signal_ttl": jnp.zeros(2, jnp.int32),
            "signal_active": jnp.zeros(2, jnp.bool_), "signal_vis": jnp.zeros((3, 2), jnp.bool_),
            "gate_pos": jnp.array([[8, 4]], jnp.int32), "gate_open": jnp.zeros(1, jnp.bool_),
            "gate_active": jnp.ones(1, jnp.bool_), "gate_vis": jnp.ones((3, 1), jnp.bool_),
            "scan_event": jnp.zeros(2, jnp.bool_),
            "capture": jnp.array(False), "escape": jnp.array(False)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        mask = jnp.zeros((3, 8), jnp.bool_).at[:, 0].set(True)
        for i in range(3):
            live = ~s["done"]
            for a in range(1, 5):
                p = s["pos"][i] + delta[a]
                inside = jnp.all((p >= 0) & (p <= 8))
                q = jnp.clip(p, 0, 8)
                mask = mask.at[i, a].set(live & inside & (s["map"][q[1], q[0]] == 0))
        mask = mask.at[:2, 5].set(~s["done"] & (s["energy"][:2] >= 2))
        mask = mask.at[2, 5].set(~s["done"] & (s["energy"][2] >= 2) & (s["budget"][2] > 0))
        mask = mask.at[:2, 6].set(~s["done"] & (s["energy"][:2] >= 1) & (s["signal_ttl"] > 0))
        mask = mask.at[2, 6].set(~s["done"] & (s["energy"][2] >= 1) & (s["cloak"][2] == 0))
        # Tag is a spatial attempt; a moving opponent may escape before settlement.
        mask = mask.at[:2, 7].set(~s["done"])
        mask = mask.at[2, 7].set(~s["done"] & s["gate_open"][0] & jnp.all(s["pos"][2] == s["gate_pos"][0]))
        return mask

    def observe(self, s):
        own = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["energy"], s["budget"], s["cloak"]), axis=1)
        clock = jnp.broadcast_to(jnp.stack((s["t"], s["gate_open"][0].astype(jnp.int32)))[None, :], (3, 2))
        av = s["agent_vis"]
        agents = jnp.concatenate((jnp.where(av[:, :, None], s["pos"][None, :, :], -1), av[:, :, None].astype(jnp.int32), jnp.where(av, s["pulse"][None, :], 0)[:, :, None]), axis=2).reshape(3, 12)
        dv = s["decoy_vis"]
        decoy = jnp.concatenate((jnp.where(dv[:, :, None], s["decoy_pos"][None, :, :], -1), dv[:, :, None].astype(jnp.int32), jnp.where(dv, s["decoy_ttl"][None, :], 0)[:, :, None]), axis=2).reshape(3, 4)
        sv = s["signal_vis"]
        signal = jnp.concatenate((jnp.where(sv[:, :, None], s["signal_pos"][None, :, :], -1), sv[:, :, None].astype(jnp.int32), jnp.where(sv, s["signal_ttl"][None, :], 0)[:, :, None]), axis=2).reshape(3, 8)
        public = jnp.concatenate((s["gate_pos"].reshape(-1), s["scan_event"].astype(jnp.int32), jnp.stack((s["capture"], s["escape"])).astype(jnp.int32)))
        return jnp.concatenate((own, jnp.eye(3, dtype=jnp.int32), clock, agents, decoy, signal, jnp.broadcast_to(public[None, :], (3, 6))), axis=1).astype(jnp.float32)

    def raw_step(self, key, s, actions):
        ac = jnp.clip(actions, 0, 7)
        a = jnp.where((actions >= 0) & (actions < 8) & self.legal_actions(s)[jnp.arange(3), ac], ac, 0)
        n = dict(s)
        n["t"] = s["t"] + jnp.array(1, jnp.int32)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        n["pos"] = s["pos"] + delta[a]
        n["pulse"] = jnp.maximum(s["pulse"] - 1, 0)
        n["cloak"] = jnp.maximum(s["cloak"] - 1, 0)
        n["decoy_ttl"] = jnp.maximum(s["decoy_ttl"] - 1, 0)
        n["signal_ttl"] = jnp.maximum(s["signal_ttl"] - 1, 0)
        cost = jnp.where(a == 5, 2, jnp.where(a == 6, 1, 0))
        n["energy"] = jnp.clip(s["energy"] - cost + (a == 0).astype(jnp.int32), 0, 8)
        # DECOY: a finite physical beacon can be mistaken for the controlled evader.
        place = a[2] == 5
        n["decoy_pos"] = jnp.where(place, n["pos"][2][None, :], s["decoy_pos"])
        n["decoy_ttl"] = jnp.where(place, jnp.full(1, 4, jnp.int32), n["decoy_ttl"])
        n["budget"] = s["budget"].at[2].add(-place.astype(jnp.int32))
        n["cloak"] = n["cloak"].at[2].set(jnp.where(a[2] == 6, 2, n["cloak"][2]))
        # RELAY: copy old contact coordinates with decayed age; relay never refreshes TTL.
        for i in range(2):
            recipient = 1 - i
            relay = (a[i] == 6) & (s["signal_ttl"][i] > 1)
            n["signal_pos"] = n["signal_pos"].at[recipient].set(jnp.where(relay, s["signal_pos"][i], n["signal_pos"][recipient]))
            n["signal_ttl"] = n["signal_ttl"].at[recipient].set(jnp.where(relay, jnp.maximum(s["signal_ttl"][i] - 1, 0), n["signal_ttl"][recipient]))
        # SCAN: radius five, nearest echo; ties favor decoy, no identity oracle.
        n["scan_event"] = a[:2] == 5
        for i in range(2):
            real_d = jnp.sum(jnp.abs(n["pos"][i] - n["pos"][2]))
            false_d = jnp.sum(jnp.abs(n["pos"][i] - n["decoy_pos"][0]))
            use_decoy = (n["decoy_ttl"][0] > 0) & (false_d <= real_d)
            nearest = jnp.where(use_decoy, false_d, real_d)
            found = n["scan_event"][i] & (nearest <= 5)
            contact = jnp.where(use_decoy, n["decoy_pos"][0], n["pos"][2])
            n["signal_pos"] = n["signal_pos"].at[i].set(jnp.where(found, contact, n["signal_pos"][i]))
            n["signal_ttl"] = n["signal_ttl"].at[i].set(jnp.where(found, 3, jnp.where(n["scan_event"][i], 0, n["signal_ttl"][i])))
            n["pulse"] = n["pulse"].at[i].set(jnp.where(n["scan_event"][i], 2, n["pulse"][i]))
        # RESOLUTION: actual proximity and explicit attempts, capture wins a tie.
        contact = jnp.sum(jnp.abs(n["pos"][:2] - n["pos"][2]), axis=1) <= 1
        n["capture"] = jnp.any((a[:2] == 7) & contact)
        n["escape"] = (a[2] == 7) & (n["t"] >= 8) & jnp.all(n["pos"][2] == n["gate_pos"][0]) & ~n["capture"]
        outcome = n["capture"].astype(jnp.int32) - n["escape"].astype(jnp.int32)
        reward = -cost + jnp.array([20, 20, -20], jnp.int32) * outcome
        n["terminated"] = n["capture"] | n["escape"]
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        n = {k: jnp.where(s["done"], s[k], n[k]) for k in s}
        reward = jnp.where(s["done"], 0, reward).astype(jnp.float32)
        return self.observe(n), n, reward, jnp.full(3, n["done"], jnp.bool_), {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
