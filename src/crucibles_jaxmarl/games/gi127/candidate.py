"""Shield Relay: directional interception, moving carts and charge exchange."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 12
    observation_size = 145
    max_steps = 80

    def refresh(self, s):
        s = dict(s)
        s["agent_active"] = s["agent_health"] > 0
        s["cart_active"] = (s["cart_health"] > 0) & ~s["cart_delivered"]
        facing = s["agent_facing"]
        perpendicular = jnp.stack([-facing[:, 1], facing[:, 0]], axis=1)
        arc = s["agent_pos"][:, None, :] + facing[:, None, :] + jnp.array([-1, 0, 1], dtype=jnp.int32)[None, :, None] * perpendicular[:, None, :]
        inside = jnp.all((arc >= jnp.array([0, 0])) & (arc <= jnp.array([10, 6])), axis=2)
        enabled = s["agent_guard"] & s["agent_active"] & (s["agent_energy"] > 0)
        s["shield_pos"] = jnp.clip(arc, jnp.array([0, 0]), jnp.array([10, 6])).reshape((9, 2))
        s["shield_active"] = (inside & enabled[:, None]).reshape(9)
        s["shield_direction"] = jnp.repeat(facing, 3, axis=0)
        s["agent_visible"] = jnp.broadcast_to(s["agent_active"], (3, 3))
        s["cart_visible"] = jnp.broadcast_to(s["cart_active"], (3, 2))
        s["hazard_visible"] = jnp.broadcast_to(s["hazard_active"], (3, 4))
        s["shield_visible"] = jnp.broadcast_to(s["shield_active"], (3, 9))
        return s

    def reset(self, key):
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "agent_pos": jnp.array([[2, 1], [2, 5], [1, 3]], dtype=jnp.int32),
            "agent_health": jnp.full(3, 3, dtype=jnp.int32),
            "agent_energy": jnp.full(3, 6, dtype=jnp.int32),
            "agent_facing": jnp.array([[0, -1], [0, 1], [1, 0]], dtype=jnp.int32),
            "agent_guard": jnp.zeros(3, dtype=jnp.bool_),
            "agent_active": jnp.ones(3, dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "cart_pos": jnp.array([[1, 2], [1, 4]], dtype=jnp.int32),
            "cart_health": jnp.full(2, 4, dtype=jnp.int32),
            "cart_delivered": jnp.zeros(2, dtype=jnp.bool_),
            "cart_active": jnp.ones(2, dtype=jnp.bool_),
            "cart_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "station_pos": jnp.array([[0, 3], [5, 3], [10, 3]], dtype=jnp.int32),
            "station_active": jnp.ones(3, dtype=jnp.bool_),
            "station_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "hazard_pos": jnp.zeros((4, 2), dtype=jnp.int32),
            "hazard_velocity": jnp.zeros((4, 2), dtype=jnp.int32),
            "hazard_ttl": jnp.zeros(4, dtype=jnp.int32),
            "hazard_active": jnp.zeros(4, dtype=jnp.bool_),
            "hazard_visible": jnp.zeros((3, 4), dtype=jnp.bool_),
            "shield_pos": jnp.zeros((9, 2), dtype=jnp.int32),
            "shield_direction": jnp.zeros((9, 2), dtype=jnp.int32),
            "shield_owner": jnp.repeat(jnp.arange(3, dtype=jnp.int32), 3),
            "shield_active": jnp.zeros(9, dtype=jnp.bool_),
            "shield_visible": jnp.zeros((3, 9), dtype=jnp.bool_),
            "intercept_event": jnp.zeros(3, dtype=jnp.int32),
            "transfer_event": jnp.zeros(3, dtype=jnp.int32),
            "receive_event": jnp.zeros(3, dtype=jnp.int32),
            "agent_hit_event": jnp.zeros(3, dtype=jnp.int32),
            "cart_hit_event": jnp.zeros(2, dtype=jnp.int32),
            "progress_event": jnp.zeros(2, dtype=jnp.bool_),
            "arrival_event": jnp.zeros(2, dtype=jnp.bool_),
            "spawn_event": jnp.zeros(4, dtype=jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        fields = ("agent_pos", "agent_health", "agent_energy", "agent_facing",
                  "agent_guard", "agent_active", "cart_pos", "cart_health",
                  "cart_delivered", "cart_active", "station_pos", "hazard_pos",
                  "hazard_velocity", "hazard_ttl", "hazard_active",
                  "shield_pos", "shield_direction", "shield_active", "shield_owner",
                  "intercept_event", "transfer_event", "receive_event",
                  "agent_hit_event", "cart_hit_event", "progress_event", "arrival_event", "spawn_event")
        pub = jnp.concatenate([s[f].ravel().astype(jnp.float32) / 16.0 for f in fields] +
                              [jnp.array([s["t"] / 128.0, s["episode_done"]], dtype=jnp.float32)])
        return jnp.concatenate([jnp.eye(3, dtype=jnp.float32),
                                jnp.broadcast_to(pub, (3, pub.shape[0]))], axis=1)

    def legal_actions(self, s):
        return (jnp.arange(12)[None, :] == 0) | (s["agent_active"][:, None] & ~s["episode_done"])

    def raw_step(self, key, state, actions):
        s = dict(state)
        a = jnp.where((actions >= 0) & (actions < 12) & s["agent_active"], actions, 0)
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        moving = (a >= 1) & (a <= 4) & s["agent_active"]
        pos = jnp.clip(s["agent_pos"] + jnp.where(moving[:, None], dirs[jnp.clip(a - 1, 0, 3)], 0), jnp.array([0, 0]), jnp.array([10, 6]))
        guarding = (a >= 5) & (a <= 8) & s["agent_active"]
        face = jnp.where(moving[:, None], dirs[jnp.clip(a - 1, 0, 3)], s["agent_facing"])
        face = jnp.where(guarding[:, None], dirs[jnp.clip(a - 5, 0, 3)], face)
        station = jnp.any(jnp.sum(jnp.abs(pos[:, None, :] - s["station_pos"][None, :, :]), axis=2) <= 1, axis=1)
        energy = jnp.minimum(6, s["agent_energy"] + ((a == 0) & station & s["agent_active"]).astype(jnp.int32) * 2)
        sent, received = jnp.zeros(3, dtype=jnp.int32), jnp.zeros(3, dtype=jnp.int32)
        # Transfers conserve charge; receiver chosen from nearby active roles by least charge, then role id.
        for k in range(3):
            i = (s["t"] + k) % 3
            nearby = (jnp.sum(jnp.abs(pos - pos[i]), axis=1) <= 1) & (jnp.arange(3) != i) & s["agent_active"]
            target = jnp.argmin(jnp.where(nearby, energy * 4 + jnp.arange(3), 99))
            amount = jnp.where((a[i] == 9) & jnp.any(nearby), jnp.minimum(jnp.minimum(2, energy[i]), 6 - energy[target]), 0)
            energy = energy.at[i].add(-amount).at[target].add(amount)
            sent = sent.at[i].add(amount)
            received = received.at[target].add(amount)
        cp, ch = s["cart_pos"], s["cart_health"]
        # Repair costs two charge for one health; no repair reward.
        for k in range(3):
            i = (s["t"] + k) % 3
            near = (jnp.sum(jnp.abs(cp - pos[i]), axis=1) <= 1) & s["cart_active"] & (ch < 4)
            c = jnp.argmax(near)
            repair = (a[i] == 10) & (energy[i] >= 2) & jnp.any(near)
            ch = ch.at[c].add(repair.astype(jnp.int32))
            energy = energy.at[i].add(-2 * repair.astype(jnp.int32))
        # Multiple escorts cannot double-speed a cart. Guards cannot also escort on this tick.
        pushing = jnp.any((jnp.sum(jnp.abs(pos[:, None, :] - cp[None, :, :]), axis=2) <= 1) &
                         (a == 11)[:, None] & s["agent_active"][:, None], axis=0) & s["cart_active"]
        cp = cp.at[:, 0].set(jnp.minimum(10, cp[:, 0] + pushing.astype(jnp.int32)))
        s["agent_pos"], s["agent_facing"], s["agent_guard"], s["agent_energy"] = pos, face, guarding, energy
        s = self.refresh(s)
        hp, hv, ha = s["hazard_pos"], s["hazard_velocity"], s["hazard_active"]
        health = s["agent_health"]
        intercepted, agent_hit = jnp.zeros(3, dtype=jnp.int32), jnp.zeros(3, dtype=jnp.int32)
        cart_hit = jnp.zeros(2, dtype=jnp.int32)
        next_hp = jnp.clip(hp + hv, jnp.array([0, 0]), jnp.array([10, 6]))
        ttl = jnp.maximum(0, s["hazard_ttl"] - ha.astype(jnp.int32))
        # Slot order is explicit. Shields intercept first, then lowest live cart, then lowest live agent.
        for h in range(4):
            candidate = hp[h] + hv[h]
            inside = jnp.all((candidate >= jnp.array([0, 0])) & (candidate <= jnp.array([10, 6])))
            flying = ha[h] & inside & (ttl[h] > 0)
            cells = s["shield_pos"].reshape((3, 3, 2))
            cell_active = s["shield_active"].reshape((3, 3))
            arc_hit = jnp.any((jnp.all(cells == candidate, axis=2) | jnp.all(cells == hp[h], axis=2)) & cell_active, axis=1)
            aimed = jnp.all(face == -hv[h], axis=1)
            can_block = arc_hit & aimed & guarding & (energy > 0) & (health > 0)
            blocker = jnp.argmax(can_block)
            blocked = flying & jnp.any(can_block)
            energy = energy.at[blocker].add(-blocked.astype(jnp.int32))
            intercepted = intercepted.at[blocker].add(blocked.astype(jnp.int32))
            hit_c = jnp.all(cp == candidate, axis=1) & (ch > 0) & ~s["cart_delivered"]
            c = jnp.argmax(hit_c)
            damage_c = flying & ~blocked & jnp.any(hit_c)
            ch = ch.at[c].add(-damage_c.astype(jnp.int32))
            cart_hit = cart_hit.at[c].add(damage_c.astype(jnp.int32))
            hit_a = jnp.all(pos == candidate, axis=1) & (health > 0)
            i = jnp.argmax(hit_a)
            damage_a = flying & ~blocked & ~damage_c & jnp.any(hit_a)
            health = health.at[i].add(-damage_a.astype(jnp.int32))
            agent_hit = agent_hit.at[i].add(damage_a.astype(jnp.int32))
            ha = ha.at[h].set(flying & ~blocked & ~damage_c & ~damage_a)
        # Scheduled telegraphed shots enter only empty slots after all current shots resolve.
        h = (s["t"] // 2) % 4
        spawn = (s["t"] % 2 == 0) & ~ha[h]
        from_top = (s["t"] // 2) % 2 == 0
        spawn_pos = jnp.array([3 + 2 * ((s["t"] // 4) % 4), jnp.where(from_top, 0, 6)], dtype=jnp.int32)
        spawn_vel = jnp.array([0, jnp.where(from_top, 1, -1)], dtype=jnp.int32)
        next_hp = next_hp.at[h].set(jnp.where(spawn, spawn_pos, next_hp[h]))
        hv = hv.at[h].set(jnp.where(spawn, spawn_vel, hv[h]))
        ttl = ttl.at[h].set(jnp.where(spawn, 7, ttl[h]))
        ha = ha.at[h].set(ha[h] | spawn)
        arrival = (cp[:, 0] == 10) & (ch > 0) & ~s["cart_delivered"]
        s["agent_health"], s["agent_energy"] = health, energy
        s["cart_pos"], s["cart_health"], s["cart_delivered"] = cp, ch, s["cart_delivered"] | arrival
        s["hazard_pos"], s["hazard_velocity"], s["hazard_active"], s["hazard_ttl"] = next_hp, hv, ha, ttl
        s["intercept_event"], s["agent_hit_event"], s["cart_hit_event"] = intercepted, agent_hit, cart_hit
        s["transfer_event"], s["receive_event"] = sent, received
        s["progress_event"], s["arrival_event"] = pushing, arrival
        s["spawn_event"] = jnp.zeros(4, dtype=jnp.bool_).at[h].set(spawn)
        s["t"] = state["t"] + jnp.array(1, dtype=jnp.int32)
        s["terminated"] = jnp.all(s["cart_delivered"] | (ch <= 0)) | jnp.all(health <= 0)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        common = jnp.sum(arrival.astype(jnp.float32)) + 0.125 * jnp.sum(pushing.astype(jnp.float32)) - 0.25 * jnp.sum(cart_hit.astype(jnp.float32))
        reward = jnp.full(3, common, dtype=jnp.float32) + 0.125 * intercepted.astype(jnp.float32) - 0.25 * agent_hit.astype(jnp.float32)
        s = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, s)
        reward = jnp.where(state["episode_done"], jnp.zeros(3, dtype=jnp.float32), reward)
        info = {"done_all": s["episode_done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full((3,), s["episode_done"], dtype=jnp.bool_), info
