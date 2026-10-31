"""Canal Seal Commonwealth. Original spatial guild game; unexecuted author source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 160
    observation_size = 242
    num_actions = 20
    max_steps = 896

    def distance(self, p, q):
        return jnp.sum(jnp.abs(p[:, None, :] - q[None, :, :]), axis=-1)

    def sight(self, p, q, active, count, radius):
        d = self.distance(p, q)
        order = jnp.argsort(jnp.where(active[None, :], d, 10000) * (q.shape[0] + 1) + jnp.arange(q.shape[0])[None, :], axis=1)
        return (jnp.argsort(order, axis=1) < count) & (d <= radius) & active[None, :], order[:, :count]

    def refresh(self, s):
        s = dict(s)
        for name in ('t', 'pos', 'guild', 'tool', 'wood', 'seal', 'knowledge', 'message', 'event', 'recipient', 'gate_pos', 'water', 'health', 'bypass', 'guest', 'ttl', 'uses', 'coins', 'grain', 'delivered', 'loss', 'toll', 'forecast', 'barge_pos', 'owner', 'cargo', 'hull', 'destination', 'barge_toll', 'tile'):
            s[name] = s[name].astype(jnp.int32)
        s['actor_visible'], _ = self.sight(s['pos'], s['pos'], s['active'], 8, 4)
        s['actor_visible'] = s['actor_visible'] | jnp.eye(160, dtype=jnp.bool_)
        s['gate_visible'], _ = self.sight(s['pos'], s['gate_pos'], s['gate_active'], 4, 9)
        s['barge_visible'], _ = self.sight(s['pos'], s['barge_pos'], s['barge_active'], 4, 6)
        return s

    def reset(self, key):
        ids = jnp.arange(160, dtype=jnp.int32)
        guild, slot = ids // 16, ids % 16
        starts = jnp.array([[4, 6], [4, 9], [1, 6], [1, 8], [1, 11], [1, 2], [2, 6], [2, 8], [3, 6], [3, 8], [5, 6], [5, 8], [2, 2], [2, 3], [2, 10], [3, 10]], jnp.int32)
        g = jnp.arange(10, dtype=jnp.int32)
        yy, xx = jnp.arange(16)[:, None], jnp.arange(80)[None, :]
        tile = jnp.where(yy == 7, 1, jnp.zeros((16, 80), jnp.int32))
        tile = jnp.where((xx % 8 == 4) & (yy == 7), 2, tile)
        tile = jnp.where((xx % 8 == 4) & (yy == 9), 3, tile)
        tile = jnp.where((xx % 8 == 1) & (yy == 2), 4, tile)
        tile = jnp.where((xx % 8 == 1) & (yy == 11), 5, tile)
        tile = jnp.where((xx % 8 == 1) & (yy == 5), 6, tile)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False),
            'pos': starts[slot] + jnp.stack((guild * 8, jnp.zeros(160, jnp.int32)), axis=1),
            'active': jnp.ones(160, jnp.bool_), 'guild': guild,
            'tool': jnp.where(slot == 0, 0, jnp.where(slot == 1, 3, jnp.where((slot == 2) | (slot == 3), 2, slot % 4))).astype(jnp.int32),
            'wood': jnp.zeros(160, jnp.int32), 'seal': jnp.full(160, -1, jnp.int32),
            'knowledge': jnp.full(160, -1, jnp.int32), 'message': jnp.zeros(160, jnp.int32),
            'event': jnp.zeros(160, jnp.int32), 'recipient': jnp.full(160, -1, jnp.int32),
            'actor_visible': jnp.zeros((160, 160), jnp.bool_),
            'gate_pos': jnp.stack((g * 8 + 4, jnp.full(10, 7)), axis=1), 'gate_active': jnp.ones(10, jnp.bool_),
            'gate_visible': jnp.zeros((160, 10), jnp.bool_), 'water': jnp.full(10, 3, jnp.int32),
            'health': jnp.full(10, 3, jnp.int32), 'open': jnp.zeros(10, jnp.bool_),
            'verified': jnp.zeros(10, jnp.bool_), 'bypass': jnp.zeros(10, jnp.int32),
            'guest': jnp.full(10, -1, jnp.int32), 'ttl': jnp.zeros(10, jnp.int32), 'uses': jnp.zeros(10, jnp.int32),
            'coins': jnp.full(10, 8, jnp.int32), 'grain': jnp.full(10, 8, jnp.int32),
            'delivered': jnp.zeros(10, jnp.int32), 'mill': jnp.zeros(10, jnp.bool_),
            'loss': jnp.zeros(10, jnp.int32), 'toll': jnp.zeros(10, jnp.int32),
            'forecast': jax.random.randint(key, (10,), 0, 3, dtype=jnp.int32),
            'barge_pos': jnp.stack((g * 8 + 1, jnp.full(10, 7)), axis=1), 'barge_active': jnp.ones(10, jnp.bool_),
            'barge_visible': jnp.zeros((160, 10), jnp.bool_), 'owner': g, 'cargo': jnp.full(10, 3, jnp.int32),
            'hull': jnp.full(10, 3, jnp.int32), 'destination': jnp.minimum(g * 8 + 9, 78),
            'arrived': jnp.zeros(10, jnp.bool_), 'moved': jnp.zeros(10, jnp.bool_),
            'barge_toll': jnp.zeros(10, jnp.int32), 'tile': tile.astype(jnp.int32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def context(self, s):
        d = self.distance(s['pos'], s['gate_pos'])
        gate = jnp.argmin(d, axis=1)
        bd = self.distance(s['pos'], s['barge_pos'])
        barge = jnp.argmin(bd, axis=1)
        return gate, d[jnp.arange(160), gate] <= 1, barge, bd[jnp.arange(160), barge] <= 1

    def legal_actions(self, s):
        gate, ng, barge, nb = self.context(s)
        p = s['pos']
        local = jnp.stack((p[:, 0] % 8, p[:, 1]), axis=1)
        def at(xy):
            return jnp.sum(jnp.abs(local - jnp.array(xy)), axis=1) <= 1
        moves = p[:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        bounded = jnp.all((moves >= 0) & (moves <= jnp.array([79, 15])), axis=-1)
        inspect = jnp.any(self.distance(p, s['gate_pos'] + jnp.array([0, 2])) == 0, axis=1)
        adjacent = (self.distance(p, p) <= 1) & ~jnp.eye(160, dtype=jnp.bool_)
        m = jnp.stack((jnp.ones(160, jnp.bool_), bounded[:, 0], bounded[:, 1], bounded[:, 2], bounded[:, 3],
                       at([1, 11]) & (s['wood'] < 3), ng & (s['tool'] == 0) & (s['health'][gate] > 0),
                       inspect & (s['tool'] == 3), ng, at([1, 2]) & (s['seal'] < 0),
                       ng & (s['seal'] >= 0) & (s['guest'][gate] < 0) & (s['seal'] != gate),
                       ng & (s['seal'] < 0) & (s['guest'][gate] >= 0) & ((s['guild'] == gate) | (s['guild'] == s['guest'][gate])),
                       nb & (s['tool'] == 2) & s['barge_active'][barge], nb & (s['tool'] == 2) & s['barge_active'][barge],
                       nb & (s['barge_pos'][barge, 0] >= s['destination'][barge]) & (s['cargo'][barge] > 0),
                       (s['wood'] > 0) & ((ng & (s['health'][gate] < 3)) | (nb & (s['hull'][barge] < 3) & s['barge_active'][barge])),
                       jnp.any(adjacent, axis=1), at([1, 2]), ng & (s['wood'] > 0) & (s['bypass'][gate] < 3), s['knowledge'] >= 0), axis=1)
        return m & ((jnp.arange(20)[None, :] == 0) | ~s['done'])

    def advance(self, key, s, actions):
        ids, gates = jnp.arange(160, dtype=jnp.int32), jnp.arange(10, dtype=jnp.int32)
        safe = jnp.clip(actions, 0, 19)
        a = jnp.where((actions == safe) & self.legal_actions(s)[ids, safe], safe, 0)
        gate, ng, barge, nb = self.context(s)
        # Inspection is a separate person at a separate downstream-safety stand.
        inspected = jnp.any((self.distance(s['pos'], s['gate_pos'] + jnp.array([0, 2])) == 0) & (a[:, None] == 7), axis=0)
        downstream_closed = jnp.concatenate((~s['open'][1:], jnp.ones(1, jnp.bool_)))
        verified = inspected & downstream_closed
        want_open = jnp.any((gate[:, None] == gates[None, :]) & (a[:, None] == 6), axis=0)
        want_close = jnp.any((gate[:, None] == gates[None, :]) & (a[:, None] == 8), axis=0)
        opened = jnp.where(want_close, False, jnp.where(want_open & verified, True, s['open']))
        release = opened & (s['water'] > 0) & (((s['t'] + 1) % 4) == 0)
        incoming = jnp.concatenate((jnp.zeros(1, jnp.int32), release[:-1].astype(jnp.int32)))
        weather = ((s['t'] + 1) % 32) == 0
        water = jnp.clip(s['water'] - release.astype(jnp.int32) + incoming + weather * s['forecast'], 0, 4)
        # Seals exist in hands or on a gate, and reciprocal rights expire or are consumed.
        post = jnp.min(jnp.where((gate[:, None] == gates[None, :]) & (a[:, None] == 10), ids[:, None], 160), axis=0)
        posted = post < 160
        remove = jnp.min(jnp.where((gate[:, None] == gates[None, :]) & (a[:, None] == 11), ids[:, None], 160), axis=0)
        removed = remove < 160
        guest = jnp.where(posted, s['seal'][jnp.minimum(post, 159)], s['guest'])
        ttl = jnp.where(posted, 24, jnp.maximum(s['ttl'] - 1, 0))
        uses = jnp.where(posted, 2, s['uses'])
        guest = jnp.where(removed | (ttl == 0) | (uses == 0), -1, guest)
        seal = jnp.where(a == 9, s['guild'], s['seal'])
        seal = jnp.where((a == 10) & (post[gate] == ids), -1, seal)
        seal = jnp.where((a == 11) & (remove[gate] == ids), s['guest'][gate], seal)
        builders = (a == 18)
        added = jnp.sum((gate[:, None] == gates[None, :]) & builders[:, None], axis=0).astype(jnp.int32)
        bypass = jnp.minimum(3, s['bypass'] + added)
        gate_repair = (a == 15) & ng & (s['health'][gate] < 3)
        hull_repair = (a == 15) & ~gate_repair & nb
        health = jnp.minimum(3, s['health'] + jnp.sum((gate[:, None] == gates[None, :]) & gate_repair[:, None], axis=0).astype(jnp.int32))
        hull = jnp.minimum(3, s['hull'] + jnp.sum((barge[:, None] == gates[None, :]) & hull_repair[:, None], axis=0).astype(jnp.int32))
        # Two ropes, one on each bank, must agree on direction; opposite proposals cancel.
        dx = s['pos'][:, None, 0] == s['barge_pos'][None, :, 0]
        north = dx & (s['pos'][:, None, 1] == 6)
        south = dx & (s['pos'][:, None, 1] == 8)
        east = jnp.any(north & (a[:, None] == 12), axis=0) & jnp.any(south & (a[:, None] == 12), axis=0)
        west = jnp.any(north & (a[:, None] == 13), axis=0) & jnp.any(south & (a[:, None] == 13), axis=0)
        direction = east.astype(jnp.int32) - west.astype(jnp.int32)
        proposed_x = jnp.clip(s['barge_pos'][:, 0] + direction, 0, 79)
        lock = proposed_x // 8
        crossing = (proposed_x % 8) == 4
        other = s['owner']
        reciprocal = (guest[lock] == other) & (guest[other] == lock) & (ttl[lock] > 0) & (ttl[other] > 0) & (uses[lock] > 0) & (uses[other] > 0)
        tariff = crossing & (lock != other) & ~reciprocal & (bypass[lock] < 3)
        capacity = ~crossing | ((opened[lock] & (water[lock] > 0)) | (bypass[lock] == 3))
        occupied = jnp.any((proposed_x[:, None] == s['barge_pos'][None, :, 0]) & ~jnp.eye(10, dtype=jnp.bool_) & s['barge_active'][None, :], axis=1)
        contested = jnp.any((proposed_x[:, None] == proposed_x[None, :]) & (gates[None, :] < gates[:, None]) & (direction[None, :] != 0) & s['barge_active'][None, :], axis=1)
        moved = (direction != 0) & capacity & ~occupied & ~contested & s['barge_active'] & (~tariff | (s['coins'][other] > 0))
        bpos = s['barge_pos'].at[:, 0].set(jnp.where(moved, proposed_x, s['barge_pos'][:, 0]))
        paid = (moved & tariff).astype(jnp.int32)
        receipts = jnp.sum(jax.nn.one_hot(lock, 10, dtype=jnp.int32) * paid[:, None], axis=0)
        expenditure = jnp.sum(jax.nn.one_hot(other, 10, dtype=jnp.int32) * paid[:, None], axis=0)
        consume = moved & crossing & reciprocal & (bypass[lock] < 3)
        used = jnp.sum(jax.nn.one_hot(lock, 10, dtype=jnp.int32) * consume[:, None], axis=0) + jnp.sum(jax.nn.one_hot(other, 10, dtype=jnp.int32) * consume[:, None], axis=0)
        uses = jnp.maximum(0, uses - used)
        guest = jnp.where(uses == 0, -1, guest)
        # Flooding and mill production are delayed material effects of upstream releases.
        mill = (water >= 2) & (water <= 3) & ~opened & (((s['t'] + 1) % 16) == 0)
        flooded = (water == 4) & (((s['t'] + 1) % 8) == 0)
        loss = jnp.minimum(s['grain'], flooded.astype(jnp.int32))
        grain = jnp.clip(s['grain'] + mill.astype(jnp.int32) - loss, 0, 32)
        health = jnp.maximum(0, health - flooded.astype(jnp.int32))
        hull = jnp.maximum(0, hull - flooded[bpos[:, 0] // 8].astype(jnp.int32))
        delivery = jnp.any((barge[:, None] == gates[None, :]) & (a[:, None] == 14), axis=0) & s['barge_active']
        destroyed = (hull == 0) & s['barge_active'] & ~delivery
        delivered = s['delivered'] + jnp.where(delivery, s['cargo'], 0)
        cargo = jnp.where(delivery | destroyed, 0, s['cargo'])
        coins = jnp.clip(s['coins'] + receipts - expenditure + mill.astype(jnp.int32) + delivery.astype(jnp.int32) * 3, 0, 100)
        wood = jnp.clip(s['wood'] + (a == 5).astype(jnp.int32) - ((a == 15) | builders).astype(jnp.int32), 0, 3)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[jnp.minimum(a, 4)]
        pos = s['pos'] + jnp.where(((a >= 1) & (a <= 4))[:, None], delta, 0)
        adjacent = (self.distance(pos, pos) <= 1) & ~jnp.eye(160, dtype=jnp.bool_)
        partner = jnp.argmin(jnp.where(adjacent & (a[None, :] == 16), ids[None, :], 1000), axis=1)
        swap = (a == 16) & adjacent[ids, partner] & (a[partner] == 16) & (partner[partner] == ids)
        tool = jnp.where(swap, s['tool'][partner], s['tool'])
        message = jnp.where(a == 19, s['knowledge'] + 1, 0)
        hear = adjacent & (message[None, :] > 0)
        knowledge = jnp.where(jnp.any(hear, axis=1), message[jnp.argmax(hear, axis=1)] - 1, s['knowledge'])
        forecast = jnp.where(weather, jax.random.randint(key, (10,), 0, 3, dtype=jnp.int32), s['forecast'])
        knowledge = jnp.where(a == 17, forecast[s['pos'][:, 0] // 8], knowledge)
        arrived = s['arrived'] | delivery
        active = s['barge_active'] & ~delivery & ~destroyed
        terminal = ~jnp.any(active)
        t = s['t'] + jnp.array(1, jnp.int32)
        reward = -0.003 * (a != 0) + 0.25 * delivery[s['guild']] - 0.2 * destroyed[s['guild']] + 0.02 * mill[s['guild']] - 0.1 * loss[s['guild']] + 0.01 * (receipts - expenditure)[s['guild']]
        reward = reward + 0.02 * (gate_repair | hull_repair) + 0.001 * jnp.sum((bypass == 3) & (s['bypass'] < 3))
        n = dict(s)
        n['t'], n['done'], n['terminated'] = t, terminal | (t >= 896), terminal
        n['pos'], n['tool'], n['wood'], n['seal'] = pos, tool, wood, seal
        n['knowledge'], n['message'], n['event'], n['recipient'] = knowledge, message, a, jnp.where(swap, partner, -1)
        n['water'], n['health'], n['open'], n['verified'], n['bypass'] = water, health, opened, verified, bypass
        n['guest'], n['ttl'], n['uses'], n['coins'], n['grain'] = guest, ttl, uses, coins, grain
        n['delivered'], n['mill'], n['loss'], n['toll'], n['forecast'] = delivered, mill, loss, receipts, forecast
        n['barge_pos'], n['barge_active'], n['cargo'], n['hull'], n['arrived'], n['moved'], n['barge_toll'] = bpos, active, cargo, hull, arrived, moved, paid
        return self.refresh(n), reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(160, jnp.float32)), lambda _: self.advance(key, state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': (n['t'] >= 896) & ~n['terminated']}
        return self.observe(n), n, reward, jnp.full(160, n['done'], jnp.bool_), info

    def encode(self, s, posname, actname, count, radius, values):
        visible, ix = self.sight(s['pos'], s[posname], s[actname], count, radius)
        mask = jnp.take_along_axis(visible, ix, axis=1)
        data = jnp.concatenate((jnp.ones((s[posname].shape[0], 1)), jnp.arange(s[posname].shape[0])[:, None], s[posname], values), axis=1)
        return jnp.where(mask[:, :, None], data[ix], -1).reshape((160, -1))

    def observe(self, s):
        own = jnp.stack((jnp.arange(160), s['guild'], s['tool'], s['pos'][:, 0], s['pos'][:, 1], s['wood'], s['seal'], s['knowledge'], s['message'], s['event'], s['recipient'], s['coins'][s['guild']], s['grain'][s['guild']], s['delivered'][s['guild']]), axis=1)
        actors = self.encode(s, 'pos', 'active', 8, 4, jnp.stack((s['guild'], s['tool'], s['wood'], s['seal'], s['message'], s['event'], s['recipient']), axis=1))
        gates = self.encode(s, 'gate_pos', 'gate_active', 4, 9, jnp.stack((s['water'], s['health'], s['open'], s['verified'], s['bypass'], s['guest'], s['ttl'], s['uses'], s['coins'], s['grain'], s['mill'], s['loss'], s['toll']), axis=1))
        barges = self.encode(s, 'barge_pos', 'barge_active', 4, 6, jnp.stack((s['owner'], s['cargo'], s['hull'], s['destination'], s['arrived'], s['moved'], s['barge_toll']), axis=1))
        offsets = jnp.stack((jnp.arange(25) % 5 - 2, jnp.arange(25) // 5 - 2), axis=1)
        q = s['pos'][:, None, :] + offsets[None, :, :]
        inside = jnp.all((q >= 0) & (q <= jnp.array([79, 15])), axis=-1)
        patch = jnp.where(inside, s['tile'][jnp.clip(q[:, :, 1], 0, 15), jnp.clip(q[:, :, 0], 0, 79)], -1)
        public = jnp.broadcast_to(jnp.stack((s['t'], s['done'], s['terminated'])), (160, 3))
        return jnp.concatenate((own, actors, gates, barges, patch, public), axis=1).astype(jnp.float32)
