"""Skyline Lunch Lifts. Original pure-array kitchen; author has not executed it."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 20
    observation_size = 196
    num_actions = 20
    max_steps = 640

    def distance(self, p, q):
        return jnp.sum(jnp.abs(p[:, None, :] - q[None, :, :]), axis=-1)

    def sight(self, p, q, active, count, radius):
        d = self.distance(p, q)
        order = jnp.argsort(jnp.where(active[None, :], d, 10000) * (q.shape[0] + 1) + jnp.arange(q.shape[0])[None, :], axis=1)
        return (jnp.argsort(order, axis=1) < count) & (d <= radius) & active[None, :], order[:, :count]

    def refresh(self, s):
        s = dict(s)
        for name in ('t', 'pos', 'team', 'item', 'recipe', 'age', 'knowledge', 'message', 'event', 'recipient', 'kitchen_pos', 'pot_recipe', 'pot_time', 'pot_age', 'order', 'stock', 'delivered', 'dirty', 'lift_pos', 'lift_item', 'lift_recipe', 'lift_age', 'loader', 'target', 'travel', 'tile'):
            s[name] = s[name].astype(jnp.int32)
        s['actor_visible'], _ = self.sight(s['pos'], s['pos'], s['active'], 8, 3)
        s['actor_visible'] = s['actor_visible'] | jnp.eye(20, dtype=jnp.bool_)
        s['kitchen_visible'], _ = self.sight(s['pos'], s['kitchen_pos'], s['kitchen_active'], 3, 6)
        s['lift_visible'], _ = self.sight(s['pos'], s['lift_pos'], s['lift_active'], 2, 4)
        return s

    def reset(self, key):
        ids = jnp.arange(20, dtype=jnp.int32)
        team, slot = ids // 4, ids % 4
        order = jax.random.randint(key, (5,), 1, 3, dtype=jnp.int32)
        start = jnp.array([[3, 1], [4, 2], [6, 2], [10, 2]], jnp.int32)
        pos = start[slot] + jnp.stack((jnp.zeros(20, jnp.int32), team * 5), axis=1)
        yy, xx = jnp.arange(25)[:, None], jnp.arange(15)[None, :]
        tile = jnp.where((yy % 5 == 4) & (xx > 0), 1, 0)
        tile = jnp.where((xx == 1) & (yy % 5 == 1), 2, tile)
        tile = jnp.where((xx == 3) & (yy % 5 == 1), 3, tile)
        tile = jnp.where((xx == 4) & (yy % 5 == 3), 4, tile)
        tile = jnp.where((xx == 5) & (yy % 5 == 1), 5, tile)
        tile = jnp.where(((xx == 6) | (xx == 8)) & (yy % 5 == 2), 6, tile)
        tile = jnp.where((xx == 11) & (yy % 5 == 1), 7, tile)
        tile = jnp.where((xx == 11) & (yy % 5 == 3), 8, tile)
        item = jnp.where((slot == 0) | (slot == 2), 1, jnp.where(slot == 3, 5, 0)).astype(jnp.int32)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False),
            'pos': pos, 'active': jnp.ones(20, jnp.bool_), 'team': team, 'item': item,
            'recipe': jnp.where(item > 0, order[team], 0), 'age': jnp.zeros(20, jnp.int32),
            'knowledge': jnp.where(slot == 3, order[team], 0), 'message': jnp.zeros(20, jnp.int32),
            'event': jnp.zeros(20, jnp.int32), 'recipient': jnp.full(20, -1, jnp.int32),
            'actor_visible': jnp.zeros((20, 20), jnp.bool_),
            'kitchen_pos': jnp.stack((jnp.full(5, 4), jnp.arange(5) * 5 + 3), axis=1).astype(jnp.int32),
            'kitchen_active': jnp.ones(5, jnp.bool_), 'kitchen_visible': jnp.zeros((20, 5), jnp.bool_),
            'freezer': jnp.zeros(5, jnp.bool_), 'pot_recipe': jnp.zeros(5, jnp.int32),
            'pot_time': jnp.zeros(5, jnp.int32), 'pot_age': jnp.zeros(5, jnp.int32),
            'order': order, 'stock': jnp.full(5, 12, jnp.int32), 'delivered': jnp.zeros(5, jnp.int32),
            'dirty': jnp.zeros(5, jnp.int32), 'served': jnp.zeros(5, jnp.bool_),
            'lift_pos': jnp.array([[6, 2], [8, 22]], jnp.int32), 'lift_active': jnp.ones(2, jnp.bool_),
            'lift_visible': jnp.zeros((20, 2), jnp.bool_), 'lift_item': jnp.zeros(2, jnp.int32),
            'lift_recipe': jnp.zeros(2, jnp.int32), 'lift_age': jnp.zeros(2, jnp.int32),
            'barrier': jnp.zeros(2, jnp.bool_), 'secured': jnp.ones(2, jnp.bool_),
            'loader': jnp.full(2, -1, jnp.int32), 'target': jnp.array([0, 4], jnp.int32),
            'travel': jnp.zeros(2, jnp.int32), 'docked': jnp.zeros(2, jnp.bool_),
            'spoil': jnp.zeros(20, jnp.bool_), 'tile': tile.astype(jnp.int32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def context(self, s):
        floor = s['pos'][:, 1] // 5
        p = jnp.stack((s['pos'][:, 0], s['pos'][:, 1] % 5), axis=1)
        ld = self.distance(s['pos'], s['lift_pos'])
        lift = jnp.argmin(ld, axis=1)
        near = (ld[jnp.arange(20), lift] <= 1) & (s['travel'][lift] == 0)
        return floor, p, lift, near

    def power(self, s):
        circuit = jnp.arange(5) // 2
        load = jnp.sum((circuit[:, None] == circuit[None, :]) * s['freezer'][None, :], axis=1)
        return 3 - load

    def legal_actions(self, s):
        f, p, lift, near = self.context(s)
        def at(xy):
            return jnp.sum(jnp.abs(p - jnp.array(xy)), axis=1) <= 1
        moves = s['pos'][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        bounded = jnp.all((moves >= 0) & (moves <= jnp.array([14, 24])), axis=-1)
        wall = s['tile'][jnp.clip(moves[:, :, 1], 0, 24), jnp.clip(moves[:, :, 0], 0, 14)] == 1
        move = bounded & ~wall
        nearest_actor = (self.distance(s['pos'], s['pos']) <= 1) & ~jnp.eye(20, dtype=jnp.bool_)
        dispatch = near & s['secured'][lift] & ~s['barrier'][lift] & (self.power(s)[f] >= 2)
        oven = at([4, 3]) & (((s['item'] == 2) & (s['pot_recipe'][f] == 0)) | ((s['item'] == 0) & (s['pot_recipe'][f] > 0) & (s['pot_time'][f] == 0)))
        m = jnp.stack((jnp.ones(20, jnp.bool_), move[:, 0], move[:, 1], move[:, 2], move[:, 3],
                       at([1, 1]) & (s['item'] == 0) & (s['stock'][f] > 0), at([3, 1]) & (s['item'] == 1), oven,
                       near & (s['lift_item'][lift] == 0) & (s['item'] > 0), near & s['barrier'][lift] & (s['loader'][lift] != jnp.arange(20)),
                       dispatch & (f < 4), dispatch & (f > 0), near & (s['item'] == 0) & (s['lift_item'][lift] > 0),
                       at([5, 1]), at([11, 1]) & (s['item'] == 3) & (s['recipe'] == s['order'][f]),
                       at([11, 3]) & ((s['item'] == 4) | (s['dirty'][f] > 0)), jnp.any(nearest_actor, axis=1),
                       s['knowledge'] > 0, at([11, 1]) & ((s['item'] == 0) | (s['item'] == 5)), s['item'] > 0), axis=1)
        return m & ((jnp.arange(20)[None, :] == 0) | ~s['done'])

    def advance(self, key, s, actions):
        ids = jnp.arange(20, dtype=jnp.int32)
        safe = jnp.clip(actions, 0, 19)
        a = jnp.where((actions == safe) & self.legal_actions(s)[ids, safe], safe, 0)
        f, lp, lift, near = self.context(s)
        # Each station or lift accepts at most one actor, lowest ID first.
        def kitchen_winner(action):
            return jnp.min(jnp.where((f[:, None] == jnp.arange(5)[None, :]) & (a[:, None] == action), ids[:, None], 20), axis=0)
        def lift_winner(action):
            return jnp.min(jnp.where((lift[:, None] == jnp.arange(2)[None, :]) & (a[:, None] == action), ids[:, None], 20), axis=0)
        toggles = jnp.sum((f[:, None] == jnp.arange(5)[None, :]) & (a[:, None] == 13), axis=0) % 2
        freezer = s['freezer'] ^ (toggles == 1)
        ns = dict(s)
        ns['freezer'] = freezer
        power = self.power(ns)
        item, recipe, age = s['item'], s['recipe'], s['age']
        take = kitchen_winner(5)
        took = (a == 5) & (take[f] == ids)
        item, recipe, age = jnp.where(took, 1, item), jnp.where(took, s['order'][f], recipe), jnp.where(took, 0, age)
        stock = s['stock'] - (take < 20).astype(jnp.int32)
        item = jnp.where(a == 6, 2, item)
        cook = kitchen_winner(7)
        ci = jnp.minimum(cook, 19)
        deposit = (cook < 20) & (s['item'][ci] == 2)
        collect = (cook < 20) & (s['item'][ci] == 0)
        pot_recipe = jnp.where(deposit, recipe[ci], jnp.where(collect, 0, s['pot_recipe']))
        pot_time = jnp.where(deposit, 4, jnp.maximum(0, s['pot_time'] - jnp.minimum(power, 2)))
        pot_age = jnp.where(deposit, age[ci], jnp.where(collect, 0, s['pot_age'] + ((s['pot_recipe'] > 0) & ~freezer).astype(jnp.int32)))
        ownoven = (a == 7) & (cook[f] == ids)
        item = jnp.where(ownoven & deposit[f], 0, jnp.where(ownoven & collect[f], 3, item))
        recipe = jnp.where(ownoven & collect[f], s['pot_recipe'][f], recipe)
        age = jnp.where(ownoven & collect[f], s['pot_age'][f], age)
        # Loading opens a physical barrier. A different actor must secure the crate.
        load = lift_winner(8)
        li = jnp.minimum(load, 19)
        loaded = load < 20
        unload = lift_winner(12)
        ui = jnp.minimum(unload, 19)
        unloaded = unload < 20
        litem = jnp.where(loaded, item[li], jnp.where(unloaded, 0, s['lift_item']))
        lrecipe = jnp.where(loaded, recipe[li], s['lift_recipe'])
        lage = jnp.where(loaded, age[li], jnp.where(unloaded, 0, s['lift_age']))
        sender = (a == 8) & (load[lift] == ids)
        receiver = (a == 12) & (unload[lift] == ids)
        item = jnp.where(sender, 0, jnp.where(receiver, s['lift_item'][lift], item))
        recipe = jnp.where(receiver, s['lift_recipe'][lift], recipe)
        age = jnp.where(receiver, s['lift_age'][lift], age)
        secure = lift_winner(9) < 20
        barrier = jnp.where(loaded | unloaded, True, jnp.where(secure, False, s['barrier']))
        secured = jnp.where(loaded | unloaded, False, jnp.where(secure, True, s['secured']))
        loader = jnp.where(loaded, load, jnp.where(unloaded, unload, s['loader']))
        up, down = lift_winner(10), lift_winner(11)
        driver = jnp.minimum(up, down)
        lf = s['lift_pos'][:, 1] // 5
        dispatch = (driver < 20) & (power[lf] >= 2) & ~barrier
        target = jnp.where(dispatch, lf + jnp.where(up <= down, 1, -1), s['target'])
        travel = jnp.where(dispatch, 3, jnp.maximum(0, s['travel'] - 1))
        docked = (s['travel'] == 1) & ~dispatch
        lpos = s['lift_pos'].at[:, 1].set(jnp.where(docked, target * 5 + 2, s['lift_pos'][:, 1]))
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[jnp.minimum(a, 4)]
        proposed = s['pos'] + jnp.where(((a >= 1) & (a <= 4))[:, None], delta, 0)
        blocked = jnp.any((self.distance(proposed, s['lift_pos']) == 0) & barrier[None, :], axis=1)
        pos = jnp.where(blocked[:, None], s['pos'], proposed)
        # A reciprocal handoff transfers an item and its age, never duplicates it.
        adjacent = (self.distance(pos, pos) <= 1) & ~jnp.eye(20, dtype=jnp.bool_)
        counterpart = jnp.argmin(jnp.where(adjacent & (a[None, :] == 16), ids[None, :], 100), axis=1)
        pair = (a == 16) & adjacent[ids, counterpart] & (a[counterpart] == 16) & (counterpart[counterpart] == ids)
        hand = pair & ((item == 0) != (item[counterpart] == 0))
        olditem, oldrecipe, oldage = item, recipe, age
        item = jnp.where(hand, olditem[counterpart], olditem)
        recipe = jnp.where(hand, oldrecipe[counterpart], oldrecipe)
        age = jnp.where(hand, oldage[counterpart], oldage)
        serve = kitchen_winner(14)
        served = serve < 20
        delivered_actor = (a == 14) & (serve[f] == ids)
        quality = jnp.maximum(0, 32 - age[jnp.minimum(serve, 19)]).astype(jnp.float32) / 32.0
        item = jnp.where(delivered_actor, 4, item)
        clean = kitchen_winner(15)
        cleaned = clean < 20
        item = jnp.where((a == 15) & (clean[f] == ids) & (item == 4), 0, item)
        dirty = jnp.clip(s['dirty'] + served.astype(jnp.int32) - cleaned.astype(jnp.int32), 0, 20)
        order = jnp.where(served, 3 - s['order'], s['order'])
        message = jnp.where(a == 17, s['knowledge'], 0)
        heard = adjacent & (message[None, :] > 0)
        speaker = jnp.argmax(heard, axis=1)
        knowledge = jnp.where(jnp.any(heard, axis=1), message[speaker], s['knowledge'])
        knowledge = jnp.where(a == 18, s['order'][f], knowledge)
        item = jnp.where(a == 18, 5, item)
        recipe = jnp.where(a == 18, s['order'][f], recipe)
        item = jnp.where(a == 19, 0, item)
        age = jnp.minimum(32, age + (((item >= 1) & (item <= 3)) & ~freezer[pos[:, 1] // 5]).astype(jnp.int32))
        spoil = (age >= 32) & (item >= 1) & (item <= 3)
        item = jnp.where(spoil, 4, item)
        lage = jnp.minimum(32, lage + (((litem >= 1) & (litem <= 3)) & ~freezer[lpos[:, 1] // 5]).astype(jnp.int32))
        litem = jnp.where((lage >= 32) & (litem >= 1) & (litem <= 3), 4, litem)
        pot_rotten = (pot_age >= 32) & (pot_recipe > 0)
        pot_recipe, pot_time, pot_age = jnp.where(pot_rotten, 0, pot_recipe), jnp.where(pot_rotten, 0, pot_time), jnp.clip(pot_age, 0, 32)
        delivered = s['delivered'] + served.astype(jnp.int32)
        terminal = jnp.all(delivered >= 6)
        t = s['t'] + jnp.array(1, jnp.int32)
        reward = -0.005 * (a != 0) - 0.3 * spoil - 0.05 * (a == 19) + 0.2 * quality[s['team']] * served[s['team']] + 0.01 * jnp.sum(served) - 0.002 * freezer[s['team']]
        n = dict(s)
        n['t'], n['done'], n['terminated'] = t, terminal | (t >= 640), terminal
        n['pos'], n['item'], n['recipe'], n['age'] = pos, item, recipe, age
        n['knowledge'], n['message'], n['event'], n['recipient'], n['spoil'] = knowledge, message, a, jnp.where(hand, counterpart, -1), spoil
        n['freezer'], n['pot_recipe'], n['pot_time'], n['pot_age'] = freezer, pot_recipe, pot_time, pot_age
        n['order'], n['stock'], n['delivered'], n['dirty'], n['served'] = order, stock, delivered, dirty, served
        n['lift_pos'], n['lift_item'], n['lift_recipe'], n['lift_age'] = lpos, litem, lrecipe, lage
        n['barrier'], n['secured'], n['loader'], n['target'], n['travel'], n['docked'] = barrier, secured, loader, target, travel, docked
        return self.refresh(n), reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(20, jnp.float32)), lambda _: self.advance(key, state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': (n['t'] >= 640) & ~n['terminated']}
        return self.observe(n), n, reward, jnp.full(20, n['done'], jnp.bool_), info

    def encode(self, s, posname, actname, count, radius, values):
        visible, ix = self.sight(s['pos'], s[posname], s[actname], count, radius)
        mask = jnp.take_along_axis(visible, ix, axis=1)
        data = jnp.concatenate((jnp.ones((s[posname].shape[0], 1)), jnp.arange(s[posname].shape[0])[:, None], s[posname], values), axis=1)
        return jnp.where(mask[:, :, None], data[ix], -1).reshape((20, -1))

    def observe(self, s):
        own = jnp.stack((jnp.arange(20), s['team'], s['pos'][:, 0], s['pos'][:, 1], s['item'], s['recipe'], s['age'], s['knowledge'], s['message'], s['event'], s['recipient'], s['spoil']), axis=1)
        actors = self.encode(s, 'pos', 'active', 8, 3, jnp.stack((s['team'], s['item'], s['recipe'], s['age'], s['message'], s['event'], s['recipient'], s['spoil']), axis=1))
        kitchens = self.encode(s, 'kitchen_pos', 'kitchen_active', 3, 6, jnp.stack((s['freezer'], s['pot_recipe'], s['pot_time'], s['pot_age'], s['order'], s['stock'], s['delivered'], s['dirty']), axis=1))
        lifts = self.encode(s, 'lift_pos', 'lift_active', 2, 4, jnp.stack((s['lift_item'], s['lift_recipe'], s['lift_age'], s['barrier'], s['secured'], s['loader'], s['target'], s['travel']), axis=1))
        offsets = jnp.stack((jnp.arange(25) % 5 - 2, jnp.arange(25) // 5 - 2), axis=1)
        q = s['pos'][:, None, :] + offsets[None, :, :]
        inside = jnp.all((q >= 0) & (q <= jnp.array([14, 24])), axis=-1)
        patch = jnp.where(inside, s['tile'][jnp.clip(q[:, :, 1], 0, 24), jnp.clip(q[:, :, 0], 0, 14)], -1)
        public = jnp.broadcast_to(jnp.stack((s['t'], s['done'], s['terminated'])), (20, 3))
        return jnp.concatenate((own, actors, kitchens, lifts, patch, public), axis=1).astype(jnp.float32)
