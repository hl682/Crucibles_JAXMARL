import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 32
    observation_size = 544
    num_actions = 20
    max_steps = 640

    def distance(self, x, y):
        return jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)

    def refresh(self, s):
        s = dict(s)
        d = self.distance(s['pos'], s['pos'])
        ix = jnp.argsort(d + jnp.arange(32, dtype=jnp.int32)[None, :] / 128.0, axis=1)[:, :4]
        s['seen'] = ix.astype(jnp.int32)
        visible = jnp.take_along_axis(d, ix, axis=1) <= 5
        s['agent_vis'] = jnp.zeros((32, 32), jnp.bool_).at[jnp.arange(32, dtype=jnp.int32)[:, None], ix].set(visible)
        s['agent_vis'] = s['agent_vis'].at[jnp.arange(32, dtype=jnp.int32), jnp.arange(32, dtype=jnp.int32)].set(True)
        for group in ('station', 'pantry', 'vessel', 'cold', 'cart'):
            s[group + '_vis'] = self.distance(s['pos'], s[group + '_pos']) <= 5
        s['order_vis'] = jnp.arange(32, dtype=jnp.int32)[:, None] // 4 == jnp.arange(8, dtype=jnp.int32)[None, :]
        return s

    def packed(self, pos, data, vis):
        xy = pos.astype(jnp.float32) / jnp.array([31.0, 23.0], jnp.float32)
        fields = jnp.concatenate((xy, data), axis=1)
        return jnp.concatenate((vis[:, :, None], jnp.where(vis[:, :, None], fields[None, :, :], 0.0)), axis=2).reshape((32, -1))

    def observe(self, s):
        crew = jnp.arange(32, dtype=jnp.int32) // 4
        own = jnp.stack((jnp.arange(32, dtype=jnp.int32) / 31.0, s['pos'][:, 0] / 31.0, s['pos'][:, 1] / 23.0, crew / 7.0, (s['carry'] + 1) / 16.0, s['starter'].astype(jnp.float32), s['dirty_tool'].astype(jnp.float32), s['message'] / 4.0, s['message_recipe'].astype(jnp.float32), s['message_culture'].astype(jnp.float32), jnp.full((32,), s['t'] / 640.0)), axis=1)
        data = jnp.stack((jnp.arange(32, dtype=jnp.int32) / 31.0, s['pos'][:, 0] / 31.0, s['pos'][:, 1] / 23.0, crew / 7.0, (s['carry'] + 1) / 16.0, s['starter'].astype(jnp.float32), s['dirty_tool'].astype(jnp.float32), s['message'] / 4.0, s['message_recipe'].astype(jnp.float32), s['message_culture'].astype(jnp.float32)), axis=1)
        vis = jnp.take_along_axis(s['agent_vis'], s['seen'], axis=1)
        people = jnp.concatenate((vis[:, :, None], jnp.where(vis[:, :, None], data[s['seen']], 0.0)), axis=2).reshape((32, 44))
        station = self.packed(s['station_pos'], jnp.stack((s['station_kind'] / 3.0, s['station_dirty']), axis=1), s['station_vis'])
        pantry = self.packed(s['pantry_pos'], jnp.stack((s['pantry_stock'] / 24.0, s['pantry_recipe']), axis=1), s['pantry_vis'])
        vessel = self.packed(s['vessel_pos'], jnp.stack((s['vessel_stage'] / 7.0, (s['vessel_recipe'] + 1) / 2.0, s['vessel_culture'], s['vessel_progress'] / 10.0, s['vessel_age'] / 25.0, s['vessel_bad'], s['vessel_heavy'], (s['vessel_slot'] + 1) / 8.0), axis=1), s['vessel_vis'])
        cold = self.packed(s['cold_pos'], s['cold_used'][:, None] / 2.0, s['cold_vis'])
        cart = self.packed(s['cart_pos'], s['cart_direction'][:, None].astype(jnp.float32), s['cart_vis'])
        order = self.packed(s['order_pos'], jnp.stack((s['order_recipe'], s['order_culture'], s['order_count'] / 3.0, jnp.clip(s['order_deadline'] - s['t'], -640, 640) / 640.0), axis=1), s['order_vis'])
        off = jnp.stack((jnp.tile(jnp.arange(-2, 3, dtype=jnp.int32), 5), jnp.repeat(jnp.arange(-2, 3, dtype=jnp.int32), 5)), axis=1)
        xy = jnp.clip(s['pos'][:, None, :] + off[None, :, :], jnp.array([0, 0], dtype=jnp.int32), jnp.array([31, 23], dtype=jnp.int32))
        patch = s['tiles'][xy[:, :, 1], xy[:, :, 0]].astype(jnp.float32)
        return jnp.concatenate((own, people, station, pantry, vessel, cold, cart, order, patch), axis=1).astype(jnp.float32)

    def reset(self, key):
        crew = jnp.arange(8, dtype=jnp.int32)
        center = jnp.stack((4 + 8 * (crew % 4), 4 + 12 * (crew // 4)), axis=1)
        ids = jnp.arange(32, dtype=jnp.int32)
        pos = center[ids // 4] + jnp.stack((ids % 2, ids % 4 // 2), axis=1)
        station_pos = jnp.repeat(center, 4, axis=0) + jnp.tile(jnp.array([[0, -2], [2, -2], [2, 2], [0, 3]], jnp.int32), (8, 1))
        xx = jnp.arange(32, dtype=jnp.int32)[None, :]
        yy = jnp.arange(24, dtype=jnp.int32)[:, None]
        tiles = ((xx % 8 == 7) & (yy % 12 >= 2) & (yy % 12 <= 8) & (yy % 12 != 5)).astype(jnp.int32)
        cold = jnp.array([[8, 6], [24, 6], [8, 18], [24, 18]], jnp.int32)
        s = {'t': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False), 'tiles': tiles, 'pos': pos, 'active': jnp.ones((32,), jnp.bool_), 'carry': jnp.full((32,), -1, jnp.int32), 'starter': ids % 2, 'dirty_tool': jnp.zeros((32,), jnp.bool_), 'message': jnp.zeros((32,), jnp.int32), 'message_recipe': jnp.zeros((32,), jnp.int32), 'message_culture': jnp.zeros((32,), jnp.int32), 'seen': jnp.zeros((32, 4), jnp.int32), 'agent_vis': jnp.zeros((32, 32), jnp.bool_), 'station_pos': station_pos, 'station_kind': jnp.arange(32, dtype=jnp.int32) % 4, 'station_dirty': jnp.arange(32, dtype=jnp.int32) % 4 == 1, 'station_active': jnp.ones((32,), jnp.bool_), 'station_vis': jnp.zeros((32, 32), jnp.bool_), 'pantry_pos': center + jnp.array([-2, 0], jnp.int32), 'pantry_stock': jnp.full((8,), 24, jnp.int32), 'pantry_recipe': crew % 2, 'pantry_active': jnp.ones((8,), jnp.bool_), 'pantry_vis': jnp.zeros((32, 8), jnp.bool_), 'vessel_pos': jnp.repeat(center, 2, axis=0) + jnp.tile(jnp.array([[0, 0], [1, 1]], jnp.int32), (8, 1)), 'vessel_stage': jnp.zeros((16,), jnp.int32), 'vessel_recipe': jnp.full((16,), -1, jnp.int32), 'vessel_culture': jnp.zeros((16,), jnp.int32), 'vessel_progress': jnp.zeros((16,), jnp.int32), 'vessel_age': jnp.zeros((16,), jnp.int32), 'vessel_bad': jnp.zeros((16,), jnp.bool_), 'vessel_heavy': jnp.arange(16, dtype=jnp.int32) % 2 == 1, 'vessel_slot': jnp.full((16,), -1, jnp.int32), 'vessel_active': jnp.ones((16,), jnp.bool_), 'vessel_vis': jnp.zeros((32, 16), jnp.bool_), 'cold_pos': cold, 'cold_used': jnp.zeros((4,), jnp.int32), 'cold_active': jnp.ones((4,), jnp.bool_), 'cold_vis': jnp.zeros((32, 4), jnp.bool_), 'cart_pos': jnp.array([[2, 11], [18, 11], [6, 23], [22, 23]], jnp.int32), 'cart_direction': jnp.array([1, -1, 1, -1], jnp.int32), 'cart_active': jnp.ones((4,), jnp.bool_), 'cart_vis': jnp.zeros((32, 4), jnp.bool_), 'order_pos': center + jnp.array([0, 3], jnp.int32), 'order_recipe': jax.random.randint(key, (8,), 0, 2).astype(jnp.int32), 'order_culture': crew % 2, 'order_count': jnp.zeros((8,), jnp.int32), 'order_deadline': jnp.full((8,), 72, jnp.int32), 'order_active': jnp.ones((8,), jnp.bool_), 'order_vis': jnp.zeros((32, 8), jnp.bool_), 'event_serve': jnp.zeros((32,), jnp.bool_), 'event_contaminate': jnp.zeros((32,), jnp.bool_), 'event_waste': jnp.zeros((32,), jnp.bool_), 'event_ripen': jnp.zeros((16,), jnp.bool_)}
        s = self.refresh(s)
        return (self.observe(s), s)

    def legal_actions(self, s):
        has = s['carry'] >= 0
        mask = jnp.ones((32, 20), jnp.bool_)
        for action in (5, 8, 9, 10, 11, 12, 13, 16, 18, 19):
            mask = mask.at[:, action].set(has)
        mask = mask.at[:, 7].set(~has)
        return mask & (~s['done'] | (jnp.arange(20, dtype=jnp.int32)[None, :] == 0))

    def raw_step(self, key, state, actions):
        s = state
        clipped = jnp.clip(actions, 0, 19)
        a = jnp.where((actions >= 0) & (actions < 20) & self.legal_actions(s)[jnp.arange(32, dtype=jnp.int32), clipped], clipped, 0)
        n = dict(s)
        n['t'] = s['t'] + 1
        n['message'] = jnp.maximum(s['message'] - 1, 0)
        n['message_recipe'] = jnp.where(n['message'] > 0, s['message_recipe'], 0)
        n['message_culture'] = jnp.where(n['message'] > 0, s['message_culture'], 0)
        for name in ('event_serve', 'event_contaminate', 'event_waste'):
            n[name] = jnp.zeros((32,), jnp.bool_)
        progress = jnp.minimum(s['vessel_progress'] + (s['vessel_stage'] == 4).astype(jnp.int32), 10)
        target = 6 + 4 * s['vessel_culture']
        ripe = (s['vessel_stage'] == 4) & (progress >= target)
        n['vessel_progress'] = progress
        n['event_ripen'] = ripe
        aging = ((s['vessel_stage'] == 5) | (s['vessel_stage'] == 6)) & ((s['vessel_slot'] < 0) | (n['t'] % 3 == 0))
        n['vessel_age'] = jnp.minimum(s['vessel_age'] + aging.astype(jnp.int32), 25)
        n['vessel_stage'] = jnp.where(n['vessel_age'] >= 25, 7, jnp.where(ripe, 5, s['vessel_stage']))
        direction = jnp.where((s['cart_pos'][:, 0] == 0) | (s['cart_pos'][:, 0] == 31), -s['cart_direction'], s['cart_direction'])
        cart_target = s['cart_pos'].at[:, 0].add(direction)
        obstructed = jnp.any(self.distance(cart_target, s['pos']) == 0, axis=1)
        n['cart_pos'] = jnp.where(obstructed[:, None], s['cart_pos'], cart_target)
        n['cart_direction'] = direction
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        motion = delta[jnp.minimum(a, 4)] * ((a >= 1) & (a <= 4))[:, None]
        to = jnp.clip(s['pos'] + motion, jnp.array([0, 0], dtype=jnp.int32), jnp.array([31, 23], dtype=jnp.int32))
        helper = jnp.any((self.distance(s['pos'], s['pos']) <= 1) & (a[None, :] == 6) & (jnp.arange(32, dtype=jnp.int32)[:, None] != jnp.arange(32, dtype=jnp.int32)[None, :]), axis=1)
        heavy = (s['carry'] >= 0) & s['vessel_heavy'][jnp.maximum(s['carry'], 0)]
        open_tile = (s['tiles'][to[:, 1], to[:, 0]] == 0) & ~jnp.any(self.distance(to, n['cart_pos']) == 0, axis=1)
        n['pos'] = jnp.where((open_tile & (~heavy | helper))[:, None], to, s['pos'])
        reward = -0.01 * (a >= 5).astype(jnp.float32)

        def work(i, pair):
            z, rew = pair
            z = dict(z)
            here = z['pos'][i]
            pot = jnp.maximum(z['carry'][i], 0)
            carried = jnp.any(z['carry'][:, None] == jnp.arange(16, dtype=jnp.int32)[None, :], axis=0)
            vd = jnp.sum(jnp.abs(z['vessel_pos'] - here), axis=1)
            choice = jnp.argmin(jnp.where(~carried, vd, 1000))
            pick = (a[i] == 7) & (vd[choice] <= 1) & ~carried[choice] & (~z['vessel_heavy'][choice] | helper[i])
            z['carry'] = z['carry'].at[i].set(jnp.where(pick, choice, z['carry'][i]))
            z['vessel_slot'] = z['vessel_slot'].at[choice].set(jnp.where(pick, -1, z['vessel_slot'][choice]))
            drop = a[i] == 8
            z['vessel_pos'] = z['vessel_pos'].at[pot].set(jnp.where(drop, here, z['vessel_pos'][pot]))
            z['carry'] = z['carry'].at[i].set(jnp.where(drop, -1, z['carry'][i]))
            pd = jnp.sum(jnp.abs(z['pantry_pos'] - here), axis=1)
            pantry = jnp.argmin(pd)
            fill = (a[i] == 5) & (pd[pantry] <= 1) & (z['pantry_stock'][pantry] > 0) & (z['vessel_stage'][pot] == 0)
            z['pantry_stock'] = z['pantry_stock'].at[pantry].add(-fill.astype(jnp.int32))
            z['vessel_recipe'] = z['vessel_recipe'].at[pot].set(jnp.where(fill, z['pantry_recipe'][pantry], z['vessel_recipe'][pot]))
            z['vessel_stage'] = z['vessel_stage'].at[pot].set(jnp.where(fill, 1, z['vessel_stage'][pot]))
            sd = jnp.sum(jnp.abs(z['station_pos'] - here), axis=1)
            desired = jnp.where(a[i] == 9, 0, jnp.where(a[i] == 10, 1, jnp.where(a[i] == 12, 2, 3)))
            st = jnp.argmin(jnp.where(z['station_kind'] == desired, sd, 1000))
            near = sd[st] <= 1
            wash = (a[i] == 9) & near & (z['vessel_stage'][pot] == 1)
            grind = (a[i] == 10) & near & (z['vessel_stage'][pot] == 2) & (z['vessel_recipe'][pot] == 0)
            finish = (a[i] == 12) & near & (z['vessel_stage'][pot] == 5)
            process = wash | grind | finish
            dirty = z['dirty_tool'][i] | z['station_dirty'][st]
            contaminate = process & dirty & ~z['vessel_bad'][pot]
            food_bad = jnp.where(wash, dirty, z['vessel_bad'][pot] | process & dirty)
            z['vessel_bad'] = z['vessel_bad'].at[pot].set(food_bad)
            z['station_dirty'] = z['station_dirty'].at[st].set(z['station_dirty'][st] | process & z['dirty_tool'][i])
            z['dirty_tool'] = z['dirty_tool'].at[i].set(z['dirty_tool'][i] | process & food_bad)
            z['vessel_stage'] = z['vessel_stage'].at[pot].set(jnp.where(wash, 2, jnp.where(grind, 3, jnp.where(finish, 6, z['vessel_stage'][pot]))))
            ready = (z['vessel_recipe'][pot] == 0) & (z['vessel_stage'][pot] == 3) | (z['vessel_recipe'][pot] == 1) & (z['vessel_stage'][pot] == 2)
            inoculate = (a[i] == 11) & ready
            z['vessel_culture'] = z['vessel_culture'].at[pot].set(jnp.where(inoculate, z['starter'][i], z['vessel_culture'][pot]))
            z['vessel_stage'] = z['vessel_stage'].at[pot].set(jnp.where(inoculate, 4, z['vessel_stage'][pot]))
            z['vessel_progress'] = z['vessel_progress'].at[pot].set(jnp.where(inoculate, 0, z['vessel_progress'][pot]))
            clean_st = jnp.argmin(sd)
            clean = (a[i] == 14) & (sd[clean_st] <= 1)
            z['station_dirty'] = z['station_dirty'].at[clean_st].set(z['station_dirty'][clean_st] & ~clean)
            z['dirty_tool'] = z['dirty_tool'].at[i].set(z['dirty_tool'][i] & ~clean)
            nd = jnp.sum(jnp.abs(z['pos'] - here), axis=1)
            friend = jnp.argmin(jnp.where(jnp.arange(32, dtype=jnp.int32) != i, nd, 1000))
            lend = (a[i] == 15) & (nd[friend] <= 1)
            z['starter'] = z['starter'].at[friend].set(jnp.where(lend, z['starter'][i], z['starter'][friend]))
            counter = jnp.min(jnp.where(z['station_kind'] == 3, sd, 1000)) <= 1
            handoff = (a[i] == 19) & counter & (nd[friend] <= 1) & (z['carry'][friend] < 0)
            z['carry'] = z['carry'].at[friend].set(jnp.where(handoff, pot, z['carry'][friend])).at[i].set(jnp.where(handoff, -1, z['carry'][i]))
            cd = jnp.sum(jnp.abs(z['cold_pos'] - here), axis=1)
            fridge = jnp.argmin(cd)
            slots = jnp.array([2, 2], jnp.int32) * fridge + jnp.arange(2, dtype=jnp.int32)
            free = ~jnp.any(z['vessel_slot'][:, None] == slots[None, :], axis=0)
            slot = slots[jnp.argmax(free)]
            store = (a[i] == 16) & (cd[fridge] <= 1) & jnp.any(free)
            coldxy = z['cold_pos'][fridge] + jnp.array([-1, 0], jnp.int32) + jnp.array([2, 0], jnp.int32) * (slot % 2)
            z['vessel_slot'] = z['vessel_slot'].at[pot].set(jnp.where(store, slot, z['vessel_slot'][pot]))
            z['vessel_pos'] = z['vessel_pos'].at[pot].set(jnp.where(store, coldxy, z['vessel_pos'][pot]))
            z['carry'] = z['carry'].at[i].set(jnp.where(store, -1, z['carry'][i]))
            crew = i // 4
            z['message'] = z['message'].at[i].set(jnp.where(a[i] == 17, 4, z['message'][i]))
            z['message_recipe'] = z['message_recipe'].at[i].set(jnp.where(a[i] == 17, z['order_recipe'][crew], z['message_recipe'][i]))
            z['message_culture'] = z['message_culture'].at[i].set(jnp.where(a[i] == 17, z['order_culture'][crew], z['message_culture'][i]))
            customer = st // 4
            match = (z['vessel_recipe'][pot] == z['order_recipe'][customer]) & (z['vessel_culture'][pot] == z['order_culture'][customer])
            serve = (a[i] == 13) & near & (z['vessel_stage'][pot] == 6) & match & (z['order_count'][customer] < 3)
            quality = jnp.where(z['vessel_bad'][pot], 0.25, 2.0) + 0.5 * (z['t'] <= z['order_deadline'][customer])
            waste = (a[i] == 18) & (z['vessel_stage'][pot] > 0)
            empty = serve | waste
            z['order_count'] = z['order_count'].at[customer].add(serve.astype(jnp.int32))
            z['order_recipe'] = z['order_recipe'].at[customer].set(jnp.where(serve, 1 - z['order_recipe'][customer], z['order_recipe'][customer]))
            z['order_culture'] = z['order_culture'].at[customer].set(jnp.where(serve, 1 - z['order_culture'][customer], z['order_culture'][customer]))
            z['order_deadline'] = z['order_deadline'].at[customer].set(jnp.where(serve, z['t'] + 72, z['order_deadline'][customer]))
            z['vessel_stage'] = z['vessel_stage'].at[pot].set(jnp.where(empty, 0, z['vessel_stage'][pot]))
            z['vessel_recipe'] = z['vessel_recipe'].at[pot].set(jnp.where(empty, -1, z['vessel_recipe'][pot]))
            z['vessel_age'] = z['vessel_age'].at[pot].set(jnp.where(empty, 0, z['vessel_age'][pot]))
            z['vessel_progress'] = z['vessel_progress'].at[pot].set(jnp.where(empty, 0, z['vessel_progress'][pot]))
            z['vessel_bad'] = z['vessel_bad'].at[pot].set(z['vessel_bad'][pot] & ~empty)
            z['event_serve'] = z['event_serve'].at[i].set(serve)
            z['event_contaminate'] = z['event_contaminate'].at[i].set(contaminate)
            z['event_waste'] = z['event_waste'].at[i].set(waste)
            rew = rew.at[i].add(quality * serve - 0.4 * waste - 0.15 * contaminate)
            rew = rew + (jnp.arange(32, dtype=jnp.int32) // 4 == customer).astype(jnp.float32) * 0.25 * serve
            return (z, rew)
        n, reward = jax.lax.fori_loop(0, 32, work, (n, reward))
        carriers = n['carry'][:, None] == jnp.arange(16, dtype=jnp.int32)[None, :]
        owner = jnp.argmax(carriers, axis=0)
        n['vessel_pos'] = jnp.where(jnp.any(carriers, axis=0)[:, None], n['pos'][owner], n['vessel_pos'])
        n['cold_used'] = jnp.sum((n['vessel_slot'][:, None] >= 0) & (n['vessel_slot'][:, None] // 2 == jnp.arange(4, dtype=jnp.int32)[None, :]), axis=0).astype(jnp.int32)
        n['order_active'] = n['order_count'] < 3
        n['terminated'] = jnp.all(n['order_count'] >= 3)
        n['done'] = n['terminated'] | (n['t'] >= 640)
        n = self.refresh(n)
        n = jax.tree.map(lambda new, old: jnp.where(s['done'], old, new), n, s)
        reward = jnp.where(s['done'], jnp.zeros_like(reward), reward).astype(jnp.float32)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': (n['t'] >= 640) & ~n['terminated']}
        return (self.observe(n), n, reward, jnp.full((32,), n['done'], jnp.bool_), info)
