"""Filter Couriers: original fictional-dust logistics; authored, never executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 251
    num_actions = 11
    max_steps = 96

    def carried(self, s):
        holds = s['owner'][None, :] == jnp.arange(3, dtype=jnp.int32)[:, None]
        return jnp.where(jnp.any(holds, axis=1), jnp.argmax(holds, axis=1), 3).astype(jnp.int32)

    def refresh(self, s):
        owner = jnp.clip(s['owner'], 0, 2)
        load_pos = jnp.where((s['owner'] >= 0)[:, None], s['pos'][owner], s['load_pos'])
        av = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2) <= 2
        lv = (jnp.sum(jnp.abs(s['pos'][:, None, :] - load_pos[None, :, :]), axis=2) <= 2) & ~s['delivered'][None, :]
        dv = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['tile_pos'][None, :, :]), axis=2) <= 2
        return dict(s, load_pos=load_pos, load_active=~s['delivered'], agent_visible=av,
                    load_visible=lv, dust_visible=dv, dust_amount=s['dust'].reshape(-1),
                    dust_active=s['dust'].reshape(-1) > 0)

    def reset(self, key):
        x = jnp.arange(9, dtype=jnp.int32)[None, :]
        y = jnp.arange(7, dtype=jnp.int32)[:, None]
        walls = ((x == 4) & ((y == 0) | (y == 2) | (y == 4) | (y == 6))).astype(jnp.int32)
        cells = jnp.stack([jnp.tile(jnp.arange(9, dtype=jnp.int32), 7),
                           jnp.repeat(jnp.arange(7, dtype=jnp.int32), 9)], axis=1)
        dust = jnp.zeros((7, 9), jnp.int32).at[3, 3].set(2).at[3, 4].set(3).at[3, 5].set(2)
        pos = jnp.array([[3, 3], [5, 3], [2, 1]], jnp.int32)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'map': walls, 'pos': pos, 'agent_active': jnp.ones((3,), jnp.bool_),
            'agent_visible': jnp.eye(3, dtype=jnp.bool_),
            'private_visible': jnp.eye(3, dtype=jnp.bool_),
            'stamina': jnp.full((3,), 20, jnp.int32),
            'filters': jnp.array([2, 0, 2], jnp.int32),
            'burden': jnp.zeros((3,), jnp.int32),
            'priority': jax.random.randint(key, (3,), 1, 4, dtype=jnp.int32),
            'load_pos': pos, 'owner': jnp.arange(3, dtype=jnp.int32),
            'destination': jnp.array([1, 0, 1], jnp.int32),
            'quality': jnp.full((3,), 6, jnp.int32),
            'load_dust': jnp.full((3,), 4, jnp.int32),
            'delivered': jnp.zeros((3,), jnp.bool_),
            'load_active': jnp.ones((3,), jnp.bool_),
            'load_visible': jnp.eye(3, dtype=jnp.bool_),
            'station_pos': jnp.array([[1, 1], [7, 5]], jnp.int32),
            'station_stock': jnp.array([2, 2], jnp.int32),
            'station_active': jnp.ones((2,), jnp.bool_),
            'station_visible': jnp.ones((3, 2), jnp.bool_),
            'tile_pos': cells, 'dust': dust, 'dust_amount': dust.reshape(-1),
            'dust_active': dust.reshape(-1) > 0,
            'dust_visible': jnp.zeros((3, 63), jnp.bool_),
            'cleaned': jnp.zeros((3,), jnp.bool_),
            'exposed': jnp.zeros((3,), jnp.bool_),
            'delivered_now': jnp.zeros((3,), jnp.bool_),
            'handoff': jnp.zeros((3,), jnp.bool_),
            'filter_used': jnp.zeros((3,), jnp.int32),
            'spilled': jnp.zeros((3,), jnp.int32),
            'vented': jnp.array(0, jnp.int32),
            'filters_consumed': jnp.array(0, jnp.int32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        directions = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
        target = s['pos'][:, None, :] + directions[None, :, :]
        clipped = jnp.clip(target, jnp.array([0, 0]), jnp.array([8, 6]))
        wall = s['map'][clipped[:, :, 1], clipped[:, :, 0]] != 0
        cost = 1 + (s['dust'][clipped[:, :, 1], clipped[:, :, 0]] >= 2).astype(jnp.int32)
        occupied = jnp.any(jnp.all(target[:, :, None, :] == s['pos'][None, None, :, :], axis=3), axis=2)
        move = jnp.all(target == clipped, axis=2) & ~wall & ~occupied & (s['stamina'][:, None] >= cost)
        load = self.carried(s)
        has = load < 3
        index = jnp.minimum(load, 2)
        at_destination = jnp.all(s['pos'] == s['station_pos'][s['destination'][index]], axis=1)
        pickup = jnp.any((s['owner'][None, :] == -1) &
                          jnp.all(s['pos'][:, None, :] == s['load_pos'][None, :, :], axis=2), axis=1)
        adjacent = (jnp.sum(jnp.abs(s['pos'][:, None, :] - s['pos'][None, :, :]), axis=2) == 1)
        can_give = jnp.any(adjacent & (load[None, :] == 3), axis=1)
        station = jnp.all(s['pos'][:, None, :] == s['station_pos'][None, :, :], axis=2)
        refill = jnp.any(station & (s['station_stock'][None, :] > 0), axis=1) & (s['filters'] < 4)
        spill = (s['burden'] > 0) | (has & (s['load_dust'][index] > 0))
        mask = jnp.concatenate([jnp.ones((3, 1), jnp.bool_), move,
                                (s['filters'] > 0)[:, None], (has & at_destination)[:, None],
                                (has | pickup)[:, None], (has & can_give)[:, None],
                                spill[:, None], refill[:, None]], axis=1)
        return mask & ((~s['done']) | (jnp.arange(11)[None, :] == 0))

    def advance(self, s, actions):
        safe = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(s)[jnp.arange(3), safe]
        a = jnp.where(valid, safe, 0)
        directions = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        target = s['pos'] + directions[a]
        same = jnp.all(target[:, None, :] == target[None, :, :], axis=2)
        loses = jnp.any(same & (jnp.arange(3)[None, :] < jnp.arange(3)[:, None]), axis=1)
        pos = jnp.where(loses[:, None], s['pos'], target)
        moved = jnp.any(pos != s['pos'], axis=1)
        cost = 1 + (s['dust'][pos[:, 1], pos[:, 0]] >= 2).astype(jnp.int32)
        stamina = jnp.minimum(20, s['stamina'] - moved.astype(jnp.int32) * cost + (a == 0).astype(jnp.int32))
        carried = self.carried(s)
        has = carried < 3
        c = jnp.minimum(carried, 2)
        # Finite load stock and carried burden leave real dust at the departure cell.
        from_load = jnp.minimum(s['load_dust'][c], jnp.where(a == 9, 2, moved.astype(jnp.int32))) * has.astype(jnp.int32)
        from_body = jnp.where(a == 9, s['burden'], jnp.minimum(s['burden'], moved.astype(jnp.int32)))
        spilled = from_load + from_body
        load_dust = s['load_dust'] - jnp.sum(jax.nn.one_hot(c, 3, dtype=jnp.int32) * from_load[:, None], axis=0)
        burden = s['burden'] - from_body
        deposited = s['dust'].at[s['pos'][:, 1], s['pos'][:, 0]].add(spilled)
        vented = jnp.sum(jnp.maximum(deposited - 3, 0))
        dust = jnp.minimum(deposited, 3)
        # Filters sweep the current cell plus four cardinal neighbors after deposition.
        clean = a == 5
        reach = jnp.sum(jnp.abs(pos[:, None, :] - s['tile_pos'][None, :, :]), axis=2) <= 1
        sweep = 3 * jnp.sum(clean[:, None] & reach, axis=0).astype(jnp.int32)
        dust = jnp.maximum(dust.reshape(-1) - sweep, 0).reshape(7, 9)
        filters = s['filters'] - clean.astype(jnp.int32)
        # Unique occupancy makes contact extraction simultaneous and unambiguous.
        contact = dust[pos[:, 1], pos[:, 0]] > 0
        filtered = contact & (filters > 0)
        exposed = contact & ~filtered
        take = contact & (filtered | (burden < 3))
        dust = dust.at[pos[:, 1], pos[:, 0]].add(-take.astype(jnp.int32))
        filters = filters - filtered.astype(jnp.int32)
        burden = burden + (take & ~filtered).astype(jnp.int32)
        damage = jnp.sum(jax.nn.one_hot(c, 3, dtype=jnp.int32) * (has & exposed)[:, None], axis=0)
        quality = jnp.maximum(s['quality'] - damage, 0)
        owner = s['owner']
        load_pos = jnp.where((owner >= 0)[:, None], pos[jnp.clip(owner, 0, 2)], s['load_pos'])
        # Delivery precedes handoff/drop/pickup; one action cannot do two of them.
        delivering = has & (a == 6)
        delivered_now = jnp.any(jax.nn.one_hot(c, 3, dtype=jnp.bool_) & delivering[:, None], axis=0)
        owner = jnp.where(delivered_now, -2, owner)
        # A waiting empty neighbor accepts; lowest load id wins simultaneous offers.
        adjacent = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=2) == 1
        accept = adjacent & (carried[None, :] == 3) & (a[None, :] == 0)
        recipient = jnp.argmin(jnp.where(accept, jnp.arange(3)[None, :], 9), axis=1)
        offer = has & (a == 8) & jnp.any(accept, axis=1)
        smaller = c[None, :] < c[:, None]
        duplicate = jnp.any((recipient[:, None] == recipient[None, :]) & offer[None, :] & smaller, axis=1)
        handoff = offer & ~duplicate
        for i in range(3):
            owner = owner.at[c[i]].set(jnp.where(handoff[i], recipient[i], owner[c[i]]))
        drop = has & (a == 7)
        for i in range(3):
            owner = owner.at[c[i]].set(jnp.where(drop[i], -1, owner[c[i]]))
        ground = (owner[None, :] == -1) & jnp.all(pos[:, None, :] == load_pos[None, :, :], axis=2)
        pick = (~has) & (a == 7) & jnp.any(ground, axis=1)
        selected = jnp.argmax(ground, axis=1)
        for i in range(3):
            owner = owner.at[selected[i]].set(jnp.where(pick[i], i, owner[selected[i]]))
        at_station = jnp.all(pos[:, None, :] == s['station_pos'][None, :, :], axis=2)
        refill = (a == 10)[:, None] & at_station
        amount = refill.astype(jnp.int32)
        stock = s['station_stock'] - jnp.sum(amount, axis=0)
        filters = filters + jnp.sum(amount, axis=1)
        delivered = s['delivered'] | delivered_now
        t = s['t'] + 1
        terminated = jnp.all(delivered)
        truncated = (t >= self.max_steps) & ~terminated
        reward = s['priority'].astype(jnp.float32) * (quality / 6.0) * delivered_now
        reward = reward + 0.5 * delivering + 0.1 * jnp.sum(delivered_now) - 0.01 * moved - 0.03 * exposed
        n = dict(s, t=t, done=terminated | truncated, terminated=terminated, truncated=truncated,
                 pos=pos, stamina=stamina, filters=filters, burden=burden, owner=owner,
                 load_pos=load_pos, quality=quality, load_dust=load_dust, delivered=delivered,
                 station_stock=stock, dust=dust, cleaned=clean, exposed=exposed,
                 delivered_now=delivered_now, handoff=handoff, spilled=spilled,
                 filter_used=clean.astype(jnp.int32) + filtered.astype(jnp.int32),
                 vented=s['vented'] + vented,
                 filters_consumed=s['filters_consumed'] + jnp.sum(clean) + jnp.sum(filtered))
        return self.refresh(n), reward.astype(jnp.float32)

    def observe(self, s):
        own = jnp.concatenate([s['pos'] / jnp.array([8.0, 6.0]), s['stamina'][:, None] / 20.0,
                               s['filters'][:, None] / 4.0, s['burden'][:, None] / 3.0,
                               jax.nn.one_hot(self.carried(s), 4), s['priority'][:, None] / 3.0,
                               jnp.full((3, 1), s['t'] / 96.0)], axis=1)
        public = jnp.concatenate([s['map'].reshape(-1),
                                  (s['station_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
                                  s['station_stock'] / 2.0, s['delivered']])
        af = jnp.concatenate([s['pos'] / jnp.array([8.0, 6.0]), s['stamina'][:, None] / 20.0,
                              s['filters'][:, None] / 4.0, s['burden'][:, None] / 3.0], axis=1)
        av = s['agent_visible']
        local_agents = jnp.concatenate([av[:, :, None], jnp.where(av[:, :, None], af[None, :, :], 0)], axis=2).reshape(3, 18)
        lf = jnp.concatenate([s['load_pos'] / jnp.array([8.0, 6.0]), (s['owner'][:, None] + 2) / 4.0,
                              s['destination'][:, None], s['quality'][:, None] / 6.0,
                              s['load_dust'][:, None] / 4.0], axis=1)
        lv = s['load_visible']
        local_loads = jnp.concatenate([lv[:, :, None], jnp.where(lv[:, :, None], lf[None, :, :], 0)], axis=2).reshape(3, 21)
        dust = jnp.where(s['dust_visible'], s['dust_amount'][None, :] / 3.0, 0)
        return jnp.concatenate([own, jnp.broadcast_to(public, (3, 72)), local_agents,
                                 local_loads, dust, s['dust_visible'], jnp.eye(3)], axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'],
                                 lambda _: (state, jnp.zeros((3,), jnp.float32)),
                                 lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((3,), n['done'], jnp.bool_), info
