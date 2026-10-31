"""Beacon Auction: original charge-backed spatial rights; authored, never executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 114
    num_actions = 10
    max_steps = 80

    def refresh(self, s):
        return dict(s, escrow_total=jnp.sum(s['escrow'], axis=0))

    def reset(self, key):
        x = jnp.arange(9, dtype=jnp.int32)[None, :]
        y = jnp.arange(7, dtype=jnp.int32)[:, None]
        tiles = jnp.where(x == 4, jnp.where((y == 1) | (y == 5), 2,
                                           jnp.where(y == 3, 3, 1)), 0).astype(jnp.int32)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'map': tiles,
            'pos': jnp.array([[3, 1], [3, 5], [5, 1], [5, 5]], jnp.int32),
            'agent_active': jnp.ones((4,), jnp.bool_),
            'agent_visible': jnp.ones((4, 4), jnp.bool_),
            'private_visible': jnp.eye(4, dtype=jnp.bool_),
            'charge': jnp.full((4,), 6, jnp.int32),
            'values': jax.random.randint(key, (4, 2), 2, 6, dtype=jnp.int32),
            'delivered': jnp.zeros((4,), jnp.bool_),
            'escrow': jnp.zeros((4, 2), jnp.int32),
            'escrow_total': jnp.zeros((2,), jnp.int32),
            'beacon_pos': jnp.array([[4, 1], [4, 5]], jnp.int32),
            'beacon_active': jnp.ones((2,), jnp.bool_),
            'beacon_visible': jnp.ones((4, 2), jnp.bool_),
            'owner': jnp.full((2,), -1, jnp.int32),
            'ttl': jnp.zeros((2,), jnp.int32),
            'terminal_pos': jnp.array([[1, 1], [1, 5], [7, 1], [7, 5]], jnp.int32),
            'terminal_active': jnp.ones((4,), jnp.bool_),
            'terminal_visible': jnp.ones((4, 4), jnp.bool_),
            'spent_charge': jnp.array(0, jnp.int32),
            'cleared': jnp.zeros((2,), jnp.bool_),
            'expired': jnp.zeros((2,), jnp.bool_),
            'toll_paid': jnp.zeros((4,), jnp.int32),
            'shared': jnp.zeros((4,), jnp.int32),
            'relayed': jnp.zeros((4,), jnp.bool_),
            'delivered_now': jnp.zeros((4,), jnp.bool_)
        }
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(4, dtype=jnp.int32)
        team = ids // 2
        directions = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
        target = s['pos'][:, None, :] + directions[None, :, :]
        clipped = jnp.clip(target, jnp.array([0, 0]), jnp.array([8, 6]))
        tile = s['map'][clipped[:, :, 1], clipped[:, :, 0]]
        matches = jnp.all(target[:, :, None, :] == s['beacon_pos'][None, None, :, :], axis=3)
        owned = (s['owner'][None, None, :] >= 0) & (s['ttl'][None, None, :] > 0)
        free = s['owner'][None, None, :] // 2 == team[:, None, None]
        permission = jnp.any(matches & owned & (free | (s['charge'][:, None, None] >= 1)), axis=2)
        occupied = jnp.any(jnp.all(target[:, :, None, :] == s['pos'][None, None, :, :], axis=3), axis=2)
        move = (jnp.all(target == clipped, axis=2) & (tile != 1) & ~occupied &
                ((tile != 2) | permission) & ((tile != 3) | (s['charge'][:, None] >= 2)))
        near = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['beacon_pos'][None, :, :]), axis=2) <= 1
        can_bid = jnp.any(near & (s['ttl'][None, :] == 0), axis=1)
        mate = jnp.array([1, 0, 3, 2], jnp.int32)
        share = (jnp.sum(jnp.abs(s['pos'] - s['pos'][mate]), axis=1) == 1) & (s['charge'] >= 2)
        on_beacon = jnp.all(s['pos'][:, None, :] == s['beacon_pos'][None, :, :], axis=2)
        origin = jnp.argmax(on_beacon, axis=1)
        exit_pos = s['beacon_pos'][1 - origin]
        exit_free = ~jnp.any(jnp.all(exit_pos[:, None, :] == s['pos'][None, :, :], axis=2), axis=1)
        both = jnp.all((s['ttl'][None, :] > 0) & (s['owner'][None, :] // 2 == team[:, None]), axis=1)
        relay = jnp.any(on_beacon, axis=1) & both & exit_free & (s['charge'] >= 1)
        goal_x = jnp.where(team == 0, 7, 1)
        deliver = (~s['delivered']) & (s['pos'][:, 0] == goal_x) & ((s['pos'][:, 1] == 1) | (s['pos'][:, 1] == 5))
        mask = jnp.concatenate([jnp.ones((4, 1), jnp.bool_), move,
                                (can_bid & (s['charge'] >= 1))[:, None],
                                (can_bid & (s['charge'] >= 2))[:, None],
                                share[:, None], relay[:, None], deliver[:, None]], axis=1)
        return mask & ((~s['done']) | (jnp.arange(10)[None, :] == 0))

    def advance(self, s, actions):
        ids = jnp.arange(4, dtype=jnp.int32)
        team = ids // 2
        safe = jnp.clip(actions, 0, 9)
        valid = (actions >= 0) & (actions < 10) & self.legal_actions(s)[ids, safe]
        a = jnp.where(valid, safe, 0)
        directions = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        on_beacon = jnp.all(s['pos'][:, None, :] == s['beacon_pos'][None, :, :], axis=2)
        origin = jnp.argmax(on_beacon, axis=1)
        target = jnp.where((a == 8)[:, None], s['beacon_pos'][1 - origin], s['pos'] + directions[a])
        clash = jnp.all(target[:, None, :] == target[None, :, :], axis=2)
        loses = jnp.any(clash & (ids[None, :] < ids[:, None]), axis=1)
        pos = jnp.where(loses[:, None], s['pos'], target)
        moved = jnp.any(pos != s['pos'], axis=1)
        relayed = moved & (a == 8)
        # Movement uses old rights, including their last tick. Exiting is always free.
        entered = jnp.all(pos[:, None, :] == s['beacon_pos'][None, :, :], axis=2) & moved[:, None]
        tolls = entered & (s['ttl'][None, :] > 0) & (s['owner'][None, :] // 2 != team[:, None])
        paid = jnp.sum(tolls, axis=1).astype(jnp.int32)
        income = jnp.sum(jax.nn.one_hot(jnp.clip(s['owner'], 0, 3), 4, dtype=jnp.int32) *
                         jnp.sum(tolls, axis=0)[:, None], axis=0)
        bypass = moved & (s['map'][pos[:, 1], pos[:, 0]] == 3)
        travel_burn = 2 * bypass.astype(jnp.int32) + relayed.astype(jnp.int32)
        # Adjacent pair gifts are handed over at old positions before movement.
        mate = jnp.array([1, 0, 3, 2], jnp.int32)
        shared = 2 * (a == 7).astype(jnp.int32)
        charge = s['charge'] - paid + income - travel_burn - shared + shared[mate]
        near = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['beacon_pos'][None, :, :]), axis=2) <= 1
        site = jnp.argmax(near & (s['ttl'][None, :] == 0), axis=1)
        bid = jnp.where(a == 5, 1, jnp.where(a == 6, 2, 0)).astype(jnp.int32)
        escrow = s['escrow'] + jax.nn.one_hot(site, 2, dtype=jnp.int32) * bid[:, None]
        charge = charge - bid
        t = s['t'] + 1
        expired = s['ttl'] == 1
        ttl = jnp.maximum(s['ttl'] - 1, 0)
        owner = jnp.where(ttl == 0, -1, s['owner'])
        # Delayed clearing every fourth transition; highest total wins, then rotating rank.
        cleared = (jnp.mod(t, 4) == 0) & (ttl == 0) & (jnp.sum(escrow, axis=0) > 0)
        priority = jnp.mod(ids - jnp.mod(t // 4, 4), 4)
        score = escrow * 5 + (3 - priority)[:, None]
        winner = jnp.argmax(score, axis=0).astype(jnp.int32)
        winners = ids[:, None] == winner[None, :]
        refund = jnp.sum(jnp.where(cleared[None, :] & ~winners, escrow, 0), axis=1)
        payment = jnp.sum(jnp.where(cleared[None, :] & winners, escrow, 0))
        charge = charge + refund
        escrow = jnp.where(cleared[None, :], 0, escrow)
        owner = jnp.where(cleared, winner, owner)
        ttl = jnp.where(cleared, 6, ttl)
        # Either opposing terminal pays a private value; quarter credit goes to the mate.
        delivered_now = a == 9
        destination = (s['pos'][:, 1] == 5).astype(jnp.int32)
        value = s['values'][ids, destination].astype(jnp.float32) * delivered_now
        reward = value + 0.25 * value[mate]
        reward = reward - 0.02 * (travel_burn + bid).astype(jnp.float32)
        delivered = s['delivered'] | delivered_now
        terminated = jnp.all(delivered)
        truncated = (t >= self.max_steps) & ~terminated
        n = dict(s, t=t, done=terminated | truncated, terminated=terminated, truncated=truncated,
                 pos=pos, charge=charge, escrow=escrow, owner=owner, ttl=ttl,
                 delivered=delivered, delivered_now=delivered_now, cleared=cleared,
                 expired=expired, toll_paid=paid, shared=shared, relayed=relayed,
                 spent_charge=s['spent_charge'] + jnp.sum(travel_burn) + payment)
        return self.refresh(n), reward.astype(jnp.float32)

    def observe(self, s):
        public = jnp.concatenate([jnp.array([s['t'] / 80.0]),
                                  (s['pos'] / jnp.array([8.0, 6.0])).reshape(-1),
                                  s['charge'] / 24.0, s['escrow'].reshape(-1) / 24.0,
                                  s['delivered'], (s['beacon_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
                                  (s['owner'] + 1) / 4.0, s['ttl'] / 6.0, s['escrow_total'] / 24.0,
                                  (s['terminal_pos'] / jnp.array([8.0, 6.0])).reshape(-1),
                                  s['map'].reshape(-1) / 3.0, jnp.array([s['done']]),
                                  jnp.array([s['spent_charge'] / 24.0])])
        return jnp.concatenate([s['values'] / 5.0, jnp.broadcast_to(public, (4, 108)),
                                 jnp.eye(4)], axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'],
                                 lambda _: (state, jnp.zeros((4,), jnp.float32)),
                                 lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((4,), n['done'], jnp.bool_), info
