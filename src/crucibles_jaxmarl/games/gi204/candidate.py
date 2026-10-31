"""Treaty Rail: original unexecuted spatial source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 188
    num_actions = 15
    max_steps = 160

    def sync(self, s):
        s = dict(s)
        s['wagon_pos'] = jnp.where((s['wagon_train'] >= 0)[:, None], s['agent_pos'][jnp.maximum(s['wagon_train'], 0)], s['wagon_pos'])
        s['train_wagons'] = jnp.sum(s['wagon_train'][None, :] == jnp.arange(3)[:, None], axis=1).astype(jnp.int32)
        s['train_cargo'] = jnp.sum((s['wagon_train'][None, :] == jnp.arange(3)[:, None]) & (s['wagon_cargo'] >= 0)[None, :], axis=1).astype(jnp.int32)
        s['agent_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=2) <= 3
        s['wagon_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['wagon_pos'][None, :, :]), axis=2) <= 3
        s['gate_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['gate_pos'][None, :, :]), axis=2) <= 3
        s['home_vis'] = (jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['home_pos'][None, :, :]), axis=2) <= 3) | jnp.eye(3, dtype=jnp.bool_)
        return s

    def reset(self, key):
        y = jnp.arange(7)[:, None]
        x = jnp.arange(13)[None, :]
        rail = ((y == 3) & (x >= 1) & (x <= 11)) | ((y == 1) & (x >= 2) & (x <= 4)) | (((x == 2) | (x == 4)) & (y >= 1) & (y <= 3)) | ((y == 5) & (x >= 6) & (x <= 8)) | (((x == 6) | (x == 8)) & (y >= 3) & (y <= 5))
        grid = jnp.where(rail, jnp.where(x <= 4, 1, jnp.where(x <= 8, 2, 3)), 0).astype(jnp.int32)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'success': jnp.array(False), 'failure': jnp.array(False), 'map': grid,
            'agent_pos': jnp.array([[6, 3], [8, 3], [10, 3]], jnp.int32),
            'agent_active': jnp.ones(3, jnp.bool_), 'agent_vis': jnp.ones((3, 3), jnp.bool_),
            'train_wagons': jnp.ones(3, jnp.int32), 'train_cargo': jnp.ones(3, jnp.int32),
            'work': jnp.zeros(3, jnp.int32), 'blocked': jnp.zeros(3, jnp.bool_),
            'gate_pos': jnp.array([[3, 3], [7, 3], [9, 3]], jnp.int32),
            'gate_active': jnp.ones(3, jnp.bool_), 'gate_vis': jnp.ones((3, 3), jnp.bool_),
            'gate_owner': jnp.arange(3, dtype=jnp.int32), 'green': jnp.ones(3, jnp.bool_),
            'permit': jnp.zeros((3, 3), jnp.int32),
            'wagon_pos': jnp.array([[6, 3], [8, 3], [10, 3]], jnp.int32),
            'wagon_active': jnp.ones(3, jnp.bool_), 'wagon_vis': jnp.ones((3, 3), jnp.bool_),
            'wagon_owner': jnp.arange(3, dtype=jnp.int32), 'wagon_lessee': jnp.arange(3, dtype=jnp.int32),
            'wagon_train': jnp.arange(3, dtype=jnp.int32), 'wagon_cargo': jnp.arange(3, dtype=jnp.int32),
            'home_pos': jnp.array([[1, 3], [6, 5], [11, 3]], jnp.int32),
            'home_active': jnp.ones(3, jnp.bool_), 'home_vis': jnp.ones((3, 3), jnp.bool_),
            'bank': jnp.full(3, 3, jnp.int32), 'depot_pos': jnp.array([[6, 3]], jnp.int32),
            'depot_active': jnp.ones(1, jnp.bool_), 'depot_vis': jnp.ones((3, 1), jnp.bool_),
            'public_delivered': jnp.zeros(3, jnp.int32), 'private_stock': jnp.zeros(3, jnp.int32),
            'delivery_now': jnp.zeros(3, jnp.int32), 'loan_now': jnp.zeros(3, jnp.bool_),
            'recall_now': jnp.zeros(3, jnp.bool_),
        }
        s = self.sync(s)
        return self.observe(s), s

    def rail_targets(self, s):
        p = s['agent_pos'][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        inside = jnp.all((p >= 0) & (p <= jnp.array([12, 6], jnp.int32)), axis=2)
        track = s['map'][jnp.clip(p[:, :, 1], 0, 6), jnp.clip(p[:, :, 0], 0, 12)] > 0
        train = jnp.any(jnp.all(p[:, :, None, :] == s['agent_pos'][None, None, :, :], axis=3), axis=2)
        wagon = jnp.any(jnp.all(p[:, :, None, :] == s['wagon_pos'][None, None, :, :], axis=3) & (s['wagon_train'] < 0)[None, None, :], axis=2)
        return p, inside & track & ~train & ~wagon

    def legal_actions(self, s):
        ids = jnp.arange(3)
        p = s['agent_pos']
        nearown = jnp.sum(jnp.abs(p - s['gate_pos']), axis=1) <= 1
        neighbor = (jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2) <= 1) & ~jnp.eye(3, dtype=jnp.bool_)
        canloan = (s['wagon_train'] == ids) & jnp.any(neighbor & (s['train_wagons'] < 2)[None, :], axis=1)
        recallnear = jnp.sum(jnp.abs(p - s['wagon_pos']), axis=1) <= 1
        recall = (s['wagon_train'] != ids) & recallnear & (s['train_wagons'] < 2)
        attached = s['wagon_train'][None, :] == ids[:, None]
        empty = jnp.any(attached & (s['wagon_cargo'] < 0)[None, :], axis=1)
        loaded = jnp.any(attached & (s['wagon_cargo'] >= 0)[None, :], axis=1)
        ownhome = jnp.all(p == s['home_pos'], axis=1)
        nearbywagon = jnp.sum(jnp.abs(p[:, None, :] - s['wagon_pos'][None, :, :]), axis=2) <= 1
        eligible = (s['wagon_train'] < 0)[None, :] & ((s['wagon_owner'][None, :] == ids[:, None]) | (s['wagon_lessee'][None, :] == ids[:, None]))
        loose_here = jnp.any(jnp.all(p[:, None, :] == s['wagon_pos'][None, :, :], axis=2) & (s['wagon_train'] < 0)[None, :], axis=1)
        _, walk = self.rail_targets(s)
        m = jnp.zeros((3, 15), jnp.bool_).at[:, 0].set(True).at[:, 1:5].set(walk)
        m = m.at[:, 5].set(nearown).at[:, 6].set(nearown).at[:, 7].set(nearown)
        m = m.at[:, 8].set(canloan).at[:, 9].set(recall)
        m = m.at[:, 10].set(empty & ownhome & (s['bank'] > 0))
        m = m.at[:, 11].set(loaded & jnp.all(p == s['depot_pos'][0], axis=1))
        m = m.at[:, 12].set(loaded & ownhome)
        m = m.at[:, 13].set((s['train_wagons'] > 0) & ~loose_here)
        m = m.at[:, 14].set((s['train_wagons'] < 2) & jnp.any(nearbywagon & eligible, axis=1))
        return jnp.where(s['done'], jnp.arange(15)[None, :] == 0, m)

    def observe(self, s):
        public = jnp.concatenate((jnp.stack((s['t'], s['done'], s['terminated'], s['truncated'], s['success'], s['failure'])), s['map'].ravel(), s['public_delivered'], s['private_stock'], s['delivery_now'], s['depot_pos'].ravel()))
        trains = jnp.concatenate((s['agent_pos'], s['train_wagons'][:, None], s['train_cargo'][:, None]), axis=1)
        gates = jnp.concatenate((s['gate_pos'], s['gate_owner'][:, None], s['green'][:, None], s['permit']), axis=1)
        wagons = jnp.concatenate((s['wagon_pos'], s['wagon_owner'][:, None], s['wagon_lessee'][:, None], s['wagon_train'][:, None], s['wagon_cargo'][:, None]), axis=1)
        homes = jnp.concatenate((s['home_pos'], s['bank'][:, None]), axis=1)
        rows = []
        for i in range(3):
            own = jnp.concatenate((jnp.stack((i, s['work'][i], s['blocked'][i], s['loan_now'][i], s['recall_now'][i])), s['permit'][:, i]))
            row = jnp.concatenate((public, own, s['agent_vis'][i], jnp.where(s['agent_vis'][i, :, None], trains, -1).ravel(), s['gate_vis'][i], jnp.where(s['gate_vis'][i, :, None], gates, -1).ravel(), s['wagon_vis'][i], jnp.where(s['wagon_vis'][i, :, None], wagons, -1).ravel(), s['home_vis'][i], jnp.where(s['home_vis'][i, :, None], homes, -1).ravel()))
            rows = rows + [row.astype(jnp.float32)]
        return jnp.stack(rows)

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 14)
        mask = self.legal_actions(s)
        a = jnp.where((actions >= 0) & (actions < 15) & mask[jnp.arange(3), clipped], clipped, 0)
        n['t'] = s['t'] + 1
        n['work'] = s['work'] + (a != 0).astype(jnp.int32)
        n['blocked'] = actions != a
        n['delivery_now'] = jnp.zeros(3, jnp.int32)
        n['loan_now'] = jnp.zeros(3, jnp.bool_)
        n['recall_now'] = jnp.zeros(3, jnp.bool_)
        n['permit'] = jnp.maximum(s['permit'] - 1, 0)
        for i in range(3):
            recipient = jnp.where(a[i] == 5, (i + 1) % 3, (i + 2) % 3)
            grant = (a[i] == 5) | (a[i] == 6)
            n['permit'] = n['permit'].at[i, recipient].set(jnp.where(grant, 8, n['permit'][i, recipient]))
            n['green'] = n['green'].at[i].set(n['green'][i] ^ (a[i] == 7))
        targetset, geometry = self.rail_targets(s)
        target = targetset[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        wants = (a >= 1) & (a <= 4) & geometry[jnp.arange(3), jnp.clip(a - 1, 0, 3)]
        atgate = jnp.all(target[:, None, :] == s['gate_pos'][None, :, :], axis=2)
        permission = n['green'][None, :] & ((s['gate_owner'][None, :] == jnp.arange(3)[:, None]) | (n['permit'].T > 0))
        passable = ~jnp.any(atgate & ~permission, axis=1)
        moving = wants & passable
        rank = (jnp.arange(3) - s['t']) % 3
        rivals = jnp.all(target[:, None, :] == target[None, :, :], axis=2) & moving[None, :]
        win = moving & ~jnp.any(rivals & (rank[None, :] < rank[:, None]), axis=1)
        n['agent_pos'] = jnp.where(win[:, None], target, s['agent_pos'])
        n['blocked'] = n['blocked'] | (wants & ~win)
        n = self.sync(n)
        bonus = jnp.zeros(3, jnp.float32)
        for k in range(3):
            i = (s['t'] + k) % 3
            p = n['agent_pos'][i]
            neighbor = (jnp.sum(jnp.abs(n['agent_pos'] - p), axis=1) <= 1) & (jnp.arange(3) != i) & (n['train_wagons'] < 2)
            recipient = jnp.argmax(neighbor)
            loan = (a[i] == 8) & (n['wagon_train'][i] == i) & jnp.any(neighbor)
            recall = (a[i] == 9) & (n['wagon_train'][i] != i) & (jnp.sum(jnp.abs(n['wagon_pos'][i] - p)) <= 1) & (n['train_wagons'][i] < 2)
            n['wagon_train'] = n['wagon_train'].at[i].set(jnp.where(loan, recipient, jnp.where(recall, i, n['wagon_train'][i])))
            n['wagon_lessee'] = n['wagon_lessee'].at[i].set(jnp.where(loan, recipient, jnp.where(recall, i, n['wagon_lessee'][i])))
            n['loan_now'] = n['loan_now'].at[i].set(loan)
            n['recall_now'] = n['recall_now'].at[i].set(recall)
            attached = n['wagon_train'] == i
            empties = attached & (n['wagon_cargo'] < 0)
            w = jnp.argmax(empties)
            load = (a[i] == 10) & jnp.any(empties) & (n['bank'][i] > 0)
            n['wagon_cargo'] = n['wagon_cargo'].at[w].set(jnp.where(load, i, n['wagon_cargo'][w]))
            n['bank'] = n['bank'].at[i].add(-load.astype(jnp.int32))
            loaded = attached & (n['wagon_cargo'] >= 0)
            w = jnp.argmax(loaded)
            origin = jnp.maximum(n['wagon_cargo'][w], 0)
            deliver = (a[i] == 11) & jnp.any(loaded)
            stash = (a[i] == 12) & jnp.any(loaded)
            n['public_delivered'] = n['public_delivered'].at[origin].add(deliver.astype(jnp.int32))
            n['private_stock'] = n['private_stock'].at[origin].add(stash.astype(jnp.int32))
            n['delivery_now'] = n['delivery_now'].at[origin].add(deliver.astype(jnp.int32))
            n['wagon_cargo'] = n['wagon_cargo'].at[w].set(jnp.where(deliver | stash, -1, n['wagon_cargo'][w]))
            bonus = bonus.at[i].add(0.1 * deliver.astype(jnp.float32) + 0.5 * stash.astype(jnp.float32)).at[origin].add(0.3 * deliver.astype(jnp.float32) + 0.2 * stash.astype(jnp.float32))
            w = jnp.argmax(attached)
            loose_here = jnp.any(jnp.all(n['wagon_pos'] == p, axis=1) & (n['wagon_train'] < 0))
            detach = (a[i] == 13) & jnp.any(attached) & ~loose_here
            n['wagon_train'] = n['wagon_train'].at[w].set(jnp.where(detach, -1, n['wagon_train'][w]))
            n['wagon_pos'] = n['wagon_pos'].at[w].set(jnp.where(detach, p, n['wagon_pos'][w]))
            eligible = (n['wagon_train'] < 0) & ((n['wagon_owner'] == i) | (n['wagon_lessee'] == i)) & (jnp.sum(jnp.abs(n['wagon_pos'] - p), axis=1) <= 1)
            w = jnp.argmax(eligible)
            couple = (a[i] == 14) & jnp.any(eligible) & (jnp.sum(n['wagon_train'] == i) < 2)
            n['wagon_train'] = n['wagon_train'].at[w].set(jnp.where(couple, i, n['wagon_train'][w]))
            n = self.sync(n)
        remaining = jnp.sum((n['wagon_cargo'][:, None] == jnp.arange(3)[None, :]).astype(jnp.int32), axis=0)
        n['success'] = jnp.all(n['public_delivered'] >= 2)
        n['failure'] = jnp.any(n['public_delivered'] + n['bank'] + remaining < 2) & ~n['success']
        n['terminated'] = n['success'] | n['failure']
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        reward = 0.5 * jnp.sum(n['delivery_now']).astype(jnp.float32) + bonus - 0.01 * (a != 0).astype(jnp.float32) - n['failure'].astype(jnp.float32)
        return n, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full(3, n['done'], jnp.bool_), info
