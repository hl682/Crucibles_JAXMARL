"""Delta Federation. Six settlements, delayed water, visible freight and trade."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 24
    num_actions = 25
    observation_size = 1042
    max_steps = 512
    public_keys = ('t', 'done', 'terminated', 'truncated', 'terrain', 'pos', 'settlement', 'tool', 'inventory', 'energy', 'village_pos', 'field_pos', 'growth', 'wood_pos', 'wood', 'river_pos', 'water', 'gate', 'levee', 'bridge', 'sediment', 'rain', 'packet_pos', 'packet_amount', 'barge_pos', 'barge_cargo', 'barge_owner', 'barge_progress', 'barge_direction', 'barge_length', 'delivered', 'flooded', 'built_now', 'shipped', 'handed', 'fed', 'reward_parts')

    def reset(self, key):
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False, jnp.bool_), 'terminated': jnp.array(False, jnp.bool_), 'truncated': jnp.array(False, jnp.bool_),
            'terrain': jnp.zeros((19, 25), jnp.int32),
            'pos': jnp.array([[5, 4], [4, 6], [6, 6], [5, 2], [11, 4], [10, 6], [12, 6], [11, 2], [21, 3], [20, 5], [22, 5], [18, 3], [21, 12], [20, 14], [22, 14], [18, 11], [11, 13], [10, 15], [12, 15], [11, 10], [3, 13], [2, 15], [4, 15], [5, 10]], jnp.int32),
            'agent_active': jnp.ones((24,), jnp.bool_), 'agent_visible': jnp.ones((24, 24), jnp.bool_),
            'settlement': jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], jnp.int32),
            'tool': jnp.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], jnp.int32),
            'inventory': jnp.zeros((24, 2), jnp.int32), 'energy': jnp.full((24,), 16, jnp.int32),
            'village_pos': jnp.array([[5, 6], [11, 6], [21, 5], [21, 14], [11, 15], [3, 15]], jnp.int32),
            'village_active': jnp.ones((6,), jnp.bool_), 'village_visible': jnp.zeros((24, 6), jnp.bool_), 'granary': jnp.full((6,), 6, jnp.int32),
            'forecast': jnp.ones((6,), jnp.int32), 'notes_food': jnp.zeros((24, 6), jnp.int32), 'notes_weather': jnp.zeros((24, 6), jnp.int32), 'note_life': jnp.zeros((24, 6), jnp.int32),
            'field_pos': jnp.array([[5, 5], [11, 5], [21, 4], [21, 13], [11, 14], [3, 14]], jnp.int32),
            'field_active': jnp.ones((6,), jnp.bool_), 'field_visible': jnp.ones((24, 6), jnp.bool_), 'growth': jnp.full((6,), 4, jnp.int32),
            'wood_pos': jnp.array([[3, 6], [9, 6], [19, 5], [19, 14], [9, 15], [1, 15]], jnp.int32),
            'wood_active': jnp.ones((6,), jnp.bool_), 'wood_visible': jnp.ones((24, 6), jnp.bool_), 'wood': jnp.full((6,), 8, jnp.int32),
            'river_pos': jnp.array([[5, 3], [11, 3], [17, 3], [17, 11], [11, 11], [5, 11]], jnp.int32),
            'river_active': jnp.ones((6,), jnp.bool_), 'river_visible': jnp.ones((24, 6), jnp.bool_),
            'water': jnp.array([5, 3, 3, 3, 3, 3], jnp.int32), 'gate': jnp.array([False, True, True, True, True, True], jnp.bool_), 'levee': jnp.zeros((6,), jnp.int32),
            'bridge': jnp.array([False, True, False, False, True, False], jnp.bool_), 'bridge_ever': jnp.array([False, True, False, False, True, False], jnp.bool_), 'sediment': jnp.zeros((6,), jnp.int32), 'rain': jnp.zeros((6,), jnp.int32),
            'water_queue': jnp.zeros((6, 3), jnp.int32),
            'packet_pos': jnp.array([[7, 3], [9, 3], [11, 3], [13, 3], [15, 3], [17, 3], [17, 5], [17, 8], [17, 11], [15, 11], [13, 11], [11, 11], [9, 11], [7, 11], [5, 11], [4, 11], [2, 11], [0, 11]], jnp.int32),
            'packet_active': jnp.zeros((18,), jnp.bool_), 'packet_visible': jnp.ones((24, 18), jnp.bool_), 'packet_amount': jnp.zeros((18,), jnp.int32),
            'barge_pos': jnp.array([[5, 3], [11, 3], [17, 3], [17, 11], [11, 11]], jnp.int32),
            'barge_active': jnp.ones((5,), jnp.bool_), 'barge_visible': jnp.ones((24, 5), jnp.bool_), 'barge_cargo': jnp.zeros((5,), jnp.int32), 'barge_owner': jnp.full((5,), -1, jnp.int32),
            'barge_progress': jnp.zeros((5,), jnp.int32), 'barge_direction': jnp.ones((5,), jnp.int32), 'barge_length': jnp.array([6, 6, 8, 6, 6], jnp.int32),
            'delivered': jnp.zeros((24,), jnp.int32), 'flooded': jnp.zeros((6,), jnp.int32), 'built_now': jnp.zeros((6,), jnp.int32), 'shipped': jnp.zeros((5,), jnp.int32), 'handed': jnp.zeros((24,), jnp.int32),
            'fed': jnp.zeros((6,), jnp.int32), 'reward_parts': jnp.zeros((24, 5), jnp.float32)
        }
        s['forecast'] = jax.random.randint(key, (6,), 0, 3, dtype=jnp.int32)
        s['inventory'] = s['inventory'].at[0::4, 0].set(1).at[1::4, 1].set(3)
        s['village_visible'] = s['settlement'][:, None] == jnp.arange(6, dtype=jnp.int32)[None, :]
        s['notes_food'] = jnp.where(s['village_visible'], s['granary'][None, :], 0)
        s['notes_weather'] = jnp.where(s['village_visible'], s['forecast'][None, :], -1)
        s['note_life'] = jnp.where(s['village_visible'], 16, 0).astype(jnp.int32)
        xx, yy = jnp.arange(25, dtype=jnp.int32)[None, :], jnp.arange(19, dtype=jnp.int32)[:, None]
        for c in range(6):
            town = (jnp.abs(xx - s['village_pos'][c, 0]) <= 2) & (jnp.abs(yy - s['village_pos'][c, 1]) <= 2)
            s['terrain'] = jnp.where(town, 10 + c, s['terrain'])
            ditch = ((yy == s['river_pos'][c, 1]) & (xx >= jnp.minimum(s['river_pos'][c, 0], s['field_pos'][c, 0])) & (xx <= jnp.maximum(s['river_pos'][c, 0], s['field_pos'][c, 0]))) | ((xx == s['field_pos'][c, 0]) & (yy >= jnp.minimum(s['river_pos'][c, 1], s['field_pos'][c, 1])) & (yy <= jnp.maximum(s['river_pos'][c, 1], s['field_pos'][c, 1])))
            s['terrain'] = jnp.where(ditch, 2, s['terrain'])
        s['terrain'] = s['terrain'].at[3, 5:18].set(1).at[3:12, 17].set(1).at[11, 0:18].set(1)
        return self.observe(s), s

    def distances(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=2, dtype=jnp.int32)

    def observe(self, s):
        public = jnp.concatenate([s[k].astype(jnp.float32).ravel() for k in self.public_keys])
        own = jnp.concatenate((jnp.arange(24, dtype=jnp.int32)[:, None], s['notes_food'], s['notes_weather'], s['note_life']), axis=1).astype(jnp.float32)
        return jnp.concatenate((own, jnp.broadcast_to(public, (24, public.shape[0]))), axis=1)

    def legal_actions(self, s):
        df = self.distances(s['pos'], s['field_pos']) <= 1
        dr = self.distances(s['pos'], s['river_pos']) <= 1
        db = self.distances(s['pos'], s['barge_pos']) <= 1
        dv = self.distances(s['pos'], s['village_pos']) <= 1
        free = jnp.sum(s['inventory'], axis=1, dtype=jnp.int32) < 6
        peer = self.distances(s['pos'], s['pos']) == 1
        farmer, builder, courier, operator = s['tool'] == 1, s['tool'] == 2, s['tool'] == 3, s['tool'] == 4
        harvest = farmer & free & jnp.any(df & (s['growth'][None, :] >= 4), axis=1)
        timber = free & jnp.any((self.distances(s['pos'], s['wood_pos']) <= 1) & (s['wood'][None, :] > 0), axis=1)
        levee = builder & (s['inventory'][:, 1] >= 2) & jnp.any(dr & (s['levee'][None, :] < 3), axis=1)
        opened = operator & jnp.any(dr & ~s['gate'][None, :], axis=1)
        closed = operator & jnp.any(dr & s['gate'][None, :], axis=1)
        dredge = operator & jnp.any(dr & (s['sediment'][None, :] > 0), axis=1)
        load = courier & (jnp.sum(s['inventory'], axis=1, dtype=jnp.int32) > 0) & jnp.any(db & (s['barge_cargo'][None, :] == 0), axis=1)
        unload = free & jnp.any(db & (s['barge_cargo'][None, :] > 0), axis=1)
        deliver = (s['inventory'][:, 0] > 0) & jnp.any(dv, axis=1)
        take = free & jnp.any(dv & s['village_visible'], axis=1)
        give = (jnp.sum(s['inventory'], axis=1, dtype=jnp.int32) > 0) & jnp.any(peer & free[None, :], axis=1)
        bridge = builder & (s['inventory'][:, 1] >= 2) & jnp.any(dr & ~s['bridge'][None, :], axis=1)
        repair = builder & (s['inventory'][:, 1] > 0) & jnp.any(dr & (s['sediment'][None, :] > 0), axis=1)
        plant = farmer & (s['inventory'][:, 0] > 0) & jnp.any(df & (s['growth'][None, :] < 4), axis=1)
        breach = operator & jnp.any(dr & (s['levee'][None, :] > 0), axis=1)
        extra = jnp.stack((harvest, timber, levee, opened, closed, dredge, load, unload, jnp.any(db, axis=1), jnp.any(db, axis=1), deliver, take, give, jnp.any(peer, axis=1), jnp.any(peer, axis=1), bridge, repair, jnp.ones((24,), jnp.bool_), plant, breach), axis=1)
        mask = jnp.concatenate((jnp.ones((24, 5), jnp.bool_), extra), axis=1) & (s['energy'][:, None] > 0) & ~s['done']
        return mask.at[:, 0].set(True).at[:, 22].set(~s['done'])

    def interact(self, i, s, a):
        b = a[i]
        inv = s['inventory'][i]
        free = jnp.sum(inv, dtype=jnp.int32) < 6
        fd = jnp.sum(jnp.abs(s['field_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        fi = jnp.argmin(fd)
        harvest = (b == 5) & (s['tool'][i] == 1) & (fd[fi] <= 1) & (s['growth'][fi] >= 4) & free
        plant = (b == 23) & (s['tool'][i] == 1) & (fd[fi] <= 1) & (s['growth'][fi] < 4) & (inv[0] > 0)
        s['growth'] = s['growth'].at[fi].set(jnp.where(harvest, 0, jnp.minimum(6, s['growth'][fi] + 2 * plant.astype(jnp.int32))))
        s['inventory'] = s['inventory'].at[i, 0].add(harvest.astype(jnp.int32) - plant.astype(jnp.int32))
        wd = jnp.sum(jnp.abs(s['wood_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        wi = jnp.argmin(wd)
        wood = (b == 6) & (wd[wi] <= 1) & (s['wood'][wi] > 0) & free
        s['wood'] = s['wood'].at[wi].add(-wood.astype(jnp.int32))
        s['inventory'] = s['inventory'].at[i, 1].add(wood.astype(jnp.int32))
        rd = jnp.sum(jnp.abs(s['river_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        ri = jnp.argmin(rd)
        near = rd[ri] <= 1
        levee = (b == 7) & near & (s['tool'][i] == 2) & (inv[1] >= 2) & (s['levee'][ri] < 3)
        open_gate = (b == 8) & near & (s['tool'][i] == 4)
        close_gate = (b == 9) & near & (s['tool'][i] == 4)
        dredge = (b == 10) & near & (s['tool'][i] == 4) & (s['sediment'][ri] > 0)
        bridge = (b == 20) & near & (s['tool'][i] == 2) & (inv[1] >= 2) & ~s['bridge'][ri]
        repair = (b == 21) & near & (s['tool'][i] == 2) & (inv[1] > 0) & (s['sediment'][ri] > 0)
        breach = (b == 24) & near & (s['tool'][i] == 4) & (s['levee'][ri] > 0)
        s['levee'] = s['levee'].at[ri].add(levee.astype(jnp.int32) - breach.astype(jnp.int32))
        s['gate'] = s['gate'].at[ri].set(jnp.where(open_gate, True, jnp.where(close_gate, False, s['gate'][ri])))
        s['sediment'] = s['sediment'].at[ri].set(jnp.maximum(0, s['sediment'][ri] - 2 * dredge.astype(jnp.int32) - repair.astype(jnp.int32)))
        s['built_now'] = s['built_now'].at[ri].add((bridge & ~s['bridge_ever'][ri]).astype(jnp.int32))
        s['bridge_ever'] = s['bridge_ever'].at[ri].set(s['bridge_ever'][ri] | bridge)
        s['bridge'] = s['bridge'].at[ri].set(s['bridge'][ri] | bridge)
        materials = 2 * levee.astype(jnp.int32) + 2 * bridge.astype(jnp.int32) + repair.astype(jnp.int32)
        s['inventory'] = s['inventory'].at[i, 1].add(-materials)
        bd = jnp.sum(jnp.abs(s['barge_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        eligible_barge = ((b == 11) & (s['barge_cargo'] == 0)) | ((b == 12) & (s['barge_cargo'] > 0)) | (b == 13) | (b == 14)
        bi = jnp.argmin(jnp.where(eligible_barge, bd, 999))
        resource = jnp.where(inv[0] > 0, 0, 1)
        load = (b == 11) & (bd[bi] <= 1) & (s['tool'][i] == 3) & (s['barge_cargo'][bi] == 0) & (inv[resource] > 0)
        oldcargo = s['barge_cargo'][bi]
        take = (b == 12) & (bd[bi] <= 1) & (oldcargo > 0) & free
        s['inventory'] = s['inventory'].at[i, resource].add(-load.astype(jnp.int32)).at[i, jnp.clip(oldcargo - 1, 0, 1)].add(take.astype(jnp.int32))
        s['barge_cargo'] = s['barge_cargo'].at[bi].set(jnp.where(load, resource + 1, jnp.where(take, 0, oldcargo)))
        s['barge_owner'] = s['barge_owner'].at[bi].set(jnp.where(load, s['settlement'][i], jnp.where(take, -1, s['barge_owner'][bi])))
        forward = (b == 13) & (bd[bi] <= 1)
        reverse = (b == 14) & (bd[bi] <= 1)
        s['barge_direction'] = s['barge_direction'].at[bi].set(jnp.where(forward, 1, jnp.where(reverse, -1, s['barge_direction'][bi])))
        vd = jnp.sum(jnp.abs(s['village_pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        vi = jnp.argmin(vd)
        deliver = (b == 15) & (vd[vi] <= 1) & (inv[0] > 0) & (s['granary'][vi] < 32)
        takefood = (b == 16) & (vd[vi] <= 1) & (vi == s['settlement'][i]) & (s['granary'][vi] > 0) & free
        s['granary'] = s['granary'].at[vi].add(deliver.astype(jnp.int32) - takefood.astype(jnp.int32))
        s['inventory'] = s['inventory'].at[i, 0].add(-deliver.astype(jnp.int32) + takefood.astype(jnp.int32))
        s['delivered'] = s['delivered'].at[i].set(deliver.astype(jnp.int32))
        stock_change = deliver.astype(jnp.float32) - takefood.astype(jnp.float32)
        s['reward_parts'] = s['reward_parts'].at[:, 0].add(0.8 * stock_change * (s['settlement'] == vi).astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[i, 1].add(0.2 * stock_change)
        pd = jnp.sum(jnp.abs(s['pos'] - s['pos'][i]), axis=1, dtype=jnp.int32)
        capacity = jnp.sum(s['inventory'], axis=1, dtype=jnp.int32) < 6
        pi = jnp.argmin(jnp.where((pd == 1) & capacity, jnp.arange(24, dtype=jnp.int32), 999))
        give = (b == 17) & (pd[pi] == 1) & capacity[pi] & (inv[resource] > 0)
        s['inventory'] = s['inventory'].at[i, resource].add(-give.astype(jnp.int32)).at[pi, resource].add(give.astype(jnp.int32))
        s['handed'] = s['handed'].at[i].set(give.astype(jnp.int32))
        pi = jnp.argmin(jnp.where(pd == 1, jnp.arange(24, dtype=jnp.int32), 999))
        swap = (b == 18) & (pd[pi] == 1)
        ti, tp = s['tool'][i], s['tool'][pi]
        s['tool'] = s['tool'].at[i].set(jnp.where(swap, tp, ti)).at[pi].set(jnp.where(swap, ti, tp))
        speak = (b == 19) & (pd == 1)
        home = s['settlement'][i]
        s['notes_food'] = s['notes_food'].at[:, home].set(jnp.where(speak, s['granary'][home], s['notes_food'][:, home]))
        s['notes_weather'] = s['notes_weather'].at[:, home].set(jnp.where(speak, s['forecast'][home], s['notes_weather'][:, home]))
        s['note_life'] = s['note_life'].at[:, home].set(jnp.where(speak, 16, s['note_life'][:, home]))
        s['reward_parts'] = s['reward_parts'].at[i, 3].add(-0.08 * materials.astype(jnp.float32) - 0.05 * plant.astype(jnp.float32))
        return s

    def advance(self, key, state, actions):
        s = dict(state)
        a = jnp.clip(actions, 0, 24)
        valid = (actions >= 0) & (actions < 25) & self.legal_actions(s)[jnp.arange(24, dtype=jnp.int32), a]
        a = jnp.where(valid, a, 0)
        s['delivered'] = jnp.zeros((24,), jnp.int32)
        s['built_now'] = jnp.zeros((6,), jnp.int32)
        s['handed'] = jnp.zeros((24,), jnp.int32)
        s['reward_parts'] = jnp.zeros((24, 5), jnp.float32)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['pos'] + delta[jnp.where(a < 5, a, 0)]
        inside = jnp.all((p >= 0) & (p <= jnp.array([24, 18], jnp.int32)), axis=1)
        q = jnp.clip(p, jnp.array([0, 0], jnp.int32), jnp.array([24, 18], jnp.int32))
        occupied = jnp.any(jnp.all(q[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(24, dtype=jnp.bool_), axis=1)
        clash = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1, dtype=jnp.int32) > 1
        bridged = jnp.any(jnp.all(q[:, None, :] == s['river_pos'][None, :, :], axis=2) & s['bridge'][None, :], axis=1)
        ground = (s['terrain'][q[:, 1], q[:, 0]] != 1) | bridged
        move = (a > 0) & (a < 5) & inside & ~occupied & ~clash & ground
        s['pos'] = jnp.where(move[:, None], q, s['pos'])
        effort = move.astype(jnp.int32) + ((a >= 5) & (a != 22)).astype(jnp.int32)
        s['energy'] = jnp.clip(s['energy'] - effort + 2 * (a == 0).astype(jnp.int32) + 5 * (a == 22).astype(jnp.int32), 0, 16)
        s['reward_parts'] = s['reward_parts'].at[:, 3].add(-0.02 * effort.astype(jnp.float32))
        s = jax.lax.fori_loop(0, 24, lambda i, carry: self.interact(i, carry, a), s)
        phase = (s['t'] % 4) == 0
        s['rain'] = (((s['t'] % 12) == 0) & (((s['t'] // 12 + s['forecast']) % 3) != 0)).astype(jnp.int32)
        incoming = jnp.concatenate((jnp.zeros((1,), jnp.int32), s['water_queue'][:-1, 2]))
        level = jnp.minimum(12, s['water'] + incoming + s['rain'])
        release = jnp.minimum(level, 1 + (level >= 5).astype(jnp.int32) + ((s['levee'] > 0) & (level >= 6)).astype(jnp.int32)) * (s['gate'] & phase).astype(jnp.int32)
        s['water_queue'] = jnp.concatenate((release[:, None], s['water_queue'][:, :2]), axis=1)
        s['water'] = jnp.clip(level - release - (phase & (level > 0)).astype(jnp.int32), 0, 9)
        s['packet_amount'] = s['water_queue'].ravel()
        s['packet_active'] = s['packet_amount'] > 0
        s['sediment'] = jnp.minimum(6, s['sediment'] + s['rain'])
        flood = phase & (s['water'] > 5 + s['levee'])
        s['flooded'] = flood.astype(jnp.int32)
        grows = phase & (s['water'] >= 2) & ~flood
        s['growth'] = jnp.where(flood, 0, jnp.minimum(6, s['growth'] + grows.astype(jnp.int32)))
        s['levee'] = jnp.maximum(0, s['levee'] - (flood & (s['water'] >= 9)).astype(jnp.int32))
        s['bridge'] = s['bridge'] & ~(flood & (s['water'] >= 9))
        navigable = phase & s['gate'][:5] & (s['water'][:5] >= 2) & (s['water'][1:] >= 2) & (s['sediment'][:5] < 4) & (s['sediment'][1:] < 4)
        progress = jnp.clip(s['barge_progress'] + navigable.astype(jnp.int32) * s['barge_direction'], 0, s['barge_length'])
        s['shipped'] = ((progress != s['barge_progress']) & (s['barge_cargo'] > 0)).astype(jnp.int32)
        s['barge_progress'] = progress
        s['barge_pos'] = s['river_pos'][:5] + jnp.sign(s['river_pos'][1:] - s['river_pos'][:5]) * progress[:, None]
        meal = (s['t'] % 12) == 11
        hungry = meal & (s['granary'] == 0)
        s['fed'] = (meal & (s['granary'] > 0)).astype(jnp.int32)
        s['granary'] = jnp.maximum(0, s['granary'] - meal.astype(jnp.int32))
        s['wood'] = jnp.minimum(8, s['wood'] + ((s['t'] % 24) == 23).astype(jnp.int32))
        s['reward_parts'] = s['reward_parts'].at[:, 0].add(0.7 * s['fed'][s['settlement']].astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[:, 2].add(0.15 * jnp.sum(s['built_now'], dtype=jnp.int32).astype(jnp.float32) + 0.02 * jnp.sum(navigable, dtype=jnp.int32).astype(jnp.float32))
        s['reward_parts'] = s['reward_parts'].at[:, 4].add(-0.4 * flood[s['settlement']].astype(jnp.float32) - 0.6 * hungry[s['settlement']].astype(jnp.float32))
        s['note_life'] = jnp.maximum(0, s['note_life'] - 1)
        s['notes_food'] = jnp.where(s['note_life'] > 0, s['notes_food'], 0)
        s['notes_weather'] = jnp.where(s['note_life'] > 0, s['notes_weather'], -1)
        s['notes_food'] = jnp.where(s['village_visible'], s['granary'][None, :], s['notes_food'])
        s['notes_weather'] = jnp.where(s['village_visible'], s['forecast'][None, :], s['notes_weather'])
        s['note_life'] = jnp.where(s['village_visible'], 16, s['note_life'])
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.array(False, jnp.bool_)
        s['truncated'] = (s['t'] >= 512) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        return s, jnp.sum(s['reward_parts'], axis=1, dtype=jnp.float32)

    def raw_step(self, key, state, actions):
        s, rewards = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros((24,), jnp.float32)), lambda _: self.advance(key, state, actions), operand=None)
        return self.observe(s), s, rewards, jnp.full((24,), s['done'], jnp.bool_), {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
