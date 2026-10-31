"""Charter Frontier: original funded, delayed and expiring corridor charter."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 140
    num_actions = 13
    max_steps = 96

    def distances(self, first, second):
        return jnp.sum(jnp.abs(first[:, None, :] - second[None, :, :]), axis=-1)

    def reset(self, key):
        values = jax.random.randint(key, (4, 4), 1, 3, dtype=jnp.int32)
        values = jnp.where(jnp.eye(4, dtype=jnp.bool_), 3, values)
        state = {
            't': jnp.array(0, jnp.int32),
            'done': jnp.array(False, jnp.bool_),
            'terminated': jnp.array(False, jnp.bool_),
            'truncated': jnp.array(False, jnp.bool_),
            'map': jnp.zeros((7, 11), jnp.int32).at[1:6, 5].set(1).at[3, 5].set(2),
            'agent_pos': jnp.array([[3, 2], [3, 3], [3, 4], [2, 3]], jnp.int32),
            'agent_active': jnp.ones((4,), jnp.bool_),
            'agent_visibility': jnp.ones((4, 4), jnp.bool_),
            'materials': jnp.full((4,), 3, jnp.int32),
            'private_value': values,
            'plot_pos': jnp.array([[8, 1], [8, 2], [8, 4], [8, 5]], jnp.int32),
            'plot_active': jnp.ones((4,), jnp.bool_),
            'plot_visibility': jnp.ones((4, 4), jnp.bool_),
            'owner': jnp.full((4,), -1, jnp.int32),
            'built': jnp.zeros((4,), jnp.bool_),
            'growth': jnp.zeros((4,), jnp.int32),
            'harvests': jnp.zeros((4,), jnp.int32),
            'council_pos': jnp.array([[3, 3]], jnp.int32),
            'council_active': jnp.ones((1,), jnp.bool_),
            'council_visibility': jnp.ones((4, 1), jnp.bool_),
            'fund': jnp.zeros((1,), jnp.int32),
            'gate_pos': jnp.array([[5, 3]], jnp.int32),
            'gate_active': jnp.ones((1,), jnp.bool_),
            'gate_visibility': jnp.ones((4, 1), jnp.bool_),
            'gate_open': jnp.zeros((1,), jnp.bool_),
            'lease': jnp.zeros((1,), jnp.int32),
            'ballot_remaining': jnp.array(0, jnp.int32),
            'proposer': jnp.array(-1, jnp.int32),
            'votes': jnp.zeros((4,), jnp.int32),
            'pending': jnp.array(0, jnp.int32),
            'launched': jnp.array(False, jnp.bool_),
            'ratified': jnp.array(False, jnp.bool_),
            'rejected': jnp.array(False, jnp.bool_),
            'opened': jnp.array(False, jnp.bool_),
            'expired': jnp.array(False, jnp.bool_),
            'claimed': jnp.zeros((4,), jnp.bool_),
            'built_event': jnp.zeros((4,), jnp.bool_),
            'donated': jnp.zeros((4,), jnp.bool_),
            'harvest_event': jnp.zeros((4,), jnp.bool_),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        pos = state['agent_pos']
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = pos[:, None, :] + delta[None, :, :]
        inside = jnp.all((proposed >= 0) & (proposed <= jnp.array([10, 6])), axis=-1)
        clipped = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([10, 6]))
        tiles = state['map'][clipped[:, :, 1], clipped[:, :, 0]]
        movement = inside & (tiles != 1) & ((tiles != 2) | state['gate_open'][0])
        nearby = self.distances(pos, state['plot_pos']) <= 1
        own = state['owner'][None, :] == jnp.arange(4)[:, None]
        near_council = self.distances(pos, state['council_pos'])[:, 0] <= 1
        ballot = state['ballot_remaining'] > 0
        propose = near_council & (state['fund'][0] >= 2) & ~ballot & (state['pending'] == 0) & (state['lease'][0] == 0)
        specials = jnp.stack([
            ~jnp.any(own, axis=1) & jnp.any(nearby & (state['owner'][None, :] < 0), axis=1),
            (state['materials'] >= 2) & jnp.any(nearby & own & ~state['built'][None, :], axis=1),
            jnp.any(nearby & own & state['built'][None, :] & (state['growth'][None, :] >= 3) & (state['harvests'][None, :] < 2), axis=1),
            near_council & (state['materials'] > 0) & (state['fund'][0] < 4),
            propose,
            jnp.full((4,), ballot, jnp.bool_), jnp.full((4,), ballot, jnp.bool_), jnp.full((4,), ballot, jnp.bool_),
        ], axis=1)
        mask = jnp.concatenate([jnp.ones((4, 1), jnp.bool_), movement, specials], axis=1)
        return jnp.where(state['done'], jnp.arange(13)[None, :] == 0, mask)

    def advance(self, key, state, actions):
        s = dict(state)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        pos = state['agent_pos'] + delta[jnp.minimum(actions, 4)] * (actions <= 4)[:, None]
        s['agent_pos'] = pos
        s['t'] = state['t'] + jnp.array(1, jnp.int32)
        nearby = self.distances(pos, state['plot_pos']) <= 1
        near_council = self.distances(pos, state['council_pos'])[:, 0] <= 1
        materials = state['materials']
        fund = state['fund'][0]
        owner = state['owner']
        built = state['built']
        claimed = jnp.zeros((4,), jnp.bool_)
        built_event = jnp.zeros((4,), jnp.bool_)
        donated = jnp.zeros((4,), jnp.bool_)

        # Claim and fund contention use ascending actor order, without extra materials.
        for i in range(4):
            available = nearby[i] & (owner < 0)
            plot = jnp.argmin(jnp.where(available, jnp.arange(4), 9))
            claim = (actions[i] == 5) & ~jnp.any(owner == i) & jnp.any(available)
            owner = owner.at[plot].set(jnp.where(claim, i, owner[plot]))
            claimed = claimed.at[plot].set(claimed[plot] | claim)
            buildable = nearby[i] & (owner == i) & ~built
            plot = jnp.argmin(jnp.where(buildable, jnp.arange(4), 9))
            construct = (actions[i] == 6) & (materials[i] >= 2) & jnp.any(buildable)
            built = built.at[plot].set(built[plot] | construct)
            built_event = built_event.at[plot].set(built_event[plot] | construct)
            materials = materials.at[i].add(-2 * construct.astype(jnp.int32))
            contribute = (actions[i] == 8) & near_council[i] & (materials[i] > 0) & (fund < 4)
            materials = materials.at[i].add(-contribute.astype(jnp.int32))
            fund = fund + contribute.astype(jnp.int32)
            donated = donated.at[i].set(contribute)
        s['materials'] = materials
        s['owner'] = owner
        s['built'] = built
        s['claimed'] = claimed
        s['built_event'] = built_event
        s['donated'] = donated

        # Four subsequent ballot ticks; votes persist until changed, deadline or next launch.
        active_ballot = state['ballot_remaining'] > 0
        votes = jnp.where(active_ballot & (actions == 10), 1, state['votes'])
        votes = jnp.where(active_ballot & (actions == 11), -1, votes)
        votes = jnp.where(active_ballot & (actions == 12), 0, votes)
        closing = state['ballot_remaining'] == 1
        ratified = closing & (jnp.sum(votes == 1) >= 3) & (fund >= 2)
        rejected = closing & ~ratified
        candidates = (actions == 9) & near_council
        launch = jnp.any(candidates) & ~active_ballot & (state['pending'] == 0) & (state['lease'][0] == 0) & (fund >= 2)
        proposer = jnp.argmin(jnp.where(candidates, jnp.arange(4), 9))
        votes = jnp.where(launch, (jnp.arange(4) == proposer).astype(jnp.int32), votes)
        s['votes'] = votes
        s['proposer'] = jnp.where(launch, proposer, state['proposer'])
        s['ballot_remaining'] = jnp.where(launch, 4, jnp.maximum(state['ballot_remaining'] - 1, 0))
        fund = fund - 2 * ratified.astype(jnp.int32)
        s['fund'] = jnp.array([fund], jnp.int32)

        # A passed charter opens only after two more transitions and expires after twelve.
        opened = state['pending'] == 1
        lease = jnp.where(opened, 12, jnp.maximum(state['lease'][0] - 1, 0))
        s['pending'] = jnp.where(ratified, 2, jnp.maximum(state['pending'] - 1, 0))
        s['lease'] = jnp.array([lease], jnp.int32)
        s['gate_open'] = jnp.array([lease > 0], jnp.bool_)
        s['launched'] = launch
        s['ratified'] = ratified
        s['rejected'] = rejected
        s['opened'] = opened
        s['expired'] = (state['lease'][0] == 1) & ~opened

        growth = jnp.minimum(state['growth'] + built.astype(jnp.int32), 3)
        safe_owner = jnp.maximum(owner, 0)
        harvest = built & (owner >= 0) & (growth >= 3) & (state['harvests'] < 2)
        harvest = harvest & (actions[safe_owner] == 7) & nearby[safe_owner, jnp.arange(4)]
        s['growth'] = jnp.where(harvest, 0, growth)
        s['harvests'] = state['harvests'] + harvest.astype(jnp.int32)
        s['harvest_event'] = harvest
        s['terminated'] = jnp.all(s['harvests'] >= 2)
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        paid = (owner[None, :] == jnp.arange(4)[:, None]) & harvest[None, :]
        rewards = jnp.sum(paid.astype(jnp.float32) * state['private_value'].astype(jnp.float32), axis=1)
        rewards = rewards - (state['materials'] - materials).astype(jnp.float32) * 0.02 + s['terminated'].astype(jnp.float32)
        return s, rewards.astype(jnp.float32)

    def observe(self, state):
        public = jnp.concatenate([
            state['t'][None] / 96.0, state['map'].reshape(-1) / 2.0,
            (state['agent_pos'] / jnp.array([10.0, 6.0])).reshape(-1), state['materials'] / 3.0,
            (state['plot_pos'] / jnp.array([10.0, 6.0])).reshape(-1), (state['owner'] + 1) / 4.0,
            state['built'].astype(jnp.float32), state['growth'] / 3.0, state['harvests'] / 2.0,
            (state['council_pos'] / jnp.array([10.0, 6.0])).reshape(-1),
            (state['gate_pos'] / jnp.array([10.0, 6.0])).reshape(-1),
            state['gate_open'].astype(jnp.float32), state['lease'] / 12.0, state['fund'] / 4.0,
            (state['ballot_remaining'] > 0)[None].astype(jnp.float32), (state['proposer'][None] + 1) / 4.0,
            state['ballot_remaining'][None] / 4.0, (state['votes'] + 1) / 2.0, state['pending'][None] / 2.0,
            jnp.stack([state['done'], state['terminated'], state['truncated']]).astype(jnp.float32),
        ])
        return jnp.concatenate([jnp.eye(4, dtype=jnp.float32), state['private_value'] / 3.0,
            jnp.broadcast_to(public[None, :], (4, 132))], axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions.astype(jnp.int32), 0, self.num_actions - 1)
        valid = (actions >= 0) & (actions < self.num_actions)
        valid = valid & self.legal_actions(state)[jnp.arange(4), clipped]
        proposed, rewards = self.advance(key, state, jnp.where(valid, clipped, 0))
        next_state = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), proposed, state)
        rewards = jnp.where(state['done'], jnp.zeros((4,), jnp.float32), rewards)
        dones = jnp.full((4,), next_state['done'], jnp.bool_)
        info = {'done_all': next_state['done'], 'terminated': next_state['terminated'], 'truncated': next_state['truncated']}
        return self.observe(next_state), next_state, rewards, dones, info
