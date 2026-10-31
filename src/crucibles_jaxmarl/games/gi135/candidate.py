"""Nine Shards Council: original cooperative spatial mosaic; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 80
    num_actions = 19
    max_steps = 96

    def refresh(self, s):
        n = dict(s)
        n["fragment_pos"] = s["agent_pos"]
        n["hint_active"] = s["hint_value"] >= 0
        return n

    def reset(self, key):
        rows = jax.random.permutation(jax.random.fold_in(key, 0), 3)
        colours = jax.random.permutation(jax.random.fold_in(key, 1), 3)
        recipe = colours[(rows[:, None] + jnp.arange(3)[None, :]) % 3].astype(jnp.int32)
        board_pos = jnp.stack((2 + jnp.arange(9) % 3, 2 + jnp.arange(9) // 3), axis=1).astype(jnp.int32)
        tilemap = jnp.zeros((7, 7), jnp.int32).at[2:5, 2:5].set(1)
        s = {
            "t": jnp.array(0, jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tilemap": tilemap,
            "agent_pos": jnp.array([[1, 2], [5, 3], [3, 5]], jnp.int32),
            "agent_active": jnp.ones((3,), jnp.bool_), "agent_visibility": jnp.ones((3, 3), jnp.bool_),
            "hands": jnp.eye(3, dtype=jnp.int32),
            "selected": jnp.arange(3, dtype=jnp.int32),
            "ink": jnp.full((3,), 2, jnp.int32),
            "recipe": recipe,
            "fragment_pos": jnp.array([[1, 2], [5, 3], [3, 5]], jnp.int32),
            "fragment_active": jnp.ones((3,), jnp.bool_), "fragment_visibility": jnp.eye(3, dtype=jnp.bool_),
            "board_pos": board_pos,
            "board_active": jnp.ones((9,), jnp.bool_), "board_visibility": jnp.ones((3, 9), jnp.bool_),
            "board_colour": jnp.full((9,), -1, jnp.int32),
            "hint_pos": board_pos,
            "hint_active": jnp.zeros((9,), jnp.bool_), "hint_visibility": jnp.ones((3, 9), jnp.bool_),
            "hint_value": jnp.full((9,), -1, jnp.int32),
            "urn_pos": jnp.array([[3, 0]], jnp.int32),
            "urn_active": jnp.ones((1,), jnp.bool_), "urn_visibility": jnp.ones((3, 1), jnp.bool_),
            "supply": jnp.full((1, 3), 2, jnp.int32),
            "placed": jnp.zeros((9,), jnp.bool_),
            "transferred": jnp.zeros((3,), jnp.bool_),
            "revealed": jnp.zeros((9,), jnp.bool_),
        }
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(3)
        alive = ~s["done"]
        mask = jnp.zeros((3, 19), jnp.bool_).at[:, 0].set(True)
        mask = mask.at[:, 1:8].set(jnp.broadcast_to(alive, (3, 7)))
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        targets = s["agent_pos"][:, None, :] + delta[None, :, :]
        matches = jnp.all(targets[:, :, None, :] == s["board_pos"][None, None, :, :], axis=3)
        empty_target = jnp.any(matches & (s["board_colour"][None, None, :] < 0), axis=2)
        has = s["hands"][ids, s["selected"]] > 0
        mask = mask.at[:, 8:12].set(empty_target & has[:, None] & alive)
        adjacent = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2) == 1
        receiver = adjacent & (jnp.sum(s["hands"], axis=1)[None, :] < 3)
        mask = mask.at[:, 12].set(jnp.any(receiver, axis=1) & has & alive)
        near_urn = jnp.sum(jnp.abs(s["agent_pos"] - s["urn_pos"][0]), axis=1) <= 1
        room = jnp.sum(s["hands"], axis=1) < 3
        mask = mask.at[:, 13:16].set(near_urn[:, None] & room[:, None] & (s["supply"] > 0) & alive)
        unknown = s["hint_value"].reshape(3, 3) < 0
        mask = mask.at[:, 16:19].set(unknown & (s["ink"] > 0)[:, None] & alive)
        return mask

    def raw_step(self, key, s, actions):
        ids = jnp.arange(3)
        clipped = jnp.clip(actions, 0, 18)
        valid = (actions >= 0) & (actions < 19) & self.legal_actions(s)[ids, clipped]
        a = jnp.where(valid, clipped, 0)
        n = dict(s)
        nt = s["t"] + jnp.array(1, jnp.int32)
        n["t"] = nt
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        moving = (a >= 1) & (a <= 4)
        proposed = s["agent_pos"] + delta[jnp.where(moving, a, 0)]
        inside = jnp.all((proposed >= 0) & (proposed <= 6), axis=1)
        safe = jnp.clip(proposed, 0, 6)
        floor = s["tilemap"][safe[:, 1], safe[:, 0]] <= 1
        occupied = jnp.any(jnp.all(proposed[:, None, :] == s["agent_pos"][None, :, :], axis=2) &
                           ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2), axis=1) > 1
        p = jnp.where((moving & inside & floor & ~occupied & ~contested)[:, None], proposed, s["agent_pos"])
        n["agent_pos"] = p
        selected = jnp.where((a >= 5) & (a <= 7), a - 5, s["selected"]).astype(jnp.int32)
        hands, supply = s["hands"], s["supply"]
        board, hint, ink = s["board_colour"], s["hint_value"], s["ink"]
        placed = jnp.zeros((9,), jnp.bool_)
        transferred = jnp.zeros((3,), jnp.bool_)
        revealed = jnp.zeros((9,), jnp.bool_)
        # Public cyclic priority resolves simultaneous competition; no tile is duplicated.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            colour = selected[i]
            target = p[i] + delta[jnp.clip(a[i] - 7, 1, 4)]
            matches = jnp.all(s["board_pos"] == target, axis=1) & (board < 0)
            q = jnp.argmax(matches)
            place = (a[i] >= 8) & (a[i] <= 11) & (hands[i, colour] > 0) & jnp.any(matches)
            board = board.at[q].set(jnp.where(place, colour, board[q]))
            hands = hands.at[i, colour].add(-place.astype(jnp.int32))
            placed = placed.at[q].set(placed[q] | place)
            adjacent = jnp.sum(jnp.abs(p - p[i]), axis=1) == 1
            accept = adjacent & (jnp.sum(hands, axis=1) < 3)
            rank = (ids - i - 1) % 3
            recipient = jnp.argmin(jnp.where(accept, rank, 9))
            give = (a[i] == 12) & (hands[i, colour] > 0) & jnp.any(accept)
            hands = hands.at[i, colour].add(-give.astype(jnp.int32))
            hands = hands.at[recipient, colour].add(give.astype(jnp.int32))
            transferred = transferred.at[i].set(give)
            draw_colour = jnp.clip(a[i] - 13, 0, 2)
            near_urn = jnp.sum(jnp.abs(p[i] - s["urn_pos"][0])) <= 1
            draw = (a[i] >= 13) & (a[i] <= 15) & near_urn & (jnp.sum(hands[i]) < 3) & (supply[0, draw_colour] > 0)
            hands = hands.at[i, draw_colour].add(draw.astype(jnp.int32))
            supply = supply.at[0, draw_colour].add(-draw.astype(jnp.int32))
            column = jnp.clip(a[i] - 16, 0, 2)
            cell = 3 * i + column
            reveal = (a[i] >= 16) & (a[i] <= 18) & (ink[i] > 0) & (hint[cell] < 0)
            hint = hint.at[cell].set(jnp.where(reveal, s["recipe"][i, column], hint[cell]))
            ink = ink.at[i].add(-reveal.astype(jnp.int32))
            revealed = revealed.at[cell].set(revealed[cell] | reveal)
        n["selected"], n["hands"], n["supply"] = selected, hands, supply
        n["board_colour"], n["hint_value"], n["ink"] = board, hint, ink
        n["placed"], n["transferred"], n["revealed"] = placed, transferred, revealed
        natural = jnp.all(board >= 0)
        # No correctness signal is released before the final irreversible board is full.
        correct = jnp.sum(board == s["recipe"].reshape(9)).astype(jnp.float32)
        reward = jnp.full((3,), jnp.where(natural, 2.0 * correct / 9.0 - 1.0, 0.0), jnp.float32)
        n["terminated"] = natural
        n["truncated"] = (nt >= self.max_steps) & ~natural
        n["done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        n = jax.tree.map(lambda old, new: jnp.where(s["done"], old, new), s, n)
        reward = jnp.where(s["done"], jnp.zeros((3,), jnp.float32), reward)
        dones = jnp.full((3,), n["done"], jnp.bool_)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, reward, dones, info

    def observe(self, s):
        public = jnp.concatenate((
            s["t"][None] / 96.0, s["agent_pos"].reshape(-1) / 6.0,
            s["hands"].reshape(-1) / 3.0, s["selected"] / 2.0,
            s["supply"].reshape(-1) / 2.0, (s["board_colour"] + 1) / 3.0,
            (s["hint_value"] + 1) / 3.0, s["ink"] / 2.0,
        ))
        own = jnp.concatenate((s["recipe"] / 2.0, jnp.eye(3, dtype=jnp.float32)), axis=1)
        events = jnp.concatenate((s["placed"], s["transferred"], s["revealed"])).astype(jnp.float32)
        return jnp.concatenate((jnp.broadcast_to(public, (3, 43)), own,
                                jnp.broadcast_to(events, (3, 21)), jnp.zeros((3, 10))), axis=1).astype(jnp.float32)
