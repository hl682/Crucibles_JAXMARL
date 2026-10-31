# Signal Bargain

Status: AUTHORED_NOT_EXECUTED. Initial blind attempt_00; train split. Declared A/O/U/H = (3, 108, 44, 60).

Three mobile traders occupy a 9 by 7 open grid. Each starts with two units of its indexed commodity; sources at (1,1),(7,1),(4,5) each hold six further units. The economy begins with exactly eight units of each commodity. Homes are (0,3),(8,3),(4,6), and markets (3,3),(5,3). Co-location and swaps are allowed. Spatial travel determines harvesting, meetings, delivery and opportunity cost under deadlines.

Reset samples private recipe index q from the supplied key. A recipe requires two units of q and one of (q+1) mod 3. No script seed was sampled or inspected by the author. Each role may deliver twice, with free inventory plus own escrow capped at four. Harvest removes one unit from the occupied source; clock-rotating role order settles contested remaining supply. Delivery consumes recipe goods and records per-commodity consumption. For each commodity, free inventories plus outstanding offered-commodity escrow plus source stock plus consumed must equal eight.

Actions 0-16 are literal in the manifest. For action 17+c, target=c//9, give=(c//3)%3, ask=c%3. Self offers and equal give/ask are masked. An offer needs an owned free unit, no pending offer, an adjacent recipient, and both roles within one cell of a common market. It commits one actual unit, stores proposer position and terms, and starts TTL five. Only proposer and recipient see terms. A partner can move away and invalidate a pre-legally selected new offer or acceptance.

Order is validation/movement, old expiry/rejection/cancellation refunds, old acceptance in proposer-ID order, new offers, harvest, delivery, role-closing refunds, signals, terminal evaluation. Old TTL decrements first; old TTL one expires before acceptance. Cancellation or recipient rejection defeats simultaneous acceptance. Acceptance exchanges escrow for one requested free unit only when endpoints remain adjacent and each is within one cell of the stored offer position. Each role chooses one action; there is no automatic or optimal matching. A proposer with an old offer can itself accept another contract; current inventories are checked at settlement. Closing either endpoint refunds its outstanding contract.

Signals select commodity 0,1,2, cost one reward and one of three private tokens, and may misrepresent the recipe. They alter observations of roles within Manhattan radius three at emission. Content lasts four ticks at the frozen broadcast position, so it does not remotely track a moving sender. Other trader positions/inventories and source stock are visible at radius two. Source locations, delivery counts, deadlines, active status, settlement and cancellation identities are public. Terms and recipes remain inaccessible to unauthorized roles. Scene attributes omit private recipes, signal budgets and escrow quantities outside addressed contract visibility.

Delivery at own home pays eighteen. An unfinished role loses three at its deadline 42,48,54 and becomes inactive; completed two-delivery roles become inactive earlier. Inactive roles can wait while others continue. All inactive ends naturally; horizon is 60. Rewards are bounded [-4,18]. Possible partner-dependent choices include honest signaling, bluffing, commitment, cancellation, recipient rejection, complementary exchange and self-sourcing a missing input. None is claimed optimal.

Source invariants: nonnegative stock/inventory/escrow, inventory plus escrow at most four, total conservation including consumed goods, escrow one exactly when an offer is live, TTL [0,5], at most two deliveries, and no post-deadline delivery. These await independent execution; private recipes and desired delivery success are not inferred from a chosen seed.


Terminal interface: a nonterminal raw transition advances scalar int32 t exactly once. Natural completion has priority at a coincident horizon. Truncated equals (t >= max_steps) and not terminated; done_all is their disjunction and equals episode_done. All agent dones share that value. A completed input preserves every state leaf, observation, clock and terminal cause, returning zero float32 reward. Legal masks retain action 0; inactive roles retain only action 0. Invalid IDs or masked actions deterministically become action 0. Raw-step never auto-resets.

Representation: fixed-shape int32/bool state, float32 observations and rewards. Small signed counters and IDs use int32. The supplied uint32 reset key is trusted trajectory provenance, not duplicated native state. Random traits, where present, use only that key. Subsequent transitions are deterministic. These are declared interface choices, not actual execution receipts.

Provenance and limits: AUTHORED_NOT_EXECUTED, initial blind attempt_00. The five supplied upstream references were read only as bytes for integrity checks, not semantically read or copied. No upstream implementation or license text is incorporated. Rules, native kernel and literal scripts are original to this authoring turn. Trusted static AST, manifest and scene data validators are the only checks performed. Candidate code was never imported or executed. JAX shape behavior, eager/jit/vmap parity, numerical bounds, causal effects, actual rendering, solvability, optimality, strategic richness, non-isomorphism, learning and meta-RL transfer remain unverified. No reward or success acceptance threshold is introduced.

Independent gpt-6.1-sol/high integration owns the JaxMARL adapter, sealed execution and real replay/animation. No CSD3, remote browser, model call, training, heldout evidence, prior source/result or unrestricted search was used. The final input freeze records all 22 actual-read inputs with paths, byte counts, hashes and use.

Playthroughs: every seed, prefix and suffix is an original literal prospective hypothesis. Source reasoning suggests legality, but no execution verified legality or outcomes. There was no seed search or numerical feedback. Each paired branch differs in exactly one scalar action; later actions are identical. Delay and intended differences explain a testable source dependency, not a measured effect or successful play.

## Literal source evidence

- candidate.py line 26: recipe = 2 * jax.nn.one_hot(priority, 3, dtype=jnp.int32) + jax.nn.one_hot((priority + 1) % 3, 3, dtype=jnp.int32)
- candidate.py line 19: n['offer_visible'] = (jnp.eye(3, dtype=jnp.bool_) | (jnp.arange(3)[:, None] == s['offer_target'][None, :])) & n['offer_active'][None, :]
- candidate.py line 21: n['signal_visible'] = s['signal_heard'] & n['signal_active'][None, :]
- candidate.py line 91: offers = (s['inventory'][:, give] > 0) & ~pending[:, None] & together[:, target] & s['agent_active'][target][None, :] & (jnp.arange(3)[:, None] != target[None, :]) & (give[None, :] != ask[None, :])
- candidate.py line 116: cancelled = (s['offer_ttl'] > 0) & ((a == 10) | (s['offer_ttl'] == 1) | (a[jnp.maximum(target, 0)] == 14 + jnp.arange(3)))
- candidate.py line 127: inv = inv.at[r, ask[i]].add(-unit).at[r, give[i]].add(unit).at[i, ask[i]].add(unit)
- candidate.py line 137: new = new & shared_market & (jnp.sum(jnp.abs(p - p[new_target]), axis=1) <= 1) & (inv[jnp.arange(3), new_give] > 0)
- candidate.py line 171: n['consumed'] = s['consumed'] + jnp.sum(used, axis=0, dtype=jnp.int32)
- candidate.py line 176: n['signal_heard'] = jnp.where(signal[None, :], heard_now, s['signal_heard'])
- candidate.py line 183: reward = 18 * deliver.astype(jnp.int32) - signal.astype(jnp.int32) - 3 * missed.astype(jnp.int32)
- candidate.py line 187: n, reward = jax.lax.cond(state['episode_done'], lambda _: (state, jnp.zeros((3,), jnp.float32)), lambda _: self.advance(state, actions), operand=None)

## Method locations

- refresh: candidate.py lines 12-22.
- reset: candidate.py lines 24-70.
- legal_actions: candidate.py lines 72-94.
- observe: candidate.py lines 96-106.
- advance: candidate.py lines 108-184.
- raw_step: candidate.py lines 186-189.
