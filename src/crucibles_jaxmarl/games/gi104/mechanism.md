Orchard Watershed Commons — game-v3-00104

Initial window 0, attempt 00. A=3, O=164, U=12, H=128. Three stewards move in a 9 by 9 public commons. Their personal storehouses are (1,4), (4,1), (7,4); the refill spring is (4,4). Six actual fruit patches form a spatial network at (2,3), (4,2), (6,3), (6,5), (4,6), (2,5). Distinct patches are ecological neighbours exactly when their Manhattan separation is at most three.

A role can carry four fruits and two units of water. Every role wants to store six fruits at its own home; banking gives that role reward equal to the number deposited. A deposited fruit is no longer available as planting material or a gift. Completing all three quotas adds +2 to every role and naturally ends the episode. Exhaustion of all living roots and all carried planting fruit, without success, naturally ends play and adds -2 to every role. Otherwise step 128 truncates. Per-step rewards lie in [-2,6]. The rule is deliberately not a claim that every reachable ecological configuration is recoverable.

Actions and local ecology

Actions 0–4 are wait and cardinal movement. Actions 5, 6 and 7 address the nearest patch within one tile, with patch index breaking equal-distance ties. Harvest takes one fruit if the patch is rooted and nonempty and the bag is not full. Taking the patch's last fruit removes one root strength. Plant consumes one carried fruit and adds one root, one stock and one soil moisture, capped at four roots, five fruits and three moisture; it can revive a zero-root patch. A completely saturated patch accepts no planting and charges no fruit. Irrigate consumes one water and sets a rooted patch's moisture to three. It can spend water on an already wet patch; that waste is part of the literal rule. Fill action 8 refills to two beside the central spring.

Deposit action 9 works only at the acting role's own exact home cell and moves as many fruits as possible into the bank without exceeding six. Action 10 offers one fruit; an adjacent role choosing 11 can receive if its bag has room. The same gift removes one unit from the donor and adds one to the recipient; it is not a reward-producing duplication.

At each fourth transition, ecology uses a common post-action snapshot. A living, moist patch below stock five grows one fruit if it retains at least two fruits or has a neighbouring patch with at least two roots and two fruits. Thus harvesting a patch below two increases dependence on neighbours retaining a seed reservoir. On the same pulse soil decays by one, but a patch with a neighbour at moisture three receives a moisture floor of one. Growth uses the pre-propagation moisture, so a dry neighbour wetted this pulse can grow only on a later pulse. Irrigation therefore has a spatial and delayed external effect. Donor and recipient checks use the same snapshot, not an order-dependent within-pulse cascade.

Simultaneous movement resolves first. Patch actions then resolve in rotating role priority, starting with t mod 3, so two harvest attempts can contest the same last fruit and a prior planting can change a later harvest opportunity. Spring refills follow, then fruit gifts in the same rotating order, then home deposits, then ecological growth/propagation, then terminal checks. Receiver ties also use rotating role priority. Every bag capacity is checked at the moment of transfer.

Causal interactions and strategies

Immediate extraction gives inventory that can be privately banked. Retaining at least two fruits supports local renewal and neighbours; investing a carried fruit restores another steward's future supply, and irrigation can moisten neighbouring patches after a delay. A steward's restraint can determine whether a competing neighbour finds a seed-supporting patch or an empty, damaged patch. A gift can help a lagging partner complete the joint objective but costs the donor material that could otherwise be banked or invested.

One plausible division of labour reserves connected seed patches, assigns a spring runner to maintain moisture, and rotates harvest lanes. Another relies on local harvesters banking early, with a later donor planting or gifting to whichever partner remains short. The second can earn individual income sooner but risks loss of shared regeneration or leaving a partner without nearby supply. These are competing rule-based possibilities, not measured equilibrium or cooperation results. Altering the ecological radius, pulse interval, home locations, capacities or gift access could change partner-dependent strategies; those variants are not implemented in this fixed source.

The proposed main script demonstrates harvesting, a costly planting investment, irrigation, personal deposits, spring travel and a later refill/irrigation cycle. It is intentionally a partial episode, not a successful-completion claim. Separate gift and paired hypotheses isolate the recipient's willingness, competition for a final local fruit, and moisture propagated to another steward's patch four ticks after irrigation. No simulated trajectory or numerical ecology trace has been generated.

Information, scene and declared invariants

Every state relevant to decisions is public. The 164 integer features divided by 256 contain own identity, time and terminal causes, positions/inventories/banks, all six patch positions and ecological levels, spring and homes, actual 9x9 map, and actual per-transition events. Bounds are [0,1]. A dead patch remains an active physical planting site, with root strength zero; it is not removed cosmetically. The scene displays actual stock, roots and moisture as levels, and growth events only on a real successful pulse. Stored fruit is an agent attribute, not a fabricated animation counter.

Declared invariants include unique agent cells and nonnegative inventory, water, stores, roots, fruit and soil. Source-level upper bounds for independent checking are fruit bag four, water bag two, store six, patch fruit five, roots four, soil three, and t at most 128. Planting and gifting conserve carried material through explicit debits; ecological growth is the only repeated new-fruit source. A dry, underseeded network with positive roots can become stranded without satisfying the strict collapse predicate and then time out; no hidden rescue or fabricated terminal success is introduced.

Interface and evidence boundary

The only candidate imports are jax and jax.numpy. All state leaves are fixed-shape int32 or bool arrays; observations and rewards are float32. All masks describe permitted attempts: a legal action may fail because of simultaneous contention or an unmet interaction condition. Out-of-range actions and masked actions deterministically become action 0. Destination contention cancels every conflicting move; no agent may enter another agent's start-of-step cell, including swaps or following a vacated cell in the same tick. No role is removed before joint termination. At terminal states only wait is legal. A nonterminal raw call advances t once; natural termination takes priority over the horizon; terminated and truncated are mutually exclusive. On an already terminal input every state leaf and observation remains unchanged, the clock is frozen, reward is zero and all dones remain true. There is no raw auto-reset.

The fixed initial layout and rules are deterministic. reset(key) and raw_step(key, ...) accept the trusted key but consume no randomness; no PRNG state is duplicated or replaced. The design did not prescribe narrow signed integer widths, so int32 is an explicit boundary representation choice. Power-of-two observation scaling is intended to keep represented values exact, but no eager/JIT/vmap parity has been executed or accepted.

The manifest binds every state leaf and every observation column. Entity coordinates affect the actual rules, and scene.json only labels those bound objects. The public tile map is explicitly declared public. Full-state debug rendering must remain labelled separately from each role view. No JaxMARL adapter, animation, replay, training, benchmark, scientific acceptance or transfer claim is supplied by this author. The coordinator must supply its separate agent-keyed adapter and independently test legal masks, inactive-role and __all__ behavior without modifying this frozen kernel.

Community reference attribution

The supplied official JaxMARL README informed the distinction between parallel agent-keyed dictionaries and this flat trusted kernel: https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md . The supplied official Pgx README informed explicit external-key, legality and termination API considerations, not board-game mechanics or heldout layouts: https://github.com/sotetsuk/pgx/blob/main/README.md . The supplied official Craftax README provided broad inspiration for persistent spatial objects and game state, and for separating raw transitions from outer auto-reset: https://github.com/MichaelTMatthews/Craftax/blob/main/README.md . The supplied official PureJaxRL README describes end-to-end JAX training and single-file algorithm implementations, not a new game library: https://github.com/luchris429/purejaxrl/blob/main/README.md . No upstream code was copied. The supplied Craftax_Baselines_ppo_rnn.py was not read for code or used; its bytes are hashed only to verify the supplied reference manifest. No external network request, dependency installation, candidate import, JAX/NumPy numerical execution or model/API call was made by the source author.

All action scripts and paired interventions in playthrough.json are UNEXECUTED_AUTHOR_HYPOTHESIS. Only source-byte, stdlib AST/JSON and supplied data-schema checks were performed. Static screening does not establish executability, exact numerical parity, legal trajectories, solvability, strategic richness, learning or generalization. Failed, negative, missing or inconclusive results from later independent evaluation must be retained.

Literal source evidence

- Last-fruit ecological damage: candidate.py:86: `last_fruit = harvest & (n["patch_stock"][p] == 1)`
- Costly planting debit: candidate.py:94: `n["agent_fruit"] = n["agent_fruit"].at[i].add(-plant.astype(jnp.int32))`
- Partner-dependent fruit transfer: candidate.py:115: `gift = (a[i] == 10) & (n["agent_fruit"][i] > 0) & jnp.any(ready)`
- Spatial ecological neighbourhood: candidate.py:130: `adjacent = (separation > 0) & (separation <= 3)`
- Neighbour seed reservoir: candidate.py:132: `seed_support = jnp.any(adjacent & donor[None, :], axis=1)`
- Delayed moisture propagation: candidate.py:138: `new_soil = jnp.maximum(n["patch_soil"] - 1, wet_neighbour.astype(jnp.int32))`
- Strict exhaustion condition: candidate.py:142: `n["collapse"] = ~n["success"] & jnp.all(n["patch_roots"] == 0) & jnp.all(n["agent_fruit"] == 0)`
- Absorbing terminal state: candidate.py:170: `n = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), proposed, state)`

Method line index

- candidate.py:12-13: distances.
- candidate.py:15-46: reset.
- candidate.py:48-59: legal_actions.
- candidate.py:61-67: move.
- candidate.py:69-148: advance.
- candidate.py:150-162: observe.
- candidate.py:164-174: raw_step.
