# Floodgate Commons

Three controlled keepers share a 9 by 7 grid and a directed three-basin waterway. Positions are actual grid cells. Land includes plot cells; channel cells y=2, x=1..7 cannot be entered. Co-location, crossing and swaps are allowed, so there is no hidden movement-priority allocator. Basin i supplies plot i, and keeper i alone can irrigate it. All keepers can operate any reachable gate, repair any reachable plot, or give finite kits.

Actions are 0 wait, 1 north, 2 east, 3 south, 4 west, 5 toggle, 6 reinforce/repair, 7 give a kit, 8 irrigate. A keeper must be within Manhattan distance one of a gate or relevant plot. Toggle selects the lowest-index reachable gate. Each gate flips once for an odd number of simultaneous operators; an even count cancels. Reinforcement accepts keepers in ascending index, targets the lowest-index still-improvable nearby plot, spends exactly one kit, adds one capacity while bank<2, and otherwise restores one integrity up to four. Gift targets the nearest eligible peer, breaking distance ties by index. A receiver accepts only its lowest-index donor. Decisions use pre-gift stocks; cycles can exchange kits, and each keeper sends/receives at most one. Capacity is four kits per keeper, initially two.

After actions, tick t receives two upstream water units iff t mod 4 is zero. For each edge i, outgoing flux is min(pre-flux pool_i,2) if its gate is open, otherwise zero. Edges 0 and 1 connect to the next basin; edge 2 drains water out of the world. Incoming edge flux is added only after all outgoing fluxes are fixed. Storage then clips at 6+bank_i, with excess discarded. Each positive spill removes one plot integrity and sets its real inundation timer to three; otherwise the timer decreases by one. Inundation does not additionally block land movement or damage on later timer-only ticks. Irrigation follows flow and spill: a legal pre-step attempt can fail to consume if the basin was emptied by this tick's outgoing flow. One unit increments growth; growth three yields a harvest and resets growth. Completed two-harvest plots stop consuming.

The temporal dependency is literal resource redistribution: releasing water now can deprive the upstream keeper of an irrigation opportunity while enabling middle and then downstream consumption later. Closing the outlet conserves downstream water but can increase flood exposure. Reinforcing a neighbor can protect common survival at a private resource cost. Retention, staggered releases, outlet drainage, preventive banks and kit transfer are plausible alternatives; their value depends on partners' gate timing and private harvest priorities. No equilibrium or strategic richness result is asserted.

Reset samples each private harvest value uniformly from integer 1..3 using the supplied key. Rewards are own harvest value on a harvest, minus one for own basin spill, plus one to every keeper on joint harvest completion, minus one to every keeper on destruction. A repair action additionally pays 0.05 times its clipped net kit loss after simultaneous gifts; a received gift can offset this small reward charge, while the repair still consumes its accepted kit. Declared reward bounds are [-3,5]; observations are normalized to [0,1]. Natural termination is all plots at two harvests or any integrity zero. H=96 is a clock limit, not a success requirement.

Declared invariants include nonnegative water, kits, bank, integrity, crop progress, harvest counts and flood timers; capacities and intended ranges follow the source. Gate, basin and plot positions bind actual operations. Plot flood is a level, while toggle/repair/gift/irrigation/harvest events describe the current tick. Private_value is never a scene attribute or peer observation. The fixed map is common rule knowledge; every varying public field shown by the scene is represented by public state or its corresponding observation.

The literal paired hypothesis starts from seed 17401, changes only keeper 0's first action from wait to toggle, and keeps keeper 1 opening the middle gate and keeper 2 closing the outlet in both branches. Two identical wait suffix steps are intended to expose delayed downstream water/opportunity differences. This is an unexecuted prediction from the pre-flux rule; no effect size or successful trajectory is reported. Prospective variation in inflow timing, private values, storage or topology would need its own frozen protocol and runtime checks.

Status: AUTHORED_NOT_EXECUTED. All strategies, scripts and paired interventions are source-reasoned hypotheses. No successful replay, effect magnitude, learnability, reward gain, optimality, non-isomorphism or meta-RL transfer is claimed.

The source uses only JAX/JAX NumPy imports; it was never imported or called. The author ran only the supplied trusted stdlib AST whitelist and manifest/scene data validators, plus data-only hashes, state-key/action-range/one-action-difference checks. Shapes, dtypes, jit/vmap parity, numerical bounds, gameplay, rendering, legal script execution and terminal absorption require independent frozen-source integration.

All agents are controlled; there is no scripted teammate, pursuer, planner or oracle. Every step receives a joint action vector. Invalid/out-of-range/masked actions become wait. A live raw step increments t once. A terminal input preserves every state leaf and observation, returns zero rewards and true dones, and preserves causes. Natural completion takes priority over H=96 truncation. The outer wrapper must perform auto-reset.

Representation: counters, identifiers and inventories use int32; flags use bool; observations/rewards use float32. The external reset key is not duplicated in state. Stochasticity, where present, uses only that key. No design-required distribution was removed to satisfy the interface.

Provenance: only the 22 current closed inputs listed in author_input_freeze.json were read. The five official reference files were read as bytes for integrity only; their text/code was not semantically consulted, copied or executed. These are original game kernels, with no copied upstream code or incorporated upstream license text. Reference URLs remain input-bundle provenance. No prior candidates, selection/training outcomes, heldout inputs, network sources or external models were used. Adapters, real trajectories and animation belong to independent integration. New A/O/U contracts do not license reuse of old policy weights or two-player ranking statistics.

Literal source evidence:
- candidate.py:L84: `gate_choice = jnp.argmin(jnp.where(gate_distance <= 1, jnp.arange(3)[None, :], 9), axis=1)`
- candidate.py:L87: `s['toggled'] = (toggle_count % 2) == 1`
- candidate.py:L102: `bank = bank.at[target].add(raise_bank.astype(jnp.int32))`
- candidate.py:L116: `accepted = requests & (winner[None, :] == jnp.arange(3)[:, None])`
- candidate.py:L125: `available = state['water'].at[0].add(inflow)`
- candidate.py:L126: `flow = jnp.where(s['gate_open'], jnp.minimum(available, 2), 0)`
- candidate.py:L127: `arrived = jnp.concatenate([jnp.zeros((1,), jnp.int32), flow[:2]])`
- candidate.py:L129: `spill = jnp.maximum(pool - (6 + bank), 0)`
- candidate.py:L139: `irrigated = (actions == 8) & own_near & (water > 0) & (integrity > 0) & (state['harvests'] < 2)`
- candidate.py:L148: `success = jnp.all(s['harvests'] >= 2)`
- candidate.py:L149: `failure = jnp.any(integrity == 0)`
- candidate.py:L172: `jnp.eye(3, dtype=jnp.float32), state['private_value'][:, None] / 3.0,`
- candidate.py:L182: `next_state = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), proposed, state)`
