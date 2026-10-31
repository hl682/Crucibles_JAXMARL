# Lantern Convoy

Three controlled explorers navigate an 11 by 7 public floor/wall grid. The central wall x=5, y=1..5 has gaps at y=2 and y=4; the outer rows provide longer alternatives. Co-location, swaps and crossing are permitted. Public radio tracking exposes explorer and cargo positions, inventories and cargo integrity; uncertainty concerns hazard locations/pulses, private cargo priorities and other roles' remembered observations. This is a discrete information/resource game, not an optical or transport simulator.

Actions are 0 wait, 1 north, 2 east, 3 south, 4 west, 5 toggle lantern, 6 pass lantern, 7 pickup/drop cargo, 8 pass cargo, 9 deliver, 10 relay. Cargo pickup/drop and delivery use exact cells. Pickup contention is resolved by ascending actor index and then lowest cargo index. Every actor carries at most one cargo. Lantern and cargo handoffs target the nearest adjacent eligible peer, then lowest peer index; a recipient accepts only the lowest-index donor. Sender and recipient eligibility is fixed before each matching phase. Lantern possession, remaining charge and switch setting move together. The two original lanterns are conserved; there is no charger or spawn mechanism.

At each transition movement precedes transfers. A switched-on held lantern with positive charge pays one battery and emits glow in that snapshot. The final paid unit still lights the snapshot even though remaining charge becomes zero and the switch is cleared. A role directly senses hazards at Manhattan distance at most one, or at distance at most four when any glowing lantern is within distance three of that hazard. Walls do not occlude this deliberately abstract overhead glow. Public tracking of a distant lamp does not itself reveal every hazard it lights: the observer-distance condition still applies.

Four native hazards have fixed positions and phase offsets. At the new clock, armed is ((t+phase) mod 4)<2. An armed contact removes one integrity from cargo currently carried by that explorer, including newly picked or handed cargo. Uncarried cargo is not damaged. Three integrity losses destroy a cargo and clear its holder; it cannot later deliver. Delivery then requires action 9, surviving cargo and its own exact destination. All cargo resolved as delivered/destroyed causes natural termination; the two outcome types remain explicit.

Direct sensing records the current armed value with capture timestamp t. A relay action offers each sample to every adjacent peer. The receiver takes a strictly newer available timestamp, breaking equally fresh sender choices by index; direct sensing and self-memory participate in the comparison. Sender offers are fixed before relay, so a message does not cascade across multiple hops in a tick. A forwarded old sample retains its original timestamp. The observation carries known/unknown and age, and the role renderer only binds the directly visible hazard state. Neither a remembered value nor a received stale report makes the current hazard sprite visible. Global contact and relay-receipt bookkeeping are declared state but deliberately omitted from scene events/attributes, preventing those debug channels from exposing peers' hidden current information.

The reset key samples off-diagonal cargo priorities in {1,2}; each explorer's own indexed cargo priority is three. Every delivery rewards each role by its own priority for that cargo, every destruction penalizes the same value, and the current light payer incurs 0.02. Thus all deliveries are useful, but private rankings and finite carrying/light capacity can produce competing priorities. Carrying a partner's cargo, giving away a charged lamp, splitting routes or preserving battery for a later chokepoint are plausible partner-dependent choices. Charge expenditure affects later observation opportunities; a handoff changes which moving position casts light and who pays. No preference, route or strategy has been evaluated.

Declared bounds are observation [0,1], reward [-10,10], battery 0..8, integrity 0..3 and nonnegative memory age for known reports. Unknown memory timestamps are -1 in state but masked in observations. Cargo holders are -1 or a controlled role index. Entity positions are native transition positions; cargo sprites follow holders, and inactive cargo corresponds to delivery or destruction. Contact effects on public cargo are visible through actual integrity changes; hidden hazard phase offsets are never scene attributes.

The paired relay hypothesis uses seed 17501. Its fixed prefix turns explorer 0's lantern on, turns it off, then moves that explorer beside explorer 2. The branch changes only explorer 0's relay action to wait. Identical waits follow. The intended distinction is whether explorer 2 acquires an aged report while the hazard remains outside its direct visibility; subsequent age should refer to capture time rather than relay time. These claims are source deductions awaiting execution. Prospective geometry, pulse, battery and priority variation would need separately frozen experiments.

Status: AUTHORED_NOT_EXECUTED. All strategies, scripts and paired interventions are source-reasoned hypotheses. No successful replay, effect magnitude, learnability, reward gain, optimality, non-isomorphism or meta-RL transfer is claimed.

The source uses only JAX/JAX NumPy imports; it was never imported or called. The author ran only the supplied trusted stdlib AST whitelist and manifest/scene data validators, plus data-only hashes, state-key/action-range/one-action-difference checks. Shapes, dtypes, jit/vmap parity, numerical bounds, gameplay, rendering, legal script execution and terminal absorption require independent frozen-source integration.

All agents are controlled; there is no scripted teammate, pursuer, planner or oracle. Every step receives a joint action vector. Invalid/out-of-range/masked actions become wait. A live raw step increments t once. A terminal input preserves every state leaf and observation, returns zero rewards and true dones, and preserves causes. Natural completion takes priority over H=96 truncation. The outer wrapper must perform auto-reset.

Representation: counters, identifiers and inventories use int32; flags use bool; observations/rewards use float32. The external reset key is not duplicated in state. Stochasticity, where present, uses only that key. No design-required distribution was removed to satisfy the interface.

Provenance: only the 22 current closed inputs listed in author_input_freeze.json were read. The five official reference files were read as bytes for integrity only; their text/code was not semantically consulted, copied or executed. These are original game kernels, with no copied upstream code or incorporated upstream license text. Reference URLs remain input-bundle provenance. No prior candidates, selection/training outcomes, heldout inputs, network sources or external models were used. Adapters, real trajectories and animation belong to independent integration. New A/O/U contracts do not license reuse of old policy weights or two-player ranking statistics.

Literal source evidence:
- candidate.py:L17: `priorities = jnp.where(jnp.eye(3, dtype=jnp.bool_), 3, priorities)`
- candidate.py:L61: `lit = jnp.any((distance <= 3) & state['glow'][:, None], axis=0)`
- candidate.py:L62: `visible = (distance <= 1) | ((distance <= 4) & lit[None, :])`
- candidate.py:L70: `offers = jnp.where(receive_from[:, :, None], stamp[None, :, :], -1)`
- candidate.py:L75: `s['memory_stamp'] = jnp.where(newer, best_stamp, stamp)`
- candidate.py:L121: `matching = self.matching(pos, (actions == 6) & state['lantern'], ~state['lantern'])`
- candidate.py:L129: `glow = lantern & lamp_on & (battery > 0)`
- candidate.py:L130: `s['battery'] = battery - glow.astype(jnp.int32)`
- candidate.py:L164: `armed = ((s['t'] + state['hazard_phase']) % 4) < 2`
- candidate.py:L166: `damaged = (holder >= 0) & contact[jnp.maximum(holder, 0)] & state['cargo_active']`
- candidate.py:L184: `s['terminated'] = jnp.all(delivered | (integrity == 0))`
- candidate.py:L225: `age = jnp.where(known, state['t'] - state['memory_stamp'], 0) / 96.0`
- candidate.py:L223: `jnp.where(visible[:, :, None], direct_values[None, :, :], 0.0)], axis=2).reshape(3, -1)`
- candidate.py:L240: `next_state = jax.tree.map(lambda new, old: jnp.where(state['done'], old, new), proposed, state)`
