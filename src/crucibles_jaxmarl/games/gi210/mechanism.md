# Lantern Grid

AUTHORED_NOT_EXECUTED. Original initial source attempt 00 for game-v3-00210. Requested author gpt-6-astra / max; provider measured identity, tokens and cost unknown.

The quarry has a wall with a central lift gate and longer routes around its north/south ends. Six ore blocks start east of the wall; two cable segments and two replacement fuses are shared finite objects. Every actor carries at most one object. The lamp starts on, the lift starts disconnected, and the sorter starts off. Moving a material, installing a cable, repairing a fuse, sorting ore and depositing it update its original object's holder/activity state.

The three branch demands are 1/3/2 with a common trunk capacity of 4. Switching on both industrial branches can therefore destroy the lift fuse; losing the light instead preserves capacity but narrows other crews' observations. After a trip, remaining connected demands receive supply together only if total charge is sufficient. A crank adds six charge at the cost of one fuel, never overflowing twelve. Switching and generation affect the next interval; current movement and sorting use input-state power. The three-tick manual gate and outer perimeter routes exchange labor or travel time for electricity. Partners can respond by coordinating switch schedules, carrying spare parts, waiting, using manual bypass, or detouring. These are source-defined alternatives, not observed strategies.

Interaction priority rotates with the clock. A branch interaction chooses the closest branch, with lowest-index distance ties. Pickup chooses lowest-index eligible material within one tile (not geometric nearest). A held item's position follows its holder; consuming it leaves its final physical position but disables activity.

## Native interface and source evidence

A/O/U/H = (3, 167, 14, 160). Humans and policies choose the same ordered native integer actions. Any subset of the three roles can be assigned to humans by the shared platform; this package adds no separate browser transition or controller restriction. Legal masks are contextual affordances, not a proof that an interaction will find an object. Out-of-range or mask-false actions become wait. Eligible interactions resolve in rotating actor order (t + k) mod 3; ties among eligible objects use lowest native object index. Movement rejects destinations occupied at the start of the tick and duplicate destinations; swaps and following into a simultaneously vacated cell therefore wait.

- candidate.py:L13-L36: reset.
- candidate.py:L38-L44: visibility.
- candidate.py:L46-L57: observe.
- candidate.py:L59-L71: legal_actions.
- candidate.py:L73-L142: advance.
- candidate.py:L144-L146: raw_step.

## Action legend

- 0: Wait.
- 1: Walk north.
- 2: Walk east.
- 3: Walk south.
- 4: Walk west.
- 5: Pick lowest-index eligible loose item within one tile; hands must be empty.
- 6: Drop held item on current tile.
- 7: Toggle closest branch switch within one tile.
- 8: Crank generator within one tile: one fuel gives six charge if charge <= 6.
- 9: Install held cable at closest unwired branch within one tile.
- 10: Repair closest blown branch with held fuse within one tile.
- 11: Hold manual lift bypass open for three ticks from within one tile.
- 12: Sort held ore beside powered sorter branch 2.
- 13: Deliver sorted held ore beside depot.

## Observations and scene

The map, clock, terminal causes, fixed circuit/facility positions and all circuit/fuel/charge/bypass/delivery indicators are public. Each crew sees itself and agents/items within Manhattan radius 1; radius expands to 4 if it is within four tiles of the powered lamp. Visibility is recomputed after each transition. Hidden agent/item records are -1 with a separate visibility bit. Inventory is shown only on visible entities. No hidden object data appears in event channels. All state leaves and exact observation slices are in manifest.json. Every visible attribute is native state; scenes introduce no hidden policy data. Public event channels contain only intentionally public values. Debug/full-state inspection must be separate from role view. Use original muted quarry blocks, brass circuit indicators and blue ore crates. The wall, lift and circuit objects bind native grid cells; no imported commercial art.

## End states, reward and invariants

Every actor receives (new deposits) - 0.01 - 0.1 times the current overload flag per nonterminal step. No reward is paid again after termination. Natural completion when all six ore blocks have been delivered. Otherwise truncate at tick 160. Natural termination has priority on the cap step. Already done input is absorbing with unchanged state, observation and clock; rewards become zero and terminal causes persist. No raw autoreset occurs. The declared bounds and invariants are source obligations, not observed numerical results. Exactly six original ore objects, two cables and two fuses exist. A material is either active on the floor/carried or inactive after use/deposit. One holder per item and at most one item per actor are intended invariants; installed or delivered items are not picked up again. Charge remains 0..12 and fuel 0..9.

## Hypotheses and provenance limits

Two literal scripts and one single actor/action paired intervention are supplied in playthrough.json; suffix lengths follow the implemented timers or locations. All have UNEXECUTED_AUTHOR_HYPOTHESIS status, and the seed is literal rather than searched. They do not certify legal trajectories, solvability or mechanism success. Prospective tests could vary fuel allocation, starting branch connectivity, item placement or partner switching policies under a separately frozen future design; none has been run. This source does not establish compatibility, numerical parity, strategic richness, learning, generalization or transfer.

The actual allowed-input inventory, full-byte hashes and excerpt extents are frozen in author_input_freeze.json. Current scope overrides historical IDs in copied generic requirements. Four supplied official README excerpts provided API context: JaxMARL parallel dictionaries, Pgx explicit-key/terminal concepts, Craftax spatial world/interface context, PureJaxRL single-file implementation philosophy. No benchmark implementation, layout, upstream code or assets were copied; the supplied PPO implementation was not read. Static author checks parse data/AST only and import only supplied trusted stdlib validators. Candidate/JAX/NumPy import, native calls, numerical replay, rendering, training, seed search and network/model calls were not performed.

## Literal rule evidence

candidate.py:L41: radius = jnp.where(lit, 4, 1)

candidate.py:L124: overload = jnp.sum(demand)>4

candidate.py:L131: n['charge'] = n['charge']-jnp.where(supplied,jnp.sum(demand,dtype=jnp.int32),0)

candidate.py:L108: cable = (a==9)&near&has&(n['kind'][h]==1)&~n['wired'][b]

candidate.py:L115: hand = (a==11)&(jnp.sum(jnp.abs(p-jnp.array([4,3],jnp.int32)))<=1)

candidate.py:L117: sort = (a==12)&has&(n['kind'][h]==0)&s['powered'][2]&(jnp.sum(jnp.abs(p-n['branch_pos'][2]))<=1)

candidate.py:L119: deliver = (a==13)&has&(n['kind'][h]==0)&n['sorted'][h]&(jnp.sum(jnp.abs(p-n['plant_pos'][1]))<=1)

candidate.py:L145: n,r = jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((3,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)

