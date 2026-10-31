# Tideworks Yard

AUTHORED_NOT_EXECUTED. Current blind train-only initial attempt_00, no numerical or selection feedback. A=3; O=254; U=14; H=168.

Tideworks Yard is a block-building harbor on an 11 by 7 grid. The channel occupies x=4..6. The permanent northern crossing at y=0 is always supported; the southern causeway at y=6 is supported only in low tide. Five persistent pontoon identities, six crates and three controllable workers occupy the board. Three initially lashed sections make a middle bridge held by one anchor; two spare sections start ashore. No coordinate exists only to draw an unrelated scalar game.

A worker carries either one crate or one pontoon. Crate pickup/drop uses the same cell and lowest eligible crate index. A pontoon can be lifted from within one cell only when deployed, isolated, unanchored, and carrying neither a worker nor an active crate. Placement uses the east/west action and requires a free channel cell in rows 1..5. Direct lashings are a symmetric matrix restricted to touching deployed sections. A five-iteration transitive closure identifies actual connected components. The owner can reserve its section against peer entry or share it. This controls stepping onto it, not passive transport of somebody already aboard.

After movements and rotating-order transactions, t advances and the tide phase is recomputed. On t divisible by six, every wholly afloat unanchored component attempts a one-row current displacement: south at t mod 12=6, north at t mod 12=0. Bounds, another deployed pontoon or a nonriding worker can prevent the entire component's move. Components resolve by their lowest section index. Each successful move translates its workers and crates by exactly the same offset. Anchors, carried sections and ashore components do not drift. Connectivity is refreshed after geometry changes.

The causal peer dependency is direct: an adjacent worker releasing the bridge's only anchor changes another worker's future position, support and reachable crossing row. Anchoring, unlash-and-carry construction, private section use, shared bridge maintenance, and taking the permanent long detour consume different time and opportunity. A worker who prioritizes its own owner depot may withhold a section that would shorten another worker's route. These are implemented alternatives, not empirical strategy or learning claims.

All six crate identities remain loose, held, delivered or sunk. Only a matched owner's depot accepts a crate. Each active step gives every role -0.01; each delivered crate gives all roles 0.2 and its owner an additional 1. Natural completion gives all roles 2; natural failure gives all roles -2. Repeated construction or anchor toggling has no direct reward.

## Native controls

- 0: Wait
- 1: North
- 2: East
- 3: South
- 4: West
- 5: Pick/drop crate on own cell
- 6: Lift an adjacent unanchored isolated empty pontoon
- 7: Place held pontoon one cell east into channel
- 8: Place held pontoon one cell west into channel
- 9: Toggle adjacent section anchor
- 10: Lash adjacent section to all touching deployed sections
- 11: Unlash adjacent linked section from its neighbors
- 12: Owner toggles adjacent pontoon private/shared entry
- 13: Deliver carried crate at its owner's depot

Humans and policies must call the same native reset/raw_step/observe/legal_actions interface. This source package does not implement or verify a public adapter, browser controls or a separate dynamics approximation. Invalid/masked actions become wait. Movement rejects old occupied agent cells, including simultaneous swaps, then uses rotating priority (role - old_t) mod 3 for shared targets. Item and engineering actions are rechecked against earlier transactions in the rotating order. At most one anchor toggle per section is applied in a step; a second toggle request fizzles.

## Information and terminal semantics

Fully observable. The role one-hot and public obs_keys vector expose all map/tide state, worker cargo, pontoon positions/holders/owners/permissions/anchors/direct and transitive connectivity, all crate identities, owner depots and events. Deployed and carried sections are actual state entities. Inactive delivered/sunk crate coordinates stay in public accounting but their entity active mask is false. Scene information is drawn from these same leaves.

Success when all six crates are delivered. Failure when a worker lacks physical support after tide/current update or any crate sinks. Failure excludes success on the same step. Natural causes take priority over H. Each nonterminal raw call advances t once; an already terminal input preserves all state, observation, clock and terminal causes, returns zero rewards/all dones true, and never resets. Terminal masks permit only wait. All three roles remain present until joint termination.

Reset layout and all transitions are deterministic; supplied key arguments are accepted but unused.

## Invariants and evidence limits

Worker positions are unique and within [0..10,0..6]. There are exactly five pontoon and six crate identities. Each worker holds at most one total item. Direct links are symmetric and connect adjacent deployed sections only; connected is their reflexive/transitive closure. Drift preserves the geometry of each moved component and transports its actual load. Delivered and sunk crates are inactive, not deleted. Total crate identities across loose/held/delivered/sunk remains six; delivered counts match delivered statuses. Values and clocks are bounded by the fixed episode cap. These are source-reasoned invariants, not executed tests.

All state leaves are fixed-shape int32/bool arrays; signed identifiers are represented losslessly as int32 and the external reset key is not copied into state. Observations and rewards are float32. Declared observation bounds [-5,256] and reward bounds [-3, 6] are reasoned bounds, not measured extrema. Static AST/schema/shape bookkeeping cannot establish runtime compatibility, eager/JIT/vmap parity, solvability, meaningful effect size, successful play, learning or transfer.

## Literal source evidence

- Explicit connectivity closure: candidate.py:32, literal 'reach = reach | (reach[:, k, None] & reach[k, None, :])'.
- Physical support: candidate.py:64, literal 'return land | floating'.
- Ownership changes peer entry: candidate.py:72, literal 'forbidden = jnp.any(hits & ~permission[:, None, :], axis=2)'.
- Anchor changes component dynamics: candidate.py:173, literal 'drift = pulse & jnp.any(member) & ~jnp.any(member & s["anchored"]) & jnp.all(~member | (water & bounds & ~other)) & ~collision'.
- Real peer transport: candidate.py:176, literal 's["agent_pos"] = s["agent_pos"] + (riders & drift)[:, None].astype(jnp.int32) * shift'.
- Native flood loss: candidate.py:180, literal 's["soaked"] = ~self.support(s, s["agent_pos"])'.
- Conserved cargo terminal: candidate.py:184, literal 'failure = jnp.any(s["soaked"]) | jnp.any(s["crate_holder"] == -3)'.
- Raw absorption: candidate.py:198, literal 'return jax.lax.cond(state["done"], self.absorb, self.transition, (key, state, actions))'.

- refresh: candidate.py:17 through line 36.
- reset: candidate.py:38 through line 58.
- support: candidate.py:60 through line 64.
- destinations: candidate.py:66 through line 75.
- legal_actions: candidate.py:77 through line 100.
- observe: candidate.py:102 through line 104.
- transition: candidate.py:106 through line 191.
- absorb: candidate.py:193 through line 195.
- raw_step: candidate.py:197 through line 198.

## Blind scripts and attribution

Two original literal seed-zero scripts and one equal-length, single-actor-action paired hypothesis are in playthrough.json. They are unexecuted author hypotheses; timing follows the implemented mechanism and no seed search occurred. JaxMARL README supplied parallel agent-dictionary API context; Pgx supplied state/mask/terminal and external-key context; Craftax supplied JAX spatial-environment context; PureJaxRL supplied the distinction between algorithm implementations and game libraries. No upstream implementation, benchmark layout, learner or commercial asset was copied or imported. The supplied PPO baseline source was not read. Actual current input paths, bytes, hashes, displayed reference ranges and uses are in author_input_freeze.json. Historical IDs in generic copied requirements are background only. A separate explicit repair dispatch is required after this source is frozen.
