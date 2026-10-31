# Signal Coast League

AUTHORED_NOT_EXECUTED. Current blind train-only initial attempt_00, no numerical or selection feedback. A=3; O=218; U=12; H=120.

Signal Coast League uses a territorial road board with three scout agents, three owned caravans, three home stores, three destinations, three finite-fuel watchtowers, nine protected stops and three moving hazards. Roads run along y=1,3,5 and connect vertically at x=1,6,11. Scouts can traverse the surrounding hinterland, while caravans follow only explicit cardinal commitments between those stops. Initial scout sight radii differ: 3,3,4.

A scout within one cell of its own stopped caravan can commit it to the next stop east, west, north or south. Horizontal segments are five cells and vertical segments two. There is no path planner or destination inference: the selected target is stored, and each subsequent tick takes one sign-of-displacement step until arrival. Mid-segment cancellation is unavailable. Old caravan cells cannot be entered or swapped; simultaneous target contention uses rotating role priority. A safe-stop destination can already be occupied, producing a real queue and potentially exposing the waiting caravan to danger. Scouts and caravans may share cells; their physical roles are distinct.

A watchtower action sends exactly the lowest-index hazard currently visible to that scout. It costs one unit from that tower's finite stock of four. Every recipient receives a stored position, horizontal direction, sender and source t for that hazard, with lifetime six. Existing lifetimes decrement at the next transition before new transmissions. A marker disappears at zero. A transmitted snapshot is sampled from the input state; hazards may move later in the same step. If multiple scouts send the same hazard, the later transaction in rotating order determines sender metadata; all paid their chosen signal cost. No text diplomacy is generated.

Actual hazards move one horizontal cell on resulting t divisible by three, reflecting between x=2 and x=10. A caravan on any protected stop is safe. Else a hazard within one Manhattan cell after motion destroys one carried supply, up to one per caravan per tick. Different lanes, temporary waiting, observation from scout positions and timely signals can change commitment choices. A message's age therefore matters: it conveys a past observation rather than granting remote live sight. Signals, driving, walking toward the next loading/unloading point and clearing another caravan's route compete for the same scout actions.

Each polity starts with two supplies in its caravan and two in its home bank. Loading transfers up to capacity two; delivery transfers only still-needed supplies at the owner's destination. Reward per role is -0.01 per active tick, plus 0.2 times all deliveries, plus its own delivered amount, minus 0.25 for its own actual cargo hit, plus 2 on joint success or minus 2 on natural impossibility. Signal tending has no separate reward and never creates supplies.

## Native controls

- 0: Wait
- 1: Scout north
- 2: Scout east
- 3: Scout south
- 4: Scout west
- 5: Tend current tower: send lowest-index currently visible hazard snapshot
- 6: Beside own stopped caravan: commit east to next protected stop
- 7: Beside own stopped caravan: commit west to next protected stop
- 8: Beside own stopped caravan: commit north to next protected stop
- 9: Beside own stopped caravan: commit south to next protected stop
- 10: Beside own caravan at home: load available supplies
- 11: Beside own caravan at destination: deliver needed cargo

Humans and policies must call the same native reset/raw_step/observe/legal_actions interface. This source package does not implement or verify a public adapter, browser controls or a separate dynamics approximation. Invalid/masked actions become wait. Movement rejects old occupied agent cells, including simultaneous swaps, then uses rotating priority (role - old_t) mod 3 for shared targets. Only caravan targets are automatic; movement follows the chosen axis and can stall at actual occupied road cells. Signal warnings are immutable snapshots between explicit replacements, not extrapolated animations.

## Information and terminal semantics

Partially observable. Static map, safe stops, destinations, towers/fuel, delivery totals, signal flags and phase are public. Scouts have radii 3,3,4. A role also sees hazards within one of its own caravan and always knows its own caravan/home; other scouts, caravans and homes require scout sight. Invisible fields are -1 with explicit masks. Warning entities show only their recipient's stored snapshot, direction, timestamp and remaining lifetime. A live warning never reveals the actual hazard's current state; actual hazard entity visibility remains local. Per-role blocked/loss flags are private observation fields and are not public scene events.

Success when all three delivery counts reach two. Failure when any polity's bank plus caravan cargo plus already delivered supplies falls below two, making its required delivery impossible. Natural causes take priority over H. Each nonterminal raw call advances t once; an already terminal input preserves all state, observation, clock and terminal causes, returns zero rewards/all dones true, and never resets. Terminal masks permit only wait. All three roles remain present until joint termination.

Reset uses only jax.random.bernoulli(key,0.5,(3,)) to choose each hazard's initial horizontal direction. Initial positions are fixed; every later transition is deterministic. The key is not stored.

## Invariants and evidence limits

For each polity, bank+cargo+delivered+lost equals four. Cargo is 0..2, delivered 0..2, and tower fuel 0..4. Caravan positions remain on the stated road network; committed targets are protected stops. Scouts and caravans separately have unique positions. Hazards remain on their lane with x in 2..10 and direction +/-1. Warning lifetimes stay 0..6 and active iff positive; warning coordinates/timestamps remain the transmitted values until a later explicit signal. Rendering hidden actual hazards from warning data is not permitted. These invariants and information rules are source declarations, not executed acceptance.

All state leaves are fixed-shape int32/bool arrays; signed identifiers are represented losslessly as int32 and the external reset key is not copied into state. Observations and rewards are float32. Declared observation bounds [-5,256] and reward bounds [-3, 6] are reasoned bounds, not measured extrema. Static AST/schema/shape bookkeeping cannot establish runtime compatibility, eager/JIT/vmap parity, solvability, meaningful effect size, successful play, learning or transfer.

## Literal source evidence

- Role-local actual hazard access: candidate.py:21, literal 's["hazard_visible"] = from_scout | from_caravan'.
- Recipient-limited warning entities: candidate.py:23, literal 's["warning_visible"] = (jnp.arange(9)[None, :] // 3 == jnp.arange(3)[:, None]) & s["warning_active"][None, :]'.
- Driver presence requirement: candidate.py:67, literal 'driver = jnp.sum(jnp.abs(s["agent_pos"] - s["caravan_pos"]), axis=1) <= 1'.
- Finite warning life decrement: candidate.py:110, literal 's["warning_ttl"] = jnp.maximum(0, s["warning_ttl"] - 1)'.
- Immutable snapshot copy: candidate.py:119, literal 's["warning_pos"] = s["warning_pos"].at[recipients].set(jnp.where(send, state["hazard_pos"][hazard], s["warning_pos"][recipients]))'.
- Physical commitment: candidate.py:126, literal 's["caravan_target"] = jnp.where(command[:, None], s["caravan_pos"] + direction, s["caravan_target"])'.
- Real route congestion: candidate.py:138, literal 's["blocked"] = moving & (old_occupied | contested)'.
- Native cargo damage: candidate.py:149, literal 's["hit"] = ~safe & danger & (s["cargo"] > 0)'.
- Supply impossibility terminal: candidate.py:152, literal 'failure = jnp.any(s["bank"] + s["cargo"] + s["delivered"] < 2)'.
- Raw absorption: candidate.py:166, literal 'return jax.lax.cond(state["done"], self.absorb, self.transition, (key, state, actions))'.

- visibility: candidate.py:12 through line 25.
- reset: candidate.py:27 through line 53.
- walking: candidate.py:55 through line 60.
- legal_actions: candidate.py:62 through line 78.
- role_observation: candidate.py:80 through line 95.
- observe: candidate.py:97 through line 98.
- transition: candidate.py:100 through line 159.
- absorb: candidate.py:161 through line 163.
- raw_step: candidate.py:165 through line 166.

## Blind scripts and attribution

Two original literal seed-zero scripts and one equal-length, single-actor-action paired hypothesis are in playthrough.json. They are unexecuted author hypotheses; timing follows the implemented mechanism and no seed search occurred. JaxMARL README supplied parallel agent-dictionary API context; Pgx supplied state/mask/terminal and external-key context; Craftax supplied JAX spatial-environment context; PureJaxRL supplied the distinction between algorithm implementations and game libraries. No upstream implementation, benchmark layout, learner or commercial asset was copied or imported. The supplied PPO baseline source was not read. Actual current input paths, bytes, hashes, displayed reference ranges and uses are in author_input_freeze.json. Historical IDs in generic copied requirements are background only. A separate explicit repair dispatch is required after this source is frozen.
