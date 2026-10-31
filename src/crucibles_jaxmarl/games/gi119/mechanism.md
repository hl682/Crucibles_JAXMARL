# Circuit crews with coupled power paths

AUTHORED_NOT_EXECUTED. This document describes source rules and author hypotheses; it is not a validation, animation, successful play, learning or transfer receipt.

## Task and state transition order

A 9 by 9 public map has a three-cell barrier at x=5 and a powered gate at (5,4), with upper and lower walking alternatives. Seven electrical nodes and eight selectable edges form branching routes. Every socket is a real installation location. Six cables, two portable switches and two batteries are individual objects, each loose, carried by exactly one role, or installed. Roles hold at most one part. Cables and switches occupy separate slot categories, and a switch may precede its cable. An unswitched cable conducts; an open installed switch blocks it.

Movement precedes timer decrement and cyclic role transactions for pickup, drop, specialty install/retrieval, toggles, handoff, pedaling and facility-mode changes. Seven fixed closure passes derive component connectivity. Each component totals available battery count and device demand. Overload removes component power, spends up to two charge per loaded battery and sets trip=3; normal demand spends one. Zero-demand components draw nothing. The pre-discharge supply is saved in power_delivered and can power this tick repair even if the last charge is consumed. After repair and completed-device mode changes, the network is recomputed to set next-state gate/tool availability.

Any nearby role can service a powered work-mode facility. The specialist contributes two units, others one, each facility capped at four. Shared reward is total service increment divided by eight. Closing a circuit or displaying a cooperation label earns no reward. Both completed facilities naturally end the task without an additional completion bonus.

## Actions and inactive-role semantics

- 0: Wait.
- 1: Move east.
- 2: Move west.
- 3: Move south.
- 4: Move north.
- 5: Pick lowest-id loose part within distance one, or drop held part.
- 6: Install held part of own specialty at an eligible nearby socket or source pad; if empty-handed, retrieve a nearby own-specialty installed part.
- 7: Toggle lowest-id nearby installed switch or battery.
- 8: Service nearest facility within distance one if powered and in work mode.
- 9: Courier pedal-recharges a carried battery near source pad 0 by one charge.
- 10: Hand held part to lowest-id adjacent empty-handed teammate.
- 11: Toggle nearest unfinished facility between idle and work mode within distance one.

All twelve intent actions are legal before joint termination. Missing position, specialty, inventory or power prerequisites cause deterministic no-ops. Malformed indices map to action 0; only 0 is legal after termination. Simultaneous movement blocks input occupied cells, walls and closed gates, cancels duplicate destinations, and blocks swaps. Object transactions use cyclic priority starting at t mod 3; ties use lowest ids except nearest eligible socket selection. A received part may be used by a later role turn that tick. Movement uses current gate power; newly installed power supplies tools now and permits gate movement from the next tick.

## Information and rendering

All terrain, crew/item positions, modes, carriers, slots, charge, trips, connectivity, node endpoints, power, loads, device modes, progress and gate state are public flat-observation fields. Own specialty and service rate are explicit. All rendered entities are public; static role ordering and capacities are task definitions.

All entity/event visibility arrays are true. Per-step events are separate from persistent node power and overload levels. power_delivered describes the supply before charge consumption, while node_powered is the post-discharge condition.

Part positions follow actual carriers and installed sockets. The generic tool glyph uses object_kind to distinguish cables, switches and batteries. Socket endpoint ids identify the actual graph; present/closed/powered flags are distinct physical states. Node demand/power, facility progress and the gate open flag bind actual leaves. power_delivered is transition supply; node_powered is current post-discharge supply. A replay must not merge them.

## Native observation columns

- [0, 3): Own crew specialty one-hot identity (self).
- [3, 4): Own powered service rate divided by 2 (self).
- [4, 85): Public 9 by 9 terrain in yx row-major order divided by 2 (public).
- [85, 91): Three crew xy positions divided by 8 (public).
- [91, 94): Held part ids divided by 9; negative means empty (public).
- [94, 97): Crew tool-power flags (public).
- [97, 117): Ten physical part xy positions divided by 8 (public).
- [117, 127): Part kinds divided by 2: cable, switch, battery (public).
- [127, 137): Part modes divided by 2: loose, carried, installed (public).
- [137, 147): Carrier ids divided by 2; -0.5 means not carried (public).
- [147, 157): Installed slot ids divided by 7; negative means uninstalled; cable/switch slots are edge ids, battery slots node ids (public).
- [157, 167): Switch closed and battery enabled flags; inert true values on cables (public).
- [167, 177): Charge divided by 12; only batteries store charge (public).
- [177, 187): Battery trip ticks divided by 3; other parts zero (public).
- [187, 195): Cable-present flags in edge order (public).
- [195, 203): Conducting-edge flags (public).
- [203, 211): Energized-edge flags (public).
- [211, 260): Seven by seven reflexive transitive connectivity matrix, row-major (public).
- [260, 267): Current powered-node flags (public).
- [267, 274): Current overloaded-node flags (public).
- [274, 281): Local node demand divided by 2 (public).
- [281, 288): Connected component load divided by 5 (public).
- [288, 290): Facility progress divided by 4 (public).
- [290, 292): Facility work-mode flags encoded as integers (public).
- [292, 299): Power delivered before battery discharge in the last transition (public).
- [299, 300): Current gate-open flag (public).
- [300, 316): Eight socket xy positions divided by 8 (public).
- [316, 330): Seven node xy positions divided by 8 (public).
- [330, 346): Eight endpoint node-id pairs divided by 6 (public).
- [346, 350): Facility xy positions divided by 8 (public).
- [350, 352): Gate xy position divided by 8 (public).
- [352, 355): Clock/112, total energy created/136, total energy spent/136 (public).

## Invariants and conservation

Object ids/kinds are fixed: exactly six cables, two switches and two batteries. Operations move an object between mode/carrier/slot states without spawning or clipping inventory. Same-kind installed edge slots and multiple carried items are excluded by admission checks. For Q=sum(charge), E=energy_spent, C=energy_created, reset is Q=24,E=0,C=24. Pedaling adds one to Q and C; use subtracts exactly used from Q and adds it to E. Thus Q+E=C. Charge stays 0..12; one courier can add at most one each tick, so C<=136. Trips are 0..3, progress 0..4 and modes 0..1. Occupancy checks preserve distinct crew positions. Graph power is computed from actual cables, switches, batteries and loads. These are algebraic source assertions, not numerical validation results.

## Partner-dependent strategies and bounded variation

Build isolated upper/lower circuits, route through the gate, or reclaim a cable after one facility finishes. Combine batteries to support concurrent loads or isolate one demand and conserve charge. Opening a switch or idling a device can keep a partner powered; closing too much with one battery can remove both partners access. Handing parts to specialists trades walking time against service speed. Pedaling competes with battery delivery. Changes in topology, capacity, source-pad positions or specialties would change these choices; no variants, strategic richness or policy transfer have been measured.

## Terminal, RNG and numeric representation contract

A nonterminal call advances scalar int32 t once. Natural termination takes priority at H: truncated=(t>=H)&~terminated; done=terminated|truncated. An already-done input returns the same entire state and observation, zero float32[A] rewards, all dones true and the preserved causes. No raw reset occurs. All roles continue to receive the joint done flag; an arrived or stunned role is not independently terminated.
Reset and step are deterministic and ignore supplied keys. The declared replay seed/key schedule remains exact provenance and must not be silently remapped. No process-global randomness exists.
The author declares seed 119 with reset_key=jax.random.PRNGKey(seed) and raw_step_key=jax.random.fold_in(reset_key, step_index+1), step_index starting at zero. The same keys are required for both intervention branches, including suffix steps. This schedule has not been executed.
All signed integer state uses int32, boolean state uses bool, and the observation/reward boundary is float32. No smaller signed integer design field or stored reset key is required. Probability draws use only the supplied explicit key. Token counts, compute cost and elapsed model-compute cost are unknown.

## Literal source evidence

- Cables create conducting opportunities: candidate.py:17; literal source: `present = jnp.any(cable[:, None] & (s['slot'][:, None] == edges[None, :]), axis=0)`.
- Fixed-pass graph closure: candidate.py:25; literal source: `reach = reach | (reach[:, k:k + 1] & reach[k:k + 1, :])`.
- Component overload: candidate.py:32; literal source: `overload = (supply > 0) & (load > 3 * supply)`.
- Specialty admission: candidate.py:161; literal source: `install = (a[i] == 6) & has_item & (kind == s['role'][i]) & place_ok`.
- Physical handoff: candidate.py:182; literal source: `transfer = (a[i] == 10) & has_item & jnp.any(neighbors)`.
- Energy ledger: candidate.py:201; literal source: `s['energy_spent'] = s['energy_spent'] + jnp.sum(used, dtype=jnp.int32)`.
- Power gates work: candidate.py:207; literal source: `work = (a[i] == 8) & (distance[facility] <= 1) & powered[facility + 4] & (s['device_mode'][facility] == 1)`.
- Power gates movement: candidate.py:126; literal source: `free = (tile == 0) | ((tile == 2) & s['gate_open'][0])`.

## Method line map

- network: candidate.py:12-34.
- decorate: candidate.py:36-54.
- reset: candidate.py:56-101.
- observe: candidate.py:103-115.
- legal_actions: candidate.py:117-118.
- advance: candidate.py:120-220.
- absorb: candidate.py:222-224.
- raw_step: candidate.py:226-228.

## Reference attribution

- JaxMARL_README.md: Reviewed the copied parallel agent-keyed API description and reset/step call sites. The native author kernel remains a distinct interface requiring a trusted adapter. Supplied SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; origin https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md.
- Pgx_README.md: Reviewed the copied legal-action-mask and explicit-key board-game API discussion. No board rule, implementation, layout or evaluation was copied. Supplied SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; origin https://github.com/sotetsuk/pgx/blob/main/README.md.
- PureJaxRL_README.md: Reviewed the copied description of a JAX-native PPO training pipeline. This is algorithm integration background, not a game library and not a learner replacement. Supplied SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; origin https://github.com/luchris429/purejaxrl/blob/main/README.md.
- Craftax_Baselines_ppo_rnn.py: Read-only search of reset/step/scan call sites in the supplied recurrent PPO integration code. No upstream source was copied or executed. Supplied SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; origin https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py.
- Craftax_README.md: Reviewed copied JAX spatial game, gymnax-interface and optional auto-reset descriptions. Original mechanics are authored here; raw terminal handling follows the assigned contract. Supplied SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; origin https://github.com/MichaelTMatthews/Craftax/blob/main/README.md.

No upstream game code is copied. JaxMARL API ideas, Pgx board APIs, Craftax game/environment ideas and PureJaxRL/Craftax PPO algorithm integration are distinct sources of conceptual context. No literature claim is treated as evidence that this game works.

## Acceptance limits

Only the supplied stdlib AST whitelist and data-only manifest/scene validators are used. Static checks do not test runtime array shapes, eager/JIT/vmap parity, terminal behavior, conservation over rollouts, paired causal outcomes, solvability, rich strategy, trainability, animation, optimality, non-isomorphism or meta-learning. Source authoring alone is not community delivery. The coordinator must retain independent dispatch provenance and perform any later numeric work only under its separately authorized, hash-bound execution boundary.
All scripts and paired branches are UNEXECUTED_AUTHOR_HYPOTHESIS. Negative, failed and indeterminate later outcomes must remain visible; no performance threshold or scientific acceptance is introduced.
