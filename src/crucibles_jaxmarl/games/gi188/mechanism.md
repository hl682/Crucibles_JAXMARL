# Canal Kitchen

Status: AUTHORED_NOT_EXECUTED. Prospective initial-window source attempt 0; no candidate import, rollout, training, seed search or learned-policy result.

The vessel is part of the production network. Keeping a bowl on one bank can stop the other bank from plating, while sending it occupies the only ferry that could carry a fresh ingredient.

## World, task and player interpretation

A 9 by 7 kitchen has a canal at x=4, the only footbridge at (4,3), and a ferry route between (3,4) and (5,4). Vegetable pantry, chopping board and sink are left; broth pantry, pot, assembly and service are right. Pantries each hold four units. Cooks start at (2,3), (5,4), (3,4); the last cook holds one clean bowl and the sink holds the other. There is no hidden procedural layout or random reset in this game; key arguments remain compatible with the native interface.

## Native controls and contextual feedback

Supply one integer action per role to raw_step. Any legal action sequence is supported; no script ID or scripted policy is consulted by the kernel. Directional actions remain selectable for able characters before joint completion; blocked moves spend the tick and stay in place. Interaction masks show the current prerequisites. Simultaneous requests can still contend for the same finite item, tile or station.

- 0: Wait. Advance bridge, ferry, cooking and freshness clocks.
- 1: Walk north. One cardinal tile; counters, canal and occupied cells block entry.
- 2: Walk east. One cardinal tile; the bridge admits entry only while open.
- 3: Walk south. One cardinal tile; counters, canal and occupied cells block entry.
- 4: Walk west. One cardinal tile; the bridge admits entry only while open.
- 5: Take raw ingredient. Empty hand beside a pantry takes one finite vegetable or broth portion.
- 6: Load preparation station. Raw vegetable goes to the adjacent empty chopping board for two ticks; raw broth goes to the adjacent empty pot for three.
- 7: Collect prepared ingredient. Empty hand beside board or pot takes its ready item, including an already burnt pot item.
- 8: Put ingredient on assembly tray. Beside assembly, place chopped vegetable or cooked broth into its empty matching tray slot.
- 9: Plate meal in clean bowl. Beside assembly with clean bowl and both ingredients, consume the tray contents and carry a plated meal.
- 10: Serve meal. Beside service counter with a meal, fulfill the earliest unresolved order; keep the returned dirty bowl.
- 11: Load cargo ferry. Beside a landed empty ferry, place your carried item aboard, preserving food age.
- 12: Launch loaded ferry. Beside a loaded landed ferry, send it to the opposite bank over four transitions.
- 13: Unload landed ferry. Beside a landed loaded ferry with empty hand, take its actual cargo and age.
- 14: Take clean bowl at sink. Beside sink with empty hand, take one of the two conserved bowls if clean.
- 15: Wash dirty bowl. Beside sink with dirty bowl and no washing underway, start two-tick washing.
- 16: Hand item to neighbor. Give item and freshness age to the lowest-index adjacent empty-handed cook.
- 17: Discard food. Discard raw/prepared/burnt food or a meal; discarded meals return a dirty bowl, so vessels are conserved.

Moves cannot enter another cook old cell, so waiting on the bridge can physically delay a partner; swaps and following into a just-vacated cell do not occur. Same-destination contention rotates with t modulo three. Station, ferry and bowl transactions then use the same rotating order with current-state rechecks. An individually legal claim can be unavailable after a prior claimant; it then leaves the relevant object unchanged.

## Exact transition, costs and completion

Directional steps and interactions each consume one tick. A processing timer is decremented after that tick transactions, so loading at duration two finishes after the loading and following transitions. Finished pot broth ages to eight and becomes already-accounted burnt food. Cooked broth and meals in hands or ferry age every transition and spoil at 12; a spoiled meal returns a dirty bowl. Tray broth also ages to 12, leaving any chopped vegetable in place. Plating resets meal age but consumes the tray ingredients. The ferry moves half a tile per transition, reaches the other landing after four including launch, and uses exact half-integer coordinates. Bridge phase is t modulo eight <4; closing prevents entry but does not destroy or trap a cook already on the deck. Each delivered meal gives +1 to every cook; an expired order costs every cook 0.2. There is no reward for raw pickup or processing.

## Information and visible scene

Map, bodies, held food, held ages, all station states, order status/deadlines, ferry cargo/age/position/remaining travel and bridge openness are in every role observation. Renderer groups bind actual station positions, ferry motion and gate state. The bridge map value identifies its permanent deck; the bridge_open entity flag determines whether its entrance gate is raised. Full-state ledgers add no policy advantage because all food losses are visible physical events.

Independent cooperative-kitchen art direction: cream stone paving, deep blue canal, terracotta and turquoise aprons, crates of greens, a timber chopping board, a copper broth pot, a ceramic sink and stacked white bowls. The ferry is a small wooden cargo skiff whose item is drawn aboard, moving at its actual half-tile positions. Stations visibly change from loaded to processing to ready/burnt; carried food sits in each cook hands. Original assets only; no copied Overcooked characters, layout or code.

The scene is complete declarative presentation data with exact source/manifest identity. It does not implement a browser, keyboard or animation engine. A later trusted human-play adapter must separate Enter game from Watch replay, pass deliberate joint actions into this unchanged kernel, show legal controls, and preserve role visibility. Human play is separate from formal experiments.

## Cross-agent mechanisms and alternative responses

A ferry launch immediately commits a shared transport slot and later creates another cook unload opportunity. A cook can instead carry a bowl over the gated bridge, leaving the ferry for perishable broth. Early vegetable stockpiling uses the nonperishable tray slot; just-in-time broth delivery reduces spoilage risk. Washing converts a temporarily unusable bowl into a teammate next plating opportunity. A stationary bridge cook blocks transport even if that cook receives no personal reward. Alternative routes are endogenous choices through the same world, not separate replay scripts.

## Declared invariants and implementation limits

Exactly two vessels exist across carried clean/meal/dirty item codes, ferry cargo, sink clean count and a positive washing timer. Food stock never replenishes. Per ingredient, count remaining raw stock, held raw/prepared food, board/pot in-process or edible output, assembly, edible meals, served meals and waste; already burnt item code eight has already entered waste and is not counted twice. Positions remain distinct on walkable cells. Ferry capacity is one; stations hold at most one job/item; order statuses only advance 0 to 1 or 2. The model deliberately uses deterministic travel and simple freshness clocks rather than hydrodynamics or physical tipping.

State leaves use fixed-shape int32, bool and, where declared, float32 arrays. Small signed fields are represented as int32; no PRNG-key surrogate is stored. Only reset uses the supplied key where explicitly stated. raw_step is deterministic for a given state/action and ignores its unused key. Natural termination has priority over the horizon. A terminal input preserves every state leaf and observation, returns zero float32 rewards and joint dones, and preserves terminated/truncated causes. No autoreset occurs in the kernel.

## Literal source evidence

- Literal canal and station geometry: candidate.py:12 (def reset).
- Bowl, ferry and station prerequisites: candidate.py:37 (def legal_actions).
- Movement admission on timed bridge: candidate.py:81 (free = (tile==0)).
- Delivery returns the same vessel: candidate.py:120 (c = jnp.where(deliver,7,c)).
- Ferry retains actual cargo age: candidate.py:123 (s["ferry_age"] = jnp.where(ship).
- Finite bowl circulation: candidate.py:152 (# Every vessel remains).
- Real perishable-food timing: candidate.py:166 (# Prepared broth and plated meals).
- Terminal absorption: candidate.py:193 (s = jax.tree.map).
- Required boundary signatures: candidate.py:12, candidate.py:61, candidate.py:37, candidate.py:69.

## Frozen author hypotheses

The ferry diagnostic sends the initially held clean bowl from the left dock to the waiting right-bank cook and unloads it. The vegetable diagnostic walks the left cook to its pantry, obtains raw vegetable, loads the board, waits and collects chopped vegetable. Neither script is a meal-completion proof. The paired intervention changes only launch versus wait after the same cargo loading; after four suffix transitions compare the right cook unload mask, which should differ because of actual ferry location.

playthrough.json contains literal joint integer vectors, seed 0 and UNEXECUTED_AUTHOR_HYPOTHESIS statuses. Seed 0 is a fixed author convention, not an observed good seed. Script legality and effects are reasoned from source only. No execution, score, success guarantee, effect magnitude, solvability, non-isomorphism or learning/transfer claim is made.

## Source and reference provenance

All game rules and kernel code were authored for this assigned game. No upstream implementation or commercial game asset was copied. The human-play and independent-family requirements are recorded with actual paths, bytes and SHA256 in author_input_freeze.json, together with every other closed input consumed.

- JaxMARL supplied README: parallel agent-keyed interfaces and __all__ for a later adapter. This package retains the assigned flat native kernel. https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md
- Pgx supplied README: explicit legal masks, immutable game state and caller-supplied randomness are integration concepts; no board-game mechanic was copied. https://github.com/sotetsuk/pgx/blob/main/README.md
- Craftax supplied README: block materials, consequential inventory and explicit raw/autoreset separation informed the category/integration discussion; no Craftax world or rule implementation is used. https://github.com/MichaelTMatthews/Craftax/blob/main/README.md
- PureJaxRL README and Craftax_Baselines ppo_rnn.py were read as end-to-end JAX/recurrent-reset integration context. They are algorithm implementations, not a game library or replacement for the frozen experimental learner. No algorithm code, weights, dependencies or reported scores enter this game. https://github.com/luchris429/purejaxrl/blob/main/README.md ; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py

This environment can supply a future cross-environment adaptation task only after independent hash-bound integration. Actual R0 interface checks, eager/JIT/vmap behavior, paired effects, human input, replay, visual fidelity, training, scientific comparisons and transfer remain unverified.
