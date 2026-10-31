# Night Market Crew

Status: AUTHORED_NOT_EXECUTED. Prospective initial-window source attempt 0; no candidate import, rollout, training, seed search or learned-policy result.

The queue is shared, but order interests differ. A vendor who lights the skewer grill can leave a partner noodle cooker waiting for heat; a washed utensil can restore another stall production at the cost of its recipe water.

## World, task and player interpretation

The 9 by 8 night market contains three finite pantries, three distinct preparation stalls, a central wash counter, shared boiler, finite well, coal crate and service counter. Vendors start beside their skewer, noodle and tea stalls, each with matching raw ingredient and one clean utensil. Pantries contain four more of each ingredient. Initial resources are heat four, water pool four, water reserve twelve and coal six. All recipes and stations are usable by any vendor; roles describe starting positions only.

## Native controls and contextual feedback

Supply one integer action per role to raw_step. Any legal action sequence is supported; no script ID or scripted policy is consulted by the kernel. Directional actions remain selectable for able characters before joint completion; blocked moves spend the tick and stay in place. Interaction masks show the current prerequisites. Simultaneous requests can still contend for the same finite item, tile or station.

- 0: Wait. Let cooking, washing, cooling, boiler loss and queues advance.
- 1: Walk north. Move one cardinal cell; fixtures and occupied cells block.
- 2: Walk east. Move one cardinal cell; fixtures and occupied cells block.
- 3: Walk south. Move one cardinal cell; fixtures and occupied cells block.
- 4: Walk west. Move one cardinal cell; fixtures and occupied cells block.
- 5: Take pantry ingredient. Beside a pantry with empty hand, take one finite raw item; lowest eligible pantry index breaks ties.
- 6: Start matching stall recipe. Beside the matching idle stall with raw item and clean utensil, consume the recipe heat/water, start processing, and dirty your utensil.
- 7: Collect finished dish. Beside a stall with an empty hand, take its hot or cold dish and current age.
- 8: Serve matching order. Beside central counter with hot/cold food, fulfill the earliest open order of that dish type.
- 9: Take one coal. At the coal crate with empty hand, take one of six initial coal units.
- 10: Stoke shared boiler. At the boiler with coal, add four shared heat if current heat is at most four.
- 11: Pump finite well water. At the well with pool at most four, transfer up to two finite reserve units into the shared water pool.
- 12: Wash dirty utensil. At the washing counter with a dirty utensil, consume one water and start two-tick washing if idle.
- 13: Take clean utensil. At the washing counter with no utensil, take one available clean utensil.
- 14: Pass carried item. Give the item and freshness age to the lowest-index adjacent empty-handed vendor.
- 15: Pass utensil. Give your clean or dirty utensil to the lowest-index adjacent vendor without one.
- 16: Discard food. Remove raw, hot or cold food and record the ingredient loss. Coal is not discarded by this action.

Movement uses conservative occupancy: another vendor old tile blocks entry, and ties for a free destination rotate with t modulo three. Interactive claims are then processed in the same rotating order, each rechecking shared resource and station state. This permits two vendors to have individually legal recipe starts even when their combined heat demand exceeds supply. The later request is visibly denied, consumes no ingredients or tool, and can be reconsidered next tick. Per-role denied flags also show blocked movement and invalid requests.

## Exact transition, costs and completion

Recipe costs (heat, water, processing ticks) are skewers (3,0,3), noodles (2,2,4), tea (1,1,2). The loading tick is the first processing transition. Output begins at age zero, then food age continues across stall, hand and handoff; skewers cool at seven, noodles at six and tea at ten. Cooling changes hot codes 4..6 to cold codes 7..9, preserving the dish and ingredient count. Hot service pays every vendor 0.5 and adds 0.4 to the server if it matches their private priority; cold service pays 0.15 and a matching bonus 0.1. Expired orders cost all vendors 0.2. Boiler heat falls by one every fourth transition. Coal stoking adds four heat up to eight. Pumping moves at most two reserve water into a pool with capacity six. Washing consumes one water and temporarily holds one dirty utensil before adding it to the common clean rack.

## Information and visible scene

Every physical stock, hand, utensil, process timer, cooled/hot dish, queue and blocked-request flag is public in both observations and rendering descriptors. The reset key only permutes private priority dish types across roles. A role view may display its own priority beside its name, but another role priority is not present in public entity attributes. Tool age is not invented; the implemented tool state is none/clean/dirty and food age belongs to actual dishes.

Independent cooperative-kitchen art direction: a warm lantern night market with distinct skewer grill, noodle pot and tea brewer, striped cloth stall roofs, produce baskets, metal utensils, copper boiler and blue well header. Aproned vendors visibly carry raw bundles, steaming plated dishes, cooled dishes or coal. Dirty and clean tools have different surface treatment. Fixture status binds stock, timer, ready food, water or heat at each literal position; counters annotate those objects rather than replacing the market with a dashboard. All artwork is original and category-inspired.

The scene is complete declarative presentation data with exact source/manifest identity. It does not implement a browser, keyboard or animation engine. A later trusted human-play adapter must separate Enter game from Watch replay, pass deliberate joint actions into this unchanged kernel, show legal controls, and preserve role visibility. Human play is separate from formal experiments.

## Cross-agent mechanisms and alternative responses

A skewer request can consume three of the four initial heat and prevent a simultaneously requested noodle recipe. A vendor can instead yield heat to noodles and tea, trade tools, transport coal, pump water or wash on behalf of another stall. Washing competes with noodle and tea water use, making maintenance timing part of the food queue. Cold delivery offers a lower-value fallback when a handoff route is delayed; waiting for a privately preferred hot order can hurt the shared queue. These resource and opportunity effects are direct game rules, not inferred from a similarity of rendered characters.

## Declared invariants and implementation limits

Three utensils are conserved across positive held tool states, the clean rack and a positive wash timer. Water pool + reserve + water_used equals sixteen. Coal in crate + held coal + coal_burned equals six. For each dish type, pantry stock plus initial/raw carried units, active cooking job, ready/held hot or cold dishes, served_kind and waste conserves five ingredients. Shared resources cannot become negative because every transaction rechecks availability. Heat lies in 0..8, water pool 0..6, reserve 0..12, wash timer 0..2, and all vendors occupy distinct walkable cells. Cooling never fabricates a new ingredient. The world uses global shared plumbing and boiler supply, not a simulated pipe-pressure network.

State leaves use fixed-shape int32, bool and, where declared, float32 arrays. Small signed fields are represented as int32; no PRNG-key surrogate is stored. Only reset uses the supplied key where explicitly stated. raw_step is deterministic for a given state/action and ignores its unused key. Natural termination has priority over the horizon. A terminal input preserves every state leaf and observation, returns zero float32 rewards and joint dones, and preserves terminated/truncated causes. No autoreset occurs in the kernel.

## Literal source evidence

- Literal fixture geometry and finite starting resources: candidate.py:12 (def reset).
- Shared heat and water admission: candidate.py:44 (recipe =).
- Visible contention denial: candidate.py:93 (now_legal =).
- Recipe consumption changes partner availability: candidate.py:106 (s["heat"] = s["heat"]-heat_cost).
- Tool cleaning and water use: candidate.py:141 (wash = act==12).
- Temporal hot-to-cold conversion: candidate.py:172 (cooled =).
- Different interests over delivered dishes: candidate.py:128 (reward = reward.at[i].add).
- Absorbing terminal state: candidate.py:194 (s = jax.tree.map).
- Required boundary signatures: candidate.py:12, candidate.py:63, candidate.py:37, candidate.py:71.

## Frozen author hypotheses

The staggered noodle-and-tea diagnostic starts those two compatible recipes, collects tea, walks it to central service and delivers it. The contention diagnostic simultaneously requests skewers and noodles. The paired hypothesis changes only agent_0 initial start to wait while agent_1 still requests noodles, predicting a different agent_1 processing opportunity, carried ingredient, clean/dirty tool and denied flag. It makes no claim about accumulated return or a learned strategy.

playthrough.json contains literal joint integer vectors, seed 0 and UNEXECUTED_AUTHOR_HYPOTHESIS statuses. Seed 0 is a fixed author convention, not an observed good seed. Script legality and effects are reasoned from source only. No execution, score, success guarantee, effect magnitude, solvability, non-isomorphism or learning/transfer claim is made.

## Source and reference provenance

All game rules and kernel code were authored for this assigned game. No upstream implementation or commercial game asset was copied. The human-play and independent-family requirements are recorded with actual paths, bytes and SHA256 in author_input_freeze.json, together with every other closed input consumed.

- JaxMARL supplied README: parallel agent-keyed interfaces and __all__ for a later adapter. This package retains the assigned flat native kernel. https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md
- Pgx supplied README: explicit legal masks, immutable game state and caller-supplied randomness are integration concepts; no board-game mechanic was copied. https://github.com/sotetsuk/pgx/blob/main/README.md
- Craftax supplied README: block materials, consequential inventory and explicit raw/autoreset separation informed the category/integration discussion; no Craftax world or rule implementation is used. https://github.com/MichaelTMatthews/Craftax/blob/main/README.md
- PureJaxRL README and Craftax_Baselines ppo_rnn.py were read as end-to-end JAX/recurrent-reset integration context. They are algorithm implementations, not a game library or replacement for the frozen experimental learner. No algorithm code, weights, dependencies or reported scores enter this game. https://github.com/luchris429/purejaxrl/blob/main/README.md ; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py

This environment can supply a future cross-environment adaptation task only after independent hash-bound integration. Actual R0 interface checks, eager/JIT/vmap behavior, paired effects, human input, replay, visual fidelity, training, scientific comparisons and transfer remain unverified.
