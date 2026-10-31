# Ember Shelter

Status: AUTHORED_NOT_EXECUTED. Prospective initial-window source attempt 0; no candidate import, rollout, training, seed search or learned-policy result.

A warm refuge is a public good assembled from complementary materials. Burning timber now can protect all three foragers later, but it can also consume a teammate wall supply and increase smoke at the kiln.

## World, task and player interpretation

A 7 by 7 original block-world clearing contains three six-unit timber stands, one six-unit clay deposit, shelter at (3,3), kiln at (5,3), and locker at (3,5). Stone rim, trunks, clay and the two structures are solid. The locker apron is walkable. Three workers start adjacent to shelter with packs [2 timber], [1 timber + 1 lining], [1 timber + 1 clay], and health eight. Materials are whole units; pack capacity three forces deliveries and handoffs.

## Native controls and contextual feedback

Supply one integer action per role to raw_step. Any legal action sequence is supported; no script ID or scripted policy is consulted by the kernel. Directional actions remain selectable for able characters before joint completion; blocked moves spend the tick and stay in place. Interaction masks show the current prerequisites. Simultaneous requests can still contend for the same finite item, tile or station.

- 0: Wait. Let processing, weather and other workers advance.
- 1: Walk north. One cardinal tile; solid fixtures and occupied tiles block entry.
- 2: Walk east. One cardinal tile; solid fixtures and occupied tiles block entry.
- 3: Walk south. One cardinal tile; solid fixtures and occupied tiles block entry.
- 4: Walk west. One cardinal tile; solid fixtures and occupied tiles block entry.
- 5: Gather nearby material. Within one cell of a finite timber/clay stand, take one material if the three-slot pack has room; lowest eligible node index breaks ties.
- 6: Build shelter wall. Within distance two of shelter, spend one timber to add a wall level up to three.
- 7: Load kiln with timber and clay. Beside the empty kiln, consume one timber and one clay to start three-tick lining production.
- 8: Take kiln lining. Beside the kiln, take its finished lining if the pack has room.
- 9: Install shelter lining. Within distance two of shelter, spend one lining to raise insulation up to three.
- 10: Feed shelter hearth. Within distance two of shelter, burn one timber for four shared heat; requires heat at most eight.
- 11: Vent kiln smoke. Beside the kiln, remove up to three smoke.
- 12: Pass one material. Give the lowest-index carried material to the lowest-index adjacent conscious worker with pack capacity.
- 13: Stash private-priority material. Beside the locker, remove one item matching your private preference for an individual reward; it leaves the common material economy.
- 14: Repair shelter wall. Same one-timber wall operation as build, explicitly available after storm/fire wear.
- 15: Rest in daytime shelter. During phase 0..11 and within two cells of shelter, regain one health up to eight.

Movement uses conservative simultaneous occupancy: no swaps or entry into another role old cell. Same-destination ties rotate by t modulo three. Transactions rotate through all three workers and recheck current inventory, station, stock and capacity, so a mask is individual eligibility rather than a guarantee that all simultaneous claims succeed. No permanent player-index winner is built in.

## Exact transition, costs and completion

A cycle is 18 transitions: source states with phase 0..11 are daytime, 12..17 nighttime. Reset independently samples three cold severities in {1,2} and a permutation of material preferences. During a night transition a conscious worker is protected iff within distance two of shelter, wall>=2, lining>=2, and heat>=that night severity. Unprotected night damage is one health. Smoke>=6 additionally damages a worker within one of the kiln by one. Health is floored at zero; collapsed workers can only wait and do not auto-revive. Hearth fuel adds four heat up to 12; one heat is consumed each night transition. Smoke rises by two when kiln loading or stoking and falls by one every third tick or up to three on vent. Kiln timers include the loading transition. Night end or qualifying fire wear removes one existing wall level, with at most one wall loss per tick. Private stash gives +0.25 to its contributor, damage costs 0.1 per health unit, and a surviving worker receives +1 at t=54. Building itself is unrewarded.

## Information and visible scene

Every role sees physical inventory and shared construction. The locker aggregate is public in both observation and station attributes. The current weather level is public; each role privately knows exactly one night forecast indexed by its role, plus its own preferred stash resource. No renderer may reveal other forecasts or priority values. Zero-health bodies remain visible as collapsed characters and occupy their cells.

Independent block-survival art direction: cedar trunks, layered clay, hand-built timber wall sections, woven lining on the refuge, a glowing hearth, a chimney with state-bound smoke and blue night frost over the clearing. Worker packs visibly carry timber, clay or lining rather than generic points. Shelter and kiln status rows are semantic render bindings: wall changes the built silhouette, lining changes its interior surface, heat lights the hearth, kiln timer changes processing, and smoke controls the local plume. These are desired faithful drawings, not a claim that frames were rendered.

The scene is complete declarative presentation data with exact source/manifest identity. It does not implement a browser, keyboard or animation engine. A later trusted human-play adapter must separate Enter game from Watch replay, pass deliberate joint actions into this unchanged kernel, show legal controls, and preserve role visibility. Human play is separate from formal experiments.

## Cross-agent mechanisms and alternative responses

One worker wall investment changes the same cold transition for all nearby workers. A clay carrier can supply kiln production while a timber carrier chooses between lining fuel, hearth heat, wall repair and private stash. Venting can protect a nearby kiln operator but consumes a work tick; taking the finished lining can block a second batch until someone installs or moves it. The privately known night forecast can support anticipatory preparation through visible actions without exposing it directly. Candidate strategies include early shared refuge construction, staggered kiln/hearth labor, and private stockpiling with later repairs. Their relative value remains unmeasured.

## Declared invariants and implementation limits

Packs remain nonnegative with total at most three; health 0..8; wall and lining 0..3; heat 0..12; smoke 0..8; kiln output 0/1; kiln timer 0..3. Three construction/material ledgers support conservation: timber across nodes, packs, stash, wall, lost_wall and burned_timber; clay across nodes, packs, stash and processed_clay; lining equals the initial one plus completed kiln conversions across packs, installed lining, stash and output. A pending kiln conversion is represented by positive timer. Timber is spent on heat and kiln combustion; no resources regenerate. The model represents a shared shelter footprint and protection radius, not arbitrary wall placement or a fire-fluid simulator.

State leaves use fixed-shape int32, bool and, where declared, float32 arrays. Small signed fields are represented as int32; no PRNG-key surrogate is stored. Only reset uses the supplied key where explicitly stated. raw_step is deterministic for a given state/action and ignores its unused key. Natural termination has priority over the horizon. A terminal input preserves every state leaf and observation, returns zero float32 rewards and joint dones, and preserves terminated/truncated causes. No autoreset occurs in the kernel.

## Literal source evidence

- Literal stocks, resources and private draws: candidate.py:12 (def reset).
- Pack capacity and spatial interactions: candidate.py:43 (def legal_actions).
- Finite complementary kiln inputs: candidate.py:100 (s["carry"] = s["carry"].at[i,0]).
- Private stash competes with materials: candidate.py:116 (stash = act==13).
- Exact weather protection: candidate.py:131 (protected = near_shelter).
- Wall maintenance timing: candidate.py:137 (wear =).
- Absorbing terminal transition: candidate.py:151 (s = jax.tree.map).
- Required boundary signatures: candidate.py:12, candidate.py:64, candidate.py:43, candidate.py:73.

## Frozen author hypotheses

The build/heat/lining diagnostic spans the first night onset without searching a seed. It installs one wall and one lining, starts the kiln, feeds the hearth, adds a second wall, takes and installs kiln output, then waits through transition 13. Its paired version changes only agent_0 first wall contribution to wait. With identical later actions, source reasoning predicts two versus one wall at first cold exposure and therefore different partner health, while both branches have enough shared heat and lining.

playthrough.json contains literal joint integer vectors, seed 0 and UNEXECUTED_AUTHOR_HYPOTHESIS statuses. Seed 0 is a fixed author convention, not an observed good seed. Script legality and effects are reasoned from source only. No execution, score, success guarantee, effect magnitude, solvability, non-isomorphism or learning/transfer claim is made.

## Source and reference provenance

All game rules and kernel code were authored for this assigned game. No upstream implementation or commercial game asset was copied. The human-play and independent-family requirements are recorded with actual paths, bytes and SHA256 in author_input_freeze.json, together with every other closed input consumed.

- JaxMARL supplied README: parallel agent-keyed interfaces and __all__ for a later adapter. This package retains the assigned flat native kernel. https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md
- Pgx supplied README: explicit legal masks, immutable game state and caller-supplied randomness are integration concepts; no board-game mechanic was copied. https://github.com/sotetsuk/pgx/blob/main/README.md
- Craftax supplied README: block materials, consequential inventory and explicit raw/autoreset separation informed the category/integration discussion; no Craftax world or rule implementation is used. https://github.com/MichaelTMatthews/Craftax/blob/main/README.md
- PureJaxRL README and Craftax_Baselines ppo_rnn.py were read as end-to-end JAX/recurrent-reset integration context. They are algorithm implementations, not a game library or replacement for the frozen experimental learner. No algorithm code, weights, dependencies or reported scores enter this game. https://github.com/luchris429/purejaxrl/blob/main/README.md ; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py

This environment can supply a future cross-environment adaptation task only after independent hash-bound integration. Actual R0 interface checks, eager/JIT/vmap behavior, paired effects, human input, replay, visual fidelity, training, scientific comparisons and transfer remain unverified.
