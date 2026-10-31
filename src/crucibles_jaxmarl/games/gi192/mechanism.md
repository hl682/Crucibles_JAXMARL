# Orchard Commons — authored native game mechanism

Status: AUTHORED_NOT_EXECUTED. Assigned game-v3-00192; three agents; source attempt 00.

## Game and human interaction
Orchard Commons is a rural community game with three personally owned plots at (1,2), (4,2), (7,2). A bee meadow at (4,0), shared pruning tool initially at (4,3), pond at (4,5) and market at (4,6) are real interactable places. Covered irrigation channel cells are walkable; the pond cell is not. Farmers may share a cell. The world is a compact farm landscape, with crop production visibly coupled through ecology and equipment.

Human players use native directional actions and contextual Plant apple/pear, Water, Tend, Pick up, Pass/drop, Harvest, Sell and Compost. The current mask and visible state identify interaction prerequisites. The tool can be carried deliberately between farmers; no canned script is required. Arbitrary legal joint action sequences work through the same reset/raw_step interface. Human controls and actual rendered gameplay are pending trusted integration.

## Rules and exact chronology
Reset draws each farmers private favorite, apple or pear, once from its supplied key. All other state and later dynamics are deterministic. Each farmer begins with two compost bags, soil quality two, empty plots and baskets. Shared water is three, bee habitat one, and the single tool is unheld.

A farmer may plant either crop within one cell of its own bare plot while completed growth ticks are below five. Water and compost target the nearest plot if within one cell; tied targets use lowest plot index. Watering is allowed for any growing or flowering plot, including someone elses. It consumes one shared water and adds up to two moisture, capped at three. Tending within one cell of the meadow adds up to two habitat, capped at four. Compost spends one private bag to improve any adjacent plot soil by one, capped at three.

Masks use incoming state, invalid commands normalize to Wait, and movement resolves simultaneously. Nonmovement intents use rotating role priority t mod 3. Shared water/tool constraints are rechecked; a lost race produces no consumption. Each player has one action, so movement cannot be combined with interaction.

Every fourth step is a growth tick. During the first six ticks, a moist seedling becomes blossom and a moist blossom becomes ripe; the stage snapshot prevents both changes on one tick. A new blossom records pollen if habitat is at least two at that moment. On ripening, fruit count is one plus captured pollen plus a soil-at-least-two bonus. Thus tending habitat before someone elses blossom affects its later ripe crop. A growing or flowering plot loses one moisture on a growth tick. Habitat then loses one and pond water refills by two to capacity four. All interaction changes precede growth, so a same-step meadow intervention can affect that steps blossom.

Ripening sets ripe age zero. On each later fourth-step tick a previously ripe plot ages, and at age two its remaining fruit spoils. Ripeness aging continues after six production ticks. Unripe plots are cleared when the sixth tick closes the growing season. Planting cannot extend this season.

Only the holder can harvest its own ripe plot, within one cell. A basket holds three fruit of a single type; the harvest removes as many as fit. Taking the last fruit makes the plot bare and reduces soil by one. Pass/drop sends the tool to the adjacent farmer with the lowest role index, using postmovement positions; if none is adjacent it is placed at the givers position. A pickup requires an unheld tool within one cell. New recipients act with it from the next decision because incoming masks were already frozen.

At the exact market tile, selling empties the basket. Unit price is one, plus one for own favorite, plus one when the crop matches completed-growth-tick parity. Those private and timing incentives can differ across farmers. Natural termination is after six growth ticks, with no ripe plots and no baskets. Otherwise H=40 truncates. The terminal state/clock/cause then absorb, with zero reward.

## Why partners matter
Watering ones own crop uses the same replenishing reserve another farmer needs. Traveling to support bee habitat can raise another persons later crop while delaying ones own field work. The unique pruning tool makes the order of pickup, harvest and handoff determine whether fruit is saved before spoilage. Composting a neighbors depleted soil spends a scarce private input. Farmers can prioritize independent baseline yields, cooperate on pollination, trade equipment access, synchronize harvests or deliberately monopolize a time-sensitive tool. No artificial gift score makes these choices automatically beneficial.

This is an ecological timing and equipment-allocation mechanism, separate from territorial trade and workshop job queues. Its source structure does not prove optimal play, strategic richness, learnability or meta-learning.

## Information and visual charter
The first 46 normalized columns are public physical state: time/season, shared reserves, farmer locations/baskets, compost, tool holder/position, plot stages/types/moisture/soil/pollen/fruit/age, and score. Three self role bits and own private favorite complete O=50. Fixed map coordinates are common knowledge. Other favorites are excluded from visual attributes.

Use warm rural artwork: aproned gardeners with baskets, recognizably different apple and pear trees, bare beds, blossom clusters, fruit-laden branches, straw paths, covered water channels, wildflowers and bees, a strapped pruning tool, and a timber market awning. Pollination and moisture should be shown on the actual plot, not as unrelated meters. Tool possession must follow its native holder and position. Decorations cannot invent flying bee state, planted locations or harvesting transitions. Scene glyphs, attributes, actions and rules are already bound; a richer procedural material treatment is downstream presentation work.

## Bounds and diagnostics
Water and habitat stay in 0..4, moisture/soil/fruit in 0..3, compost in 0..2, tool holder in {-1,0,1,2}, baskets in 0..3, stages in 0..3, ripe ages in 0..2. Basket/crop type -1 means empty/unplanted; encoded values are nonnegative. Six growth ticks and finite planting opportunities bound total production and score. Per-step reward is 0..9; observations are in [0,1]. These are source-derived bounds, not measured results.

The meadow script has the outer farmers plant and water while the central farmer travels to the habitat. Its paired hypothesis changes only the central Tend-versus-Wait at the first blossom tick, then waits through the next growth tick. Pollen and later fruit on the other two plots are intended comparisons. A second script makes a real pickup and adjacent tool handoff. Seed 192 is fixed without search. These sequences have not been executed and do not certify successful gameplay.

## Complete native action dictionary

- 0: Wait.
- 1: Walk north.
- 2: Walk east.
- 3: Walk south.
- 4: Walk west.
- 5: Plant apple at own empty plot.
- 6: Plant pear at own empty plot.
- 7: Water nearest adjacent growing plot: 1 shared water.
- 8: Tend adjacent bee meadow: add up to 2 habitat.
- 9: Pick up nearby unheld pruning tool.
- 10: Pass tool to adjacent lowest-index farmer, or put down.
- 11: Harvest own ripe plot with pruning tool.
- 12: Sell whole basket at market.
- 13: Compost nearest adjacent plot: 1 own compost.

## Literal source-line evidence

- Sites and reservoir: candidate.py line 12; literal source: def reset(self, key):
- Human interaction prerequisites: candidate.py line 55; literal source: def legal_actions(self, s):
- Finite-resource/tool arbitration: candidate.py line 95; literal source: for k in range(3):
- Real tool handoff: candidate.py line 116; literal source: next_holder = jnp.where(jnp.any(near), recipient, -1)
- Delayed blossom and ripening: candidate.py line 144; literal source: old_stage = n['stage']
- Fruit spoilage: candidate.py line 153; literal source: spoil = tick & (old_stage == 3) & (n['ripe_age'] >= 2)
- Private favorite observation: candidate.py line 173; literal source: def observe(self, s):


## Provenance, reference use and verification limits
This is source attempt 00 of the original blind initial window, not a repair or feedback-selected revision. Current batch ordinals 186..193 were reserved inside the original cap 512; cumulative used-or-reserved count was 194 and feedback count was zero. This author owns only GI190..193 attempt_00 and does not alter other work. Exact actual input paths, byte sizes, hashes and purposes are in author_input_freeze.json, including the prospective human-playability addendum. All four assigned design/request pairs were read; no other candidate, result, selection, project training, develop/confirm, heldout, IR or final-test input was accessed.

The supplied official JaxMARL README informed a prospective dictionary-per-agent adapter boundary. Pgx informed explicit external random keys, masks and environment identity. Craftax README informed native spatial state/presentation and raw versus outer autoreset. PureJaxRL README and Craftax_Baselines_ppo_rnn.py informed only reset/step/scan integration. They are algorithm integration references, not new game libraries or replacements for the frozen learner. No upstream game, algorithm code or visual asset was copied. No network request, model/API invocation, remote session, credential inspection, download or candidate execution was performed.

Signed enums, counters and ownership fields use lossless int32; state booleans use bool. The supplied reset key is not duplicated inside state. This representation adaptation does not imply changing an executed earlier distribution. Each new private-parameter distribution is stated above. Raw terminal semantics follow the addendum: a live call advances t once, natural termination overrides horizon, and an already-done call preserves state/observation/time/cause with zero rewards.

Only stdlib AST, JSON and hash checks were authorized. No candidate or JAX import, eager call, JIT, vmap, rollout, random policy, seed search, replay, RGB render, adapter test or training was run. Shapes/dtypes, masks over trajectories, numerical parity, invariant preservation and paired effects await independent frozen-source integration. Positive, zero, negative, failed and indeterminate results remain valid reports. No reward, seed-count or success-rate acceptance threshold is introduced. Source authorship is not human-playability acceptance or evidence of learning/transfer.
