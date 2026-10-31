# Floodstone Builders

Status: AUTHORED_NOT_EXECUTED. Prospective initial-window source attempt 0; no candidate import, rollout, training, seed search or learned-policy result.

A teammate can excavate the shortcut you need and simultaneously connect your worksite to floodwater. Protection is a material and labor commitment: one worker installs support, another brings the scarce wrench, and someone must keep pumping.

## World, task and player interpretation

The 9 by 7 quarry is a literal grid with a protected boundary rim, a spring at (7,1), camp at (1,5), and three one-unit veins at (6,2), (6,3), (6,4). Each interior stone or ore block yields one building stone. Floor, support, pump, spring and camp cells are walkable; rock, ore and floodwalls are not. Water is independently represented on the same cells. Three miners start at (2,4), (3,4), (2,5), with breath six, stone [2,1,2], and tools pick/empty/wrench. Tool names describe initial equipment, not locked player classes.

## Native controls and contextual feedback

Supply one integer action per role to raw_step. Any legal action sequence is supported; no script ID or scripted policy is consulted by the kernel. Directional actions remain selectable for able characters before joint completion; blocked moves spend the tick and stay in place. Interaction masks show the current prerequisites. Simultaneous requests can still contend for the same finite item, tile or station.

- 0: Wait. Advance water and other workers while holding position.
- 1: Walk north / face north. A blocked move still turns the miner north.
- 2: Walk east / face east. A blocked move still turns the miner east.
- 3: Walk south / face south. A blocked move still turns the miner south.
- 4: Walk west / face west. A blocked move still turns the miner west.
- 5: Mine front block. Requires a pick, an interior stone/ore tile, stone capacity and ore capacity; remove the block and receive one stone plus its ore if present.
- 6: Install support in front. Requires one stone, empty floor and no miner on that cell.
- 7: Install pump in front. Requires two stones, a wrench, empty unoccupied floor and a cardinally adjacent support post.
- 8: Work front pump. Requires the wrench and a supported adjacent pump; drain Manhattan radius two after this tick water spread.
- 9: Build front floodwall. Costs one stone; the unoccupied front floor becomes a movement- and water-blocking wall.
- 10: Salvage front structure. Recover one stone from a support/wall or two from a pump, only with sufficient capacity and no occupant.
- 11: Deposit ore at camp. Within Manhattan one of camp, bank all carried ore.
- 12: Pass one stone forward. The miner standing in the front cell receives one stone if below capacity four.
- 13: Rescue miner ahead. A breath-zero miner in front regains three breath; one stone is permanently spent on rescue equipment.
- 14: Take pick at camp. Within one cell of camp, take one of the two conserved picks if empty-handed.
- 15: Take wrench at camp. Within one cell of camp, take the single conserved wrench if empty-handed.
- 16: Return tool at camp. Put the held pick or wrench back into camp stock.
- 17: Face north. Turn without moving.
- 18: Face east. Turn without moving.
- 19: Face south. Turn without moving.
- 20: Face west. Turn without moving.

Movement resolves simultaneously: entering any other miner current cell is blocked, including swaps and following into a vacated cell. Competing moves to the same otherwise free destination use rotating priority (agent index minus t) modulo three. Object transactions then run in order t modulo three, next role, next role, and recheck masks. Invalid or newly unavailable requests become no-ops. Builders cannot enclose an occupied target tile.

## Exact transition, costs and completion

Water spreads after transactions on ticks whose next clock is divisible by three. Existing wet cells persist unless a wall or worked supported pump removes their water. Walls exclude water; support posts do not. Pump radius uses Manhattan distance two and the spring is restored after pumping. Breath loss follows the final water field; dry conscious miners recover up to six, while zero-breath miners remain collapsed until rescue. Rewards are +0.25 to every role for each deposited ore and +0.5 to the depositor if it matches that role private vein preference, minus 0.05 for a conscious miner exposed to water that tick. There is no construction, excavation or waiting reward.

## Information and visible scene

All bodies, tools, carried ore, stones, terrain and inundation are public and appear in every role observation. The reset key permutes three preferred ore indices; each role sees only its own scalar. Agent entity attributes omit priorities. Full-state/debug access must be labeled separately. Water entity active bits are exactly the flattened water map, including the real spring cell.

Independent block-survival art direction: wet layered ochre quarry blocks, chipped copper seams, timber support posts, green-patinated pump heads and small miners with readable helmets and carried packs. The trusted scene binds actual map codes and a water-cell group, so excavated passages and installed structures must change in place. The desired illustration is a quarry game scene; counters can annotate the scene but cannot replace it. Original artwork only; no copied Minecraft assets.

The scene is complete declarative presentation data with exact source/manifest identity. It does not implement a browser, keyboard or animation engine. A later trusted human-play adapter must separate Enter game from Watch replay, pass deliberate joint actions into this unchanged kernel, show legal controls, and preserve role visibility. Human play is separate from formal experiments.

## Cross-agent mechanisms and alternative responses

Opening the right-hand seam can flood a partner path later, whereas a floodwall consumes reusable stone and closes that route. A pump is complementary infrastructure: a builder provides support, a wrench bearer supplies labor, and another miner can use the temporarily drained space. Donation can finance another miner construction or rescue but leaves the donor with less material. Preferred-vein bonuses create a choice between banking another miner preferred ore for team gain and reserving one own vein. Proposed alternatives include a wall-first southern route, supported pump teamwork, or opportunistic extraction with rescue reserves; none has been ranked by execution.

## Declared invariants and implementation limits

For source-reachable states: miner positions remain distinct and interior; stone stays in 0..4; breath in 0..6; ore count per vein is conserved across an intact vein, all inventories and bank. Each miner carries at most two ore units. Stone conservation counts remaining mineable interior blocks, carried stones, one per support/wall, two per pump, and spent rescue stone. Picks in hands plus cache equal two; wrenches equal one. Water is absent from solid rock, ore and floodwalls. The source does not simulate fluid volume or structural collapse; the spring supplies water and pumps remove it. These are declared rules, not executed invariant results.

State leaves use fixed-shape int32, bool and, where declared, float32 arrays. Small signed fields are represented as int32; no PRNG-key surrogate is stored. Only reset uses the supplied key where explicitly stated. raw_step is deterministic for a given state/action and ignores its unused key. Natural termination has priority over the horizon. A terminal input preserves every state leaf and observation, returns zero float32 rewards and joint dones, and preserves terminated/truncated causes. No autoreset occurs in the kernel.

## Literal source evidence

- Literal map and starting equipment: candidate.py:12 (def reset).
- Spatial and tool prerequisites: candidate.py:52 (def legal_actions).
- Conserved dig/build/salvage transactions: candidate.py:124 (refund =).
- Cross-agent donation and rescue: candidate.py:135 (recipient = jnp.argmax).
- Water connectivity timing: candidate.py:147 (# Connected inundation).
- Pump labor and support complementarity: candidate.py:153 (# A supported installed pump).
- Terminal absorption: candidate.py:174 (s = jax.tree.map).
- Required boundary signatures: candidate.py:12, candidate.py:80, candidate.py:52, candidate.py:88.

## Frozen author hypotheses

The construction diagnostic transfers stone, installs a support, moves the wrench bearer into a south-side operating position, constructs a pump and works it. It does not claim to drain a flooded worksite or finish the ore objective. The paired hypothesis changes only agent_0 initial pass versus wait, then gives identical support-building and turn-east actions to agent_1; the predicted difference is agent_1 remaining stone and its next legal floodwall action toward empty (4,4).

playthrough.json contains literal joint integer vectors, seed 0 and UNEXECUTED_AUTHOR_HYPOTHESIS statuses. Seed 0 is a fixed author convention, not an observed good seed. Script legality and effects are reasoned from source only. No execution, score, success guarantee, effect magnitude, solvability, non-isomorphism or learning/transfer claim is made.

## Source and reference provenance

All game rules and kernel code were authored for this assigned game. No upstream implementation or commercial game asset was copied. The human-play and independent-family requirements are recorded with actual paths, bytes and SHA256 in author_input_freeze.json, together with every other closed input consumed.

- JaxMARL supplied README: parallel agent-keyed interfaces and __all__ for a later adapter. This package retains the assigned flat native kernel. https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md
- Pgx supplied README: explicit legal masks, immutable game state and caller-supplied randomness are integration concepts; no board-game mechanic was copied. https://github.com/sotetsuk/pgx/blob/main/README.md
- Craftax supplied README: block materials, consequential inventory and explicit raw/autoreset separation informed the category/integration discussion; no Craftax world or rule implementation is used. https://github.com/MichaelTMatthews/Craftax/blob/main/README.md
- PureJaxRL README and Craftax_Baselines ppo_rnn.py were read as end-to-end JAX/recurrent-reset integration context. They are algorithm implementations, not a game library or replacement for the frozen experimental learner. No algorithm code, weights, dependencies or reported scores enter this game. https://github.com/luchris429/purejaxrl/blob/main/README.md ; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py

This environment can supply a future cross-environment adaptation task only after independent hash-bound integration. Actual R0 interface checks, eager/JIT/vmap behavior, paired effects, human input, replay, visual fidelity, training, scientific comparisons and transfer remain unverified.
