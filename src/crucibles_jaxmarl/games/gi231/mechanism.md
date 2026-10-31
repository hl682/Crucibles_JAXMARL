# Orchard Pollination Commons

Status: AUTHORED_NOT_EXECUTED. Candidate game-v3-00231, source attempt 0, train-only prospective source stream. Assigned author identity: gpt-6-astra / max; root dispatch is the authority for actual model execution. A/O/U/H = 48/315/23/768. Source SHA-256: b96f8fe5123543183ed78649d8991a974ff64e0299f0a83505d29b761a179467.

## Native game and dependencies

Twelve households occupy one 48 by 24 valley. Each household has four individually controlled people, initially tender/grafter, irrigator, apiarist and harvester. Twelve bees and twelve apiaries are separate ecology objects; they never contribute to A=48. The world contains twenty-four actual planting plots, connected water nodes, gates, buildable/removable hedges and three shared market service rows.

A plot progresses through empty, sapling, flowering, fruiting and spent. Moisture above zero, disease below 3 and a non-winter season allow growth age to advance, capped at 32. Saplings flower at age 6, or 4 if grafted. A flower with pollen at least 2 fruits at age 14, or 12 if grafted, producing two fruits. A basket harvests one fruit per action. A tender can prune one disease unit or compost a spent plot, and replant using a finite seed.

Every second non-winter tick, each living bee associated with a placed hive attempts one horizontal cell. It patrols within five cells of the hive; if relocation leaves it outside that range, it heads inward toward the new anchor. A blocked edge, hedge or closed gate reverses its direction. Bees do not pathfind around barriers, and relocation can leave a poor or temporarily blocked route. People can carry an apiary; its actual position follows its carrier and its bee waits until the hive is placed on that hive's original corridor lane.

A flying bee within Manhattan distance 1 of a flower increases pollen. Spores carried by a bee can add disease to another flower; visits to sufficiently diseased flowers replenish the bee's spore stock. Excessively wet soil can also add disease at eight-tick boundaries. A newly constructed hedge or newly closed gate directly on a living bee reduces its health by one. A zero-health bee becomes inactive. Merely being near a closed gate does not count as injury. These are game abstractions, not a claim of biological realism.

Each open sluice draws up to two units from its incoming native packet and applies that amount to each of its two plots. Remaining water flows to the next household in the same row on the next tick; the last node's remainder leaves that row. The first node receives a three-unit rain packet every fourth non-winter tick. Thus diversion, manual bucket filling, overwatering and passing water downstream change other households' opportunities. A filled bucket can instead irrigate the particular plot ahead.

Fruit deposited in a store enters a capacity-four queue. Every fourth tick, each row market serves one nonempty eligible household queue out of four. Search starts at a rotating column and skips empty or completed households. Another queue can consume the shared slot and delay a neighbor. The actual shipped counter, rather than a planned or merely deposited fruit, counts toward the task.

Plausible alternatives include opening a bee corridor to share visits, relocating a hive, grafting for earlier growth, pruning disease before bees transport it, reserving water for downstream households, manually watering a dry plot and choosing which store should receive a carried fruit. Barriers can provide local control but harm shared ecology. None of these choices has a measured value or proof of viability.

## Actions, observations and presentation

After actions 0..8, actions 9..22 plant, graft, tend/compost, harvest, deposit, fill/apply water, toggle sluice, toggle gate, hand off, lift/place hive, swap tools, signal, build hedge and remove hedge. Giving prioritizes fruit, then water, then compost. Tool exchange can change who performs specialist work. A carried hive is placed only on its original lane in a clear grass cell.

Each 315-column row includes 16 self values, a 49-cell four-channel terrain/growth/occupancy/bee-count patch, four detailed actors, two detailed plots, one detailed hive, local water/store records and four detailed bees. The patch explicitly encodes gate openness. Detailed entity masks are narrowed to exactly the selected records; other visible bee/hive occupancy cannot leak hidden attributes through rendering. Bees outside the local patch or inactive bees are not drawn in a role view. Public season is floor(t/64) modulo 4, with phase 3 winter.

The warm rural scene binds true crop stages, soil moisture, spores, bee health, gate state, moving hives and shipping queues. It does not add a second ecology simulation or count decorative animals as controllable roles. The human-control frontend and real animation are still integration work.

## Rewards and termination

reward_parts has five columns.

- Column 0 charges the actor 0.01 for movement and 0.02 for planting, grafting, tending, irrigating or changing a hedge.
- Column 1 gives the actor 0.1 per harvested fruit and 0.05 per deposited fruit.
- Column 2 gives each member of the store's household 0.25 when one fruit is actually shipped.
- Column 3 gives each actor 0.005 per newly ripened plot within Manhattan distance 8, and subtracts 0.005 per bee injured this tick within the same distance measured after the environmental pass. A per-bee injury flag counts once in this shared component.
- Column 4 charges the builder 0.02 per constructed hedge, charges a bucket filler 0.005 per water unit removed, and charges the acting barrier modifier 0.1 for each bee it directly injures. This last recipient is the actual causative actor, rather than all nearby people.

The interval [-8, 8] bounds one actor action, at most twelve injury recipients, twenty-four possible local ripening events and one shared-market pass. It is a finite contract range, not a score requirement. Crops, carried quantities, queues, spores and health have explicit small caps. Lost pollinators or poor watering can make the objective unattainable; failure and timeout remain valid outcomes.

Natural completion requires two actual shipments from every household store. Otherwise H=768 truncates. NPC health never changes the 48 agent done rows.

## Hypotheses and limits

One literal script explores closing and reopening a gate, grafting and water diversion. The other lifts, carries and places a hive, then manually waters a plot. The paired hypothesis changes only irrigator 1's initial gate action, with the same seed, empty prefix and eight identical waiting steps. It intends to inspect the actual bee trajectory, downstream plot pollen/crop and harvester 3's legal harvest opportunity. No branch has been run or observed.

The source uses fixed arrays and native actions for every object interaction. Only static author checks have occurred. Native interfaces, causality, reachability, real animation, human play, memory cost, throughput, learning and transfer remain unverified.

## Literal source evidence

These line references identify authored implementation expressions, not measured effects.

- Tool and spatial affordances: candidate.py line 39, literal source: `def legal_row(self, s, i):`.
- Held hive position binding: candidate.py line 80, literal source: `z["hive_pos"] = jnp.where((z["hive_carrier"] >= 0)[:, None], p[carrier], z["hive_pos"])`.
- Actor-attributed ecological damage: candidate.py line 251, literal source: `injury = (fence | ((action == 16) & s["gate_open"][g])) & jnp.all(z["bee_pos"] == q, axis=1) & z["bee_active"]`.
- Upstream water subtraction: candidate.py line 279, literal source: `remaining = z["canal"] - draw`.
- Separate ecological motion eligibility: candidate.py line 286, literal source: `flying = (tick % 2 == 0) & (season != 3) & (z["hive_carrier"] < 0) & z["bee_active"]`.
- Spatial flower pollination: candidate.py line 300, literal source: `visits = near & flying[None, :] & (z["stage"][:, None] == 2)`.
- Competing shipping queues: candidate.py line 319, literal source: `ranked = eligible[:, priority]`.
- Absorbing native boundary: candidate.py line 339, literal source: `def raw_step(self, key, state, actions):`.

## Read-only reference use

Only the current batch's authorized files were read. author_input_freeze.json records all 26 actual input files, bytes, hashes and use. The 24 payload files include this batch's four current designs/requests, shared contracts, current population/family control, general design and human-play requirements, and five named official reference snapshots. Three optional historical population addenda were not read. Earlier copied example numbers are noncontrolling.

JaxMARL README informed the distinction between a simultaneous native action-row boundary and a later agent-keyed dictionary adapter. Pgx README informed explicit legal masks, separate termination causes and external key provenance; no Pgx game was copied. Craftax README informed bounded persistent world state and the separation between mechanics and rendering. PureJaxRL README and Craftax_Baselines_ppo_rnn.py were read only for JAX scan/vectorization and wrapper/algorithm separation. No upstream game code, PPO code, assets or performance claims were copied into this kernel. No reference links were opened, dependencies installed, local models invoked or scientific outcomes read.

## Boundary and evidence status

The native interface is reset(key), observe(state), legal_actions(state), and raw_step(key,state,actions). All state leaves are fixed-shape int32, bool or float32 arrays declared in manifest.json. Keys are accepted but intentionally unused: these original rules and layouts are deterministic. Small integer quantities use int32 at the boundary; there is no hidden PRNG key in state.

The joint input is one int32 action per actor. Resolution uses rotating priority (k + input t) modulo A. Each actor's action is rechecked against the current intermediate state, so contention can turn a previously legal intention into a deterministic wait. Environmental processes run once afterward. This is a simultaneous input interface with explicitly sequential conflict resolution, not a claim that actions commute.

A nonterminal step advances the int32 clock once. Natural completion takes priority over H. An already-terminal raw call preserves the entire state, observation, clock and terminal causes and returns zero reward; it never auto-resets. Event flags refer to the last actual nonabsorbing step and are not new events on repeated absorbing calls. A separate trusted wrapper or public adapter remains outside this author package.

No candidate was imported or executed. No JAX/NumPy runtime, validator, trajectory script, local model, remote job, HTTP endpoint, browser, training or evaluation was run. Standard-library AST parsing and data-only JSON/static consistency checks are not native validation. Source authoring is not an R0, causal, animation, solvability, learning or transfer receipt. No additional proposal or repair attempt was created.

