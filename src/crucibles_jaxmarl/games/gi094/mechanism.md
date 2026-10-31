# Three-Hand Kiln

A/O/U/H = 3/143/10/96. This cooperative kitchen uses four physical ingredients, a unique shared cleaver, a cutting bench, an unattended timed oven and an ordered dispatch station. All roles have the same action capabilities; their reset positions suggest an initial division of labor without forcing it.

## Rules and causally coupled decisions

The 9 by 7 map has a central wall interrupted only at [4,3]. Cooks occupy distinct cells and cannot enter any other cook's pre-step cell, even when that cook intends to leave; competing destinations also block. Ingredients and the cleaver may share their carrier's cell. The cleaver starts at the cutting bench [2,2], the oven is [6,2], and dispatch is [7,5]. Station positions govern interactions, not merely the illustration.

A cook can carry one ingredient and also carry the cleaver. A raw ingredient needs two work actions while its carrier stands at the bench and holds the cleaver. The ingredient then becomes cut. Dropping a cut ingredient on the oven causes cumulative heat to increase on each nonabsorbing tick, including the drop tick. At heat 4 it becomes baked. If it remains on the oven until heat 8 it burns. Heat is retained if it is lifted and returned. First pickup also starts a cumulative age clock; age 50 spoils any unresolved ingredient. Waiting therefore changes partners' available resources.

Dispatch accepts only baked ingredients whose flavor matches the next public order [0,1,0,1]. Each accepted ingredient becomes inactive and gives all roles +1. A spoilage makes an ingredient inactive and gives all roles -0.5. Natural completion is all four ingredients resolved, including failures; successful fulfillment of all orders is not assumed by termination. The horizon is 96.

Commands are 0 wait; 1/2/3/4 north/east/south/west; 5 pick up; 6 put down; 7 work at bench or dispatch; 8 consented adjacent handoff; 9 pick up or drop the cleaver. The complete labels are in the manifest and scene. Object operations resolve in rotating role order starting at t modulo 3. A handoff requires both neighboring roles to choose 8, an empty receiver, and an ingredient not already handed off that tick. This once-per-ingredient rule prevents a consenting receiver from immediately handing the same ingredient back later in the priority loop.

The handoff changes the receiver's inventory and the item's actual position. A cook retaining the only cleaver changes every partner's cutting opportunities. A slow oven collector can destroy another cook's preparatory work. Occupying the central doorway delays both delivery and food aging, so independently maximizing movement is insufficient.

Two plausible plans are a stationed cutter plus oven and dispatch runners, or a rotating whole-chain carrier with the other cooks moving ingredients and clearing traffic. Either may be useful depending on partner timing. Both are hypotheses, not observed solutions. Passing food saves travel for the cleaver holder but adds consent and rendezvous requirements; carrying the entire chain avoids failed handoffs but ties up the tool. Preparing multiple ingredients early increases pipeline capacity while exposing them to the age clock.

## Information, visible state and declared bounds

Every ingredient position, carrier, stage, flavor, work count, heat, age and first-pickup flag is actual state. The cleaver's position follows its carrier; station coordinates are the same arrays used by transitions. Scene attributes, action labels and events bind these leaves directly. Complete public state plus own-role one-hot produces 143 float32 columns. There is no private order or observation asymmetry to invent.

All agent positions remain distinct on walkable floor. Capacity is always three role slots, four ingredient slots, three station slots and one cleaver slot. Every cook holds at most one ingredient; a held active ingredient follows exactly its holder. Cut work stays in [0,2], heat in [0,8], age in [0,50], order_index and spoiled in [0,4]. Ingredient stage is 0 raw, 1 cut, 2 baked or 3 spoiled. No new ingredient is spawned. The shared reward is bounded by [-2,4] from four ingredients; this safe finite bound is not an experimental performance threshold.

## Literal source evidence

- Actual map and reset entities: `candidate.py:23`, literal `def reset(self, key):`.
- Public observation packing: `candidate.py:54`, literal `def observe(self, s):`.
- Spatial collision rule: `candidate.py:71`, literal `occupied = jnp.any(jnp.all(q[:, None, :] == p[None, :, :], axis=2) &`.
- Cutting requires the actual shared tool: `candidate.py:113`, literal `chopping = ((a[i] == 7) & carrying & (holder[j] == i) & (stage[j] == 0) &`.
- Flavor-specific delivery: `candidate.py:117`, literal `serving = ((a[i] == 7) & carrying & (holder[j] == i) & (stage[j] == 2) &`.
- Two-sided handoff and same-tick protection: `candidate.py:128`, literal `giving = (a[i] == 8) & carrying & active[j] & (holder[j] == i) & ~touched[j] & jnp.any(receivers)`.
- Heat clock on the real oven tile: `candidate.py:134`, literal `in_oven = active & (holder < 0) & jnp.all(ip == s["station_pos"][1], axis=1) & (stage >= 1)`.
- Persistent spoilage: `candidate.py:137`, literal `ruined = active & ((age >= 50) | (in_oven & (heat >= 8)))`.
- Natural end rule: `candidate.py:150`, literal `s["terminated"] = ~jnp.any(active)`.
- Absorbing terminal branch: `candidate.py:164`, literal `def absorb(self, key, state, actions):`.

## Witnesses and possible future variation

The first script attempts cleaver pickup, ingredient 3 preparation, a consented transfer, oven processing and one flavor-0 delivery. The paired hypothesis changes only receiver consent at the rendezvous while keeping the later runner motions literal. Its intended evidence is item_holder, item_pos, carry and handed_now; there is no execution receipt.

Future separately assigned rules could alter the doorway, cleaver number, heat/age timing or flavor sequence, changing the balance between station specialization and flexible carriers. None of those variants is silently sampled or implemented here, and no adaptation benefit is claimed.

## Provenance, interface and status

Source SHA-256: 2beb035ce5becb46b53f5a6776d4e5eac5f13517a7126da146ba170a339a99ec.

This is original author-written game code and layout for this assigned train-stream proposal, source attempt 00, generation window 0. No previous candidate, execution result, replay, ranking, training output or final-test material was read. No upstream game code, benchmark layout, texture, heldout implementation or scoring code was copied. Original authorship does not prove structural non-isomorphism.

The supplied official JaxMARL README informed the distinction between the native flat kernel and an agent-keyed public integration API. The supplied Pgx README informed explicit state, legal-mask and terminal API concepts. The supplied Craftax README informed JAX game-state and reset/render separation concepts. The supplied PureJaxRL README was read as an algorithm/integration reference, not a game library; no PPO implementation was copied and the frozen scientific learner is unchanged. The supplied Craftax_Baselines/ppo_rnn.py was not read or used. Exact copied-reference hashes and official source URLs are in generation_record.json.

Every declared nonterminal command is legal. A physically blocked movement or unavailable object operation is a deterministic failed attempt, not an out-of-range action. Out-of-range integers map to command 0. No role is individually removed before joint termination. Each role retains command 0; after joint termination only 0 is legal. The absorbing raw call ignores the provided actions, returns the same complete state and observation, leaves t unchanged, returns zero float32 rewards and all-true dones, and preserves actual terminal causes. Otherwise t advances exactly once. Natural termination has priority over the horizon, so terminated and truncated cannot both be true. No reset occurs inside raw_step.

All observation columns are covered exactly once in manifest.json, in the literal concatenate order used by observe. Raw integer/bool state is cast to float32 without normalization. Visibility is fully public: active entity visibility is broadcast to all roles; map.public is true. Fixed or derived active/visibility masks do not add hidden policy information. Resolved objects retain their last public position and status in flat state, while the role renderer suppresses their inactive glyphs. A separate full-state debug view must remain labelled as such. The source uses only int32 and bool state, representing small categorical design values losslessly as int32. The reset key remains external provenance and is intentionally unused by these deterministic initial rules.

The native kernel is not a public adapter implementation. Agent-keyed dictionaries, __all__, space objects and replay/UI integration belong to the coordinator's separate presentation layer. The author has not exercised that layer. The six author files satisfy a source-delivery scope; they do not by themselves establish community product acceptance.

No candidate import, reset, raw_step, JAX call, validator, training run, replay, browser playback or numerical parity test was executed by this author. Pure text and JSON metadata review and byte hashing are the only checks performed. playthrough.json contains literal finite action vectors, seeds and explicitly unexecuted paired intervention hypotheses. A later sealed replay must retain mismatches, zero effects, failures and early terminals as actual evidence. No script is a successful trajectory receipt, solvability proof, learned policy, optimal strategy, meta-learning result or generalization result. Model tokens and API charges are unknown rather than invented.

Reference URLs read as supplied snapshots:
- https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md
- https://github.com/sotetsuk/pgx/blob/main/README.md
- https://github.com/luchris429/purejaxrl/blob/main/README.md
- https://github.com/MichaelTMatthews/Craftax/blob/main/README.md
