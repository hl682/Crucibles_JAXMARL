# Lockwake Convoys

A/O/U/H = 3/155/9/80. Three boats move physical cargo through two basins connected by a one-cell-wide canal. The transport problem couples integer movement commitments, simultaneous passage reservations, resource use, paired towing and partly shared delivery payoffs.

## Rules and causally coupled decisions

The original 11 by 7 map has west and east basins and the central row-3 passage. Boats start at [2,1], [8,5] and [2,5], each sharing a tile with an unheld cargo item. Cargo destinations are respectively [8,5], [2,1] and [8,1]. Four docks use those two basin corners plus [2,5]. These coordinates determine interaction, movement and refueling; they are not renderer decorations.

Each boat has a committed heading and integer speed 0, 1 or 2. A directional throttle spends one fuel if available. From rest it chooses the requested heading and starts at speed 1. Repeating the same heading accelerates up to 2. A different heading while moving decreases speed by one and keeps the old heading; a later command from rest can turn. Coast retains momentum. Brake sets speed zero immediately. At the start of a tick, a stopped boat on a dock choosing coast gains one fuel, capped at 12.

A stopped cargo interaction picks up an unheld item on the same tile if empty. Holding an item at its own destination delivers it; elsewhere the same action drops it if no unheld cargo already occupies the tile. At most one cargo can be carried by a boat, but boats may navigate over unheld cargo. A laden independent boat is capped at speed one. Loading, delivering and dropping consume the command and cannot occur while initially moving.

Two adjacent initially stopped boats both choosing tow form a symmetric tether; the lower-index boat is the tug. No chains or third member are allowed. A tether synchronizes heading and speed to the tug. An empty tug can carry a laden follower at speed two, whereas a laden tug remains capped at one. Towing spends one extra tug fuel per moving tick and stalls both boats if that fuel is unavailable. A follower's directional command does not take control or consume throttle fuel. Either partner can brake both; either may release the tether. If a tether is released while moving, existing momentum remains.

Motion resolves up to two real one-cell crossings. Simultaneous movement into the same target or a head-on edge swap is blocked. Moving into a tile that another boat vacates is allowed, enabling a tandem convoy. Blocking propagates through occupied targets and tether partners before movement is committed. A collision stops affected momentum. A tug therefore cannot pull through a bank while leaving the follower behind, and an opposing boat's passage decision changes the feasible crossings of the other boats.

A colliding carrier's cargo takes one damage per tick, capped at 3; at 3 it is permanently lost. A contract outcome gives its original owner +1 for delivery or -1 for loss, and every role also receives +0.25 or -0.25. The role physically performing a delivery need not be its contract owner. Each colliding boat additionally receives -0.125. All cargo resolved ends the episode; otherwise the horizon is 80. Completion can include losses and is not reported as successful delivery.

Commands are 0 coast/refuel; 1/2/3/4 directional throttle; 5 brake; 6 cargo interaction; 7 tow consent; 8 release. Two plausible plans are scheduled independent crossings with early basin yielding, or a tug temporarily leaving its own contract to accelerate another owner's load. Empty-tug speed trades against tug fuel, paired consent and rigid convoy geometry. Late braking inside the narrow passage may leave nowhere to yield. Racing a competing delivery can create persistent damage to both owners' cargo. No plan has been measured or optimized.

## Information, visible state and declared bounds

All positions, headings, speeds, fuel, tethers, cargo ownership/holder/damage/status and destinations are public actual state. Complete public state plus own-role one-hot yields 155 float32 columns. Destination entities are active only while their matching cargo remains unresolved. Flat observations retain public final positions/status of resolved cargo; role rendering hides inactive cargo and destination glyphs.

Agent positions remain unique on navigable tiles. Heading is in [0,4], speed in [0,2], fuel in [0,12], and damage in [0,3]. Tethers are symmetric pairs with distinct adjacent members and have no cycles/chains. Each boat carries at most one active cargo. Active plus delivered plus lost remains three cargo items, and resolved statuses cannot reverse. Reward lies within the declared conservative [-2,2] bound. These finite quantities are game rules and interface bounds, not research acceptance gates.

## Literal source evidence

- Original map and cargo contract coordinates: `candidate.py:23`, literal `def reset(self, key):`.
- Exact public state packing: `candidate.py:52`, literal `def observe(self, s):`.
- Stopped physical cargo interaction: `candidate.py:74`, literal `interact = (a[i] == 6) & (s["speed"][i] == 0)`.
- Paired tether consent: `candidate.py:96`, literal `attach = (a[i] == 7) & (partner[i] < 0) & (s["speed"][i] == 0) & jnp.any(eligible)`.
- Momentum and throttle cost: `candidate.py:103`, literal `throttle = (a >= 1) & (a <= 4) & (s["fuel"] > 0) & is_leader`.
- Laden speed and tug synchronization: `candidate.py:114`, literal `heading,speed = heading[leader],speed[leader]`.
- Resource-constrained towing: `candidate.py:117`, literal `tow_fuel_ok = fuel[leader] > 0`.
- Cell-crossing reservation and swap rule: `candidate.py:130`, literal `swap = jnp.any(to_old & jnp.transpose(to_old) & ~jnp.eye(3,dtype=jnp.bool_) &`.
- Blocked occupancy/tether propagation: `candidate.py:133`, literal `for propagation in range(3):`.
- Persistent physical cargo damage: `candidate.py:141`, literal `damage = jnp.minimum(s["cargo_damage"] + (active & (holder >= 0) &`.
- Owner and shared outcome rewards: `candidate.py:158`, literal `rewards = private + jnp.sum(outcomes)*0.25 - crash.astype(jnp.float32)*0.125`.
- Natural end from unresolved cargo: `candidate.py:152`, literal `s["terminated"] = ~jnp.any(active)`.
- Absorbing terminal branch: `candidate.py:167`, literal `def absorb(self, key, state, actions):`.

## Witnesses and possible future variation

The counterflow script attempts a head-on reservation conflict. Its paired branch brakes and turns in the east basin before entering. The intended comparison is feasible motion, stopped momentum, actual cargo damage and returned reward. A second pair changes only follower consent to a stopped tandem tow; it inspects follower motion, speed and tug fuel. These are literal unexecuted hypotheses, not replay receipts or successful deliveries.

Future separately assigned canal widths, dock placement, fuel costs, cargo ownership or towing rules could change whether yielding, solo transport or cooperation is useful. No hidden parameter sweep is implemented, and no adaptation or generalization claim follows from source authorship.

## Provenance, interface and status

Source SHA-256: e079e07b06792718a4201360563bca1e92b24fc4be75d9f6899be8787072c89f.

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
