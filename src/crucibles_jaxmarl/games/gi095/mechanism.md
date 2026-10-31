# Copper Relay

A/O/U/H = 4/393/9/80. Two teams of two surveyors build, defend and cut actual supply networks. Team membership is public and fixed: roles 0 and 1 are west team 0, roles 2 and 3 are east team 1. Homogeneous commands allow construction, raids, defense and resource handoffs.

## Rules and causally coupled decisions

The original 9 by 7 map has 35 interior site slots, with inactive rocks at [4,1] and [4,5]. Home sites [1,3] and [7,3] start owned with durability 3. Other sites start neutral. Every active site's position, owner, durability, linked status and enemy-occupation status are real state. The scoring relay sites are [3,2] and [5,4]; either team may own either relay.

Supply follows cardinally adjacent sites with the same owner from that owner's home. A site occupied by an enemy is excluded from reachability, even if its durability is still positive. Connectivity is recomputed by a bounded 35-hop flood through the actual sites. A physically present raider can therefore interrupt multiple downstream opportunities without receiving an immediate shaping bonus. Home hardware cannot be destroyed, but occupation can interrupt its network.

Each role begins with 4 supply and can carry at most 6. Laying a neutral conduit on the current tile costs 2 and sets durability 3. Repairing a friendly conduit costs 1 and adds one durability up to 3. Cutting chooses the lowest-index enemy nonhome site at Manhattan distance at most 1, costs 1, and removes one durability; zero durability returns a site to neutral. Choices resolve in rotating role order, making same-tick repair/attack order a visible clock-dependent rule.

Two adjacent teammates both choosing transfer move one unit from the richer pack to the poorer pack. Each participant transfers at most once that tick. Restocking on a connected friendly tile draws up to 2 from the team's shared finite stock, capped by pack capacity. Restocking uses the post-movement network before that step's operations; new construction becomes usable next tick. Both teammates consume the same stock. Every fourth tick each stock gains one up to 24, so excessive private restocking constrains a partner's future construction.

Every third tick, disconnected nonhome conduits decay by one durability. Connectivity is recomputed after this decay. Every fourth tick, each connected owned relay generates one pulse. Each teammate receives 0.5 times own-team pulses minus 0.5 times opposing-team pulses. Reaching 12 cumulative pulses ends the game; both teams may reach that level together, and a tie is retained. Otherwise the horizon is 80. The number 12 is a game termination rule, not an acceptance or performance gate.

Commands are 0 wait; 1/2/3/4 movement; 5 lay/repair; 6 cut; 7 teammate handoff; 8 restock. Movement into pre-step occupied cells and competing destinations is blocked. A defender standing on a relay can stop a raider from entering; leaving for supplies opens that physical opportunity. Enemy-controlled tiles otherwise remain traversable.

Plausible plans include short defended relay connections, or two-branch expansion with one teammate resupplying and one raiding. A redundant path costs extra carried supply but can survive a cut that a cheap single chain cannot. Holding a relay may preserve production while foregoing repairs or a raid elsewhere. Teammates who build independently can drain shared stock or leave the same cut vertex undefended. These tradeoffs are mechanistic hypotheses; no policy has been evaluated.

## Information, visible state and declared bounds

All four roles see the same public map, agents, team stocks, site ownership, durability, actual supply links, occupation and scores, plus their own identity. There are 393 float32 columns. Neutral sites are drawn as neutral sites, not fabricated active supply. Rock slots are inactive. Agent team and pack attributes permit a role view to explain resource access without introducing private information.

Agents occupy distinct traversable cells. Site positions are fixed and unique. Pack is in [0,6], durability in [0,3], stock in [0,24], and ownership in {-1,0,1}. Home ownership/durability remain fixed. Linked sites belong to a team, are uncontested and reach its home through actual eligible cells. The two relays bound each role's step reward to [-1,1]. No artificial winner, learned defense or significant advantage is inferred from those definitions.

## Literal source evidence

- Bounded physical supply flood: `candidate.py:12`, literal `def network(self, owner, contested, active, positions):`.
- Enemy occupation changes links: `candidate.py:27`, literal `s["contested"] = s["cell_active"] & (s["owner"] >= 0) & jnp.any(`.
- Original sites and map: `candidate.py:32`, literal `def reset(self, key):`.
- Public observation packing: `candidate.py:62`, literal `def observe(self, s):`.
- Build cost: `candidate.py:101`, literal `building = (a[i] == 5) & (owner[here] < 0) & (pack[i] >= 2)`.
- Enemy-cut opportunity: `candidate.py:108`, literal `targets = ((owner >= 0) & (owner != team) & ~home_cell & s["cell_active"] &`.
- Paired finite-resource transfer: `candidate.py:120`, literal `give = (a[i] == 7) & ~used[i] & (pack[i] > 0) & jnp.any(partners)`.
- Linked shared-stock restocking: `candidate.py:126`, literal `take = (a[i] == 8) & (s["owner"][here] == team) & s["linked"][here] & (pack[i] < 6) & (stock[team] > 0)`.
- Delayed decay: `candidate.py:133`, literal `decay = s["cell_active"] & (owner >= 0) & ~s["linked"] & ~home_cell & (jnp.mod(t,3) == 0)`.
- Actual relay pulse scoring: `candidate.py:138`, literal `pulses = jnp.stack([jnp.sum((s["relay"] & s["linked"] & (owner == team)).astype(jnp.int32),dtype=jnp.int32)`.
- Natural end and tie preservation: `candidate.py:145`, literal `s["terminated"] = jnp.any(s["score"] >= 12)`.
- Absorbing terminal branch: `candidate.py:158`, literal `def absorb(self, key, state, actions):`.

## Witnesses and possible future variation

One script attempts two initial connected networks, then west role 1 enters east's vacated relay. Its paired branch keeps the east defender in place. The intended comparison is physical entry, contested, linked, durability and a later pulse, not a claimed winning raid. A second script and pair isolate consent to a teammate supply transfer. Both remain unexecuted.

Future separately assigned layouts, home positions, relay placement, replenishment timing or team composition could change whether defense, redundant links or raids are sensible. Those variants are not implemented as hidden reset randomness and do not constitute measured transfer.

## Provenance, interface and status

Source SHA-256: 2040229c883b0abaa228b1d952c982d6fa7fb453b0371148dd299504e6a88af3.

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
