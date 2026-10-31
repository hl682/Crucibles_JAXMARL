# Bonded Bridges

**AUTHOR_ONLY_NOT_EXECUTED.** This is initial parentless proposal 153, source attempt 0, in the frozen train-only window. Candidate runtime execution, numerical R0, replay, rendered animation, learning and scientific acceptance have not occurred in this authoring task. Python AST/syntax and declarative data-shape/schema screens are authoring checks only; they do not establish any game outcome.

Native contract: A=4, O=182, U=14, H=160. Observations are float32 `[A,O]`, actions int32 `[A]`, rewards float32 `[A]`, masks bool `[A,U]`, and state leaves are fixed-shape int32/bool arrays. Source SHA-256: `e9d1a6de2281874cecfbf6b2f2ce57605925694d281ce97040c40634e54779c6`.

## Task and rules

Transport four complementary parcels to their bilateral private target depots by negotiating scarce expiring bridge access; choose toll revenue, prompt passage and reciprocal delivery assistance.

A river divides the world; the only traversable river cells are the two narrow bridges. North issuer 0 and south issuer 1 each transport a parcel and also need the other issuer's bridge for their own crossing. Merchants 2 and 3 can request either route. This embeds reciprocal access in reachable spatial opportunities rather than attaching decorative contracts to unrelated movement.

A request at an approach locks the currently advertised toll and lasts four ticks. A grant reads the prior step's public requests, takes an issuer action near its control point, locks one issuer token in escrow and names one beneficiary for six ticks. An issuer cannot grant itself. Contending requests are chosen by rotating agent priority. A last-tick permit may still be used before timers decay. Once a beneficiary enters, its quote is paid to the issuer and the issuer's bond returns; entry consumes the permit. A carrier can subsequently leave or retreat without being trapped by expiry, and a used toll is not refunded for retreat. An unused expired permit returns the bond. New requests are recorded after grants, and simultaneous toll changes affect subsequent quotes rather than rewriting a submitted request.

Issuing a permit ties up a token, consumes a turn and requires the issuer near its control point. While an issuer is transporting its own parcel away from that point, new passage requests may wait. A toll-seeking strategy trades access for future purchasing power; a free-access strategy may obtain reciprocal service sooner. Issuers may coordinate by moving one cell toward their own controls to grant each other's requests and then returning to the foreign bridge. Merchants can choose routes, wait, or carry another role's parcel after a hand-off. These are prospective partner-dependent strategies, not tested outcomes.

Four immutable parcel types have cyclic recipients [1,2,3,0]. Every reset independently samples each parcel's target row as 2 or 4 from the supplied key; its opposite-bank x coordinate is fixed. The sender, recipient and present carrier know that address; others see only public possible depot locations. Cargo does not respawn, hand-offs cannot duplicate it, and each recipient can be fulfilled exactly once. Delivered cargo remains accounted for but is inactive in rendering. A parcel label's type/recipient is fixed by id and is not extra hidden policy information.

The intended token identity is `sum(wallet) + sum(bond) = 8`. Toll payments and bond deposits/refunds change ownership but create no tokens. Permits have one slot per bridge; request lifetime is at most four and permit lifetime at most six. Wallets remain nonnegative under legal moves. Private addresses and local carrier positions bind to actual role visibility. Rule variants could change bridge/control layout, permit duration, toll options or recipient cycles; only sampled target rows vary in this implementation, and no adaptation claim follows from that fact.

## Actions and simultaneous timing

- `0`: Wait.
- `1`: Move north.
- `2`: Move east.
- `3`: Move south.
- `4`: Move west.
- `5`: Request north bridge at its approach; quote current toll.
- `6`: Request south bridge at its approach; quote current toll.
- `7`: As issuer, grant one prior request at own control point and escrow one token.
- `8`: As issuer, toggle advertised own-bridge toll between zero and one.
- `9`: Hand carried parcel to adjacent empty agent 0.
- `10`: Hand carried parcel to adjacent empty agent 1.
- `11`: Hand carried parcel to adjacent empty agent 2.
- `12`: Hand carried parcel to adjacent empty agent 3.
- `13`: Attempt delivery of held parcel at its private target.

Movement rejects another agent's start-of-tick cell even if that agent moves; swaps are blocked. Among proposals to the same empty cell, the smallest (agent_id - t) mod A wins. Terrain or joint contention may make a legal movement attempt stationary.

Out-of-range or masked actions deterministically become action 0. Legal interaction attempts may still do nothing when their resolution-phase prerequisites are absent; the action labels state their intended operation. All roles remain active until joint termination, so no eliminated role is silently deleted. Terminal masks retain action 0.

## Observation and renderer contract

Public terrain and contract board includes wallets, requests, quotes, permit lifetimes, grants, expiries and receipts. Moving agents and active parcels have Manhattan-radius-3 visibility without wall occlusion. Each random delivery address is visible only to its original sender, named recipient and current carrier. Full-state debug views must be labeled separately.

- Columns `0:7` (self): Own x,y,id,carried parcel id,received count,clock,joint done.
- Columns `7:53` (public): Contract board: wallets[4]; request triples gate,ttl,quote for four agents[12]; bridge records x,y,beneficiary,ttl,advertised toll,locked price,bond[14]; delivery flags[4],crossing flags[4],grant flags[2],expiry flags[2],received counts[4].
- Columns `53:73` (local): Four agent records: x,y,carried parcel,in-transit bridge,visible. Radius 3 without wall occlusion; hidden fields -1.
- Columns `73:93` (local): Four parcel records: x,y,carrier,active,visible. Active parcels within radius 3; immutable type equals record id and recipients are [1,2,3,0].
- Columns `93:105` (local): Four private address records: x,y,visible. Known to original sender, named recipient and current carrier; otherwise -1. Delivered target becomes inactive and hidden.
- Columns `105:182` (public): Static 7 by 11 public tile map, row-major yx.

All coordinates are integer `[x,y]`; the tile map uses `[y,x]` indexing and a downward y axis. `scene.json` binds these actual map, position, active, visibility, attribute and event paths. It contains no invented motion, external image, executable markup or policy-only information. Public event channels are also present in observations. Counters and persistent levels are labeled separately from per-step events. Full-state inspection must be distinguished from role-visible playback. A real trajectory and its animation remain to be produced by the trusted implementation process.

## Termination, rewards and conservation

Terminate when all four unique parcels have been delivered. Otherwise truncate at 160; undelivered demands remain explicitly unsatisfied.

A nonterminal call advances the int32 clock once. Natural completion takes precedence over the horizon. Terminal raw calls preserve every leaf, observation and clock, return zero rewards and all dones true, and preserve terminated/truncated. No episode reset occurs in raw_step.

Declared finite observation bounds are -1..160; reward bounds are -0.1..2. These are game data bounds, not outcome thresholds. Rule-specific reward terms and conservation claims above are literal source design claims; they remain untested numerically. `manifest.json` declares every state leaf and safe structural invariants; source-dependent accounting identities require independent execution review.

## Source-line evidence

- Bilateral address and local entity visibility: `candidate.py:12` — `def refresh(self, s):`
- Explicit-key private demand sampling: `candidate.py:34` — `target_y = jnp.where(jax.random.bernoulli(key,0.5,(4,)),2,4).astype(jnp.int32)`
- Permits constrain movement masks: `candidate.py:61` — `def legal_actions(self, s):`
- Actual bridge occupancy and toll transfer: `candidate.py:113` — `# Entering consumes one permit and transfers its quoted toll; escrow returns to issuer.`
- Refund and expiry: `candidate.py:123` — `# Last-tick permits were usable above. Unused expiring bonds are now refunded.`
- Bonded delayed grants: `candidate.py:132` — `# Grants read the previous public requests, require a real control-point visit and bond.`
- Quote locking and request lifetime: `candidate.py:148` — `# Requests posted now are actionable next tick and lock the old advertised price.`
- Parcel-preserving hand-offs: `candidate.py:157` — `# Hand-offs preserve parcel identity, recipient and private delivery address.`
- Complementary delivery rewards: `candidate.py:166` — `for p in range(4):`
- Public versus role-specific observation construction: `candidate.py:187` — `def observe(self, s):`

## Prospective playthroughs and single-action interventions

`playthrough.json` contains literal source-derived scripts and paired branches that change exactly one role action at one decision. Each script is explicitly `UNEXECUTED_AUTHOR_HYPOTHESIS`; its action dimensions and ranges were checked as data, while runtime legality, actual trajectories, numerical parity and success were not checked. No script result, score, learning gain or browser acceptance is invented. The paired hypotheses identify state paths for later comparison and do not substitute for measured causal traces.

## Community interface and provenance

The supplied official JaxMARL README informed the separation between the homogeneous array kernel and a later agent-keyed parallel API with `__all__`; no upstream environment was copied. Pgx informed explicit legality, raw terminal state and explicit-key provenance, rather than importing a board-game implementation. Craftax supplied a reference for spatial objects, inventories and persistent consequences; this game is an original multiplayer design. PureJaxRL is an algorithm implementation reference, not a game library. The supplied Craftax recurrent PPO baseline credits PureJaxRL; its training/network/wrapper code was not copied, run or substituted for the frozen learner.

Reference snapshots are hash-frozen in `author_input_freeze.json` and the assigned request: [JaxMARL README](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), [Pgx README](https://github.com/sotetsuk/pgx/blob/main/README.md), [Craftax README](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), [PureJaxRL README](https://github.com/luchris429/purejaxrl/blob/main/README.md), and [Craftax recurrent PPO baseline](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py). API/design ideas only; copied upstream code: none.

The public JaxMARL adapter belongs to the trusted integration layer: it should map fixed ids `agent_0` through `agent_3` to the existing array boundary, expose masks and set done `__all__` from `done_all`. No agent dictionary adapter or new learner is smuggled into this restricted one-class kernel. Source integer categories use int32, flags use bool, and the supplied uint32 reset key is external provenance rather than a hidden mutable state object.

The source/manifest/scene hashes are bound in `generation_record.json`; `author_input_freeze.json` binds the verified frozen author inputs. No prior candidate, result, replay, rank, training feedback or final-test material was read. No numeric candidate execution, CSD3 operation, model download or background task was performed. Initial-window accounting remains four parentless proposals 150..153, 154 used or reserved of the existing 512 cap, with later windows unchanged. Authoring and visualization support alone do not establish learnability, non-isomorphism, generalization or meta-learning.
