# Forest Frontier Commons

**AUTHORED_NOT_EXECUTED** — A128 / O321 / U26 / H896. No candidate execution, numerical acceptance, replay, trained policy, human session or transfer result is claimed.

## Native world and population

Sixteen eight-person settlements occupy a40x40 bounded, connected woodland. There are128 actual independently controllable native agent rows: two woodcutters, farmers, courier/builders and rangers per village. Professions describe initial specialities; exchangeable axes, hoes and buckets determine abilities. The16 pack animals are separate NPC shipment objects and are not counted among128.

The world has64 persistent tree sites,16 crop plots/wells/stores,8 stream bridges and connected orthogonal trail rows/columns. Village centers are spaced10 cells apart. Trees of height3 physically block entry; chopping to2 opens the same cell to settlers and pack animals. The public map supplies actual trail/stream geometry, while tree state supplies dynamic obstruction. Source: candidate.py:32-78, candidate.py:122-128.

Agents carry bounded timber, food and water; tools are persistent state. Stores start with six timber/food. Couriers start with two timber; farmers/rangers carry one food, rangers one water. Trees start mature, crops at stage2 but dry, wells at4. Some bridges are broken. These are original deterministic constants, not seed-search outcomes.

## Actions and step order

0 rest;1–4 cardinal movement;5 chop;6 harvest;7 irrigate;8 fill bucket;9 bridge repair;10/11 load timber/food onto a home-idle pack animal;12/13 deposit timber/food;14 tend crop;15 eat to recover;16 swap equipment;17 aid an injured neighbor;18 make firebreak;19 upgrade store;20 seed crop;21 replant tree;22 extinguish;23/24 withdraw timber/food;25 send a nearby-fire reading to the next village in the row. Every action id has an implemented affordance. Source: candidate.py:130-163, candidate.py:165-261.

Work uses the nearest object of the appropriate group within Manhattan distance1; id resolves ties. A nearby ineligible object may require repositioning. Waiting accepts a tool swap with the lowest-id eligible neighbor, at most one exchange per actor per turn. Aid chooses the lowest-id injured adjacent settler and consumes food first, otherwise water. This is a contextual action protocol, not an implemented free-text agreement system.

Input legality is checked first; invalid commands become0. Actor intents resolve in order(old_t+k) mod128 with updated resource/target checks. Each tree can be chopped at most once that turn; contention can waste a legal intent. Citizen occupancy is exclusive. Moving on meadow costs one stamina; trails cost none. Wait recovers one stamina. Eating consumes food and restores health/stamina. At health0 only wait is legal; another settler can restore health, or a waiting injured settler at a store can consume store food every16 turns. Actors are never removed. Source: candidate.py:165-261, candidate.py:295-352.

After actors,16 pack animals resolve in rotating order, then ecology updates. Loaded animals move at most one actual horizontal trail cell toward the next village in their row, unload if storage permits, and return home empty before accepting another load. Trees, streams/broken bridges, settlers and other animals block trail movement. Store docks permit co-location, so a parked destination animal does not force all villages to launch at once. Source: candidate.py:263-293.

## Ecology and multi-agent consequences

Chopping yields one timber, reduces height and resets regrowth age. A living height1/2 tree ages only when not burning/firebroken, its village well has water, and at least two mature trees lie within distance6. Each24 eligible turns adds a stage. Bare stumps require replanting with food seed. A site can regrow under an occupant; it prevents later entry, while the occupant may leave. Source: candidate.py:165-261, candidate.py:295-352, candidate.py:122-128.

A well recharges by one every10 turns if at least two mature trees lie within distance6. Simultaneous aquifer flow moves one unit west-to-east across each neighboring pair whose level difference is at least2. Drawing water, cutting cover and preserving woodland thus affect another settler's later water opportunities. Watering spends a carried bucket unit to add two soil moisture, capped3. Wet crops naturally grow every6 turns; hoe tending adds growth immediately; soil decays every12. Harvest at6 supplies two food and clears the plot, which then needs seed. Source: candidate.py:165-261, candidate.py:295-352.

Every64 turns a deterministic spark visits a declared tree index. It ignites only mature, dry, non-firebroken woodland; dryness means its owner well has less than6 water. Existing fire can spread to tree sites within Manhattan4 at equal/eastern x, one ecology update at a time. Fire consumes one stage every4 turns. Trees reaching0 stop burning. Nearby crop plots lose growth; smoke within distance3 drains stamina and injures every8 turns. Firebreaks consume timber and clear the site, preventing fire/regrowth until replanted. Bucket water extinguishes; source fire may later spread again. These are actual conditional transitions, not promised ambient animations. Source: candidate.py:295-352, candidate.py:165-261.

Important partner-dependent choices: clearing a shipment route versus preserving mature cover/recharge; sending timber versus retaining it for crossings/firebreaks; spending water on crops versus firefighting or keeping a moist woodland; loaning equipment to relieve a local bottleneck; opening a stream crossing versus walking to another bridge; and rescue versus saving supplies. Local pairs can farm or clear routes while everyone else waits. Cross-village traffic, aquifer flow, fires and finite commissions prevent an interpretation as128 independent single-player copies. No all-to-all interaction or all128-action requirement is imposed.

## Local observation and faithful scene

The321-column native observation contains12 self/public fields and capped local entity slots:8 settlers,12 tree sites and3 each of crops, wells, villages, bridges and pack animals. Visibility is Manhattan5, sorted by distance then id. Every slot has seen and all visible render attributes; unseen slots are zero. Source: candidate.py:12-19, candidate.py:21-30, candidate.py:80-87, candidate.py:89-116.

Fire reports copy the sender's radius-five burning-tree count, capped4, into the next village board; age increments to896. They convey local information but not a global map of fires. Public terrain is static. Dynamic tree stage, fire, crop moisture, stocks, bridges, carried goods and moving shipments come only from native state. Scene events are levels of already-observed fields. Per-step diagnostic flags remain full-state diagnostics, not extra information in a role view.

The visual family is warm rural farming/woodland, separate from the canal federation's territorial water infrastructure. Metadata identifies villages/roles and camera focus suitable for selecting one or multiple human actors. It does not claim a shipped UI, rendered frames or actual human play. The frontend must use this same kernel and legal masks, and label any privileged full-state view as debug.

## Reward components and ranges

Six float32 columns sum to native reward:

1. Actor−0.01 for successful chopping, watering, bridge repair, firebreak, upgrade, planting/seeding, extinguishing, aid or eating.
2. Actor+0.16 per two-food harvest; aid gives helper and injured recipient+0.02 each.
3. Every beneficiary-village member+0.06 per store upgrade; settlers within5 of a repaired bridge+0.005.
4. A matching finite shipment commission gives each recipient member+0.08 and home-village member+0.04. Required cargo is1+destination mod2; orders start4 and decrement permanently.
5. Each member of an extinguished tree's owner village+0.02; each newly rematured tree gives that village+0.002.
6. Smoke injury costs the injured actor0.03; each burning tree-stage loss costs its village members0.02; a local crop loss costs each member0.01.

Sources: candidate.py:165-261, candidate.py:263-293, candidate.py:295-352. Manual transfers/withdrawals, merely loading and idle status pay nothing. Store upgrades consume two carried timber and one food; facilities cap3. Orders are not replenished, avoiding a paid withdraw/redeposit loop. Renewable harvest/forest production still has real time, water and resource consequences.

A conservative lower bound is−0.13: one0.01 expenditure,0.03 injury, at most four own-village tree losses0.08, and0.01 crop loss; manifest−1 safely covers it. An intentionally loose upper bound sums0.16 harvest +0.02 helping +0.06 received aid +0.18 village upgrades +0.12 all24 bridge increments +1.92 all16 possible commissions +0.08 own-village extinguishing +0.008 own-village rematuration =2.548, below manifest4. These analytic envelopes are not measured returns or success criteria.

## Episode and invariants

There is no forced early success/failure endpoint. The896-turn season truncates; construction/harvest does not reset the world. A live call increments t once. Already-terminal raw calls preserve state and observation, return zero reward/all dones, and retain causes. Natural priority holds with terminated always false. Source: candidate.py:295-352, candidate.py:354-356, candidate.py:358-362.

Declared invariants awaiting independent checking: unique in-bounds settler positions; fixed entity capacities; inventories timber/food0–3, water0–2; health0–3/stamina0–6; tree0–3, age0–24; wells0–8; crops0–6/moisture0–3; stores0–32; facilities0–3; orders0–4; bridge0–3; pack cargo0/1/2 and phase0/1/2. All visible values normalize into[0,1]. Int32 counters/coordinates, bool masks and float32 observations/rewards follow the boundary. The external PRNG key is accepted but this original scenario is deterministic.

## Literal scripts and single-action falsification

Two seed0 scripts are unexecuted local proposals. The first chops tree0, loads pack animal0 and clears the next trail cell; the second has a ranger fetch water and irrigate while a farmer tends/harvests. Neither proves a complete delivery, solvability or learning.

Paired branches share the empty prefix and identical subsequent commands. Only actor0's first action changes: chop5 versus wait0. Actor2 loads timber in both; actor0 moves north on step2 in both. From source reasoning, the loaded animal should encounter tree0's cell on step3: the cut branch may permit entry while the uncut branch stalls. Comparison targets the other actor2's visible shipment information, native cart position/stalled/cargo and tree height. No effect magnitude or observed success is recorded. Independent replay may falsify legality, timing or the proposed contrast.

## Provenance

Only current clean226/227 inputs and named official snapshots were read. Population223 controls96/128, with no future population ceiling. Earlier population197/203 payloads, project history, other candidates/results, final-test/heldout data and memory were not used. No network/model API or resource probe occurred.

JaxMARL/Pgx API concepts, Craftax bounded persistence/local visibility and PureJaxRL/Craftax baseline integration context informed the design; no upstream code, art, layouts, weights or benchmark/training results were copied. Input hashes are in author_input_freeze.json; output identities and Astra/max self-report in generation_record.json; parent dispatch independently retains model assignment. Text/JSON/AST/hash checks do not establish source validation, numerical parity, native animation, scalable throughput, human playability, learnability, successful learning, non-isomorphism or transfer.
