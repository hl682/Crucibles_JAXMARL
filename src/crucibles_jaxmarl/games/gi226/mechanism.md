# Tidal Canal Federation

**AUTHORED_NOT_EXECUTED** — A96 / O184 / U19 / H768. No candidate import, execution, numerical validation, replay, successful play, learning or transfer claim.

## Physical world and individually actionable roles

Twelve districts each have two keepers, builders, haulers and growers: all96 are independently controllable, with fixed native action/observation/reward/done rows. The12 barges are physical objects, not extra agents. No global quorum or simultaneous action by all96 is required.

A41x31 public map contains a111-cell serpentine canal, twelve reservoirs, sluice/footbridge sites, stores and fields. Citizen coordinates govern movement, handoffs and work reach. Barge coordinates are indexed from the same canal path used by movement. Land detours remain available. Reset starts upstream water6, other water1, stores timber8/grain0, dry crop maturity2, six town commissions and levee0. Builders carry timber; profession-associated equipment is exchangeable. Source: candidate.py:12-18, candidate.py:40-94, candidate.py:139-145, candidate.py:147-156.

## Native actions and contention

0 wait;1–4 north/east/south/west;5 withdraw timber;6 deposit cargo;7 swap tools;8 hand over cargo;9/10 open/close gate;11 repair crossing;12 irrigate;13 harvest;14/15 tow downstream/upstream;16 load empty barge;17 post upstream water reading downstream;18 raise levee.

Work uses nearest object in the corresponding group within Manhattan distance1; id breaks ties. An ineligible nearest object can require repositioning. Waiting explicitly accepts a neighbor's exchange: lowest eligible citizen id, empty hands for cargo, and at most one exchange per actor per turn. Tools1/2/3 are wrench/can/rope; profession only retains the private keeper forecast. Source: candidate.py:158-182, candidate.py:184-267.

Invalid or masked commands become0. The initial mask is evaluated before changes, then intents resolve in rotating order (old_t+k) mod96 with updated preconditions. Contention may turn a legal intent into a no-op. Citizens cannot enter occupied cells, including cells occupied at initial mask time. Barges move at most once each turn; canal capacity is one, reservoir docking capacity two. Source: candidate.py:269-303, candidate.py:184-267, candidate.py:147-156.

## Delayed water, production and conflicting interests

An open gate spends up to one local water unit and inserts it into slot2 of the next district's three-slot queue. Subsequent shifts move2→1→0, then release the packet: step1 release arrives on step4. The last gate drains outside the federation. A spring supplies two upstream units every12 turns; evaporation removes one every16. Reservoir capacity is8. Source: candidate.py:269-303.

Irrigation immediately spends one reservoir unit for two soil units, capped3. Wet unflooded crops gain maturity every4 turns; soil decays every8; harvest resets maturity6 to0 and puts grain in empty hands. Towing requires water>=2 in the current and next reaches, plus an open boundary gate. Thus upstream irrigation or retention can alter another crew's future observation, action availability and cargo delivery. Source: candidate.py:184-267, candidate.py:147-156, candidate.py:269-303.

Water>=7+levee constitutes flooding. Every8 turns a flooded district loses one crop maturity and bridge strength. A repair consumes carried timber; a levee increment costs carried timber plus two store timber, at most two increments. Deliveries physically move barges to their target reservoir and require storage space. Local deposits, finite recipient/sender commissions and shared flood/navigation rewards have different beneficiaries. Source: candidate.py:184-267, candidate.py:269-303.

Plausible choices include retention versus downstream navigation, irrigation versus towing, bridge investment versus detouring, local grain deposits versus trade cargo, sharing equipment versus retaining control, and posting versus withholding a fresh reading. Their value depends on partner actions and delay. These are source-derived strategic possibilities, not measured strategic richness.

## Local information and presentation

Dynamic visibility is Manhattan radius5 and nearest-distance/id order: eight citizen slots and three slots for each of the other five groups. Each slot begins with seen; all hidden-slot values are zero. Every displayed entity attribute is included in the native slot. Only keepers see the home incoming-water total remotely. Posting copies current upstream water into the next town board, whose age increments to768. Other roles only see nearby boards. Source: candidate.py:20-27, candidate.py:29-38, candidate.py:96-103, candidate.py:105-133.

Civilization-style territories, stores, fields, water and cargo use actual positions/attributes. Event channels are state levels, not invented opening/delivery animations. Additional per-step diagnostic arrays are not role-visible UI channels. Role cameras must obey visibility; full-state inspection needs a debug label. Manifest camera and multiple-actor assignment fields are integration intentions, not an implemented frontend. Human and AI commands must enter the same native transition.

## Rewards and analytic finite bounds

The six columns of reward_components[A,6] sum to returned reward:

1. Acting citizen −0.02 for repair, upgrade or irrigation.
2. Acting citizen +0.10 for harvest.
3. Every member of the receiving district +0.05 per hand-deposited grain; timber deposits pay nothing.
4. Citizens within radius5 of a repaired bridge +0.01; each beneficiary-district member +0.03 per levee increment.
5. A correct finite commission pays each recipient member +0.10 and sender member +0.05. Required cargo is1+(destination+1) mod2. Orders decrement permanently.
6. A damaging flood costs each local member0.04 and everybody0.01 times the damaging-district fraction. Every actual barge delivery gives everybody0.002.

Sources: candidate.py:184-267, candidate.py:269-303. Timber transfers and gate toggles alone do not pay. Grain cannot be withdrawn, so a town can accept at most32 rewarded hand deposits. Conservative lower bound −0.07. A loose upper sum is1.6 for32 grain deposits +0.36 for36 possible crossing increments +0.06 for two own-town levees +1.8 for12 commissions +0.10 personal harvest +0.024 shared deliveries =3.944. Manifest bounds[-1,4] contain these components and are not a performance gate.

## Termination and declared invariants

The finite charter truncates at768; there is no early success/failure threshold or internal reset. A live step increments t once. Terminal calls preserve every state leaf and observation, return zero reward/all done, and preserve terminal causes. Natural priority is respected with terminated always false. Source: candidate.py:269-303, candidate.py:305-307, candidate.py:309-313.

Authored invariants awaiting independent checks: unique citizen positions; fixed in-bounds entities; reservoir0–8; transit cells0/1; bridge/soil0–3; growth0–6; stock0–32; levee0–2; orders0–6; cargo0/1/2; canal index0–110; finite float32 observations in[0,1]. Small fields are int32, masks bool. External key remains accepted; this deterministic original scenario intentionally does not consume it.

## Unexecuted scripts and paired hypothesis

Two literal scripts use seed0. One opens gate0 and waits through the delay. Another has actor6 clear a bank cell while actor2 tows the loaded barge and follows it. They are local diagnostic proposals, not complete solutions or legal-replay receipts.

The paired hypothesis changes exactly actor0's first scalar command:9 versus0; all other commands and subsequent waits match. On source reasoning, the downstream packet should arrive at reservoir1 on step4 and change actor10's local water observation and potentially its tow mask. The JSON names native comparison paths, receiving role and delay, with no observed result or effect magnitude. Independent frozen-source replay may falsify the hypothesis.

## Provenance boundaries

Only current clean226/227 inputs and supplied official snapshots were read; population223 controls and128 is not a future ceiling. Population197/203 payloads were not read. No historical candidate/result, heldout/test task, memory, network search or model API was used.

JaxMARL/Pgx informed API/mask/key conventions; Craftax informed bounded persistent-world and visibility concepts; PureJaxRL/Craftax baseline snapshots provided integration context only. No upstream code/assets/layouts/weights/results were copied. Actual inputs are in author_input_freeze.json, exact output hashes in generation_record.json. Astra/max is author self-report backed separately by parent dispatch. Text/JSON/AST/hash checks do not establish native execution, numerical parity, runtime scalability, playability, learnability, successful learning, non-isomorphism or transfer.
