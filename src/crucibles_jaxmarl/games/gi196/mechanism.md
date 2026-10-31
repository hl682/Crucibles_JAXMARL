# Beacon Marches

AUTHORED_NOT_EXECUTED; train-only attempt_00. No runtime, compatibility, replay, solvability, learning or transfer claim.

Native dimensions A=3, O=301, U=12, H=90. Humans and policies use the same reset/raw_step/observe/legal_actions. Human UI and public adapter are later integration work.

Beacon Marches is a territorial frontier board with three mobile couriers, home supply settlements, opposite-bank outposts, three fortified river passes and six relay sockets. Its muted territory palette, flags and beacon counters are independent of the kitchen, quarry and rural scenes. River topology and pass locations are actual movement constraints. Six fixed possible rough-route cells are sampled once from the reset key, creating local information worth revealing without importing a benchmark layout.

Every raw step is one of three explicit phases: council at t mod 3 = 0, march at 1, works at 2. Each polity starts with two carried supply units and six at its own home; capacity is two. Movement is available only in march, and workers do not take imaginary diplomacy actions while travelling. In council at home, an owner may offer next or previous polity passage for one carried supply. An offer is a public row-owner/column-guest timer, not simulated text. Acceptance requires a pre-existing offer, a recipient at home and one recipient supply. Newly created offers cannot be accepted in that same input step. An accepted grant is a non-revocable four-works-tick commitment. It may expire or be renewed through a new offer and acceptance. Simultaneous acceptance consumes an offer even if its owner renewed it that council; both chosen actions retain their actual costs.

During works, loading transfers bank supplies into cargo, delivery removes one cargo supply into a public owner delivery count and adds two defense, or building/refueling consumes one cargo supply. An unbuilt socket becomes the builder's beacon; a built beacon keeps its original owner. Only its owner or an active guest can refuel it. Each investment adds three fuel up to six. Offers, grants and beacon fuel each lose one at the end of a works tick, including the tick on which a beacon is built. Thus a new three-unit beacon is visible with two units after that step. A dark owned beacon still reports its own status to its owner, but reveals no route cells.

Pass ownership is north=1, middle=2, south=0. Entry checks ownership or an accepted grant from that owner. Once inside, expiration does not prevent leaving for an ordinary adjacent cell. A polity can take its own distant pass, invest supplies in a shorter agreement, build information infrastructure, support another polity's relay under a grant, or spend its courier time on direct deliveries. Fuel and time spent on offers and relays are not simultaneously available for outpost supply. A grant changes another role's later movement permissions and relay information; those effects are represented by the actual action mask, map visibility and terrain transitions.

Action order is old-state masks, simultaneous march movement, simultaneous council offers/acceptances, works transfers and relay transactions, works-clock expiry, global t increment, the fifteen-step public defense pressure and terminal evaluation. Pressure occurs at resulting t=15,30,... and removes one defense after any same-step delivery. There is no hidden combat planner. Reward is -0.01 per active role, plus 0.25 per delivery to every role and 0.75 additional to its owner, minus 0.1 on a pressure tick, plus 2 on all-outpost success or minus 2 on natural defense failure.

## Actions

- 0: Wait / decline by doing nothing
- 1: North during march
- 2: East during march
- 3: South during march
- 4: West during march
- 5: Load up to two supplies at own home during works
- 6: Deliver one supply at own outpost during works
- 7: Build or refuel an accessible relay socket during works
- 8: At home in council: offer passage to next role (i+1 mod 3)
- 9: At home in council: offer passage to previous role (i+2 mod 3)
- 10: At home in council: accept next role's pending offer
- 11: At home in council: accept previous role's pending offer

## Information and boundary

Partially observable. Every role sees Manhattan radius two around its courier and radius three around its own powered relays or relays owned by a polity currently granting it passage. Other roles' couriers, cargo and home stocks are masked outside revealed cells; owned home and owned beacon telemetry remain known. Public outposts/defenses, pass locations/owners, pending offers, accepted agreement timers and specified public events are broadcast. The world map declares map_visible and never public=true. Hidden map/objects are -1 in the flat observation with explicit visibility bits. Role renderers bind the same masks; hidden blocked-move events and unrevealed refuel events are not public event channels.

Masked/invalid actions become wait. Old occupied agent cells cannot be entered, even if vacated simultaneously; shared destinations use rotating priority (role - old_t) mod 3. Roles remain active until joint termination and terminal masks permit only wait. Leaves are int32/bool and observations/rewards float32. Signed IDs are int32 losslessly and keys are not copied into state. Only reset uses jax.random.bernoulli(key, 0.35, (6,)) for six declared obstruction candidates. No step randomness, state PRNG key, hidden planner or seed search is used.

Failure when any defense is zero after pressure. Otherwise success when every outpost has at least two deliveries. If these coincide, defense loss prevents success; both are natural completion, with natural causes taking priority over H. Nonterminal calls advance once. Natural causes outrank the cap. Terminal inputs preserve state/clock/observation/causes and return zero reward/all dones true without resetting.

## Invariants and limits

Bank and carried supplies are nonnegative; carried capacity is two. Sum(bank)+sum(carried)+sum(spent)+sum(delivered)=24, where spent includes offers, acceptances and relay investments. A fuel unit is a duration, not a second conserved supply unit. Offers stay 0..3, grants 0..4, fuel 0..6, defenses 0..6 and deliveries 0..2 per polity. Diagonal offers/grants stay zero. Owner IDs are -1 for unbuilt relays and 0..2 otherwise. Couriers are unique and in board bounds. Every hidden local field is masked together with its visibility indicator. Numerical/JIT and visibility enforcement remain unexecuted integration obligations.

Bounds [-5,256] for observations and [-3, 4] for rewards are declared, not measured. Static/data checks do not establish numerical parity or successful play.

## Literal source evidence

- Key-only local obstructions: candidate.py:35, literal 'rough = jax.random.bernoulli(key, 0.35, (6,))'.
- Real relay information sharing: candidate.py:23, literal 's["map_visible"] = local | jnp.any(access[:, :, None, None] & radius[None, :, :, :], axis=1)'.
- Pass permission: candidate.py:68, literal 'permit = (owner == jnp.arange(3)[:, None]) | (s["grants"][owner, jnp.arange(3)[:, None]] > 0)'.
- Finite phase gating: candidate.py:83, literal 'works = s["phase"] == 2'.
- Accept only existing offers: candidate.py:91, literal 'mask = mask.at[:, 10].set(council & (s["offers"][jnp.mod(ids + 1, 3), ids] > 0))'.
- Commitment timer creation: candidate.py:137, literal 's["grants"] = jnp.where(s["accepted_now"], 4, s["grants"])'.
- Supply cost: candidate.py:140, literal 's["spent"] = s["spent"] + cost'.
- Works-tick expiry: candidate.py:163, literal 's["grants"] = jnp.maximum(0, s["grants"] - works)'.
- Natural defense consequence: candidate.py:168, literal 'failure = jnp.any(s["defense"] == 0)'.
- Exact role observations: candidate.py:95, literal 'def role_observation(self, s, i):'.

- refresh: candidate.py:12 through line 30.
- reset: candidate.py:32 through line 59.
- movement: candidate.py:61 through line 72.
- legal_actions: candidate.py:74 through line 93.
- role_observation: candidate.py:95 through line 110.
- observe: candidate.py:112 through line 113.
- advance: candidate.py:115 through line 175.
- absorb: candidate.py:177 through line 179.
- raw_step: candidate.py:181 through line 182.

## Provenance

Literal seed-zero scripts and the one-component paired hypothesis have never run; timing follows source reasoning, not seed search. JaxMARL, Pgx, Craftax and PureJaxRL supplied READMEs provide only API/design context. Craftax game mechanics and Pgx board APIs are distinct from PureJaxRL algorithms. No implementation, benchmark layout, commercial asset or learner code was copied/imported. Craftax PPO source was not read; public performance claims or incidentally displayed scores were not used. All 23 actually read input files and uses are frozen in author_input_freeze.json. Current 194-197 authority controls work, not copied historical IDs. Separate explicit dispatch is required before any repair or numerical execution.
