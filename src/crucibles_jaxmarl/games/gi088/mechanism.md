# Borrowed Current

**Status: AUTHORED_NOT_EXECUTED.**

Three operators share the work of power production and routing while pursuing private alternating service orders. Crossing a useful passage subtracts from the exact capacitor a waiting customer needs, so spatial access and service allocation cannot be optimized independently.

## Rules and real cross-agent dependencies

A divider at x=6 can be crossed at [6,2] or [6,6]. Its associated stations are [9,2] and [9,6], with initial capacitor levels two and two. The common generator at [2,4] starts with five units, capped at ten. Every supplied unit goes to north, south, or the clock-alternating route, and the source loses that unit. A full destination capacitor stops that turn's supply without consuming buffer. CRANK adds two shared units per cranking role. The roles can trade jobs because each has the same physical interaction capabilities.

The console at [4,4] cycles 2 -> 0 -> 1 -> 2. A switch consumes up to one buffer unit and sets cooldown two. There is no supply on the switch turn or next turn; supply resumes when the decremented cooldown reaches zero. Repeated switching can waste power and perpetually delay a distant partner. A handcell recharge at the generator consumes two shared units to gain one portable unit, with capacity two and one rotating-priority socket. A role may instead carry its charge to a station and donate it to a partner.

After supply is applied, a successful move entering a gate spends one unit from that route's capacitor. A gate without energy cannot be entered. Once occupied, a safety latch displays it open until its occupant leaves, without creating additional charge or allowing another agent to overlap it. Donations are then applied, so a donor can enable another role's service on the same turn. A service consumes two capacitor units, increases the requester's count, and flips its private target. One donation and one service can be accepted per station per turn, each with rotating identity priority. Every fourth turn, all capacitors lose one remaining unit. This creates timing choices between hoarding, travel, donation and immediate service.

Each role needs two completed services. The initial dispatcher starts with a north order, the north customer with north, and the south customer with south. The dispatcher eventually has to leave the console/generator to obtain services; a previously served customer can take over supporting jobs. A finished customer remains able to move, crank, route, donate and signal. This is a joint task with private immediate service rewards, not three separate errands.

## Kernel, information and rewards

A=3, O=54, U=11, H=96. Coordinates are integer [x,y], with y increasing downward. All state leaves are fixed-shape int32, float32 or bool arrays; all observations and rewards are float32.

Private order is always in that role's observation and is not an entity rendering attribute. RADIO consumes the role's action for that turn, publishes its current target, and makes its position, portable charge, completion count and last radio fields visible to all roles for three recorded states. After a service, the private target flips but a prior radio message does not automatically update. Calling again is a real information/timing decision. Outside a live call, agent fields and station capacitors are visible only within Manhattan distance three. Public route/buffer displays, gate indicators, supply pulses and aggregate service count are included in every observation.

The transition applies route changes/cranking/socket charge, one-unit supply, physical walking and crossing cost, local donations, local service, periodic leakage, radio lifetime and terminal assessment. A movement intent may be legal yet blocked by a wall, a closed/empty gate, any other role's old cell or a shared destination. Service, donate and handcell-charge actions are legal intents when their local prerequisites hold, with actual simultaneous energy availability deciding realization. This permits a correctly placed partner donation to enable a service that had insufficient stored energy at the previous observation.

Rewards are -0.01 per live role/turn, an additional -0.03 for a crank and -0.02 for a routing action; a service pays its requester 2 and every role 0.2 per accepted service that turn. Joint completion pays every role 6. The possible per-step values fit [-1,9]. All observations fit [0,1] and have exactly 54 columns. The maximum six services and request quotas are game objectives, not evidence thresholds.

The binding addendum is implemented: a live input advances the scalar int32 clock by one; natural completion takes priority over the horizon. Already-terminal raw calls preserve the complete state, observation, clock, terminal causes and final event flags, return zero rewards and all-true dones. They are not extra gameplay transitions. The only legal terminal action is WAIT. Autoreset belongs to an external wrapper. No individual is removed from the world before joint completion. Invalid or out-of-range actions become WAIT; blocked cardinal directions remain legal attempted movements and may produce no motion.

Each observation slice, entity position, active mask, visibility mask and displayed attribute is declared in manifest.json. scene.json repeats exactly the hash-bound map and declared channels. Its motion comes only from real state positions. A debug full-state view must remain distinct from role views. Public event channels are explicitly present in each role observation; local object/agent attributes are zero-masked together with a visible bit. A hidden entity is not a position hint.

## Literal source evidence

- Partial local information with public tracking calls: candidate.py:L19 — `ns["agent_visible"] = (distance <= 3) | (s["call_ttl"][None, :] > 0)`
- Switching has a persistent timing cost: candidate.py:L124 — `cooldown = jnp.where(switched, 2, jnp.maximum(0, s["cooldown"][0] - 1))`
- Finite shared generation competes with portable charging: candidate.py:L129 — `buffer = buffer - 2 * jnp.sum(charging.astype(jnp.int32))`
- One route receives each common supply unit: candidate.py:L134 — `power_event = jax.nn.one_hot(destination, 2, dtype=jnp.int32) * powered.astype(jnp.int32)`
- Movement consumes a distant service resource: candidate.py:L153 — `capacitor = capacitor - crossing`
- A partner donation enters the same capacitor before service: candidate.py:L157 — `capacitor = capacitor + jnp.sum(at * donating[:, None].astype(jnp.int32), axis=0)`
- Service depends on energy after other roles crossing/donation: candidate.py:L159 — `serving = self.select_local((a == 7) & (capacitor[station] >= 2), station, s["t"])`
- An order changes privately after service: candidate.py:L162 — `order = jnp.where(serving, 1 - s["order"], s["order"])`
- Stored radio messages can become stale: candidate.py:L167 — `call_target = jnp.where(call, s["order"], s["call_target"])`
- All three roles matter to joint completion: candidate.py:L169 — `success = jnp.all(completed == 2)`

Method extents in this exact source:

- `direction`: candidate.py:L12-L14.
- `refresh`: candidate.py:L16-L24.
- `reset`: candidate.py:L26-L70.
- `observe`: candidate.py:L72-L92.
- `local_station`: candidate.py:L94-L96.
- `legal_actions`: candidate.py:L98-L109.
- `select_local`: candidate.py:L111-L115.
- `advance`: candidate.py:L117-L196.
- `absorb`: candidate.py:L198-L201.
- `raw_step`: candidate.py:L203-L204.

## Fixed witnesses and intended intervention differences

These fixed scripts inspect local allocation, delayed supply and information rather than claiming all six services. The crossing pair isolates one mover while holding the remote requester fixed. The first script is intended to show calls becoming stale when private orders flip, real portable-cell donations, and delayed south supply after a routing change. No actual outcome is asserted.

No scripted outcome is an execution receipt. A successful-looking path in prose is only a prediction from the rules. Scripts that exercise local mechanisms intentionally do not assert whole-task completion.

## Invariants, variation and limits

Declared invariants include unique positions and nonnegative clock, buffer, capacitors, handcells, order/call fields and service counters. Construction bounds are buffer<=10, capacitor<=4, handcell<=2, completed<=2, route<=2, cooldown<=2, call_ttl<=3 and service_total<=6. Orders are private dynamic state even though their initial values are fixed by this instance; first service and subsequent timing still change information needs. Other original layouts, private-order distributions, leakage schedules or generator constraints could change partner-dependent allocation strategies; such variants and any learning advantage remain untested.

Integer sign-bearing fields (for example no-owner=-1) remain exact int32 values. There is no design-specified int8 distribution to adapt. The external uint32 PRNG key is not duplicated as a state leaf. The authored reset and transitions are deterministic; keys are accepted but not consumed.

## Provenance and reference use

This is an original authored kernel for initial window0, attempt00, in the train source stream. No prior candidate, project execution result, heldout layout, evaluation score, ranking or final test was read. No candidate or JAX was imported or run. The action lists are fixed hypotheses, not recorded trajectories. Interface/R0 checks, numerical parity, actual legality along the scripts, intervention replays, animation, training, solvability, optimality, strategic richness and meta-learning effects remain unestablished.

Only supplied official references were used. JaxMARL's README informed the intended simultaneous multi-agent adapter boundary (agent-keyed dictionaries and __all__ belong in the later integration layer). Pgx's README informed the separation of explicit transition keys, legal masks and raw terminal state. Craftax's README informed the distinction between a world kernel, state-based rendering and an external reset wrapper. PureJaxRL is an algorithm implementation reference, not a game library; its performance statements are not evidence about this game. Craftax_Baselines/ppo_rnn.py was read only at its environment/reset wrapper interfaces; no PPO/network code or benchmark mechanics were copied. No upstream implementation or map is incorporated. These references are attributed in generation_record.json with exact supplied file hashes and upstream URLs.

The copied community_delivery_requirements.json still lists IDs 82-85. The exact assigned design/request IDs 86-89 control this batch; the general interaction and presentation requirements are applied without rewriting that input. Actual dispatch identity, backend and usage evidence are retained by the coordinator, not invented here.
