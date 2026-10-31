# Festival Steam Kitchens

Status: AUTHORED_NOT_EXECUTED. Candidate game-v3-00229, source attempt 0, train-only prospective source stream. Assigned author identity: gpt-6-astra / max; root dispatch is the authority for actual model execution. A/O/U/H = 24/160/23/512. Source SHA-256: 97209e05ff4bc4806ce64dc4b090707c2c39e4874da15eae748c9af2ad077583.

## Native game and dependencies

Six kitchens share one connected 30 by 18 floor, a directed steam circuit and reusable dishes. Each kitchen has four individually controlled workers initially holding knife, tongs, tray and wrench tools. Seven physical counter types block movement: pantry, chopping board, pot, cooling rack, serving hatch, valve and sink. A worker can change job by swapping tools with a facing neighbor.

The two ingredient choices are roots and beans. Raw ingredients must be chopped with a knife, then placed in an empty pot that has a clean plate. Progress reaches readiness at 4 steam units. Tongs or a tray can remove a ready meal; that removes the plate from the pot. A carried hot meal cools by one temperature unit per tick but still needs the explicit cooling-rack action before serving. A requested ingredient earns quality 2; substituting the other ingredient earns quality 1. Initial pots already contain one prepared requested meal, at progress zero; later meals use finite pantry stocks and the return-plate chain.

Serving emits the used dirty dish into a reserved physical cart position in the serving lane. That cart blocks entry into its cell, though a worker already there can leave. Other workers can push it into a clear adjacent cell, pick up the dirty dish, wash it using sink water, and bring the clean plate back to a pot. One cart capacity per kitchen prevents overwriting an outstanding return. Cart pickup deactivates its world object; the same dish then exists in the carrier's inventory.

Steam is not a global scalar bonus. Each of six nodes has two transport buffers. An open valve draws at most two units from its arriving buffer. Remaining steam moves into the next node's first buffer, while that node's former first buffer becomes its second. The circuit's node 0 additionally receives two units, capped at 6. A valve decision therefore changes another kitchen's arriving steam after the declared transport delay. The circuit links nodes in index order, including its return from 5 to 0. Native station positions identify the endpoints; pipe buffering is an explicit physical-network rule.

Ready food left on a heated pot accumulates one burn count per heated tick and spoils at 3. Spoiling clears the meal, retaining the pot's plate. Sink water starts at 4, caps at 8, and receives condensation on each eighth tick if that kitchen drew heat. A closed upstream valve can aid a cooking neighbor, overheat its unattended meal, or change future washing capacity.

Plausible plans include a specialist handoff line, swapping tools to follow a changing bottleneck, accepting lower-quality ingredient substitutions, sharing steam in phases, and spending local time to return dishes for another cook. Congestion, food heat, private menus, scarce plate circulation and finite stocks make these partner-dependent choices. Their quality has not been measured.

## Actions, observations and presentation

After common actions 0..8, actions 9..22 collect roots, collect beans, chop, load pot/plate, take cooked food, finish cooling, serve, toggle valve, hand off, swap tools, wash, collect cart contents, push cart and signal. The native mask controls tool and spatial conditions. A handoff also transfers hot-food temperature.

The 160-column row contains 14 self values, a 49-cell two-channel patch, three detailed actor records, local pot/valve/menu records, pantry stock plus its visibility flag, and one explicit zero padding column. A menu is readable only within Manhattan distance 1. Coarse signals travel distance 6; the maximum code is heard. No remote order list, extra actor inventory or new policy information is supplied by the scene.

The warm kitchen scene uses colored counter categories, native carried-item codes, actual pot progress/burn, valve buffers, local menus and movable return carts. Menu visibility is stricter than ordinary object visibility. This is declarative presentation metadata; human controls and animation have not been implemented or tested by this author.

## Rewards and termination

reward_parts has five columns.

- Column 0 charges the actor 0.01 for a successful move, chop, wash or cart push.
- Column 1 gives the serving actor 0.4 times the actual quality.
- Column 2 gives each of the served kitchen's four workers 0.25 times quality.
- Column 3 charges each worker of a missed-deadline kitchen 0.2 and each worker of a spoiled-meal kitchen 0.075.
- Column 4 charges each worker 0.005 per steam unit drawn by its kitchen. This makes the native energy tradeoff visible in payoffs as well as downstream state.

A deadline starts at 80, counts down while that kitchen has fewer than two completed orders, and resets to 80 on service or expiry. It alternates the wanted ingredient after each service. Expiry does not reset the world. The post-action environmental pass can decrement a freshly reset deadline to 79.

The reward interval [-32, 32] is a conservative bound from 24 action resolutions, quality at most 2, and one spoil/deadline/energy pass. It is not a reward target. Natural completion is two served orders at every kitchen; H=512 otherwise truncates. These order quantities are native task rules, not scientific admission criteria.

## Hypotheses and limits

The first literal script attempts one complete plate circulation: take the initial cooked meal, cool, serve, pick up the blocking dish, wash, and restore the plate. It does not claim to finish both orders. The second script coordinates bean preparation, hot-food handoff, a tool swap and the initial root delivery; its prepared bean remains intermediate inventory.

The paired hypothesis changes only engineer 3's first valve action. Both branches share the same seed and all waiting suffix actions. The intended delayed readout is node 1's steam buffers, heat, pot progress and burn. More downstream heat need not be beneficial. No script, mask or effect was executed. JAX compilation, public adapter, animation, throughput, solvability, learning and transfer remain unverified.

## Literal source evidence

These line references identify authored implementation expressions, not measured effects.

- Station/tool/plate conditions: candidate.py line 35, literal source: `def legal_row(self, s, i):`.
- Local observation and menu restriction: candidate.py line 113, literal source: `def observe(self, s):`.
- Ingredient substitution payoff: candidate.py line 197, literal source: `quality = 1 + ((old_carry - 6) == z["wanted"][g]).astype(jnp.int32)`.
- Physical returned dish location: candidate.py line 201, literal source: `z["cart_pos"] = z["cart_pos"].at[g].set(jnp.where(serve, z["menu_pos"][g] + jnp.array([0, 1], jnp.int32), z["cart_pos"][g]))`.
- Steam siphoning: candidate.py line 231, literal source: `draw = jnp.where(z["valve"], jnp.minimum(2, z["steam"][:, 1]), 0)`.
- Downstream delayed transport: candidate.py line 232, literal source: `stage0 = jnp.roll(z["steam"][:, 1] - draw, 1)`.
- Actual spoil event: candidate.py line 240, literal source: `z["spoiled"] = cooking & (z["burn"] == 3)`.
- Absorbing native boundary: candidate.py line 260, literal source: `def raw_step(self, key, state, actions):`.

## Read-only reference use

Only the current batch's authorized files were read. author_input_freeze.json records all 26 actual input files, bytes, hashes and use. The 24 payload files include this batch's four current designs/requests, shared contracts, current population/family control, general design and human-play requirements, and five named official reference snapshots. Three optional historical population addenda were not read. Earlier copied example numbers are noncontrolling.

JaxMARL README informed the distinction between a simultaneous native action-row boundary and a later agent-keyed dictionary adapter. Pgx README informed explicit legal masks, separate termination causes and external key provenance; no Pgx game was copied. Craftax README informed bounded persistent world state and the separation between mechanics and rendering. PureJaxRL README and Craftax_Baselines_ppo_rnn.py were read only for JAX scan/vectorization and wrapper/algorithm separation. No upstream game code, PPO code, assets or performance claims were copied into this kernel. No reference links were opened, dependencies installed, local models invoked or scientific outcomes read.

## Boundary and evidence status

The native interface is reset(key), observe(state), legal_actions(state), and raw_step(key,state,actions). All state leaves are fixed-shape int32, bool or float32 arrays declared in manifest.json. Keys are accepted but intentionally unused: these original rules and layouts are deterministic. Small integer quantities use int32 at the boundary; there is no hidden PRNG key in state.

The joint input is one int32 action per actor. Resolution uses rotating priority (k + input t) modulo A. Each actor's action is rechecked against the current intermediate state, so contention can turn a previously legal intention into a deterministic wait. Environmental processes run once afterward. This is a simultaneous input interface with explicitly sequential conflict resolution, not a claim that actions commute.

A nonterminal step advances the int32 clock once. Natural completion takes priority over H. An already-terminal raw call preserves the entire state, observation, clock and terminal causes and returns zero reward; it never auto-resets. Event flags refer to the last actual nonabsorbing step and are not new events on repeated absorbing calls. A separate trusted wrapper or public adapter remains outside this author package.

No candidate was imported or executed. No JAX/NumPy runtime, validator, trajectory script, local model, remote job, HTTP endpoint, browser, training or evaluation was run. Standard-library AST parsing and data-only JSON/static consistency checks are not native validation. Source authoring is not an R0, causal, animation, solvability, learning or transfer receipt. No additional proposal or repair attempt was created.

