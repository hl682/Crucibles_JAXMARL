# Nine Shards Council

A=3, O=80, U=19, H=96. The workshop is a 7 by 7 grid. The nine mosaic cells are (2..4,2..4), in row-major order. Both workshop floor and mosaic substrate are walkable, including already laid tiles, so the centre cell is physically reachable. A tile must be placed on a cardinally adjacent board cell, not under the acting role's feet. Agents cannot occupy the same physical cell. The communal urn is (3,0); drawing is possible at Manhattan distance at most one.

## Recipe, finite resources and decision order

Role i holds the private target row i, corresponding to y=2+i. At reset a uniformly permuted row-offset vector and an independently permuted colour vector are drawn using fold_in(key,0) and fold_in(key,1). Target[i,j]=colours[(rows[i]+j) mod 3]. Consequently each row and column contains one of every colour, and the board needs exactly three tiles of each colour. This published construction rule permits logical inference; it does not reveal the random row assignment directly.

Role i begins with one colour-i tile, selected colour i and two ink. The communal supply contains two of each colour. Thus exactly nine tiles exist. Public hand capacity is three total tiles per role. Actions 5..7 select colours 0..2 without consuming them. Actions 8..11 lay the selected colour north/east/south/west respectively, requiring an empty actual board cell and a tile in hand. A wrong tile is still a legal, irreversible placement. Tiles cannot be removed or recoloured.

Action 12 gives one selected tile to an adjacent role with hand capacity. If two receivers qualify, use the first role in cyclic order after the giver. Actions 13..15 draw colours 0..2 near the urn. Actions 16..18 reveal columns 0..2 of one's own row, consuming one ink and binding a truthful colour hint to that exact board cell. Already disclosed cells cannot be revealed again. These three clue symbols are finite actions, not arbitrary language, and a role cannot disclose another role's hidden fragment.

All roles act simultaneously against input-state masks. Movement resolves first: wall/out-of-bounds attempts, attempts into another role's old cell, and duplicate proposed destinations stay put; no swaps or immediate following. Colour selections then update. Interactions are resolved in the public order (input_t+0) mod 3, (input_t+1) mod 3, (input_t+2) mod 3. Each placement, gift, draw and disclosure rechecks its current resources, so a legal contested request may lose to an earlier action without duplicating tiles. A newly received tile cannot make an action legal retroactively if the input-state mask disallowed it.

All preterminal rewards are zero. On the step filling the final empty cell, each role receives 2*(number of correct cells/9)-1 and the episode naturally terminates. The terminal board is evaluated even if mistakes were made. H=96 truncates unfinished work without a correctness payout. No individual correctness marker or dense reward leaks a hidden recipe entry before commitment is complete.

## Cross-agent effects and alternative conventions

A truthful hint changes partners' available knowledge while using a finite resource and decision tick. A draw removes a tile from everybody else's communal options. A gift changes a partner's feasible placement colours. A placement changes a shared physical board permanently and consumes a tile needed elsewhere; incorrect early placement may force a lower final match count because no repair operation exists. Agents also obstruct access paths and the urn.

Roles could specialize by their private target rows, converge to the urn and exchange supplies, or disclose informative cells so a well-positioned partner lays multiple rows. A team can use the Latin construction rule to infer undisclosed cells from fewer clues. The usefulness of one more hint therefore depends on what partners already know and where they can place tiles. Rushing a contested cell can save travel while precluding the colour that a better-informed partner intended. These are potential conventions and tradeoffs, not observed learned behavior.

The first paired hypothesis changes role 0's disclosure while other roles wait; the intended effect is a durable change in role 1's native public-hint observation. The second moves role 1 next to the same cell as role 0 and changes whether role 0 commits its tile. At input t=3 the priority starts with role 0, so the intended difference is which colour occupies the contested cell and whether role 1 retains its tile. The main fixed script requests truthful clues, three irreversible placements, an urn draw and a physical tile gift. It makes no claim that those randomly targeted placements are correct or that the whole mosaic is solved.

Different clue budgets, public construction rules, initial inventories, urn locations or interaction priorities would favor different conventions. They are suggested axes for later independently frozen tasks, not evidence that this source generalizes.

## State visibility and intended invariants

Public agents bind actual positions, hand counts, selected colours and ink. Recipe-fragment entities travel at their steward's physical position and use an identity visibility matrix. Board entities bind the actual nine placement cells. Hint entities share those cells but activate only when their true colour has been publicly disclosed. The urn binds the live three-colour supply. Scene metadata never exposes the other two recipe rows.

Intended invariants: for each colour, remaining communal supply plus all hands plus placed tiles of that colour is three; total tile count remains nine; no hand exceeds three total tiles; inventory, supply and ink are nonnegative; filled cells and disclosed hints never revert; hints equal the corresponding true recipe entries; active agent positions are distinct and within the 7 by 7 walkable floor. State-shape and metadata checks do not prove these dynamic claims; conservation and information noninterference require later sealed review.

## Literal source evidence

Evidence refers to the frozen candidate.py with SHA-256 729b52d4f841f8cce8d6bfe93771c99cf046dcc2ef7f2056e58ebe4a000ebbc8. Line numbers were obtained by parsing/reading source text without importing it.
- candidate.py:18-49: reset implementation.
- candidate.py:51-70: legal_actions implementation.
- candidate.py:72-142: raw_step implementation.
- candidate.py:144-154: observe implementation.
- candidate.py:12-16: refresh implementation.
- candidate.py:21: recipe = colours[ - Complementary private recipe rows are generated only at reset.
- candidate.py:85: floor = s["tilemap"][safe[:, 1], safe[:, 0]] <= 1 - Mosaic substrate is walkable, including access to its centre.
- candidate.py:99: i = (s["t"] + offset) % 3 - Cyclic priority resolves shared commitments.
- candidate.py:114: hands = hands.at[recipient, colour].add - A spatial gift directly changes another role's resources.
- candidate.py:124: hint = hint.at[cell].set - A finite reveal publishes an actual hidden recipe cell.
- candidate.py:132: correct = jnp.sum(board == s["recipe"].reshape(9)) - Correctness is used for terminal reward only.

## Boundary and source provenance

This package is an initial author artifact, status AUTHORED_NOT_EXECUTED. It has not been imported, reset, stepped, JIT-compiled, vmapped, trained, rendered, or subjected to action search. Source AST screening and the copied data-only manifest/scene validators are separate from R0 and numerical parity. All playthroughs and paired interventions remain UNEXECUTED_AUTHOR_HYPOTHESIS; a later executor must use their literal keys and record actual legality, causes and failures.

Every nonterminal raw call increments scalar int32 t once. Natural termination has priority on the horizon: truncated=(t>=H)&~terminated. done is their union. An input whose done flag is already true preserves every state leaf and its observation, returns zero reward and all dones true, and retains both terminal-cause flags. No raw reset occurs. Invalid indices and actions disallowed by the input-state mask are replaced deterministically with wait. Attempted movement is legal even when the destination is blocked; geometry/collision resolution may leave position unchanged. All externally visible state uses int32, bool or float32 arrays of fixed shapes. The supplied reset key is not duplicated in state.

The agent-keyed JaxMARL adapter, real replay, RGB/animation, browser playback controls, sealed R0, numerical parity, learning and transfer measurement are separate integration/execution work. The author supplies actual state bindings for them, not an acceptance receipt. Inactive roles have action 0 available until joint termination.

Community sources consulted were the supplied JaxMARL README (parallel agent dictionaries and __all__), Pgx README (explicit step keys, legal masks and board-state interfaces), Craftax README (persistent spatial object interaction and a JAX-native environment boundary), and PureJaxRL README (end-to-end functional integration). These are interface/design inspiration only. No upstream source code, layout, game rule, PPO implementation, assets or weights were copied. The supplied Craftax_Baselines_ppo_rnn.py was hash-verified but its body was not used. Craftax mechanics, Pgx board APIs and PureJaxRL training implementations are distinct; no learner replacement is claimed. All source rules in candidate.py were authored prospectively for this assignment.

Only the assigned design/request, supplied source contracts/schema parsers and copied official references were inputs. The 20-file batch manifest and authorization SHA were verified. No prior candidate, project outcome, replay, ranking, training log, heldout file, result_N or other author's directory was read. This source stream is train-only and final_test_used=false. Author identity is a self-report; dispatch provenance remains with the coordinator.

Rule, map, role or resource variations discussed here are research possibilities, not generated extra tasks and not evidence of generalization. Shape/mask/schema checks cannot establish strategic richness, non-isomorphism, solvability, learning or meta-learning. Negative, failed, zero-effect and indeterminate execution findings remain valid outcomes.
