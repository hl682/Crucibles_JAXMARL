# Canal Charters — original multiplayer infrastructure board

AUTHORED_NOT_EXECUTED. Three player-pawns take turns on a 9 by 9 board; 13 actions, 435 float32 observations, 120-turn horizon. Each player owns one farm but shares the spring and canal network. The initial public trunk favors the middle farm; this explicit spatial asymmetry is part of this authored instance, not an assertion of balanced starts. All signed integer world quantities use int32. The supplied key is unused by the deterministic rule set.

## Strategic relationship, not a board-game skin

A player can lay a directed conduit under its pawn, rotate the current conduit, open/close its shutter, or permanently charter its own conduit as public. These decisions change future water motion through physical cells. Every three player turns, the spring produces two units; each open conduit sends up to two units exactly one edge along its actual direction into another conduit, a farm or the drain. All outgoing transfers are calculated from the same pre-flow water vector, so water cannot skip a newly built chain in one pulse. Farming consumes two actual units; drained, bailed and overflowing units are removed. Source and topology are in candidate.py:12-58; the material-flow transition is in candidate.py:99-178.

The network links private returns to public infrastructure. Entering a foreign conduit pays its owner one coin; rotating it pays a one-coin permit. A player with no coin loses those action opportunities. Tolls transfer currency rather than creating it. Construction costs one finite kit and one coin, while farm harvest creates two owner coins. A charter is irreversible: the owner relinquishes future toll income and unilateral shutter control in exchange for a route that teammates or competitors can traverse and rotate freely. Each player only has six construction kits, so no one can assume unlimited topology changes.

Who affects whom: a west player building a branch and a middle player rotating the public junction can redirect future supply from the middle farm to the west farm. The east player may then wait for a rotation, build an alternative connection, pay for access, or preserve a public routing agreement. A shutter upstream can remove downstream harvests several pulses later and cause stored water to overflow. Paying an owner supplies that owner with the coin needed for its next construction or bailout; refusing the toll changes where both pawns can go next.

Concrete partner-dependent response: once the west branch is built, rotating the junction west can create harvests after a travel delay. If the west player instead closes its branch shutter, the same rotation stocks water upstream and risks a flood; keeping the middle route open or rotating toward the east is then a different reasonable response. Another case: a charter on the shortcut removes a movement toll and lets a coin-poor neighbor reach a routing junction. The neighbor's feasible moves change immediately even before any water arrives.

Mixed incentives are explicit. The owning player receives one reward unit per harvest; every player receives 0.05 per harvest anywhere. Every spill harms all, and flood closure costs all one. Completing three harvests at every farm adds one shared reward. A player can pursue private irrigation or toll income, but doing so may delay collective completion or increase flood loss. There is no asserted equilibrium, performance improvement or learned cooperation.

## Moves, timing and fair accounting

Actions 0–4 pass/move north/east/south/west. Actions 5–8 lay north/east/south/west conduits. Action 9 rotates clockwise, 10 toggles the shutter, 11 charters the current owned conduit, and 12 pays one coin to bail up to two water. All local operations concern the actual pawn cell. Farm, spring and drain terrain cannot receive new conduits; the source conduit cannot be rotated or shuttered. Existing public trunk tiles can be operated by anyone. Masks and active-turn restrictions are in candidate.py:73-93.

Only player `t mod 3` acts. Every other player has exactly no-op as its legal action. The chosen player's illegal/out-of-range action becomes pass, then the nonterminal clock advances exactly once. On the last player's turn of each round, the hydraulic pulse follows the chosen local action. This order lets a late-round edit affect the next actual flow, and gives meaning to turn order without simultaneous movement conflicts.

Each cell holds at most six units after a pulse. Outgoing connections are unpressurized canals: incoming water to a full cell becomes overflow instead of backing pressure through previous edges. The drain removes all water reaching (4,7). Overflow is counted before the farm consumes two units for an available harvest. At most one harvest per farm occurs in each pulse. These are explicit simplified transport rules, not claims about physical fluid fidelity.

Reward for each role is its newly gained harvest plus the common term `0.05 * all new harvests - 0.02 * min(10, spill_this_step) + successful_collective_completion - flood_closure`, less 0.005 only for the player whose turn advanced. Tolls, bailing and construction affect actual coin resources rather than hidden reward bonuses. Reward bounds [-3,3] are conservative interface limits. No construction, charter or rotation bonus can be farmed by repeating an action.

## Visibility and genuine board dynamics

candidate.py:60-71 exposes the public board, pawn positions, currencies, kits, harvests, all conduit owners/directions/shutters/water, turn identity and own observer one-hot. All 435 columns are labeled, every entity visibility is true, and all policies see the same strategic information. All pipe coordinates are real grid cells used by water transfer and legal movement, not display-only coordinates.

The scene binds active conduit tiles, owner labels, shutters, real stored water and `tile_flow` counts to the actual state. A stream animation may only move the recorded outgoing flow along the recorded direction for that hydraulic pulse; `tile_flow` is zero on intervening turns. Farms display actual harvest counts and remaining water. Per-turn construction, rotation, charter, shutter, toll, bail and harvest channels record real state changes. `total_spill` is cumulative and must not be rendered as a fresh event every frame.

## End conditions and remaining verification

The season naturally closes when every farm has at least three harvests or cumulative overflow reaches eighteen. If both happen in one pulse, flood takes priority for the success flag. Otherwise 120 turns truncates, with natural completion taking precedence at that boundary. candidate.py:180-181 and candidate.py:95-97 preserve already-terminal state, observations, clock and actual cause, returning zero rewards; no raw autoreset occurs.

Declared invariants include unique pawn positions, nonnegative coins/kits/harvest/water, finite board coordinates, one owner and one outgoing direction per constructed conduit, tile water in [0,6], per-pulse flow in [0,2], kit stock in [0,6], and coin cap 30. Integer material conservation across transfer, source, farm consumption, bailout, drainage and overflow is a necessary later multi-leaf check; it has not been run. No more than eighty spring units can enter within the horizon.

The 66-turn script is unexecuted and manually schedules branch construction, partial public charter, a real foreign-tile toll, and delayed west-then-east routing. It is a hypothesis about events, not a success certificate. Later paired interventions should compare a partner's open versus closed shutter, public versus private branch access, or clockwise rotation versus passing, while holding other actions and source state fixed. Record the immediate partner mask/coin change and delayed farm/overflow consequences over several pulses.

Prospective variants could alter network geometry, spring timing, toll policy, charter rights or beneficiary placement. Those rule relationships would change negotiation and routing incentives. No heldout variant, trained strategy, measured non-isomorphism or meta-RL outcome is used or claimed here.

## Reference provenance

Supplied Pgx documentation motivated explicit turn ownership, legal-action masks and a readable board. JaxMARL informed multi-agent returned rewards and outcomes; Craftax informed persistent material transformations; PureJaxRL and the supplied Craftax baseline excerpts informed fixed-shape pure-JAX integration. All rules and source are original. No standard chess/go, heldout Hanabi, previous project candidate or upstream game implementation was copied. Only the supplied public references and author contract were read; their exact hashes are recorded.
