# Thawforge Caravan — authored rules, not executed

Status: AUTHORED_NOT_EXECUTED. Three agents; 11 by 9 public grid; 12 discrete actions; 209 float32 observations; horizon 192. All resource and signed integer state uses int32. The supplied key is accepted but this version is deterministic and does not store or synthesize a PRNG key. Source is original; no upstream game or learner code was copied.

## Interaction structure

The crew share one cold-exposed workshop. Miner, weaver and engineer have different gathering or processing advantages, but every crew member can perform each ordinary job. Two ore and one timber become one plate after four labor; two reeds and one timber become one gasket after four labor. The actual objects, buffers and unfinished jobs persist when the first worker leaves. Carrying is spatial and limited to six units. Resource accounting and the production transition are in candidate.py:129-245; the world and all initial inventories are in candidate.py:16-56.

A worker loading the foundry consumes its private carried ore/timber and creates another worker's opportunity to operate it. Weaver loom labor and engineer foundry labor count twice, so finishing a partner's preparation can outperform everyone gathering their own complete recipe. Handoffs move an actual item between adjacent bags; a receiver's future inventory and remaining carrying capacity change. Shared buffers are scarce: taking a product prevents another worker from taking that same product. `ev_load`, `ev_take`, `ev_pass`, `ev_craft`, bag and station-stock changes expose these causal transitions in a real replay.

The hearth burns one stored fuel turn each transition even when no one uses it. Loading one timber buys twelve fuel turns, while an emergency field fire consumes one timber for ten self warmth and five warmth for each adjacent teammate. Fuel also competes with both recipes. A crew member can keep another alive by choosing a fuel run or a nearby fire; it can also exhaust the common material budget on its own warmth. Cold lasts in a public twelve-turn cycle: the final four turns add one extra warmth loss. An optional plate-plus-gasket shelter upgrade lowers future cold loss and expands the heating radius, but consumes the exact same products needed for rescue. This is an intertemporal allocation decision, not a separate score counter.

Concrete partner-dependent response: if the engineer has already loaded foundry timber, the miner should usually load ore or supply labor rather than carry another timber there. If the engineer instead spends that timber on a field fire, the miner's previous operate action is unavailable until somebody loads fuel. A second concrete case is the final rescue: if one beacon operator leaves, charging loses one unit per step; the remaining operator cannot finish alone and may rationally return for fuel or recruit a replacement. At least two physical adjacent agents, complete plate/gasket installation, and live hearth fuel are jointly necessary for each effective charge increment.

The cooperative return is shared, but scarce products, buffer access, carrying room, and movement cells create coordination conflicts. Plausible unmeasured strategies include early insulation and a protected manufacturing hub; direct rescue assembly with field fires; and specialist gathering with relay handoffs versus flexible nearby workers. No best strategy, learning gain, transfer effect or non-isomorphism has been measured.

## Transition and controls

Actions 0–4 wait or move north/east/south/west; 5 harvests; 6 loads one eligible item; 7 takes a product; 8 operates; 9 hands an item to the adjacent least-loaded teammate; 10 rests in the active heated radius; 11 burns carried timber. Options are in candidate.py:109-123, with load/recipe prerequisites in candidate.py:85-95 and candidate.py:97-107. Ambiguous eligible deposits or stations choose lowest object index; loading then chooses lowest item index. Passing ties choose lowest teammate index and passes the first nonzero item type. Resource contention resolves in rotating order `(t+offset) mod 3`.

Movement is simultaneous over proposals. Initially occupied cells are unavailable even if the occupant proposes leaving; shared empty targets resolve by rotating priority. Thus no swaps or overlapping live crew positions arise. Positions actually govern gathering, station interaction, warmth and handoffs. See candidate.py:77-83 and candidate.py:129-245.

In each nonterminal transition: sanitize illegal/out-of-range actions to wait; move; resolve inventory actions in rotating order; pool labor and transform materials; invest in insulation; update synchronized beacon charge; burn fuel and apply warmth; advance the clock; record natural or horizon completion. Interaction masks describe the pre-step state, so a same-step newly loaded input does not retroactively legalize another agent's previously illegal operate command. Recipe work itself persists and may be resumed by another agent.

Shared reward per transition is `-0.005 + 0.10 * newly manufactured products + 0.15 * newly installed beacon items + 0.20 * insulation upgrade + 3 * successful rescue - 3 * freezing failure`. Reward bounds [-4,4] are conservative numerical bounds, not acceptance or scientific performance gates. No reward is assigned to repeated pickup/deposit of the same product. Recipe output buffers cap at four, and installed beacon demand caps at three plates and two gaskets.

## Observation and presentation

candidate.py:58-75 supplies the full public dynamic state needed for decisions, normalized to [0,1], plus observer one-hot identity. The manifest exhaustively labels all 209 columns. The terrain is public, every entity visibility mask is true, and no policy gets renderer-only information. Facing and one-step event fields merely explain the last visible transition. Scene positions, warmth bars, inventory counts, remaining station work, insulation and heat bind to real state; no cosmetic animation supplies missing game logic.

The scene shows finite deposits, physical crew, foundry/loom buffers, the hearth, shelter and rescue beacon. `ev_harvest`, `ev_load`, `ev_take`, `ev_pass`, `ev_fire`, `ev_craft` and `ev_upgrade` are one-transition flags. `manufactured` is cumulative and `success` is a level. A renderer must not emit repeated craft events from a persistent output count.

## Termination, invariants and later checks

Rescue succeeds at beacon charge three only if every crew member is still warm. Any zero warmth naturally fails. Otherwise the clock truncates at 192. Natural completion wins a simultaneous horizon boundary. candidate.py:247-248 dispatches to absorbing candidate.py:125-127: an already-done input preserves every leaf and clock, returns zero rewards and true dones, and preserves terminal cause; no raw autoreset occurs.

Declared invariants include unique active crew positions, integer nonnegative inventories and stores, six total carried items per agent, node stock in [0,8], warmth in [0,80], station progress in [0,4], insulation in [0,2], and heat in [0,60]. The total bag bound is a documented cross-leaf invariant for later checking; the manifest also provides supported per-leaf range checks. Objects and map cannot move independently.

`playthrough.json` is a 42-step manually constructed, unexecuted hypothesis that aims to demonstrate material transformations, refueling, passing, a shared fire and insulation investment. It is not a successful trajectory receipt. Later sealed replay should inspect actual legality and event timing, and paired action interventions should hold state/key/other actions fixed while varying load versus fire, operate versus leave, and pass versus retain. Compare the next and later partner masks, warmth, remaining products and beacon charge; do not substitute a scalar reward difference alone for mechanism evidence.

Candidate variants could change station geometry, recipe tradeoffs, specialist assignments, heat radius or leakage to change the useful coordination structure. None of those variants is implemented or measured here, and this train-only source does not read heldout tasks. A game count, successful animation or interface pass is not evidence of meta-RL success.

## Public reference use

The supplied JaxMARL README informed simultaneous multi-agent interfaces; Craftax README informed persistent resource transformations and a spatial survival world; Pgx README informed explicit legal-action/state/version boundaries; PureJaxRL README and the supplied Craftax baseline excerpts informed pure array-state and scan-compatible integration. The project interface is deliberately its supplied flat homogeneous subset rather than upstream API compatibility. No upstream source was copied and no learner, environment or reference code was executed. Exact consulted reference bytes are recorded in generation_record.json.
