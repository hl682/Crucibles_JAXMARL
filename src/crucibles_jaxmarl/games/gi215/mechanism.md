# Washwater Bistro

AUTHORED_NOT_EXECUTED. Current initial attempt 00. Requested author gpt-6-astra / max; provider measured model, effort, tokens, elapsed time and cost are unknown.

Two workers begin with different physical utensils: one cooking implement and one cleaning brush. These persist and follow their holders. A single adjacent exchange action swaps capabilities for both workers; simultaneous exchange requests still cause only one swap. Seven item identities comprise two dirty reusable bowls, four raw ingredients and one portable filter. Carrying, placing, washing, preparing, filling and serving act on these same objects.

The common tank has capacity twelve. Preparing an ingredient, washing a bowl and assembling a soup each consume one unit; scrubbing pan contamination/grease consumes two. A pump replenishes two units per use, while washing a held filter at that pump consumes two and restores its capacity. Washing adds dirt to an upstream sink with capacity three. Its valve releases one parcel per tick. Parcels move one actual channel cell per native transition, including a two-step delay from drain emission to the pan. A filter at the second cell removes parcels until exhausted; a diversion at that cell instead fills a three-unit holding basin that slowly empties. The coworker therefore has a real preemptive route/filter choice, a later scrub recovery choice, or an alternative work sequence while shared resources are unavailable.

Assembly consumes a prepared loose ingredient adjacent to the pan and fills a clean held bowl. Each assembly adds grease; three grease units block further preparation until scrubbed. Serving a fresh soup fulfills the lowest-index pending customer and returns the same bowl dirty at the serving location. Food freshness and individual customer patience continue through unrelated work. A scene can distinguish the flowing cause, delayed downstream contamination and the object/tool recovery action.

## Native roles, controls and observations

A/O/U/H = (2, 190, 16, 256). U=16 is compact and contextual. Tool identity, proximity and held objects determine preparation/wash/clean affordances; the role mask changes after a native physical tool exchange. Human and AI controllers use exactly the same scalar int32 joint-action interface and native state transition. The shared platform can assign any subset of roles to humans. No separate browser dynamics are provided.

Fully observable compact kitchen. Both roles receive identical flat native maps, actors/items/tools and holders, clean/prepared/filled/age/spoiled flags, every drainage-cell parcel, tank/sink/basin/filter/pan state, customer patience/outcomes and reward components. Inactive ingredients and completed orders remain fixed slots in observations but disappear from active scene groups.

0: Wait.
1: Walk north.
2: Walk east.
3: Walk south.
4: Walk west.
5: Pick lowest-index adjacent loose bowl, ingredient or filter.
6: Put held item on current floor tile.
7: Use equipped utensil: prepare held food, wash held empty bowl, or scrub pan with empty hands.
8: Pump two clean-water units, or wash a held filter using two units at pump.
9: Toggle upstream sink drain.
10: Toggle channel diversion into holding basin beside valve.
11: Install held filter beside channel valve.
12: Assemble a prepared nearby ingredient into a held clean bowl; cooking utensil required.
13: Serve held fresh clean soup beside serving station.
14: Discard food or empty a filled bowl beside bin.
15: Exchange cooking and cleaning utensils with adjacent coworker.

## Resolution, reward and time

Invalid or mask-false actions become wait. Simultaneous movement rejects old-occupied or duplicate destinations. Interactions alternate actor priority by (t+k) mod 2; eligible objects/orders use lowest index. Utensil exchange is resolved once before individual interactions. Interaction changes precede downstream flow and then aging/patience.

Four native components are shared +6 per newly served order, shared -0.75 per newly contaminated pan, shared -1 per newly spoiled item or deliberate food disposal, and own -0.02 for a valid contextual action code >=7. A spoiled dish later disposed of incurs both separately named events. Four original food units exist across active ingredients and filled bowls, so at most four units newly spoil or are disposed in a step; disposal precedes aging. At most two orders can be served. These bounds imply a conservative reward interval [-8,13]. Components are native float32 state, with no generic per-tick living reward.

Natural end when each of the four customers has either been served or left after its own patience expired. H=256 is the upper truncation window; natural causes take priority. Natural causes take priority over H, and already-terminal input is absorbing: unchanged state/observation/clock, zero reward and preserved terminal causes. Raw transitions never reset. Each of four ingredients can be consumed or discarded once. Two bowls and two utensils are reusable and never respawn as new identities. The filter is a movable original object; installation clears its holder and pickup removes installation. Tank, sink, basin and filter capacities remain bounded. Pump input and slow basin drainage are explicit sources/sinks.

## Literal source evidence

candidate.py:L49: swap=jnp.any(ac==15)&(jnp.sum(jnp.abs(n['agent_pos'][0]-n['agent_pos'][1]))<=1)

candidate.py:L61: prep=(a==7)&cook&near[0]&has&(n['kind'][h]==1)&~n['spoiled'][h]&~n['prepared'][h]&~n['pan_dirty']&(n['grease']<3)&(n['tank']>0)

candidate.py:L62: wash=(a==7)&~cook&near[1]&has&(n['kind'][h]==0)&~n['filled'][h]&(n['tank']>0)&(n['sink_dirt']<3)

candidate.py:L63: scrub=(a==7)&~cook&near[0]&~has&(n['tank']>=2)&(n['pan_dirty']|(n['grease']>0))

candidate.py:L79: assemble=(a==12)&cook&near[0]&has&(n['kind'][h]==0)&n['clean'][h]&~n['filled'][h]&~n['pan_dirty']&(n['tank']>0)&jnp.any(ingredients)

candidate.py:L94: shifted=jnp.concatenate((emission.astype(jnp.int32)[None],s['flow'][:3]))

candidate.py:L95: filtered=(shifted[1]>0)&n['filter_installed']&(n['filter_charge']>0)

candidate.py:L97: diverted=(shifted[1]>0)&~filtered&n['divert']&(n['basin']<3)

candidate.py:L100: contamination=(n['flow'][2]>0)&~n['pan_dirty'];n['pan_dirty']=n['pan_dirty']|(n['flow'][2]>0)

candidate.py:L112: n['reward_parts']=jnp.stack((jnp.full((2,),6.0*served,jnp.float32),jnp.full((2,),-0.75*contamination.astype(jnp.float32),jnp.float32),jnp.full((2,),-waste.astype(jnp.float32),jnp.float32),-0.02*(ac>=7).astype(jnp.float32)),axis=1)

candidate.py:L111: n['terminated']=jnp.all(n['order_status']!=0);n['truncated']=(n['t']>=self.max_steps)&~n['terminated'];n['done']=n['terminated']|n['truncated']

## Presentation and evidence boundary

Warm cream kitchen floor, blue sink/counter hardware, terracotta workers, food objects and four actual wastewater markers. Real bowl cleanliness and carried utensils supply the visual affordances. All scene geometry, entity attributes and visibility refer to declared native leaves. Public scene events are restricted to actually public information. Debug/full-state view must be separate from role view.

Two literal scripts and a single actor/action paired hypothesis are supplied, with fixed integer seed and source-derived suffix lengths. All remain UNEXECUTED_AUTHOR_HYPOTHESIS; neither legality of the full trajectory nor any effect, solvability, strategic richness, learning or transfer is certified. Prospective variation could examine water capacities, customer arrival/patience, physical drain routing or partner work sequences under future frozen designs. No such variation or candidate has been run.

Current author checks use stdlib AST/literal shape arithmetic, byte hashes and the supplied trusted manifest/scene/whitelist validators only. No candidate/JAX/NumPy import, native method call, numerical rollout, rendering, training, network/model call or seed search occurred. Four official README excerpts supplied API context only: JaxMARL parallel dictionaries, Pgx explicit-key and terminal interfaces, Craftax spatial game framing, PureJaxRL single-file implementation philosophy. No upstream implementation, layout or asset was copied. The supplied PPO implementation was not read. Exact current input paths, bytes, hashes and excerpt extents are recorded in author_input_freeze.json. Current diversity addendum controls A/H and the scope of open worlds.
