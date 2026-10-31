# Floodplain Council

AUTHORED_NOT_EXECUTED. Current initial attempt 00. Requested author gpt-6-astra / max; provider measured model, effort, tokens, elapsed time and cost are unknown.

The square floodplain has four settlements with independent bounded stores and physical field regions of Manhattan radius two. Each worker starts with a small material load and a distinct home food valuation. A backpack carries four units total. Resource withdrawal requires the owner's home, while deposits can reach any adjacent settlement; transfers to a peer require adjacency. Harvesting and grain trade are repeatable choices rather than a compulsory global quest.

At most 32 physical water packets enter from north-edge columns that sweep across the floodplain. Rain increases arrivals. Existing packets advance south every second tick; an encountered levee/closed culvert diverts them outward from the center, and a blocked side holds them. Packets leaving the finite map deactivate. A native per-cell depth class counts colocated packets and saturates at four. Levees can therefore protect one approach while diverting water into a neighboring field. Opening a culvert releases its flow path and restores walking access. Raised roads allow movement in deep cells and two-cell hauling, but periodic deep exposure reduces their durability and can erase them. Construction, conversion, dismantling and repair all modify the actual front cell.

Field grain grows on periodic safe pulses; flooded fields lose crop and stored grain. A road builder receives a small utility payment when another worker actually enters its road. Foreign grain deposits benefit both trader and recipient, while field harvesting by another worker imposes an explicit local opportunity cost. Workers may protect, reroute, repair, trade, gift material, wait for water to pass or exchange dated observations. These competing responses are source hypotheses, not measured strategies.

## Native roles, controls and observations

A/O/U/H = (4, 3001, 24, 384). U=24 combines directional construction, three withdrawal resources, material selection/gifts, gates, repair/dismantling, road hauling, harvest/trade and dated report exchange. Four owners have distinct stores and valuations. Human and AI controllers use exactly the same scalar int32 joint-action interface and native state transition. The shared platform can assign any subset of roles to humans. No separate browser dynamics are provided.

Public information consists of clock/terminal causes, rain, aggregate flood-alert count, settlement coordinates and fixed food valuations. Each worker sees map cells, peers and water packets within Manhattan radius four; hidden map/entity records are -1 with a visibility bit. Own settlement stores/field water are continuously available as its explicit home telemetry; other settlement contents require local visibility. Each actor additionally sees its own dated report memory and six reward components. Public safety rewards and the alerts counter reveal only the aggregate exposed-settlement count, not remote identities or crop/store values. The scene map uses the actual map_visible mask.

0: Wait.
1: Walk north and face north.
2: Walk east and face east.
3: Walk south and face south.
4: Walk west and face west.
5: Withdraw one stone from own home; pack capacity four total.
6: Withdraw one timber from own home.
7: Withdraw one grain from own home.
8: Deposit carried resources at adjacent settlement; foreign grain counts as trade.
9: Build levee on front empty cell for two stone.
10: Build raised road on front empty cell for two timber.
11: Convert front levee into open culvert for one stone and one timber.
12: Open or close front culvert.
13: Dismantle front structure; recover one appropriate material.
14: Repair front damaged road with one timber.
15: Harvest one grain from adjacent settlement field.
16: Haul two cells north along existing raised roads.
17: Haul two cells east along existing raised roads.
18: Haul two cells south along existing raised roads.
19: Haul two cells west along existing raised roads.
20: Give one selected resource to lowest-index adjacent peer.
21: Cycle selected resource: stone, timber, grain.
22: Record dated water reports from locally visible settlements and own telemetry.
23: Hand latest stored reports to lowest-index adjacent peer.

## Resolution, reward and time

Invalid/mask-false commands become wait. Moves reject start-occupied and duplicate destinations; a two-cell haul also rejects a start-occupied intermediate cell. A haul is one native movement command along two raised-road cells. Sequential interactions rotate by (t+k) mod 4; eligible home/peer ties use lowest index. Construction precedes packet routing, then crop/flood consequences and visibility refresh. Reports are stamped with the pre-flow source clock and are never refreshed merely by transmission.

Six native components are: trader +2 per foreign grain deposited; recipient +0.2*its food priority per imported grain and harvester +0.4*its priority per unit; shared +0.05 per unexposed settlement on each eight-tick safety pulse; builder -0.2 per consumed construction/repair resource; owner -priority per flood-lost stored grain and -0.15 per grain harvested from its field by another actor; road owner +0.1 per peer entry onto its road. A backpack holds four, there are four workers and at most one flood loss per home on a pulse. Own trade at most 8, remote imports at most 3.84, shared safety at most 0.2 and road entries at most 0.3 give a conservative upper bound below 14. Building/flood/foreign-harvest costs stay above -4. Components are native float32 state, with no zero-sum assumption.

No single settlement quest ends this persistent strategy world. terminated remains false and tick 384 truncates; harvest, trade, construction and flood losses never reset the state. Natural causes take priority over H, and already-terminal input is absorbing: unchanged state/observation/clock, zero reward and preserved terminal causes. Raw transitions never reset. Construction consumes real carried stone/timber; dismantling recovers one unit and cannot duplicate the original investment. Stores cap at twenty per resource and backpacks at four total. Grain regrowth and river inflow/outflow are explicit autonomous sources/sinks. Packet count is capped at 32, and depth is a capped class rather than an unbounded volume claim. Reports retain original timestamps.

## Literal source evidence

candidate.py:L78: deposited=jnp.where((a==8)&jnp.any(near),jnp.minimum(bag,20-n['stores'][home]),jnp.zeros((3,),jnp.int32))

candidate.py:L84: levee=(a==9)&inside&(old==0)&~occupied&(bag[0]>=2)

candidate.py:L86: culvert=(a==11)&inside&(old==1)&(bag[0]>=1)&(bag[1]>=1)

candidate.py:L97: n['gate_open']=n['gate_open'].at[f[1],f[0]].set(jnp.where(culvert,True,n['gate_open'][f[1],f[0]]^toggle)&(new==3))

candidate.py:L110: send=(a==23)&jnp.any(peers)&(n['report_tick'][i]>n['report_tick'][peer])

candidate.py:L119: proposed=jnp.where(dam[:,None],jnp.where(sideblock[:,None],wp,side),down)

candidate.py:L127: n['depth']=jnp.zeros((20,20),jnp.int32).at[n['water_pos'][:,1],n['water_pos'][:,0]].add(n['water_active'].astype(jnp.int32));n['depth']=jnp.minimum(4,n['depth'])

candidate.py:L133: n['home_water']=jnp.max(jnp.where(field,n['depth'][None,:,:],0),axis=(1,2)).astype(jnp.int32)

candidate.py:L138: n['reward_parts']=n['reward_parts'].at[:,2].set(jnp.where(s['t']%8==7,0.05*(4-n['alerts']),0.0)).at[:,4].add(-n['priority']*loss.astype(jnp.float32))

candidate.py:L140: n['t']=s['t']+1;n['rain']=(n['t']%32)<12;n['terminated']=jnp.array(False);n['truncated']=n['t']>=self.max_steps;n['done']=n['truncated']

## Presentation and evidence boundary

Parchment ground, muted settlement flags, blue water packets and native colored levee/road/culvert tiles. Terrain and infrastructure are the actual masked cell state, not an invented hex board. All scene geometry, entity attributes and visibility refer to declared native leaves. Public scene events are restricted to actually public information. Debug/full-state view must be separate from role view.

Two literal scripts and a single actor/action paired hypothesis are supplied, with fixed integer seed and source-derived suffix lengths. All remain UNEXECUTED_AUTHOR_HYPOTHESIS; neither legality of the full trajectory nor any effect, solvability, strategic richness, learning or transfer is certified. Prospective variation could examine rain routing, material stores, settlement valuations, road durability or report-sharing conventions under a separate prospective source. No such variation or candidate has been run.

Current author checks use stdlib AST/literal shape arithmetic, byte hashes and the supplied trusted manifest/scene/whitelist validators only. No candidate/JAX/NumPy import, native method call, numerical rollout, rendering, training, network/model call or seed search occurred. Four official README excerpts supplied API context only: JaxMARL parallel dictionaries, Pgx explicit-key and terminal interfaces, Craftax spatial game framing, PureJaxRL single-file implementation philosophy. No upstream implementation, layout or asset was copied. The supplied PPO implementation was not read. Exact current input paths, bytes, hashes and excerpt extents are recorded in author_input_freeze.json. Current diversity addendum controls A/H and the scope of open worlds.
