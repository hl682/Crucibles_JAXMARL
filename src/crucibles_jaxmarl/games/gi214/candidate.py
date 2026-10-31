"""Magnet Frontier: six complementary crews in a finite persistent world."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 6
    observation_size=938
    num_actions = 28
    max_steps = 384
    obs_keys = ('t','done','terminated','truncated','terrain','agent_pos','role','facing','inventory','selected','carrying','visits','tool_pos','tool_kind','tool_holder','node_pos','node_kind','stock','magnet_pos','magnet_active','magnet_on','magnet_powered','magnet_radius','polarity','cargo_pos','cargo_active','holder','cargo_kind','jam','anchored','integrity','cooldown','last_touch','facility_pos','charge','weather','weather_offset','deliveries','builds','reward_parts')

    def reset(self,key):
        s = {
            't':jnp.array(0,jnp.int32),'done':jnp.array(False),'terminated':jnp.array(False),'truncated':jnp.array(False),
            'terrain':jnp.zeros((24,24),jnp.int32),
            'agent_pos':jnp.array([[7,10],[7,12],[9,9],[8,9],[10,12],[10,11]],jnp.int32),
            'agent_active':jnp.ones((6,),jnp.bool_),'agent_visible':jnp.ones((6,6),jnp.bool_),
            'role':jnp.array([0,1,2,3,4,5],jnp.int32),'facing':jnp.array([2,2,3,1,1,2],jnp.int32),
            'inventory':jnp.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[2,2,1,0,0,0],[1,1,0,0,1,0],[0,0,0,1,0,0],[0,2,0,1,0,0]],jnp.int32),
            'selected':jnp.zeros((6,),jnp.int32),'carrying':jnp.full((6,),-1,jnp.int32),'visits':jnp.zeros((6,8),jnp.bool_),
            'tool_pos':jnp.array([[7,10],[7,12],[9,9],[8,9],[10,12],[10,11]],jnp.int32),'tool_active':jnp.ones((6,),jnp.bool_),'tool_visible':jnp.ones((6,6),jnp.bool_),'tool_kind':jnp.array([0,1,2,3,4,5],jnp.int32),'tool_holder':jnp.array([0,1,2,3,4,5],jnp.int32),
            'node_pos':jnp.array([[8,10],[18,5],[17,18],[4,18],[8,12],[5,5],[18,14],[4,14]],jnp.int32),
            'node_active':jnp.ones((8,),jnp.bool_),'node_visible':jnp.ones((6,8),jnp.bool_),
            'node_kind':jnp.array([0,0,0,0,1,1,1,1],jnp.int32),'stock':jnp.full((8,),8,jnp.int32),
            'magnet_pos':jnp.array([[10,10],[13,10],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],jnp.int32),
            'magnet_active':jnp.array([True,True,False,False,False,False,False,False]),'magnet_visible':jnp.zeros((6,8),jnp.bool_),
            'magnet_on':jnp.zeros((8,),jnp.bool_),'magnet_powered':jnp.zeros((8,),jnp.bool_),'magnet_radius':jnp.full((8,),4,jnp.int32),'polarity':jnp.array([1,-1,1,1,1,1,1,1],jnp.int32),
            'cargo_pos':jnp.array([[9,10],[9,12],[14,10],[14,12],[18,5],[17,18],[4,18],[5,5]],jnp.int32),
            'cargo_active':jnp.ones((8,),jnp.bool_),'cargo_visible':jnp.ones((6,8),jnp.bool_),
            'holder':jnp.full((8,),-1,jnp.int32),'cargo_kind':jnp.array([0,1,0,1,0,1,0,1],jnp.int32),
            'jam':jnp.zeros((8,),jnp.bool_),'anchored':jnp.zeros((8,),jnp.bool_),'integrity':jnp.full((8,),3,jnp.int32),'cooldown':jnp.zeros((8,),jnp.int32),'last_touch':jnp.full((8,),4,jnp.int32),
            'facility_pos':jnp.array([[8,8],[5,5],[18,18]],jnp.int32),'facility_active':jnp.ones((3,),jnp.bool_),'facility_visible':jnp.ones((6,3),jnp.bool_),
            'charge':jnp.array(20,jnp.int32),'weather':jnp.array(0,jnp.int32),'weather_offset':jnp.array(0,jnp.int32),
            'deliveries':jnp.zeros((6,),jnp.int32),'builds':jnp.array(0,jnp.int32),'reward_parts':jnp.zeros((6,5),jnp.float32)
        }
        x=jnp.arange(24,dtype=jnp.int32)[None,:]
        y=jnp.arange(24,dtype=jnp.int32)[:,None]
        s['terrain']=jnp.where((x==11)&(y>=4)&(y<=19),1,jnp.where((y==10)&(x>=8)&(x<=15),3,0)).astype(jnp.int32)
        variation=jax.random.randint(key,(8,),0,3,dtype=jnp.int32)
        s['stock']=s['stock']+variation
        s['node_pos']=s['node_pos'].at[1:4,0].add(variation[1:4]-1)
        s['weather_offset']=variation[7]
        s['magnet_visible']=jnp.broadcast_to(s['magnet_active'],(6,8))
        return self.observe(s),s

    def observe(self,s):
        flat=jnp.concatenate([s[k].ravel().astype(jnp.float32) for k in self.obs_keys])
        return jnp.broadcast_to(flat,(6,self.observation_size))

    def legal_actions(self,s):
        role=s['tool_kind'];p=s['agent_pos'];d=jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        q=p[:,None,:]+d[None,:,:]
        move=jnp.all((q>=0)&(q<24),axis=2)&(s['terrain'][jnp.clip(q[:,:,1],0,23),jnp.clip(q[:,:,0],0,23)]!=1)
        forge=jnp.sum(jnp.abs(p-s['facility_pos'][0]),axis=1)<=1
        nm=jnp.any((jnp.sum(jnp.abs(p[:,None,:]-s['magnet_pos'][None,:,:]),axis=2)<=1)&s['magnet_active'][None,:],axis=1)
        anyrole=jnp.ones((6,),jnp.bool_)
        extra=jnp.stack(((role==0)|(role==1),(role==0)|(role==4),s['carrying']>=0,(role==2)&forge,(role==5)&forge,(role==3)&forge,role==2,role==2,role==5,nm,nm,(role==2)|(role==5),(role==3)&forge,s['carrying']>=0,anyrole,anyrole,(role==4)|(role==5),forge,role==5,anyrole,anyrole,anyrole,anyrole),axis=1)
        mask=jnp.concatenate((move,extra),axis=1)&~s['done']
        return mask.at[:,0].set(True)

    def advance(self,s,actions):
        n=dict(s);n['reward_parts']=jnp.zeros((6,5),jnp.float32)
        ac=jnp.clip(actions,0,27)
        ac=jnp.where((actions>=0)&(actions<28)&self.legal_actions(s)[jnp.arange(6,dtype=jnp.int32),ac],ac,0)
        d=jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        target=s['agent_pos']+d[jnp.minimum(ac,4)]*((ac>0)&(ac<5))[:,None]
        other=~jnp.eye(6,dtype=jnp.bool_)
        blocked=jnp.any(jnp.all(target[:,None,:]==s['agent_pos'][None,:,:],axis=2)&other,axis=1)|jnp.any(jnp.all(target[:,None,:]==target[None,:,:],axis=2)&other,axis=1)
        n['agent_pos']=jnp.where(blocked[:,None],s['agent_pos'],target)
        n['tool_pos']=n['agent_pos'][n['tool_holder']]
        n['facing']=jnp.where((ac>0)&(ac<5),ac,jnp.where(ac>=24,ac-23,s['facing']))
        newbuild=jnp.array(0,jnp.int32)
        for k in range(6):
            i=(s['t']+k)%6;p=n['agent_pos'][i];a=ac[i];inv=n['inventory'][i]
            near=(jnp.sum(jnp.abs(n['node_pos']-p),axis=1)<=1)&(n['stock']>0)&(n['node_kind']==n['tool_kind'][i])
            node=jnp.argmax(near).astype(jnp.int32);kind=n['node_kind'][node]
            gather=(a==5)&jnp.any(near)&(inv[kind]<6)
            n['stock']=n['stock'].at[node].add(-gather.astype(jnp.int32))
            inv=inv.at[kind].add(gather.astype(jnp.int32))
            loose=n['cargo_active']&(n['holder']==-1)&(jnp.sum(jnp.abs(n['cargo_pos']-p),axis=1)<=1)
            c=jnp.argmax(loose).astype(jnp.int32);held=n['holder']==i;h=jnp.argmax(held).astype(jnp.int32);has=jnp.any(held)
            pick=(a==6)&~has&jnp.any(loose)
            n['holder']=n['holder'].at[c].set(jnp.where(pick,i,n['holder'][c]))
            n['last_touch']=n['last_touch'].at[c].set(jnp.where(pick,i,n['last_touch'][c]))
            n['jam']=n['jam'].at[c].set(n['jam'][c]&~pick)
            n['anchored']=n['anchored'].at[c].set(n['anchored'][c]&~pick)
            drop=(a==7)&has
            n['holder']=n['holder'].at[h].set(jnp.where(drop,-1,n['holder'][h]))
            n['cargo_pos']=n['cargo_pos'].at[h].set(jnp.where(drop,p,n['cargo_pos'][h]))
            coil=(a==8)&(inv[0]>=2)&(inv[2]<6)
            brace=(a==9)&(inv[1]>=2)&(inv[3]<6)
            pack=(a==10)&(inv[0]>=1)&(inv[1]>=1)&(inv[4]<6)
            inv=inv+jnp.stack((-2*coil.astype(jnp.int32)-pack.astype(jnp.int32),-2*brace.astype(jnp.int32)-pack.astype(jnp.int32),coil.astype(jnp.int32),brace.astype(jnp.int32),pack.astype(jnp.int32),jnp.array(0,jnp.int32)))
            front=p+d[n['facing'][i]];inside=jnp.all((front>=0)&(front<24));f=jnp.clip(front,0,23);tile=n['terrain'][f[1],f[0]]
            occupied=jnp.any(jnp.all(n['agent_pos']==front,axis=1))|jnp.any(jnp.all(n['magnet_pos']==front,axis=1)&n['magnet_active'])
            free=~n['magnet_active'];m=jnp.argmax(free).astype(jnp.int32)
            build=(a==11)&inside&(tile!=1)&~occupied&jnp.any(free)&(inv[2]>0)&(inv[1]>0)
            rail=(a==12)&inside&((tile==0)|(tile==2))&(inv[0]>0)&(inv[1]>0)
            bridge=(a==13)&inside&(tile==1)&(inv[3]>0)
            n['magnet_pos']=n['magnet_pos'].at[m].set(jnp.where(build,front,n['magnet_pos'][m]))
            n['magnet_active']=n['magnet_active'].at[m].set(n['magnet_active'][m]|build)
            n['magnet_on']=n['magnet_on'].at[m].set(jnp.where(build,False,n['magnet_on'][m]))
            n['polarity']=n['polarity'].at[m].set(jnp.where(build,1,n['polarity'][m]))
            n['terrain']=n['terrain'].at[f[1],f[0]].set(jnp.where(bridge,2,jnp.where(rail,jnp.where(tile==2,4,3),tile)))
            inv=inv.at[0].add(-rail.astype(jnp.int32)).at[1].add(-rail.astype(jnp.int32)-build.astype(jnp.int32)).at[2].add(-build.astype(jnp.int32)).at[3].add(-bridge.astype(jnp.int32))
            newbuild=newbuild+(build|rail|bridge).astype(jnp.int32)
            md=jnp.sum(jnp.abs(n['magnet_pos']-p),axis=1);eligible=n['magnet_active']&(md<=1);m=jnp.argmin(jnp.where(eligible,md,99))
            toggle=(a==14)&jnp.any(eligible);reverse=(a==15)&jnp.any(eligible)
            n['magnet_on']=n['magnet_on'].at[m].set(n['magnet_on'][m]^toggle)
            n['polarity']=n['polarity'].at[m].set(jnp.where(reverse,-n['polarity'][m],n['polarity'][m]))
            remove=(a==16)&inside&jnp.any(jnp.all(n['magnet_pos']==front,axis=1)&n['magnet_active'])&(inv[2]<6)
            rm=jnp.argmax(jnp.all(n['magnet_pos']==front,axis=1)&n['magnet_active'])
            n['magnet_active']=n['magnet_active'].at[rm].set(n['magnet_active'][rm]&~remove)
            n['magnet_on']=n['magnet_on'].at[rm].set(n['magnet_on'][rm]&~remove)
            inv=inv.at[2].add(remove.astype(jnp.int32))
            charge=(a==17)&(inv[4]>0)&(n['charge']<=24)
            inv=inv.at[4].add(-charge.astype(jnp.int32));n['charge']=n['charge']+8*charge.astype(jnp.int32)
            dest=n['facility_pos'][1+n['cargo_kind'][h]]
            deliver=(a==18)&has&(n['integrity'][h]>0)&(jnp.sum(jnp.abs(p-dest))<=1)
            n['cargo_active']=n['cargo_active'].at[h].set(n['cargo_active'][h]&~deliver)
            n['holder']=n['holder'].at[h].set(jnp.where(deliver,-1,n['holder'][h]))
            n['cooldown']=n['cooldown'].at[h].set(jnp.where(deliver,16,n['cooldown'][h]))
            n['deliveries']=n['deliveries'].at[i].add(deliver.astype(jnp.int32))
            n['reward_parts']=n['reward_parts'].at[i,1].add(deliver.astype(jnp.float32)*(2.0+2.0*n['cargo_kind'][h]))
            n['selected']=n['selected'].at[i].set(jnp.where(a==19,(n['selected'][i]+1)%5,n['selected'][i]))
            peers=(jnp.sum(jnp.abs(n['agent_pos']-p),axis=1)<=1)&(jnp.arange(6,dtype=jnp.int32)!=i)
            to=jnp.argmax(peers).astype(jnp.int32);res=n['selected'][i]
            give=(a==20)&jnp.any(peers)&(inv[res]>0)&(n['inventory'][to,res]<6)
            inv=inv.at[res].add(-give.astype(jnp.int32));n['inventory']=n['inventory'].at[to,res].add(give.astype(jnp.int32))
            recover=(a==21)&jnp.any(loose)&(inv[3]>0)&(n['jam'][c]|(n['integrity'][c]<3))
            inv=inv.at[3].add(-recover.astype(jnp.int32));n['jam']=n['jam'].at[c].set(n['jam'][c]&~recover);n['integrity']=n['integrity'].at[c].set(jnp.where(recover,3,n['integrity'][c]))
            anchor=(a==23)&jnp.any(loose)&(inv[3]>0)&~n['anchored'][c]
            inv=inv.at[3].add(-anchor.astype(jnp.int32));n['anchored']=n['anchored'].at[c].set(n['anchored'][c]|anchor)
            handpower=(a==22)&(n['charge']<32)
            n['charge']=n['charge']+handpower.astype(jnp.int32)
            n['inventory']=n['inventory'].at[i].set(inv)
            cost=2*coil.astype(jnp.int32)+2*brace.astype(jnp.int32)+2*pack.astype(jnp.int32)+2*build.astype(jnp.int32)+2*rail.astype(jnp.int32)+bridge.astype(jnp.int32)+charge.astype(jnp.int32)+recover.astype(jnp.int32)+anchor.astype(jnp.int32)
            n['reward_parts']=n['reward_parts'].at[i,3].add(-0.05*cost.astype(jnp.float32))
        # A common power pack supplies all mounted fields, or none if insufficient.
        n['cargo_pos']=jnp.where((n['holder']>=0)[:,None],n['agent_pos'][jnp.clip(n['holder'],0,5)],n['cargo_pos'])
        demand=n['magnet_active']&n['magnet_on'];draw=jnp.sum(demand.astype(jnp.int32),dtype=jnp.int32)
        powered=demand&(n['charge']>=draw);used=jnp.where(n['charge']>=draw,draw,0)
        n['magnet_powered']=powered
        n['charge']=n['charge']-used
        diff=n['magnet_pos'][None,:,:]-n['cargo_pos'][:,None,:]
        reach=(jnp.sum(jnp.abs(diff),axis=2)<=n['magnet_radius'][None,:])&powered[None,:]
        force=jnp.sum(jnp.sign(diff)*n['polarity'][None,:,None]*reach[:,:,None],axis=1,dtype=jnp.int32)
        horizontal=jnp.abs(force[:,0])>=jnp.abs(force[:,1])
        delta=jnp.stack((jnp.where(horizontal,jnp.sign(force[:,0]),0),jnp.where(horizontal,0,jnp.sign(force[:,1]))),axis=1)
        cq=n['cargo_pos']+delta;clipped=jnp.clip(cq,0,23)
        tile=n['terrain'][clipped[:,1],clipped[:,0]]
        current=n['terrain'][n['cargo_pos'][:,1],n['cargo_pos'][:,0]]
        movable=n['cargo_active']&(n['holder']==-1)&~n['jam']&~n['anchored']&(n['integrity']>0)&jnp.any(delta!=0,axis=1)&((current>=2)|(tile>=2)|(tile==1))
        other=~jnp.eye(8,dtype=jnp.bool_)
        hit=jnp.any(jnp.all(cq[:,None,:]==n['cargo_pos'][None,:,:],axis=2)&n['cargo_active'][None,:]&other,axis=1)|jnp.any(jnp.all(cq[:,None,:]==cq[None,:,:],axis=2)&movable[None,:]&other,axis=1)|~jnp.all((cq>=0)&(cq<24),axis=1)
        jammed=movable&hit
        n['cargo_pos']=jnp.where((movable&~hit)[:,None],cq,n['cargo_pos'])
        n['jam']=n['jam']|jammed;n['integrity']=jnp.maximum(0,n['integrity']-jammed.astype(jnp.int32))
        n['cargo_pos']=jnp.where((n['holder']>=0)[:,None],n['agent_pos'][jnp.clip(n['holder'],0,5)],n['cargo_pos'])
        n['t']=s['t']+1;n['weather']=((n['t']+n['weather_offset'])%24>=16).astype(jnp.int32)
        n['charge']=jnp.minimum(32,n['charge']+jnp.where((n['weather']==0)&(n['t']%8==0),2,0))
        regrow=((n['node_kind']==0)&(n['t']%48==0))|((n['node_kind']==1)&(n['weather']==1)&(n['t']%8==0))
        n['stock']=jnp.minimum(12,n['stock']+regrow.astype(jnp.int32))
        n['cooldown']=jnp.maximum(0,n['cooldown']-1)
        for c in range(8):
            node=c%4;spawn=~n['cargo_active'][c]&(n['cooldown'][c]==0)&(n['stock'][node]>0)
            n['stock']=n['stock'].at[node].add(-spawn.astype(jnp.int32));n['cargo_active']=n['cargo_active'].at[c].set(n['cargo_active'][c]|spawn)
            n['cargo_pos']=n['cargo_pos'].at[c].set(jnp.where(spawn,n['node_pos'][node],n['cargo_pos'][c]));n['integrity']=n['integrity'].at[c].set(jnp.where(spawn,3,n['integrity'][c]));n['jam']=n['jam'].at[c].set(n['jam'][c]&~spawn)
        visiting=jnp.sum(jnp.abs(n['agent_pos'][:,None,:]-n['node_pos'][None,:,:]),axis=2)<=2
        n['reward_parts']=n['reward_parts'].at[:,0].set(0.02*jnp.sum((visiting&~s['visits']).astype(jnp.float32),axis=1))
        n['visits']=s['visits']|visiting;n['builds']=s['builds']+newbuild
        n['reward_parts']=n['reward_parts'].at[:,2].set(0.15*newbuild.astype(jnp.float32)).at[:,3].add(-0.01*used.astype(jnp.float32))
        damage=jnp.sum((jammed[None,:]&(n['last_touch'][None,:]==jnp.arange(6,dtype=jnp.int32)[:,None])).astype(jnp.float32),axis=1)
        n['reward_parts']=n['reward_parts'].at[:,4].set(-0.25*damage)
        matches=n['holder'][None,:]==jnp.arange(6,dtype=jnp.int32)[:,None]
        n['carrying']=jnp.where(jnp.any(matches,axis=1),jnp.argmax(matches,axis=1),-1).astype(jnp.int32)
        n['cargo_visible']=jnp.broadcast_to(n['cargo_active'],(6,8));n['magnet_visible']=jnp.broadcast_to(n['magnet_active'],(6,8))
        n['terminated']=jnp.array(False);n['truncated']=n['t']>=self.max_steps;n['done']=n['truncated']
        return n,jnp.sum(n['reward_parts'],axis=1,dtype=jnp.float32)

    def raw_step(self,key,state,actions):
        n,r=jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((6,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)
        return self.observe(n),n,r,jnp.full((6,),n['done'],jnp.bool_),{'done_all':n['done'],'terminated':n['terminated'],'truncated':n['truncated']}
