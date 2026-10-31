"""Floodplain Council: four stores, physical levees, trade and local reports."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents=4
    observation_size=3001
    num_actions=24
    max_steps=384
    public_keys=('t','done','terminated','truncated','rain','alerts','home_pos','priority')

    def reset(self,key):
        s={
            't':jnp.array(0,jnp.int32),'done':jnp.array(False),'terminated':jnp.array(False),'truncated':jnp.array(False),
            'terrain':jnp.zeros((20,20),jnp.int32),'structure':jnp.zeros((20,20),jnp.int32),'gate_open':jnp.zeros((20,20),jnp.bool_),'durability':jnp.zeros((20,20),jnp.int32),'builder':jnp.full((20,20),-1,jnp.int32),'depth':jnp.zeros((20,20),jnp.int32),'map_visible':jnp.zeros((4,20,20),jnp.bool_),
            'agent_pos':jnp.array([[8,5],[11,5],[8,12],[11,12]],jnp.int32),'agent_active':jnp.ones((4,),jnp.bool_),'agent_visible':jnp.ones((4,4),jnp.bool_),
            'facing':jnp.array([2,4,2,4],jnp.int32),'backpack':jnp.array([[2,0,0],[0,2,0],[2,0,0],[0,2,0]],jnp.int32),'selected':jnp.zeros((4,),jnp.int32),'priority':jnp.array([1.0,1.3,0.8,1.6],jnp.float32),
            'home_pos':jnp.array([[6,6],[13,6],[6,13],[13,13]],jnp.int32),'home_active':jnp.ones((4,),jnp.bool_),'home_visible':jnp.ones((4,4),jnp.bool_),
            'stores':jnp.full((4,3),8,jnp.int32),'crop':jnp.full((4,),3,jnp.int32),'home_water':jnp.zeros((4,),jnp.int32),
            'report':jnp.full((4,4),-1,jnp.int32),'report_tick':jnp.full((4,4),-1,jnp.int32),
            'water_pos':jnp.zeros((32,2),jnp.int32),'water_active':jnp.zeros((32,),jnp.bool_),'water_visible':jnp.zeros((4,32),jnp.bool_),
            'rain':jnp.array(True),'alerts':jnp.array(0,jnp.int32),'trade_receipts':jnp.zeros((4,),jnp.int32),'flood_losses':jnp.zeros((4,),jnp.int32),'reward_parts':jnp.zeros((4,6),jnp.float32)
        }
        s=self.visibility(s)
        return self.observe(s),s

    def visibility(self,s):
        n=dict(s);p=s['agent_pos'];x=jnp.arange(20,dtype=jnp.int32)[None,:];y=jnp.arange(20,dtype=jnp.int32)[:,None]
        n['map_visible']=(jnp.abs(x[None,:,:]-p[:,0,None,None])+jnp.abs(y[None,:,:]-p[:,1,None,None]))<=4
        n['agent_visible']=jnp.sum(jnp.abs(p[:,None,:]-p[None,:,:]),axis=2)<=4
        n['home_visible']=(jnp.sum(jnp.abs(p[:,None,:]-s['home_pos'][None,:,:]),axis=2)<=4)|jnp.eye(4,dtype=jnp.bool_)
        n['water_visible']=(jnp.sum(jnp.abs(p[:,None,:]-s['water_pos'][None,:,:]),axis=2)<=4)&s['water_active'][None,:]
        return n

    def record(self,data,visible):
        return jnp.concatenate((visible[:,None],jnp.where(visible[:,None],data,-1)),axis=1).ravel().astype(jnp.float32)

    def observe(self,s):
        public=jnp.concatenate([s[k].ravel().astype(jnp.float32) for k in self.public_keys])
        grid=jnp.stack((s['terrain'].ravel(),s['depth'].ravel(),s['structure'].ravel(),s['durability'].ravel(),s['builder'].ravel(),s['gate_open'].ravel()),axis=1)
        agents=jnp.concatenate((s['agent_pos'],s['backpack'],s['selected'][:,None],s['facing'][:,None]),axis=1)
        homes=jnp.concatenate((s['home_pos'],s['stores'],s['crop'][:,None],s['home_water'][:,None]),axis=1)
        rows=[]
        for i in range(4):
            own=jnp.concatenate((jnp.array([i],jnp.float32),s['backpack'][i],s['selected'][i,None],s['stores'][i],s['priority'][i,None],s['report'][i],s['report_tick'][i],s['reward_parts'][i]))
            rows=rows+[jnp.concatenate((public,own,self.record(grid,s['map_visible'][i].ravel()),self.record(agents,s['agent_visible'][i]),self.record(homes,s['home_visible'][i]),self.record(s['water_pos'],s['water_visible'][i]))).astype(jnp.float32)]
        return jnp.stack(rows)

    def legal_actions(self,s):
        p=s['agent_pos'];d=jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32);q=p[:,None,:]+d[None,:,:];c=jnp.clip(q,0,19)
        road=s['structure'][c[:,:,1],c[:,:,0]]==2
        barrier=(s['structure'][c[:,:,1],c[:,:,0]]==1)|((s['structure'][c[:,:,1],c[:,:,0]]==3)&~s['gate_open'][c[:,:,1],c[:,:,0]])
        move=jnp.all((q>=0)&(q<20),axis=2)&~barrier&((s['depth'][c[:,:,1],c[:,:,0]]<=1)|road)
        home=jnp.sum(jnp.abs(p-s['home_pos']),axis=1)<=1
        anyhome=jnp.any(jnp.sum(jnp.abs(p[:,None,:]-s['home_pos'][None,:,:]),axis=2)<=1,axis=1)
        dash=p[:,None,:]+2*d[None,1:,:];dc=jnp.clip(dash,0,19)
        dashok=jnp.all((dash>=0)&(dash<20),axis=2)&road[:,1:]&(s['structure'][dc[:,:,1],dc[:,:,0]]==2)
        yes=jnp.ones((4,),jnp.bool_)
        extra=jnp.stack((home,home,home,anyhome,yes,yes,yes,yes,yes,yes,anyhome,dashok[:,0],dashok[:,1],dashok[:,2],dashok[:,3],yes,yes,yes,yes),axis=1)
        return (jnp.concatenate((move,extra),axis=1)&~s['done']).at[:,0].set(True)

    def advance(self,s,actions):
        n=dict(s);n['reward_parts']=jnp.zeros((4,6),jnp.float32)
        ac=jnp.clip(actions,0,23);ac=jnp.where((actions>=0)&(actions<24)&self.legal_actions(s)[jnp.arange(4,dtype=jnp.int32),ac],ac,0)
        d=jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        direction=jnp.where((ac>=16)&(ac<=19),ac-15,jnp.where((ac>0)&(ac<5),ac,0));distance=jnp.where((ac>=16)&(ac<=19),2,1).astype(jnp.int32)
        target=s['agent_pos']+d[direction]*distance[:,None];other=~jnp.eye(4,dtype=jnp.bool_)
        block=jnp.any(jnp.all(target[:,None,:]==s['agent_pos'][None,:,:],axis=2)&other,axis=1)|jnp.any(jnp.all(target[:,None,:]==target[None,:,:],axis=2)&other,axis=1)
        middle=s['agent_pos']+d[direction]
        block=block|((distance==2)&jnp.any(jnp.all(middle[:,None,:]==s['agent_pos'][None,:,:],axis=2)&other,axis=1))
        n['agent_pos']=jnp.where(block[:,None],s['agent_pos'],target);n['facing']=jnp.where(direction>0,direction,s['facing'])
        for k in range(4):
            i=(s['t']+k)%4;p=n['agent_pos'][i];a=ac[i];bag=n['backpack'][i]
            near=jnp.sum(jnp.abs(n['home_pos']-p),axis=1)<=1;home=jnp.argmax(near).astype(jnp.int32)
            kind=jnp.clip(a-5,0,2);take=(a>=5)&(a<=7)&near[i]&(n['stores'][i,kind]>0)&(jnp.sum(bag,dtype=jnp.int32)<4)
            bag=bag.at[kind].add(take.astype(jnp.int32));n['stores']=n['stores'].at[i,kind].add(-take.astype(jnp.int32))
            deposited=jnp.where((a==8)&jnp.any(near),jnp.minimum(bag,20-n['stores'][home]),jnp.zeros((3,),jnp.int32))
            bag=bag-deposited;n['stores']=n['stores'].at[home].add(deposited)
            traded=jnp.where(home!=i,deposited[2],0);n['trade_receipts']=n['trade_receipts'].at[i].add(traded)
            n['reward_parts']=n['reward_parts'].at[i,0].add(2.0*traded.astype(jnp.float32)).at[home,1].add(0.2*traded.astype(jnp.float32)*n['priority'][home])
            front=p+d[n['facing'][i]];inside=jnp.all((front>=0)&(front<20));f=jnp.clip(front,0,19);old=n['structure'][f[1],f[0]]
            occupied=jnp.any(jnp.all(n['agent_pos']==front,axis=1))|jnp.any(jnp.all(n['home_pos']==front,axis=1))
            levee=(a==9)&inside&(old==0)&~occupied&(bag[0]>=2)
            road=(a==10)&inside&(old==0)&~occupied&(bag[1]>=2)
            culvert=(a==11)&inside&(old==1)&(bag[0]>=1)&(bag[1]>=1)
            toggle=(a==12)&inside&(old==3)
            reclaim=jnp.where(old==2,1,0)
            remove=(a==13)&inside&(old>0)&(jnp.sum(bag,dtype=jnp.int32)<4)
            repair=(a==14)&inside&(old==2)&(n['durability'][f[1],f[0]]<3)&(bag[1]>0)
            bag=bag.at[0].add(-2*levee.astype(jnp.int32)-culvert.astype(jnp.int32)).at[1].add(-2*road.astype(jnp.int32)-culvert.astype(jnp.int32)-repair.astype(jnp.int32)).at[reclaim].add(remove.astype(jnp.int32))
            built=levee|road|culvert
            new=jnp.where(remove,0,jnp.where(levee,1,jnp.where(road,2,jnp.where(culvert,3,old))))
            n['structure']=n['structure'].at[f[1],f[0]].set(new)
            n['durability']=n['durability'].at[f[1],f[0]].set(jnp.where(remove,0,jnp.where(built|repair,3,n['durability'][f[1],f[0]])))
            n['builder']=n['builder'].at[f[1],f[0]].set(jnp.where(remove,-1,jnp.where(built,i,n['builder'][f[1],f[0]])))
            n['gate_open']=n['gate_open'].at[f[1],f[0]].set(jnp.where(culvert,True,n['gate_open'][f[1],f[0]]^toggle)&(new==3))
            expense=2*(levee|road|culvert).astype(jnp.int32)+repair.astype(jnp.int32)
            n['reward_parts']=n['reward_parts'].at[i,3].add(-0.2*expense.astype(jnp.float32))
            harvest=(a==15)&jnp.any(near)&(n['crop'][home]>0)&(jnp.sum(bag,dtype=jnp.int32)<4)
            bag=bag.at[2].add(harvest.astype(jnp.int32));n['crop']=n['crop'].at[home].add(-harvest.astype(jnp.int32))
            n['reward_parts']=n['reward_parts'].at[i,1].add(0.4*n['priority'][i]*harvest.astype(jnp.float32)).at[home,4].add(-0.15*(harvest&(home!=i)).astype(jnp.float32))
            n['selected']=n['selected'].at[i].set(jnp.where(a==21,(n['selected'][i]+1)%3,n['selected'][i]))
            peers=(jnp.sum(jnp.abs(n['agent_pos']-p),axis=1)<=1)&(jnp.arange(4,dtype=jnp.int32)!=i);peer=jnp.argmax(peers).astype(jnp.int32);res=n['selected'][i]
            share=(a==20)&jnp.any(peers)&(bag[res]>0)&(jnp.sum(n['backpack'][peer],dtype=jnp.int32)<4)
            bag=bag.at[res].add(-share.astype(jnp.int32));n['backpack']=n['backpack'].at[peer,res].add(share.astype(jnp.int32));n['backpack']=n['backpack'].at[i].set(bag)
            visible=(jnp.sum(jnp.abs(n['home_pos']-p),axis=1)<=4)|(jnp.arange(4,dtype=jnp.int32)==i)
            inspect=(a==22)&visible
            n['report']=n['report'].at[i].set(jnp.where(inspect,s['home_water'],n['report'][i]));n['report_tick']=n['report_tick'].at[i].set(jnp.where(inspect,s['t'],n['report_tick'][i]))
            send=(a==23)&jnp.any(peers)&(n['report_tick'][i]>n['report_tick'][peer])
            n['report']=n['report'].at[peer].set(jnp.where(send,n['report'][i],n['report'][peer]));n['report_tick']=n['report_tick'].at[peer].set(jnp.where(send,n['report_tick'][i],n['report_tick'][peer]))
            owner=s['builder'][p[1],p[0]];used=(direction[i]>0)&~block[i]&(s['structure'][p[1],p[0]]==2)&(owner>=0)&(owner!=i)
            n['reward_parts']=n['reward_parts'].at[jnp.clip(owner,0,3),5].add(0.1*used.astype(jnp.float32))
        # Every second tick packets travel down; blocked gates divert outward.
        wp=s['water_pos'];down=wp+jnp.array([0,1],jnp.int32);dc=jnp.clip(down,0,19)
        dam=(n['structure'][dc[:,1],dc[:,0]]==1)|((n['structure'][dc[:,1],dc[:,0]]==3)&~n['gate_open'][dc[:,1],dc[:,0]])
        side=wp+jnp.stack((jnp.where(wp[:,0]<10,-1,1).astype(jnp.int32),jnp.zeros((32,),jnp.int32)),axis=1)
        sc=jnp.clip(side,0,19);sideblock=(n['structure'][sc[:,1],sc[:,0]]==1)|((n['structure'][sc[:,1],sc[:,0]]==3)&~n['gate_open'][sc[:,1],sc[:,0]])
        proposed=jnp.where(dam[:,None],jnp.where(sideblock[:,None],wp,side),down)
        moving=(s['t']%2==0)&s['water_active'];alive=jnp.all((proposed>=0)&(proposed<20),axis=1)
        n['water_active']=s['water_active']&(~moving|alive);n['water_pos']=jnp.where((moving&alive)[:,None],proposed,wp)
        for j in range(2):
            free=~n['water_active'];slot=jnp.argmax(free).astype(jnp.int32)
            emit=jnp.any(free)&(s['rain']|((s['t']%8==0)&(j==0)))
            start=jnp.stack((7+(s['t']//4+j)%6,jnp.array(0,jnp.int32)))
            n['water_pos']=n['water_pos'].at[slot].set(jnp.where(emit,start,n['water_pos'][slot]));n['water_active']=n['water_active'].at[slot].set(n['water_active'][slot]|emit)
        n['depth']=jnp.zeros((20,20),jnp.int32).at[n['water_pos'][:,1],n['water_pos'][:,0]].add(n['water_active'].astype(jnp.int32));n['depth']=jnp.minimum(4,n['depth'])
        wetroad=(n['structure']==2)&(n['depth']>=3)&(s['t']%8==7)
        n['durability']=jnp.maximum(0,n['durability']-wetroad.astype(jnp.int32));broken=(n['structure']==2)&(n['durability']==0)
        n['structure']=jnp.where(broken,0,n['structure']);n['builder']=jnp.where(broken,-1,n['builder'])
        xs=jnp.arange(20,dtype=jnp.int32)[None,:];ys=jnp.arange(20,dtype=jnp.int32)[:,None]
        field=(jnp.abs(xs[None,:,:]-n['home_pos'][:,0,None,None])+jnp.abs(ys[None,:,:]-n['home_pos'][:,1,None,None]))<=2
        n['home_water']=jnp.max(jnp.where(field,n['depth'][None,:,:],0),axis=(1,2)).astype(jnp.int32)
        flood=(n['home_water']>=2)&(s['t']%4==3);loss=jnp.minimum(n['stores'][:,2],flood.astype(jnp.int32))
        n['stores']=n['stores'].at[:,2].add(-loss);n['flood_losses']=s['flood_losses']+loss
        n['crop']=jnp.clip(n['crop']+((n['home_water']<=1)&(s['t']%8==7)).astype(jnp.int32)-flood.astype(jnp.int32),0,6)
        n['alerts']=jnp.sum((n['home_water']>=2).astype(jnp.int32),dtype=jnp.int32)
        n['reward_parts']=n['reward_parts'].at[:,2].set(jnp.where(s['t']%8==7,0.05*(4-n['alerts']),0.0)).at[:,4].add(-n['priority']*loss.astype(jnp.float32))
        n['terrain']=jnp.where(n['structure']==1,3,jnp.where(n['structure']==2,4,jnp.where(n['structure']==3,jnp.where(n['gate_open'],6,5),jnp.minimum(n['depth'],2)))).astype(jnp.int32)
        n['t']=s['t']+1;n['rain']=(n['t']%32)<12;n['terminated']=jnp.array(False);n['truncated']=n['t']>=self.max_steps;n['done']=n['truncated']
        n=self.visibility(n)
        return n,jnp.sum(n['reward_parts'],axis=1,dtype=jnp.float32)

    def raw_step(self,key,state,actions):
        n,r=jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((4,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)
        return self.observe(n),n,r,jnp.full((4,),n['done'],jnp.bool_),{'done_all':n['done'],'terminated':n['terminated'],'truncated':n['truncated']}
