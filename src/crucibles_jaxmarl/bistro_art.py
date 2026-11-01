# Adapted from the existing original scene renderer.
# Source SHA256: c6722e2a6a60c668717585638e8212eb86d59fc951a3b43dad3ae426b68b6d0d
# Changes: remove offline-only execution guard, use bundled Pillow font,
# translate in-world kitchen labels. Original geometry and artwork retained.
# Only GI215 is called by the adapter; instructional panels are cropped off.
"""Four original 2D scene families. Only filtered actual saved-view data enters.
No transition, inferred events, private state lookup or persistent seed identity.
"""
from pathlib import Path
from functools import lru_cache
import colorsys,math

from PIL import Image,ImageDraw,ImageFont
ACTION={215: [str(i) for i in range(16)]}
W,H=1120,760
SCENE=(18,132,748,718)
INFO={214:('磁力边境','搭桥铺轨、共享电力，把货物送往对应目的地','方块采矿 · 六位专业工种协作'),215:('洗水小馆','分工清洗备餐，管理排水与过滤，及时上菜','双人厨房 · 清水与脏水共用设施'),216:('洪原议会','筑堤修路、调节涵洞，交换物资与带日期水情','领土策略 · 局部视野与公共洪水警报'),217:('种风果园','采种晾晒、播种浇水，管理树荫与共享蓄水','田园生产 · 八位园丁与有限种子槽')}
PALETTES={214:('#182733','#bca178','#8b795a'),215:('#302b35','#e9d6b9','#c1aa8f'),216:('#172f3b','#c7b991','#90846e'),217:('#22352f','#b4c58a','#829c67')}
@lru_cache(maxsize=40)
def font(size):return ImageFont.load_default(size=size)
def color(i):
 rgb=colorsys.hsv_to_rgb((i*.61803398875+.08)%1,.49,.94);return '#%02x%02x%02x'%tuple(round(x*255)for x in rgb)
def short(x):
 if type(x)is float:return f'{x:.4f}'
 if isinstance(x,list):return '/'.join(short(y)for y in x)
 return str(x)
def geometry(v):
 h,w=v['map_geometry']['shape'];cell=min(696/w,548/h);return cell,35+(696-w*cell)/2,151+(548-h*cell)/2
def map_values(v):
 if 'public_map'in v:return v['public_map']
 if 'role_map'in v:return v['role_map']['values']
 return v['audit_map']['values']
def xy(v,e):
 cell,ox,oy=geometry(v);origin=v['map_geometry']['cell_origin_xy'];size=v['map_geometry']['cell_size_xy'];return ox+((e['xy'][0]-origin[0])/size[0]+.5)*cell,oy+((e['xy'][1]-origin[1])/size[1]+.5)*cell
@lru_cache(maxsize=64)
def background(n,grid_json):
 import json
 grid=json.loads(grid_json);h,w=len(grid),len(grid[0]);cell=min(696/w,548/h);ox=35+(696-w*cell)/2;oy=151+(548-h*cell)/2
 im=Image.new('RGB',(W,H),PALETTES[n][0]);d=ImageDraw.Draw(im);d.rounded_rectangle(SCENE,12,fill='#263b42',outline='#657c7d',width=2)
 for y,row in enumerate(grid):
  for x,t in enumerate(row):
   px,py=ox+x*cell,oy+y*cell
   if t is None:
    d.rectangle((px,py,px+cell,py+cell),fill='#172b37',outline='#1e3541');continue
   colors={214:['#baa382','#34485c','#b78e62','#8d8c7e','#ae8d65'],215:['#eddcba','#a8bcb8'],216:['#d1c59f','#94c4cc','#548da9','#99836a','#b99f68','#797c76','#bdad91'],217:['#b6ca8c','#79a9bc']}[n]
   fill=colors[int(t)];d.rectangle((px,py,px+cell,py+cell),fill=fill,outline=PALETTES[n][2])
   if n==214:
    if t==1:
     d.polygon([(px+2,py+cell*.3),(px+cell*.6,py+cell*.55),(px+cell*.25,py+cell-2)],fill='#263c50')
    elif t in[2,4]:
     for k in range(4):d.line((px+2,py+cell*(k+.5)/4,px+cell-2,py+cell*(k+.5)/4),fill='#694f3d',width=2)
    else:
     d.rectangle((px+cell*.16,py+cell*.27,px+cell*.28,py+cell*.35),fill='#d6bf91');d.rectangle((px+cell*.64,py+cell*.68,px+cell*.8,py+cell*.75),fill='#927b60')
    if t in[3,4]:
     for k in [.27,.73]:d.line((px+cell*k,py+2,px+cell*k,py+cell-2),fill='#3e4950',width=2)
     for k in [.2,.5,.8]:d.line((px+3,py+cell*k,px+cell-3,py+cell*k),fill='#c9b088',width=2)
   elif n==215:
    d.line((px,py+cell*.5,px+cell,py+cell*.5),fill='#d0bd9a');d.line((px+cell*.5,py,px+cell*.5,py+cell),fill='#d0bd9a')
    if t:d.rounded_rectangle((px+4,py+6,px+cell-4,py+cell-4),8,fill='#b6cbc8',outline='#6e9096',width=3)
   elif n==216:
    if t in[1,2]:d.arc((px+cell*.12,py+cell*.2,px+cell*.88,py+cell*.65),0,155,fill='#c5e1da',width=2)
    else:d.line((px+cell*.2,py+cell*.8,px+cell*.3,py+cell*.58),fill='#a4ad76',width=1)
   else:
    if t:d.arc((px+3,py+4,px+cell-3,py+cell*.65),0,150,fill='#b1d5cd',width=2)
    else:
     d.line((px+cell*.2,py+cell*.75,px+cell*.3,py+cell*.52),fill='#83a76b');d.rectangle((px+cell*.71,py+cell*.28,px+cell*.79,py+cell*.36),fill='#d0d9a1')
 return im

def render(m,scene,v,track,index,total):
 import json
 n=int(m['candidate_id'][-3:]);im=background(n,json.dumps(map_values(v),separators=(',',':'))).copy();d=ImageDraw.Draw(im);cell,ox,oy=geometry(v);p=v['public_scene_fields'];audit=v['view']=='PRIVILEGED_ACTUAL_STATE';role=-1 if audit else int(v['view'].split('_')[-1]);g=INFO[n]
 def text(x,y,s,size=13,c='#efe7d0'):d.text((x,y),str(s),font=font(size),fill=c)
 def line(points,c,width=2):d.line(points,fill=c,width=width)
 def box(rect,c,outline='#536d72',r=4,width=1):d.rounded_rectangle(rect,r,fill=c,outline=outline,width=width)
 def tag(x,y,s,c='#edf1dc',size=10):
  tw=d.textlength(str(s),font=font(size))+8;box((x-tw/2,y,x+tw/2,y+size+7),'#253b41',r=3);text(x-tw/2+4,y+1,s,size,c)
 def bar(x,y,a,maximum,w=24,c='#87b7d0'):
  box((x-w/2,y,x+w/2,y+5),'#554e3d',r=2);d.rectangle((x-w/2+1,y+1,x-w/2+1+(w-2)*max(0,min(float(a)/maximum,1)),y+4),fill=c)
 def flag(x,y,i):
  line((x,y+8,x,y-20),'#e0ca9b',2);d.polygon([(x,y-20),(x+15,y-15),(x,y-10)],fill=color(i))
 def crate(x,y,r=10,c='#c5a36e'):
  box((x-r,y-r,x+r,y+r),c,'#6b543b',2,2);line((x-r+3,y-r+3,x+r-3,y+r-3),'#e8cf92');line((x-r+3,y+r-3,x+r-3,y-r+3),'#7c6348')
 def house(x,y,i,label):
  r=max(9,cell*.39);box((x-r,y-r*.4,x+r,y+r), '#ddcba2','#766a54',3,2);d.polygon([(x-r-3,y-r*.4),(x,y-r-5),(x+r+3,y-r*.4)],fill=color(i),outline='#6f6554');box((x-3,y+2,x+4,y+r),'#6b6250',r=1);flag(x+r,y,i);tag(x,y+r+4,label,size=9)
 text(23,12,g[0],27);text(24,51,g[1],14);text(24,79,g[2],12,'#a7c3be')
 box((764,12,1100,105),'#30464d',r=11);text(779,24,'全场审阅（特权）'if audit else '角色 '+str(role)+' · '+('全员广播'if n in[214,215]else'原生局部视野'),17,'#ecc292'if audit else'#a5d9ca');text(779,53,'当前原生状态 · 原服务执行动作',12);text(779,78,'时刻 '+str(v['clock'])+'  /  画面 '+str(index+1)+'/'+str(total),12)
 if n==216:
  fields=v['visible_grid_fields'];grid=map_values(v)
  for y,row in enumerate(grid):
   for x,t in enumerate(row):
    if t is None:continue
    px,py=ox+(x+.5)*cell,oy+(y+.5)*cell;depth=fields['depth'][y][x];st=fields['structure'][y][x];gate=fields['gate_open'][y][x]
    if depth:
     d.rectangle((px-cell*.45,py-cell*.45,px+cell*.45,py+cell*.45),fill=['#8ec2ce','#619fb9','#4e859e'][min(depth,3)-1]);line((px-cell*.3,py,px+cell*.25,py+cell*.1),'#c8e5df',2)
    if st==1:box((px-cell*.4,py-cell*.15,px+cell*.4,py+cell*.15),'#a18962','#60584c',2,2)
    elif st==2:
     box((px-cell*.42,py-cell*.32,px+cell*.42,py+cell*.32),'#d1ac69','#786d51',2,2)
     for dy in[-.2,0,.2]:line((px-cell*.33,py+cell*dy,px+cell*.33,py+cell*dy),'#88704a',1)
    elif st==3:
     box((px-cell*.4,py-cell*.3,px+cell*.4,py+cell*.3),'#9b9a85','#5d665f',2,2)
     line((px-cell*.3,py,px+cell*.3,py),'#7fc5ce'if gate else'#5e5c50',3)
  # Public home coordinates are public pins only; remote contents remain hidden.
  visible_homes={e['index']for e in v['entities']if e['group']=='settlements'}
  for i,position in enumerate(p['home_pos']):
   if i not in visible_homes:
    x,y=xy(v,{'xy':position});d.ellipse((x-3,y-3,x+3,y+3),fill=color(i));tag(x,y+5,'城址'+str(i),size=8)
 # Layer only already-visible entities. Co-located characters receive a display
 # offset, not a changed native coordinate; no connection to hidden geometry.
 layer={'sites':0,'stations':0,'facilities':0,'drying_slots':0,'settlements':1,'deposits':2,'plots':2,'windbreaks':3,'magnets':3,'drainage':4,'freight':5,'kitchen_items':5,'baskets':5,'kits':6,'utensils':6,'customers':7,'floodwater':7,'drifting_seeds':8}
 for e in sorted(v['entities'],key=lambda e:(layer.get(e['group'],20),e['index'])):
  if e['kind']=='agent':continue
  x,y=xy(v,e);a=e['attributes'];i=e['index'];k=e['group'];r=max(7,min(18,cell*.36))
  if n==214:
   if k=='sites':
    if i==0:
     box((x-r,y-r*.4,x+r,y+r*.6),'#b58e5c','#685137',3,2);line((x-r*.6,y-r*.6,x+r*.6,y+r*.3),'#e0c482',3);tag(x,y+r+4,'工作台',size=9)
    else:flag(x,y,i);box((x-r,y-r*.3,x+r,y+r*.6),'#96734c','#d2b574',3,2);tag(x,y+r+4,'西货站'if i==1 else'东南货站',size=9)
   elif k=='deposits':
    if a['node_kind']==0:
     d.polygon([(x-r,y+r*.5),(x-r*.4,y-r),(x+r*.4,y-r*.8),(x+r,y+r*.6)],fill='#687b83',outline='#cbd7c7');d.polygon([(x-3,y),(x+1,y-6),(x+5,y)],fill='#d8b66b')
    else:
     line((x,y+r,x,y-r),'#7e583e',5);d.polygon([(x-r,y),(x,y-r*1.6),(x+r,y)],fill='#52764d')
    bar(x,y+r+2,a['stock'],12,c='#d6ba72')
   elif k=='magnets':
    powered=a['magnet_powered'];c='#efa256'if a['polarity']==1 else'#6dc3d2'
    box((x-r*.7,y-r,x+r*.7,y+r*.8),'#495c67','#c1c4ac',4,2);line((x-r*.45,y-r*.5,x-r*.45,y+r*.3,x+r*.45,y+r*.3,x+r*.45,y-r*.5),c,5);d.ellipse((x-3,y-3,x+3,y+3),fill='#f3d47c'if powered else'#293f48');tag(x,y+r+3,('通电'if powered else'断电')+('吸'if a['polarity']==1 else'斥'),size=8)
   elif k=='freight':
    crate(x,y,r,c='#d8ad62'if a['cargo_kind']==0 else'#7fb2bd');bar(x,y+r+2,a['integrity'],3,c='#ca8563');
    if a['anchored']:line((x-r,y-r,x+r,y+r),'#383f42',3)
    if a['jam']:tag(x,y-r-16,'卡住',c='#f0b49a',size=8)
   elif k=='kits':
    line((x-r*.4,y+r*.4,x+r*.4,y-r*.4),'#dbca9b',2)
    if a['tool_kind']==0:line((x-r*.6,y-r*.3,x+r*.5,y-r*.6),'#7ca7b9',3)
    elif a['tool_kind']==1:line((x-r*.6,y-r*.1,x+r*.7,y-r*.4),'#dce1cc',3)
  elif n==215:
   if k=='stations':
    names=['Pan','Sink','Pump','Serve','Valve','Bin','Basin'];c=['#546e77','#99bdc5','#e0ba75','#b49b68','#6c9c94','#6d716a','#8f9ca2'][i];box((x-r*1.2,y-r,x+r*1.2,y+r),c,'#485c60',5,3)
    if i==0:
     d.ellipse((x-r*.7,y-r*.7,x+r*.7,y+r*.6),fill='#635d54',outline='#ded0af',width=3);d.ellipse((x-r*.48,y-r*.45,x+r*.48,y+r*.35),fill='#8c795e'if p['pan_dirty']else'#cdb56c')
    elif i==1:d.ellipse((x-r*.7,y-r*.6,x+r*.7,y+r*.6),fill='#557a8e',outline='#e0ded2',width=3)
    elif i==2:line((x,y+r*.6,x,y-r*.6,x+r*.6,y-r*.6),'#e9d9ae',4)
    elif i==4:line((x-r*.6,y,x+r*.6,y),'#dbb779',5)
    tag(x,y+r+4,names[i],size=10)
   elif k=='drainage':
    box((x-r*.7,y-r*.3,x+r*.7,y+r*.3),'#7b8a88','#4d656b',2)
    if a['flow']:d.ellipse((x-6,y-6,x+6,y+6),fill='#a17c56',outline='#d5b48e',width=2)
   elif k=='kitchen_items':
    if a['kind']==0:
     d.ellipse((x-r,y-r*.6,x+r,y+r*.6),fill='#e5daca'if a['clean']else'#978a71',outline='#716b5a',width=2)
     if a['filled']:d.ellipse((x-r*.6,y-r*.35,x+r*.6,y+r*.25),fill='#d8a55c')
    elif a['kind']==1:
     d.ellipse((x-r*.6,y-r*.7,x+r*.6,y+r*.7),fill='#90af65'if not a['spoiled']else'#81715b',outline='#55714f',width=2);line((x,y-r*.7,x+4,y-r),'#658d51',2)
    else:box((x-r*.6,y-r*.8,x+r*.6,y+r*.8),'#b4d3c5','#658c88',3,2)
   elif k=='customers':
    c='#d8ad80'if a['order_status']==0 else('#8dbda7'if a['order_status']==1 else'#8a8182');box((x-r*.7,y-r*.3,x+r*.7,y+r*.8),c,r=4);d.ellipse((x-7,y-17,x+7,y-3),fill='#ead2a6');bar(x,y+r+4,a['patience'],175,w=40,c='#cfab6d');tag(x,y-r-35,'Waiting'if a['order_status']==0 else('Served'if a['order_status']==1 else'Left'),size=9)
   elif k=='utensils':line((x-8,y+9,x+8,y-8),'#e7dbc1',3)
  elif n==216:
   if k=='settlements':house(x,y,i,'村'+str(i));bar(x,y+cell*.8,a['crop'],6,c='#d5b76b')
   elif k=='floodwater':d.polygon([(x,y-r*.8),(x+r*.45,y+r*.2),(x,y+r*.7),(x-r*.45,y+r*.2)],fill='#b4dbe0',outline='#4e7f9a')
  else:
   if k=='facilities':
    if i==0:d.ellipse((x-r,y-r*.7,x+r,y+r*.7),fill='#679daa',outline='#d7cf9e',width=3);tag(x,y+r+3,'共享井',size=9)
    else:house(x,y,2,'晾晒棚 Lv'+str(p['roof']))
   elif k=='drying_slots':box((x-r,y-r*.65,x+r,y+r*.65),'#c8ad79','#8e7951',2,2)
   elif k=='plots':
    d.rectangle((x-r,y-r*.6,x+r,y+r*.6),fill='#725f43'if not a['shade']else'#52694a',outline=color(a['bed_owner']))
    for dy in[-.35,0,.35]:line((x-r*.8,y+dy*r,x+r*.8,y+dy*r),'#9b8054',1)
    if a['trait']>=0:
     h=3+a['growth']*2;line((x,y+3,x,y-h),'#547b4b',2);d.ellipse((x-r*.6,y-h-5,x+r*.6,y-h+5),fill=['#a5bc64','#6c9a61','#b59abc'][a['trait']],outline='#507449')
     if a['growth']>=6:
      for dx in[-4,4]:d.ellipse((x+dx-2,y-h-2,x+dx+2,y-h+2),fill='#edbc6e')
    bar(x,y+r*.7,a['moisture'],6,c='#78b2ba')
   elif k=='baskets':
    crate(x,y,r*.8,c='#c3a16b');wet=sum(a['wet']);dry=sum(a['dry'])
    if wet:d.ellipse((x-7,y-5,x-1,y+1),fill='#6eb5c0')
    if dry:d.ellipse((x+1,y-5,x+7,y+1),fill='#eed086')
    tag(x,y+r+1,str(wet)+'湿/'+str(dry)+'干',size=8)
   elif k=='windbreaks':
    if a['deployed']:
     q=(x-r,y,x+r,y)if a['break_axis']==1 else(x,y-r,x,y+r);line(q,'#54744d',7);line(q,'#9eb580',2)
    else:crate(x,y,r*.5,c='#8e9e65')
   elif k=='drifting_seeds':
    c=['#dcc372','#8bc38a','#c5a5d1'][a['seed_trait']];d.ellipse((x-3,y-2,x+3,y+2),fill=c,outline='#6d7358');line((x,y-3,x+4,y-6),'#cbd5b0',1)
 # Characters come last with tools and team colors. Palette extends to any A.
 crews=[e for e in v['entities']if e['kind']=='agent'];at={}
 for e in crews:at.setdefault(tuple(e['xy']),[]).append(e)
 for e in crews:
  x,y=xy(v,e);i=e['index'];a=e['attributes'];same=at[tuple(e['xy'])];j=same.index(e);offset=(j-(len(same)-1)/2)*min(cell*.2,7);x+=offset;r=max(6,min(18,cell*.33));c=color(i)
  d.ellipse((x-r*.75,y+r*.65,x+r*.75,y+r),fill='#4b544e');box((x-r*.65,y-r*.1,x+r*.65,y+r*.7),c,'#34434b',3,1);d.ellipse((x-r*.48,y-r*.85,x+r*.48,y+r*.1),fill='#e3bd8e',outline='#6b6552')
  if n==215:
   box((x-r*.65,y-r*1.1,x+r*.65,y-r*.62),'#f2eddb','#aab7b1',4);d.ellipse((x-r*.6,y-r*1.5,x+r*.6,y-r*.8),fill='#f2eddb');box((x-r*.4,y+r*.1,x+r*.4,y+r*.65),'#f0e5ca',r=1)
  elif n==214:box((x-r*.65,y-r*.95,x+r*.65,y-r*.56),c,'#5a635b',3);d.ellipse((x-2,y-r*.78,x+2,y-r*.5),fill='#fff0b1')
  elif n==217:d.ellipse((x-r*.8,y-r*.85,x+r*.8,y-r*.5),fill='#d4ba7d',outline='#92754f')
  else:flag(x+r*.6,y-r*.4,i)
  if i==role:d.ellipse((x-r-3,y-r*1.1-3,x+r+3,y+r+3),outline='#fff1b3',width=2)
  tag(x,y+r+4,str(i),color(i),size=9)
 # Right-side HUD: concise visible gameplay. All raw technical details stay in
 # separate view JSON/audit, not unfiltered state dumps on the main scene.
 box((764,119,1100,718),'#263d45',r=10);text(780,133,'当前可见状态',17,'#f0cc8f');yy=168
 rows=[]
 if n==214:rows=[('共享电量',str(p['charge'])+'/32'),('天气','降雨'if p['weather']else'晴朗'),('各工班交付',short(p['deliveries'])),('已建造',p['builds'])]
 elif n==215:rows=[('清水 / 脏水',str(p['tank'])+' / '+str(p['sink_dirt'])),('排水 / 分流',str(bool(p['draining']))+' / '+str(bool(p['divert']))),('蓄污盆 / 容量',str(p['basin'])+' / 3'),('过滤剩余',p['filter_charge']),('煮锅', '污染'if p['pan_dirty']else'清洁'),('上菜 / 废弃',str(p['served'])+' / '+str(p['wasted'])),('新污染累计',p['contaminations'])]
 elif n==216:rows=[('上游天气','降雨'if p['rain']else'无雨'),('洪水警报（公共总数）',p['alerts'])]
 else:rows=[('风向',short(p['wind'])),('天气 / 光照',('降雨'if p['rain']else'晴')+' / '+('白昼'if p['daylight']else'夜晚')),('共享井水',str(p['reservoir'])+'/20'),('晾晒棚等级',p['roof'])]
 for label,value in rows:text(780,yy,label,12,'#9fbbb9');text(780,yy+20,short(value),16);yy+=49
 if role!=-1:
  if 'own_reward'in v:text(780,yy,'此步本人奖励 '+short(v['own_reward']),14,'#e9c48b');yy+=29
  if 'own_action'in v:
   act=v['own_action'];label=ACTION[n][act];text(780,yy,'动作 '+str(act)+' · '+label[:17],12,'#bdd8d1');yy+=23
  own=next((e for e in crews if e['index']==role),None)
  if own:
   a=own['attributes']
   for key,label in ({214:[('inventory','矿/木/线圈/支架/电包/保留'),('carrying','所持货物')],215:[('carrying','所持物品')],216:[('backpack','石/木/粮 背包')],217:[('wood','木材'),('water','携水'),('selected','选中种型')]}[n]):
    if yy>640:break
    text(780,yy,label+' '+short(a[key]),12);yy+=24
  if n==216:
   o=v['own_scene_fields'];text(780,yy,'本人仓储 '+short(o['stores']),12);yy+=26;text(780,yy,'水情记忆（数值@记录时刻）',11,'#d2b986');yy+=22
   for i,(value,tick)in enumerate(zip(o['report'],o['report_tick'])):text(780,yy,'村'+str(i)+'  '+str(value)+' @ '+str(tick),11);yy+=20
  elif n==217:text(780,yy,'本人首获标记 '+short(v['own_scene_fields']['achieved']),12)
 else:text(780,yy,'包含真实全状态，仅供独立审阅。',12,'#e5bb99')
 text(26,731,('全状态审阅'if audit else'角色实际信息条件')+' · 原坐标不变；同格角色仅显示偏移 · 原生投影；此绘图不推进世界',10,'#a2b7b4')
 return im
