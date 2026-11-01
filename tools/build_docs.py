"""Static bilingual catalogs and manuals; runs with Python standard library only."""
from pathlib import Path
import html,json,re
from manual_content import ROOT as R,SRC as S,THEMES,profiles,read_json
from manual_pages import manual,choose,GH
from gallery_media import original_previews,primary,caption,recording_metadata
from scene_viewports import profiles as viewport_profiles,image_markup
D=R/'docs'
def esc(x):return html.escape(str(x))
def inline(s):
    s=esc(s);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
def markdown(s):
    out=[];para=[];code=None;listing=False
    def flush():
        if para:out.append('<p>'+inline(' '.join(para))+'</p>');para.clear()
    for line in s.splitlines():
        if line.startswith('```'):
            flush()
            if code is None:code=[]
            else:out.append('<pre><code>'+esc('\n'.join(code))+'</code></pre>');code=None
            continue
        if code is not None:code.append(line);continue
        if not line or line.startswith(('#','- ')):
            flush()
            if listing and not line.startswith('- '):out.append('</ul>');listing=False
        if not line:continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>')
        elif line.startswith('- '):
            if not listing:out.append('<ul>');listing=True
            out.append('<li>'+inline(line[2:])+'</li>')
        else:para.append(line)
    flush()
    if listing:out.append('</ul>')
    return ''.join(out)
def page(path,title,body,lang):
    depth=len(Path(path).parts)-1;root='../'*depth;local=path[3:] if lang=='zh' else path
    switch=root+(local if lang=='zh' else 'zh/'+local);base=root+('zh/' if lang=='zh' else '')
    nav=[('index.html','All games','全部游戏'),('guide.html','Install & play','安装操作'),('api.html','Training API','训练接口'),('design.html','Design & classification','理念与分类'),('verification.html','Evidence','证据')]
    links=''.join('<a href="'+base+p+'">'+choose(en,zh,lang)+'</a>' for p,en,zh in nav)
    links+='<a class="language-switch" lang="'+('en' if lang=='zh' else 'zh-CN')+'" href="'+switch+'">'+('English' if lang=='zh' else '中文')+'</a><a href="'+GH+'">GitHub ↗</a>'
    footer=choose('Original JAX transitions · Full-state debug previews · Untrained policies · Human scores unverified · Apache 2.0','原生状态转移 · 完整状态调试预览 · 策略未训练 · 人类成绩未核验 · 开源许可：Apache 2.0',lang)
    result='<!doctype html><html lang="'+('zh-CN' if lang=='zh' else 'en')+'"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+' · Crucibles JAXMARL</title><link rel="stylesheet" href="'+root+'style.css"></head><body><header><a href="'+base+'index.html">CRUCIBLES <small>JAXMARL</small></a><nav>'+links+'</nav></header>'+body+'<footer>'+footer+'</footer></body></html>'
    out=D/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(result,encoding='utf-8')
def main():
    rows=read_json(S/'catalog.json')['games'];ps=profiles();assert len(ps)==256
    originals=original_previews()
    viewports=viewport_profiles()
    rows=sorted(rows,key=lambda r:(r['family']!='spatial',int(re.sub(r'\D','',r['key'])),r['key']))
    for lang in ['en','zh']:
        t=lambda a,b:choose(a,b,lang);base='zh/' if lang=='zh' else '';asset='../' if lang=='zh' else '';cards=[]
        (D/'manuals'/lang).mkdir(parents=True,exist_ok=True)
        for row in rows:
            key=row['key'];p=ps[key];g=S/'games'/key;m=read_json(g/'manifest.json');sc=read_json(g/'scene.json') if (g/'scene.json').exists() else {}
            md=manual(row,p,m,sc,lang);(D/'manuals'/lang/(key+'.md')).write_text(md,encoding='utf-8')
            title=p[lang]['title'];prefix='../../' if lang=='zh' else '../'
            media=primary(key,originals)
            figure='<figure><a href="'+prefix+media+'"><img class="demo" src="'+prefix+media+'" width="640" alt="'+esc(title)+'"></a><figcaption>'+esc(caption(key,originals,lang))+'</figcaption></figure>'
            if key in originals:
                original=originals[key]
                scope,mode=recording_metadata(original,lang)
                figure+='<p>'+t('Original recording scope: ','原始录像信息范围：')+esc(scope)+' · '+t('Mode: ','记录模式：')+esc(mode)+'</p>'
                figure+='<details><summary>'+t('Additional generated state replay (different trajectory / full-state debug)','补充：新生成状态重放（不同轨迹／全状态调试）')+'</summary><figure><a href="'+prefix+'gifs/'+key+'.gif"><img loading="lazy" class="demo" src="'+prefix+'gifs/'+key+'.gif" width="640" alt="'+esc(title)+' — '+t('generated diagnostic','生成诊断')+'"></a><figcaption>'+t('Simplified diagnostic graphics; this does not replace the original scene recording.','简化诊断图，不替代上方原版场景录像。')+'</figcaption></figure></details>'
            if key in viewports:
                initial='<a href="'+prefix+media+'"><img class="demo" src="'+prefix+media+'" width="640" alt="'+esc(title)+'"></a>'
                figure=figure.replace(initial,image_markup(key,prefix+media,title,viewports,lazy=False),1)
            if key=='gi214':
                figure+='<section><h2>'+t('Build the route and control the field','修建道路并控制磁场')+'</h2><img class="demo" src="'+prefix+'walkthroughs/gi214-causal.gif" alt="'+t('Mining, bridge building, rail laying and powered freight','采矿、搭桥、铺轨和磁力货运')+'"><p>'+t('All six workers stay in place during this four-step native demonstration. Mining transfers stock into inventory, construction changes the map, and power/polarity changes move the freight. Single-action wait controls verify these effects. This is not a completed delivery or a human score.','这段四步原生演示中，六名角色都没有移动。采矿使矿床资源进入背包，建造改变地图，供电和磁体极性改变货物运动。单动作等待对照验证了这些效果。这不代表已完成配送或获得人类基准分。')+'</p><p>'+t('Play locally with ','本地运行 ')+'<code>crucibles play --game 214</code>'+t('. Play now assigns all six roles. Click workers to select, WASD to move, or click nearby deposits and magnets for their actions.','。Play now 会分配六名角色。点击角色选择，WASD 移动，点击附近矿床、磁体等物体选择动作。')+'</p></section>'
            if key=='gi215':
                walkthrough=t('A real cooperative order','真实协作完成一单')
                figure+='<section><h2>'+walkthrough+'</h2><img class="demo" src="'+prefix+'walkthroughs/gi215-first-order.gif" alt="'+walkthrough+'"><p>'+t('The cleaner washes and transfers a bowl; the cook prepares an ingredient, assembles soup and serves it. This recording contains 19 native transitions, with display-only animation between them. Removing the washing action prevents this order.','清洁角色洗碗并交接，厨师加工食材、装汤并上菜。这段录像包含19次原生状态更新，中间动画仅用于显示。去掉洗碗动作后，同一路线不能完成这一单。')+'</p><p>'+t('Play locally: run ','本地游玩：运行 ')+'<code>crucibles play --game 215</code>'+t(', click Play now, select a worker and use WASD or click adjacent items. Direct controls make your other owned kitchen worker wait; switch them off to queue both actions. This first-order engineering check is not a human score or all-game acceptance.','，点击 Play now，选择角色后使用 WASD，或点击身边物品。直接控制模式会让另一名自有角色等待；关闭后可分别排队两人的动作。这是一单的工程验证，不是人类成绩或全部游戏验收。')+'</p></section>'
            if key=='gi215':
                figure+='<section><h2>'+t('Shared water: one decision changes a teammate’s work','共用水路：一个决定改变队友的工作')+'</h2><p>'+t('These runs differ only at step 2: the cook opens the diversion valve, or waits. In both runs the cleaner washes a bowl and opens the drain. Unprotected water reaches the pan and prevents preparation; diversion fills the waste basin and the cook can prepare the ingredient. All changes below come from native state transitions.','两次运行仅在第2步不同：厨师打开分流阀，或者等待。两次清洁角色都洗碗并排水。未分流时脏水流入锅中，之后无法加工；分流后蓄污盆接住脏水，厨师可以加工食材。下面所有变化都来自原生状态更新。')+'</p><div class="grid">'
                for suffix,en,zh in [('unprotected','Unprotected drain: contaminated pan','直接排水：锅被污染'),('diverted','Diverted drain: preparation succeeds','提前分流：可以加工')]:
                    figure+='<figure><img class="demo" src="'+prefix+'walkthroughs/gi215-water-'+suffix+'.gif" alt="'+t(en,zh)+'"><figcaption>'+t(en,zh)+'</figcaption></figure>'
                figure+='</div></section>'
            body=markdown(md);at=body.find('</h1>')+5;body=body[:at]+figure+body[at:]
            download='<p><a class="button" download href="'+prefix+'manuals/'+lang+'/'+key+'.md">'+t('Download this manual','下载本款说明书')+'</a></p>'
            page(base+'games/'+key+'.html',title,'<main class="manual">'+download+body+'</main>',lang)
            tags=' · '.join(THEMES[v['code']][0 if lang=='en' else 1] for v in p['themes']);brief=p[lang]['brief']
            cards.append('<a class="card" data-family="'+row['family']+'" data-themes="'+' '.join(v['code'] for v in p['themes'])+'" data-search="'+esc(key+' '+title+' '+brief+' '+tags)+'" href="games/'+key+'.html"><img loading="lazy" decoding="async" width="640" height="440" src="'+asset+'gifs/'+key+'.gif" alt="'+esc(title)+'"><div class="card-copy"><span class="eyebrow">'+key.upper()+' · '+t('Spatial world' if row['family']=='spatial' else 'Abstract state','空间世界' if row['family']=='spatial' else '抽象状态')+'</span><h3>'+esc(title)+'</h3><p>'+esc(brief[:190])+('…' if len(brief)>190 else '')+'</p><p class="tags">'+esc(tags)+'</p><span class="meta">'+str(row['dimensions']['num_agents'])+t(' roles · ',' 角色 · ')+str(row['dimensions']['max_steps'])+t(' steps',' 步')+'</span><span class="manual-link">'+t('Gameplay, controls & RL manual →','玩法、操作与训练说明书 →')+'</span></div></a>')
        cards=[card.replace('src="'+asset+'gifs/'+row['key']+'.gif"','src="'+asset+primary(row['key'],originals)+'"').replace('<span class="manual-link">','<p class="media-scope">'+esc(caption(row['key'],originals,lang))+'</p><span class="manual-link">') for card,row in zip(cards,rows)]
        for i,row in enumerate(rows):
            key=row['key']
            if key in viewports:
                start=cards[i].index('<img ');end=cards[i].index('>',start)+1
                cards[i]=cards[i][:start]+image_markup(key,asset+primary(key,originals),ps[key][lang]['title'],viewports)+cards[i][end:]
        opts=''.join('<option value="'+c+'">'+THEMES[c][0 if lang=='en' else 1]+'</option>' for c in THEMES)
        hero='<div class="hero"><p class="eyebrow">'+t('MULTI-AGENT WORLDS · NATIVE JAX','多智能体世界 · 原生内核')+'</p><h1>'+t('Every world.<br>Every consequence.','每一个世界，<br>每一次行动的后果。')+'</h1><p class="lead">'+t('All 256 games, animated. Open any card for its name, explanation, gameplay, controls, design ideas and reinforcement-learning manual.','二百五十六款游戏逐款展示动态预览。打开任一卡片，查看名称、解释、玩法、操作、设计理念与强化学习说明书。')+'</p><div class="stats"><span>180 '+t('spatial games','款空间游戏')+'</span><span>76 '+t('abstract games','款抽象游戏')+'</span><span>256 '+t('GIFs · 512 manuals','份动图 · 512 份说明书')+'</span></div><p><a class="button" href="guide.html">'+t('Install & play →','安装与游玩 →')+'</a><a href="design.html">'+t('Read the design and classification rationale','阅读完整设计与分类依据')+'</a></p></div>'
        gallery='<section id="collection"><h2>'+t('All games, animated','全部游戏动态目录')+'</h2><p>'+t('Scroll to load each game\'s GIF. Every card has its own recording. Full-state debug may reveal private information. These are saved untrained trajectories, not solved-task claims. Game text stays English; the site and manuals switch language.','滚动即可加载各款动图，每张卡片都有独立预览。完整状态调试视角可能暴露私人信息；保存的未训练轨迹不代表已解出任务。游戏文字固定英文，网站与说明书可以切换语言。')+'</p><div class="filters"><input id="search" aria-label="'+t('Search games','搜索游戏')+'" placeholder="'+t('Search name, mechanic or ID','搜索名称、机制或编号')+'"><select id="family" aria-label="'+t('Interface','接口')+'"><option value="">'+t('All interfaces','全部接口')+'</option><option value="spatial">'+t('Spatial worlds','空间世界')+'</option><option value="legacy_array">'+t('Abstract states','抽象状态')+'</option></select><select id="theme" aria-label="'+t('Mechanism','机制')+'"><option value="">'+t('All mechanisms','全部机制')+'</option>'+opts+'</select></div><p id="count" aria-live="polite">256 '+t('games','款游戏')+'</p><div class="grid">'+''.join(cards)+'</div></section>'
        hero=hero.replace(t('All 256 games, animated.','二百五十六款游戏逐款展示动态预览。'),t('Explore 180 spatial games first, with the existing scene recordings restored. All 256 environments remain available.','优先浏览一百八十款空间游戏，恢复已有原版场景预览；全部二百五十六个环境仍可检索。'))
        hero=hero.replace('256 '+t('GIFs · 512 manuals','份动图 · 512 份说明书'),'157 '+t('original previews · 512 manuals','份原版预览 · 512 份说明书'))
        gallery=gallery.replace(t('All games, animated','全部游戏动态目录'),t('Spatial games & original scenes','空间游戏与原版场景'))
        gallery=gallery.replace(t('Scroll to load each game\'s GIF. Every card has its own recording. Full-state debug may reveal private information. These are saved untrained trajectories, not solved-task claims. Game text stays English; the site and manuals switch language.','滚动即可加载各款动图，每张卡片都有独立预览。完整状态调试视角可能暴露私人信息；保存的未训练轨迹不代表已解出任务。游戏文字固定英文，网站与说明书可以切换语言。'),t('Existing scene previews take priority for 157 games; other games show generated native-state replays. Original previews retain their text and information scope. Two originals are still images. Additional diagnostic replays are available on each game page. Abstract environments remain available through the interface filter. Recordings do not establish task success or human scores.','157款优先展示已有原版场景预览，其余展示生成的原生状态重放。原版保留图中文字与信息范围，其中两款原版为静态图；游戏页另有补充诊断重放。抽象环境可通过接口筛选查看。录像不代表任务成功或人类成绩。'))
        gallery=gallery.replace('<option value="spatial">','<option value="spatial" selected>')
        page(base+'index.html',t('All games','全部游戏'),'<main>'+hero+gallery+'</main><script src="'+asset+'catalog.js"></script>',lang)
        for name,title in [('guide',t('Install & play','安装操作')),('api',t('Training API','训练接口'))]:
            body=(D/(name+'-body.html')).read_text(encoding='utf-8') if lang=='en' else '<main class="manual">'+markdown((R/'manuals'/(name+'.zh.md')).read_text(encoding='utf-8'))+'</main>'
            page(base+name+'.html',title,body,lang)
        design=(R/'manuals'/('design.'+lang+'.md')).read_text(encoding='utf-8')
        (D/'manuals'/('design.'+lang+'.md')).write_text(design,encoding='utf-8')
        page(base+'design.html',t('Design & classification','理念与分类'),'<main class="manual">'+markdown(design)+'</main>',lang)
        evidence=t('# Evidence and limits\n\nThe v0.1.0 archive records 256 native full-episode checks, 256 browser-control checks and 180 spatial keyboard checks. Original candidate and manifest hashes remain unchanged.\n\n179 identities had exact eager/JIT parity; 77 had measured nonzero floating differences and remain numerically undetermined. Human scores remain unverified. A pass or a GIF does not establish learning, transfer or strategic depth.\n\nNew presentation receipts record native replay steps, captured endpoints, source identity, GIF hashes and reward differences against saved traces, without a new numerical tolerance.\n\n[Browse actual receipts]('+GH+'/tree/main/verification)\n\n[Read the complete evaluation limits](design.html)','# 证据与边界\n\n初版档案记录全部二百五十六款原生完整回合检查、二百五十六款浏览器控制检查及一百八十款空间游戏键盘检查。原始内核和清单哈希保持不变。\n\n一百七十九款即时与编译数值完全一致，七十七款存在实测非零差异，数值结论仍未定。人类成绩尚未核验。工程通过或动图不能证明学习、迁移或策略深度。\n\n新的展示回执记录原生重放步数、真实端点、源码身份、动图哈希及相对保存轨迹的奖励差异，不增加数值验收容差。\n\n[查看真实回执]('+GH+'/tree/main/verification)\n\n[阅读完整评测边界](design.html)')
        page(base+'verification.html',t('Evidence','证据'),'<main class="manual">'+markdown(evidence)+'</main>',lang)
    (D/'catalog.json').write_text(json.dumps(dict(games=[dict(family=r['family'],dimensions=r['dimensions'],**ps[r['key']]) for r in rows]),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Built 2 catalogs, 512 complete game manuals and paired design/guide/API pages')
if __name__=='__main__':main()
