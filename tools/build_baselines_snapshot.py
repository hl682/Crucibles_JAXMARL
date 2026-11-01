"""Build the frozen 2026-11-01 cloud baseline Pages snapshot (stdlib only).

Default: rebuild from the public, repository-local source bundle.
Import: --review REVIEW.json --runs RETURNED_RUNS_DIRECTORY
Only runs explicitly named by the review are imported; source directories are never scanned.
No training, evaluation, JAX import, network connection, or checkpoint loading.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "baselines"
TOTAL_GAMES = 256
FILES = ("config.json", "status.json", "provenance.json", "eval_curve.jsonl", "learning_curve.svg", "learning_curve.png")
PUBLIC_FILES = FILES + ("tracking_public.json",)
TRACKING_FIELDS = ("project", "experiment_name", "run_id", "url", "mode")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reviewed_runs(review):
    require(review.get('status') == 'RETURNED_CLOUD_ARTIFACTS_REVIEWED_NOT_FULL_LIBRARY_ACCEPTANCE', 'An explicit independent review receipt is required')
    rows = review.get('reviewed_runs', [])
    require(isinstance(rows, list) and 0 < len(rows) <= TOTAL_GAMES, 'Invalid reviewed run collection')
    require(review.get('reviewed_count') == len(rows), 'Review count does not match its explicit collection')
    names, games = [], []
    for row in rows:
        name = row['run']
        match = re.fullmatch(r'((?:gi\d{3}|v1_\d{3}))_ippo_v2_seed\d+', name)
        require(match is not None, 'Unsupported or unsafe reviewed run identity')
        names.append(name)
        games.append(match.group(1))
        require(set(row['critical_file_hashes']) == set(FILES) | {'checkpoint.pkl'}, 'Incomplete or extra critical file receipt')
        for value in [row['archive_sha256'], *row['critical_file_hashes'].values()]:
            require(re.fullmatch('[0-9a-f]{64}', value) is not None, 'Invalid review SHA-256')
        require(row['positive_return_difference'] == (row['native_return'] > row['random_return']), 'Inconsistent comparison label')
        require(not row['task_completion_or_generalization_acceptance'], 'This snapshot does not claim task/generalization acceptance')
    require(len(set(names)) == len(names) and len(set(games)) == len(games), 'Duplicate reviewed run or game')
    require(review['positive_return_difference_count'] == sum(r['positive_return_difference'] for r in rows), 'Inconsistent comparison count')
    require(not review['scientific_completion'], 'Scientific completion is outside this baseline snapshot')
    return names


def verify_run(folder, receipt):
    config, status, provenance = [read(folder / name) for name in FILES[:3]]
    curve = [json.loads(line) for line in (folder / 'eval_curve.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(curve) == receipt['eval_points'], 'Initial-stage evaluation points do not match review')
    require(len(set(provenance['eval_seeds'])) == 4, 'Expected four distinct evaluation seeds')
    require(config['train_seed'] == provenance['training_seed'] == receipt['training_seed'], 'Training seed mismatch')
    require(config['eval_seeds'] == provenance['eval_seeds'] == receipt['eval_seeds'], 'Evaluation seed mismatch')
    require(status['status'] == receipt['status'], 'Execution status mismatch')
    budget, interval = config['total_joint_steps'], config['eval_interval_joint_steps']
    require(curve[-1]['train_joint_steps'] == status['train_joint_steps'] == receipt['joint_steps'] == budget, 'Initial-stage joint-step endpoint mismatch')
    require(status['planned_joint_steps'] == budget, 'Initial-stage planned budget mismatch')
    require(curve[-1]['ippo_mean_agent_return'] == receipt['native_return'], 'IPPO endpoint mismatch')
    require(curve[-1]['random_mean_agent_return'] == receipt['random_return'], 'Random endpoint mismatch')
    require(curve[-1]['ippo_native_event_means'] == receipt['native_event_means'], 'Native event mismatch')
    require(config['source_sha256'] == provenance['game_source_sha256'], 'Source identity mismatch')
    require(config['manifest_sha256'] == provenance['game_manifest_sha256'], 'Manifest identity mismatch')
    require(provenance['no_reward_shaping'] is True, 'Native reward contract mismatch')
    require([p['train_joint_steps'] for p in curve] == list(range(0,budget,interval)) + [budget], 'Evaluation order or joint-step grid mismatch')
    return config, status, provenance, curve


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def public_text_check(text):
    # Abort instead of silently publishing paths or credentials from a later input.
    forbidden = (r"(?<![A-Za-z])[A-Za-z]:[\\/]", r"/root/", r"/home/", r"/Users/", r"\\\\[^\\]+\\",
                 r'"(?:password|api_key|access_token|secret|hostname|host_ip)"\s*:')
    for pattern in forbidden:
        if re.search(pattern, text, re.I):
            raise ValueError("Non-public material found in candidate public artifact")


def import_sources(review_path, runs_path):
    review = read(review_path)
    names = reviewed_runs(review)
    public_text_check(review_path.read_text(encoding="utf-8"))
    catalog = {r["key"]: r for r in read(ROOT / "docs" / "catalog.json")["games"]}
    existing = OUT / 'data'
    if existing.exists():
        require(all(p.is_dir() and p.name in names for p in existing.iterdir()), 'Extra unreviewed published run; resolve it explicitly before importing')
    # Finish the entire read-only preflight before mutating the published bundle.
    # The source run directory may contain later returns; none are discovered or imported.
    for receipt in review['reviewed_runs']:
        folder = runs_path / receipt['run']
        for name, expected in receipt['critical_file_hashes'].items():
            require(sha(folder / name) == expected, f'Receipt hash mismatch: {receipt["run"]}/{name}')
        verify_run(folder, receipt)
        for name in FILES:
            if not name.endswith('.png'):
                public_text_check((folder / name).read_text(encoding='utf-8'))
        tracking = read(folder / 'tracking.json')
        public_tracking = {k: tracking[k] for k in TRACKING_FIELDS}
        public_text_check(json.dumps(public_tracking))
        url = urlsplit(public_tracking['url'])
        require(url.scheme == 'https' and url.netloc == 'swanlab.cn' and '/runs/' in url.path, 'Invalid public tracking URL')
        game = receipt['run'].rsplit('_ippo_v2_seed',1)[0]
        require(game in catalog, 'Reviewed game is absent from the catalog')
        destination = existing / receipt['run']
        if destination.exists():
            require(set(p.name for p in destination.iterdir()) == set(PUBLIC_FILES), 'Extra or missing published run file')
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(review_path, OUT / "root_review.json")
    records = []
    for receipt in review["reviewed_runs"]:
        run = receipt["run"]
        source = runs_path / run
        destination = OUT / "data" / run
        destination.mkdir(parents=True, exist_ok=True)
        # Verify every critical receipt hash, including the unexported checkpoint.
        # Read bytes only; never deserialize executable checkpoint objects.
        for name, expected in receipt["critical_file_hashes"].items():
            assert sha(source / name) == expected, f"Receipt hash mismatch: {run}/{name}"
        for name in FILES:
            if not name.endswith(".png"):
                public_text_check((source / name).read_text(encoding="utf-8"))
            shutil.copyfile(source / name, destination / name)
            assert sha(destination / name) == receipt["critical_file_hashes"][name]
        tracking = read(source / "tracking.json")
        url = urlsplit(tracking["url"])
        assert url.scheme == "https" and url.netloc == "swanlab.cn" and "/runs/" in url.path
        # tracking.json contains a private logdir; export only this allowlist.
        write_json(destination / "tracking_public.json", {k: tracking[k] for k in ("project", "experiment_name", "run_id", "url", "mode")})
        game = run.rsplit('_ippo_v2_seed',1)[0]
        records.append({"run": run, "game": game, "title": catalog[game]["en"]["title"], "title_zh": catalog[game]["zh"]["title"],
                        "tracking_source_sha256": sha(source / "tracking.json"),
                        "public_file_sha256": {name: sha(destination / name) for name in FILES + ("tracking_public.json",)}})
    write_json(OUT / "snapshot_source.json", {
        "schema": "crucibles-cloud-baseline-static-snapshot-v1", "snapshot_date": datetime.fromtimestamp(review["observed_unix"], timezone.utc).strftime("%Y-%m-%d"),
        "root_review_sha256": sha(OUT / "root_review.json"),
        "frozen_coverage": {"reviewed_games": len(names), "library_games": TOTAL_GAMES, "runs": names},
        "policy": "Frozen reviewed first-round runs only; no directory discovery; no performance selection.",
        "records": records,
    })


def load_verified():
    source = read(OUT / "snapshot_source.json")
    review = read(OUT / "root_review.json")
    names = reviewed_runs(review)
    assert sha(OUT / "root_review.json") == source["root_review_sha256"]
    require(source["frozen_coverage"] == {"reviewed_games": len(names), "library_games": TOTAL_GAMES, "runs": names}, 'Published coverage does not equal the review')
    require([r['run'] for r in source['records']] == names, 'Missing, extra, or reordered published run')
    require(set(p.name for p in (OUT/'data').iterdir()) == set(names), 'Published run directory differs from the review')
    receipts = {r["run"]: r for r in review["reviewed_runs"]}
    results = []
    for record in source["records"]:
        run = record["run"]
        folder = OUT / "data" / run
        require(record['game'] == run.rsplit('_ippo_v2_seed',1)[0], 'Published game identity mismatch')
        require(set(record['public_file_sha256']) == set(PUBLIC_FILES), 'Missing or extra public file hash')
        require(set(p.name for p in folder.iterdir()) == set(PUBLIC_FILES), 'Missing or extra published run file')
        for name, expected in record["public_file_sha256"].items():
            assert sha(folder / name) == expected, f"Public input hash mismatch: {run}/{name}"
            if name in FILES:
                assert expected == receipts[run]["critical_file_hashes"][name]
        receipt = receipts[run]
        config, status, provenance, curve = verify_run(folder,receipt)
        require(set(read(folder/'tracking_public.json')) == set(TRACKING_FIELDS), 'Tracking allowlist mismatch')
        results.append(dict(record, config=config, status=status, provenance=provenance, curve=curve,
                            receipt=receipt, tracking=read(folder / "tracking_public.json")))
    return source, review, results


def esc(value):
    return html.escape(str(value), quote=True)


def number(value):
    return f"{value:,.6f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"


def status_label(value):
    return {
        'BUDGET_COMPLETE_NOT_PROOF_OF_LEARNING': 'Initial budget completed; learning has not been established',
        'PREPARED_NOT_RUN': 'Prepared; not yet run',
        'RUNNING': 'Run in progress',
        'FAILED': 'Run failed; see the execution receipt',
        'STOPPED': 'Run stopped; see the execution receipt',
    }.get(value, 'See the execution receipt for the recorded run state')


def link(href, label):
    return f'<a href="{esc(href)}">{esc(label)}</a>'


def table(headers, rows, caption=None):
    return '<div class="table-scroll"><table>' + (f'<caption>{esc(caption)}</caption>' if caption else '') + '<thead><tr>' + ''.join(f'<th scope="col">{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


CSS = '''.baseline-main{max-width:1320px}.baseline-main h1{font-size:clamp(38px,5.5vw,70px)}.snapshot-note{padding:22px 26px;background:#182a34;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px}.snapshot-note strong{color:var(--fg)}.baseline-card{padding:28px;margin-top:32px;border:1px solid var(--line);border-radius:8px;background:#15252e;scroll-margin-top:20px}.baseline-card h2{margin-top:6px}.baseline-card figure{margin:22px 0}.baseline-card figure img{max-width:100%;width:100%;background:white;border-radius:5px}.baseline-card .status{font-size:12px;overflow-wrap:anywhere;color:var(--accent)}.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;text-align:left;font-variant-numeric:tabular-nums}th,td{padding:12px 15px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--fg);font-weight:600}td{color:var(--muted)}caption{text-align:left;color:var(--muted);padding:12px 0}code{overflow-wrap:anywhere;font-size:12px}details pre{max-height:420px;overflow:auto}.native-warning{color:#ffd0a3;border-left:3px solid #ffd0a3;padding-left:14px}.evidence-links{display:flex;flex-wrap:wrap;gap:14px;font-size:13px}.protocol{max-width:1000px}.baseline-main section{padding-top:26px}.baseline-main .hero{padding-top:25px}.jump-list{display:flex;flex-wrap:wrap;gap:12px}.jump-list a{padding:6px 12px;border:1px solid var(--line);border-radius:4px}@media(max-width:720px){.baseline-card{padding:16px}th,td{padding:10px}header{padding:18px 5%}}'''
CSS += '''
.baseline-main{padding-top:18px}.curve-intro h1{font-size:clamp(28px,3vw,40px);letter-spacing:-1px;line-height:1.15;margin:8px 0 12px}.curve-intro p{margin:5px 0;font-size:14px;line-height:1.5}.curve-intro .coverage{color:var(--fg)}.curve-intro .snapshot-time{font-size:12px}.baseline-card{margin-top:18px;padding:20px}.baseline-card h2{font-size:24px;margin:4px 0 10px}.baseline-card>.eyebrow{margin:0 0 5px}.baseline-card figure{margin:12px 0 18px}.context-details>summary{font-size:20px}.context-details .snapshot-note{margin-top:20px}@media(max-width:720px){.baseline-main{padding-top:12px}.curve-intro h1{font-size:28px;margin:4px 0 10px}.curve-intro p{font-size:12px}.baseline-card{padding:12px;margin-top:14px}.baseline-card h2{font-size:20px}.baseline-card figure{margin-top:8px}header{padding-top:12px;padding-bottom:12px}header>a{font-size:20px}nav{font-size:11px;gap:10px;margin-top:10px}}
'''


ZH = {
    'Initial budget completed; learning has not been established': '首轮预算完成；未据此判定学会',
    'Prepared; not yet run': '已准备；尚未运行',
    'Run in progress': '运行中',
    'Run failed; see the execution receipt': '运行失败；详见执行回执',
    'Run stopped; see the execution receipt': '运行已停止；详见执行回执',
    'See the execution receipt for the recorded run state': '运行状态详见执行回执',
    'Joint environment steps': '联合环境步',
    'Joint environment steps (completed / planned)': '联合环境步（完成／计划）',
    'The joint environment step count': '联合环境步数',
    'Cloud learning curves': '云端学习曲线',
    'First-round, single-seed results; see each card for task-completion evidence.': '首轮单种子结果，任务完成情况见各卡片。',
    'Methods, coverage, interpretation and source integrity': '方法、覆盖范围、结果解读与来源完整性',
    'Cloud baseline snapshot · Crucibles JAXMARL': '云端基线阶段快照 · Crucibles JAXMARL',
    'All games': '全部游戏', 'Install & play': '安装操作', 'Training API': '训练接口', 'Evidence': '证据', 'Cloud baselines': '云端基线',
    'CLOUD BASELINES · FROZEN FIRST-ROUND SNAPSHOT': '云端基线 · 首轮冻结快照',
    'Learning curves.': '学习曲线。', 'Evidence in context.': '同时呈现证据边界。',
    '1 training seed per game': '每款游戏 1 个训练种子',
    '4 evaluation seeds per game': '每款游戏 4 个评估种子', '262,144 joint steps per run': '每次运行 262,144 个联合步',
    'Budget completion is not proof of learning, task solution, statistical significance, or generalization.': '预算完成不等于已经学会、完成任务、具有统计显著性或能够泛化。',
    'Returns use each game\'s own reward scale. Do not rank games or aggregate performance across games. Four evaluation seeds from one trained policy are not four independent training replications. Failures, negative results and missing event evidence remain visible.': '回报沿用各游戏自己的奖励尺度，不应跨游戏排名或聚合性能。同一训练策略的 4 个评估种子不等于 4 次独立训练重复。失败、负向结果和缺失的事件证据均保留。',
    'How to read this snapshot': '如何阅读这份快照',
    'The learner is role-conditioned, parameter-shared IPPO with a feedforward policy and four frames of own-observation history. It uses native legal actions, native rewards without shaping, and no privileged critic. Evaluation uses stochastic legal-action sampling with the same independent seeds and native horizon for IPPO and the uniform-legal random baseline.': '学习器采用角色条件化、参数共享的 IPPO，策略为前馈网络，输入包含自身观测的 4 帧历史。它使用原生合法动作与未经塑形的原生奖励，评论器不使用特权信息。IPPO 与均匀合法随机基线使用相同的独立评估种子和原生时域，以随机合法动作采样进行评估。',
    'Every retained evaluation point is shown. The reported endpoint is the final budget checkpoint; evaluation scores were not used to choose a checkpoint. The mean native return averages across roles and evaluation episodes; the role vector is retained below because averaging can conceal opposing outcomes. ': '所有留存的评估点均展示。报告的终点是预算结束时的检查点，没有用评估分数选择检查点。平均原生回报对角色和评估回合取平均；下方同时保留角色向量，因为平均值可能掩盖相反的结果。',
    ' counts joint environment transitions across parallel environments, not individual agent actions.': ' 统计并行环境中的联合环境转移次数，而非单个智能体的动作数。',
    'The original cloud SVG and PNG files are copied byte for byte. Exact values and paired episode differences remain in the original JSONL. No local training or evaluation was performed to generate this page.': '云端原始 SVG 与 PNG 文件逐字节复制。精确数值和配对回合差值保留在原始 JSONL 中。本页生成过程没有执行本地训练或评估。原始曲线、数据附件与溯源字段沿用英文，不作翻译或重绘。',
    'Reviewed runs in game-ID order': '按游戏编号排列的已复核运行',
    'Final checkpoint only; values are not comparable across different reward scales.': '仅列最终检查点；不同奖励尺度之间的数值不可直接比较。',
    'Game': '游戏', 'IPPO native return': 'IPPO 原生回报', 'Random native return': '随机基线原生回报', 'Training seed': '训练种子', 'Tracking': '运行追踪', 'SwanLab run ↗': 'SwanLab 运行记录 ↗',
    'Negative comparison retained:': '保留负向对照结果：',
    'Zero comparison retained:': '保留零差异对照结果：',
    ' final IPPO mean native return does not exceed the random baseline in this run. This is an observed single-training-seed result.': ' 本次运行的最终 IPPO 平均原生回报未高于随机基线。这是单个训练种子的观测结果。',
    'No orders served:': '没有完成订单：', ' final mean ': ' 最终平均 ',
    ' for both IPPO and random. IPPO has a less negative native return here, but this does not establish task completion. ': '，IPPO 与随机基线均如此。IPPO 的原生回报在此较不负，但不能据此判定任务完成。',
    ' remains zero at every recorded evaluation point.': ' 在所有已记录评估点均为零。',
    'Download original PNG': '下载原始 PNG', 'Open original SVG': '打开原始 SVG',
    'Endpoint evidence': '终点评估证据', 'Observed value': '观测值',
    'IPPO / random mean native episode return': 'IPPO / 随机基线平均原生回合回报',
    'IPPO per-role return': 'IPPO 各角色回报', 'Random per-role return': '随机基线各角色回报',
    'Training seed (one independent trained policy)': '训练种子（1 个独立训练策略）',
    'Evaluation seeds (four episodes per policy checkpoint)': '评估种子（每个策略检查点 4 个回合）',
    'Runtime / updates': '运行耗时 / 更新次数', 'Recorded backend / numerical versions': '记录的后端 / 数值库版本',
    'Native terminal event or state metric': '原生终止事件或状态指标', 'IPPO mean': 'IPPO 均值', 'Random mean': '随机基线均值',
    'Native metrics retain their original semantics: counters, state levels and termination flags are not interchangeable.': '原生指标保留原义：计数器、状态量与终止标记不能相互替代。',
    'Native task-event metrics were not recorded for this game in this source bundle. Nonzero return alone does not prove that its task was solved.': '这份来源包没有记录该游戏的原生任务事件指标。仅凭非零回报不能证明任务已解决。',
    'All evaluation points': '全部评估点', 'Protocol, identity and provenance': '协议、身份与溯源',
    'No selection by evaluation; initial final checkpoint reported, every intermediate point retained': '不根据评估结果进行选择；报告首轮预算结束时的最终检查点，保留所有中间点。',
    'Complete frozen joint-step exposure. Stop safely for nonfinite values, native contract error, resource limit, or explicit coordinator/user instruction. Wall slices resume same identity.': '完成冻结的联合步数预算。遇到非有限数值、原生接口约定错误、资源限制或协调者／用户明确指令时安全停止。分时段运行续接同一身份。',
    'The configuration retains its original prepared, not-yet-run state. Execution state comes from the separate actual status receipt shown above.': '配置保留运行前的“已准备、尚未运行”状态。实际执行状态来自上方单独展示的真实状态回执。',
    'Original reviewed archive SHA-256: ': '已复核原始归档的 SHA-256：',
    'Original evaluation JSONL': '原始评估 JSONL', 'Configuration': '配置', 'Execution status': '执行状态', 'Provenance': '溯源', 'Public tracking record': '公开追踪记录',
    'Source bundle and integrity': '来源包与完整性',
    'This page is generated from a repository-local frozen bundle. The review receipt identifies the exact reviewed runs and source hashes. Public curves, configuration, execution status, provenance and evaluation JSONL match those hashes. Tracking metadata is exported through an explicit public-field allowlist; private logging paths are excluded. Checkpoint hashes are retained in the review, but executable checkpoint files are not published.': '本页由仓库内的冻结资料包生成。复核回执明确标识本快照中的运行及其来源哈希。公开曲线、配置、执行状态、溯源和评估 JSONL 与对应哈希一致。追踪元数据仅导出明确允许公开的字段，排除私有日志路径。复核回执保留检查点哈希，但不公开可执行的检查点文件。',
    'Snapshot source JSON': '快照来源 JSON', 'Independent review receipt': '独立复核回执', 'SHA-256 file manifest': '文件 SHA-256 清单', 'Rebuild and provenance notes': '重建与溯源说明（英文）',
    'Known limits: one training seed per game; fixed initial exposure; no convergence test; no significance claim; no library-wide coverage; no task-solution or generalization acceptance. All source statuses remain separate from scientific acceptance.': '已知边界：每款游戏仅 1 个训练种子；固定首轮预算；没有收敛检验；不声称显著性；未覆盖完整游戏库；未验收任务解决或泛化。所有来源状态均与科学验收分开。',
    'Frozen cloud baseline evidence · Unmodified native rewards · Single training seed · No task-completion or generalization acceptance · Apache 2.0': '冻结云端基线证据 · 未修改原生奖励 · 单训练种子 · 未验收任务完成或泛化 · Apache 2.0',
}


def budget_range(results):
    values = [r['config']['total_joint_steps'] for r in results]
    low, high = min(values), max(values)
    return f'{low:,}' if low == high else f'{low:,}–{high:,}'


def chinese_page(page, results, observed):
    translations = dict(ZH)
    count = len(results)
    budgets = budget_range(results)
    translations[f'{count} / {TOTAL_GAMES} games · 1 training seed · 4 evaluation seeds · {budgets} joint steps, frozen per-game budgets'] = f'{count} / {TOTAL_GAMES} 个游戏 · 1 个训练种子 · 4 个评估种子 · 按各款冻结预算运行 {budgets} 个联合步'
    translations['This scalar curve summary has no task-event metrics. Consult the native event definitions and per-role returns; missing summary fields are not zero-valued observations.'] = '这份标量曲线摘要没有任务事件指标。请结合原生事件定义和逐角色回报；摘要中缺少字段不代表观测值为零。'
    text = f'Coverage is frozen to the {count} runs in the review receipt observed at {observed}. Later files and runs do not enter this snapshot automatically. These fixed-learner baselines do not implement or validate the planned LLM/UED or meta-gradient method.'
    translations[text] = f'覆盖范围冻结为 {observed} 复核回执中的 {count} 次运行。后续文件和运行不会自动进入本快照。这些固定学习器基线没有实现或验证计划中的 LLM/UED 或元梯度方法。'
    translations[f'Reviewed snapshot: {observed}'] = f'复核快照时间：{observed}'
    for r in results:
        translations[r['title']] = r['title_zh']
        translations[r['game'].upper()+' · '+r['title']] = r['game'].upper()+' · '+r['title_zh']
        translations[r['game'].upper()+' · INITIAL BASELINE'] = r['game'].upper()+' · 首轮基线'
        s = r['status']
        translations[f'{number(s["wall_seconds"])} seconds / {number(s["updates"])} updates'] = f'{number(s["wall_seconds"])} 秒 / {number(s["updates"])} 次更新'
        translations[f'Original cloud learning curve; {len(r["curve"])} evaluation points, one trained seed, four evaluation seeds. Axis ranges are game-specific. '] = f'云端原始学习曲线；{len(r["curve"])} 个评估点，1 个训练种子，4 个评估种子。坐标轴范围按游戏独立设置。'
    # Only translate visible text nodes. Original JSON provenance and attributes
    # remain verbatim; links and shared original assets keep identical identities.
    parts = re.split(r'(<[^>]+>)', page)
    in_pre = False
    for i, part in enumerate(parts):
        if part.startswith('<pre'):
            in_pre = True
        elif part == '</pre>':
            in_pre = False
        elif not part.startswith('<') and not in_pre:
            value = html.unescape(part)
            if value in translations:
                parts[i] = esc(translations[value])
    result = ''.join(parts).replace('<html lang="en">','<html lang="zh-CN">')
    result = result.replace('href="../index.html"','href="../zh/index.html"').replace('href="../guide.html"','href="../zh/guide.html"').replace('href="../api.html"','href="../zh/api.html"').replace('href="../verification.html"','href="../zh/verification.html"')
    result = result.replace('<a href="index.html">云端基线</a>','<a href="zh.html">云端基线</a>')
    result = result.replace('<a class="language-switch" lang="zh-CN" href="zh.html">中文</a>','<a class="language-switch" lang="en" href="index.html">English</a>')
    return result


def render(source, review, results):
    count = len(results)
    observed = datetime.fromtimestamp(review["observed_unix"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    content = ''
    content += f'''<div class="snapshot-note"><strong>Budget completion is not proof of learning, task solution, statistical significance, or generalization.</strong><p>Coverage is frozen to the {count} runs in the review receipt observed at {esc(observed)}. Later files and runs do not enter this snapshot automatically. These fixed-learner baselines do not implement or validate the planned LLM/UED or meta-gradient method.</p><p>Returns use each game's own reward scale. Do not rank games or aggregate performance across games. Four evaluation seeds from one trained policy are not four independent training replications. Failures, negative results and missing event evidence remain visible.</p></div>'''
    content += '''<section class="protocol"><h2>How to read this snapshot</h2><p>The learner is role-conditioned, parameter-shared IPPO with a feedforward policy and four frames of own-observation history. It uses native legal actions, native rewards without shaping, and no privileged critic. Evaluation uses stochastic legal-action sampling with the same independent seeds and native horizon for IPPO and the uniform-legal random baseline.</p><p>Every retained evaluation point is shown. The reported endpoint is the final budget checkpoint; evaluation scores were not used to choose a checkpoint. The mean native return averages across roles and evaluation episodes; the role vector is retained below because averaging can conceal opposing outcomes. <code>The joint environment step count</code> counts joint environment transitions across parallel environments, not individual agent actions.</p><p>The original cloud SVG and PNG files are copied byte for byte. Exact values and paired episode differences remain in the original JSONL. No local training or evaluation was performed to generate this page.</p></section>'''
    rows = []
    for r in results:
        f = r["curve"][-1]
        rows.append([link('#'+r['game'],r['game'].upper()+' · '+r['title']), number(f['train_joint_steps']), number(f['ippo_mean_agent_return']), number(f['random_mean_agent_return']), str(r['provenance']['training_seed']), link(r['tracking']['url'],'SwanLab run ↗')])
    content += '<section><h2>Reviewed runs in game-ID order</h2>' + table(['Game','Joint environment steps','IPPO native return','Random native return','Training seed','Tracking'],rows,'Final checkpoint only; values are not comparable across different reward scales.') + '</section>'
    context_content = content
    content = f'''<div class="curve-intro"><h1>Cloud learning curves</h1><p class="coverage">{count} / {TOTAL_GAMES} games · 1 training seed · 4 evaluation seeds · {budget_range(results)} joint steps, frozen per-game budgets</p><p class="snapshot-time">Reviewed snapshot: {esc(observed)}</p><p>First-round, single-seed results; see each card for task-completion evidence.</p></div>'''
    for r in results:
        f, p, c, s = r['curve'][-1], r['provenance'], r['config'], r['status']
        base = 'data/'+r['run']+'/'
        content += f'<article class="baseline-card" id="{r["game"]}"><p class="eyebrow">{r["game"].upper()} · INITIAL BASELINE</p><h2>{esc(r["title"])}</h2>'
        if f['ippo_mean_agent_return'] <= f['random_mean_agent_return']:
            comparison_label = 'Zero comparison retained:' if f['ippo_mean_agent_return'] == f['random_mean_agent_return'] else 'Negative comparison retained:'
            content += '<p class="native-warning"><strong>'+comparison_label+'</strong> final IPPO mean native return does not exceed the random baseline in this run. This is an observed single-training-seed result.</p>'
        if r['game']=='gi215':
            content += '<p class="native-warning"><strong>No orders served:</strong> final mean <code>served = 0</code> for both IPPO and random. IPPO has a less negative native return here, but this does not establish task completion. <code>served</code> remains zero at every recorded evaluation point.</p>'
        loading = 'eager' if r is results[0] else 'lazy'
        content += f'<figure><a href="{base}learning_curve.svg"><img loading="{loading}" src="{base}learning_curve.svg" alt="{esc(r["game"].upper())} original cloud learning curve: IPPO and random native return by joint environment steps"></a><figcaption>Original cloud learning curve; {len(r["curve"])} evaluation points, one trained seed, four evaluation seeds. Axis ranges are game-specific. {link(base+"learning_curve.png","Download original PNG")} · {link(base+"learning_curve.svg","Open original SVG")}</figcaption></figure><p class="status">{esc(status_label(s["status"]))}</p>'
        content += table(['Endpoint evidence','Observed value'],[
            ['Joint environment steps (completed / planned)',f'{number(s["train_joint_steps"])} / {number(s["planned_joint_steps"])}'],
            ['IPPO / random mean native episode return',f'{number(f["ippo_mean_agent_return"])} / {number(f["random_mean_agent_return"])}'],
            ['IPPO per-role return',esc(', '.join(number(v) for v in f['ippo_per_agent_return']))],
            ['Random per-role return',esc(', '.join(number(v) for v in f['random_per_agent_return']))],
            ['Training seed (one independent trained policy)',str(p['training_seed'])],
            ['Evaluation seeds (four episodes per policy checkpoint)',esc(', '.join(map(str,p['eval_seeds'])))],
            ['Runtime / updates',f'{number(s["wall_seconds"])} seconds / {number(s["updates"])} updates'],
            ['Recorded backend / numerical versions',esc(f'{p["backend"]}; JAX {p["jax"]}; NumPy {p["numpy"]}')]])
        events = f['ippo_native_event_means']
        if events:
            content += table(['Native terminal event or state metric','IPPO mean','Random mean'],[[esc(k),number(v),number(f['random_native_event_means'][k])] for k,v in events.items()], 'Native metrics retain their original semantics: counters, state levels and termination flags are not interchangeable.')
        else:
            content += '<p>This scalar curve summary has no task-event metrics. Consult the native event definitions and per-role returns; missing summary fields are not zero-valued observations.</p>'
        point_rows = [[number(q['train_joint_steps']),number(q['ippo_mean_agent_return']),number(q['random_mean_agent_return'])] for q in r['curve']]
        content += '<details><summary>All evaluation points</summary>' + table(['Joint environment steps','IPPO native return','Random native return'],point_rows) + '</details>'
        content += '<details><summary>Protocol, identity and provenance</summary><p>'+esc(c['selection_rule'])+'</p><p>'+esc(c['stop_rule'])+'</p><p>The configuration retains its original prepared, not-yet-run state. Execution state comes from the separate actual status receipt shown above.</p><pre>'+esc(json.dumps(p,indent=2))+'</pre><p>Original reviewed archive SHA-256: <code>'+esc(r['receipt']['archive_sha256'])+'</code></p></details>'
        content += '<p class="evidence-links">'+' '.join(link(base+n,label) for n,label in [('eval_curve.jsonl','Original evaluation JSONL'),('config.json','Configuration'),('status.json','Execution status'),('provenance.json','Provenance'),('tracking_public.json','Public tracking record')])+link(r['tracking']['url'],'SwanLab run ↗')+'</p></article>'
    content += '<section><details class="context-details"><summary>Methods, coverage, interpretation and source integrity</summary>' + context_content
    content += '''<section><h2>Source bundle and integrity</h2><p>This page is generated from a repository-local frozen bundle. The review receipt identifies the exact reviewed runs and source hashes. Public curves, configuration, execution status, provenance and evaluation JSONL match those hashes. Tracking metadata is exported through an explicit public-field allowlist; private logging paths are excluded. Checkpoint hashes are retained in the review, but executable checkpoint files are not published.</p><p class="evidence-links">'''+link('snapshot_source.json','Snapshot source JSON')+link('root_review.json','Independent review receipt')+link('hash_manifest.json','SHA-256 file manifest')+link('README.md','Rebuild and provenance notes')+'''</p><p>Known limits: one training seed per game; fixed initial exposure; no convergence test; no significance claim; no library-wide coverage; no task-solution or generalization acceptance. All source statuses remain separate from scientific acceptance.</p></section></details></section>'''
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cloud baseline snapshot · Crucibles JAXMARL</title><meta name="description" content="Frozen reviewed first-round IPPO cloud baseline snapshot, native returns, original learning curves and provenance."><link rel="stylesheet" href="../style.css"><link rel="stylesheet" href="snapshot.css"></head><body><header><a href="../index.html">CRUCIBLES <small>JAXMARL</small></a><nav>'+link('../index.html','All games')+link('../guide.html','Install & play')+link('../api.html','Training API')+link('../verification.html','Evidence')+link('index.html','Cloud baselines')+'</nav></header><main class="baseline-main">'+content+'</main><footer>Frozen cloud baseline evidence · Unmodified native rewards · Single training seed · No task-completion or generalization acceptance · Apache 2.0</footer></body></html>'
    page = page.replace('</nav>', '<a class="language-switch" lang="zh-CN" href="zh.html">中文</a></nav>')
    (OUT / 'index.html').write_text(page,encoding='utf-8')
    (OUT / 'zh.html').write_text(chinese_page(page,results,observed),encoding='utf-8')
    (OUT / 'snapshot.css').write_text(CSS.rstrip()+'\n',encoding='utf-8')
    (OUT / 'README.md').write_text(f'''# Frozen cloud baseline snapshot — {source['snapshot_date']}

Entry points: `index.html` (English), `zh.html` (Chinese). Original cloud plots
and source attachments remain in their original language and bytes.
Pages publishes the repository `docs/` directory, so
the site URL is `baselines/` below the existing project URL. All site-local
links are relative and work under a GitHub Pages repository prefix.

This bundle includes exactly {count} / {TOTAL_GAMES} reviewed games, not a live
directory count. The explicit collection is in `snapshot_source.json` and must
match `root_review.json` exactly. Every run uses one training seed, four evaluation seeds and
its frozen per-game budget ({budget_range(results)} joint environment steps in this snapshot).
Budget completion does not establish learning,
significance, task solution or generalization. All non-positive comparisons are preserved on their own game cards.
GI215's zero served orders are preserved when that game is in the review. No LLM/UED method is validated here.

## Rebuild from the public repository bundle

From the repository root, run `python tools/build_baselines_snapshot.py`.
Python standard library only; no network, JAX, training or evaluation required.
Existing public input hashes are checked before rendering. The existing Pages
builder leaves this independent directory intact; no workflow edit is needed.
Run `python tools/build_baselines_snapshot.py --check` for a read-only integrity,
coverage, privacy and local-link check, also called by the release checker.

## Import an explicitly reviewed cloud bundle

Run `python tools/build_baselines_snapshot.py --review REVIEW.json --runs RUNS`.
Both arguments are local paths supplied by the operator; they are never written
to public output. The published run list must equal the explicit independently reviewed set.
Missing, extra or duplicate entries and unmatched hashes are rejected. Source
folders are never scanned to discover runs; later unreviewed returns are ignored.
Every critical input file is checked against the independent review, including
the checkpoint bytes (never deserialized). Curves and the config, status,
provenance and evaluation JSONL are copied unchanged. Tracking is reduced to
public fields because its original logdir is private. No checkpoint is exported.

`snapshot_source.json` stores the frozen coverage and per-run public hashes.
`root_review.json` is the original independent review receipt, copied unchanged.
`hash_manifest.json` hashes every delivered file other than itself, plus the
builder. The manifest cannot contain its own hash. Exact floating-point values,
role vectors, paired episode differences and all evaluation points remain in
the original JSONL. Display numbers are rounded to six decimal places.

Original archive hashes in the review are provenance references, not a claim
that archives are published here. The four evaluation seeds are repeated
measurements of one learned policy, not four independent training seeds.
The config's PREPARED_NOT_RUN records preparation; execution status is in the
separate status receipt. Original plots are not regenerated or edited.
''',encoding='utf-8')


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href','src'):
            if key in attrs:
                self.links.append(attrs[key])


def validate_and_manifest(results, *, write_manifest=True):
    manifest = {p.relative_to(OUT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='hash_manifest.json'}
    expected_manifest = {'algorithm': 'sha256', 'self_excluded': True, 'builder_sha256': sha(Path(__file__)), 'files': manifest}
    if write_manifest:
        write_json(OUT / 'hash_manifest.json', expected_manifest)
    else:
        require(read(OUT / 'hash_manifest.json') == expected_manifest, 'Published hash manifest or builder identity mismatch')
    parser = Links()
    expected_ids = [r['game'] for r in results]
    for page in ['index.html','zh.html']:
        markup = (OUT/page).read_text(encoding='utf-8')
        require(re.findall(r'<article class="baseline-card" id="([^"]+)"',markup) == expected_ids, 'Missing, extra or reordered page card')
        require(markup.count('<figure>') == len(results), 'Missing or extra plotted curve')
        parser.feed(markup)
    local_links = 0
    for href in parser.links:
        parsed = urlsplit(href)
        if parsed.scheme:
            assert parsed.scheme == 'https' and parsed.netloc == 'swanlab.cn'
            continue
        assert not parsed.path.startswith('/'), 'Site-local URLs must be relative'
        if parsed.path:
            assert (OUT / unquote(parsed.path)).is_file(), f'Broken link: {href}'
        elif parsed.fragment:
            assert parsed.fragment in parser.ids
        local_links += 1
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix != '.png':
            public_text_check(path.read_text(encoding='utf-8'))
    for gi215 in (r for r in results if r['game']=='gi215'):
        assert all(q['ippo_native_event_means']['served']==q['random_native_event_means']['served']==0 for q in gi215['curve'])
    return {'status':'VERIFIED_STATIC_SNAPSHOT','reviewed_games':len(results),'library_games':TOTAL_GAMES,'evaluation_points':sum(len(r['curve']) for r in results),'unchanged_cloud_curves':2*len(results),'non_positive_comparisons':sum(not r['receipt']['positive_return_difference'] for r in results),'checked_local_links':local_links,'manifest_files':len(manifest),'entry':'docs/baselines/index.html'}


def check_snapshot():
    _, _, results = load_verified()
    return validate_and_manifest(results, write_manifest=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--runs',type=Path)
    parser.add_argument('--check', action='store_true', help='Check the existing public snapshot without writing files')
    args = parser.parse_args()
    if args.check:
        if args.review or args.runs:
            parser.error('--check cannot be combined with import arguments')
        print(json.dumps(check_snapshot()))
        return
    if bool(args.review) != bool(args.runs):
        parser.error('--review and --runs must be supplied together')
    if args.review:
        import_sources(args.review,args.runs)
    source, review, results = load_verified()
    render(source,review,results)
    print(json.dumps(validate_and_manifest(results)))


if __name__=='__main__':
    main()
