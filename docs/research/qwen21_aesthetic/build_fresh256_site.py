"""Publish a complete fixed test snapshot; never select examples by reward."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora/fresh_test256_window15")


def build(destination):
    aesthetic = json.loads((ROOT / "eval_step15/aggregate/summary.json").read_text())
    pickscore = json.loads((ROOT / "pickscore/vs_base/summary.json").read_text())
    integrity = json.loads((ROOT / "eval_step15/pair_integrity.json").read_text())
    assert integrity["valid"] and integrity["pairs"] == 256
    output = destination / "fresh256"
    output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for gpu in range(4):
        rows = [
            json.loads(s)
            for s in (ROOT / f"eval_step15/gpu{gpu}/report/scores.jsonl").read_text().splitlines()
        ]
        for index in (0, 21, 42, 63):
            pair = [
                next(
                    x for x in rows if x["prompt_index"] == index and x["checkpoint_label"] == arm
                )
                for arm in ("base", "lora15")
            ]
            assert pair[0]["prompt"] == pair[1]["prompt"] and pair[0]["seed"] == pair[1]["seed"]
            item = dict(
                number=gpu * 64 + index, prompt=pair[0]["prompt"], seed=pair[0]["seed"], images=[]
            )
            for row in pair:
                source = Path(row["image_path"])
                assert hashlib.sha256(source.read_bytes()).hexdigest() == row["image_sha256"]
                name = f"{item['number']:03d}_{row['checkpoint_label']}.png"
                shutil.copyfile(source, output / name)
                assert (
                    hashlib.sha256((output / name).read_bytes()).hexdigest() == row["image_sha256"]
                )
                item["images"].append(dict(src="fresh256/" + name, label=row["checkpoint_label"]))
            pairs.append(item)
    table = []
    for title, summary, metric in [
        ("美学评分", aesthetic, "aesthetic"),
        ("独立 PickScore", pickscore, "pickscore"),
    ]:
        score = summary["arms"]["lora15"]["scores"][metric]
        delta = score["paired_delta_from_base"]
        assert delta["count"] == 256
        lo, hi = delta["bootstrap_95ci"]
        table.append(
            f"<tr><td>{title}</td><td>{delta['mean']:+.5f}</td><td>[{lo:+.5f}, {hi:+.5f}]</td><td>{round(delta['win_rate'] * 256)}/256</td></tr>"
        )
    for name, data in [
        ("aesthetic", aesthetic),
        ("pickscore", pickscore),
        ("integrity", integrity),
        ("pairs", pairs),
    ]:
        (output / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen 2.1 · 256 条固定测试</title><style>body{margin:0;background:#f5f4f0;color:#202923;font:17px/1.7 system-ui}main{max-width:1250px;margin:auto;padding:24px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}a{color:#23624b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;display:block}button,select{font:inherit;padding:8px;margin:4px}@media(max-width:650px){.grid{grid-template-columns:1fr}section{padding:12px}table{font-size:13px}}</style><main><a href="index.html">← 实验首页</a><h1>256 条固定测试：美学改善尚未确认</h1><p>Qwen-Image-2.1 · 两步加噪训练 · LoRA 第 15 轮（30 个优化步）vs Base</p><p>这是已完成测试的结果快照，并非实时训练监控。</p><section><h2>全部 256 对的统计</h2><table><tr><th>指标</th><th>LoRA − Base</th><th>95% 配对区间</th><th>分数胜出</th></tr>TABLE</table><p>美学差值区间跨零；246/256 张变亮，但变亮不等于更好看。分数胜出不是人类偏好胜率。PickScore 未用于训练，仍属于模型代理评分。</p><a href="fresh256/aesthetic.json">美学统计</a> · <a href="fresh256/pickscore.json">PickScore 统计</a> · <a href="fresh256/integrity.json">配对校验</a></section><section><h2>原生采样配对原图</h2><p>1024 × 1024，40 步，同提示词、同种子。按每个分片固定序号 0、21、42、63 展示 16 对，未按评分筛选。点击查看原图。</p><button id="prev">上一组</button><select id="sel" aria-label="选择测试提示词"></select><button id="next">下一组</button><p id="prompt"></p><p id="seed"></p><div class="grid" id="images"></div></section><section><h2>训练安排</h2><p>已安排从第 15 轮 checkpoint 继续，目标累计 240 个优化步（120 轮），在第 60、120 轮检查固定验证集。相同步数不保证复现论文效果。这批测试结果已被查看，后续版本不能再称其为全新测试。</p></section></main><script>fetch('fresh256/pairs.json').then(r=>r.json()).then(data=>{let i=0;const sel=document.getElementById('sel');data.forEach((p,j)=>{const o=document.createElement('option');o.value=j;o.textContent='测试提示词 '+p.number;sel.append(o)});function render(){const p=data[i];sel.value=i;document.getElementById('prompt').textContent=p.prompt;document.getElementById('seed').textContent='Seed '+p.seed;const grid=document.getElementById('images');grid.replaceChildren();p.images.forEach(x=>{const f=document.createElement('figure'),c=document.createElement('figcaption'),a=document.createElement('a'),im=document.createElement('img');c.textContent=x.label==='base'?'Base':'LoRA · 第 15 轮';a.href=x.src;a.target='_blank';a.rel='noopener';im.src=x.src;im.alt=c.textContent+' — '+p.prompt;a.append(im);f.append(c,a);grid.append(f)})}sel.onchange=()=>{i=Number(sel.value);render()};document.getElementById('prev').onclick=()=>{i=(i+data.length-1)%data.length;render()};document.getElementById('next').onclick=()=>{i=(i+1)%data.length;render()};render()});</script></html>"""
    (destination / "fresh256.html").write_text(page.replace("TABLE", "".join(table)))
    print(f"Built {len(pairs)} original pairs with complete 256-prompt statistics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    build(parser.parse_args().destination)
