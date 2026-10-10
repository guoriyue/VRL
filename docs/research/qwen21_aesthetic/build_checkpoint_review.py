"""Fixed64 native comparison with verified PNG sources and compact WebP previews."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from vrl.scripts.eval.score_report import write_curve_report

ROOT = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora/window2_trial")
DEST = Path("/mnt/nvme/work/qwen21-aesthetic-review-site/dist")


def build(steps=(15, 43, 47), destination=DEST, quality=95):
    steps = list(steps)
    assert len(set(steps)) == len(steps) and 43 in steps
    destination = Path(destination)
    out = destination / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)

    def pick_source(step):
        return ROOT / f"pickscore_15_{47 if step == 15 else step}"

    arms = {}
    stats = []
    for step in steps:
        summary = json.loads((ROOT / f"eval_step{step}/aggregate/summary.json").read_text())
        ps = json.loads((pick_source(step) / "vs_base/summary.json").read_text())
        for g in range(4):
            for row in map(
                json.loads,
                (ROOT / f"eval_step{step}/gpu{g}/report/scores.jsonl").read_text().splitlines(),
            ):
                key = (row["checkpoint_label"], g * 16 + row["prompt_index"])
                if key in arms:
                    assert arms[key]["image_sha256"] == row["image_sha256"]
                arms[key] = row
        label = f"lora{step}"
        stats.append(
            dict(
                step=step,
                aesthetic=summary["arms"][label]["scores"]["aesthetic"],
                pickscore=ps["arms"][label]["scores"]["pickscore"],
            )
        )
    assert len(arms) == 64 * (len(steps) + 1)
    comparisons = []
    for ref in [s for s in steps if s != 43]:
        for metric in ["aesthetic", "pickscore"]:
            selected = []
            if metric == "aesthetic":
                for (label, idx), v in arms.items():
                    if label in [f"lora{ref}", "lora43"]:
                        v = dict(v)
                        v["prompt_index"] = idx
                        selected.append(v)
            else:
                for step in [ref, 43]:
                    source = pick_source(step) / "raw_scores.jsonl"
                    selected.extend(
                        v
                        for v in map(json.loads, source.read_text().splitlines())
                        if v["checkpoint_label"] == f"lora{step}"
                    )
            report = write_curve_report(
                selected,
                out / f"{metric}_43_vs_{ref}",
                base_label=f"lora{ref}",
                bootstrap_resamples=10000,
                seed=260925,
            )
            comparisons.append(
                dict(
                    reference=ref,
                    metric=metric,
                    delta=report["arms"]["lora43"]["scores"][metric]["paired_delta_from_base"],
                )
            )
    items = []
    for i in range(64):
        base = arms[("base", i)]
        item = dict(index=i, prompt=base["prompt"], seed=base["seed"], images=[])
        for label in ["base"] + [f"lora{s}" for s in steps]:
            row = arms[(label, i)]
            assert (row["prompt"], row["seed"]) == (base["prompt"], base["seed"])
            path = Path(row["image_path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"]
            name = f"{i:03d}_{label}.webp"
            with Image.open(path) as img:
                assert img.size == (1024, 1024)
                img.save(out / name, format="WEBP", quality=quality, method=4)
                with Image.open(out / name) as saved:
                    assert saved.size == img.size
            item["images"].append(
                dict(label=label, src="checkpoints/" + name, reward=row["r_aesthetic"])
            )
        items.append(item)
    (out / "data.json").write_text(
        json.dumps(
            dict(
                stats=stats,
                items=items,
                comparisons=comparisons,
                display_encoding=dict(
                    format="WebP", quality=quality, lossless=False, scores_from="original PNG"
                ),
            ),
            ensure_ascii=False,
        )
    )
    shutil.copyfile(
        "/mnt/nvme/work/qwen_aesthetic_figures/training_and_validation.png", out / "curves.png"
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen 美学 · 15 / 43 / 47 轮对比</title><style>
body{margin:0;background:#f4f5f1;color:#192b2b;font:16px/1.6 system-ui}main{max-width:1650px;margin:auto;padding:28px}h1{font-size:clamp(25px,4vw,44px);line-height:1.2}section{background:white;border:1px solid #dce3de;border-radius:16px;padding:22px;margin:22px 0}.muted{color:#526762}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;display:block;border-radius:8px}figcaption{padding:10px 0;font-weight:650}a{color:#126254}button,select{font:inherit;padding:9px;border:1px solid #adc3bb;border-radius:7px;background:white;max-width:100%;margin:4px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3de}.scroll{overflow:auto}.badge{display:inline-block;background:#e1eee7;padding:5px 12px;border-radius:30px}.curve{max-width:1000px;margin:auto}#prompt{overflow-wrap:anywhere}@media(max-width:1000px){.grid{grid-template-columns:1fr 1fr}}@media(max-width:550px){main{padding:12px}section{padding:12px}.grid{grid-template-columns:1fr}}
</style><main><a href="index.html">← 实验首页</a><p class="badge">固定 64 条 · 原生采样 · 已完成评测快照</p><h1>奖励最高的第 43 轮，画质也更好吗？</h1><p>对比 Base 与第 15、43、47 轮。第 43 轮训练奖励最高（6.2503）；不同轮次训练提示词不同，不能直接据此选出画质最佳模型。</p><section><h2>固定验证集上的结果</h2><p class="muted">全部 64 对，1024 × 1024、40 步、相同提示词和种子。括号是 95% 配对 bootstrap 区间。分数胜率不是人工偏好胜率；这是反复查看的验证集，不是全新测试。</p><div class="scroll"><table><thead><tr><th>Checkpoint</th><th>美学均分</th><th>美学差值 vs Base [95% CI]</th><th>PickScore 差值 vs Base [95% CI]</th></tr></thead><tbody id="stats"></tbody></table></div><p id="finding"></p><h3>第 43 轮相对其他 checkpoint</h3><div class="scroll"><table><thead><tr><th>对比</th><th>指标</th><th>配对差值 [95% CI]</th></tr></thead><tbody id="direct"></tbody></table></div><a href="checkpoints/data.json">下载统计与原图索引</a></section><section><h2>同提示词、同种子的原图</h2><p>全部 64 组均可查看，未按奖励挑选图片。点击图片打开 1024 原图。快捷入口仅按主题选择。</p><button data-i="13">动漫肖像</button><button data-i="26">写实肖像</button><button data-i="43">油画肖像</button><button data-i="37">Moto-Moto / furry</button><br><button id="prev">← 上一组</button><select id="sel" aria-label="提示词"></select><button id="next">下一组 →</button><p id="prompt"></p><p class="muted" id="seed"></p><div id="images" class="grid"></div></section><section><h2>训练奖励与已评测 checkpoint</h2><p>训练图覆盖第 1–47 轮；下方误差条为第 5、15、43、47 轮评测，全部使用同一固定验证集。最低 loss 不代表最佳画质。</p><a href="checkpoints/curves.png" target="_blank"><img class="curve" src="checkpoints/curves.png" alt="训练奖励、loss 与固定验证集配对置信区间"></a></section></main><script>
const labels={base:'Base',lora15:'第 15 轮',lora43:'第 43 轮 · 训练奖励最高',lora47:'第 47 轮'};
fetch('checkpoints/data.json').then(r=>{if(!r.ok)throw Error('数据加载失败');return r.json()}).then(data=>{let i=13;const sel=document.getElementById('sel');const fmt=x=>(x>=0?'+':'')+x.toFixed(5);const delta=s=>{const d=s.paired_delta_from_base;return fmt(d.mean)+' ['+d.bootstrap_95ci.map(fmt).join(', ')+']'};
for(const s of data.stats){const tr=document.createElement('tr');for(const v of [s.step,s.aesthetic.absolute.mean.toFixed(4),delta(s.aesthetic),delta(s.pickscore)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}document.getElementById('stats').append(tr)}
for(const c of data.comparisons){const tr=document.createElement('tr');for(const v of ['43 − '+c.reference,c.metric,delta({paired_delta_from_base:c.delta})]){const td=document.createElement('td');td.textContent=v;tr.append(td)}document.getElementById('direct').append(tr)}
const best=data.stats.reduce((a,b)=>a.aesthetic.absolute.mean>b.aesthetic.absolute.mean?a:b);document.getElementById('finding').textContent='已评测 15/43/47 中，美学均分数值最高：第 '+best.step+' 轮。数值排名不等于差异显著，请结合区间与原图判断。';
data.items.forEach((p,j)=>{const o=document.createElement('option');o.value=j;o.textContent=(j+1)+'/64 · '+p.prompt.slice(0,65);sel.append(o)});
function render(){const p=data.items[i];sel.value=i;document.getElementById('prompt').textContent=p.prompt;document.getElementById('seed').textContent='Seed '+p.seed;const grid=document.getElementById('images');grid.replaceChildren();for(const x of p.images){const f=document.createElement('figure'),c=document.createElement('figcaption'),a=document.createElement('a'),im=document.createElement('img');c.textContent=labels[x.label]+' · 美学 '+x.reward.toFixed(3);a.href=x.src;a.target='_blank';a.rel='noopener';im.src=x.src;im.alt=labels[x.label]+' — '+p.prompt;a.append(im);f.append(c,a);grid.append(f)}}
sel.onchange=()=>{i=+sel.value;render()};document.getElementById('prev').onclick=()=>{i=(i+63)%64;render()};document.getElementById('next').onclick=()=>{i=(i+1)%64;render()};document.querySelectorAll('[data-i]').forEach(b=>b.onclick=()=>{i=+b.dataset.i;render()});render();}).catch(e=>{document.getElementById('finding').textContent=e.message});
</script></html>"""
    names = "/".join(map(str, steps))
    labels = {"base": "Base", **{f"lora{s}": f"第 {s} 轮" for s in steps}}
    start = page.index("const labels=")
    end = page.index(";", start)
    page = page[:start] + "const labels=" + json.dumps(labels, ensure_ascii=False) + page[end:]
    page = page.replace("15 / 43 / 47", names).replace("15/43/47", names)
    page = page.replace(
        "对比 Base 与第 15、43、47 轮。", "对比 Base 与第 " + "、".join(map(str, steps)) + " 轮。"
    )
    page = page.replace(
        "第 43 轮训练奖励最高（6.2503）", "前 47 轮中，第 43 轮训练奖励最高（6.2503）"
    )
    page = page.replace("repeat(4,minmax(0,1fr))", f"repeat({len(steps) + 1},minmax(0,1fr))")
    page = page.replace("同提示词、同种子的原图", "同提示词、同种子的图片")
    page = page.replace(
        "点击图片打开 1024 原图。",
        f"点击图片打开 1024 图片；网页使用 WebP quality {quality} 有损压缩，评分来自原始 PNG。",
    )
    page = page.replace("下载统计与原图索引", "下载统计与图片索引")
    (destination / "checkpoints.html").write_text(page)
    print(
        f"Built 64 groups, {64 * (len(steps) + 1)} images; source hashes verified; display WebP quality {quality}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[15, 43, 47])
    parser.add_argument("--destination", type=Path, default=DEST)
    parser.add_argument("--quality", type=int, choices=range(1, 101), default=95)
    args = parser.parse_args()
    build(args.steps, args.destination, args.quality)
