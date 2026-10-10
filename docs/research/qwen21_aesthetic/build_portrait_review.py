"""Publishable portrait diagnostic page; requires all 144 verified outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_portrait_probe import PROBE, prepare, verify
from PIL import Image


def build(destination):
    jobs = prepare()
    verify(jobs)
    prompts = [json.loads(line) for line in (PROBE / "prompts.jsonl").read_text().splitlines()]
    rows = [
        json.loads(line) for line in (PROBE / "scores.verified.jsonl").read_text().splitlines()
    ]
    cells = {(r["prompt_index"], r["sample_index"], r["checkpoint_label"]): r for r in rows}
    out = destination / "portraits"
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for p, prompt in enumerate(prompts):
        for seed_index in range(4):
            item = dict(
                prompt=prompt["prompt"],
                style=prompt["style"],
                subject="Alice" if p < 6 else "Portrait",
                seed=prompt["seeds"][seed_index],
                images=[],
            )
            for arm in ("base", "lora43", "lora60"):
                row = cells[p, seed_index, arm]
                source = Path(row["image_path"])
                assert hashlib.sha256(source.read_bytes()).hexdigest() == row["image_sha256"]
                name = f"{p:02d}_{seed_index}_{arm}.webp"
                with Image.open(source) as image:
                    image.save(out / name, format="WEBP", quality=95, method=4)
                item["images"].append(
                    dict(
                        label=arm,
                        src=f"portraits/{name}",
                        reward=row["r_aesthetic"],
                        original_sha256=row["image_sha256"],
                    )
                )
            items.append(item)
    data = dict(
        items=items,
        summary=json.loads((PROBE / "paired_summary.json").read_text()),
        display="1024px WebP quality95; scores computed on original PNG",
    )
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False))
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alice 与多风格肖像 · Base / 43 / 60</title><style>
body{margin:0;background:#f4f5f1;color:#192b2b;font:16px/1.6 system-ui}main{max-width:1650px;margin:auto;padding:24px}section{background:white;border:1px solid #dce3de;border-radius:12px;padding:20px;margin:20px 0}h1{line-height:1.2}a{color:#126254}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}img{width:100%;display:block}button,select{font:inherit;padding:8px;margin:5px;max-width:100%}#prompt{overflow-wrap:anywhere}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:12px}}
</style><main><a href="index.html">← 固定 64 条验证集</a><h1>Alice 与不同风格的肖像</h1>
<p>Base / 第 43 轮 / 第 60 轮。12 条预先固定的提示词，每条 4 个相同种子，全部 144 张均展示。原生采样，1024 × 1024，40 步。</p>
<section><h2>先比较图片，再看分数</h2><p>观察人物结构、面部与手部、细节、构图和风格忠实度；风格改变或颜色更浓不自动等于质量改善。网页使用 WebP quality95 有损压缩，评分来自原始 PNG。</p>
<button id="prev">← 上一组</button><select id="sel" aria-label="风格和种子"></select><button id="next">下一组 →</button><label><input id="blind" type="checkbox" checked>隐藏模型标签与分数</label><p id="prompt"></p><p id="seed"></p><div id="images" class="grid"></div></section>
<section><h2>这批肖像的美学评分</h2><p>这是一组针对性的风格诊断，只有两个主体、六种风格，不能代表通用生成能力。先在每条提示词内平均四个种子的差值，再按 12 条提示词 bootstrap；区间没有多重比较校正，也不代表人工偏好。</p><div class="scroll"><table><thead><tr><th>对比</th><th>平均差值</th><th>95% 区间</th></tr></thead><tbody id="stats"></tbody></table></div><a href="portraits/data.json">下载全部分数和图片索引</a></section></main><script>
fetch('portraits/data.json').then(r=>{if(!r.ok)throw Error('数据加载失败');return r.json()}).then(data=>{
let i=0;const names={base:'Base',lora43:'第 43 轮',lora60:'第 60 轮'},sel=document.getElementById('sel'),blind=document.getElementById('blind');
data.items.forEach((p,j)=>{const o=document.createElement('option');o.value=j;o.textContent=(j+1)+'/48 · '+p.subject+' · '+p.style+' · seed '+p.seed;sel.append(o)});
function render(){const p=data.items[i];sel.value=i;document.getElementById('prompt').textContent=p.prompt;document.getElementById('seed').textContent='Seed '+p.seed;const grid=document.getElementById('images');grid.replaceChildren();const orders=[[0,2,1],[2,1,0],[1,0,2],[2,0,1],[0,1,2],[1,2,0]];const order=blind.checked?orders[i%6]:[0,1,2];order.forEach((k,j)=>{const x=p.images[k],f=document.createElement('figure'),c=document.createElement('figcaption'),a=document.createElement('a'),im=document.createElement('img');c.textContent=blind.checked?'图片 '+String.fromCharCode(65+j):names[x.label]+' · 美学 '+x.reward.toFixed(3);a.href=x.src;a.target='_blank';a.rel='noopener';im.src=x.src;im.alt=c.textContent+' — '+p.prompt;a.append(im);f.append(c,a);grid.append(f)})}
sel.onchange=()=>{i=+sel.value;render()};blind.onchange=render;document.getElementById('prev').onclick=()=>{i=(i+47)%48;render()};document.getElementById('next').onclick=()=>{i=(i+1)%48;render()};
for(const [key,value] of Object.entries(data.summary.comparisons)){const tr=document.createElement('tr');for(const text of [key,value.mean_delta.toFixed(5),'['+value.prompt_bootstrap_ci95.map(x=>x.toFixed(5)).join(', ')+']']){const td=document.createElement('td');td.textContent=text;tr.append(td)}document.getElementById('stats').append(tr)}render();
}).catch(e=>{document.getElementById('prompt').textContent=e.message});</script></html>"""
    (destination / "portraits.html").write_text(page)
    print(f"Built {len(items)} paired groups, {len(rows)} images in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    build(parser.parse_args().destination)
