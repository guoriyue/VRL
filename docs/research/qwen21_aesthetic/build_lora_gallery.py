"""Build the review site's native base/LoRA gallery from completed evaluations."""

# Chinese UI copy intentionally uses full-width punctuation.
# ruff: noqa: RUF001

import hashlib
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora")
SITE = Path("/mnt/nvme/work/qwen21-aesthetic-review-site/dist")


def build():
    runs = []
    for epoch in (2, 5, 10, 20):
        directory = ROOT / f"eval_step{epoch}"
        if not (directory / "aggregate" / "summary.json").is_file():
            continue
        pairs = []
        destination = SITE / "lora" / str(epoch)
        destination.mkdir(parents=True, exist_ok=True)
        for gpu in range(4):
            report = directory / f"gpu{gpu}" / "report"
            if not (report / "evaluation_complete.json").is_file():
                raise ValueError(f"Incomplete evaluation: {report}")
            grouped = {}
            for line in (report / "scores.jsonl").read_text().splitlines():
                row = json.loads(line)
                grouped.setdefault(row["prompt_index"], []).append(row)
            for index, rows in sorted(grouped.items()):
                if len(rows) != 2 or {r["checkpoint_label"] for r in rows} != {
                    "base",
                    f"lora{epoch}",
                }:
                    raise ValueError("Expected one native base/LoRA pair per prompt")
                number = gpu * 16 + index
                rows.sort(key=lambda row: row["checkpoint_label"])
                random.Random(f"gallery:260926:{epoch}:{number}").shuffle(rows)
                pair = {
                    "id": f"{epoch}:{number}",
                    "number": number,
                    "prompt": rows[0]["prompt"],
                    "seed": rows[0]["seed"],
                    "images": [],
                }
                for label, row in zip(("A", "B"), rows, strict=True):
                    source = Path(row["image_path"])
                    name = f"{number:04d}_{label}.png"
                    shutil.copyfile(source, destination / name)
                    pair["images"].append(
                        {
                            "src": f"lora/{epoch}/{name}",
                            "arm": row["checkpoint_label"],
                            "score": row["r_aesthetic"],
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        }
                    )
                pairs.append(pair)
        summary = json.loads((directory / "aggregate" / "summary.json").read_text())
        runs.append({"epoch": epoch, "pairs": pairs, "summary": summary})
    if not runs:
        raise RuntimeError("No completed native evaluations; refusing an empty gallery")
    (SITE / "lora_results.json").write_text(
        json.dumps(
            {
                "built_at": datetime.now(UTC).isoformat(),
                "runs": runs,
                "note": "Native 1024px/40-step paired validation; training-reward scores are not independent quality proof.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (SITE / "lora.html").write_text(HTML)
    index = SITE / "index.html"
    old = index.read_text()
    if 'href="lora.html"' not in old:
        old = old.replace(
            "<nav>",
            '<nav><a href="lora.html"><strong>LoRA training: native A/B results</strong></a>',
            1,
        )
        old = old.replace(
            "No RL was launched.",
            "No RL was launched for this historical qualification. The later exploratory LoRA experiment is shown on the new results page.",
        )
        index.write_text(old)
    print(json.dumps({"epochs": [r["epoch"] for r in runs], "page": str(SITE / "lora.html")}))


HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen 2.1 · LoRA 原生对比</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f4f0;color:#202723;font:16px/1.6 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}h1{font-size:clamp(28px,4vw,48px);line-height:1.15;margin-bottom:16px}
nav{display:flex;gap:20px;flex-wrap:wrap}a{color:#226151}small,.muted{color:#606861}
.tools{position:sticky;top:0;background:#f5f4f0f2;padding:16px 0;z-index:2;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
button,select{font:inherit;border:1px solid #b8c6bd;background:white;padding:9px 15px;border-radius:8px;cursor:pointer}
button.selected{background:#245d4b;color:white}.pair{background:white;padding:20px;border:1px solid #dbe2dc;border-radius:14px;margin:20px 0}
.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;min-width:0}img{width:100%;height:auto;display:block;border-radius:6px}figcaption{padding:8px 0}.votes{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
#stats{padding:14px;background:#e7eee8;border-radius:10px}.hidden{display:none}@media(max-width:650px){main{padding:14px}.pair{padding:12px}.images{grid-template-columns:1fr}}
</style><main><nav><a href="clean_sampling.html">Base 采样参考</a><a href="index.html">历史奖励资格实验</a></nav>
<h1>Qwen-Image-2.1<br>LoRA 美学 RL · 原生采样对比</h1>
<p>相同提示词、相同种子，1024 × 1024，40 步原生确定性采样。两张中一张是 base，一张是训练后的 LoRA；A/B 顺序随机。点击图片查看原图。</p>
<p class="muted">这是验证集上的探索性实验。奖励涨分不等于肉眼画质改善。请同时检查美感、细节、提示词符合度和明显瑕疵。</p>
<div class="tools"><label>训练轮次 <select id="round"></select></label><button id="reveal">揭晓模型与奖励</button><button id="export">导出我的判断</button><span id="count"></span></div>
<div id="stats" class="hidden"></div><div id="pairs"></div><small id="stamp"></small>
<script>
let data, revealed=false;const storageKey='qwen21-lora-review-v1';let votes={};try{votes=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
const el=(tag,text)=>{let n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n};
function render(){const run=data.runs.find(r=>r.epoch===Number(document.querySelector('#round').value));const target=document.querySelector('#pairs');target.replaceChildren();
document.querySelector('#count').textContent=run.pairs.length+' 组固定提示词';
let stats=document.querySelector('#stats');stats.classList.toggle('hidden',!revealed);let d=run.summary.arms['lora'+run.epoch].scores.aesthetic.paired_delta_from_base;
stats.textContent='美学奖励配对差：'+d.mean.toFixed(4)+'；95% CI ['+d.bootstrap_95ci.map(v=>v.toFixed(4)).join(', ')+']。该评分与训练奖励相同，不是独立画质验证。';
for(const pair of run.pairs){let card=el('section');card.className='pair';card.append(el('h2','提示词 '+pair.number),el('p',pair.prompt),el('small','Seed '+pair.seed));let grid=el('div');grid.className='images';
pair.images.forEach((im,i)=>{let fig=el('figure'),a=el('a'),img=el('img');a.href=im.src;a.target='_blank';a.rel='noopener';img.src=im.src;img.loading='lazy';img.alt='提示词 '+pair.number+'，图 '+('AB'[i]);a.append(img);fig.append(a,el('figcaption',('AB'[i])+(revealed?' · '+im.arm+' · 奖励 '+Number(im.score).toFixed(3):'')));grid.append(fig)});card.append(grid);
for(const [key,title,options] of [['appeal','美感',['A','B','相当']],['adherence','提示词符合度',['A','B','相当']],['artifacts','明显瑕疵',['A','B','都有','都无']]]){let line=el('div');line.className='votes';line.append(el('span',title+'：'));for(const choice of options){let b=el('button',choice);b.classList.toggle('selected',votes[pair.id]?.[key]?.choice===choice);b.onclick=()=>{votes[pair.id]={...votes[pair.id],[key]:{choice,revealed_when_rated:revealed,time:new Date().toISOString()}};localStorage.setItem(storageKey,JSON.stringify(votes));for(const x of line.querySelectorAll('button'))x.classList.toggle('selected',x===b)};line.append(b)}card.append(line)}target.append(card)}}
document.querySelector('#reveal').onclick=()=>{revealed=!revealed;document.querySelector('#reveal').textContent=revealed?'隐藏模型与奖励':'揭晓模型与奖励';render()};
document.querySelector('#export').onclick=()=>{let a=el('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({exported_at:new Date().toISOString(),votes},null,2)],{type:'application/json'}));a.download='qwen21-lora-my-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
fetch('lora_results.json').then(r=>{if(!r.ok)throw Error('无法加载结果');return r.json()}).then(d=>{data=d;let s=document.querySelector('#round');for(const run of d.runs){let o=el('option','第 '+run.epoch+' 轮');o.value=run.epoch;s.append(o)}s.value=d.runs.at(-1).epoch;s.onchange=render;document.querySelector('#stamp').textContent='结果快照生成于 '+d.built_at+'。判断仅保存在当前浏览器，导出后可分享。';render()}).catch(e=>{document.querySelector('#pairs').textContent=e.message});
</script></main></html>"""


if __name__ == "__main__":
    build()
