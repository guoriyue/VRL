"""Build a checkpoint-series page: one strip per gallery prompt, one column per checkpoint, GPT reference last.

usage: build_series_page.py out_dir title eyebrow lede col=label=/path/with/{s}/images/subdir ...
  e.g. base=base=/mnt/nvme/eval/visual_v11ck20_s{s}/images/base v13ck10="v13 ck10"=/mnt/nvme/eval/visual_v13ck10_s{s}/images/v13ck10
Prompts come from /mnt/nvme/aes2/visual_prompts_{0,1,2}.jsonl, interleaved across shards (s0[0], s1[0], s2[0], s0[1], ...).
"""

import html
import json
import sys
from pathlib import Path

from PIL import Image

TILE = 340


def main():
    out = Path(sys.argv[1])
    title, eyebrow, lede = sys.argv[2:5]
    cols = []
    for spec in sys.argv[5:]:
        key, label, pat = spec.split("=", 2)
        cols.append((key, label, pat))
    (out / "strips").mkdir(parents=True, exist_ok=True)
    shards = [
        [json.loads(l) for l in open(f"/mnt/nvme/aes2/visual_prompts_{s}.jsonl")] for s in range(3)
    ]
    order = [(s, i) for i in range(max(map(len, shards))) for s in range(3) if i < len(shards[s])]
    meta = []
    for n, (s, i) in enumerate(order):
        row = shards[s][i]
        paths = [Path(pat.format(s=s)) / f"prompt{i:04d}_sample00.png" for _, _, pat in cols]
        paths.append(Path(row["metadata"]["judge_reference"]))
        strip = Image.new("RGB", (TILE * len(paths) + 4 * (len(paths) - 1), TILE), (255, 255, 255))
        for k, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
                strip.paste(im, (k * (TILE + 4), 0))
        strip.save(out / "strips" / f"s{n:02d}.jpg", quality=88)
        meta.append({"n": n, "prompt": row["prompt"]})
    labels = [label for _, label, _ in cols] + ["GPT"]
    ncol = len(labels)
    page = f"""<title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
:root{{--bg:#f3f5f7;--panel:#fff;--ink:#17202a;--ink2:#51606f;--mute:#8b96a3;--line:#dde3e9;--acc:#0f6e8c}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0f141a;--panel:#161d25;--ink:#e6ebf0;--ink2:#aab6c2;--mute:#76838f;--line:#2a3440;--acc:#4fb3d1;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0f141a;--panel:#161d25;--ink:#e6ebf0;--ink2:#aab6c2;--mute:#76838f;--line:#2a3440;--acc:#4fb3d1;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.55;margin:0}}
.wrap{{max-width:1600px;margin:0 auto;padding-block:32px 64px;padding-inline:16px}}
h1{{font-family:"Sora",system-ui,sans-serif;font-size:clamp(22px,4vw,32px);margin:0;text-wrap:balance}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--acc)}}
.lede{{color:var(--ink2);max-width:72ch}}
.labels{{display:grid;grid-template-columns:repeat({ncol},minmax(0,1fr));gap:6px;font-size:11px;color:var(--mute);font-family:'IBM Plex Mono',monospace;margin-top:4px;min-width:{150 * ncol}px}}
.strip{{overflow-x:auto}}.strip img{{width:100%;min-width:{150 * ncol}px;display:block;border-radius:4px}}
.row{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:12px}}
.row .p{{font-size:14px;margin-bottom:8px}}.idx{{font-family:"IBM Plex Mono",monospace;color:var(--mute);margin-right:8px}}
</style>
<div class="wrap">
<div class="eyebrow">{html.escape(eyebrow)}</div>
<h1>{html.escape(title)}</h1>
<p class="lede">{html.escape(lede)}</p>
<div id="rows"></div>
</div>
<script>
const M = {json.dumps(meta, ensure_ascii=False)};
const L = {json.dumps(labels, ensure_ascii=False)};
document.getElementById('rows').innerHTML = M.map((m,i)=>`<div class="row"><div class="p"><span class="idx">${{String(i+1).padStart(2,'0')}}</span>${{m.prompt.replace(/</g,'&lt;')}}</div><div class="strip"><a href="strips/s${{String(m.n).padStart(2,'0')}}.jpg" target="_blank"><img loading="lazy" src="strips/s${{String(m.n).padStart(2,'0')}}.jpg" alt="checkpoint strip"></a><div class="labels">${{L.map(c=>`<span>${{c}}</span>`).join('')}}</div></div></div>`).join('');
</script>
"""
    (out / "index.html").write_text(page)
    print("rows", len(meta), "cols", labels)


main()
