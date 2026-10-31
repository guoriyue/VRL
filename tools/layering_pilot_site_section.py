"""Render measured pilot results for insertion into the existing private report."""

# Chinese report copy intentionally uses native punctuation.
# ruff: noqa: RUF001

import argparse
import datetime
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--decision", required=True)
    args = parser.parse_args()
    run = args.run
    comparison = json.loads((run / "comparison.json").read_text())
    history = json.loads((run / "historical_comparison.json").read_text())
    removal = json.loads((run / "removal_comparison.json").read_text())
    stress = json.loads((run / "stress_comparison.json").read_text())
    weights = json.loads((run / "weight_changes.json").read_text())
    assert len(stress["rows"]) == 16
    assert all(len(a["cases"]) == 4 for a in removal["arms"].values())
    arms = {**history["arms"], **comparison["arms"]}
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    markup = [
        f"""<section id="pilot"><div class="section-head"><div><div class="eyebrow">00 / COMPLETED PILOT · {now}</div><h2>堵住了两个漏洞，试训仍未通过。</h2></div></div>
<p class="section-intro">{html.escape(args.decision)}</p>
<p>从 ckpt-28 完成两步纯 GRPO：4 个 prompt × 4 个样本，lr 5e-5、Adam eps 1e-12、KL 0.04。评测使用真实权重、同一 seed 4401 和 10 步采样。以下对照覆盖同一批 15 张 held-out COCO 照片中的 60 个已标注物体。</p>
<div class="panel table-wrap"><table><thead><tr><th>模型</th><th>完整且纯净物体 / 60</th><th>背景遗漏</th><th>混合多个物体的层</th></tr></thead><tbody>"""
    ]
    for key, label in [
        ("v3b_20", "v3b-20"),
        ("baseline_28", "ckpt-28 基线"),
        ("v3e_44", "历史 step 44"),
        ("step_30", "integrity v2 · step 30"),
    ]:
        rows = [r for r in arms[key]["rows"] if r["id"].startswith("coco")]
        means = arms[key]["groups"]["coco"]["means"]
        complete = sum(r["complete_pure_objects"] for r in rows)
        merges = sum(r["merge_layers"] for r in rows)
        markup.append(
            f"<tr><td>{label}</td><td>{complete}</td><td>{means['object_background_share']:.2%}</td><td>{merges}</td></tr>"
        )
    markup.append("""</tbody></table></div><p class="note">完整且纯净：匹配层覆盖该物体 ≥90%，且该层标注像素中该物体占比 ≥90%。背景遗漏是逐图平均的物体可见区域占比。这些指标不覆盖未标注物体，也不证明隐藏区域语义正确。</p>
<div class="audit-grid"><article class="panel"><h3>奖励修复已验证</h3><p>旧试验完整层评分可被“复制混合层并藏在后面”利用：integrity 从 0.677 升到 0.750。首轮因此中止，未完成优化器更新。</p><p>v2 将完整层匹配与可见物体区域绑定。86 个真实生成层 / 真值层的隐藏复制攻击均降分；20 个真值场景的隐藏复制与背景残留对照通过。17 项相关测试通过。空层允许。</p><p class="note">完整层监督仅用于 1200 个渲染训练场景；其余 4239 行不启用此项。没有完整参考时，真实图片的隐藏重复仍是验证难点。</p></article>
<article class="panel"><h3>预先设定的模型检查</h3><ul>""")
    labels = {
        "more_complete_pure_photo_objects": "完整且纯净物体至少增加 2 个",
        "photo_background_not_worse_by_more_than_one_point": "背景遗漏增加不超过 1 个百分点",
        "no_more_photo_merge_layers": "混合物体的层数不增加",
        "rendered_edit_quality_not_worse": "渲染编辑质量不明显下降",
        "rendered_nonredundancy_not_worse": "渲染非冗余指标不明显下降",
    }
    for key, passed in comparison["promotion_gates"]["step_30"].items():
        markup.append(f"<li>{'通过' if passed else '未通过'}：{labels[key]}</li>")
    delta = weights["comparisons"][0]
    markup.append(f"""</ul><p>真实权重变化：{delta["changed_live_tensors"]} / {delta["live_tensor_count"]} 个张量，L2 相对变化 {delta["relative_delta_l2"]:.3%}。</p><p class="note">单 seed、小样本筛查。即使通过，也需要逐层复核与另一 seed 确认。增加监督奖励还会改变 global_std 下的 advantage 尺度，不能把结果仅归因于一个评分项。</p></article></div>
<h3>逐层查看变化最大的案例</h3><p class="note">按完整物体数变化选取排序两端的照片，再选渲染编辑质量变化最低的场景。它们用于找问题，不代表整体分布；层编号也不是物体身份。</p><a href="reward/pilot_review.png"><img src="reward/pilot_review.png" alt="基线与新 checkpoint 的源图和完整 RGBA 分层对照" style="width:100%;height:auto"></a>
<h3>独立移除物体</h3><p>4 个外部参考场景：既检查全部目标一起移除，也检查逐个移除时其他物体是否保留。参考图包含生成或修复内容；这是像素诊断，不是真实遮挡内容的绝对答案。</p><div class="panel table-wrap"><table><thead><tr><th>模型</th><th>全部目标移除质量</th><th>单物体移除质量</th></tr></thead><tbody>""")
    for key, value in removal["arms"].items():
        markup.append(
            f"<tr><td>{html.escape(key)}</td><td>{value['mean_removal_quality']:.4f}</td><td>{value['mean_single_object_removal_quality']:.4f}</td></tr>"
        )
    markup.append(
        """</tbody></table></div><h3>增加 4 层后的完整物体数</h3><p class="note">4 个固定诊断场景，其中 Luma 是训练参考图。同一 seed 在不同层数下不代表相同噪声张量。完整结果包含混层、背景残留和隐藏层检查。</p><div class="panel table-wrap"><table><thead><tr><th>场景</th><th>ckpt-28 原层数 → +4</th><th>v2 原层数 → +4</th></tr></thead><tbody>"""
    )
    rows = stress["rows"]
    for key in dict.fromkeys(r["id"] for r in rows):
        counts = {}
        for arm in ["baseline_28", "integrity_v2_30"]:
            counts[arm] = [
                next(
                    r["complete_pure_objects"]
                    for r in rows
                    if r["id"] == key and r["arm"] == arm and r["extra_layers"] == extra
                )
                for extra in [0, 4]
            ]
        markup.append(
            f"<tr><td>{html.escape(key)}</td><td>{counts['baseline_28'][0]} → {counts['baseline_28'][1]}</td><td>{counts['integrity_v2_30'][0]} → {counts['integrity_v2_30'][1]}</td></tr>"
        )
    markup.append(
        """</tbody></table></div><p><a href="reward/pilot_comparison.json" download>下载逐图主评测</a> · <a href="reward/removal_comparison.json" download>物体移除评测</a> · <a href="reward/stress_comparison.json" download>层数压力测试</a></p><p class="note">下方保留前一轮可见区域奖励修复和 72 组历史输出复算；它们不是本轮试训结果。</p></section>"""
    )
    (run / "site_pilot_section.html").write_text("\n".join(markup))
    print("Rendered measured pilot section")


if __name__ == "__main__":
    main()
