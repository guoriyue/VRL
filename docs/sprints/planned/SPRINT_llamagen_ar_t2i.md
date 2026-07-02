# SPRINT: 接入 LlamaGen AR T2I 家族（10 个中优先级最低，可降级）

状态：planned（2026-07-01）。性质：新增离散 AR T2I 家族——学术基线价值 > 生产价值，排最后。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 10 项。

## 0. 一句话 + 诚实定位

**LlamaGen 是"LLaMA 架构能不能直接做图像 AR"的学术基线（VQ tokenizer + vanilla next-token）。
它没有 HF transformers 原生类——代码在 FoundationVision 的 GitHub 仓库。** 接它的价值是凑齐
"离散 AR 谱系基线"（janus/emu3/llamagen 三点对照）；若 vendor 成本高于价值，**允许降级为
`*_probe` 一次性验证或直接关闭**（对齐 AGENTS.md 一次性验证 vs 长期资产的边界）。

## 1. 模型事实（落地第一步核对权重与代码形态）

- checkpoint：`FoundationVision/LlamaGen`（HF 上有权重集合；T2I 变体 ~775M XL，class-cond 到 3B）。
- **无 diffusers/transformers 原生类**——官方 PyTorch 仓库代码。第一动作：评估其 GPT 模型定义能否
  用 transformers `LlamaForCausalLM` + 自定义 embedding 复现（架构就是 LLaMA），能则**零 vendor**；
  不能则 vendor 最小模型文件（单文件，进 `vrl/models/ar/llamagen/upstream.py` 并写明来源 commit）。
- 视觉 tokenizer：VQGAN（官方 repo 自带，codebook 16384）——同样评估最小化引入。
- 文本编码器：T2I 变体用 FLAN-T5-XL 出条件 embeds（prefix conditioning）。
- 生成：categorical next-token + CFG → `ar_discrete` seam，janus 模板。

## 2. 技术点

1. **零 vendor 优先**：LLaMA 架构 → 尽量 `LlamaForCausalLM.from_pretrained` + 权重 key 映射，
   只有 VQ tokenizer 可能必须带最小实现。
2. decode loop / runner / capability 全套照 [[SPRINT_emu3_ar_t2i]]（先落 emu3 再做这里，增量最小）。
3. 775M 是 10 个里最小的模型——适合当 AR seam 的快速回归测试家族（CI 友好）。

## 3. 落地文件

同 [[SPRINT_emu3_ar_t2i]] §3 形状：`vrl/models/ar/llamagen/` 四件（+可能的 `upstream.py`）+
registry + `configs/model/ar/llamagen/t2i_xl.yaml` + tests。

## 4. 验收

- CPU 结构绿；GPU parity 对照官方 repo 推理脚本（同 seed 出图 + logprob 合理性）。
- 短 GRPO 曲线；另一个隐性验收：**它够小，能进常规回归**——加一个 tiny-config 的 CI 级测试。

## 5. 非目标

- 不接 class-conditional 变体（只要 T2I）；不追官方 repo 的 vLLM serving 集成。
- vendor 超过"一个模型文件 + 一个 tokenizer 文件"即触发降级讨论（§0）。

## 参考

- https://github.com/FoundationVision/LlamaGen ；https://huggingface.co/FoundationVision/LlamaGen
- 模板：[[SPRINT_emu3_ar_t2i]]、`vrl/models/ar/janus_pro/`
