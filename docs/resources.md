# Resources: Models, Training Objectives, and References

Comprehensive reference for all encoder models and training paradigms supported by auto-mlm-pipes.

---

## Table of Contents

1. [Standard Encoder Models](#1-standard-encoder-models)
2. [Late-Interaction Models (ColBERT Family)](#2-late-interaction-models-colbert-family)
3. [Hybrid / Infilling Models (GLM Family)](#3-hybrid--infilling-models-glm-family)
4. [Sparse Retrieval Models (SPLADE Family)](#4-sparse-retrieval-models-splade-family)
5. [Masked Auto-Encoder Models (RetroMAE Family)](#5-masked-auto-encoder-models-retromae-family)
6. [Decoder-to-Encoder Conversion (LLM2Vec / MNTP)](#6-decoder-to-encoder-conversion-llm2vec--mntp)
7. [Training Objectives Reference](#7-training-objectives-reference)
8. [Frameworks and Libraries](#8-frameworks-and-libraries)
9. [Benchmarks and Evaluation](#9-benchmarks-and-evaluation)

---

## 1. Standard Encoder Models

Models compatible with `AutoModelForMaskedLM`. Trained via the core `train_encoder.py` pipeline.

### 1.1 ModernBERT

- **Paper**: Warner et al., "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference" (2024)
- **arXiv**: [2412.13663](https://arxiv.org/abs/2412.13663)
- **GitHub**: [AnswerDotAI/ModernBERT](https://github.com/AnswerDotAI/ModernBERT)
- **HuggingFace**: `answerdotai/ModernBERT-base` (149M), `answerdotai/ModernBERT-large` (395M)
- **Context**: 8,192 tokens
- **Architecture**:
  - Alternating local-global attention (every 3rd layer global, others 128-token sliding window)
  - Rotary Position Embeddings (RoPE): theta=160K global, theta=10K local
  - GeGLU activations
  - Flash Attention 2 (local) / Flash Attention 3 (global)
  - Unpadding for efficiency
- **Training**: 2T tokens in 3 phases (1.7T@1024 → 250B@8192 → 50B annealing)
- **Masking rate**: 30% (vs BERT's 15%)
- **Performance**: 14.5-30.9% faster (short), 98.8-118.8% faster (long) vs other encoders

### 1.2 NomicBERT

- **Paper**: Nussbaum et al., "Nomic Embed: Training a Reproducible Long Context Text Embedder" (2024)
- **arXiv**: [2402.01613](https://arxiv.org/abs/2402.01613)
- **GitHub**: [nomic-ai/contrastors](https://github.com/nomic-ai/contrastors)
- **HuggingFace**: `nomic-ai/nomic-bert-2048` (137M)
- **Context**: 8,192 tokens (trained at 2,048, extended)
- **Architecture**:
  - Rotary Position Embeddings (RoPE)
  - SwiGLU activations
  - Flash Attention 2 integration
- **Training**: 3-stage (MLM → contrastive pre-training on 235M pairs → contrastive fine-tuning)
- **Fully open**: code, data, weights, training logs

### 1.3 NeoBERT

- **Paper**: Gonçalves et al., "NeoBERT: A Next-Generation BERT" (2025)
- **arXiv**: [2502.19587](https://arxiv.org/abs/2502.19587)
- **HuggingFace**: Available as standard HF model
- **Parameters**: 250M
- **Context**: 4,096 tokens
- **Architecture**: Optimal depth-to-width ratio, plug-and-play BERT replacement
- **Training**: 2T+ tokens on RefinedWeb dataset
- **Masking rate**: 20%
- **Key claim**: Only MLM pre-training (no contrastive stage needed), outperforms BERT-large and NomicBERT on GLUE

### 1.4 DeBERTaV3

- **Paper**: He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing" (2023, ICLR)
- **arXiv**: [2111.09543](https://arxiv.org/abs/2111.09543)
- **GitHub**: [microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)
- **HuggingFace**: `microsoft/deberta-v3-xsmall` (22M), `microsoft/deberta-v3-small` (44M), `microsoft/deberta-v3-base` (86M), `microsoft/deberta-v3-large` (304M)
- **Context**: 512 tokens
- **Architecture**:
  - Disentangled attention: separate content and position representations
  - Enhanced mask decoder
  - ELECTRA-style Replaced Token Detection (RTD) pre-training
  - Gradient-Disentangled Embedding Sharing (GDES) for generator-discriminator stability
- **Key finding**: Surpasses ModernBERT in sample efficiency when trained on identical data

### 1.5 BERT

- **Paper**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019, NAACL)
- **arXiv**: [1810.04805](https://arxiv.org/abs/1810.04805)
- **HuggingFace**: `bert-base-uncased` (110M), `bert-large-uncased` (340M)
- **Context**: 512 tokens
- **Training objectives**: MLM (15% masking) + Next Sentence Prediction (NSP)
- **Foundation model** for all subsequent encoder architectures

### 1.6 RoBERTa

- **Paper**: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
- **arXiv**: [1907.11692](https://arxiv.org/abs/1907.11692)
- **HuggingFace**: `FacebookAI/roberta-base` (125M), `FacebookAI/roberta-large` (355M)
- **Context**: 512 tokens
- **Key improvements over BERT**:
  - Dynamic masking (different mask per epoch)
  - Removed NSP objective
  - Trained on 160GB text (10x BERT)
  - Larger batch sizes
- **Still the most downloaded encoder on HuggingFace** (2024+)

### 1.7 ALBERT

- **Paper**: Lan et al., "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (2020, ICLR)
- **arXiv**: [1909.11942](https://arxiv.org/abs/1909.11942)
- **HuggingFace**: `albert-base-v2` (12M), `albert-large-v2` (18M), `albert-xlarge-v2` (60M), `albert-xxlarge-v2` (235M)
- **Context**: 512 tokens
- **Key innovations**:
  - Factorized embedding parameterization (embedding size ≠ hidden size)
  - Cross-layer parameter sharing (dramatically reduces params)
  - Sentence Order Prediction (SOP) replaces NSP
- **Use case**: When model size / memory is critical

### 1.8 XLNet

- **Paper**: Yang et al., "XLNet: Generalized Autoregressive Pretraining for Language Understanding" (2019, NeurIPS)
- **arXiv**: [1906.08237](https://arxiv.org/abs/1906.08237)
- **HuggingFace**: `xlnet-base-cased` (110M), `xlnet-large-cased` (340M)
- **Context**: 512 tokens (with Transformer-XL recurrence for longer)
- **Architecture**:
  - Permutation language modeling: all factorization orders, not just left-to-right
  - Captures bidirectional context without [MASK] token (no pre-train/fine-tune discrepancy)
  - Transformer-XL backbone with segment recurrence
- **Key advantage**: Avoids BERT's independence assumption between masked tokens

### 1.9 ELECTRA

- **Paper**: Clark et al., "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (2020, ICLR)
- **arXiv**: [2003.10555](https://arxiv.org/abs/2003.10555)
- **GitHub**: [google-research/electra](https://github.com/google-research/electra)
- **HuggingFace**: `google/electra-small-discriminator` (14M), `google/electra-base-discriminator` (110M), `google/electra-large-discriminator` (335M)
- **Architecture**:
  - Generator: small MLM that proposes replacement tokens
  - Discriminator: main encoder classifies each token as original/replaced (RTD)
  - 3-7x more efficient per FLOP than MLM
  - Discriminator is the final model; generator is discarded
- **Mathematically equivalent** to Noise-Contrastive Estimation (NCE)
- **Performance**: Matches RoBERTa at 1/4 pre-training FLOPs

### 1.10 M2-BERT (Monarch Mixer)

- **Paper**: Fu et al., "Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture" (2023)
- **arXiv**: [2310.12109](https://arxiv.org/abs/2310.12109)
- **GitHub**: [HazyResearch/m2](https://github.com/HazyResearch/m2)
- **Parameters**: ~80M (25% fewer params/FLOPs than BERT-base)
- **Architecture**:
  - Replaces attention with Monarch matrices (sub-quadratic complexity)
  - Sequence mixer: DFT-based gated convolutions
  - Dimension mixer: learned block-diagonal matrices
- **Training**: Standard MLM, compatible with `AutoModelForMaskedLM`
- **Performance**: Outperforms Transformers by 23.3+ points on LoCo long-context benchmark despite 90x fewer params

### 1.11 FNet

- **Paper**: Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms" (2022, NAACL)
- **arXiv**: [2105.03824](https://arxiv.org/abs/2105.03824)
- **HuggingFace**: `google/fnet-base`, `google/fnet-large`
- **Architecture**: Replaces self-attention with Fast Fourier Transform for token mixing
- **Performance**: 92-97% of BERT accuracy on GLUE
- **Efficiency**: 80% faster on GPU, 70% faster on TPU at seq_len=512
- **Use case**: When inference speed is critical on limited hardware

### 1.12 UniLMv2

- **Paper**: Bao et al., "UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training" (2020, ICML)
- **arXiv**: [2002.12804](https://arxiv.org/abs/2002.12804)
- **GitHub**: [microsoft/unilm](https://github.com/microsoft/unilm)
- **Architecture**:
  - Pseudo-Masked Language Model (PMLM)
  - Combines conventional masks (autoencoding) + pseudo masks (partially autoregressive)
  - Single unified network (not encoder-decoder)
  - Context encodings reused to avoid redundant computation
- **Functions as**: Bidirectional encoder AND sequence-to-sequence decoder

### 1.13 ERNIE 3.0

- **Paper**: Sun et al., "ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation" (2021)
- **arXiv**: [2107.02137](https://arxiv.org/abs/2107.02137)
- **GitHub**: [PaddlePaddle/ERNIE](https://github.com/PaddlePaddle/ERNIE)
- **Architecture**: Unified framework fusing auto-regressive and auto-encoding networks
- **Training data**: 4TB corpus + large-scale knowledge graph
- **Key innovation**: Knowledge-integrated MLM (phrase masking, named entity masking)
- **Performance**: Surpassed human on SuperGLUE (90.6% vs 89.8% human)
- **Note**: PaddlePaddle framework (not PyTorch native)

---

## 2. Late-Interaction Models (ColBERT Family)

Trained via standalone `train_colbert.py` pipeline. These produce per-token multi-vector representations.

### 2.1 ColBERT

- **Paper**: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" (2020, SIGIR)
- **arXiv**: [2004.12832](https://arxiv.org/abs/2004.12832)
- **GitHub**: [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)
- **Architecture**:
  - Independent query/document encoding via BERT
  - Per-token 128-dim embeddings (via linear projection)
  - MaxSim operation: max(sim(q_token, d_token)) for each query token, then sum
  - Two orders of magnitude faster than cross-encoder BERT ranking
- **Training**: Contrastive with query-document pairs, in-batch negatives
- **Data format**: Triplets (query, positive_passage, negative_passage)

### 2.2 ColBERTv2

- **Paper**: Santhanam et al., "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction" (2022, NAACL)
- **arXiv**: [2112.01488](https://arxiv.org/abs/2112.01488)
- **Key improvements**:
  - Aggressive residual compression (6-10x smaller index: 154GB → 16-25GB on MS MARCO)
  - Denoised supervision: cross-encoder knowledge distillation via MiniLM
  - w-way tuple training (32-64 way)

### 2.3 ColBERT-Zero

- **Paper**: "ColBERT-Zero: To Pre-train Or Not To Pre-train ColBERT Models" (2025)
- **arXiv**: [2602.16609](https://arxiv.org/abs/2602.16609)
- **Key innovation**: Pre-training directly in multi-vector setting (3-phase pipeline):
  1. Unsupervised contrastive pre-training (GradCache, batch size ~16k)
  2. Supervised training with BM25-mined hard negatives
  3. Knowledge distillation from Gemma-based cross-encoder
- **Result**: 55.43 nDCG@10; supervised+KD alone reaches 99.4% at 10x lower cost

### 2.4 GTE-ModernColBERT

- **Source**: LightOn AI / Answer.AI collaboration
- **GitHub**: [lightonai/pylate](https://github.com/lightonai/pylate)
- **Backbone**: ModernBERT (8,192 token context)
- **BEIR score**: 54.89 (vs 53.79 for ColBERT-small)
- **Key**: Trained with PyLate framework in ~80 lines of code
- **Advantage**: 9+ point NDCG@10 lead on long-context benchmarks vs standard BERT-based ColBERT

### 2.5 Liquid ColBERT (LFM2-ColBERT-350M)

- **Source**: Liquid AI (2025)
- **Blog**: [liquid.ai/blog/lfm2-colbert-350m](https://www.liquid.ai/blog/lfm2-colbert-350m-one-model-to-embed-them-all)
- **Parameters**: 350M
- **Backbone**: LFM2 (Liquid Functionary Model 2) — non-Transformer architecture
- **Multilingual**: English, French, Spanish, Italian, Portuguese, German
- **Key**: Maintains ColBERT late-interaction paradigm with efficient non-Transformer backbone
- **Efficiency**: Despite 2.3x larger, achieves throughput comparable to smaller models

### 2.6 Jina-ColBERT

- **Source**: Jina AI
- **HuggingFace**: `jinaai/jina-colbert-v2`
- **Backbone**: jina-bert-v2-base-en
- **Context**: 8,192 tokens
- **Key**: Extended context ColBERT for long-document retrieval

### 2.7 PLAID (Efficient ColBERT Indexing)

- **Paper**: Santhanam et al., "PLAID: An Efficient Engine for Late Interaction Retrieval" (2022)
- **arXiv**: [2205.09707](https://arxiv.org/abs/2205.09707)
- **Key**: Centroid-based clustering + residual compression for efficient multi-vector search
- **Speed**: 6.8x GPU and 45x CPU speedup vs vanilla ColBERTv2
- **Space**: 2.7x savings (MS MARCO v2: 71GB → 27GB)

---

## 3. Hybrid / Infilling Models (GLM Family)

Trained via standalone `train_glm.py` pipeline. These use autoregressive blank infilling.

### 3.1 GLM (General Language Model)

- **Paper**: Du et al., "GLM: General Language Model Pretraining with Autoregressive Blank Infilling" (2022, ACL)
- **arXiv**: [2103.10360](https://arxiv.org/abs/2103.10360)
- **ACL**: [2022.acl-long.26](https://aclanthology.org/2022.acl-long.26/)
- **GitHub**: [THUDM/GLM](https://github.com/THUDM/GLM)
- **Sizes**: 110M, 335M, 410M, 515M, 2B, 10B (English); 335M, 10B (Chinese); 1B (multilingual, 104 languages)
- **Architecture**:
  - **Autoregressive blank infilling**: randomly blank continuous spans, reconstruct autoregressively
  - **Span shuffling**: randomly permutes reconstruction order of masked spans
  - **2D positional encoding**: (position-in-text, position-in-span)
  - Prefix decoder: bidirectional on uncorrupted text, autoregressive on infills
- **Performance**: Outperforms BERT on SuperGLUE by 4.6-5.0%, outperforms T5 on NLU + generation
- **Key advantage**: Single unified model for NLU, conditional generation, and unconditional generation

### 3.2 GLM-130B

- **Paper**: Zeng et al., "GLM-130B: An Open Bilingual Pre-Trained Model" (2023, ICLR)
- **arXiv**: [2210.02414](https://arxiv.org/abs/2210.02414)
- **Parameters**: 130B (bilingual EN/ZH)
- **Architecture evolution**:
  - Prefix decoder architecture
  - DeepNorm layer normalization
  - RoPE positional encoding
  - GeLU-gated FFN
- **Training**: 400B tokens on 96 DGX-A100 nodes
- **Note**: Too large for typical encoder pipeline; included for reference

### 3.3 ILM (Infilling Language Model)

- **Paper**: Donahue et al., "Enabling Language Models to Fill in the Blanks" (2020, ACL)
- **arXiv**: [2005.05339](https://arxiv.org/abs/2005.05339)
- **ACL**: [2020.acl-main.225](https://aclanthology.org/2020.acl-main.225/)
- **GitHub**: [chrisdonahue/ilm](https://github.com/chrisdonahue/ilm)
- **Approach**: Trains standard causal LMs on restructured sequences (masked → original concatenation)
- **Key insight**: No architectural changes needed — data restructuring enables infilling in any autoregressive LM
- **Difference from GLM**: ILM restructures data; GLM adds 2D positions + span shuffling
- **Applications**: Short stories, scientific abstracts, lyrics, antibody sequence design

---

## 4. Sparse Retrieval Models (SPLADE Family)

Trained via standalone `train_splade.py` pipeline. These learn sparse token expansion.

### 4.1 SPLADE

- **Paper**: Formal et al., "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking" (2021, SIGIR)
- **arXiv**: [2107.05720](https://arxiv.org/abs/2107.05720)
- **Architecture**:
  - Uses BERT's MLM head to produce vocab-sized sparse representations
  - Max pooling across token positions to select dominant activations
  - FLOPS regularization for sparsity: `lambda * mean(sum(log(1 + relu(scores))))`
- **Key insight**: MLM head weights naturally map tokens to vocabulary space — ideal for learned sparse retrieval

### 4.2 SPLADEv2

- **Paper**: Formal et al., "From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective" (2022, SIGIR)
- **arXiv**: [2205.04733](https://arxiv.org/abs/2205.04733)
- **HuggingFace**: `naver/splade-cocondenser-selfdistil`
- **Improvements**: Self-distillation + hard negative mining
- **CoCondenser init**: Pre-trained with corpus-aware contrastive learning

### 4.3 SPLADEv3

- **Paper**: Lassance & Clinchant, "An Efficiency Study for SPLADE Models" (2024)
- **arXiv**: [2403.06789](https://arxiv.org/abs/2403.06789)
- **GitHub**: [naver/splade](https://github.com/naver/splade)
- **HuggingFace**: `naver/splade-v3`
- **Key improvements**:
  - Document-only expansion variants
  - Ensemble of cross-encoder re-rankers for distillation
  - 50 hard negatives + 50 random from top-1k per query
  - Weighted distillation losses (KL-Div + MarginMSE)
- **Training config**: Hydra-based YAML configuration

### 4.4 CoCondenser (SPLADE Pre-training Backbone)

- **Paper**: Gao & Callan, "Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval" (2022, ACL)
- **arXiv**: [2108.05540](https://arxiv.org/abs/2108.05540)
- **GitHub**: [luyug/Condenser](https://github.com/luyug/Condenser)
- **Architecture**: Modified BERT where CLS actively conditions input representations
- **Training**: 2-stage (universal Condenser MLM → corpus-aware contrastive on document spans)
- **Key**: Often used as initialization for SPLADE models

---

## 5. Masked Auto-Encoder Models (RetroMAE Family)

Trained via standalone `train_retromae.py` pipeline. Asymmetric encoder-decoder pre-training.

### 5.1 RetroMAE

- **Paper**: Xiao & Liu, "RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder" (2022, EMNLP)
- **arXiv**: [2205.12035](https://arxiv.org/abs/2205.12035)
- **GitHub**: [staoxiao/RetroMAE](https://github.com/staoxiao/RetroMAE)
- **Architecture**:
  - **Encoder**: Full BERT-scale, 15-30% masking
  - **Decoder**: 1-layer transformer, 50-70% masking
  - Both see same text with different corruption
  - Decoder reconstructs from [CLS] embedding + heavily-masked tokens
- **Key insight**: Asymmetric masking forces encoder to produce information-rich [CLS] representations
- **Used by**: BGE models (BAAI) use RetroMAE for pre-training

### 5.2 RetroMAE-2 (Duplex)

- **Paper**: Xiao & Liu, "RetroMAE v2: Duplex Masked Auto-Encoder For Pre-Training Retrieval-Oriented Language Models" (2023)
- **arXiv**: [2305.02564](https://arxiv.org/abs/2305.02564)
- **Improvements**:
  - Two complementary tasks trained jointly on unified encoder:
    1. Reconstruct input sentence from [CLS] embedding
    2. Predict bag-of-words features from token embeddings
  - No separate decoder needed

---

## 6. Decoder-to-Encoder Conversion (LLM2Vec / MNTP)

Supported in `train_encoder.py` via `objective: mntp`.

### 6.1 LLM2Vec

- **Paper**: BehnamGhader et al., "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders" (2024)
- **arXiv**: [2404.05961](https://arxiv.org/abs/2404.05961)
- **GitHub**: [McGill-NLP/llm2vec](https://github.com/McGill-NLP/llm2vec)
- **PyPI**: `pip install llm2vec`
- **Approach**: 3-stage conversion of decoder-only LLMs to bidirectional encoders:
  1. **Enable bidirectional attention** (remove causal mask)
  2. **MNTP** (Masked Next Token Prediction): mask tokens, predict next token (not masked token)
  3. **Unsupervised contrastive learning** (SimCSE-style)
- **Supported models**: LLaMA-3, Mistral, LLaMA-2, Sheared-LLaMA
- **Config**: `mlm_probability` varies (0.2 for LLaMA-3, 0.8 for Mistral)
- **Key**: Enables reuse of decoder pre-training (NWP pipeline output) as encoder

### 6.2 NV-Embed

- **Paper**: Lee et al., "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models" (2024)
- **arXiv**: [2405.17428](https://arxiv.org/abs/2405.17428)
- **Parameters**: 7B (based on Mistral)
- **Key innovation**: Latent Attention Layer for pooling (superior to mean pooling)
- **Training**: 2-stage (retrieval contrastive → multi-task blending with in-batch negatives disabled)
- **Performance**: #1 on MTEB (72.31 as of Aug 2024)
- **Note**: Decoder-based, but relevant as encoder conversion reference

---

## 7. Training Objectives Reference

### 7.1 Masked Language Modeling (MLM)

- **Original**: BERT (Devlin et al., 2019) — 15% masking
- **Modern standard**: 30% masking (ModernBERT, MosaicBERT) for consistent accuracy gains
- **Masking strategy**: 80% [MASK], 10% random, 10% unchanged (BERT default)
- **Dynamic masking**: Different mask per epoch (RoBERTa improvement)
- **HF support**: `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)`

### 7.2 Whole-Word Masking (WWM)

- **Paper**: Originally introduced in BERT's GitHub repo, formalized in various papers
- **Mechanism**: If any subword token of a word is selected for masking, all subword tokens of that word are masked
- **Advantage**: Better for morphologically rich text (learner corpora!)
- **HF support**: `DataCollatorForWholeWordMask(mlm_probability=0.15)`
- **Requires**: Fast tokenizer with `word_ids()` method

### 7.3 Replaced Token Detection (RTD) — ELECTRA

- **Paper**: Clark et al., 2020 (see ELECTRA above)
- **Mechanism**: Generator proposes replacements → Discriminator classifies original/replaced per token
- **Loss**: `gen_mlm_loss + weight * disc_binary_cross_entropy_loss`
- **Efficiency**: Applied to every token (not just 15%), 3-7x more efficient per FLOP
- **Key parameter**: `discriminator_weight` (typically 50.0)

### 7.4 Masked Next Token Prediction (MNTP)

- **Paper**: BehnamGhader et al., 2024 (see LLM2Vec above)
- **Mechanism**: Mask tokens in bidirectional context, predict the *next* token (shifted by 1)
- **Implementation**: Remove causal mask from decoder → apply MLM-style masking → shift labels
- **Use case**: Converting pre-trained decoders to bidirectional encoders
- **Key parameter**: `mlm_probability` (model-dependent: 0.2-0.8)

### 7.5 Span Corruption (T5-style)

- **Paper**: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020)
- **arXiv**: [1910.10683](https://arxiv.org/abs/1910.10683)
- **Mechanism**: Mask contiguous spans, replace each with single sentinel token
- **Span selection**: Geometric distribution with mean `mean_span_length` (default 3.0)
- **Typical masking**: 15% of tokens across all spans

### 7.6 Autoregressive Blank Infilling (GLM)

- **Paper**: Du et al., 2022 (see GLM above)
- **Mechanism**: Blank spans → reconstruct autoregressively in shuffled order
- **Key differences from span corruption**:
  - Autoregressive generation (not encoder-decoder)
  - Span order permutation (captures inter-span dependencies)
  - 2D positional encoding
- **Loss**: Standard autoregressive cross-entropy on infill tokens

### 7.7 Late Interaction (ColBERT)

- **Paper**: Khattab & Zaharia, 2020 (see ColBERT above)
- **Mechanism**: Per-token embeddings + MaxSim scoring
- **Loss**: Contrastive (maximize query-positive MaxSim, minimize query-negative MaxSim)
- **Negatives**: In-batch negatives + mined hard negatives
- **Key parameter**: `temperature` (typically 0.05), `dim` (typically 128)

### 7.8 Sparse Expansion (SPLADE)

- **Paper**: Formal et al., 2021 (see SPLADE above)
- **Mechanism**: MLM head logits → sparse vocab activations
- **Loss**: Contrastive + FLOPS regularization + optional distillation
- **FLOPS formula**: `lambda * mean(sum(log(1 + relu(logits))))` per vocab dimension
- **Key parameters**: `lambda_q`, `lambda_d` (sparsity strength)

### 7.9 Asymmetric Masked Auto-Encoding (RetroMAE)

- **Paper**: Xiao & Liu, 2022 (see RetroMAE above)
- **Mechanism**: Full encoder (low mask) + lightweight decoder (high mask) → reconstruct
- **Loss**: MLM loss on decoder output
- **Key parameters**: `encoder_mask_ratio` (0.15-0.30), `decoder_mask_ratio` (0.50-0.70)

---

## 8. Frameworks and Libraries

### 8.1 Training Frameworks

| Framework | YAML Config | pip Install | Pre-train | Objectives |
|-----------|:-----------:|:-----------:|:---------:|------------|
| **auto-mlm-pipes** (this project) | Yes | Yes | Yes | MLM, WWM, RTD, MNTP, span corruption, ColBERT, GLM, SPLADE, RetroMAE |
| [HF Transformers](https://github.com/huggingface/transformers) (run_mlm.py) | No | Yes | Yes (MLM) | MLM |
| [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) | Yes (Composer) | No | Yes | MLM |
| [Nomic Contrastors](https://github.com/nomic-ai/contrastors) | Partial | No | Yes | Contrastive, MLM |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) v3+ | No | Yes | No | Contrastive fine-tuning |
| [PyLate](https://github.com/lightonai/pylate) | No | Yes | No | ColBERT fine-tuning |
| [RAGatouille](https://github.com/AnswerDotAI/RAGatouille) | No | Yes | No | ColBERT fine-tuning |
| [colbert-ai](https://github.com/stanford-futuredata/ColBERT) | No | Yes | No | ColBERT fine-tuning |
| [THUDM/GLM](https://github.com/THUDM/GLM) | Shell scripts | No | Yes | GLM blank infilling |
| [naver/splade](https://github.com/naver/splade) | Yes (Hydra) | No | Yes | SPLADE |
| [staoxiao/RetroMAE](https://github.com/staoxiao/RetroMAE) | No | No | Yes | RetroMAE |
| [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | No | Yes | No | Contrastive fine-tuning |
| [LLM2Vec](https://github.com/McGill-NLP/llm2vec) | JSON | Yes | Yes (MNTP) | MNTP + contrastive |
| [Tevatron](https://github.com/texttron/tevatron) | No | Yes | No | Dense retrieval |
| [GradCache](https://github.com/luyug/GradCache) | N/A | Yes | N/A | Memory-efficient contrastive (utility) |

### 8.2 Key HuggingFace Components

| Component | Use Case |
|-----------|----------|
| `AutoModelForMaskedLM` | MLM, WWM, SPLADE (uses MLM head) |
| `AutoModel` | ColBERT (per-token embeddings), RetroMAE (encoder) |
| `AutoModelForCausalLM` | GLM (autoregressive infilling), MNTP (decoder-to-encoder) |
| `DataCollatorForLanguageModeling` | MLM collation with dynamic masking |
| `DataCollatorForWholeWordMask` | WWM collation respecting word boundaries |
| `Trainer` | Standard training loop (MLM, WWM) |
| `TrainingArguments` | HF training config dataclass |

---

## 9. Benchmarks and Evaluation

### 9.1 NLU Benchmarks

| Benchmark | Tasks | Use Case |
|-----------|-------|----------|
| **GLUE** | 9 tasks (CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI) | General NLU |
| **SuperGLUE** | 8 harder tasks (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC) | Advanced NLU |

### 9.2 Retrieval Benchmarks

| Benchmark | Tasks | Use Case |
|-----------|-------|----------|
| **MTEB** | 58+ tasks across 8 categories | Comprehensive embedding evaluation |
| **BEIR** | 18 diverse retrieval datasets | Zero-shot retrieval |
| **MS MARCO** | Passage + document ranking | Information retrieval |
| **LoCo** | Long-context retrieval (up to 32K tokens) | Long-document evaluation |

### 9.3 MTEB Leaderboard Highlights (2024-2025)

| Rank | Model | Type | Score |
|------|-------|------|-------|
| 1 | NV-Embed (7B) | Decoder-as-encoder | 72.31 |
| 2 | Jasper (2B, Stella distillation) | Distilled | 71.54 |
| 3 | Cohere embed-v4 | Proprietary | 65.2 |
| 4 | OpenAI text-embedding-3-large | Proprietary | 64.6 |
| 5 | BGE-M3 | Open-source encoder | 63.0 |

---

## Key Papers by Year

### 2019
- BERT (Devlin et al.) — MLM + NSP, foundation model
- RoBERTa (Liu et al.) — optimized BERT training
- ALBERT (Lan et al.) — parameter-efficient BERT
- XLNet (Yang et al.) — permutation LM

### 2020
- ELECTRA (Clark et al.) — replaced token detection
- ILM (Donahue et al.) — infilling language models
- ColBERT (Khattab & Zaharia) — late interaction retrieval
- UniLMv2 (Bao et al.) — unified language model

### 2021
- SPLADE (Formal et al.) — sparse lexical expansion
- ERNIE 3.0 (Sun et al.) — knowledge-enhanced pre-training
- Condenser (Gao & Callan) — dense retrieval pre-training

### 2022
- GLM (Du et al.) — autoregressive blank infilling
- ColBERTv2 (Santhanam et al.) — lightweight late interaction
- RetroMAE (Xiao & Liu) — retrieval-oriented MAE
- CoCondenser (Gao & Callan) — corpus-aware contrastive
- SPLADEv2 (Formal et al.) — self-distillation + hard negatives
- FNet (Lee-Thorp et al.) — Fourier attention replacement
- E5 (Wang et al.) — weakly-supervised contrastive

### 2023
- DeBERTaV3 (He et al.) — disentangled attention + RTD
- RetroMAE-2 (Xiao & Liu) — duplex MAE
- GLM-130B (Zeng et al.) — large-scale bilingual GLM
- M2-BERT (Fu et al.) — sub-quadratic Monarch Mixer

### 2024
- ModernBERT (Warner et al.) — modern bidirectional encoder
- NomicBERT (Nussbaum et al.) — open reproducible embeddings
- LLM2Vec (BehnamGhader et al.) — decoder-to-encoder conversion
- NV-Embed (Lee et al.) — LLM embeddings with latent attention
- SPLADEv3 (Lassance & Clinchant) — efficient sparse models
- Jina Embeddings v3 — task-specific LoRA adapters
- BGE-M3 — multi-functionality (dense + sparse + multi-vector)
- GTE-ModernColBERT — ModernBERT backbone for ColBERT
- Liquid ColBERT (LFM2-ColBERT-350M) — non-Transformer ColBERT

### 2025
- NeoBERT (Gonçalves et al.) — next-gen BERT, 2T tokens
- ColBERT-Zero — pre-training in multi-vector setting
- CDE (Contextual Document Embeddings) — corpus-level context conditioning
