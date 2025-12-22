# FSMT Evolution Strategies Finetuning Project

## Project Overview

This project implements **Evolution Strategies (ES)** finetuning for **FSMT (FairSeq Machine Translation)** models, enabling gradient-free optimization for neural machine translation tasks.

## Model Configuration

### Target Model: FSMT English-German Translation

```json
{
  "model_type": "fsmt",
  "is_encoder_decoder": true,
  "d_model": 1024,
  "encoder_layers": 8,
  "decoder_layers": 2,
  "encoder_attention_heads": 16,
  "decoder_attention_heads": 16,
  "encoder_ffn_dim": 4096,
  "decoder_ffn_dim": 4096,
  "src_vocab_size": 40963,
  "tgt_vocab_size": 40963,
  "max_position_embeddings": 512,
  "langs": ["en", "de"]
}
```

### Parameter Statistics

**Total Parameters**: ~87M parameters (estimated for FSMT Base)

**Parameter Breakdown**:
- **Embeddings**: ~42M params
  - Source embeddings: 40963 × 1024 = ~42M
  - Target embeddings: 40963 × 1024 = ~42M (shared)
  - Position embeddings: 512 × 1024 = ~0.5M

- **Encoder** (8 layers): ~25M params per model
  - Self-attention: ~4M per layer (Q, K, V, O projections)
  - FFN: ~8M per layer (two linear layers: 1024→4096→1024)
  - Layer norms: ~2K per layer
  - Total: ~96M params

- **Decoder** (2 layers): ~6M params per model
  - Self-attention: ~4M per layer
  - Cross-attention: ~4M per layer
  - FFN: ~8M per layer
  - Layer norms: ~2K per layer
  - Total: ~32M params

## Evolution Strategy Configuration

### LoRA-Style Evolution (Recommended)
**Purpose**: Memory-efficient finetuning, only evolve critical parameters

**Evolved Parameters** (~20% of total):
- ✅ All attention projection matrices (Q, K, V, O)
- ✅ Encoder self-attention: 8 layers × 4M = 32M params
- ✅ Decoder self-attention: 2 layers × 4M = 8M params
- ✅ Decoder cross-attention: 2 layers × 4M = 8M params
- **Total**: ~48M evolved params

**Frozen Parameters**:
- ❌ Embeddings (42M params)
- ❌ FFN layers (96M params in encoder, 16M in decoder)
- ❌ Layer norms

### Full Evolution (Optional)
**Purpose**: Maximum adaptation capacity, higher memory requirement

**Evolved Parameters**: All ~87M parameters

## ES Algorithm Selection

### Available Noisers

1. **EggRoll** (Recommended for translation)
   - Adaptive learning rates per parameter
   - Momentum-based updates
   - Good for seq2seq tasks

2. **OpenES**
   - Vanilla ES with fitness ranking
   - Simple and robust
   - Good baseline

3. **Base Noiser**
   - Simple Gaussian perturbation
   - Fast but less sophisticated

## Project Structure

```
HyperscaleES/
├── src/hyperscalees/
│   ├── models/
│   │   ├── fsmt_config.py          ✅ DONE (Task 1)
│   │   ├── fsmt_analysis.py        ✅ DONE (Task 1)
│   │   ├── fsmt_loader.py          ⏳ TODO (Task 2)
│   │   └── fsmt/
│   │       ├── __init__.py         ⏳ TODO (Task 3)
│   │       ├── forward.py          ⏳ TODO (Task 3)
│   │       └── generation.py       ⏳ TODO (Task 3)
│   ├── environments/
│   │   └── translation_task.py     ⏳ TODO (Task 5)
│   └── noiser/
│       └── (existing noisers)      ✅ Already exists
├── llm_experiments/
│   └── fsmt_do_evolution.py        ⏳ TODO (Task 7)
├── configs/
│   ├── fsmt_small.yaml             ⏳ TODO (Task 9)
│   ├── fsmt_base.yaml              ⏳ TODO (Task 9)
│   └── fsmt_lora.yaml              ⏳ TODO (Task 9)
└── docs/
    └── FSMT_ES_PROJECT.md          ✅ DONE (Task 1)
```

## Task Progress

### ✅ Task 1: Setup & Analysis (COMPLETED)
- [x] Created `fsmt_config.py` with model configuration
- [x] Created `fsmt_analysis.py` for parameter analysis
- [x] Documented parameter structure and evolution strategy
- [x] Created project documentation

**Key Outputs**:
- `FSMTConfig` dataclass with all model settings
- `FSMTParameterAnalyzer` for analyzing parameter structures
- ES map strategy (LoRA vs Full evolution)
- Parameter statistics and breakdown

### ⏳ Task 2: Model Loader (NEXT)
- [ ] Create `fsmt_loader.py`
- [ ] Load FSMT from Hugging Face
- [ ] Convert PyTorch → JAX parameters
- [ ] Test parameter extraction

### ⏳ Task 3: Forward Pass & Generation
- [ ] Implement encoder-decoder forward pass
- [ ] Implement beam search generation
- [ ] Test with pretrained weights

### ⏳ Task 4: Noiser Adaptation
- [ ] Adapt ES noiser for FSMT structure
- [ ] Implement noise injection
- [ ] Test parameter mutation

### ⏳ Task 5: Translation Task Environment
- [ ] Create `TranslationTask` class
- [ ] Implement BLEU score fitness
- [ ] Load translation datasets

### ⏳ Task 6: Update Function
- [ ] Adapt ES update for FSMT
- [ ] Handle encoder/decoder separately
- [ ] Test convergence

### ⏳ Task 7: Main Training Script
- [ ] Create main training loop
- [ ] Integrate all components
- [ ] Add checkpointing & logging

### ⏳ Task 8: Testing & Optimization
- [ ] Test on small dataset
- [ ] Debug and optimize
- [ ] Validate BLEU improvements

### ⏳ Task 9: Documentation
- [ ] Write usage README
- [ ] Create example configs
- [ ] Document hyperparameters

## Expected Hyperparameters

```yaml
# Example configuration
model:
  name: "facebook/wmt19-en-de"
  freeze_nonlora: true

evolution:
  noiser: "eggroll"
  sigma: 1e-3
  lr_scale: 1.0
  noise_reuse: 1
  generations_per_prompt: 8
  parallel_generations_per_gpu: 256

training:
  num_epochs: 50
  validate_every: 5
  log_output_every: 10

task:
  dataset: "wmt14"
  src_lang: "en"
  tgt_lang: "de"
  max_length: 200
```

## Key Differences from RWKV Evolution

| Aspect | RWKV (Original) | FSMT (This Project) |
|--------|----------------|---------------------|
| Architecture | RNN-style recurrent | Encoder-decoder transformer |
| State | Recurrent hidden state | No state (parallel encoder) |
| Generation | Autoregressive only | Encoder once, decode with beam search |
| Attention | Linear attention | Multi-head self & cross attention |
| Parameters | ~1.5B (typical) | ~87M (base model) |
| Task | General LLM | Translation-specific |
| Fitness | Task-specific | BLEU score |

## Next Steps

**Immediate**: Proceed to Task 2 - Create `fsmt_loader.py` to load models from Hugging Face and convert to JAX format.

---

*Last Updated: Task 1 completed*
*Next: Task 2 - Model Loader*
