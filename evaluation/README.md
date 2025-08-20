# Dual-Head Model Evaluation Suite

This directory contains a comprehensive evaluation framework for comparing the Dual-Head model against baseline methods (DPO, SimPO, ARGS, GenARM) using the methodology described in the paper.

## Directory Structure

```
dual_head_evaluation/
├── scripts/
│   ├── generate_dualhead.py    # Generate responses using Dual-Head model
│   ├── generate_dpo.py         # Generate responses using DPO model
│   ├── generate_simpo.py       # Generate responses using SimPO model
│   ├── generate_args.py        # Generate responses using ARGS model
│   ├── generate_genarm.py      # Generate responses using GenARM model
│   └── evaluate_models.py      # Comprehensive evaluation script
├── outputs/                    # Generated responses from each model 
├── results/                    # Evaluation results and reports
├── generate_all_responses.sh   # Shell script to run all generation scripts
└── README.md                   # This file
```

## Usage

### Step 1: Generate Responses from All Models

Run the shell script to generate responses from all models:

```bash
cd /home/ibel/research/dual_head_evaluation
./generate_all_responses.sh 300
```

This will:
- Generate 300 random responses from HH-RLHF test dataset for each model
- Save responses to `outputs/` directory
- Use the same random seed for consistency across models

### Step 2: Evaluate All Models

Run the comprehensive evaluation:

```bash
python scripts/evaluate_models.py 300
```

This performs the evaluations described in the paper:

#### Pairwise Comparisons (GPT-4)
- Head-to-head comparisons between all model pairs
- GPT-4 judges responses on helpfulness, harmlessness, honesty, and coherence
- Calculates win rates as reported in paper tables

#### Quality Analysis
- **Average Reward**: Using reward models or GPT-4 scoring (scale 1-5)
- **Diversity**: Lexical diversity using distinct n-gram ratios
- **Coherence**: Semantic coherence and structure analysis

#### Efficiency Analysis
- **Latency**: Average generation time per sample
- **Tokens/sec**: Generation speed
- **Speedup**: Relative performance compared to slowest method

## Paper Metrics Implementation

The evaluation implements all metrics from the Dual-Head paper:

### Main Results (Table 1)
- Pairwise comparison win rates via GPT-4 evaluation
- 300 test prompts from HH-RLHF dataset
- Win rates show how often Dual-Head is preferred over each baseline

### Quality Analysis (Table 2) 
- Average Reward using trajectory-level reward models
- Diversity using distinct-1 token ratios
- Coherence based on semantic similarity and structure

### Efficiency Analysis (Table 3)
- Latency measurements for generating 128 tokens
- Memory usage tracking
- Forward passes per token comparison
- Speedup calculations relative to baselines

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- OpenAI API key (for GPT-4 evaluation)
- Datasets library
- All model implementations (Dual-Head, DPO, SimPO, ARGS, GenARM)

## Configuration

Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export CUDA_VISIBLE_DEVICES=0
```

## Expected Output

The evaluation generates:

1. **Console Output**: Formatted tables matching paper results
2. **JSON Report**: Detailed results in `results/evaluation_results_300.json`
3. **Summary Statistics**: Win rates, quality metrics, efficiency comparisons

Example output tables:

```
--- PAIRWISE COMPARISON RESULTS ---
Model       dualhead    dpo         simpo       args        genarm     
----------------------------------------------------------------
dualhead    ---         52.3%       58.7%       76.2%       64.8%      
dpo         47.7%       ---         45.2%       68.9%       51.4%      
simpo       41.3%       54.8%       ---         71.3%       49.7%      
...

--- QUALITY AND ALIGNMENT ANALYSIS ---
Method       Avg Reward   Diversity    Coherence   
------------------------------------------------
dualhead        4.30        0.53        0.69
dpo             3.90        0.51        0.67
simpo           3.60        0.49        0.66
...
```

## Troubleshooting

1. **Missing Models**: Scripts will fallback to base models if trained versions not found
2. **OpenAI Issues**: GPT-4 evaluation will use random fallback if API unavailable  
3. **Memory Issues**: Reduce batch size or sample count if CUDA OOM occurs
4. **Timing Issues**: Some models may take hours to generate 300 responses

## Paper Reproduction

This evaluation suite reproduces the experimental setup from the Dual-Head paper:
- Same dataset (HH-RLHF test set)
- Same evaluation protocol (GPT-4 pairwise comparison)
- Same metrics (reward, diversity, coherence, efficiency)
- Same baseline comparisons (DPO, SimPO, ARGS, GenARM)