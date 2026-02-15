# Phase 5: Evaluation Framework

Comprehensive evaluation framework for the Memory System using RAGAS-style metrics.

## Quick Start

```bash
# Install evaluation dependencies
pip install -r requirements_evaluation.txt

# Run full evaluation (generates 200 test conversations)
python run_evaluation.py
```

## What's Evaluated

### 1. Extraction Accuracy (Phases 1-3)
Tests how accurately the system extracts memories from conversations:
- **Precision**: Proportion of extracted memories that are correct
- **Recall**: Proportion of ground truth memories that were extracted
- **F1 Score**: Harmonic mean of precision and recall

### 2. Retrieval Quality (Phase 2 + RAGAS)
Tests how well the system retrieves relevant memories:
- **Context Precision**: Relevance of retrieved memories (RAGAS metric)
- **Context Recall**: Coverage of relevant memories (RAGAS metric)
- **MRR**: Mean Reciprocal Rank (ranking quality)
- **Top-K Accuracy**: Performance at different retrieval limits

### 3. Distance Sweep (Long-Term Recall)
Tests ability to remember information over long conversations:
- Recall at 10 turns ago
- Recall at 50 turns ago
- Recall at 100 turns ago
- Recall at 500 turns ago
- Recall at 1000 turns ago

**Critical Test**: Can the system remember info from 1000+ turns ago?

### 4. Consolidation Quality (Phase 4)
Tests background consolidation effectiveness:
- **Decay**: Are old/unused memories appropriately decayed?
- **Merging**: Are similar memories correctly merged?
- **Promotion**: Are high-value memories promoted to core?

## Test Data

### Generated Conversations
- **200 synthetic conversations** with ground truth memories
- 10-30 turns per conversation
- ~60% memory density (realistic mix of informative and filler turns)
- Multiple personas for variety

### Distance Sweep Test
- Special conversation with memories at 10, 50, 100, 500, 1000 turns
- Tests long-term recall capability
- Critical for validating 1000+ turn requirement

## Architecture

```
evaluation/
├── __init__.py                    # Package initialization
├── README.md                      # This file
├── conversation_generator.py      # Generates synthetic test data
├── metrics.py                     # Evaluation metrics
├── evaluator.py                   # Main evaluation orchestrator
├── fixtures/                      # Test data
│   ├── test_conversations.json    # 200 generated conversations
│   └── distance_sweep_test.json   # Long-term recall test
└── results/                       # Evaluation results
    └── evaluation_results.json    # Latest results
```

## Running Individual Tests

```python
from evaluation import MemorySystemEvaluator
from src import MemorySystem

# Initialize
evaluator = MemorySystemEvaluator()

# Test extraction only
extraction_results = evaluator.evaluate_extraction(
    conversations=conversations,
    verbose=True
)

# Test retrieval only
retrieval_results = evaluator.evaluate_retrieval(
    conversations=conversations,
    verbose=True
)

# Test distance sweep only
distance_results = evaluator.evaluate_distance_sweep(
    distance_sweep_conversation=distance_sweep,
    verbose=True
)

# Test consolidation only
consolidation_results = evaluator.evaluate_consolidation(
    conversation=conversation,
    verbose=True
)
```

## Customizing Test Data

```python
from evaluation.conversation_generator import ConversationGenerator

# Generate custom conversations
generator = ConversationGenerator(seed=42)

# Generate 100 conversations with 50-100 turns each
conversations = generator.generate_batch(
    num_conversations=100,
    turns_per_conversation=(50, 100),
    output_file="my_test_data.json"
)

# Generate custom distance sweep
distance_sweep = generator.generate_distance_sweep_conversation(
    target_distances=[20, 100, 500, 2000]  # Custom distances
)
```

## Understanding Results

### Good Performance Targets

**Extraction:**
- Precision: >85% (few false positives)
- Recall: >80% (catches most memories)
- F1: >82%

**Retrieval:**
- Context Precision: >90% (retrieved memories are relevant)
- Context Recall: >85% (finds most relevant memories)
- MRR: >0.8 (relevant memories ranked high)

**Distance Sweep:**
- 10 turns: >95% recall
- 50 turns: >90% recall
- 100 turns: >85% recall
- 500 turns: >70% recall
- 1000 turns: >60% recall

**Consolidation:**
- Decay: Appropriate confidence reduction
- Merge: >90% duplicate reduction
- Promotion: >80% success rate

## RAGAS Integration

This framework uses RAGAS-style metrics for retrieval evaluation:

```python
from evaluation.metrics import RetrievalMetrics

metrics = RetrievalMetrics()

# Calculate context precision (like RAGAS)
precision = metrics.calculate_context_precision(
    retrieved_memories=retrieved,
    ground_truth_memories=ground_truth
)

# Calculate context recall (like RAGAS)
recall = metrics.calculate_context_recall(
    retrieved_memories=retrieved,
    ground_truth_memories=ground_truth
)

# Calculate ranking quality
ranking = metrics.calculate_ranking_quality(
    retrieved_memories=retrieved,
    ground_truth_memories=ground_truth
)
```

## Continuous Evaluation

Integrate into CI/CD:

```bash
# Run evaluation as part of CI
python run_evaluation.py

# Check exit code (0 = success, 1 = failure)
if [ $? -eq 0 ]; then
    echo "Evaluation passed"
else
    echo "Evaluation failed"
    exit 1
fi
```

## Future Enhancements

- [ ] Adversarial test cases (contradictions, updates)
- [ ] Multi-language support testing
- [ ] Performance benchmarking under load
- [ ] A/B testing framework for parameter tuning
- [ ] Automated regression detection
- [ ] Visual dashboards for results

## Troubleshooting

### "System health check failed"
- Make sure Redis is running: `docker-compose up -d`
- Check Qdrant is accessible (optional but recommended)

### "No test data found"
- Run `python run_evaluation.py` - it will generate data automatically
- Or manually run: `python evaluation/conversation_generator.py`

### "Low recall at distance X"
Check configuration in `src/config.py`:
- `RECENCY_DECAY_RATE` - controls how fast memories decay
- `RANKING_WEIGHTS_5_SIGNAL` - adjust signal weights
- `CONSOLIDATION_INTERVAL_TURNS` - when consolidation runs

## References

- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- [Memory System Specification](../README.md)
- [Evaluation Best Practices](https://arxiv.org/abs/2404.12387)
