#!/usr/bin/env python3
"""
Quick demonstration of Phase 5 evaluation capabilities
Runs a smaller test (10 conversations) for quick validation
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.conversation_generator import ConversationGenerator
from evaluation.evaluator import MemorySystemEvaluator

logging.basicConfig(level=logging.WARNING)


def main():
    print("\n" + "="*70)
    print("  PHASE 5 EVALUATION - Quick Demo")
    print("  Testing with 10 conversations")
    print("="*70)
    
    # Generate small test set
    print("\nGenerating 10 test conversations...")
    generator = ConversationGenerator(seed=42)
    conversations = generator.generate_batch(
        num_conversations=10,
        turns_per_conversation=(10, 20),
    )
    
    distance_sweep = generator.generate_distance_sweep_conversation(
        target_distances=[10, 50, 100]
    )
    
    print(f"✓ Generated {len(conversations)} conversations")
    print(f"✓ Total turns: {sum(c['num_turns'] for c in conversations)}")
    print(f"✓ Total memories: {sum(c['total_memories'] for c in conversations)}")
    
    # Run evaluation
    evaluator = MemorySystemEvaluator()
    
    print("\n--- Testing Extraction ---")
    extraction = evaluator.evaluate_extraction(
        conversations=conversations[:5],
        verbose=False
    )
    print(f"Precision: {extraction['avg_precision']:.1%}")
    print(f"Recall:    {extraction['avg_recall']:.1%}")
    print(f"F1 Score:  {extraction['avg_f1_score']:.1%}")
    
    print("\n--- Testing Retrieval ---")
    retrieval = evaluator.evaluate_retrieval(
        conversations=conversations[:5],
        verbose=False
    )
    print(f"Context Precision: {retrieval['avg_context_precision']:.1%}")
    print(f"Context Recall:    {retrieval['avg_context_recall']:.1%}")
    print(f"MRR (Ranking):     {retrieval['avg_mrr']:.3f}")
    
    print("\n--- Testing Distance Sweep ---")
    distance = evaluator.evaluate_distance_sweep(
        distance_sweep_conversation=distance_sweep,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("  Quick demo complete!")
    print("  For full evaluation (200 conversations):")
    print("    python run_evaluation.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
