#!/usr/bin/env python3
"""
Phase 5: Run Complete Evaluation Suite
Tests memory system with 200 conversation samples using RAGAS-style metrics
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.conversation_generator import ConversationGenerator
from evaluation.evaluator import MemorySystemEvaluator

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING to reduce noise during evaluation
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run complete evaluation pipeline"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          MEMORY SYSTEM EVALUATION - Phase 5                          ║
║          RAGAS-based Testing with 200 Conversation Samples           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Generate test data
    print("\n[STEP 1/3] Generating Test Data")
    print("─" * 70)
    
    fixtures_dir = Path("evaluation/fixtures")
    test_conversations_file = fixtures_dir / "test_conversations.json"
    distance_sweep_file = fixtures_dir / "distance_sweep_test.json"
    
    if not test_conversations_file.exists():
        print("Generating 200 test conversations...")
        generator = ConversationGenerator(seed=42)
        
        # Generate 200 conversations
        generator.generate_batch(
            num_conversations=200,
            turns_per_conversation=(10, 30),
            output_file=str(test_conversations_file)
        )
        
        # Generate distance sweep test
        distance_sweep = generator.generate_distance_sweep_conversation(
            target_distances=[10, 50, 100, 500, 1000]
        )
        
        import json
        distance_sweep_file.parent.mkdir(parents=True, exist_ok=True)
        with open(distance_sweep_file, 'w') as f:
            json.dump(distance_sweep, f, indent=2)
        
        print("✓ Test data generated")
    else:
        print("✓ Test data already exists")
    
    # Step 2: Check system health
    print("\n[STEP 2/3] System Health Check")
    print("─" * 70)
    
    from src import MemorySystem
    
    try:
        test_system = MemorySystem(user_id="health_check")
        health = test_system.health_check()
        
        print(f"  Redis: {'✓' if health['redis'] else '✗'}")
        print(f"  Flat Files: {'✓' if health['flat_files'] else '✗'}")
        print(f"  Vector Store: {'✓' if health['vector_store'] else '✗ (optional)'}")
        
        if not health['redis'] or not health['flat_files']:
            print("\n✗ System health check failed!")
            print("  Make sure Redis is running: docker-compose up -d")
            return 1
        
        print("\n✓ System is healthy")
    
    except Exception as e:
        print(f"\n✗ System initialization failed: {e}")
        print("  Make sure Redis is running: docker-compose up -d")
        return 1
    
    # Step 3: Run evaluation
    print("\n[STEP 3/3] Running Evaluation Suite")
    print("─" * 70)
    
    evaluator = MemorySystemEvaluator()
    
    results = evaluator.run_full_evaluation(
        test_conversations_file=str(test_conversations_file),
        distance_sweep_file=str(distance_sweep_file),
        output_file="evaluation/results/evaluation_results.json"
    )
    
    # Print summary
    print("\n" + "="*70)
    print("  EVALUATION SUMMARY")
    print("="*70)
    
    print("\n📊 EXTRACTION METRICS (Phases 1-3)")
    print(f"  Precision:  {results['extraction']['avg_precision']:.1%}")
    print(f"  Recall:     {results['extraction']['avg_recall']:.1%}")
    print(f"  F1 Score:   {results['extraction']['avg_f1_score']:.1%}")
    print(f"  Extracted:  {results['extraction']['total_extracted']} memories")
    print(f"  Expected:   {results['extraction']['total_ground_truth']} memories")
    
    print("\n📊 RETRIEVAL METRICS (Phase 2 + RAGAS)")
    print(f"  Context Precision: {results['retrieval']['avg_context_precision']:.1%}")
    print(f"  Context Recall:    {results['retrieval']['avg_context_recall']:.1%}")
    print(f"  MRR (Ranking):     {results['retrieval']['avg_mrr']:.3f}")
    print(f"  Avg Position:      {results['retrieval']['avg_position']:.1f}")
    
    print("\n📊 DISTANCE SWEEP (Long-Term Recall)")
    for distance, result in results['distance_sweep'].items():
        if result and result != 'None':
            recall_pct = result['recall'] * 100
            status = "✓" if recall_pct >= 80 else "⚠" if recall_pct >= 60 else "✗"
            print(f"  {status} {distance.replace('distance_', '')} turns ago: "
                  f"{recall_pct:.0f}% ({result['found']}/{result['expected']})")
    
    print("\n📊 CONSOLIDATION METRICS (Phase 4)")
    decay = results['consolidation']['decay']
    merge = results['consolidation']['merge']
    promotion = results['consolidation']['promotion']
    
    print(f"  Decayed:    {decay['decayed_count']} memories")
    print(f"  Deleted:    {decay['deleted_count']} low-confidence")
    print(f"  Merged:     {merge['merged_count']} similar memories")
    print(f"  Promoted:   {promotion['promoted_count']} to core")
    print(f"  Success:    {promotion['promotion_success_rate']:.1%}")
    
    print("\n" + "="*70)
    print(f"  ✓ Evaluation completed in {results['duration_seconds']:.1f}s")
    print(f"  ✓ Detailed results: evaluation/results/evaluation_results.json")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
