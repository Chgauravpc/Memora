"""
Phase 4 Demo: Memory Consolidation & 5-Signal Ranking

Demonstrates:
1. 5-signal ranking (semantic + type + recency + frequency + confidence)
2. Access tracking and frequency scoring
3. Background consolidation (decay, merge, promote)
4. Memory promotion to Core Memory
"""

import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check Phase 4 dependencies
def check_dependencies():
    """Verify Phase 4 dependencies are available"""
    try:
        from src.consolidation_worker import ConsolidationWorker
        from src.config import RANKING_WEIGHTS_5_SIGNAL, CONSOLIDATION_ENABLED
        return True
    except ImportError as e:
        print(f"✗ Missing Phase 4 components: {e}")
        return False

if not check_dependencies():
    sys.exit(1)

print("=" * 70)
print(" PHASE 4 DEMO: Consolidation & 5-Signal Ranking")
print("=" * 70)
print("✓ Phase 4 dependencies available\n")

from src import MemorySystem
from src.config import (
    RANKING_WEIGHTS_5_SIGNAL,
    CONSOLIDATION_INTERVAL_TURNS,
    PROMOTION_CONFIDENCE_THRESHOLD,
    PROMOTION_MENTION_THRESHOLD,
    PROMOTION_ACCESS_THRESHOLD,
    DECAY_TURNS_THRESHOLD,
)

# Sample conversation with repeating information (for confidence boosting)
SAMPLE_CONVERSATION = [
    # Initial information
    "Hi, I'm Marcus. I work as a data scientist at TechCorp.",
    "I prefer Python for data analysis work.",
    "My manager's name is Jennifer Wilson.",
    "I always need my code reviewed before deployment.",
    
    # Reinforcing information (should boost confidence)
    "As I mentioned, Python is my go-to language for data work.",
    "Remember, I'm Marcus from TechCorp.",
    "Jennifer Wilson is still my manager.",
    
    # More context
    "I'm working on a machine learning project called Project Alpha.",
    "I usually work late nights, that's when I'm most focused.",
    "I have a meeting every Monday at 10 AM.",
    
    # More reinforcement
    "Python is definitely my favorite for ML work.",
    "Did I mention Project Alpha? It's our main priority.",
    
    # New preferences
    "I prefer dark mode in all my IDEs.",
    "Coffee is essential for my workflow.",
    "Never call me before 9 AM.",
    
    # More repetition to boost
    "Just to confirm, I'm Marcus, data scientist at TechCorp.",
    "Python and dark mode - my essentials.",
    "Project Alpha deadline is next month.",
    
    # More turns to trigger consolidation
    "Working on feature extraction today.",
    "The ML model is showing good results.",
    "Need to update Jennifer about progress.",
    "Late night coding session coming up.",
    "Another productive day with Python.",
]


def demo_5_signal_ranking():
    """Demonstrate 5-signal ranking"""
    print("\n" + "=" * 70)
    print(" DEMO 1: 5-Signal Ranking")
    print("=" * 70)
    
    print("\n--- 5-Signal Ranking Configuration ---")
    print("Ranking Weights:")
    for signal, weight in RANKING_WEIGHTS_5_SIGNAL.items():
        print(f"  - {signal}: {weight:.0%}")
    
    print("\nNew signals in Phase 4:")
    print("  - FREQUENCY: Based on how often a memory is retrieved")
    print("  - CONFIDENCE: Based on extraction certainty and repetition")


def demo_consolidation():
    """Demonstrate memory consolidation"""
    print("\n" + "=" * 70)
    print(" DEMO 2: Memory Consolidation")
    print("=" * 70)
    
    print("\n--- Consolidation Configuration ---")
    print(f"Consolidation interval: Every {CONSOLIDATION_INTERVAL_TURNS} turns")
    print(f"Decay threshold: After {DECAY_TURNS_THRESHOLD} turns")
    print(f"Promotion requirements:")
    print(f"  - Confidence >= {PROMOTION_CONFIDENCE_THRESHOLD}")
    print(f"  - Mentions >= {PROMOTION_MENTION_THRESHOLD}")
    print(f"  - Accesses >= {PROMOTION_ACCESS_THRESHOLD}")
    print(f"\nNote: This demo has {len(SAMPLE_CONVERSATION)} turns.")
    print(f"      Automatic consolidation won't trigger (needs {CONSOLIDATION_INTERVAL_TURNS} turns).")
    print(f"      Demo will force manual consolidation at the end.")


def main():
    """Main demo function"""
    
    demo_5_signal_ranking()
    demo_consolidation()
    
    print("\n" + "=" * 70)
    print(" DEMO 3: Processing Conversation with Phase 4 Features")
    print("=" * 70)
    
    # Initialize memory system
    print("\n--- Initializing Memory System ---")
    try:
        memory = MemorySystem(user_id="phase4_demo")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("Make sure Redis and Qdrant are running:")
        print("  docker-compose up -d")
        return
    
    # Check health
    health = memory.health_check()
    print(f"Health Check: {health}")
    
    # Check if consolidation is enabled
    print(f"Consolidation enabled: {memory.is_consolidation_enabled()}")
    
    if not all(v is not False for v in health.values() if v is not None):
        print("\n✗ Some services are not healthy. Please check your setup.")
        return
    
    # Clear previous demo data
    print("\n--- Clearing Previous Demo Data ---")
    memory.clear_memories()
    
    # Process conversation
    print("\n--- Processing Sample Conversation ---")
    total_extracted = 0
    total_stored = 0
    
    for i, message in enumerate(SAMPLE_CONVERSATION, 1):
        memory_context, stats = memory.process_turn(message)
        total_extracted += stats['extracted_count']
        total_stored += stats['stored_count']
        
        # Show brief progress
        if stats['extracted_count'] > 0:
            print(f"Turn {i}: Extracted={stats['extracted_count']}, Stored={stats['stored_count']}")
        
        # Check if consolidation ran
        if 'consolidation' in stats:
            print(f"  [CONSOLIDATION] decayed={stats['consolidation'].get('decayed', 0)}, "
                  f"merged={stats['consolidation'].get('merged', 0)}, "
                  f"promoted={stats['consolidation'].get('promoted', 0)}")
    
    # Show memory statistics
    print("\n--- Memory Statistics ---")
    final_stats = memory.get_statistics()
    print(f"Total turns: {final_stats['total_turns']}")
    print(f"Total memories: {final_stats['total_memories']}")
    print(f"Memories by type: {final_stats['memories_by_type']}")
    
    # Show consolidation stats
    if memory.is_consolidation_enabled():
        consolidation_stats = memory.get_consolidation_stats()
        if consolidation_stats:
            print(f"\nConsolidation Stats:")
            print(f"  - Decayed: {consolidation_stats.get('decayed', 0)}")
            print(f"  - Deleted: {consolidation_stats.get('deleted', 0)}")
            print(f"  - Merged: {consolidation_stats.get('merged', 0)}")
            print(f"  - Promoted: {consolidation_stats.get('promoted', 0)}")
    
    # Demonstrate 5-signal retrieval
    print("\n" + "=" * 70)
    print(" DEMO 4: 5-Signal Retrieval in Action")
    print("=" * 70)
    
    test_queries = [
        ("What's my name?", "Should retrieve Marcus entity"),
        ("What language do I use?", "Should retrieve Python preference - high frequency"),
        ("Tell me about my project", "Should retrieve Project Alpha"),
        ("What are my rules?", "Should retrieve constraints/instructions"),
    ]
    
    for query, expected in test_queries:
        print(f"\n📝 Query: '{query}'")
        print(f"   Expected: {expected}")
        
        start = time.time()
        context, stats = memory.process_turn(query)
        elapsed = (time.time() - start) * 1000
        
        print(f"   Time: {elapsed:.1f}ms")
        print(f"   Retrieved {stats['retrieved_count']} memories")
        
        # Show first few lines of context
        if context:
            lines = context.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"     {line[:80]}...")
    
    # Force consolidation to demonstrate
    print("\n" + "=" * 70)
    print(" DEMO 5: Manual Consolidation")
    print("=" * 70)
    
    if memory.is_consolidation_enabled():
        print("\n--- Running Manual Consolidation (Forced) ---")
        result = memory.run_consolidation(force=True)
        if result:
            print(f"Consolidation results:")
            print(f"  - Decayed: {result.get('decayed', 0)}")
            print(f"  - Deleted: {result.get('deleted', 0)}")
            print(f"  - Merged: {result.get('merged', 0)}")
            print(f"  - Promoted: {result.get('promoted', 0)}")
            print(f"  - Duration: {result.get('duration_ms', 0):.1f}ms")
    else:
        print("Consolidation worker not available")
    
    # Final statistics
    print("\n--- Final Statistics ---")
    final_stats = memory.get_statistics()
    print(f"Total memories after consolidation: {final_stats['total_memories']}")
    print(f"Memories by type: {final_stats['memories_by_type']}")
    
    print("\n" + "=" * 70)
    print(" Demo Complete!")
    print("=" * 70)
    
    print("""
Phase 4 Features Demonstrated:
✓ 5-signal ranking (semantic + type + recency + frequency + confidence)
✓ Access tracking for frequency scoring
✓ Background consolidation worker
✓ Memory decay for old/unused memories
✓ Memory merging for similar content
✓ Promotion to Core Memory for high-value memories

Key Configuration (in config.py):
- RANKING_WEIGHTS_5_SIGNAL: Tune signal weights
- CONSOLIDATION_INTERVAL_TURNS: When to run consolidation
- DECAY_TURNS_THRESHOLD: When decay starts
- PROMOTION_*_THRESHOLD: Criteria for core memory promotion
""")


if __name__ == "__main__":
    main()
