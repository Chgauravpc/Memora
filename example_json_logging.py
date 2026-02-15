#!/usr/bin/env python3
"""
Example: JSON Logging for Per-Turn Statistics

Demonstrates how to enable JSON logging to save all turn statistics
to a JSON file for later analysis, monitoring, or API integration.

Run: python example_json_logging.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import MemorySystem

def main():
    print("="*70)
    print(" EXAMPLE: JSON Logging - Per-Turn Statistics")
    print("="*70)
    
    # Initialize memory system WITH JSON logging enabled
    json_output = "output/conversation_stats.json"
    
    print(f"\n📝 JSON logging enabled: {json_output}")
    print("   All turn statistics will be saved to this file\n")
    
    memory = MemorySystem(
        user_id="json_logging_example",
        enable_semantic_search=True,
        json_log_path=json_output  # Enable JSON logging
    )
    
    memory.clear_memories()
    
    # Simulate a conversation
    conversation = [
        "Hi, I'm Alex. I work at TechCorp as a senior engineer.",
        "I prefer Python for backend development.",
        "My manager's name is Jennifer Wilson.",
        "I'm allergic to peanuts, please remember that.",
        "I always need code reviews before deployment.",
        "When should you call me?",  # Query that retrieves memories
        "What's my manager's name again?",  # Another query
        "I also prefer dark mode in all my IDEs.",
        "Coffee is essential for my workflow.",
        "Never schedule meetings before 9 AM.",
    ]
    
    print("Processing conversation...")
    print("-" * 70)
    
    for i, message in enumerate(conversation, 1):
        print(f"\nTurn {i}: \"{message[:50]}...\"" if len(message) > 50 else f"\nTurn {i}: \"{message}\"")
        
        _, stats = memory.process_turn(message)
        
        print(f"  ✓ Extracted: {stats['extracted_count']}")
        print(f"  ✓ Retrieved: {stats['retrieved_count']}")
        print(f"  ✓ Processing: {stats['extraction_time_ms']:.1f}ms")
        print(f"  ✓ Retrieval: {stats['retrieval_time_ms']:.1f}ms")
    
    print("\n" + "="*70)
    print(" JSON Output Generated")
    print("="*70)
    
    # Read and display the JSON output
    with open(json_output, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 Summary:")
    print(f"  User ID: {data['user_id']}")
    print(f"  Total Turns: {data['total_turns']}")
    print(f"  Logged Turns: {len(data['conversation_turns'])}")
    
    # Show sample of first turn
    print(f"\n📝 Sample Turn Data (Turn 1):")
    print(json.dumps(data['conversation_turns'][0], indent=2))
    
    # Show statistics
    print(f"\n📈 Conversation Statistics:")
    total_extracted = sum(t['extracted_count'] for t in data['conversation_turns'])
    total_retrieved = sum(t['retrieved_count'] for t in data['conversation_turns'])
    avg_processing = sum(t['extraction_time_ms'] for t in data['conversation_turns']) / len(data['conversation_turns'])
    avg_retrieval = sum(t['retrieval_time_ms'] for t in data['conversation_turns']) / len(data['conversation_turns'])
    
    print(f"  Total Memories Extracted: {total_extracted}")
    print(f"  Total Memories Retrieved: {total_retrieved}")
    print(f"  Avg Processing Time: {avg_processing:.1f}ms")
    print(f"  Avg Retrieval Time: {avg_retrieval:.1f}ms")
    
    print(f"\n✅ Complete JSON log saved to: {json_output}")
    print(f"   You can use this for:")
    print(f"   • API responses")
    print(f"   • Performance monitoring")
    print(f"   • Debugging and analysis")
    print(f"   • Integration with other systems")
    
    # Show how to access the log programmatically
    print(f"\n💡 Programmatic Access:")
    print("   # Get all logged turns:")
    print("   stats_log = memory.get_turn_stats_log()")
    print(f"   # Returns: {len(memory.get_turn_stats_log())} turn stats")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
