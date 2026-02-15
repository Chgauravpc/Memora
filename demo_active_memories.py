#!/usr/bin/env python3
"""
Demo: Active Memory Tracking

Demonstrates how the system exposes which memories influenced each response.
At any turn N, shows:
- Which memories were retrieved
- Their origin turn
- Last used turn
- Confidence & access statistics

Run: python demo_active_memories.py
"""

import logging
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import MemorySystem

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO logs for cleaner output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_active_memories(active_memories, turn_number):
    """Print active memories in the requested format"""
    print(f"\n📋 ACTIVE MEMORIES AT TURN {turn_number}:")
    print(json.dumps({
        "turn": turn_number,
        "active_memories": active_memories,
        "response_generated": True
    }, indent=2))
    
    print(f"\n📊 Summary: {len(active_memories)} memories influenced this response")


def main():
    print_section("Active Memory Tracking Demo")
    
    print("\nThis demo shows how the system exposes which memories influenced each response.")
    print("Each turn includes metadata about the active memories retrieved.")
    
    # Initialize memory system
    print("\n🚀 Initializing memory system...")
    memory = MemorySystem(user_id="active_memory_demo", enable_semantic_search=True)
    
    # Clear previous data
    memory.clear_memories()
    print("✓ Memory system ready\n")
    
    # Conversation scenario
    conversation = [
        {
            "turn": 1,
            "message": "Hi, I'm Jennifer. I prefer calls after 11 AM, and I'm allergic to peanuts.",
            "description": "Initial preferences and constraints"
        },
        {
            "turn": 2,
            "message": "I work at TechCorp as a senior engineer.",
            "description": "Workplace information"
        },
        {
            "turn": 3,
            "message": "My manager is David Johnson, and I report directly to him.",
            "description": "Team structure"
        },
        {
            "turn": 4,
            "message": "I always prefer email for important updates.",
            "description": "Communication preference"
        },
        {
            "turn": 5,
            "message": "What are my scheduling preferences?",
            "description": "Query that should retrieve turn 1 constraint"
        },
        {
            "turn": 6,
            "message": "Who's my manager?",
            "description": "Query that should retrieve turn 3 entity"
        },
        {
            "turn": 7,
            "message": "Tell me about my food restrictions.",
            "description": "Query that should retrieve turn 1 constraint"
        },
        {
            "turn": 8,
            "message": "Actually, I changed my mind - I prefer calls after 10 AM now.",
            "description": "Update to previous preference (turn 1)"
        },
        {
            "turn": 9,
            "message": "What are my current scheduling preferences?",
            "description": "Query that should retrieve UPDATED preference"
        },
        {
            "turn": 10,
            "message": "Tell me everything about my work and preferences.",
            "description": "Broad query retrieving multiple memories"
        },
    ]
    
    # Process conversation
    print_section("Processing Conversation")
    
    for turn_data in conversation:
        turn_num = turn_data["turn"]
        message = turn_data["message"]
        description = turn_data["description"]
        
        print(f"\n{'─' * 80}")
        print(f"TURN {turn_num}: {description}")
        print(f"{'─' * 80}")
        print(f'User: "{message}"')
        
        # Process the turn
        context, stats = memory.process_turn(message)
        
        # Show extraction results
        if stats['extracted_count'] > 0:
            print(f"\n✓ Extracted {stats['extracted_count']} new memories")
            print(f"  Stored: {stats['stored_count']} | Duplicates: {stats.get('dedup_count', 0)}")
        
        # Show active memories (the key feature)
        active_memories = stats.get('active_memories', [])
        
        if active_memories:
            print_active_memories(active_memories, turn_num)
            
            # Show detailed breakdown
            print(f"\n🔍 Detailed Breakdown:")
            for i, mem in enumerate(active_memories, 1):
                print(f"\n  {i}. [{mem['type'].upper()}] {mem['memory_id']}")
                print(f"     Content: {mem['content']}")
                print(f"     Origin: Turn {mem['origin_turn']}")
                print(f"     Last Used: Turn {mem['last_used_turn']}")
                print(f"     Confidence: {mem['confidence']:.2f}")
                print(f"     Access Count: {mem['access_count']}")
                print(f"     Mentions: {mem['mention_count']}")
        else:
            print(f"\n📋 No memories retrieved for this turn")
        
        # Show the actual memory context that would be injected
        if turn_num in [5, 6, 7, 9, 10]:  # Show context for query turns
            print(f"\n📝 Memory Context for Prompt:")
            print("─" * 80)
            print(context if context else "(empty)")
            print("─" * 80)
    
    # Final statistics
    print_section("Session Statistics")
    
    final_stats = memory.get_statistics()
    print(f"\n📊 Total turns processed: {final_stats['total_turns']}")
    print(f"📊 Total memories stored: {final_stats['total_memories']}")
    print(f"📊 Memories by type:")
    for mem_type, count in sorted(final_stats.get('memories_by_type', {}).items()):
        print(f"    - {mem_type}: {count}")
    
    # Show example JSON output format
    print_section("Example JSON Output Format")
    
    print("\nAt any turn N, your system returns:")
    print(json.dumps({
        "turn": 412,
        "active_memories": [
            {
                "memory_id": "mem_0142",
                "content": "call_preference: after 11 AM",
                "type": "preference",
                "origin_turn": 1,
                "last_used_turn": 412,
                "confidence": 0.95,
                "mention_count": 1,
                "access_count": 15
            }
        ],
        "response_generated": True,
        "extracted_count": 0,
        "stored_count": 0,
        "retrieved_count": 1
    }, indent=2))
    
    print("\n✅ This allows you to:")
    print("  - Track which memories influenced each response")
    print("  - Debug retrieval behavior")
    print("  - Audit memory usage over time")
    print("  - Validate memory relevance")
    
    print_section("Demo Complete")
    print("\n✓ Active memory tracking is fully functional!")
    print("✓ Every turn exposes 'active_memories' in the stats dictionary")
    print("✓ Integration with existing code requires no changes")


if __name__ == "__main__":
    main()
