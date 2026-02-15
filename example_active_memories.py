#!/usr/bin/env python3
"""
Simple example showing active memory tracking output format.

Run: python example_active_memories.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import MemorySystem

# Initialize
memory = MemorySystem(user_id="example_user", enable_semantic_search=True)
memory.clear_memories()

# Turn 1: Add preference
print("Turn 1: Setting preference...")
_, stats = memory.process_turn("I prefer calls after 11 AM")
print(f"Extracted: {stats['extracted_count']}, Stored: {stats['stored_count']}\n")

# Turn 2-4: Add more context
print("Turn 2-4: Adding more information...")
memory.process_turn("I work at TechCorp as a senior engineer")
memory.process_turn("My manager is David Rodriguez")
memory.process_turn("I'm allergic to peanuts")
print()

# Turn 5: Query preference (memory should be retrieved)
print("Turn 5: Querying preference...")
_, stats = memory.process_turn("When should you call me?")

# Show the output format
output = {
    "turn": stats['turn_number'],
    "active_memories": stats.get('active_memories', []),
    "response_generated": True,
    "extracted_count": stats['extracted_count'],
    "stored_count": stats['stored_count'],
    "retrieved_count": stats['retrieved_count']
}

print("\nOutput at turn 5:")
print(json.dumps(output, indent=2))

print("\n✅ This is the format returned by process_turn() in the stats dictionary")
print(f"\n📊 The preference from turn 1 was retrieved at turn 5")
print(f"    Origin turn: {stats['active_memories'][0]['origin_turn']}")
print(f"    Last used: {stats['active_memories'][0]['last_used_turn']}")
print(f"    Access count: {stats['active_memories'][0]['access_count']}")
