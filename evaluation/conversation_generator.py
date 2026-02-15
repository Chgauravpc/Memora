"""
Synthetic Conversation Generator
Generates realistic conversations with extractable memories for testing
"""

import json
import random
from typing import Dict, List, Tuple
from pathlib import Path

class ConversationGenerator:
    """Generate synthetic conversations with known ground truth memories"""
    
    # Natural conversation templates - more realistic and conversational
    # Format: message with clear memory markers
    PREFERENCE_TEMPLATES = [
        "Oh, I definitely prefer {item} over {alternative}. Much better for my workflow.",
        "I've always been a {item} person, to be honest. Can't imagine working without it.",
        "Yeah, I really can't stand {item}. I avoid it whenever possible.",
        "My absolute favorite {category} has to be {item}. I use it every single day.",
        "I always use {item} for {purpose}. It just works better for me.",
    ]
    
    CONSTRAINT_TEMPLATES = [
        "Just a heads up - I can't eat {item} due to my {reason}. Been dealing with that for years.",
        "Oh no, I'm severely allergic to {item}. I have to be really careful about that.",
        "I absolutely must {action} before {event}. It's non-negotiable for me.",
        "Never {action} when {condition}. That's a hard rule I follow.",
        "Please don't call me before {time}. I'm really not functional in the early mornings.",
    ]
    
    ENTITY_TEMPLATES = [
        "By the way, my name is {name} - that's important for the records!",
        "I work at {company} as a {role}. Been there for about two years now, really enjoy it.",
        "My manager {name} specifically asked me to handle this. She trusts me with these projects.",
        "I'm based in {location}, so I'm on {timezone} time. Keep that in mind for scheduling!",
        "I'm currently working on {project}. It's a major initiative for our team this quarter.",
    ]
    
    COMMITMENT_TEMPLATES = [
        "I'll absolutely make sure to {action} by {deadline}. You have my word on that, I'm very reliable with these things.",
        "I promise I'll finish {task} before {date}. Setting a reminder right now - this is really important to me.",
        "I need to complete {task} before {date}. That's my top priority this week, and I'll make it happen.",
        "The deadline for {project} is {date}, so I need to wrap this up soon. I'm fully committed to delivering on time.",
    ]
    
    INSTRUCTION_TEMPLATES = [
        "Just so you know, I always {action} when {condition}. It's become a habit.",
        "Oh, before I forget - remember to {action}. That's super important.",
        "Make sure you {action}. I learned that the hard way!",
        "Never forget to {action}. Trust me, it'll save you headaches later.",
    ]
    
    # Natural filler messages (conversational but no extractable memories)
    FILLER_MESSAGES = [
        "Hi there!", "Hello!", "Hey, how's it going?", "Good morning!",
        "Thanks so much!", "I really appreciate that.", "That's helpful, thank you.",
        "OK, got it.", "Perfect, that works for me.", "Great, sounds good!",
        "Sure, no problem.", "Absolutely, I can do that.", "I see what you mean.",
        "That makes total sense.", "Understood, thanks for clarifying.",
        "Alright, let me check that.", "One moment please.", "Let me look into that.",
        "Got it, I'll take care of that.", "Cool, thanks for the update.",
        "Nice, that should work well.", "Interesting, tell me more.",
        "What's the latest on that?", "How are things progressing?",
        "Sounds like a plan!", "I'm all set then.", "No worries at all.",
    ]
    
    # Data for template filling - more realistic and diverse
    ITEMS = ["Python", "JavaScript", "TypeScript", "dark mode", "light mode", "VS Code", 
             "morning meetings", "async communication", "pair programming", "code reviews",
             "Slack", "email", "video calls", "remote work", "office work"]
    ALTERNATIVES = ["Java", "C++", "dark theme", "light theme", "IntelliJ", "afternoon meetings"]
    CATEGORIES = ["programming language", "IDE", "communication tool", "work style", "editor theme"]
    PURPOSES = ["development", "debugging", "testing", "documentation", "collaboration"]
    COMPANIES = ["TechCorp", "DataSystems", "CloudInc", "AILabs", "DevCo", "StartupHub"]
    ROLES = ["software engineer", "senior developer", "data scientist", "product manager", 
             "engineering manager", "tech lead", "UX designer"]
    NAMES = ["Alex Chen", "Sarah Johnson", "Michael Park", "Emily Rodriguez", "David Kim",
             "Jennifer Martinez", "Robert Taylor", "Lisa Anderson", "James Wilson", "Maria Garcia"]
    LOCATIONS = ["San Francisco", "New York", "Seattle", "Austin", "Boston", "Portland", 
                 "Denver", "Chicago", "Los Angeles"]
    TIMEZONES = ["PST", "EST", "CST", "MST"]
    PROJECTS = ["API refactor", "dashboard redesign", "ML pipeline upgrade", "mobile app v2", 
                "database migration", "authentication system", "analytics platform"]
    ALLERGIES = ["gluten", "peanuts", "dairy products", "shellfish", "tree nuts"]
    REASONS = ["allergies", "dietary restrictions", "health issues", "personal preference"]
    TIMES = ["8 AM", "9 AM", "10 AM", "7 PM", "8 PM"]
    DATES = ["Friday", "end of the week", "March 15th", "next Monday", "end of month"]
    ACTIONS = ["review the code", "run the tests", "update the docs", "check the logs", 
               "backup the database", "deploy to staging", "merge the PR"]
    TASKS = ["the report", "the presentation", "code review", "testing", "documentation"]
    CONDITIONS = ["deploying to production", "pushing to main", "releasing", "before standup"]
    EVENTS = ["deployment", "the meeting", "code review", "standup", "the deadline"]
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility"""
        random.seed(seed)
        self.conversations = []
    
    def generate_conversation(
        self, 
        conversation_id: int,
        num_turns: int = 20,
        memory_density: float = 0.6,
    ) -> Dict:
        """
        Generate a single conversation with ground truth memories.
        
        Args:
            conversation_id: Unique ID for this conversation
            num_turns: Number of conversation turns
            memory_density: Probability that a turn contains extractable info (0-1)
        
        Returns:
            Dictionary with conversation, memories, and metadata
        """
        turns = []
        ground_truth_memories = []
        
        # Generate persona for consistency
        persona = self._generate_persona()
        
        for turn_num in range(1, num_turns + 1):
            # Decide if this turn should have memory
            if random.random() < memory_density:
                # Generate memory-containing turn
                message, memory = self._generate_memory_turn(persona, turn_num)
                turns.append({
                    "turn": turn_num,
                    "message": message,
                    "has_memory": True,
                })
                ground_truth_memories.append(memory)
            else:
                # Generate filler turn
                message = random.choice(self.FILLER_MESSAGES)
                turns.append({
                    "turn": turn_num,
                    "message": message,
                    "has_memory": False,
                })
        
        return {
            "conversation_id": conversation_id,
            "persona": persona,
            "num_turns": num_turns,
            "turns": turns,
            "ground_truth_memories": ground_truth_memories,
            "total_memories": len(ground_truth_memories),
        }
    
    def _generate_persona(self) -> Dict:
        """Generate a consistent persona for the conversation"""
        return {
            "name": random.choice(self.NAMES),
            "company": random.choice(self.COMPANIES),
            "role": random.choice(self.ROLES),
            "location": random.choice(self.LOCATIONS),
            "preferred_language": random.choice(["Python", "JavaScript", "Java", "Go"]),
        }
    
    def _generate_memory_turn(self, persona: Dict, turn_num: int) -> Tuple[str, Dict]:
        """
        Generate a turn with extractable memory - natural conversation style.
        
        Returns:
            (message, ground_truth_memory)
        """
        # Choose memory type
        memory_type = random.choice(["preference", "constraint", "entity", 
                                     "commitment", "instruction"])
        
        if memory_type == "preference":
            template = random.choice(self.PREFERENCE_TEMPLATES)
            item = random.choice(self.ITEMS)
            message = template.format(
                item=item,
                alternative=random.choice([i for i in self.ALTERNATIVES if i != item]),
                category=random.choice(self.CATEGORIES),
                purpose=random.choice(self.PURPOSES)
            )
            key = "preference"
            value = f"prefers {item}"
            
        elif memory_type == "constraint":
            template = random.choice(self.CONSTRAINT_TEMPLATES)
            item_or_action = random.choice(self.ALLERGIES + self.ACTIONS)
            message = template.format(
                item=random.choice(self.ALLERGIES),
                reason=random.choice(self.REASONS),
                action=random.choice(self.ACTIONS),
                event=random.choice(self.EVENTS),
                time=random.choice(self.TIMES),
                condition=random.choice(self.CONDITIONS)
            )
            key = "constraint"
            # Extract the key constraint from the message
            if "can't eat" in message or "allergic" in message:
                value = f"allergic to {[a for a in self.ALLERGIES if a in message.lower()][0] if any(a in message.lower() for a in self.ALLERGIES) else 'unknown'}"
            elif "must" in message:
                value = message.split("must")[1].strip() if "must" in message else message.lower()
            elif "don't call" in message.lower():
                value = f"no calls before {[t for t in self.TIMES if t in message][0] if any(t in message for t in self.TIMES) else '9 AM'}"
            else:
                value = message.lower()
            
        elif memory_type == "entity":
            template = random.choice(self.ENTITY_TEMPLATES)
            message = template.format(
                name=persona["name"],
                company=persona["company"],
                role=persona["role"],
                location=persona["location"],
                timezone=random.choice(self.TIMEZONES),
                project=random.choice(self.PROJECTS),
            )
            key = "entity"
            if "work at" in message:
                value = f"{persona['company']} - {persona['role']}"
            elif "based in" in message:
                value = persona["location"]
            elif "manager" in message.lower():
                value = [n for n in self.NAMES if n in message][0] if any(n in message for n in self.NAMES) else "manager"
            else:
                value = persona["name"]
            
        elif memory_type == "commitment":
            template = random.choice(self.COMMITMENT_TEMPLATES)
            task = random.choice(self.TASKS)
            deadline = random.choice(self.DATES)
            message = template.format(
                action=random.choice(self.ACTIONS),
                deadline=deadline,
                task=task,
                date=deadline,
                project=random.choice(self.PROJECTS),
            )
            key = "commitment"
            value = f"{task} by {deadline}"
            
        else:  # instruction
            template = random.choice(self.INSTRUCTION_TEMPLATES)
            action = random.choice(self.ACTIONS)
            message = template.format(
                action=action,
                condition=random.choice(self.CONDITIONS)
            )
            key = "instruction"
            value = f"always {action}"
        
        ground_truth = {
            "turn": turn_num,
            "type": memory_type,
            "key": key,
            "value": value,
            "message": message,
            "confidence": 0.8 + random.random() * 0.2,  # 0.8-1.0
        }
        
        return message, ground_truth
    
    def generate_batch(
        self, 
        num_conversations: int = 200,
        turns_per_conversation: Tuple[int, int] = (10, 30),
        output_file: str = None,
    ) -> List[Dict]:
        """
        Generate batch of conversations.
        
        Args:
            num_conversations: Number of conversations to generate
            turns_per_conversation: Min and max turns per conversation
            output_file: Optional path to save JSON file
        
        Returns:
            List of conversation dictionaries
        """
        conversations = []
        
        for i in range(num_conversations):
            num_turns = random.randint(*turns_per_conversation)
            conversation = self.generate_conversation(
                conversation_id=i + 1,
                num_turns=num_turns,
                memory_density=0.6,
            )
            conversations.append(conversation)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "num_conversations": num_conversations,
                    "total_turns": sum(c["num_turns"] for c in conversations),
                    "total_memories": sum(c["total_memories"] for c in conversations),
                    "conversations": conversations,
                }, f, indent=2)
            print(f"Generated {num_conversations} conversations -> {output_file}")
        
        return conversations
    
    def generate_distance_sweep_conversation(
        self, 
        target_distances: List[int] = [10, 50, 100, 500, 1000]
    ) -> Dict:
        """
        Generate a conversation specifically for distance sweep testing.
        Places key memories at specific turn distances FROM THE QUERY.
        
        Args:
            target_distances: List of turn distances to test (e.g., [10, 50, 100])
        
        Returns:
            Conversation with memories at specific distances before the query
        """
        # Query will be at max_distance + 10
        max_distance = max(target_distances)
        query_turn = max_distance + 10
        turns = []
        ground_truth_memories = []
        
        # Generate persona
        persona = self._generate_persona()
        
        # Place memories at target distances BEFORE the query
        for distance in target_distances:
            # Memory should be at query_turn - distance
            memory_turn = query_turn - distance
            message, memory = self._generate_memory_turn(persona, distance)
            memory["test_distance"] = distance
            memory["turn"] = memory_turn
            turns.append({
                "turn": memory_turn,
                "message": message,
                "has_memory": True,
                "is_test_marker": True,
            })
            ground_truth_memories.append(memory)
        
        # Fill remaining turns with filler
        for turn_num in range(1, query_turn):
            # Skip turns that already have memories
            if not any(t['turn'] == turn_num for t in turns):
                turns.append({
                    "turn": turn_num,
                    "message": random.choice(self.FILLER_MESSAGES),
                    "has_memory": False,
                })
        
        # Sort by turn number
        turns.sort(key=lambda t: t["turn"])
        
        # Add final query turn
        turns.append({
            "turn": query_turn,
            "message": "Tell me everything you remember about me",
            "has_memory": False,
            "is_query": True,
        })
        
        return {
            "conversation_id": "distance_sweep",
            "persona": persona,
            "num_turns": query_turn,
            "turns": turns,
            "ground_truth_memories": ground_truth_memories,
            "target_distances": target_distances,
            "query_turn": query_turn,
        }


def main():
    """Generate test conversations"""
    generator = ConversationGenerator(seed=42)
    
    # Generate 200 conversations
    print("Generating 200 test conversations...")
    conversations = generator.generate_batch(
        num_conversations=200,
        turns_per_conversation=(10, 30),
        output_file="evaluation/fixtures/test_conversations.json"
    )
    
    # Generate distance sweep test
    print("\nGenerating distance sweep test conversation...")
    distance_sweep = generator.generate_distance_sweep_conversation(
        target_distances=[10, 50, 100, 500, 1000]
    )
    
    with open("evaluation/fixtures/distance_sweep_test.json", 'w') as f:
        json.dump(distance_sweep, f, indent=2)
    
    print(f"\n✓ Generated {len(conversations)} conversations")
    print(f"✓ Total turns: {sum(c['num_turns'] for c in conversations)}")
    print(f"✓ Total memories: {sum(c['total_memories'] for c in conversations)}")
    print(f"✓ Distance sweep test ready")


if __name__ == "__main__":
    main()
