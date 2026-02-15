#!/usr/bin/env python3
"""
SAMPLE: Test Memory Extraction from Customer Service Conversation

This demonstrates how the memory system extracts information from a realistic
customer service conversation between an agent and Jennifer Martinez.

The system should extract:
- Entities: Names, company, dates, amounts
- Events: Payments, transactions
- Facts: Account numbers, payment methods

Also demonstrates active memory tracking showing which memories influenced each response.

Run: python test_customer_conversation.py
"""

import logging
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import MemorySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_customer_conversation():
    """Test memory extraction from a customer service conversation"""
    
    print("\n" + "="*80)
    print("  CUSTOMER SERVICE CONVERSATION - MEMORY EXTRACTION TEST")
    print("="*80)
    
    # Initialize memory system for Jennifer Martinez
    print("\nInitializing memory system for Jennifer Martinez...")
    memory = MemorySystem(user_id="jennifer_martinez", enable_semantic_search=True)
    
    # Clear previous data
    memory.clear_memories()
    
    # Customer service conversation turns - 60 turn conversation
    conversation = [
        {"speaker": "AGENT", "message": "Good evening, this is David calling from National Payment Services. May I speak with Jennifer Martinez?"},
        {"speaker": "USER", "message": "Speaking. Who is this?"},
        {"speaker": "AGENT", "message": "Ms. Martinez, I'm calling regarding your account ending in 3847. We show a pending balance on your account."},
        {"speaker": "USER", "message": "A payment reminder? I already paid that bill three days ago!"},
        {"speaker": "AGENT", "message": "I apologize for the confusion. Let me verify your account. Can you confirm the last four digits of your account number?"},
        {"speaker": "USER", "message": "It's 3847."},
        {"speaker": "AGENT", "message": "Thank you. Let me pull up your payment history. One moment please."},
        {"speaker": "USER", "message": "Sure."},
        {"speaker": "AGENT", "message": "Ms. Martinez, I can see you're correct. There is a payment of $650 that was submitted on February 7th."},
        {"speaker": "USER", "message": "So my payment went through then?"},
        {"speaker": "AGENT", "message": "Yes, it has been processed successfully. The payment is showing as cleared in our system now."},
        {"speaker": "USER", "message": "Okay, good. I was worried something went wrong with my bank."},
        {"speaker": "AGENT", "message": "Your payment method was a bank transfer, correct?"},
        {"speaker": "USER", "message": "Yes, that's right."},
        {"speaker": "AGENT", "message": "Everything looks good. The funds were received and your account is current."},
        {"speaker": "USER", "message": "Alright, that's a relief. Thank you for checking."},
        {"speaker": "AGENT", "message": "You're welcome. Is there anything else I can help you with today?"},
        {"speaker": "USER", "message": "Actually, yes. When is my next payment due?"},
        {"speaker": "AGENT", "message": "Your next payment of $650 is due on March 7th, 2026."},
        {"speaker": "USER", "message": "Can I set up automatic payments?"},
        {"speaker": "AGENT", "message": "Absolutely! We can set that up right now. Would you like to use the same bank account?"},
        {"speaker": "USER", "message": "Yes, the same account ending in 2891."},
        {"speaker": "AGENT", "message": "Perfect. I'll set up automatic payments from your Bank of America checking account ending in 2891."},
        {"speaker": "USER", "message": "When will the automatic payments start?"},
        {"speaker": "AGENT", "message": "They'll begin with your March 7th payment. You'll receive a confirmation email within 24 hours."},
        {"speaker": "USER", "message": "Great. What email address do you have on file?"},
        {"speaker": "AGENT", "message": "I show jennifer.martinez@email.com. Is that correct?"},
        {"speaker": "USER", "message": "Yes, that's my email."},
        {"speaker": "AGENT", "message": "Excellent. Is this phone number, 555-0142, still the best way to reach you?"},
        {"speaker": "USER", "message": "Yes, but I prefer text messages over calls for reminders."},
        {"speaker": "AGENT", "message": "I've noted your preference for text message notifications. I'll update that in your account."},
        {"speaker": "USER", "message": "Thank you. What time do you usually send reminders?"},
        {"speaker": "AGENT", "message": "Payment reminders are typically sent 5 days before the due date, around 10 AM Pacific Time."},
        {"speaker": "USER", "message": "That works for me. Can you also tell me my current balance?"},
        {"speaker": "AGENT", "message": "Your current balance is $0. Your account is paid in full until the March payment."},
        {"speaker": "USER", "message": "Perfect. One more thing - what's my total loan amount?"},
        {"speaker": "AGENT", "message": "Your original loan amount was $15,600 with a 3-year term. You have 18 months remaining."},
        {"speaker": "USER", "message": "So I'm halfway through?"},
        {"speaker": "AGENT", "message": "That's correct. You've made 18 payments and have 18 remaining."},
        {"speaker": "USER", "message": "What's my interest rate?"},
        {"speaker": "AGENT", "message": "Your interest rate is 5.9% APR, which is fixed for the entire loan term."},
        {"speaker": "USER", "message": "Can I pay extra to finish early?"},
        {"speaker": "AGENT", "message": "Yes, there are no prepayment penalties. You can pay additional amounts anytime."},
        {"speaker": "USER", "message": "How would I do that?"},
        {"speaker": "AGENT", "message": "You can make additional payments through our website, mobile app, or by calling us."},
        {"speaker": "USER", "message": "I'll keep that in mind. Do you offer payment extensions if needed?"},
        {"speaker": "AGENT", "message": "Yes, we offer extensions in certain situations. You would need to call us at least 3 days before your due date."},
        {"speaker": "USER", "message": "Good to know. What documentation would I need for that?"},
        {"speaker": "AGENT", "message": "Typically proof of hardship, like a medical bill or job loss documentation, but policies vary by situation."},
        {"speaker": "USER", "message": "Understood. Is there a fee for extensions?"},
        {"speaker": "AGENT", "message": "There may be a $25 processing fee, but we waive it for first-time requests."},
        {"speaker": "USER", "message": "That's fair. What about if I miss a payment?"},
        {"speaker": "AGENT", "message": "There's a $35 late fee if payment is more than 10 days overdue, and it may affect your credit score."},
        {"speaker": "USER", "message": "I'll make sure to avoid that with the automatic payments."},
        {"speaker": "AGENT", "message": "That's a smart decision. The automatic payments will help you stay on track."},
        {"speaker": "USER", "message": "Can I change the payment date if needed?"},
        {"speaker": "AGENT", "message": "Yes, you can request a payment date change once per year. We can adjust it to any date between the 1st and 28th."},
        {"speaker": "USER", "message": "The 7th works well for me right after payday."},
        {"speaker": "AGENT", "message": "That's great planning. Many customers align their payments with their payday."},
        {"speaker": "USER", "message": "Do you send statements monthly?"},
    ]
    
    print("\n" + "="*80)
    print("  Processing conversation turns...")
    print("="*80)
    
    total_extracted = 0
    total_stored = 0
    stage3_count = 0
    
    start_time = time.time()
    
    for i, turn in enumerate(conversation, 1):
        speaker = turn["speaker"]
        message = turn["message"]
        
        print(f"\n--- Turn {i} ({speaker}) ---")
        print(f'"{message}"')
        
        # Process the turn
        turn_start = time.time()
        memory_context, stats = memory.process_turn(
            user_message=message
        )
        turn_time_ms = (time.time() - turn_start) * 1000
        
        extracted = stats.get('extracted_count', 0)
        stored = stats.get('stored_count', 0)
        
        total_extracted += extracted
        total_stored += stored
        
        if extracted > 0:
            print(f"\n> EXTRACTED: {extracted} memories")
            stage3_count += 1
        else:
            print(f"\n> No memories extracted")
        
        # Show active memories that influenced this turn
        active_memories = stats.get('active_memories', [])
        if active_memories:
            print(f"\nACTIVE MEMORIES (Retrieved {len(active_memories)}):")
            output = {
                "turn": stats['turn_number'],
                "active_memories": active_memories,
                "response_generated": True
            }
            print(json.dumps(output, indent=2))
        
        print(f"\n> Processing time: {turn_time_ms:.2f}ms")
        print(f"  - Extraction: {stats.get('extraction_time_ms', 0):.2f}ms")
        print(f"  - Storage: {stats.get('storage_time_ms', 0):.2f}ms")
        print(f"  - Retrieval: {stats.get('retrieval_time_ms', 0):.2f}ms")
        print("-" * 80)
    
    total_time_ms = (time.time() - start_time) * 1000
    
    # Show all extracted memories
    print("\n" + "="*80)
    print("  EXTRACTED MEMORIES FROM CONVERSATION")
    print("="*80)
    
    all_memories = memory.redis_store.get_recent_memories(limit=100)
    
    if all_memories:
        print(f"\nTotal memories stored: {len(all_memories)}\n")
        
        # Group by type
        by_type = {}
        for mem in all_memories:
            mem_type = mem['type']
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(mem)
        
        for mem_type, mems in sorted(by_type.items()):
            print(f"\n[{mem_type.upper()}] ({len(mems)} memories):")
            for mem in mems:
                print(f"  - {mem['key']}: {mem['value']}")
                print(f"    Confidence: {mem['confidence']:.2f}, Turn: {mem['turn_number']}")
                print(f"    Source: \"{mem['source_text'][:60]}...\"")
    else:
        print("\nNo memories were extracted from this conversation.")
    
    # Performance metrics
    print("\n" + "="*80)
    print("  PERFORMANCE METRICS")
    print("="*80)
    
    print(f"\nTotal conversation turns: {len(conversation)}")
    print(f"Total memories extracted: {total_extracted}")
    print(f"Total memories stored: {total_stored}")
    print(f"Turns with extractions: {stage3_count}")
    print(f"Total processing time: {total_time_ms:.2f}ms")
    print(f"Average time per turn: {total_time_ms/len(conversation):.2f}ms")
    
    # LLM stats if available
    extractor = memory.extractor
    if hasattr(extractor, 'llm_extractor') and extractor.llm_extractor:
        llm = extractor.llm_extractor
        if llm.api_call_count > 0:
            avg_llm_response = llm.total_response_time_ms / llm.api_call_count
            print(f"\nLLM API Calls: {llm.api_call_count}")
            print(f"Average LLM response time: {avg_llm_response:.2f}ms")
            print(f"Total LLM response time: {llm.total_response_time_ms:.2f}ms")
    
    print("\n" + "="*80)
    print("  TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_customer_conversation()
