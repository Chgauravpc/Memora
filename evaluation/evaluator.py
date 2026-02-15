"""
Main Evaluation Orchestrator
Integrates RAGAS and custom metrics for comprehensive evaluation
"""

import logging
import json
import time
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import MemorySystem
from .metrics import (
    ExtractionMetrics,
    RetrievalMetrics,
    DistanceSweepMetrics,
    ConsolidationMetrics,
)

logger = logging.getLogger(__name__)


class MemorySystemEvaluator:
    """
    Comprehensive evaluator for memory system (Phases 1-4).
    Uses RAGAS-style metrics for retrieval and custom metrics for memory-specific tests.
    """
    
    def __init__(self, memory_system: Optional[MemorySystem] = None):
        """
        Initialize evaluator.
        
        Args:
            memory_system: Optional pre-initialized memory system
        """
        self.memory_system = memory_system
        
        # Initialize metrics
        self.extraction_metrics = ExtractionMetrics()
        self.retrieval_metrics = RetrievalMetrics()
        self.distance_metrics = DistanceSweepMetrics()
        self.consolidation_metrics = ConsolidationMetrics()
    
    def evaluate_extraction(
        self,
        conversations: List[Dict],
        verbose: bool = False,
    ) -> Dict:
        """
        Evaluate extraction accuracy across conversations.
        
        Args:
            conversations: List of conversation dicts with ground truth
            verbose: Print detailed results
        
        Returns:
            Aggregated extraction metrics
        """
        if not self.memory_system:
            self.memory_system = MemorySystem(user_id="eval_extraction")
        
        all_results = []
        total_extracted = 0
        total_ground_truth = 0
        
        for conv in conversations:
            conv_id = conv['conversation_id']
            
            # Clear memories for this conversation
            self.memory_system.clear_memories()
            
            extracted_memories = []
            
            # Process each turn
            for turn in conv['turns']:
                message = turn['message']
                _, stats = self.memory_system.process_turn(message)
                
                # Collect extracted memories
                # Note: We need to get the actual memories from Redis
                if stats['stored_count'] > 0:
                    recent = self.memory_system.redis_store.get_recent_memories(limit=100)
                    extracted_memories.extend(recent)
            
            # Evaluate against ground truth
            ground_truth = conv['ground_truth_memories']
            
            metrics = self.extraction_metrics.evaluate(
                extracted_memories=extracted_memories,
                ground_truth_memories=ground_truth,
            )
            
            all_results.append({
                "conversation_id": conv_id,
                **metrics
            })
            
            total_extracted += metrics['total_extracted']
            total_ground_truth += metrics['total_ground_truth']
            
            if verbose:
                print(f"Conv {conv_id}: P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        # Aggregate results
        avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
        avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
        avg_f1 = sum(r['f1_score'] for r in all_results) / len(all_results)
        
        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1_score": avg_f1,
            "total_extracted": total_extracted,
            "total_ground_truth": total_ground_truth,
            "num_conversations": len(conversations),
            "per_conversation": all_results,
        }
    
    def evaluate_retrieval(
        self,
        conversations: List[Dict],
        verbose: bool = False,
    ) -> Dict:
        """
        Evaluate retrieval quality using RAGAS-style metrics.
        
        Args:
            conversations: List of conversations with ground truth
            verbose: Print detailed results
        
        Returns:
            Aggregated retrieval metrics
        """
        if not self.memory_system:
            self.memory_system = MemorySystem(user_id="eval_retrieval")
        
        all_results = []
        
        for conv in conversations:
            conv_id = conv['conversation_id']
            
            # Clear and populate memories
            self.memory_system.clear_memories()
            
            # Process all turns to populate memories
            for turn in conv['turns']:
                self.memory_system.process_turn(turn['message'])
            
            # Test retrieval with a query
            query = "Tell me everything you remember about this user"
            _, stats = self.memory_system.process_turn(query)
            
            retrieved_memories = stats.get('active_memories', [])
            ground_truth = conv['ground_truth_memories']
            
            # Calculate RAGAS-style metrics
            precision = self.retrieval_metrics.calculate_context_precision(
                retrieved_memories, ground_truth
            )
            recall = self.retrieval_metrics.calculate_context_recall(
                retrieved_memories, ground_truth
            )
            ranking = self.retrieval_metrics.calculate_ranking_quality(
                retrieved_memories, ground_truth
            )
            
            result = {
                "conversation_id": conv_id,
                "context_precision": precision,
                "context_recall": recall,
                "mrr": ranking['mrr'],
                "avg_position": ranking['avg_position'],
                "top_k": ranking['found_in_top_k'],
            }
            
            all_results.append(result)
            
            if verbose:
                print(f"Conv {conv_id}: Precision={precision:.3f}, "
                      f"Recall={recall:.3f}, MRR={ranking['mrr']:.3f}")
        
        # Aggregate
        return {
            "avg_context_precision": sum(r['context_precision'] for r in all_results) / len(all_results),
            "avg_context_recall": sum(r['context_recall'] for r in all_results) / len(all_results),
            "avg_mrr": sum(r['mrr'] for r in all_results) / len(all_results),
            "avg_position": sum(r['avg_position'] for r in all_results) / len(all_results),
            "num_conversations": len(conversations),
            "per_conversation": all_results,
        }
    
    def evaluate_distance_sweep(
        self,
        distance_sweep_conversation: Dict,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate long-term recall at different turn distances.
        Critical test: Can system remember info from 10, 50, 100, 500, 1000 turns ago?
        
        Args:
            distance_sweep_conversation: Special conversation with memories at target distances
            verbose: Print results
        
        Returns:
            Recall at each distance
        """
        if not self.memory_system:
            self.memory_system = MemorySystem(user_id="eval_distance_sweep")
        
        self.memory_system.clear_memories()
        
        # Process all turns
        for turn in distance_sweep_conversation['turns']:
            if not turn.get('is_query'):
                self.memory_system.process_turn(turn['message'])
        
        # Query at the end
        query_turn = distance_sweep_conversation['query_turn']
        query = "Tell me everything you remember about this user"
        _, stats = self.memory_system.process_turn(query)
        
        retrieved_memories = stats.get('active_memories', [])
        ground_truth = distance_sweep_conversation['ground_truth_memories']
        target_distances = distance_sweep_conversation['target_distances']
        
        # Evaluate recall at each distance
        results = self.distance_metrics.evaluate_recall_at_distance(
            retrieved_memories=retrieved_memories,
            ground_truth_memories=ground_truth,
            distances=target_distances,
            current_turn=query_turn,
        )
        
        if verbose:
            print("\n=== Distance Sweep Results ===")
            for distance, result in results.items():
                if result:
                    print(f"{distance}: Recall={result['recall']:.3f} "
                          f"({result['found']}/{result['expected']})")
                else:
                    print(f"{distance}: No data")
        
        return results
    
    def evaluate_consolidation(
        self,
        conversation: Dict,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate Phase 4 consolidation quality.
        
        Args:
            conversation: Conversation to test consolidation on
            verbose: Print results
        
        Returns:
            Consolidation quality metrics
        """
        if not self.memory_system:
            self.memory_system = MemorySystem(user_id="eval_consolidation")
        
        self.memory_system.clear_memories()
        
        # Process conversation
        for turn in conversation['turns']:
            self.memory_system.process_turn(turn['message'])
        
        # Get memories before consolidation
        memories_before = self.memory_system.redis_store.get_all_memories()
        
        # Run consolidation
        consolidation_stats = self.memory_system.run_consolidation(force=True)
        
        # Get memories after consolidation
        memories_after = self.memory_system.redis_store.get_all_memories()
        
        # Evaluate decay
        decay_metrics = self.consolidation_metrics.evaluate_decay(
            memories_before, memories_after
        )
        
        # Evaluate merging
        merge_metrics = self.consolidation_metrics.evaluate_merging(
            memories_before, memories_after
        )
        
        # Evaluate promotion
        promoted = [m for m in memories_after if m.get('promoted_to_core')]
        core_files = {
            "CORE.md": self.memory_system.flat_file_store.get_file_path("CORE.md").read_text(),
            "PREFERENCES.md": self.memory_system.flat_file_store.get_file_path("PREFERENCES.md").read_text(),
        }
        promotion_metrics = self.consolidation_metrics.evaluate_promotion(
            promoted, core_files
        )
        
        results = {
            "decay": decay_metrics,
            "merge": merge_metrics,
            "promotion": promotion_metrics,
            "consolidation_stats": consolidation_stats,
        }
        
        if verbose:
            print("\n=== Consolidation Results ===")
            print(f"Decay: {decay_metrics['decayed_count']} memories, "
                  f"{decay_metrics['deleted_count']} deleted")
            print(f"Merge: {merge_metrics['merged_count']} merges, "
                  f"{merge_metrics['reduction']} reduction")
            print(f"Promotion: {promotion_metrics['promoted_count']} promoted, "
                  f"{promotion_metrics['found_in_core_files']} in core files")
        
        return results
    
    def run_full_evaluation(
        self,
        test_conversations_file: str,
        distance_sweep_file: str,
        output_file: str = None,
    ) -> Dict:
        """
        Run complete evaluation suite.
        
        Args:
            test_conversations_file: Path to test conversations JSON
            distance_sweep_file: Path to distance sweep test JSON
            output_file: Optional path to save results
        
        Returns:
            Complete evaluation results
        """
        print("\n" + "="*70)
        print("  MEMORY SYSTEM EVALUATION - Phase 5")
        print("="*70)
        
        # Load test data
        print("\nLoading test data...")
        with open(test_conversations_file) as f:
            test_data = json.load(f)
            conversations = test_data['conversations']
        
        with open(distance_sweep_file) as f:
            distance_sweep = json.load(f)
        
        print(f"✓ Loaded {len(conversations)} test conversations")
        print(f"✓ Loaded distance sweep test")
        
        start_time = time.time()
        
        # 1. Extraction Evaluation
        print("\n--- Evaluating Extraction (Phases 1-3) ---")
        extraction_results = self.evaluate_extraction(
            conversations=conversations[:50],  # Use first 50 for speed
            verbose=False
        )
        print(f"Extraction - Precision: {extraction_results['avg_precision']:.3f}, "
              f"Recall: {extraction_results['avg_recall']:.3f}, "
              f"F1: {extraction_results['avg_f1_score']:.3f}")
        
        # 2. Retrieval Evaluation
        print("\n--- Evaluating Retrieval (Phase 2 + RAGAS metrics) ---")
        retrieval_results = self.evaluate_retrieval(
            conversations=conversations[:50],
            verbose=False
        )
        print(f"Retrieval - Precision: {retrieval_results['avg_context_precision']:.3f}, "
              f"Recall: {retrieval_results['avg_context_recall']:.3f}, "
              f"MRR: {retrieval_results['avg_mrr']:.3f}")
        
        # 3. Distance Sweep
        print("\n--- Evaluating Long-Term Recall (1000+ turns) ---")
        distance_results = self.evaluate_distance_sweep(
            distance_sweep_conversation=distance_sweep,
            verbose=True
        )
        
        # 4. Consolidation
        print("\n--- Evaluating Consolidation (Phase 4) ---")
        consolidation_results = self.evaluate_consolidation(
            conversation=conversations[0],
            verbose=True
        )
        
        duration = time.time() - start_time
        
        # Compile results
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "test_data": {
                "num_conversations": len(conversations),
                "total_turns": sum(c['num_turns'] for c in conversations),
                "total_ground_truth_memories": sum(c['total_memories'] for c in conversations),
            },
            "extraction": extraction_results,
            "retrieval": retrieval_results,
            "distance_sweep": distance_results,
            "consolidation": consolidation_results,
        }
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to {output_file}")
        
        print("\n" + "="*70)
        print(f"  EVALUATION COMPLETE ({duration:.1f}s)")
        print("="*70)
        
        return results
