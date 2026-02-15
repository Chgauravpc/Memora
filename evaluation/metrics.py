"""
Evaluation Metrics for Memory System
"""

import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExtractionMetrics:
    """Metrics for extraction accuracy (Phases 1-3)"""
    
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
    
    def evaluate(
        self, 
        extracted_memories: List[Dict], 
        ground_truth_memories: List[Dict],
        match_threshold: float = 0.7,
    ) -> Dict:
        """
        Calculate precision, recall, F1 for extraction.
        
        Args:
            extracted_memories: Memories extracted by system
            ground_truth_memories: Known correct memories
            match_threshold: Similarity threshold for matching
        
        Returns:
            Dictionary with precision, recall, F1, and details
        """
        # Match extracted to ground truth
        matched_gt = set()
        matched_ext = set()
        
        for i, extracted in enumerate(extracted_memories):
            for j, gt in enumerate(ground_truth_memories):
                if j in matched_gt:
                    continue
                
                # Check if they match
                if self._memories_match(extracted, gt, match_threshold):
                    matched_gt.add(j)
                    matched_ext.add(i)
                    self.true_positives += 1
                    break
        
        # False positives: extracted but not in ground truth
        self.false_positives = len(extracted_memories) - len(matched_ext)
        
        # False negatives: in ground truth but not extracted
        self.false_negatives = len(ground_truth_memories) - len(matched_gt)
        
        # Calculate metrics
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_extracted": len(extracted_memories),
            "total_ground_truth": len(ground_truth_memories),
        }
    
    def _memories_match(self, mem1: Dict, mem2: Dict, threshold: float) -> bool:
        """Check if two memories match"""
        # Type must match
        if mem1.get('type') != mem2.get('type'):
            return False
        
        # Check value similarity (simple contains check)
        val1 = mem1.get('value', '').lower()
        val2 = mem2.get('value', '').lower()
        
        if val1 in val2 or val2 in val1:
            return True
        
        # Check key similarity
        key1 = mem1.get('key', '').lower()
        key2 = mem2.get('key', '').lower()
        
        if key1 and key2 and (key1 in key2 or key2 in key1):
            return True
        
        return False


class RetrievalMetrics:
    """Metrics for retrieval quality using RAGAS-style evaluation"""
    
    def calculate_context_precision(
        self,
        retrieved_memories: List[Dict],
        ground_truth_memories: List[Dict],
    ) -> float:
        """
        Context Precision: Proportion of retrieved memories that are relevant.
        Similar to RAGAS context_precision.
        
        Args:
            retrieved_memories: Memories retrieved by system
            ground_truth_memories: Memories that should be retrieved
        
        Returns:
            Precision score (0-1)
        """
        if not retrieved_memories:
            return 0.0
        
        relevant_count = 0
        
        for retrieved in retrieved_memories:
            # Check if this retrieved memory matches any ground truth
            for gt in ground_truth_memories:
                if self._memories_similar(retrieved, gt):
                    relevant_count += 1
                    break
        
        return relevant_count / len(retrieved_memories)
    
    def calculate_context_recall(
        self,
        retrieved_memories: List[Dict],
        ground_truth_memories: List[Dict],
    ) -> float:
        """
        Context Recall: Proportion of ground truth memories that were retrieved.
        Similar to RAGAS context_recall.
        
        Args:
            retrieved_memories: Memories retrieved by system
            ground_truth_memories: Memories that should be retrieved
        
        Returns:
            Recall score (0-1)
        """
        if not ground_truth_memories:
            return 1.0
        
        retrieved_count = 0
        
        for gt in ground_truth_memories:
            # Check if this ground truth was retrieved
            for retrieved in retrieved_memories:
                if self._memories_similar(retrieved, gt):
                    retrieved_count += 1
                    break
        
        return retrieved_count / len(ground_truth_memories)
    
    def calculate_ranking_quality(
        self,
        retrieved_memories: List[Dict],
        ground_truth_memories: List[Dict],
    ) -> Dict:
        """
        Evaluate ranking quality (are most relevant memories ranked higher?)
        
        Returns:
            Dictionary with MRR, NDCG, and position stats
        """
        if not retrieved_memories or not ground_truth_memories:
            return {"mrr": 0.0, "avg_position": 0.0, "found_in_top_k": {}}
        
        # Mean Reciprocal Rank (MRR)
        first_relevant_position = None
        for i, retrieved in enumerate(retrieved_memories, 1):
            for gt in ground_truth_memories:
                if self._memories_similar(retrieved, gt):
                    first_relevant_position = i
                    break
            if first_relevant_position:
                break
        
        mrr = 1.0 / first_relevant_position if first_relevant_position else 0.0
        
        # Average position of relevant memories
        positions = []
        for gt in ground_truth_memories:
            for i, retrieved in enumerate(retrieved_memories, 1):
                if self._memories_similar(retrieved, gt):
                    positions.append(i)
                    break
        
        avg_position = sum(positions) / len(positions) if positions else 0.0
        
        # Found in top K
        top_k_stats = {}
        for k in [1, 3, 5, 10]:
            top_k = retrieved_memories[:k]
            found = sum(1 for gt in ground_truth_memories 
                       if any(self._memories_similar(r, gt) for r in top_k))
            top_k_stats[f"top_{k}"] = found / len(ground_truth_memories)
        
        return {
            "mrr": mrr,
            "avg_position": avg_position,
            "found_in_top_k": top_k_stats,
        }
    
    def _memories_similar(self, mem1: Dict, mem2: Dict) -> bool:
        """Check if two memories are similar enough to be considered a match"""
        # Type must match
        if mem1.get('type') != mem2.get('type'):
            return False
        
        # Check content similarity
        content1 = f"{mem1.get('key', '')} {mem1.get('value', '')}".lower()
        content2 = f"{mem2.get('key', '')} {mem2.get('value', '')}".lower()
        
        # Simple substring matching (can be improved with embeddings)
        if content1 in content2 or content2 in content1:
            return True
        
        return False


class DistanceSweepMetrics:
    """Metrics for long-term recall (1000+ turns)"""
    
    def evaluate_recall_at_distance(
        self,
        retrieved_memories: List[Dict],
        ground_truth_memories: List[Dict],
        distances: List[int],
        current_turn: int,
    ) -> Dict:
        """
        Evaluate recall at different turn distances.
        
        Args:
            retrieved_memories: Memories retrieved at query time
            ground_truth_memories: Ground truth with turn numbers
            distances: List of distances to test (e.g., [10, 50, 100, 500, 1000])
            current_turn: Current turn number
        
        Returns:
            Dictionary with recall at each distance
        """
        results = {}
        
        for distance in distances:
            target_turn = current_turn - distance
            
            # Find ground truth memories from that distance
            gt_at_distance = [
                m for m in ground_truth_memories 
                if abs(m.get('turn', 0) - target_turn) <= 2  # ±2 turn tolerance
            ]
            
            if not gt_at_distance:
                results[f"distance_{distance}"] = None
                continue
            
            # Check how many were retrieved
            retrieved_count = 0
            for gt in gt_at_distance:
                for retrieved in retrieved_memories:
                    if self._memories_match(retrieved, gt):
                        retrieved_count += 1
                        break
            
            recall = retrieved_count / len(gt_at_distance)
            results[f"distance_{distance}"] = {
                "recall": recall,
                "found": retrieved_count,
                "expected": len(gt_at_distance),
            }
        
        return results
    
    def _memories_match(self, mem1: Dict, mem2: Dict) -> bool:
        """Check if memories match"""
        if mem1.get('type') != mem2.get('type'):
            return False
        
        content1 = f"{mem1.get('key', '')} {mem1.get('value', '')}".lower()
        content2 = f"{mem2.get('key', '')} {mem2.get('value', '')}".lower()
        
        return content1 in content2 or content2 in content1


class ConsolidationMetrics:
    """Metrics for Phase 4 consolidation quality"""
    
    def evaluate_decay(
        self,
        memories_before: List[Dict],
        memories_after: List[Dict],
    ) -> Dict:
        """
        Evaluate decay appropriateness.
        
        Returns:
            Statistics about decay application
        """
        decayed_count = 0
        deleted_count = 0
        total_decay_applied = 0.0
        
        # Build lookup for after memories
        after_lookup = {m['memory_id']: m for m in memories_after}
        
        for mem_before in memories_before:
            mem_id = mem_before['memory_id']
            
            if mem_id not in after_lookup:
                deleted_count += 1
            else:
                mem_after = after_lookup[mem_id]
                conf_before = float(mem_before.get('confidence', 0.5))
                conf_after = float(mem_after.get('confidence', 0.5))
                
                if conf_after < conf_before:
                    decayed_count += 1
                    total_decay_applied += (conf_before - conf_after)
        
        return {
            "decayed_count": decayed_count,
            "deleted_count": deleted_count,
            "avg_decay": total_decay_applied / decayed_count if decayed_count > 0 else 0,
            "total_before": len(memories_before),
            "total_after": len(memories_after),
        }
    
    def evaluate_merging(
        self,
        memories_before: List[Dict],
        memories_after: List[Dict],
    ) -> Dict:
        """
        Evaluate memory merging quality.
        
        Returns:
            Statistics about merge operations
        """
        merged_memories = [
            m for m in memories_after 
            if m['memory_id'].startswith('merged_')
        ]
        
        # Check for duplicates that should have been merged
        duplicates_remaining = self._count_duplicates(memories_after)
        
        return {
            "merged_count": len(merged_memories),
            "duplicates_remaining": duplicates_remaining,
            "total_before": len(memories_before),
            "total_after": len(memories_after),
            "reduction": len(memories_before) - len(memories_after),
        }
    
    def evaluate_promotion(
        self,
        promoted_memories: List[Dict],
        core_memory_files: Dict[str, str],
    ) -> Dict:
        """
        Evaluate promotion to core memory.
        
        Returns:
            Statistics about promotions
        """
        promoted_count = len(promoted_memories)
        
        # Check if promoted memories appear in core files
        found_in_core = 0
        for mem in promoted_memories:
            key = mem.get('key', '')
            for filename, content in core_memory_files.items():
                if key.lower() in content.lower():
                    found_in_core += 1
                    break
        
        return {
            "promoted_count": promoted_count,
            "found_in_core_files": found_in_core,
            "promotion_success_rate": found_in_core / promoted_count if promoted_count > 0 else 0,
        }
    
    def _count_duplicates(self, memories: List[Dict]) -> int:
        """Count potential duplicates (same type and similar content)"""
        duplicates = 0
        seen = set()
        
        for mem in memories:
            signature = (
                mem.get('type'),
                mem.get('key', '').lower()[:20],  # First 20 chars of key
            )
            
            if signature in seen:
                duplicates += 1
            seen.add(signature)
        
        return duplicates
