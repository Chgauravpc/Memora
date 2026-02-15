"""
Memory Extraction - Phase 1, 2 & 3
Stage 1: Heuristic filter
Stage 2: Pattern-based classifier
Stage 3: LLM-based extraction (optional)
"""

import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import (
    SENSORY_FILTER_THRESHOLD,
    EXTRACTION_CLASSIFIER_THRESHOLD,
    HEURISTIC_WEIGHTS,
    EXTRACTION_KEYWORDS,
    MEMORY_TYPES,
    STAGE_3_ENABLED,
    STAGE_3_CONFIDENCE_THRESHOLD,
    MIN_CONFIDENCE_TO_STORE,
    CONFIDENCE_MODIFIERS,
)

logger = logging.getLogger(__name__)

# Lazy import for Stage 3
_llm_extractor = None


def _get_llm_extractor():
    """Lazy load LLM extractor"""
    global _llm_extractor
    if _llm_extractor is None and STAGE_3_ENABLED:
        try:
            from .llm_extractor import LLMExtractor
            _llm_extractor = LLMExtractor()
            logger.info("Stage 3 LLM extractor loaded")
        except Exception as e:
            logger.warning(f"Failed to load Stage 3 extractor: {e}")
    return _llm_extractor


class MemoryExtractor:
    """Extracts memories from user messages using multi-stage pipeline"""

    def __init__(self):
        self.extraction_count = 0
        self.stage3_enabled = STAGE_3_ENABLED
        self.llm_extractor = _get_llm_extractor() if self.stage3_enabled else None
        self.context_buffer = []  # Store last 3 turns for Stage 3 context

    def should_extract(self, message: str) -> Tuple[bool, float]:
        """
        Stage 1: Sensory Memory - Fast heuristic filter
        
        Filters out ~60% of turns before expensive processing:
        - Greetings ("hi", "hello")
        - Acknowledgments ("ok", "thanks")
        - Filler ("um", "uh")
        
        Args:
            message: User's message text
        
        Returns:
            (should_process, score) - True if worth extracting
        """
        message_lower = message.lower().strip()
        
        # Immediate reject: Very short or common greetings
        if len(message_lower) < 5:
            return False, 0.0
        
        greetings = ["hi", "hello", "hey", "thanks", "ok", "okay", "cool", "nice"]
        if message_lower in greetings:
            return False, 0.0
        
        # Calculate heuristic score
        score = 0.0
        
        # Length score (normalized)
        length_score = min(len(message) / 100, 1.0)
        score += length_score * HEURISTIC_WEIGHTS["length"]
        
        # Keyword score
        keyword_matches = 0
        total_keywords = sum(len(kws) for kws in EXTRACTION_KEYWORDS.values())
        
        for keyword_list in EXTRACTION_KEYWORDS.values():
            for keyword in keyword_list:
                if keyword in message_lower:
                    keyword_matches += 1
        
        keyword_score = min(keyword_matches / 3, 1.0)  # Normalize to 3 matches
        score += keyword_score * HEURISTIC_WEIGHTS["keywords"]
        
        # Question score
        question_score = 1.0 if "?" in message else 0.0
        score += question_score * HEURISTIC_WEIGHTS["question"]
        
        # Specificity score (presence of numbers, proper nouns, specific details)
        specificity_indicators = [
            r'\d+',  # Numbers
            r'[A-Z][a-z]+',  # Capitalized words (potential proper nouns)
            r'\b(am|pm|AM|PM)\b',  # Time indicators
            r'@',  # Email/mentions
        ]
        
        specificity_matches = sum(
            1 for pattern in specificity_indicators 
            if re.search(pattern, message)
        )
        specificity_score = min(specificity_matches / 3, 1.0)
        score += specificity_score * HEURISTIC_WEIGHTS["specificity"]
        
        should_process = score >= SENSORY_FILTER_THRESHOLD
        
        logger.debug(
            f"Heuristic filter: score={score:.2f}, "
            f"threshold={SENSORY_FILTER_THRESHOLD}, "
            f"process={should_process}"
        )
        
        return should_process, score

    def classify_and_extract(self, message: str, turn_number: int) -> List[Dict]:
        """
        Stage 2: Simple rule-based classifier
        
        Identifies memory type and extracts key-value pairs.
        Phase 1 uses pattern matching; Phase 3 will add LLM extraction.
        
        Args:
            message: User's message text
            turn_number: Current turn number
        
        Returns:
            List of extracted memory dictionaries
        """
        memories = []
        message_lower = message.lower()
        timestamp = datetime.now().timestamp()
        
        # Pattern-based extraction for each memory type
        
        # PREFERENCE patterns
        preference_patterns = [
            (r"i (?:prefer|like|love|enjoy) (.+)", "preference", 0.8),
            (r"my favorite (.+) is (.+)", "favorite_{}", 0.9),
            (r"i (?:always|usually) (.+)", "habitual_behavior", 0.7),
            (r"i (?:hate|dislike|avoid) (.+)", "negative_preference", 0.8),
        ]
        
        for pattern, key_template, confidence in preference_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                if "{}" in key_template:
                    key = key_template.format(match.group(1).strip())
                    value = match.group(2).strip()
                else:
                    key = key_template
                    value = match.group(1).strip()
                
                memories.append(self._create_memory(
                    memory_type="preference",
                    key=key,
                    value=value,
                    confidence=confidence,
                    turn_number=turn_number,
                    timestamp=timestamp,
                    source_text=message,
                ))
        
        # CONSTRAINT patterns
        constraint_patterns = [
            (r"i (?:can't|cannot|won't) (.+)", "cannot", 0.9),
            (r"i (?:must|have to|need to) (.+)", "must", 0.9),
            (r"i'm (?:allergic to|allergic) (.+)", "allergy", 1.0),
            (r"(?:don't|do not) (.+)", "restriction", 0.7),
        ]
        
        for pattern, key_template, confidence in constraint_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                memories.append(self._create_memory(
                    memory_type="constraint",
                    key=key_template,
                    value=match.group(1).strip(),
                    confidence=confidence,
                    turn_number=turn_number,
                    timestamp=timestamp,
                    source_text=message,
                ))
        
        # ENTITY patterns
        entity_patterns = [
            (r"my (\w+) is (?:named |called )?(\w+)", "my_{}", 0.8),
            (r"my name is (\w+)", "user_name", 0.95),
            (r"i work (?:at|for) (.+?)(?:\.|$)", "employer", 0.8),
            (r"i live in (.+?)(?:\.|$)", "location", 0.85),
        ]
        
        for pattern, key_template, confidence in entity_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                if "{}" in key_template:
                    key = key_template.format(match.group(1).strip())
                    value = match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
                else:
                    key = key_template
                    value = match.group(1).strip()
                
                memories.append(self._create_memory(
                    memory_type="entity",
                    key=key,
                    value=value,
                    confidence=confidence,
                    turn_number=turn_number,
                    timestamp=timestamp,
                    source_text=message,
                ))
        
        # COMMITMENT patterns
        commitment_patterns = [
            (r"i(?:'ll| will) (.+?) by (.+?)(?:\.|$)", "deadline", 0.85),
            (r"i promise (?:to )?(.+)", "promise", 0.9),
            (r"i'm committed to (.+)", "commitment", 0.9),
        ]
        
        for pattern, key_template, confidence in commitment_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                memories.append(self._create_memory(
                    memory_type="commitment",
                    key=key_template,
                    value=match.group(1).strip(),
                    confidence=confidence,
                    turn_number=turn_number,
                    timestamp=timestamp,
                    source_text=message,
                ))
        
        # INSTRUCTION patterns
        instruction_patterns = [
            (r"(?:always|remember to) (.+)", "always", 0.8),
            (r"(?:never|don't ever) (.+)", "never", 0.8),
            (r"make sure (?:to )?(.+)", "make_sure", 0.75),
        ]
        
        for pattern, key_template, confidence in instruction_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                memories.append(self._create_memory(
                    memory_type="instruction",
                    key=key_template,
                    value=match.group(1).strip(),
                    confidence=confidence,
                    turn_number=turn_number,
                    timestamp=timestamp,
                    source_text=message,
                ))
        
        # PAYMENT/FINANCIAL patterns (fact type)
        payment_patterns = [
            # Account numbers
            (r"account(?: number| ending in|:)? (\d+)", "account_number", 0.95),
            # Payment amounts
            (r"(?:payment|balance|amount|bill) (?:of )?\$(\d+(?:,\d+)?)", "payment_amount", 0.9),
            (r"\$(\d+(?:,\d+)?).*(?:due|owed|outstanding)", "amount_due", 0.9),
            # Due dates
            (r"due (?:on |date )?([A-Z][a-z]+ \d+(?:st|nd|rd|th)?)", "due_date", 0.9),
            (r"(?:by|before) ([A-Z][a-z]+ \d+(?:st|nd|rd|th)?)", "deadline_date", 0.85),
            # Payment status
            (r"(?:already |just )?paid.*?(\d+) days? ago", "payment_made", 0.9),
            (r"payment.*?(?:received|processed|submitted)", "payment_status_received", 0.85),
            (r"outstanding balance", "status_outstanding", 0.8),
            # Payment arrangements
            (r"payment (?:plan|extension) (?:of )?(\d+) days?", "payment_arrangement", 0.85),
            (r"(?:set up|setup) (?:a )?payment plan", "request_payment_plan", 0.8),
            # Customer info in payment context
            (r"(?:calling|speaking with) ([A-Z][a-z]+)", "customer_name", 0.85),
            (r"(?:this is |my name is )([A-Z][a-z]+)", "speaker_name", 0.9),
        ]
        
        for pattern, key_template, confidence in payment_patterns:
            matches = re.finditer(pattern, message)
            for match in matches:
                value = match.group(1).strip() if match.groups() else "true"
                memories.append(self._create_memory(
                    memory_type="fact",
                    key=key_template,
                    value=value,
                    confidence=confidence,
                    turn_number=turn_number,
                    timestamp=timestamp,
                    source_text=message,
                ))
        
        if memories:
            logger.info(f"Extracted {len(memories)} memories from turn {turn_number}")
        
        return memories

    def _apply_confidence_modifiers(self, confidence: float, text: str) -> float:
        """Apply confidence modifiers based on certainty words"""
        text_lower = text.lower()
        
        for word, modifier in CONFIDENCE_MODIFIERS.items():
            if word in text_lower:
                confidence += modifier
                logger.debug(f"Confidence modifier '{word}': {modifier:+.2f}")
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def _create_memory(
        self,
        memory_type: str,
        key: str,
        value: str,
        confidence: float,
        turn_number: int,
        timestamp: float,
        source_text: str,
        is_update: bool = False,
    ) -> Dict:
        """Create a structured memory record"""
        # Apply confidence modifiers
        confidence = self._apply_confidence_modifiers(confidence, source_text)
        
        return {
            "memory_id": f"mem_{uuid.uuid4().hex[:8]}",
            "type": memory_type,
            "key": key,
            "value": value,
            "confidence": confidence,
            "turn_number": turn_number,
            "timestamp": timestamp,
            "source_text": source_text,
            "mention_count": 1,
            "superseded_by": '',  # Empty string instead of None
            "supersedes": '',     # Empty string instead of None
            "is_update": is_update,
            "last_accessed_turn": turn_number,
        }

    def extract(self, message: str, turn_number: int) -> List[Dict]:
        """
        Full extraction pipeline (Phase 1, 2 & 3)
        
        Args:
            message: User's message
            turn_number: Current turn number
        
        Returns:
            List of extracted memories (empty if filtered out)
        """
        # Update context buffer for Stage 3
        self.context_buffer.append(message)
        if len(self.context_buffer) > 3:
            self.context_buffer.pop(0)
        
        # Stage 1: Sensory filter
        should_process, heuristic_score = self.should_extract(message)
        
        if not should_process:
            logger.debug(f"Turn {turn_number}: Filtered out by sensory layer")
            return []
        
        # Stage 2: Classify and extract
        stage2_memories = self.classify_and_extract(message, turn_number)
        
        # Check if we should escalate to Stage 3
        should_use_stage3 = False
        stage2_hint = None
        
        if self.stage3_enabled and self.llm_extractor:
            # Escalate if Stage 2 has low confidence or no results
            if not stage2_memories:
                should_use_stage3 = heuristic_score > 0.5  # Message looks important
                logger.debug("Escalating to Stage 3: no Stage 2 results")
            elif stage2_memories:
                max_confidence = max(m['confidence'] for m in stage2_memories)
                if max_confidence < STAGE_3_CONFIDENCE_THRESHOLD:
                    should_use_stage3 = True
                    stage2_hint = stage2_memories[0]['type']  # Hint from Stage 2
                    logger.debug(f"Escalating to Stage 3: low confidence ({max_confidence:.2f})")
        
        # Stage 3: LLM extraction (if needed)
        if should_use_stage3:
            try:
                stage3_memories = self.llm_extractor.extract(
                    message=message,
                    turn_number=turn_number,
                    context_turns=self.context_buffer[:-1],  # Exclude current message
                    stage2_hint=stage2_hint
                )
                
                # Merge Stage 2 and Stage 3, preferring Stage 3 if higher confidence
                memories = self._merge_stage_results(stage2_memories, stage3_memories)
            except Exception as e:
                logger.error(f"Stage 3 failed, falling back to Stage 2: {e}")
                memories = stage2_memories
        else:
            memories = stage2_memories
        
        # Filter by minimum confidence threshold
        high_confidence_memories = [
            m for m in memories 
            if m['confidence'] >= MIN_CONFIDENCE_TO_STORE
        ]
        
        logger.info(
            f"Turn {turn_number}: Extracted {len(high_confidence_memories)}/{len(memories)} "
            f"memories (heuristic={heuristic_score:.2f}, stage3={should_use_stage3})"
        )
        
        self.extraction_count += len(high_confidence_memories)
        
        return high_confidence_memories
    
    def _merge_stage_results(self, stage2: List[Dict], stage3: List[Dict]) -> List[Dict]:
        """
        Merge Stage 2 and Stage 3 results, preferring Stage 3 when there's overlap.
        
        Args:
            stage2: Memories from Stage 2
            stage3: Memories from Stage 3
        
        Returns:
            Merged list of memories
        """
        merged = {}
        
        # Add Stage 2 memories
        for mem in stage2:
            key = (mem['type'], mem['key'])
            merged[key] = mem
        
        # Override with Stage 3 if higher confidence
        for mem in stage3:
            key = (mem['type'], mem['key'])
            if key not in merged or mem['confidence'] > merged[key]['confidence']:
                merged[key] = mem
                if key in merged:
                    logger.debug(f"Stage 3 overrode Stage 2 for {key}")
        
        return list(merged.values())
