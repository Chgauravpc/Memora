"""
LLM-Based Memory Extraction - Phase 3 (Stage 3)
Uses LLM for complex extraction cases that pattern matching can't handle
"""

import logging
import json
import re
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import (
    LLM_PROVIDER,
    LLM_EXTRACTION_MODEL,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    GROQ_API_KEYS,
    STAGE_3_MAX_TOKENS,
    STAGE_3_TEMPERATURE,
    UPDATE_PATTERNS,
)

logger = logging.getLogger(__name__)

# Lazy imports for LLM clients
_openai_client = None
_anthropic_client = None
_groq_clients = []  # List of Groq clients (one per API key)
_current_groq_key_index = 0  # Track which key we're currently using


def _get_openai_client():
    """Lazy load OpenAI client"""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized")
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    return _openai_client


def _get_anthropic_client():
    """Lazy load Anthropic client"""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            from anthropic import Anthropic
            _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Anthropic client initialized")
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    return _anthropic_client


def _get_groq_client():
    """Lazy load Groq clients (one per API key)"""
    global _groq_clients
    if not _groq_clients:
        try:
            from groq import Groq
            for i, api_key in enumerate(GROQ_API_KEYS, 1):
                # Disable automatic retries so we can handle rotation ourselves
                client = Groq(api_key=api_key, max_retries=0)
                _groq_clients.append(client)
            logger.info(f"Initialized {len(_groq_clients)} Groq client(s) with multiple API keys")
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Groq clients: {e}")
            raise
    return _groq_clients


def _rotate_groq_key():
    """Rotate to the next Groq API key"""
    global _current_groq_key_index
    if len(GROQ_API_KEYS) > 1:
        _current_groq_key_index = (_current_groq_key_index + 1) % len(GROQ_API_KEYS)
        logger.info(f"Rotated to Groq API key #{_current_groq_key_index + 1} of {len(GROQ_API_KEYS)}")
    return _current_groq_key_index


class LLMExtractor:
    """
    Stage 3 extractor using LLM for complex cases.
    
    Handles:
    - Complex multi-entity messages
    - Uncertain extractions from Stage 2
    - Update detection
    - Nuanced preference statements
    """
    
    EXTRACTION_PROMPT = """You are a memory extraction system. Extract memorable facts from user messages.

Extract information that is:
- Personal preferences, constraints, or instructions
- Important entities (people, places, companies)
- Commitments or deadlines
- Factual information about the user

DO NOT extract:
- Generic greetings or acknowledgments
- Temporary conversation flow
- Your own responses

OUTPUT FORMAT: Valid JSON array with this schema:
[
  {{
    "type": "preference|constraint|entity|instruction|commitment|fact|event",
    "key": "short_key_name",
    "value": "extracted value",
    "confidence": 0.0-1.0,
    "is_update": true/false,
    "reasoning": "why this is worth storing"
  }}
]

Return [] if nothing worth storing.

EXAMPLES:

User: "I'm Alex, a software engineer at Google in San Francisco."
Output:
[
  {{"type": "entity", "key": "user_name", "value": "Alex", "confidence": 0.95, "is_update": false, "reasoning": "Direct statement of name"}},
  {{"type": "entity", "key": "employer", "value": "Google", "confidence": 0.95, "is_update": false, "reasoning": "Current employer"}},
  {{"type": "entity", "key": "job_title", "value": "software engineer", "confidence": 0.9, "is_update": false, "reasoning": "Professional role"}},
  {{"type": "entity", "key": "location", "value": "San Francisco", "confidence": 0.9, "is_update": false, "reasoning": "Work location"}}
]

User: "I prefer Python over JavaScript for backend work."
Output:
[
  {{"type": "preference", "key": "backend_language", "value": "Python over JavaScript", "confidence": 0.85, "is_update": false, "reasoning": "Explicit technology preference"}}
]

User: "Actually, I changed my mind - I prefer dark mode now."
Output:
[
  {{"type": "preference", "key": "ui_theme", "value": "dark mode", "confidence": 0.9, "is_update": true, "reasoning": "Explicit update to previous preference"}}
]

User: "I can't eat gluten because of celiac disease."
Output:
[
  {{"type": "constraint", "key": "dietary_restriction", "value": "no gluten (celiac disease)", "confidence": 1.0, "is_update": false, "reasoning": "Critical health constraint"}}
]

User: "thanks"
Output: []

User: "Maybe I'll try Python sometime."
Output:
[
  {{"type": "preference", "key": "interest_python", "value": "considering learning Python", "confidence": 0.5, "is_update": false, "reasoning": "Tentative interest, low confidence due to 'maybe'"}}
]

Now extract from this message:

CONTEXT (last 3 turns):
{context}

CURRENT MESSAGE:
{message}

Output (valid JSON array only):"""

    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model = LLM_EXTRACTION_MODEL
        self.extraction_count = 0
        self.escalation_count = 0
        self.total_response_time_ms = 0.0
        self.api_call_count = 0
        self.key_rotation_count = 0
        
        # Log API key configuration
        if self.provider == "groq" and len(GROQ_API_KEYS) > 1:
            logger.info(f"LLM Extractor initialized (provider={self.provider}, model={self.model}, keys={len(GROQ_API_KEYS)})")
        else:
            logger.info(f"LLM Extractor initialized (provider={self.provider}, model={self.model})")
    
    def extract(
        self, 
        message: str, 
        turn_number: int,
        context_turns: Optional[List[str]] = None,
        stage2_hint: Optional[str] = None,
    ) -> List[Dict]:
        """
        Extract memories using LLM.
        
        Args:
            message: User's message
            turn_number: Current turn number
            context_turns: Last 3 turns for context (optional)
            stage2_hint: Hint from Stage 2 about likely type (optional)
        
        Returns:
            List of extracted memory dictionaries
        """
        self.escalation_count += 1
        
        try:
            # Build context
            context_text = "\n".join(context_turns) if context_turns else "No prior context"
            
            # Format prompt
            logger.debug(f"Formatting prompt for message: {message[:50]}...")
            prompt = self.EXTRACTION_PROMPT.format(
                context=context_text,
                message=message
            )
            
            # Add stage 2 hint if available
            if stage2_hint:
                prompt += f"\n\nHINT: Stage 2 detected possible {stage2_hint} type."
            
            # Call LLM with timing
            logger.debug(f"Calling {self.provider} LLM...")
            start_time = time.time()
            response_text = self._call_llm(prompt)
            response_time_ms = (time.time() - start_time) * 1000
            self.total_response_time_ms += response_time_ms
            self.api_call_count += 1
            logger.info(f"LLM API call completed in {response_time_ms:.2f}ms")
            logger.debug(f"LLM response received: {response_text[:100]}...")
            
            memories = self._parse_and_validate(response_text, message, turn_number)
            
            self.extraction_count += len(memories)
            
            logger.info(f"Stage 3 extracted {len(memories)} memories from turn {turn_number}")
            return memories
            
        except Exception as e:
            logger.error(f"Stage 3 extraction failed at line {e.__traceback__.tb_lineno}: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider"""
        try:
            if self.provider == "openai":
                return self._call_openai(prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt)
            elif self.provider == "groq":
                return self._call_groq(prompt)
            else:
                raise ValueError(f"Unknown LLM provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        client = _get_openai_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=STAGE_3_MAX_TOKENS,
            temperature=STAGE_3_TEMPERATURE,
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        client = _get_anthropic_client()
        
        response = client.messages.create(
            model=self.model,
            max_tokens=STAGE_3_MAX_TOKENS,
            temperature=STAGE_3_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API with automatic key rotation on rate limits"""
        clients = _get_groq_client()
        global _current_groq_key_index
        
        # Try all available keys
        max_retries = len(clients)
        for attempt in range(max_retries):
            try:
                client = clients[_current_groq_key_index]
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=STAGE_3_MAX_TOKENS,
                    temperature=STAGE_3_TEMPERATURE,
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error (429)
                if "429" in error_str or "rate_limit" in error_str.lower() or "too many requests" in error_str.lower():
                    if attempt < max_retries - 1:  # Not the last attempt
                        logger.warning(f"Rate limit hit on key #{_current_groq_key_index + 1}, rotating to next key...")
                        _rotate_groq_key()
                        self.key_rotation_count += 1
                        continue
                    else:
                        logger.error(f"Rate limit hit on all {max_retries} API keys")
                        raise
                else:
                    # Not a rate limit error, re-raise
                    raise
        
        # Should not reach here
        raise Exception("Failed to call Groq API after trying all keys")
    
    def _parse_and_validate(
        self, 
        response_text: str, 
        original_message: str,
        turn_number: int
    ) -> List[Dict]:
        """
        Parse LLM response and validate against schema.
        
        Args:
            response_text: Raw LLM response
            original_message: Original user message
            turn_number: Current turn number
        
        Returns:
            List of validated memory dictionaries
        """
        timestamp = datetime.now().timestamp()
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if json_text.startswith("```"):
                json_text = re.sub(r'^```(?:json)?\n', '', json_text)
                json_text = re.sub(r'\n```$', '', json_text)
            
            # Parse JSON
            extracted = json.loads(json_text)
            
            if not isinstance(extracted, list):
                logger.warning(f"LLM returned non-list: {type(extracted)}")
                return []
            
            # Validate and enrich each memory
            validated_memories = []
            
            for item in extracted:
                # Validate required fields
                if not all(k in item for k in ["type", "key", "value", "confidence"]):
                    logger.warning(f"Skipping invalid memory (missing fields): {item}")
                    continue
                
                # Add missing fields with defaults
                memory = {
                    "memory_id": f"mem_{turn_number}_{len(validated_memories)}",
                    "type": item["type"],
                    "key": item["key"],
                    "value": item["value"],
                    "confidence": float(item["confidence"]),
                    "is_update": item.get("is_update", False),
                    "turn_number": turn_number,
                    "timestamp": timestamp,
                    "source_text": original_message,
                    "mention_count": 1,
                    "superseded_by": '',  # Empty string instead of None
                    "supersedes": '',     # Empty string instead of None
                    "last_accessed_turn": turn_number,
                }
                
                # Clamp confidence to [0, 1]
                memory["confidence"] = max(0.0, min(1.0, memory["confidence"]))
                
                validated_memories.append(memory)
            
            logger.debug(f"Validated {len(validated_memories)}/{len(extracted)} memories")
            return validated_memories
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Response was: {response_text[:200]}")
            
            # Retry once with error feedback
            return self._retry_with_error(response_text, original_message, turn_number, str(e))
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return []
    
    def _retry_with_error(
        self,
        bad_response: str,
        original_message: str,
        turn_number: int,
        error_msg: str
    ) -> List[Dict]:
        """Retry extraction with error feedback"""
        logger.info("Retrying extraction with error feedback")
        
        retry_prompt = f"""The previous response had an error:
ERROR: {error_msg}

PREVIOUS RESPONSE:
{bad_response[:300]}

Please provide a valid JSON array output for this message:
{original_message}

Output (valid JSON array only):"""
        
        try:
            response_text = self._call_llm(retry_prompt)
            return self._parse_and_validate(response_text, original_message, turn_number)
        except Exception as e:
            logger.error(f"Retry also failed: {e}")
            return []
    
    def detect_update_intent(self, message: str) -> bool:
        """
        Detect if message is an update to existing memory.
        
        Args:
            message: User message
        
        Returns:
            True if update pattern detected
        """
        message_lower = message.lower()
        
        for pattern in UPDATE_PATTERNS:
            if re.search(pattern, message_lower):
                logger.debug(f"Update pattern detected: {pattern}")
                return True
        
        return False
