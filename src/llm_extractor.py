"""
LLM-Based Memory Extraction - Phase 3 (Stage 3)
Uses LLM for complex extraction cases that pattern matching can't handle
"""

import logging
import json
import random
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
    STAGE_3_MAX_ATTEMPTS,
    STAGE_3_BACKOFF_BASE,
    STAGE_3_BACKOFF_MAX,
    SPEAKER_AWARE_EXTRACTION,
    UPDATE_PATTERNS,
    EXTRACTION_CACHE_VERSION,
)
from .extraction_cache import build_key, get_extraction_cache

logger = logging.getLogger(__name__)

# Why a Stage 3 call produced nothing.
#
# `extract()` returns [] for a turn that genuinely held no facts, for a truncated JSON
# response, for a dead API key and for a schema violation -- four situations with completely
# different fixes, reported identically. At the measured ~19% empty rate that is roughly one
# in five extraction calls whose outcome is unknown, and it is how a decommissioned model
# produced a silent 0% score with no error anywhere in the scorecard.
#
# The distinction is also load-bearing for the cache: CLEAN is the only reason safe to store.
EMPTY_REASON_CLEAN = "clean_empty"        # model replied, valid JSON, genuinely no facts
EMPTY_REASON_PARSE = "parse_error"        # JSON did not parse, even after the one retry
EMPTY_REASON_SCHEMA = "invalid_schema"    # parsed, but not a list / no item passed validation
EMPTY_REASON_API = "api_error"            # the call itself raised

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

        # Why the most recent extract() returned nothing. None means it returned memories.
        self.last_empty_reason: Optional[str] = None
        # Tally across the process, so a run can report WHY extraction came up empty rather
        # than only how often. Surfaced through empty_reason_counts().
        self.empty_reasons: Dict[str, int] = {}

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
        speaker: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Extract memories using LLM.

        Args:
            message: User's message
            turn_number: Current turn number
            context_turns: Last 3 turns for context (optional)
            stage2_hint: Hint from Stage 2 about likely type (optional)
            speaker: Who is speaking. The base prompt is written for a single
                first-person user ("factual information about the user"), which mis-frames
                every turn of a conversation between several people -- it invites keys
                like `user_name` for whoever spoke last and gives the model no way to
                attribute a fact to the right person. Naming the speaker fixes the frame.
                This matters more than it looks: at the ~80% Stage 3 escalation measured
                on conversational text, this prompt writes nearly every memory in the
                store, so its framing is the ceiling on everything downstream.
            event_date: When the turn's content is set, so the model can keep relative
                references ("last week") interpretable rather than dropping them.

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
            
            # Re-frame for multi-party conversation.
            #
            # Prepended rather than templated into EXTRACTION_PROMPT so the base prompt --
            # and therefore the legacy profile -- is untouched and remains directly
            # comparable. The instructions below are about conversational structure
            # (several speakers, third-person narrative, past events); nothing here is
            # specific to any dataset.
            if SPEAKER_AWARE_EXTRACTION and speaker:
                preamble = (
                    f"This is a conversation between several people. The current turn is "
                    f"spoken by {speaker}.\n"
                    f"- Extract facts about ANY person mentioned, not only about a 'user'.\n"
                    f"- Attribute correctly: a fact {speaker} states about someone else "
                    f"belongs to that person, not to {speaker}.\n"
                    f"- Do NOT invent generic keys like 'user_name'. Name the subject.\n"
                    f"- Past events are worth remembering; use type 'event' for things "
                    f"that happened and 'fact' for stable information.\n"
                )
                if event_date:
                    preamble += (
                        f"- This turn takes place on {event_date}. Resolve relative time "
                        f"references ('last week', 'yesterday') against that date and "
                        f"keep them in the value.\n"
                    )
                # OVER-COMPRESSION IS THE MEASURED FAILURE MODE.
                #
                # Every example in the base prompt has a bare-token value -- "Alex",
                # "Google", "dark mode" -- which teaches maximal compression. On
                # conversational input that produces memories like
                #     charity race: 18 May 2023
                # from a turn that also said the race raised awareness for mental health.
                # The date survives, the purpose is gone, and "what did the race raise
                # awareness for" is then unanswerable from a memory that is otherwise
                # about exactly the right thing. Retrieval cannot recover a detail that
                # extraction discarded, so this bounds every downstream component.
                #
                # Two rules address it: values must stand alone, and one turn may yield
                # several memories rather than being squeezed into one.
                preamble += (
                    "\nVALUE QUALITY -- this matters more than brevity:\n"
                    "- Each value must be SELF-CONTAINED: understandable on its own, "
                    "without the key and without the original sentence. Someone reading "
                    "only the value should learn the fact.\n"
                    "- KEEP the specifics that make a fact answerable: what it was for, "
                    "why, with whom, where, how much, which one. Do not reduce a fact to "
                    "its date or its bare subject.\n"
                    "- Emit ONE MEMORY PER DISTINCT FACT. A turn mentioning an event, its "
                    "purpose and a realisation about it is three memories, not one.\n"
                    "- Record what someone IS or HAS as well as what they DID: "
                    "relationship status, where they live, what they own.\n"
                    "\nGood:  {\"key\": \"charity race\", \"value\": \"ran a charity race "
                    "on 18 May 2023 raising awareness for mental health\"}\n"
                    "Bad:   {\"key\": \"charity race\", \"value\": \"18 May 2023\"}   "
                    "(purpose lost)\n"
                    "Good:  {\"key\": \"relationship status\", \"value\": \"single since "
                    "her breakup in 2019\"}\n"
                    "Bad:   {\"key\": \"breakup\", \"value\": \"after 2019\"}   "
                    "(status not stated)\n"
                )
                prompt = preamble + "\n" + prompt

            # Add stage 2 hint if available
            if stage2_hint:
                prompt += f"\n\nHINT: Stage 2 detected possible {stage2_hint} type."

            self.last_empty_reason = None

            # Cache lookup.
            #
            # Keyed on the finished prompt, so every input that shapes the result -- message,
            # context, speaker, event date, hint, and the prompt template itself -- is
            # already covered without maintaining a list of them. A hit skips the API call
            # entirely and, more importantly, guarantees this turn contributes exactly the
            # memories it contributed last time.
            cache = get_extraction_cache()
            cache_key = build_key(
                prompt=prompt,
                provider=self.provider,
                model=self.model,
                temperature=STAGE_3_TEMPERATURE,
                max_tokens=STAGE_3_MAX_TOKENS,
                turn_number=turn_number,
                version=EXTRACTION_CACHE_VERSION,
            ) if cache.enabled else None

            if cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    self.extraction_count += len(cached)
                    if not cached:
                        self._note_empty(EMPTY_REASON_CLEAN)
                    logger.debug(
                        f"Stage 3 cache hit for turn {turn_number} ({len(cached)} memories)"
                    )
                    return cached

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

            if not memories and self.last_empty_reason is None:
                # Parsed fine and yielded nothing -- a real "no facts here".
                self._note_empty(EMPTY_REASON_CLEAN)

            # Store only clean outcomes. A truncated response or a rejected schema is a
            # transient failure; freezing it would replay that failure on every future run
            # of this conversation and quietly cap the score.
            if cache_key is not None and (memories or self.last_empty_reason == EMPTY_REASON_CLEAN):
                cache.put(cache_key, memories, meta={
                    "turn_number": turn_number,
                    "model": self.model,
                    "provider": self.provider,
                })

            logger.info(f"Stage 3 extracted {len(memories)} memories from turn {turn_number}")
            return memories

        except Exception as e:
            logger.error(f"Stage 3 extraction failed at line {e.__traceback__.tb_lineno}: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            self._note_empty(EMPTY_REASON_API)
            return []

    def _note_empty(self, reason: str) -> None:
        """Record why this call produced no memories."""
        self.last_empty_reason = reason
        self.empty_reasons[reason] = self.empty_reasons.get(reason, 0) + 1

    def empty_reason_counts(self) -> Dict[str, int]:
        """Copy of the empty-reason tally, for the results file."""
        return dict(self.empty_reasons)
    
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
        """
        Call Groq API with key rotation AND exponential backoff on rate limits.

        Previously the attempt count was `len(clients)`, so a single-key deployment made
        exactly one attempt with no backoff and any transient 429 propagated. That is
        survivable for a demo but fatal for a long batch run (a benchmark makes tens of
        thousands of these calls), so attempts are now floored at
        STAGE_3_MAX_ATTEMPTS and each retry sleeps.

        Key rotation is retained and still happens first on each retry -- with multiple
        accounts, another key may have quota even when this one does not.
        """
        clients = _get_groq_client()
        global _current_groq_key_index

        max_attempts = max(len(clients), STAGE_3_MAX_ATTEMPTS)
        for attempt in range(max_attempts):
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
                lowered = error_str.lower()
                # Transient: rate limits plus upstream capacity/gateway errors, which
                # behave the same way (wait and retry) even though they are not 429s.
                is_transient = (
                    "429" in error_str
                    or "rate_limit" in lowered
                    or "too many requests" in lowered
                    or "overloaded" in lowered
                    or "503" in error_str
                    or "502" in error_str
                    or "504" in error_str
                )
                if not is_transient or attempt == max_attempts - 1:
                    if is_transient:
                        logger.error(
                            f"Rate limited after {max_attempts} attempts "
                            f"({len(clients)} key(s))"
                        )
                    raise

                if len(clients) > 1:
                    _rotate_groq_key()
                    self.key_rotation_count += 1

                # Prefer the provider's own hint ("try again in 4.2s") when present.
                delay = None
                hint = re.search(r"try again in ([0-9.]+)s", error_str, re.IGNORECASE)
                if hint:
                    try:
                        delay = float(hint.group(1))
                    except ValueError:
                        delay = None
                if delay is None:
                    delay = min(
                        STAGE_3_BACKOFF_BASE * (2 ** attempt), STAGE_3_BACKOFF_MAX
                    )
                # Jitter so concurrent benchmark workers do not retry in lockstep.
                delay += random.uniform(0, 0.5 * delay + 0.1)
                logger.warning(
                    f"Transient Groq error (attempt {attempt + 1}/{max_attempts}), "
                    f"sleeping {delay:.1f}s: {error_str[:120]}"
                )
                time.sleep(delay)

        # Unreachable: the loop either returns or raises on its final attempt.
        raise RuntimeError("Failed to call Groq API after exhausting all attempts")
    
    def _parse_and_validate(
        self, 
        response_text: str, 
        original_message: str,
        turn_number: int,
        attempt: int = 0
    ) -> List[Dict]:
        """
        Parse LLM response and validate against schema.
        
        Args:
            response_text: Raw LLM response
            original_message: Original user message
            turn_number: Current turn number
            attempt: 0 on the first parse, 1 when re-parsing a retry response.
                Retries are capped at one; see the JSONDecodeError handler.
        
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
                self._note_empty(EMPTY_REASON_SCHEMA)
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
            
            # A non-empty response whose every item failed validation is a schema problem,
            # not an empty turn -- and must not be cached as one.
            if extracted and not validated_memories:
                self._note_empty(EMPTY_REASON_SCHEMA)

            logger.debug(f"Validated {len(validated_memories)}/{len(extracted)} memories")
            return validated_memories
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Response was: {response_text[:200]}")
            
            # Retry once, and only once. Without the attempt guard a model that
            # never returns JSON recurses until RecursionError, burning one API
            # call per frame.
            if attempt > 0:
                logger.error("Giving up on this turn after one failed retry")
                self._note_empty(EMPTY_REASON_PARSE)
                return []
            return self._retry_with_error(response_text, original_message, turn_number, str(e))

        except Exception as e:
            logger.error(f"Validation error: {e}")
            self._note_empty(EMPTY_REASON_SCHEMA)
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
            return self._parse_and_validate(response_text, original_message, turn_number, attempt=1)
        except Exception as e:
            logger.error(f"Retry also failed: {e}")
            self._note_empty(EMPTY_REASON_API)
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
