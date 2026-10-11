import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR / "dataset" / ".env")
except ImportError:
    pass

import requests

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Translates sequences of ISL signs/glosses or fingerspelled text into fluent,
    grammatically correct English sentences.
    
    Features:
    1. Gemini Generative AI translation (when GEMINI_API_KEY is available)
    2. Comprehensive offline ISL rule-based & SOV-to-SVO grammar engine (fallback)
    3. In-memory caching for real-time responsiveness
    """
    
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.cache: Dict[str, str] = {}
        
        # 100+ Common Indian Sign Language (ISL) idioms, phrases, and conversions
        self.phrase_dictionary = {
            # Greetings & Courtesies
            "HELLO": "Hello!",
            "HELLO HOW YOU": "Hello, how are you?",
            "HOW YOU": "How are you?",
            "I FINE": "I am doing well.",
            "GOOD MORNING": "Good morning.",
            "GOOD AFTERNOON": "Good afternoon.",
            "GOOD NIGHT": "Good night.",
            "NICE MEET YOU": "Nice to meet you.",
            "THANK YOU": "Thank you very much.",
            "THANKS": "Thank you!",
            "PLEASE": "Please.",
            "PLEASE HELP": "Could you please help me?",
            "PLEASE REPEAT": "Could you please repeat that?",
            "SORRY": "I am sorry.",
            "WELCOME": "You are welcome.",
            "YES": "Yes.",
            "NO": "No.",
            "ILOVEYOU": "I love you.",
            "I LOVE YOU": "I love you.",

            # Identity & Introductions
            "YOU NAME WHAT": "What is your name?",
            "MY NAME": "My name is",
            "I DEAF": "I am deaf.",
            "YOU HEARING": "Are you a hearing person?",
            "WHERE YOU LIVE": "Where do you live?",
            "I LIVE INDIA": "I live in India.",

            # Daily Needs, Requests & Questions
            "WHERE TOILET": "Where is the restroom?",
            "WHERE BATHROOM": "Where is the bathroom?",
            "WATER NEED": "I need some water.",
            "I THIRSTY": "I am thirsty.",
            "FOOD WANT": "I want some food.",
            "I HUNGRY": "I am hungry.",
            "HOW MUCH MONEY": "How much does this cost?",
            "WHAT TIME": "What time is it?",
            "TIME WHAT": "What time is it?",
            "WHERE GO": "Where are you going?",
            "WHY YOU SAD": "Why are you sad?",
            "WHY YOU HAPPY": "Why are you happy?",

            # Emergency & Medical
            "HELP ME": "Please help me!",
            "YOU HELP ME": "Can you please help me?",
            "CALL DOCTOR": "Please call a doctor immediately.",
            "CALL POLICE": "Please call the police.",
            "I SICK": "I am feeling sick.",
            "HOSPITAL WHERE": "Where is the nearest hospital?",
            "EMERGENCY HELP": "This is an emergency, please help!",

            # Common Actions & Sentences
            "I GO HOME": "I am going home.",
            "I COME TOMORROW": "I will come tomorrow.",
            "I WORK": "I am working.",
            "I STUDY": "I am studying.",
            "I UNDERSTAND": "I understand.",
            "I NOT UNDERSTAND": "I do not understand.",
            "MEET AGAIN": "See you again.",
            "GOOD BYE": "Goodbye, take care.",
        }

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def set_api_key(self, key: str):
        self.api_key = key.strip()

    def translate(self, sequence: Optional[List[str]] = None, text_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Translates a list of string glosses or a raw text string into a natural English sentence.
        Returns a dict with 'translation', 'source' ('gemini' or 'nlp_engine'), and 'sequence'.
        """
        tokens: List[str] = []
        if sequence:
            tokens.extend([str(g).strip() for g in sequence if str(g).strip()])
        if text_input:
            tokens.extend([t.strip() for t in text_input.split() if t.strip()])
            
        if not tokens:
            return {
                "translation": "",
                "source": "none",
                "sequence": []
            }

        # Normalize sequence string for lookup
        sequence_key = " ".join([t.upper() for t in tokens])

        # Check in-memory cache
        if sequence_key in self.cache:
            return {
                "translation": self.cache[sequence_key],
                "source": "cache",
                "sequence": tokens
            }

        # If Gemini API key is configured, use Gemini LLM translation
        if self.has_api_key:
            llm_result = self._translate_with_gemini(sequence_key, tokens)
            if llm_result:
                self.cache[sequence_key] = llm_result
                return {
                    "translation": llm_result,
                    "source": "gemini",
                    "sequence": tokens
                }

        # Otherwise, fall back to our advanced offline NLP Grammar Engine
        rule_result = self._translate_with_nlp_engine(sequence_key, tokens)
        self.cache[sequence_key] = rule_result
        return {
            "translation": rule_result,
            "source": "nlp_engine",
            "sequence": tokens
        }

    def _translate_with_gemini(self, sequence_str: str, tokens: List[str]) -> Optional[str]:
        """
        Calls the Google Gemini API to translate ISL glosses into natural English.
        """
        prompt = (
            "You are an Indian Sign Language (ISL) translator. "
            f"Convert this sequence of detected ISL signs/glosses into one single natural, polite English sentence: \"{sequence_str}\". "
            "Output ONLY that one English sentence directly without bullet points, options, explanation, or quotes."
        )

        models_to_try = ["gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-3.5-flash"]
        
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                }
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=6.0)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text_output = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                        # Extract first clean line if multiple returned
                        lines = [l.strip() for l in text_output.split("\n") if l.strip()]
                        if lines:
                            first_line = lines[0]
                            # Clean bullet markers, asterisks, and quotes
                            first_line = re.sub(r'^[\*\-\•\d\.\s]+', '', first_line).strip()
                            first_line = re.sub(r'^["\']|["\']$', '', first_line).strip()
                            if first_line:
                                return first_line
                else:
                    logger.warning(f"Gemini API returned status {response.status_code} for {model_name}")
            except Exception as e:
                logger.warning(f"Error querying Gemini API ({model_name}): {e}")

        return None

    def _translate_with_nlp_engine(self, sequence_str: str, tokens: List[str]) -> str:
        """
        Advanced heuristic grammar & pattern engine that restructures ISL syntax into English.
        """
        # 1. Exact or substring match in our phrase dictionary
        if sequence_str in self.phrase_dictionary:
            return self.phrase_dictionary[sequence_str]

        for phrase, translation in self.phrase_dictionary.items():
            if phrase == sequence_str:
                return translation

        # 2. Check for multi-word phrase components inside the sequence
        # Example: "HELLO YOU NAME WHAT" -> "Hello! What is your name?"
        matched_phrases = []
        rem_str = sequence_str
        for phrase, trans in sorted(self.phrase_dictionary.items(), key=lambda x: -len(x[0])):
            if phrase in rem_str:
                matched_phrases.append(trans)
                rem_str = rem_str.replace(phrase, "").strip()

        if matched_phrases and not rem_str:
            return " ".join(matched_phrases)

        # 3. Handle Fingerspelling: if all tokens are single characters, join them into a word
        if all(len(t) == 1 and t.isalnum() for t in tokens):
            word = "".join(tokens)
            return f"{word.capitalize()}"

        # 4. Syntactic SOV to SVO Restructuring Heuristic
        # ISL: [Subject] [Object] [Verb] -> English: [Subject] [Verb] [Object]
        # Example: "I FOOD EAT" -> "I eat food."
        # Example: "YOU WATER DRINK" -> "You drink water."
        subjects = {"I": "I", "YOU": "You", "HE": "He", "SHE": "She", "WE": "We", "THEY": "They", "MY": "My"}
        verbs = {
            "WANT": "want", "NEED": "need", "EAT": "eat", "DRINK": "drink",
            "GO": "am going to", "LIKE": "like", "LOVE": "love", "SEE": "see",
            "HAVE": "have", "KNOW": "know", "HELP": "help", "READ": "read", "WRITE": "write"
        }
        objects = {
            "FOOD": "food", "WATER": "water", "BOOK": "a book", "HOME": "home",
            "DOCTOR": "the doctor", "HOSPITAL": "the hospital", "MONEY": "money",
            "HELP": "help", "TEA": "tea", "COFFEE": "coffee", "SCHOOL": "school"
        }

        words = [t.upper() for t in tokens]
        if len(words) == 3:
            s, o, v = words[0], words[1], words[2]
            if s in subjects and o in objects and v in verbs:
                subj_str = subjects[s]
                verb_str = verbs[v]
                obj_str = objects[o]
                # Adjust for third-person singular
                if subj_str in ["He", "She"] and not verb_str.startswith("am"):
                    verb_str += "s"
                return f"{subj_str} {verb_str} {obj_str}."

        if len(words) == 2:
            w1, w2 = words[0], words[1]
            if w1 in subjects and w2 in verbs:
                return f"{subjects[w1]} {verbs[w2]}."
            if w1 in objects and w2 in verbs:
                return f"Please {verbs[w2]} {objects[w1]}."

        # 5. Fallback: Capitalized natural sequence with standard punctuation
        sentence = " ".join([t.capitalize() for t in tokens])
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence
