import re

class TranslationService:
    """
    Translates sequences of ISL glosses into English sentences.
    ISL typically follows Subject-Object-Verb (SOV) structure.
    English follows Subject-Verb-Object (SVO) structure.
    """
    def __init__(self):
        # A simple rule-based dictionary for demonstration.
        # In a production system, this would be a Sequence-to-Sequence model (like a Transformer)
        # trained on an ISL-to-English aligned corpus.
        
        self.rules = {
            "HELLO HOW YOU": "Hello, how are you?",
            "YOU NAME WHAT": "What is your name?",
            "I FINE": "I am fine.",
            "YOU HELP ME": "Can you help me?",
            "I DEAF": "I am deaf.",
            "YOU HEARING": "Are you hearing?",
            "PLEASE REPEAT": "Please repeat that.",
            "THANK YOU": "Thank you!",
            "GOOD MORNING": "Good morning.",
            "WHERE TOILET": "Where is the toilet?",
        }
        
    def translate(self, gloss_sequence):
        """
        Translates a list of string glosses into an English sentence.
        """
        if not gloss_sequence:
            return ""
            
        # Join sequence into a single normalized string
        sequence_str = " ".join([g.upper().strip() for g in gloss_sequence if g])
        
        # Exact match rule-based translation
        for rule, translation in self.rules.items():
            if rule in sequence_str:
                return translation
                
        # Basic heuristic translation if no exact match
        # Try to restructure SOV to SVO naively if we detect 3 words
        words = sequence_str.split()
        if len(words) == 3:
            # S O V -> S V O
            # e.g., "I BOOK READ" -> "I READ BOOK"
            subject, obj, verb = words[0], words[1], words[2]
            
            # Very naive check to see if verb is likely an action
            if verb in ["READ", "EAT", "WANT", "LIKE", "DRINK", "GO"]:
                return f"{subject.capitalize()} {verb.lower()} {obj.lower()}."
                
        # Fallback: Just return the capitalized sequence as a placeholder
        return sequence_str.capitalize() + "."
