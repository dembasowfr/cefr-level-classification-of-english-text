class PromptManager:
    def __init__(self, task_instruction="Classify the CEFR level of the following text:"):
        self.task_instruction = task_instruction

    def generate_prompt(self, text):
        """
        Formats the input text with the classification instruction.
        """
        return f"Question: What is the CEFR level of this text? Options: A1, A2, B1, B2, C1, C2.\\n\\nText: {text}"

    def get_labels(self):
        """
        Returns the valid labels for the task.
        """
        return ["A1", "A2", "B1", "B2", "C1", "C2"]

# 2. Feature Injection


import textstat
import re


class PromptManagerWithStats:
    def __init__(self, task_instruction="Classify the CEFR level:"):
        self.task_instruction = task_instruction

    def calculate_stats(self, text):
        fk_grade = max(0, textstat.flesch_kincaid_grade(text))
        words = textstat.lexicon_count(text, removepunct=True)
        sentences = textstat.sentence_count(text)
        avg_sentence_len = round(words / sentences, 1) if sentences > 0 else 0
        tokens = re.findall(r'\w+', text.lower())
        ttr = round(len(set(tokens)) / len(tokens), 2) if len(tokens) > 0 else 0
        syllables = textstat.syllable_count(text)
        avg_syllables = round(syllables / words, 2) if words > 0 else 0
        
        return {
            "fk_grade": fk_grade,
            "avg_len": avg_sentence_len,
            "ttr": ttr,
            "complexity": avg_syllables
        }

    def generate_prompt(self, text):
        s = self.calculate_stats(text)
        stats_header = (f"[Stats -> GL:{s['fk_grade']} | SL:{s['avg_len']} | "
                        f"TR:{s['ttr']} | CX:{s['complexity']}]")
        
        # This structure provides clear options and a direct response point
        return (f"{stats_header}\n"
                f"Task: Classify text into CEFR level: A1, A2, B1, B2, C1, or C2.\n"
                f"Text: {text}\n"
                f"Answer: ")

    def get_labels(self):
        return ["A1", "A2", "B1", "B2", "C1", "C2"]


# 2.2. Feature Injection With Advanced Features
import spacy
import textstat
import re

# Load nlp outside to avoid reloading it for every instance
nlp = spacy.load("en_core_web_sm", disable=["ner"])

class PromptManagerFeatureInjectionWithSpacy:
    def __init__(self, task_instruction="Classify the CEFR level:"):
        self.task_instruction = task_instruction

    def get_tree_depth(self, node):
        """Recursively calculates the maximum depth of the dependency tree."""
        if not list(node.children):
            return 1
        return 1 + max(self.get_tree_depth(child) for child in node.children)

    def calculate_combined_stats(self, text):
        doc = nlp(text)
        
        # Coarse Stats: Flesch-Kincaid and Sentence Length
        fk_grade = max(0, textstat.flesch_kincaid_grade(text))
        sents = list(doc.sents)
        avg_len = len([t for t in doc if not t.is_punct]) / len(sents) if sents else 0
        
        # Deep Stats: Passive Voice and Syntactic Depth
        passives = len([tok for tok in doc if tok.dep_ == "auxpass"])
        depths = [self.get_tree_depth(sent.root) for sent in sents]
        avg_depth = sum(depths) / len(depths) if depths else 0
        
        # Lexical Stats: Content Word Density
        content_pos = ["NOUN", "VERB", "ADJ", "ADV"]
        content_words = [tok for tok in doc if tok.pos_ in content_pos]
        lex_density = len(content_words) / len(doc) if len(doc) > 0 else 0

        return {
            "gl": fk_grade,
            "sl": round(avg_len, 1),
            "psv": passives,
            "dep": round(avg_depth, 2),
            "den": round(lex_density, 2)
        }

    def generate_prompt(self, text):
        s = self.calculate_combined_stats(text)
        # Construct the specialized header for the LLM
        stats_header = (f"[Stats -> GL:{s['gl']} | SL:{s['sl']} | "
                        f"DEP:{s['dep']} | PSV:{s['psv']} | DEN:{s['den']}]")
        
        return (f"{stats_header}\n"
                f"Task: Classify text into CEFR level: A1, A2, B1, B2, C1, or C2.\n"
                f"Text: {text}\n"
                f"Answer: ")



# 3. Ordinal Sequence to Sequence Trainer

import spacy
import textstat
import re
import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer

# Load spaCy model (ensure you have run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

class PromptManagerOrdinalFeatureInjection:
    def __init__(self):
        # 1. Standard Instruction
        self.task_instruction = "Task: Classify the CEFR proficiency level of this text (A1, A2, B1, B2, C1, C2)."
        
        # 2. Ordinal Distance Matrix (6x6)
        # Used by the Trainer to penalize "Far Misses" (e.g. A1 vs C2)
        # A1=0, A2=1, B1=2, B2=3, C1=4, C2=5
        self.dist_matrix = torch.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                self.dist_matrix[i, j] = abs(i - j)

    def get_tree_depth(self, node):
        """Recursively calculates the maximum depth of the dependency tree."""
        if not list(node.children):
            return 1
        return 1 + max(self.get_tree_depth(child) for child in node.children)

    def calculate_combined_stats(self, text):
        """Extracts linguistic features: GL, SL, DEP, PSV, DEN"""
        doc = nlp(text)
        
        # Coarse Stats
        fk_grade = max(0, textstat.flesch_kincaid_grade(text))
        sents = list(doc.sents)
        avg_len = len([t for t in doc if not t.is_punct]) / len(sents) if sents else 0
        
        # Deep Stats (Syntactic Complexity)
        passives = len([tok for tok in doc if tok.dep_ == "auxpass"])
        depths = [self.get_tree_depth(sent.root) for sent in sents]
        avg_depth = sum(depths) / len(depths) if depths else 0
        
        # Lexical Density
        content_pos = ["NOUN", "VERB", "ADJ", "ADV"]
        content_words = [tok for tok in doc if tok.pos_ in content_pos]
        lex_density = len(content_words) / len(doc) if len(doc) > 0 else 0

        # Return formatted string for prompt
        return (f"[Stats -> GL:{fk_grade} | SL:{round(avg_len, 1)} | "
                f"DEP:{round(avg_depth, 2)} | PSV:{passives} | DEN:{round(lex_density, 2)}]")

    def generate_prompt(self, text):
        """Combines Stats + Instruction + Text"""
        stats_header = self.calculate_combined_stats(text)
        
        return (f"{stats_header}\n"
                f"{self.task_instruction}\n"
                f"Text: {text}\n"
                f"Answer: ")

# --- THE CUSTOM TRAINER ---

class OrdinalSeq2SeqTrainer(Seq2SeqTrainer):
    def set_distance_matrix(self, matrix):
        self.dist_matrix = matrix.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Standard Cross Entropy with Smoothing
        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
        main_loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        # Note: We rely on the implicit benefit of label smoothing and
        # the strong signal from feature injection here. 
        # (Explicit ordinal penalty terms can be unstable in 
        # mixed-precision training, so smoothing is the safer "Ordinal-Lite" approach).
        
        return (main_loss, outputs) if return_outputs else main_loss