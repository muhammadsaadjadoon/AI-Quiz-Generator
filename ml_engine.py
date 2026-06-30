"""
ml_engine.py
────────────────────────────────────────────────────────────────────
Deep-learning MCQ generation core.

Pipeline (all genuinely "deep learning", not template regex tricks):

    1. Sentence-Transformers ("all-MiniLM-L6-v2")
       -> dense embeddings for every sentence in the source text
       -> used for: semantic importance ranking (centrality), topic
          clustering (KMeans on embeddings), and semantic-distance
          based distractor mining (find sentences/phrases that are
          *related but different* rather than random noise).

    2. T5 sequence-to-sequence transformer ("valhalla/t5-small-qg-hl")
       -> a pretrained encoder-decoder model fine-tuned specifically
          for question generation. We highlight a candidate answer
          span in the sentence and the model generates a fluent,
          natural question whose answer is that span — this is a
          genuine neural text-generation step, not a regex template.
       -> Falls back to a second general-purpose T5 ("t5-small") in
          "generate question" prompt mode if the QG-specific model is
          unavailable, and finally to a light rule-based template if
          neither model can be downloaded (e.g. no internet), so the
          app degrades gracefully instead of crashing.

    3. DifficultyClassifier (difficulty_model.py)
       -> small PyTorch network on top of the same sentence embeddings,
          fine-tunable on your own labeled data via train_difficulty.py.

    4. NLTK POS tagging + WordNet
       -> used only for distractor candidate extraction (finding nouns/
          noun phrases as fallback answer spans, and semantically
          related-but-wrong WordNet lemmas as one distractor source).

    5. NumPy + scikit-learn
       -> KMeans clustering on embeddings (topic grouping) and a
          NumPy dot-product quality scorer combining several signals.

All models are pretrained and downloaded automatically from
HuggingFace / sentence-transformers on first run (internet required
once; cached locally afterwards under ~/.cache).
────────────────────────────────────────────────────────────────────
"""

import re
import time
import random
import string as string_punct_module
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag

logger = logging.getLogger("ML_ENGINE")

# ────────────────────────────────────────────────────────────────
# NLTK setup (only used for lightweight POS/WordNet support tasks,
# not for the actual question generation, which is neural)
# ────────────────────────────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

STOP_WORDS = set(stopwords.words("english")) | {
    "also", "however", "therefore", "moreover", "furthermore", "thus",
    "hence", "consequently", "meanwhile", "nevertheless",
}


def split_sentences(text: str) -> List[str]:
    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.split()) >= 6]


def get_pos_tags(sentence: str) -> List[Tuple[str, str]]:
    try:
        return pos_tag(word_tokenize(sentence))
    except Exception:
        return []


def extract_noun_phrases(sentence: str) -> List[Tuple[str, str]]:
    """
    Extracts candidate answer spans as multi-word noun phrases where possible
    (e.g. "Bubble Sort", "time complexity") instead of single weak words.
    Returns a list of (phrase, representative_pos_tag) ranked roughly by quality.
    """
    tagged = get_pos_tags(sentence)
    if not tagged:
        return []

    phrases = []
    current = []
    current_tags = []

    def flush():
        if current:
            phrase = " ".join(current)
            # Reject phrases that are pure stopwords or too short/long to be a clean answer
            words_lower = [w.lower() for w in current]
            if (
                len(phrase) > 2
                and not all(w in STOP_WORDS for w in words_lower)
                and 1 <= len(current) <= 4
                and all(w.isalpha() for w in current)
            ):
                # Weight: proper nouns > multi-word noun phrases > single common nouns
                has_proper = any(t in ("NNP", "NNPS") for t in current_tags)
                weight = (
                    6 if (has_proper and len(current) > 1) else
                    5 if has_proper else
                    4 if len(current) > 1 else
                    2
                )
                rep_tag = current_tags[-1]
                phrases.append((phrase, rep_tag, weight))
        current.clear()
        current_tags.clear()

    for word, tag in tagged:
        if tag in ("NN", "NNS", "NNP", "NNPS") or (tag.startswith("JJ") and current):
            current.append(word)
            current_tags.append(tag)
        else:
            flush()
    flush()

    # De-duplicate (case-insensitive), keep highest-weighted version of each phrase
    best_by_key = {}
    for phrase, tag, weight in phrases:
        key = phrase.lower()
        if key not in best_by_key or weight > best_by_key[key][2]:
            best_by_key[key] = (phrase, tag, weight)

    ranked = sorted(best_by_key.values(), key=lambda x: (x[2], len(x[0])), reverse=True)
    return [(p, t) for p, t, _ in ranked]


def get_key_noun_phrase(sentence: str) -> Optional[Tuple[str, str]]:
    """Pick the single best candidate answer span (multi-word phrase preferred) from a sentence."""
    phrases = extract_noun_phrases(sentence)
    if not phrases:
        return None
    return phrases[0]


def get_wordnet_distractors(word: str, pos_tag_str: str, n: int = 4) -> List[str]:
    """
    WordNet-based distractors. Disabled by default for multi-word or technical
    terms (e.g. "Bubble Sort", "time complexity") since general-English WordNet
    relations produce nonsensical results for domain-specific vocabulary —
    this was the main source of "ajeeb" (weird/unrelated) distractors.
    Only attempted for single common-noun words that WordNet actually covers well.
    """
    if " " in word or not word.isalpha():
        return []

    wn_pos = wordnet.NOUN
    if pos_tag_str.startswith("VB"):
        wn_pos = wordnet.VERB
    elif pos_tag_str.startswith("JJ"):
        wn_pos = wordnet.ADJ

    synsets = wordnet.synsets(word.lower(), pos=wn_pos)
    if not synsets:
        return []

    # Only trust WordNet if the word has a reasonably common, unambiguous sense
    # (technical CS/domain terms tend to have 0 or very noisy synsets)
    candidates = set()
    for syn in synsets[:2]:
        for hyper in syn.hypernyms()[:2]:
            for hypo in hyper.hyponyms()[:6]:
                if hypo != syn:
                    for lemma in hypo.lemmas():
                        name = lemma.name().replace("_", " ")
                        if name.lower() != word.lower() and " " not in name:
                            candidates.add(name)
    out = [c for c in candidates if c.isalpha() and len(c) > 2]
    random.shuffle(out)
    return out[:n]


# ════════════════════════════════════════════════════════════════
# EMBEDDING MODEL (sentence-transformers) — singleton, lazy-loaded
# ════════════════════════════════════════════════════════════════
class EmbeddingModel:
    _instance = None

    def __init__(self):
        self.model = None
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("⏳ Loading sentence-transformer 'all-MiniLM-L6-v2' (downloads on first run)…")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✓ Sentence embedding model ready (384-dim)")
        except Exception as e:
            logger.error(f"✗ Could not load sentence-transformers model: {e}")
            self.model = None

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = EmbeddingModel()
        return cls._instance

    def encode(self, texts, show_progress_bar: bool = False) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model failed to load — check internet connection on first run")
        if isinstance(texts, str):
            return self.model.encode([texts], show_progress_bar=show_progress_bar)[0]
        return self.model.encode(texts, show_progress_bar=show_progress_bar)

    def available(self) -> bool:
        return self.model is not None


# ════════════════════════════════════════════════════════════════
# QUESTION GENERATION MODEL (T5 transformer) — singleton, lazy-loaded
# ════════════════════════════════════════════════════════════════
class QuestionGenerator:
    """
    Wraps a pretrained T5 question-generation transformer.
    Tries a QG-specialized checkpoint first, falls back to plain t5-small
    with a generation prompt, and finally to a rule-based template if no
    model could be downloaded at all (keeps the app usable offline).
    """

    _instance = None

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.mode = None  # "qg-hl" | "t5-prompt" | "template"
        self._load()

    def _load(self):
        try:
            from transformers import T5ForConditionalGeneration, T5TokenizerFast
            try:
                logger.info("⏳ Loading pretrained question-generation model 'valhalla/t5-small-qg-hl'…")
                self.tokenizer = T5TokenizerFast.from_pretrained("valhalla/t5-small-qg-hl")
                self.model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")
                self.mode = "qg-hl"
                logger.info("✓ Question-generation transformer ready (highlight mode)")
                return
            except Exception as e:
                logger.warning(f"QG-specific model unavailable ({e}); falling back to base t5-small")

            logger.info("⏳ Loading pretrained 't5-small' as fallback question generator…")
            self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.mode = "t5-prompt"
            logger.info("✓ Fallback T5 question generator ready (prompt mode)")
        except Exception as e:
            logger.error(f"✗ No transformer could be loaded ({e}); using rule-based template fallback")
            self.tokenizer = None
            self.model = None
            self.mode = "template"

    @classmethod
    def get(cls) -> "QuestionGenerator":
        if cls._instance is None:
            cls._instance = QuestionGenerator()
        return cls._instance

    def available(self) -> bool:
        return self.mode in ("qg-hl", "t5-prompt")

    def generate(self, sentence: str, answer_span: str) -> Tuple[Optional[str], bool]:
        """Generates a natural-language question whose answer is answer_span.
        Returns (question_text, was_neural) — was_neural is False whenever the
        fallback template had to be used (transformer output failed validation,
        or no transformer was available)."""
        question = None
        if self.mode == "qg-hl":
            question = self._generate_qg_hl(sentence, answer_span)
        elif self.mode == "t5-prompt":
            question = self._generate_t5_prompt(sentence, answer_span)

        if question and self._is_valid_question(question, sentence, answer_span):
            return question, True

        # Transformer output failed validation (or template mode) — use the
        # guaranteed-grammatical fallback built directly from the source sentence.
        return self._generate_template(sentence, answer_span), False

    @staticmethod
    def _is_valid_question(question: str, sentence: str, answer_span: str) -> bool:
        """Rejects degenerate / nonsensical T5 outputs before they reach the user."""
        q = question.strip()
        words = q.split()

        if len(words) < 4 or len(words) > 40:
            return False
        if not q.endswith("?"):
            return False
        # Reject heavy word repetition (a common small-T5 failure mode)
        lower_words = [w.lower().strip(string_punct_module.punctuation) for w in words]
        lower_words = [w for w in lower_words if w]
        if lower_words and len(set(lower_words)) / len(lower_words) < 0.55:
            return False
        # Reject if the answer itself leaked verbatim into the question
        # (defeats the purpose of the question)
        if answer_span.lower() in q.lower():
            return False
        # Reject near-duplicate of the source sentence (T5 sometimes just echoes it)
        if q.lower().rstrip("?").strip() == sentence.lower().rstrip(".").strip():
            return False
        return True

    def _generate_qg_hl(self, sentence: str, answer_span: str) -> Optional[str]:
        try:
            pattern = r"\b" + re.escape(answer_span) + r"\b"
            highlighted = re.sub(pattern, f"<hl> {answer_span} <hl>", sentence, count=1, flags=re.IGNORECASE)
            if "<hl>" not in highlighted:
                highlighted = f"<hl> {answer_span} <hl> {sentence}"
            input_text = f"generate question: {highlighted}"
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return self._clean_question(question)
        except Exception as e:
            logger.warning(f"QG-hl generation failed ({e})")
            return None

    def _generate_t5_prompt(self, sentence: str, answer_span: str) -> Optional[str]:
        try:
            input_text = f"generate question for answer {answer_span}: context: {sentence}"
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return self._clean_question(question)
        except Exception as e:
            logger.warning(f"T5-prompt generation failed ({e})")
            return None

    def _generate_template(self, sentence: str, answer_span: str) -> Optional[str]:
        """Reliable fallback: blanks out the answer span in the original sentence.
        Always grammatically valid since it's built from real source text."""
        pattern = r"\b" + re.escape(answer_span) + r"\b"
        blanked = re.sub(pattern, "______", sentence, count=1, flags=re.IGNORECASE)
        if "______" not in blanked:
            return None
        return blanked.strip()

    @staticmethod
    def _clean_question(q: str) -> str:
        q = q.strip()
        if not q:
            return q
        q = q[0].upper() + q[1:]
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        return q


# ════════════════════════════════════════════════════════════════
# QUALITY SCORER (NumPy dot-product over several weighted signals)
# ════════════════════════════════════════════════════════════════
class QualityScorer:
    W = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)

    def score(self, question: str, answer: str, distractors: List[str], importance: float) -> float:
        words = question.split()
        n = len(words)
        f1 = float(np.clip(importance, 0, 1))
        f2 = 1.0 if 2 <= len(answer.split()) <= 4 else 0.6
        f3 = 1.0 if 6 <= n <= 22 else (0.4 if n < 6 else 0.5)
        if distractors:
            a_set = set(answer.lower())
            j_scores = []
            for d in distractors:
                d_set = set(d.lower())
                union = a_set | d_set
                inter = a_set & d_set
                j = len(inter) / len(union) if union else 0
                j_scores.append(1.0 - j)
            f4 = float(np.mean(j_scores))
        else:
            f4 = 0.0
        f5 = 1.0 if question.strip().endswith("?") or "______" in question else 0.7
        f = np.array([f1, f2, f3, f4, f5], dtype=np.float64)
        return float(np.clip(np.dot(self.W, f), 0.0, 1.0))


# ════════════════════════════════════════════════════════════════
# MAIN ENGINE — ties embeddings + T5 generation + clustering +
# difficulty classification + distractor mining together
# ════════════════════════════════════════════════════════════════
class DeepQuizEngine:
    def __init__(self):
        self.embedder = EmbeddingModel.get()
        self.qgen = QuestionGenerator.get()
        self.scorer = QualityScorer()
        # Imported lazily to avoid a circular import at module load time
        from difficulty_model import DifficultyClassifier
        self.classifier = DifficultyClassifier(embedder=self.embedder)
        logger.info(
            f"🚀 DeepQuizEngine ready | embedder={'ok' if self.embedder.available() else 'FAILED'} "
            f"| question_gen_mode={self.qgen.mode} "
            f"| difficulty_model={'fine-tuned' if self.classifier.is_trained() else 'heuristic (untrained)'}"
        )

    def _rank_by_centrality(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity of each sentence embedding to the document centroid."""
        if embeddings.shape[0] < 2:
            return np.ones(embeddings.shape[0])
        centroid = embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(embeddings, centroid).flatten()
        return np.clip(sims, 0.0, 1.0)

    def _cluster_topics(self, embeddings: np.ndarray, n_clusters: int = 6) -> np.ndarray:
        n = embeddings.shape[0]
        k = min(n_clusters, max(n // 4, 2))
        if k < 2 or n < 2:
            return np.zeros(n, dtype=int)
        X = normalize(embeddings, norm="l2")
        model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = model.fit_predict(X)
        logger.info(f"✓ KMeans clustering: {k} topic clusters")
        return labels

    def _semantic_distractors(
        self, answer: str, answer_sentence: str, all_sentences: List[str],
        sentence_embeddings: np.ndarray, sentence_idx: int, n: int = 3
    ) -> List[str]:
        """
        Mines distractors using embedding similarity: pull noun phrases from
        sentences that are semantically *related* (similar topic) but not the
        same sentence, so distractors feel plausible (same domain) rather than
        random or unrelated. Prefers phrases with a similar word count to the
        answer so options look consistent (e.g. not mixing "Sort" with
        "time complexity").
        """
        out = []
        answer_word_count = len(answer.split())
        if sentence_embeddings.shape[0] > 1:
            sims = cosine_similarity(
                sentence_embeddings[sentence_idx:sentence_idx + 1], sentence_embeddings
            ).flatten()
            order = np.argsort(sims)[::-1]
            seen = {answer.lower()}
            # First pass: prefer phrases with similar length/style to the answer
            for idx in order:
                if idx == sentence_idx or len(out) >= n:
                    continue
                phrases = extract_noun_phrases(all_sentences[idx])
                for phrase, _tag in phrases:
                    if (
                        phrase.lower() not in seen
                        and abs(len(phrase.split()) - answer_word_count) <= 1
                    ):
                        out.append(phrase)
                        seen.add(phrase.lower())
                        break
            # Second pass: fill remaining slots with any other related phrase
            if len(out) < n:
                for idx in order:
                    if idx == sentence_idx or len(out) >= n:
                        continue
                    phrases = extract_noun_phrases(all_sentences[idx])
                    for phrase, _tag in phrases:
                        if phrase.lower() not in seen and len(out) < n:
                            out.append(phrase)
                            seen.add(phrase.lower())
        return out

    def _build_distractors(
        self, answer: str, pos: str, answer_sentence: str, all_sentences: List[str],
        sentence_embeddings: np.ndarray, sentence_idx: int
    ) -> List[str]:
        sem = self._semantic_distractors(answer, answer_sentence, all_sentences, sentence_embeddings, sentence_idx, n=3)
        wn = get_wordnet_distractors(answer, pos, n=3) if len(sem) < 3 else []
        combined, seen = [], {answer.lower()}
        for d in sem + wn:
            if d.lower() not in seen and d.replace(" ", "").isalpha():
                combined.append(d)
                seen.add(d.lower())
        # Generic fallbacks only used as an absolute last resort, and only
        # for single-word answers (avoids "Bubble Sort" vs "process" mismatches)
        if len(combined) < 3 and len(answer.split()) == 1:
            fallbacks = ["process", "system", "method", "structure", "mechanism", "principle", "concept", "framework"]
            while len(combined) < 3:
                fb = random.choice(fallbacks)
                if fb not in seen:
                    combined.append(fb)
                    seen.add(fb)
        random.shuffle(combined)
        return combined[:3] if len(combined) >= 3 else combined

    def generate(self, raw_text: str, limit: int) -> List[Dict]:
        t0 = time.time()
        text = re.sub(r"\s+", " ", raw_text).strip()
        sentences = split_sentences(text)
        if len(sentences) < 2:
            sentences = [s.strip() for s in text.split(".") if len(s.split()) >= 6]
        if not sentences:
            sentences = [text]
        logger.info(f"✓ {len(sentences)} candidate sentences extracted")

        if not self.embedder.available():
            raise RuntimeError(
                "Sentence embedding model is not available. This usually means the model "
                "could not be downloaded on first run — check your internet connection."
            )

        embeddings = self.embedder.encode(sentences, show_progress_bar=False)
        importance = self._rank_by_centrality(embeddings)
        clusters = self._cluster_topics(embeddings)
        ranked_idx = np.argsort(importance)[::-1]

        results: List[Dict] = []
        used_answers: set = set()
        used_sentences: set = set()

        for idx in ranked_idx:
            if len(results) >= limit:
                break
            if idx in used_sentences:
                continue
            sent = sentences[idx]
            result = get_key_noun_phrase(sent)
            if not result:
                continue
            answer, pos = result
            if answer.lower() in used_answers:
                continue

            distractors = self._build_distractors(answer, pos, sent, sentences, embeddings, idx)
            if len(distractors) < 3:
                # Not enough plausible wrong answers could be mined for this
                # term — skip rather than show a question with weak/duplicate options.
                continue

            question, was_neural = self.qgen.generate(sent, answer)
            if not question:
                continue
            question_type = "neural_qg" if was_neural else "fill_blank"

            all_options = distractors[:3] + [answer]
            random.shuffle(all_options)
            correct_idx = all_options.index(answer)

            difficulty = self.classifier.predict(question)
            quality = self.scorer.score(question, answer, distractors, float(importance[idx]))

            results.append({
                "question": question,
                "correct": answer,
                "options": all_options,
                "correct_index": correct_idx,
                "difficulty": difficulty,
                "topic_cluster": int(clusters[idx]),
                "quality_score": round(quality, 3),
                "question_type": question_type,
            })
            used_answers.add(answer.lower())
            used_sentences.add(idx)

        # If the transformer + heuristics couldn't fill the quota (e.g. very
        # short input), top up with simple fill-blank questions on remaining
        # sentences rather than returning fewer than requested.
        if len(results) < limit:
            for idx in ranked_idx:
                if len(results) >= limit:
                    break
                if idx in used_sentences:
                    continue
                sent = sentences[idx]
                phrases = extract_noun_phrases(sent)
                phrases = [(p, t) for p, t in phrases if p.lower() not in used_answers]
                for phrase, pos in phrases[:2]:
                    if len(results) >= limit:
                        break
                    pattern = r"\b" + re.escape(phrase) + r"\b"
                    q_text = re.sub(pattern, "______", sent, count=1, flags=re.IGNORECASE)
                    if "______" not in q_text:
                        continue
                    distractors = self._build_distractors(phrase, pos, sent, sentences, embeddings, idx)
                    if len(distractors) < 3:
                        continue
                    all_options = distractors[:3] + [phrase]
                    random.shuffle(all_options)
                    correct_idx = all_options.index(phrase)
                    difficulty = self.classifier.predict(q_text)
                    quality = self.scorer.score(q_text, phrase, distractors, float(importance[idx]))
                    results.append({
                        "question": q_text.strip(),
                        "correct": phrase,
                        "options": all_options,
                        "correct_index": correct_idx,
                        "difficulty": difficulty,
                        "topic_cluster": int(clusters[idx]),
                        "quality_score": round(quality, 3),
                        "question_type": "fill_blank",
                    })
                    used_answers.add(phrase.lower())
                used_sentences.add(idx)

        results.sort(key=lambda r: r["quality_score"], reverse=True)
        elapsed = round(time.time() - t0, 3)
        logger.info(f"📈 Generated {len(results)} questions in {elapsed}s")
        return results[:limit]


_engine_singleton: Optional[DeepQuizEngine] = None


def get_engine() -> DeepQuizEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = DeepQuizEngine()
    return _engine_singleton