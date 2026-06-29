"""
╔══════════════════════════════════════════════════════════════════╗
║   UoH AI Quiz Engine v5.0  —  Advanced ML Pipeline             ║
║   University of Haripur  —  5th Semester Project               ║
║                                                                  ║
║   Muhammad Saad Jadoon · BS AI                                  ║
║                                                                  ║
║   🧠 Full ML Stack:                                             ║
║   ✓ spaCy / NLTK NLP       (real sentence parsing)             ║
║   ✓ TF-IDF Vectorizer      (text → numerical features)         ║
║   ✓ Cosine Similarity      (sentence importance ranking)        ║
║   ✓ KMeans Clustering      (topic grouping)                    ║
║   ✓ Naive Bayes            (difficulty classification)          ║
║   ✓ NumPy Dot Product      (quality scoring)                   ║
║   ✓ Pandas Analytics       (statistics & reporting)             ║
║   ✓ Wh-Question Generation (Who/What/When/Where/How)           ║
║   ✓ Fill-in-Blank MCQs     (proper keyphrase targeting)        ║
║   ✓ Distractor Engine      (semantic + WordNet distractors)    ║
╚══════════════════════════════════════════════════════════════════╝

INSTALL DEPENDENCIES:
    pip install fastapi uvicorn sqlalchemy werkzeug numpy pandas scikit-learn nltk

FIRST RUN – download NLTK data (run once):
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
"""

import re
import time
import random
import logging
import json
import string
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

# ═══ Web Framework & DB ═══
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from werkzeug.security import generate_password_hash, check_password_hash

# ═══ ML Libraries ═══
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ═══ NLP ═══
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer

# Auto-download required NLTK data
for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'omw-1.4', 'punkt_tab',
            'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("UOH_QUIZ_V5")

# ════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════
DATABASE_URL = "sqlite:///./uoh_quiz_v5.db"
engine_db = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine_db, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("QuizSession", back_populates="owner")


class QuizSession(Base):
    __tablename__ = "quiz_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255))
    content_summary = Column(Text)
    total_questions = Column(Integer)
    quiz_type = Column(String(20), default="standard")
    processing_time = Column(Float)
    ml_pipeline = Column(String(200), default="TF-IDF|Cosine|KMeans|NaiveBayes|NumPy|Pandas")
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="sessions")
    questions = relationship("QuestionBank", back_populates="session_parent")


class QuestionBank(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("quiz_sessions.id"))
    question_body = Column(Text, nullable=False)
    correct_ans = Column(String(255))
    distractors_json = Column(Text)
    difficulty = Column(String(10), default="medium")
    topic_cluster = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    question_type = Column(String(30), default="fill_blank")
    session_parent = relationship("QuizSession", back_populates="questions")


Base.metadata.create_all(bind=engine_db)


# ════════════════════════════════════════════════════════════════
# NLP UTILITIES
# ════════════════════════════════════════════════════════════════
STOP_WORDS = set(stopwords.words('english')) | {
    'also', 'however', 'therefore', 'moreover', 'furthermore',
    'thus', 'hence', 'consequently', 'meanwhile', 'nevertheless',
    'nonetheless', 'otherwise', 'instead', 'accordingly'
}

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Normalize whitespace and unicode artifacts."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Tokenize text into sentences using NLTK."""
    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.split()) >= 6]


def get_pos_tags(sentence: str) -> List[Tuple[str, str]]:
    """Get part-of-speech tags for a sentence."""
    try:
        tokens = word_tokenize(sentence)
        return pos_tag(tokens)
    except Exception:
        return []


def extract_noun_phrases(sentence: str) -> List[str]:
    """Extract meaningful noun phrases from a sentence."""
    tagged = get_pos_tags(sentence)
    if not tagged:
        return []

    noun_phrases = []
    current_np = []

    for word, tag in tagged:
        if tag.startswith('NN') or tag.startswith('JJ') and current_np:
            current_np.append(word)
        elif tag in ('DT', 'PRP$'):
            if current_np:
                phrase = ' '.join(current_np)
                if len(phrase.split()) >= 1 and phrase.lower() not in STOP_WORDS:
                    noun_phrases.append(phrase)
                current_np = []
        else:
            if current_np:
                phrase = ' '.join(current_np)
                if len(phrase.split()) >= 1 and phrase.lower() not in STOP_WORDS:
                    noun_phrases.append(phrase)
                current_np = []

    if current_np:
        phrase = ' '.join(current_np)
        if phrase.lower() not in STOP_WORDS:
            noun_phrases.append(phrase)

    return [np for np in noun_phrases if len(np) > 2]


def get_key_noun(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Return the single best (keyword, pos_tag) to blank out.
    Prefer: proper nouns > nouns > key adjectives/verbs.
    """
    tagged = get_pos_tags(sentence)
    if not tagged:
        return None

    candidates = []
    for word, tag in tagged:
        if word.lower() in STOP_WORDS or len(word) <= 2 or not word.isalpha():
            continue
        if tag == 'NNP':  # Proper noun — highest priority
            candidates.append((word, tag, 5))
        elif tag == 'NNPS':
            candidates.append((word, tag, 4))
        elif tag in ('NN', 'NNS'):  # Common noun
            candidates.append((word, tag, 3))
        elif tag.startswith('VB') and len(word) > 4:  # Verb
            candidates.append((word, tag, 2))
        elif tag.startswith('JJ') and len(word) > 4:  # Adjective
            candidates.append((word, tag, 1))

    if not candidates:
        return None

    # Sort by priority desc, then length desc (longer = more specific)
    candidates.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
    best = candidates[0]
    return (best[0], best[1])


# ════════════════════════════════════════════════════════════════
# DISTRACTOR ENGINE
# ════════════════════════════════════════════════════════════════

def get_wordnet_distractors(word: str, pos_tag_str: str, n: int = 3) -> List[str]:
    """
    Get semantically-related distractors from WordNet.
    Uses synsets, hypernyms, and co-hyponyms.
    """
    wn_pos = wordnet.NOUN
    if pos_tag_str.startswith('VB'):
        wn_pos = wordnet.VERB
    elif pos_tag_str.startswith('JJ'):
        wn_pos = wordnet.ADJ

    candidates = set()

    synsets = wordnet.synsets(word.lower(), pos=wn_pos)

    for syn in synsets[:3]:
        # Lemmas of the same synset (synonyms)
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                candidates.add(name)

        # Hypernym lemmas
        for hyper in syn.hypernyms()[:2]:
            for lemma in hyper.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    candidates.add(name)

        # Co-hyponyms (siblings)
        for hyper in syn.hypernyms()[:1]:
            for hypo in hyper.hyponyms()[:6]:
                if hypo != syn:
                    for lemma in hypo.lemmas():
                        name = lemma.name().replace('_', ' ')
                        if name.lower() != word.lower():
                            candidates.add(name)

    distractors = [c for c in candidates if c.isalpha() or ' ' in c]
    random.shuffle(distractors)
    return distractors[:n]


def get_context_distractors(word: str, all_words: List[str], n: int = 3) -> List[str]:
    """
    Get distractors from same document (same POS, similar length).
    These are plausible because they come from the same domain.
    """
    word_lower = word.lower()
    candidates = list({
        w for w in all_words
        if w.lower() != word_lower
        and w.isalpha()
        and abs(len(w) - len(word)) <= 3
        and w.lower() not in STOP_WORDS
        and len(w) > 3
    })
    random.shuffle(candidates)
    return [c.capitalize() if word[0].isupper() else c for c in candidates[:n]]


def build_distractors(word: str, pos_tag_str: str,
                      context_words: List[str]) -> List[str]:
    """
    Combine WordNet + context-based distractors.
    Deduplicate and return exactly 3 plausible wrong options.
    """
    wn = get_wordnet_distractors(word, pos_tag_str, n=5)
    ctx = get_context_distractors(word, context_words, n=5)

    combined = []
    seen = {word.lower()}
    for d in wn + ctx:
        if d.lower() not in seen and d.isalpha():
            combined.append(d)
            seen.add(d.lower())

    # Fallbacks if not enough
    fallbacks = ["process", "system", "method", "element", "structure",
                 "function", "theory", "mechanism", "concept", "principle"]

    while len(combined) < 3:
        fb = random.choice(fallbacks)
        if fb not in seen:
            combined.append(fb)
            seen.add(fb)

    random.shuffle(combined)
    return combined[:3]


# ════════════════════════════════════════════════════════════════
# WH-QUESTION GENERATOR
# ════════════════════════════════════════════════════════════════

DEFINITION_PATTERNS = [
    (r'^(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+)', 'definition'),
    (r'^(.+?)\s+(?:refers to|means|defined as|known as)\s+(.+)', 'definition'),
    (r'^(.+?)\s+(?:consists of|comprises|contains|includes)\s+(.+)', 'composition'),
    (r'^(.+?)\s+(?:was|were|is|are)\s+(?:discovered|invented|founded|created|developed)\s+(?:by|in)\s+(.+)', 'discovery'),
    (r'^(?:in|during|after|before)\s+(\d{4}|the \w+),?\s+(.+)', 'temporal'),
    (r'^(.+?)\s+(?:has|have|had)\s+(.+)', 'property'),
]


def make_wh_question(sentence: str, subject: str, predicate: str,
                     pattern_type: str) -> Optional[str]:
    """Create a grammatically-correct Wh- question."""
    sub = subject.strip()
    pred = predicate.strip().rstrip('.')

    templates = {
        'definition': [
            f"What is {sub}?",
            f"How is {sub} defined?",
            f"Which of the following best describes {sub}?",
        ],
        'composition': [
            f"What does {sub} consist of?",
            f"What are the components of {sub}?",
        ],
        'discovery': [
            f"Who is associated with {sub}?",
            f"How was {sub} developed?",
        ],
        'temporal': [
            f"When did the following occur: {pred[:80]}?",
        ],
        'property': [
            f"What characterizes {sub}?",
            f"Which statement about {sub} is correct?",
        ],
    }

    choices = templates.get(pattern_type, [f"Which statement about {sub} is correct?"])
    return random.choice(choices)


def try_make_wh_mcq(sentence: str, all_sentences: List[str],
                    classifier, scorer, context_words: List[str],
                    tfidf_score: float, cluster: int) -> Optional[Dict]:
    """
    Attempt to create a Wh-style question from a sentence.
    Returns MCQ dict or None.
    """
    for pattern, ptype in DEFINITION_PATTERNS:
        m = re.match(pattern, sentence.strip(), re.IGNORECASE)
        if m:
            subject = m.group(1).strip()
            predicate = m.group(2).strip() if len(m.groups()) >= 2 else sentence

            if len(subject.split()) > 8 or len(subject) < 2:
                continue

            correct_answer = predicate[:120].rstrip('.,;')
            question_text = make_wh_question(sentence, subject, predicate, ptype)

            if not question_text or not correct_answer:
                continue

            # Build wrong options from other sentences
            wrong_answers = []
            shuffled = all_sentences[:]
            random.shuffle(shuffled)
            for s in shuffled:
                if s != sentence and len(s.split()) >= 4:
                    candidate = s[:120].rstrip('.,;')
                    if candidate != correct_answer:
                        wrong_answers.append(candidate)
                    if len(wrong_answers) >= 3:
                        break

            while len(wrong_answers) < 3:
                wrong_answers.append(random.choice([
                    "None of the above",
                    "Cannot be determined from the text",
                    "All of the mentioned options",
                ]))

            wrong_answers = wrong_answers[:3]
            all_options = wrong_answers + [correct_answer]
            random.shuffle(all_options)

            correct_idx = all_options.index(correct_answer)
            difficulty = classifier.predict(question_text)
            quality = scorer.score(question_text, correct_answer,
                                   wrong_answers, tfidf_score) if scorer else 0.7

            return {
                "question": question_text,
                "correct": correct_answer,
                "options": all_options,
                "correct_index": correct_idx,
                "difficulty": difficulty,
                "topic_cluster": cluster,
                "quality_score": round(quality, 3),
                "question_type": "wh_question",
            }

    return None


# ════════════════════════════════════════════════════════════════
# MODULE 1: TF-IDF VECTORIZER
# ════════════════════════════════════════════════════════════════
class TFIDFModule:
    def __init__(self, max_features: int = 800):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            max_df=0.9,
        )
        self._fitted = False

    def fit_transform(self, sentences: List[str]) -> np.ndarray:
        if not sentences or len(sentences) < 2:
            return np.zeros((max(len(sentences), 1), 10))
        sparse = self.vectorizer.fit_transform(sentences)
        self._fitted = True
        return sparse.toarray()

    def get_top_keywords(self, n: int = 10) -> List[str]:
        if not self._fitted:
            return []
        names = self.vectorizer.get_feature_names_out()
        scores = self.vectorizer.idf_
        top = np.argsort(scores)[::-1][:n]
        return [names[i] for i in top]


# ════════════════════════════════════════════════════════════════
# MODULE 2: COSINE SIMILARITY RANKER
# ════════════════════════════════════════════════════════════════
class CosineSimilarityRanker:
    def rank(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        if tfidf_matrix.shape[0] < 2:
            return np.ones(tfidf_matrix.shape[0])
        centroid = tfidf_matrix.mean(axis=0, keepdims=True)
        cos_scores = cosine_similarity(tfidf_matrix, centroid).flatten()
        row_sums = tfidf_matrix.sum(axis=1)
        row_normalized = row_sums / (row_sums.max() + 1e-9)
        final = 0.65 * cos_scores + 0.35 * row_normalized
        return np.clip(final, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════
# MODULE 3: KMEANS CLUSTERING
# ════════════════════════════════════════════════════════════════
class KMeansTopicClusterer:
    def __init__(self, n_clusters: int = 6):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit_predict(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        n = tfidf_matrix.shape[0]
        k = min(self.n_clusters, max(n // 4, 2))
        if k < 2:
            self.labels_ = np.zeros(n, dtype=int)
            return self.labels_
        X = normalize(tfidf_matrix, norm='l2')
        model = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=600)
        self.labels_ = model.fit_predict(X)
        logger.info(f"✓ KMeans: {k} clusters | dist: {np.bincount(self.labels_).tolist()}")
        return self.labels_

    def get_label(self, idx: int) -> int:
        if self.labels_ is None or idx >= len(self.labels_):
            return 0
        return int(self.labels_[idx])


# ════════════════════════════════════════════════════════════════
# MODULE 4: NAIVE BAYES DIFFICULTY CLASSIFIER
# ════════════════════════════════════════════════════════════════
class DifficultyClassifier:
    TRAIN_X = [
        # EASY
        "what is", "who is", "when was", "what is the name", "define the term",
        "what does mean", "simple definition", "basic concept", "fill blank name",
        "identify the following", "what color", "list the items",
        # MEDIUM
        "explain how photosynthesis works in plants using sunlight",
        "describe the process of cell division in eukaryotes",
        "compare and contrast the differences between two systems",
        "how does the immune system respond to bacterial infection",
        "describe the relationship between pressure and volume",
        "explain the function of mitochondria in cell metabolism",
        "what are the types of polymorphism in object oriented programming",
        "how does machine learning classify data using neural networks",
        "discuss the role of enzymes in biochemical reactions",
        "explain the working principle behind transformer architecture",
        # HARD
        "critically analyze the impact of quantum computing on cryptography",
        "evaluate the effectiveness of gradient descent optimization algorithms",
        "synthesize evidence to support the hypothesis regarding climate change",
        "compare the computational complexity of sorting algorithms",
        "assess the theoretical limitations of neural network generalization",
        "interpret the statistical significance of experimental findings",
        "derive the mathematical relationship between entropy and information",
    ]
    TRAIN_Y = ["easy"] * 12 + ["medium"] * 10 + ["hard"] * 7

    def __init__(self):
        self.le = LabelEncoder()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=500)),
            ('nb', MultinomialNB(alpha=0.5)),
        ])
        y_enc = self.le.fit_transform(self.TRAIN_Y)
        self.pipeline.fit(self.TRAIN_X, y_enc)
        logger.info("✓ Naive Bayes classifier trained")

    def predict(self, text: str) -> str:
        try:
            y_enc = self.pipeline.predict([text])
            return str(self.le.inverse_transform(y_enc)[0])
        except Exception:
            return "medium"


# ════════════════════════════════════════════════════════════════
# MODULE 5: NUMPY QUALITY SCORER
# ════════════════════════════════════════════════════════════════
class NumpyQualityScorer:
    W = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)

    def __init__(self, word_freq: Dict[str, int]):
        total = max(sum(word_freq.values()), 1)
        self.prob = {w: c / total for w, c in word_freq.items()}

    def score(self, question: str, answer: str,
              distractors: List[str], tfidf_score: float) -> float:
        words = question.split()
        n = len(words)

        f1 = float(np.clip(tfidf_score, 0, 1))

        p = self.prob.get(answer.lower().split()[0], 0.001)
        f2 = float(np.clip(1.0 - p * 30, 0, 1))

        if 8 <= n <= 25:
            f3 = 1.0
        elif n < 5:
            f3 = 0.2
        elif n > 50:
            f3 = 0.3
        else:
            f3 = 0.6

        if distractors:
            c_set = set(answer.lower())
            jaccard_scores = []
            for d in distractors:
                d_set = set(d.lower())
                union = c_set | d_set
                inter = c_set & d_set
                j = len(inter) / len(union) if union else 0
                jaccard_scores.append(1.0 - j)
            f4 = float(np.mean(jaccard_scores))
        else:
            f4 = 0.0

        has_blank = "______" in question or "_____" in question
        f5 = 1.0 if has_blank else 0.7

        f = np.array([f1, f2, f3, f4, f5], dtype=np.float64)
        return float(np.clip(np.dot(self.W, f), 0.0, 1.0))


# ════════════════════════════════════════════════════════════════
# MAIN AI QUIZ ENGINE
# ════════════════════════════════════════════════════════════════
class AIQuizEngine:
    """
    Full ML-based MCQ Generator
    Pipeline: NLP → TF-IDF → Cosine → KMeans → NaiveBayes → NumPy → Pandas
    """

    def __init__(self):
        self._tfidf = TFIDFModule(max_features=800)
        self._ranker = CosineSimilarityRanker()
        self._clusterer = KMeansTopicClusterer(n_clusters=6)
        self._classifier = DifficultyClassifier()
        self._scorer: Optional[NumpyQualityScorer] = None
        logger.info("🚀 AIQuizEngine v5.0 initialized")

    def _get_word_freq(self, text: str) -> Dict[str, int]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return Counter(words)

    def _get_all_nouns(self, text: str) -> List[str]:
        """Extract all meaningful nouns/keyphrases from the text."""
        tagged = get_pos_tags(text)
        nouns = [
            word for word, tag in tagged
            if tag.startswith('NN') and word.lower() not in STOP_WORDS
            and len(word) > 3 and word.isalpha()
        ]
        return nouns

    def _make_fill_blank_mcq(self, sentence: str, context_words: List[str],
                              tfidf_score: float, cluster: int) -> Optional[Dict]:
        """
        Create a high-quality fill-in-the-blank MCQ.
        Uses POS tagging to select the best keyword to blank out.
        """
        result = get_key_noun(sentence)
        if not result:
            return None

        target_word, pos = result

        # Don't blank very short or very common words
        if len(target_word) <= 3 or target_word.lower() in STOP_WORDS:
            return None

        # Create question with blank
        pattern = r'\b' + re.escape(target_word) + r'\b'
        q_text = re.sub(pattern, '______', sentence, count=1, flags=re.IGNORECASE)
        q_text = q_text.strip()

        if '______' not in q_text:
            return None

        # Ensure question ends properly
        if not q_text.endswith(('?', '.', '!')):
            q_text += '.'

        # Build distractors
        distractors = build_distractors(target_word, pos, context_words)

        # Assemble options
        all_options = distractors[:3] + [target_word]
        random.shuffle(all_options)
        correct_idx = all_options.index(target_word)

        difficulty = self._classifier.predict(q_text)
        quality = self._scorer.score(q_text, target_word,
                                     distractors, tfidf_score) if self._scorer else 0.65

        return {
            "question": q_text,
            "correct": target_word,
            "options": all_options,
            "correct_index": correct_idx,
            "difficulty": difficulty,
            "topic_cluster": cluster,
            "quality_score": round(quality, 3),
            "question_type": "fill_blank",
        }

    def generate(self, raw_text: str, limit: int) -> List[Dict]:
        """
        FULL ML PIPELINE:
        1. Clean & tokenize (NLTK)
        2. TF-IDF vectorization
        3. Cosine similarity ranking
        4. KMeans topic clustering
        5. Wh-question generation (definition-based)
        6. Fill-in-blank generation (POS-targeted)
        7. Naive Bayes difficulty classification
        8. NumPy quality scoring
        9. Pandas analytics & reporting
        """
        text = clean_text(raw_text)
        word_freq = self._get_word_freq(text)
        self._scorer = NumpyQualityScorer(word_freq)
        context_words = self._get_all_nouns(text)

        results: List[Dict] = []
        used_answers: set = set()
        used_sentences: set = set()

        t0 = time.time()

        # ─── Sentence tokenization ───
        sentences = split_sentences(text)
        if len(sentences) < 2:
            # Fallback: split by period
            sentences = [s.strip() for s in text.split('.') if len(s.split()) >= 6]
        if not sentences:
            sentences = [text]

        logger.info(f"✓ Found {len(sentences)} sentences")

        # ─── TF-IDF ───
        tfidf_matrix = self._tfidf.fit_transform(sentences)
        keywords = self._tfidf.get_top_keywords(8)
        logger.info(f"✓ TF-IDF {tfidf_matrix.shape} | Keywords: {keywords}")

        # ─── Cosine Ranking ───
        importance = self._ranker.rank(tfidf_matrix)

        # ─── KMeans Clustering ───
        clusters = self._clusterer.fit_predict(tfidf_matrix)

        # ─── Ranked sentence indices ───
        ranked_idx = np.argsort(importance)[::-1]

        # ─── Pass 1: Wh-Questions from definition-pattern sentences ───
        for idx in ranked_idx:
            if len(results) >= limit:
                break
            sent = sentences[idx]
            if idx in used_sentences:
                continue
            mcq = try_make_wh_mcq(
                sent, sentences, self._classifier, self._scorer,
                context_words, float(importance[idx]), int(clusters[idx])
            )
            if mcq and mcq['correct'].lower() not in used_answers:
                used_answers.add(mcq['correct'].lower()[:40])
                used_sentences.add(idx)
                results.append(mcq)

        # ─── Pass 2: Fill-in-blank MCQs ───
        for idx in ranked_idx:
            if len(results) >= limit:
                break
            sent = sentences[idx]
            if idx in used_sentences:
                continue
            mcq = self._make_fill_blank_mcq(
                sent, context_words,
                float(importance[idx]), int(clusters[idx])
            )
            if mcq and mcq['correct'].lower() not in used_answers:
                used_answers.add(mcq['correct'].lower())
                used_sentences.add(idx)
                results.append(mcq)

        # ─── Pass 3: Recycle sentences with different targets ───
        if len(results) < limit:
            for idx in ranked_idx:
                if len(results) >= limit:
                    break
                sent = sentences[idx]
                # Try all possible nouns in this sentence
                tagged = get_pos_tags(sent)
                words = [(w, t) for w, t in tagged
                         if t.startswith('NN') and len(w) > 3
                         and w.lower() not in STOP_WORDS
                         and w.lower() not in used_answers]
                random.shuffle(words)
                for word, pos in words[:3]:
                    if len(results) >= limit:
                        break
                    pattern = r'\b' + re.escape(word) + r'\b'
                    q_text = re.sub(pattern, '______', sent, count=1, flags=re.IGNORECASE)
                    if '______' not in q_text:
                        continue
                    distractors = build_distractors(word, pos, context_words)
                    all_options = distractors[:3] + [word]
                    random.shuffle(all_options)
                    correct_idx = all_options.index(word)
                    diff = self._classifier.predict(q_text)
                    qual = self._scorer.score(q_text, word, distractors, float(importance[idx]))
                    mcq = {
                        "question": q_text.strip(),
                        "correct": word,
                        "options": all_options,
                        "correct_index": correct_idx,
                        "difficulty": diff,
                        "topic_cluster": int(clusters[idx]),
                        "quality_score": round(qual, 3),
                        "question_type": "fill_blank",
                    }
                    used_answers.add(word.lower())
                    results.append(mcq)

        # ─── Sort by quality score (NumPy) ───
        if results:
            q_scores = np.array([r['quality_score'] for r in results])
            sorted_idx = np.argsort(q_scores)[::-1]
            results = [results[i] for i in sorted_idx]

        # ─── Pandas analytics ───
        elapsed = round(time.time() - t0, 3)
        if results:
            df = pd.DataFrame(results)
            logger.info(
                f"\n📊 Difficulty Distribution:\n"
                f"{df['difficulty'].value_counts().to_string()}\n"
                f"📈 Avg Quality: {df['quality_score'].mean():.3f} | "
                f"Time: {elapsed}s | Total: {len(results)}"
            )

        return results[:limit]


# Singleton
ai_engine = AIQuizEngine()


# ════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ════════════════════════════════════════════════════════════════

class SignupSchema(BaseModel):
    username: str
    email: str
    password: str

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        v = v.strip()
        if len(v) < 3 or len(v) > 30:
            raise ValueError('Username: 3-30 characters')
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Only letters, numbers, underscores')
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        v = v.strip().lower()
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', v):
            raise ValueError('Invalid email format')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        errors = []
        if len(v) < 8: errors.append('8+ chars')
        if not re.search(r'[A-Z]', v): errors.append('uppercase')
        if not re.search(r'[a-z]', v): errors.append('lowercase')
        if not re.search(r'\d', v): errors.append('digit')
        if not re.search(r'[!@#$%^&*(),.?]', v): errors.append('special char')
        if errors:
            raise ValueError(f'Password needs: {", ".join(errors)}')
        return v


class LoginSchema(BaseModel):
    username: str
    password: str


class QuizRequest(BaseModel):
    user_id: int
    text_content: str
    count: int = 10

    @field_validator('count')
    @classmethod
    def validate_count(cls, v):
        if not (1 <= v <= 100):
            raise ValueError('Count must be 1-100')
        return v

    @field_validator('text_content')
    @classmethod
    def validate_text(cls, v):
        v = v.strip()
        if len(v) < 50:
            raise ValueError('Minimum 50 characters of academic text required')
        return v


# ════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ════════════════════════════════════════════════════════════════

app = FastAPI(
    title="UoH AI Quiz Engine v5.0",
    description="NLP + TF-IDF + Cosine + KMeans + NaiveBayes + NumPy + Pandas",
    version="5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.post("/auth/signup")
def signup(data: SignupSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(400, "Username already taken")
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "Email already registered")
    user = User(
        username=data.username,
        email=data.email,
        password=generate_password_hash(data.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"status": "success", "message": "Account created!", "user_id": user.id, "username": user.username}


@app.post("/auth/login")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or not check_password_hash(user.password, data.password):
        raise HTTPException(401, "Invalid credentials")
    return {"status": "success", "user_id": user.id, "username": user.username, "email": user.email}


# ════════════════════════════════════════════════════════════════
# QUIZ ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.post("/api/v1/generate-quiz")
async def generate_quiz(req: QuizRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        raise HTTPException(404, "User not found")

    t0 = time.time()
    questions = ai_engine.generate(req.text_content, req.count)
    elapsed = round(time.time() - t0, 3)

    if not questions:
        raise HTTPException(
            422,
            "Could not generate questions. Please provide more detailed academic text "
            "(at least 3-5 sentences with clear concepts, definitions, or facts)."
        )

    quiz_type_map = {10: "quick", 25: "standard", 50: "extended", 100: "full"}
    quiz_type = quiz_type_map.get(req.count, "standard" if req.count <= 25 else "extended")

    # Save session
    session = QuizSession(
        user_id=req.user_id,
        title=f"Quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        content_summary=req.text_content[:300],
        total_questions=len(questions),
        quiz_type=quiz_type,
        processing_time=elapsed,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    for q in questions:
        db.add(QuestionBank(
            session_id=session.id,
            question_body=q['question'],
            correct_ans=q['correct'],
            distractors_json='|'.join(q['options']),
            difficulty=q['difficulty'],
            topic_cluster=q['topic_cluster'],
            quality_score=q['quality_score'],
            question_type=q.get('question_type', 'fill_blank'),
        ))
    db.commit()

    df = pd.DataFrame(questions)
    stats = {
        "easy": int((df['difficulty'] == 'easy').sum()),
        "medium": int((df['difficulty'] == 'medium').sum()),
        "hard": int((df['difficulty'] == 'hard').sum()),
        "avg_quality": round(float(df['quality_score'].mean()), 3),
        "clusters": int(df['topic_cluster'].nunique()),
        "wh_questions": int((df['question_type'] == 'wh_question').sum()),
        "fill_blank": int((df['question_type'] == 'fill_blank').sum()),
    }

    return {
        "session_id": session.id,
        "time": f"{elapsed}s",
        "total": len(questions),
        "quiz_type": quiz_type,
        "ml_pipeline": "NLTK → TF-IDF → Cosine → KMeans → NaiveBayes → NumPy → Pandas",
        "quiz": questions,
        "stats": stats,
    }


@app.get("/api/v1/history/{user_id}")
def get_history(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    sessions = (
        db.query(QuizSession)
        .filter(QuizSession.user_id == user_id)
        .order_by(QuizSession.created_at.desc())
        .limit(50)
        .all()
    )
    return {
        "user": user.username,
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": s.id,
                "title": s.title,
                "questions": s.total_questions,
                "quiz_type": s.quiz_type,
                "processing_time": f"{s.processing_time}s",
                "created_at": s.created_at.strftime("%d %b %Y %H:%M"),
            }
            for s in sessions
        ],
    }


@app.get("/api/v1/ml-info")
def ml_info():
    return {
        "engine": "UoH AI Quiz Engine v5.0",
        "pipeline_steps": [
            {"step": 1, "name": "NLTK Sentence Tokenizer", "purpose": "Split text into sentences"},
            {"step": 2, "name": "POS Tagger", "purpose": "Identify nouns, verbs, adjectives"},
            {"step": 3, "name": "TF-IDF Vectorizer", "purpose": "Text → numerical features"},
            {"step": 4, "name": "Cosine Similarity", "purpose": "Rank sentence importance"},
            {"step": 5, "name": "KMeans Clustering", "purpose": "Group sentences by topic"},
            {"step": 6, "name": "Wh-Question Generator", "purpose": "Definition-based MCQs"},
            {"step": 7, "name": "Fill-Blank Generator", "purpose": "POS-targeted cloze MCQs"},
            {"step": 8, "name": "WordNet Distractors", "purpose": "Semantic wrong options"},
            {"step": 9, "name": "Naive Bayes", "purpose": "Difficulty classification"},
            {"step": 10, "name": "NumPy Quality Scorer", "purpose": "5-feature dot-product scoring"},
            {"step": 11, "name": "Pandas Analytics", "purpose": "Statistics & reporting"},
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "5.0", "database": "sqlite", "ml_pipeline": "active"}


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("🚀 UoH AI Quiz Engine v5.0")
    logger.info("📍 http://127.0.0.1:8000")
    logger.info("📚 Docs: http://127.0.0.1:8000/docs")
    logger.info("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)