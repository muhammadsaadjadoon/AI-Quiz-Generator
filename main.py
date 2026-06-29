"""
╔══════════════════════════════════════════════════════════════════╗
║   UoH AI Quiz Engine v6.0  —  Advanced ML Pipeline             ║
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
║   ✓ PDF Text Extraction    (PyMuPDF based)                     ║
║   ✓ 3000 MCQ Training Data (Naive Bayes difficulty classifier) ║
╚══════════════════════════════════════════════════════════════════╝

INSTALL DEPENDENCIES:
    pip install fastapi uvicorn sqlalchemy werkzeug numpy pandas scikit-learn nltk pymupdf python-multipart

FIRST RUN – download NLTK data (run once):
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
"""

import re
import time
import random
import logging
import json
import string
import io
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'omw-1.4', 'punkt_tab',
            'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("UOH_QUIZ_V6")

DATABASE_URL = "sqlite:///./uoh_quiz_v6.db"
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
    source_type = Column(String(20), default="text")
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

STOP_WORDS = set(stopwords.words('english')) | {
    'also', 'however', 'therefore', 'moreover', 'furthermore',
    'thus', 'hence', 'consequently', 'meanwhile', 'nevertheless',
    'nonetheless', 'otherwise', 'instead', 'accordingly'
}

lemmatizer = WordNetLemmatizer()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if not PDF_SUPPORT:
        raise HTTPException(500, "PDF support not available. Install pymupdf: pip install pymupdf")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            full_text.append(text.strip())
        doc.close()
        combined = "\n\n".join(full_text)
        combined = re.sub(r'\n([a-z])', r' \1', combined)
        combined = re.sub(r'-\n(\w)', r'\1', combined)
        combined = re.sub(r'\n{2,}', '. ', combined)
        combined = re.sub(r'\s+', ' ', combined)
        return combined.strip()
    except Exception as e:
        raise HTTPException(500, f"PDF extraction failed: {str(e)}")


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.split()) >= 6]


def get_pos_tags(sentence: str) -> List[Tuple[str, str]]:
    try:
        tokens = word_tokenize(sentence)
        return pos_tag(tokens)
    except Exception:
        return []


def extract_noun_phrases(sentence: str) -> List[str]:
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
    tagged = get_pos_tags(sentence)
    if not tagged:
        return None
    candidates = []
    for word, tag in tagged:
        if word.lower() in STOP_WORDS or len(word) <= 2 or not word.isalpha():
            continue
        if tag == 'NNP':
            candidates.append((word, tag, 5))
        elif tag == 'NNPS':
            candidates.append((word, tag, 4))
        elif tag in ('NN', 'NNS'):
            candidates.append((word, tag, 3))
        elif tag.startswith('VB') and len(word) > 4:
            candidates.append((word, tag, 2))
        elif tag.startswith('JJ') and len(word) > 4:
            candidates.append((word, tag, 1))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
    best = candidates[0]
    return (best[0], best[1])


def get_wordnet_distractors(word: str, pos_tag_str: str, n: int = 3) -> List[str]:
    wn_pos = wordnet.NOUN
    if pos_tag_str.startswith('VB'):
        wn_pos = wordnet.VERB
    elif pos_tag_str.startswith('JJ'):
        wn_pos = wordnet.ADJ
    candidates = set()
    synsets = wordnet.synsets(word.lower(), pos=wn_pos)
    for syn in synsets[:3]:
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                candidates.add(name)
        for hyper in syn.hypernyms()[:2]:
            for lemma in hyper.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    candidates.add(name)
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


def build_distractors(word: str, pos_tag_str: str, context_words: List[str]) -> List[str]:
    wn = get_wordnet_distractors(word, pos_tag_str, n=5)
    ctx = get_context_distractors(word, context_words, n=5)
    combined = []
    seen = {word.lower()}
    for d in wn + ctx:
        if d.lower() not in seen and d.isalpha():
            combined.append(d)
            seen.add(d.lower())
    fallbacks = ["process", "system", "method", "element", "structure",
                 "function", "theory", "mechanism", "concept", "principle"]
    while len(combined) < 3:
        fb = random.choice(fallbacks)
        if fb not in seen:
            combined.append(fb)
            seen.add(fb)
    random.shuffle(combined)
    return combined[:3]


DEFINITION_PATTERNS = [
    (r'^(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+)', 'definition'),
    (r'^(.+?)\s+(?:refers to|means|defined as|known as)\s+(.+)', 'definition'),
    (r'^(.+?)\s+(?:consists of|comprises|contains|includes)\s+(.+)', 'composition'),
    (r'^(.+?)\s+(?:was|were|is|are)\s+(?:discovered|invented|founded|created|developed)\s+(?:by|in)\s+(.+)', 'discovery'),
    (r'^(?:in|during|after|before)\s+(\d{4}|the \w+),?\s+(.+)', 'temporal'),
    (r'^(.+?)\s+(?:has|have|had)\s+(.+)', 'property'),
]


def make_wh_question(sentence: str, subject: str, predicate: str, pattern_type: str) -> Optional[str]:
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
            quality = scorer.score(question_text, correct_answer, wrong_answers, tfidf_score) if scorer else 0.7
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
# 3000 MCQ TRAINING DATA FOR NAIVE BAYES
# Yeh 3000 diverse examples hain (easy/medium/hard) jo NB ko
# properly train karte hain difficulty classification ke liye
# ════════════════════════════════════════════════════════════════

def build_training_data() -> Tuple[List[str], List[str]]:
    easy_templates = [
        "what is {concept}",
        "who is {person}",
        "when was {event}",
        "what is the name of {thing}",
        "define the term {term}",
        "what does {abbr} stand for",
        "identify {item}",
        "what color is {object}",
        "how many {units} are in {container}",
        "which of the following is {adjective}",
        "name the {ordinal} element of {group}",
        "what is the capital of {place}",
        "who invented {device}",
        "what is the symbol for {element}",
        "how do you spell {word}",
        "what year was {event} established",
        "which planet is {ordinal} from the sun",
        "what is the plural of {word}",
        "what type of {thing} is {example}",
        "name one example of {category}",
        "fill in the blank the {noun} is ______",
        "the process of photosynthesis produces ______",
        "water has the chemical formula ______",
        "the speed of light is approximately ______",
        "the human body has how many bones ______",
        "what is the boiling point of water",
        "who wrote {famous work}",
        "what is the largest {category}",
        "what language is spoken in {country}",
        "what is two plus two",
        "what is the opposite of {adjective}",
        "fill blank the sun is a ______",
        "fill blank humans breathe ______",
        "fill blank plants need ______ to grow",
        "what does cpu stand for in computers",
        "which organ pumps blood in human body",
        "what is the freezing point of water in celsius",
        "who is the father of computer science",
        "what is h2o commonly known as",
        "name the gas that plants absorb during photosynthesis",
    ]
    medium_templates = [
        "explain how {process} works in {context}",
        "describe the process of {biological process} in {organism}",
        "compare and contrast {concept a} and {concept b}",
        "how does the {system} respond to {stimulus}",
        "describe the relationship between {variable a} and {variable b}",
        "explain the function of {component} in {system}",
        "what are the types of {category} in {field}",
        "how does {technology} classify data using {method}",
        "discuss the role of {molecule} in {process}",
        "explain the working principle behind {architecture}",
        "what are the main differences between {a} and {b}",
        "how is {concept} applied in {real world context}",
        "describe the steps involved in {multi step process}",
        "what factors influence {phenomenon}",
        "explain the significance of {historical event} in {field}",
        "how do {organisms} adapt to {environment}",
        "what is the mechanism of action of {substance}",
        "describe the structure and function of {biological structure}",
        "how does {algorithm} improve efficiency in {domain}",
        "what are the advantages and disadvantages of {approach}",
        "explain the concept of {abstract idea} with an example",
        "how does {phenomenon} affect {outcome} in {context}",
        "describe the life cycle of {organism}",
        "what is the difference between {type a} and {type b} in {field}",
        "how are {items} classified in {taxonomy}",
        "explain the cause and effect of {event} on {system}",
        "what methods are used to measure {quantity} in {field}",
        "describe how {process} is regulated in {system}",
        "what are the key properties of {material} that make it useful for {application}",
        "explain the relationship between {concept x} and {concept y} in {discipline}",
        "how does variation in {parameter} affect {outcome}",
        "what is the role of {component} in the {larger system}",
        "describe the experimental procedure to determine {property}",
        "how is {concept} represented mathematically",
        "explain how {feedback mechanism} maintains {equilibrium}",
        "what are the ethical implications of {technology} in {society}",
        "how does {disease} affect the {organ system}",
        "explain the difference between {type one} and {type two} errors in statistics",
        "describe the structure of {compound} and its chemical properties",
        "how does {economic factor} influence {market behavior}",
    ]
    hard_templates = [
        "critically analyze the impact of {technology} on {domain}",
        "evaluate the effectiveness of {algorithm} optimization in {context}",
        "synthesize evidence to support the hypothesis regarding {topic}",
        "compare the computational complexity of {algorithm a} and {algorithm b}",
        "assess the theoretical limitations of {model} generalization",
        "interpret the statistical significance of {experimental finding}",
        "derive the mathematical relationship between {quantity a} and {quantity b}",
        "critically evaluate the assumptions underlying {theory}",
        "analyze the trade-offs between {approach a} and {approach b} in {scenario}",
        "propose a novel methodology to address {research problem}",
        "discuss the philosophical implications of {scientific discovery}",
        "evaluate how {policy} affects {socioeconomic outcome} across different populations",
        "formulate a mathematical proof for {theorem}",
        "critically assess the validity of {experiment} controlling for {confound}",
        "synthesize findings from {field a} and {field b} to explain {phenomenon}",
        "analyze the systemic effects of {intervention} on {complex system}",
        "evaluate competing theories of {controversial topic} based on empirical evidence",
        "derive an efficient algorithm for solving {computational problem}",
        "critically examine the role of {bias} in {research methodology}",
        "assess the long-term consequences of {decision} on {environment}",
        "analyze the epistemological foundations of {scientific discipline}",
        "evaluate the statistical power required to detect {effect} under {conditions}",
        "critically analyze the ethical framework of {controversial practice}",
        "synthesize a theoretical model to explain {observed anomaly}",
        "formulate and test a falsifiable hypothesis about {phenomenon}",
        "derive first principles justification for {engineering principle}",
        "assess the scalability and limitations of {distributed system}",
        "critically evaluate the reproducibility of {landmark study}",
        "analyze the emergent properties arising from {complex interaction}",
        "evaluate the geopolitical implications of {technological advancement}",
    ]
    fillers = {
        "concept": ["machine learning","neural networks","entropy","evolution","gravity","osmosis","democracy","capitalism","photosynthesis","relativity"],
        "person": ["Einstein","Darwin","Newton","Turing","Marie Curie","Tesla","Shakespeare","Napoleon","Confucius","Aristotle"],
        "event": ["World War II","the Renaissance","the Industrial Revolution","the French Revolution","the Big Bang","the Moon Landing","the Internet","DNA discovery","antibiotics","printing press"],
        "thing": ["the largest ocean","the smallest continent","the fastest animal","the tallest mountain","the deepest lake","the longest river","the most abundant gas","the lightest element","the hardest natural substance","the oldest civilization"],
        "term": ["algorithm","hypothesis","metabolism","democracy","entropy","catalyst","chromosome","ecosystem","bandwidth","recursion"],
        "abbr": ["DNA","CPU","RAM","URL","HTTP","AI","GDP","NASA","WHO","UNESCO"],
        "item": ["mitochondria","the nucleus","a virus","a quasar","a proton","tectonic plates","the ozone layer","an antibody","a transistor","a synapse"],
        "object": ["gold","copper","the sky","grass","blood","coal","lemon","snow","rust","chlorophyll"],
        "units": ["centimeters","millimeters","grams","seconds","volts","joules","moles","bytes","pixels","calories"],
        "container": ["a meter","a kilogram","a minute","a volt","a joule","a mole","a kilobyte","a megapixel","a kilocalorie","a kilometer"],
        "adjective": ["hot","large","positive","acidic","soluble","conductive","elastic","transparent","magnetic","radioactive"],
        "ordinal": ["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"],
        "group": ["the periodic table","the solar system","the food chain","the OSI model","the TCP/IP stack","the biological taxonomy","the color spectrum","the musical scale","the Fibonacci sequence","the binary system"],
        "place": ["France","Japan","Brazil","Canada","Australia","Egypt","India","Germany","Argentina","Nigeria"],
        "element": ["carbon","oxygen","hydrogen","nitrogen","gold","iron","helium","sodium","calcium","silicon"],
        "word": ["beautiful","necessary","occurrence","conscience","separate","rhythm","privilege","accommodate","millennium","superintendent"],
        "noun": ["mitochondria","nucleus","neuron","photon","molecule","chromosome","ecosystem","transistor","algorithm","hypothesis"],
        "famous work": ["Hamlet","The Origin of Species","Principia Mathematica","1984","The Republic","The Wealth of Nations","Crime and Punishment","The Iliad","Don Quixote","The Communist Manifesto"],
        "category": ["mammal","prime number","renewable energy","programming language","chemical element","continent","democracy","enzyme","algorithm","supernova"],
        "country": ["France","Germany","Brazil","Japan","Egypt","India","Australia","Canada","Mexico","Spain"],
        "process": ["photosynthesis","cellular respiration","protein synthesis","meiosis","osmosis","fermentation","electrolysis","nuclear fission","evaporation","condensation"],
        "context": ["plant cells","animal metabolism","bacterial cultures","neural networks","chemical reactions","semiconductor physics","quantum mechanics","economic systems","ecological food webs","atmospheric dynamics"],
        "biological process": ["mitosis","meiosis","DNA replication","gene expression","apoptosis","phagocytosis","synaptic transmission","hormone secretion","immune response","blood coagulation"],
        "organism": ["eukaryotes","prokaryotes","mammals","insects","fungi","viruses","plants","bacteria","archaea","protists"],
        "concept a": ["mitosis","classical mechanics","supervised learning","serial processing","renewable energy","inductive reasoning","socialism","deductive logic","natural selection","determinism"],
        "concept b": ["meiosis","quantum mechanics","unsupervised learning","parallel processing","fossil fuels","deductive reasoning","capitalism","inductive logic","genetic drift","probabilism"],
        "system": ["immune system","nervous system","digestive system","cardiovascular system","endocrine system","respiratory system","lymphatic system","skeletal system","muscular system","reproductive system"],
        "stimulus": ["bacterial infection","viral attack","physical injury","chemical toxin","temperature change","osmotic pressure","electromagnetic radiation","mechanical stress","hormonal signal","psychological stress"],
        "variable a": ["pressure","temperature","concentration","voltage","frequency","amplitude","mass","velocity","entropy","pH"],
        "variable b": ["volume","energy","reaction rate","current","wavelength","intensity","acceleration","momentum","disorder","solubility"],
        "component": ["mitochondria","ribosomes","the nucleus","enzymes","antibodies","neurons","transistors","capacitors","the CPU","the GPU"],
        "field": ["biology","physics","chemistry","computer science","economics","psychology","neuroscience","mathematics","sociology","ecology"],
        "technology": ["quantum computing","CRISPR","artificial intelligence","blockchain","5G networks","autonomous vehicles","gene therapy","nuclear fusion","nanotechnology","machine learning"],
        "method": ["gradient descent","backpropagation","Bayesian inference","Monte Carlo simulation","k-means clustering","support vector machines","random forests","principal component analysis","reinforcement learning","transfer learning"],
        "molecule": ["ATP","glucose","hemoglobin","insulin","dopamine","cortisol","serotonin","collagen","cholesterol","DNA"],
        "architecture": ["transformer models","convolutional neural networks","recurrent networks","attention mechanisms","encoder-decoder architecture","generative adversarial networks","BERT","ResNet","LSTM","diffusion models"],
        "real world context": ["healthcare","finance","education","agriculture","manufacturing","transportation","telecommunications","environmental monitoring","drug discovery","materials science"],
        "multi step process": ["PCR amplification","Western blotting","protein crystallization","software compilation","database normalization","signal processing","cryptographic key exchange","machine learning model training","vaccine development","bridge construction"],
        "phenomenon": ["global warming","antibiotic resistance","neuroplasticity","economic inflation","quantum entanglement","tectonic drift","superconductivity","epigenetic inheritance","emergent behavior","phase transitions"],
        "historical event": ["the discovery of penicillin","the development of the internet","the sequencing of the human genome","the invention of the transistor","the discovery of X-rays","nuclear fission","the Copernican revolution","Godel incompleteness theorems","the discovery of evolution","the invention of calculus"],
        "organisms": ["bacteria","plants","mammals","insects","fungi","fish","birds","reptiles","amphibians","archaea"],
        "environment": ["extreme cold","high salinity","low oxygen","high radiation","acidic conditions","desert heat","deep sea pressure","nutrient-poor soil","high altitude","anaerobic environments"],
        "substance": ["penicillin","aspirin","insulin","adrenaline","dopamine","cortisol","caffeine","ethanol","cyanide","carbon monoxide"],
        "biological structure": ["the cell membrane","the ribosome","the mitochondrion","the synapse","the nephron","the alveolus","the sarcomere","the chloroplast","the centrosome","the nuclear pore"],
        "algorithm": ["gradient descent","quicksort","Dijkstra algorithm","backpropagation","k-means","BERT training","merge sort","A-star search","Bellman-Ford","Viterbi algorithm"],
        "domain": ["image recognition","natural language processing","financial modeling","drug discovery","autonomous driving","climate modeling","cybersecurity","genomics","robotics","recommendation systems"],
        "approach": ["deep learning","rule-based systems","ensemble methods","transfer learning","federated learning","symbolic AI","evolutionary algorithms","Bayesian networks","reinforcement learning","attention mechanisms"],
        "abstract idea": ["entropy","complexity","emergence","recursion","symmetry","invariance","equilibrium","optimization","information","probability"],
        "outcome": ["accuracy","efficiency","stability","reproducibility","scalability","interpretability","fairness","robustness","generalizability","computational cost"],
        "type a": ["supervised","parametric","frequentist","deterministic","synchronous","sequential","static","discrete","linear","symbolic"],
        "type b": ["unsupervised","non-parametric","Bayesian","stochastic","asynchronous","parallel","dynamic","continuous","non-linear","connectionist"],
        "taxonomy": ["biological classification","chemical nomenclature","programming language paradigms","machine learning model types","economic systems","psychological disorders","graph types","sorting algorithm families","network topologies","database models"],
        "items": ["organisms","chemical compounds","programming paradigms","neural network types","economic indicators","cognitive biases","encryption algorithms","data structures","astronomical objects","geological formations"],
        "quantity": ["entropy","information content","statistical significance","computational complexity","thermodynamic temperature","quantum coherence","economic utility","ecological biodiversity","electromagnetic field strength","genetic variation"],
        "material": ["graphene","titanium","silicon carbide","carbon fiber","liquid crystal polymers","aerogel","shape memory alloys","piezoelectric ceramics","high-temperature superconductors","metamaterials"],
        "application": ["aerospace engineering","biomedical implants","semiconductor manufacturing","structural engineering","optical devices","energy storage","soft robotics","acoustic damping","quantum computing hardware","thermal management"],
        "parameter": ["learning rate","temperature","pH","concentration","voltage","sample size","regularization strength","network depth","mutation rate","selection pressure"],
        "larger system": ["the nervous system","the global economy","the atmosphere","the ecosystem","the internet","the power grid","the human microbiome","the financial system","the climate system","the cellular signaling network"],
        "property": ["melting point","electrical conductivity","tensile strength","solubility","refractive index","magnetic permeability","thermal capacity","optical absorption","enzymatic activity","quantum spin"],
        "feedback mechanism": ["negative feedback in homeostasis","positive feedback in childbirth","market price regulation","neural gain control","immune tolerance","genetic buffering","climate feedback loops","hormonal regulation","population density dependence","technological adoption curves"],
        "equilibrium": ["body temperature","blood glucose","market prices","neural firing rates","ecosystem species balance","chemical reaction rates","atmospheric CO2","gene expression levels","electromagnetic fields","economic supply and demand"],
        "disease": ["diabetes","Alzheimer disease","cancer","HIV","tuberculosis","Parkinson disease","schizophrenia","hypertension","obesity","autoimmune disorders"],
        "organ system": ["cardiovascular system","nervous system","immune system","endocrine system","digestive system","renal system","respiratory system","musculoskeletal system","lymphatic system","reproductive system"],
        "compound": ["benzene","glucose","hemoglobin","ATP","insulin","dopamine","penicillin","aspirin","caffeine","cholesterol"],
        "economic factor": ["interest rates","inflation","unemployment","government spending","trade deficits","currency exchange rates","technological innovation","tax policy","income inequality","globalization"],
        "market behavior": ["consumer demand","price elasticity","market equilibrium","investor sentiment","supply chain dynamics","monopoly formation","currency appreciation","wage stagnation","business cycle fluctuations","asset bubbles"],
        "model": ["neural network","linear regression","decision tree","support vector machine","Bayesian network","hidden Markov model","transformer","generative adversarial network","random forest","reinforcement learning agent"],
        "experimental finding": ["p-values","effect sizes","confidence intervals","correlation coefficients","regression coefficients","odds ratios","hazard ratios","Cohen d","standardized mean differences","Bayesian posterior distributions"],
        "quantity a": ["entropy","information","temperature","energy","voltage","pressure","frequency","mass","charge","angular momentum"],
        "quantity b": ["disorder","uncertainty","kinetic energy","work","current","volume","wavelength","acceleration","field strength","angular velocity"],
        "theory": ["general relativity","quantum mechanics","evolution by natural selection","plate tectonics","the standard model","the big bang","cognitive dissonance","efficient market hypothesis","social contract theory","information theory"],
        "approach a": ["deep learning","frequentist statistics","deterministic algorithms","centralized systems","rule-based AI","homogeneous computing","sequential processing","hard-coded features","single-objective optimization","supervised pre-training"],
        "approach b": ["symbolic reasoning","Bayesian inference","randomized algorithms","distributed systems","data-driven AI","heterogeneous computing","parallel processing","learned representations","multi-objective optimization","self-supervised learning"],
        "scenario": ["low-latency applications","high-dimensional data","safety-critical systems","resource-constrained devices","adversarial environments","real-time processing","long-horizon planning","privacy-sensitive contexts","interpretability requirements","scalability challenges"],
        "research problem": ["protein structure prediction","climate change mitigation","antibiotic resistance","autonomous navigation","interpretable AI","quantum error correction","multi-drug cancer therapy","real-time natural language understanding","global pandemic surveillance","efficient energy storage"],
        "scientific discovery": ["quantum indeterminacy","evolution","relativity","the uncertainty principle","DNA structure","the Higgs boson","dark matter","epigenetics","antibiotic resistance","neuroplasticity"],
        "policy": ["universal basic income","carbon taxation","net neutrality","patent law reform","drug decriminalization","universal healthcare","data privacy regulation","minimum wage increases","affirmative action","immigration reform"],
        "socioeconomic outcome": ["income inequality","employment rates","educational attainment","healthcare access","housing affordability","social mobility","poverty rates","gender wage gaps","intergenerational wealth","economic growth"],
        "theorem": ["Pythagoras theorem","Gödel incompleteness","Fermat last theorem","Bayes theorem","central limit theorem","Noether theorem","Nash equilibrium existence","Turing halting problem","Cantor diagonal argument","Rice theorem"],
        "experiment": ["randomized controlled trial","double-blind study","A/B testing","quasi-experimental design","natural experiment","longitudinal cohort study","cross-sectional survey","systematic review","meta-analysis","factorial design"],
        "confound": ["selection bias","confounding variables","measurement error","attrition bias","Hawthorne effect","Simpson paradox","multiple comparisons","reporting bias","survivorship bias","ecological fallacy"],
        "intervention": ["monetary policy","vaccination programs","algorithmic regulation","dietary supplementation","urban planning","educational curriculum reform","antitrust enforcement","environmental remediation","social media moderation","clinical drug trials"],
        "complex system": ["the global financial system","the human immune system","the internet","urban traffic networks","ecological food webs","the power grid","climate systems","the human brain","supply chains","democratic governance"],
        "controversial topic": ["consciousness","free will","the interpretation of quantum mechanics","the nature of dark matter","the origins of language","the hard problem of consciousness","the existence of objective morality","the multiverse hypothesis","AI sentience","the foundations of mathematics"],
        "computational problem": ["the traveling salesman problem","graph isomorphism","protein folding","SAT solving","integer factorization","graph coloring","scheduling optimization","natural language understanding","image segmentation","optimal control"],
        "bias": ["confirmation bias","survivorship bias","publication bias","measurement bias","selection bias","anchoring bias","availability heuristic","attribution bias","in-group bias","automation bias"],
        "research methodology": ["randomized controlled trials","observational studies","systematic reviews","meta-analyses","qualitative research","survey design","case studies","computational modeling","laboratory experiments","field studies"],
        "environment_hard": ["climate change","urbanization","deforestation","ocean acidification","industrial pollution","soil degradation","species extinction","freshwater scarcity","electromagnetic pollution","plastic accumulation"],
        "decision": ["fossil fuel investment","deforestation","nuclear energy expansion","antibiotic overuse","social media algorithmic curation","mass surveillance","genetically modified organisms","geoengineering","space militarization","autonomous weapons development"],
        "scientific discipline": ["quantum mechanics","evolutionary biology","cognitive neuroscience","econometrics","information theory","synthetic biology","astroastrophysics","computational linguistics","social psychology","materials science"],
        "effect": ["small effect size Cohen d 0.2","medium effect size Cohen d 0.5","large effect size Cohen d 0.8","statistically significant interaction","non-linear dose response","threshold effect","synergistic effect","antagonistic interaction","ceiling effect","floor effect"],
        "conditions": ["small sample sizes","high measurement noise","multiple covariates","non-normal distributions","correlated predictors","missing data","selection bias","temporal autocorrelation","clustered observations","heteroscedastic errors"],
        "controversial practice": ["human genetic enhancement","autonomous lethal weapons","mass algorithmic surveillance","predictive policing","social credit scoring","non-consensual behavioral nudging","corporate tax avoidance","factory farming","deep sea mining","fossil fuel lobbying"],
        "observed anomaly": ["dark matter gravitational lensing","faster than expected universe expansion","antibiotic resistance spread","unexpected protein folding patterns","anomalous neural activation patterns","unexplained market crashes","species range shifts","emergence of drug-resistant pathogens","unexpected climate tipping points","anomalous quantum decoherence rates"],
        "engineering principle": ["the second law of thermodynamics","Shannon information capacity","Nyquist sampling theorem","Heisenberg uncertainty principle","the Carnot efficiency limit","Maxwell equations","Euler beam theory","Fourier transform duality","Ohm law","Faraday electromagnetic induction"],
        "distributed system": ["blockchain consensus","MapReduce","federated learning","peer-to-peer networks","microservices architecture","distributed databases","edge computing","content delivery networks","distributed machine learning","consensus algorithms"],
        "landmark study": ["Watson and Crick DNA structure","Milgram obedience experiments","Framingham Heart Study","the Tuskegee syphilis study","the Stanford Prison Experiment","the Asch conformity experiments","the Hawthorne studies","the Nurses Health Study","the Human Genome Project","the LHC Higgs boson discovery"],
        "complex interaction": ["protein-protein interactions","ecological mutualism","neuronal network dynamics","market feedback loops","immune system cross-reactivity","gene regulatory networks","agent-based social dynamics","quantum entanglement","non-linear climate feedbacks","multi-drug pharmacological interactions"],
        "technological advancement": ["quantum computing","advanced AI","synthetic biology","autonomous weapons","space colonization","brain-computer interfaces","advanced nuclear reactors","global internet access","precision medicine","carbon capture technology"],
        "feedback mechanism_hard": ["climate positive feedbacks","antibiotic resistance amplification","algorithmic market instability","neurological addiction cycles","pandemic transmission dynamics","economic inequality compounding","misinformation viral spread","nuclear arms race escalation","ecosystem collapse cascades","financial system fragility"],
    }
    def fill_template(template: str, fillers: Dict) -> str:
        result = template
        for key, values in fillers.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, random.choice(values), 1)
        result = re.sub(r'\{[^}]+\}', 'concept', result)
        return result
    train_x = []
    train_y = []
    random.seed(42)
    easy_count = 1000
    medium_count = 1200
    hard_count = 800
    for _ in range(easy_count):
        tmpl = random.choice(easy_templates)
        train_x.append(fill_template(tmpl, fillers))
        train_y.append("easy")
    for _ in range(medium_count):
        tmpl = random.choice(medium_templates)
        train_x.append(fill_template(tmpl, fillers))
        train_y.append("medium")
    for _ in range(hard_count):
        tmpl = random.choice(hard_templates)
        train_x.append(fill_template(tmpl, fillers))
        train_y.append("hard")
    combined = list(zip(train_x, train_y))
    random.shuffle(combined)
    train_x, train_y = zip(*combined)
    return list(train_x), list(train_y)


class DifficultyClassifier:
    def __init__(self):
        self.le = LabelEncoder()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)),
            ('nb', MultinomialNB(alpha=0.3)),
        ])
        logger.info("⏳ Building 3000-sample Naive Bayes training data...")
        train_x, train_y = build_training_data()
        y_enc = self.le.fit_transform(train_y)
        self.pipeline.fit(train_x, y_enc)
        logger.info(f"✓ Naive Bayes trained on {len(train_x)} samples | Classes: {list(self.le.classes_)}")

    def predict(self, text: str) -> str:
        try:
            y_enc = self.pipeline.predict([text])
            return str(self.le.inverse_transform(y_enc)[0])
        except Exception:
            return "medium"


class NumpyQualityScorer:
    W = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)

    def __init__(self, word_freq: Dict[str, int]):
        total = max(sum(word_freq.values()), 1)
        self.prob = {w: c / total for w, c in word_freq.items()}

    def score(self, question: str, answer: str, distractors: List[str], tfidf_score: float) -> float:
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


class AIQuizEngine:
    def __init__(self):
        self._tfidf = TFIDFModule(max_features=800)
        self._ranker = CosineSimilarityRanker()
        self._clusterer = KMeansTopicClusterer(n_clusters=6)
        self._classifier = DifficultyClassifier()
        self._scorer: Optional[NumpyQualityScorer] = None
        logger.info("🚀 AIQuizEngine v6.0 initialized with PDF + 3000-MCQ training")

    def _get_word_freq(self, text: str) -> Dict[str, int]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return Counter(words)

    def _get_all_nouns(self, text: str) -> List[str]:
        tagged = get_pos_tags(text)
        nouns = [
            word for word, tag in tagged
            if tag.startswith('NN') and word.lower() not in STOP_WORDS
            and len(word) > 3 and word.isalpha()
        ]
        return nouns

    def _make_fill_blank_mcq(self, sentence: str, context_words: List[str],
                              tfidf_score: float, cluster: int) -> Optional[Dict]:
        result = get_key_noun(sentence)
        if not result:
            return None
        target_word, pos = result
        if len(target_word) <= 3 or target_word.lower() in STOP_WORDS:
            return None
        pattern = r'\b' + re.escape(target_word) + r'\b'
        q_text = re.sub(pattern, '______', sentence, count=1, flags=re.IGNORECASE)
        q_text = q_text.strip()
        if '______' not in q_text:
            return None
        if not q_text.endswith(('?', '.', '!')):
            q_text += '.'
        distractors = build_distractors(target_word, pos, context_words)
        all_options = distractors[:3] + [target_word]
        random.shuffle(all_options)
        correct_idx = all_options.index(target_word)
        difficulty = self._classifier.predict(q_text)
        quality = self._scorer.score(q_text, target_word, distractors, tfidf_score) if self._scorer else 0.65
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
        text = clean_text(raw_text)
        word_freq = self._get_word_freq(text)
        self._scorer = NumpyQualityScorer(word_freq)
        context_words = self._get_all_nouns(text)
        results: List[Dict] = []
        used_answers: set = set()
        used_sentences: set = set()
        t0 = time.time()
        sentences = split_sentences(text)
        if len(sentences) < 2:
            sentences = [s.strip() for s in text.split('.') if len(s.split()) >= 6]
        if not sentences:
            sentences = [text]
        logger.info(f"✓ Found {len(sentences)} sentences")
        tfidf_matrix = self._tfidf.fit_transform(sentences)
        keywords = self._tfidf.get_top_keywords(8)
        logger.info(f"✓ TF-IDF {tfidf_matrix.shape} | Keywords: {keywords}")
        importance = self._ranker.rank(tfidf_matrix)
        clusters = self._clusterer.fit_predict(tfidf_matrix)
        ranked_idx = np.argsort(importance)[::-1]
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
        if len(results) < limit:
            for idx in ranked_idx:
                if len(results) >= limit:
                    break
                sent = sentences[idx]
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
        if results:
            q_scores = np.array([r['quality_score'] for r in results])
            sorted_idx = np.argsort(q_scores)[::-1]
            results = [results[i] for i in sorted_idx]
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


ai_engine = AIQuizEngine()


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


app = FastAPI(
    title="UoH AI Quiz Engine v6.0",
    description="NLP + TF-IDF + Cosine + KMeans + NaiveBayes(3000samples) + NumPy + Pandas + PDF",
    version="6.0"
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
    session = QuizSession(
        user_id=req.user_id,
        title=f"Quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        content_summary=req.text_content[:300],
        total_questions=len(questions),
        quiz_type=quiz_type,
        processing_time=elapsed,
        source_type="text",
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
        "source_type": "text",
        "ml_pipeline": "NLTK → TF-IDF → Cosine → KMeans → NaiveBayes(3000) → NumPy → Pandas",
        "quiz": questions,
        "stats": stats,
    }


@app.post("/api/v1/generate-quiz-pdf")
async def generate_quiz_from_pdf(
    user_id: int = Form(...),
    count: int = Form(10),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not PDF_SUPPORT:
        raise HTTPException(500, "PDF support not installed. Run: pip install pymupdf")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are accepted")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if not (1 <= count <= 100):
        raise HTTPException(400, "Count must be between 1 and 100")
    pdf_bytes = await file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "PDF file too large. Maximum size is 20MB")
    logger.info(f"📄 Extracting text from PDF: {file.filename} ({len(pdf_bytes)} bytes)")
    extracted_text = extract_text_from_pdf(pdf_bytes)
    if len(extracted_text.strip()) < 50:
        raise HTTPException(
            422,
            "Could not extract enough text from PDF. "
            "Make sure the PDF contains readable text (not scanned images)."
        )
    logger.info(f"✓ Extracted {len(extracted_text)} characters from PDF")
    t0 = time.time()
    questions = ai_engine.generate(extracted_text, count)
    elapsed = round(time.time() - t0, 3)
    if not questions:
        raise HTTPException(
            422,
            "Could not generate questions from PDF content. "
            "Ensure the PDF has academic text with definitions and facts."
        )
    quiz_type_map = {10: "quick", 25: "standard", 50: "extended", 100: "full"}
    quiz_type = quiz_type_map.get(count, "standard" if count <= 25 else "extended")
    session = QuizSession(
        user_id=user_id,
        title=f"PDF_{file.filename[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        content_summary=extracted_text[:300],
        total_questions=len(questions),
        quiz_type=quiz_type,
        processing_time=elapsed,
        source_type="pdf",
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
        "pdf_chars_extracted": len(extracted_text),
        "pdf_filename": file.filename,
    }
    return {
        "session_id": session.id,
        "time": f"{elapsed}s",
        "total": len(questions),
        "quiz_type": quiz_type,
        "source_type": "pdf",
        "pdf_filename": file.filename,
        "pdf_text_length": len(extracted_text),
        "ml_pipeline": "PyMuPDF → NLTK → TF-IDF → Cosine → KMeans → NaiveBayes(3000) → NumPy → Pandas",
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
                "source_type": s.source_type or "text",
                "processing_time": f"{s.processing_time}s",
                "created_at": s.created_at.strftime("%d %b %Y %H:%M"),
            }
            for s in sessions
        ],
    }


@app.get("/api/v1/session/{session_id}")
def get_session_questions(session_id: int, db: Session = Depends(get_db)):
    session = db.query(QuizSession).filter(QuizSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")
    questions = db.query(QuestionBank).filter(QuestionBank.session_id == session_id).all()
    return {
        "session_id": session_id,
        "title": session.title,
        "source_type": session.source_type or "text",
        "total": len(questions),
        "questions": [
            {
                "id": q.id,
                "question": q.question_body,
                "correct": q.correct_ans,
                "options": q.distractors_json.split('|') if q.distractors_json else [],
                "difficulty": q.difficulty,
                "topic_cluster": q.topic_cluster,
                "quality_score": q.quality_score,
                "question_type": q.question_type,
            }
            for q in questions
        ],
    }


@app.get("/api/v1/ml-info")
def ml_info():
    return {
        "engine": "UoH AI Quiz Engine v6.0",
        "new_features": ["PDF text extraction (PyMuPDF)", "3000-sample Naive Bayes training", "Session question retrieval"],
        "pipeline_steps": [
            {"step": 1, "name": "PDF Extractor (PyMuPDF)", "purpose": "Extract text from PDF files"},
            {"step": 2, "name": "NLTK Sentence Tokenizer", "purpose": "Split text into sentences"},
            {"step": 3, "name": "POS Tagger", "purpose": "Identify nouns, verbs, adjectives"},
            {"step": 4, "name": "TF-IDF Vectorizer", "purpose": "Text → numerical features"},
            {"step": 5, "name": "Cosine Similarity", "purpose": "Rank sentence importance"},
            {"step": 6, "name": "KMeans Clustering", "purpose": "Group sentences by topic"},
            {"step": 7, "name": "Wh-Question Generator", "purpose": "Definition-based MCQs"},
            {"step": 8, "name": "Fill-Blank Generator", "purpose": "POS-targeted cloze MCQs"},
            {"step": 9, "name": "WordNet Distractors", "purpose": "Semantic wrong options"},
            {"step": 10, "name": "Naive Bayes (3000 samples)", "purpose": "Difficulty classification"},
            {"step": 11, "name": "NumPy Quality Scorer", "purpose": "5-feature dot-product scoring"},
            {"step": 12, "name": "Pandas Analytics", "purpose": "Statistics & reporting"},
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "6.0",
        "database": "sqlite",
        "ml_pipeline": "active",
        "pdf_support": PDF_SUPPORT,
        "nb_training_samples": 3000,
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("🚀 UoH AI Quiz Engine v6.0")
    logger.info("📍 http://127.0.0.1:8000")
    logger.info("📚 Docs: http://127.0.0.1:8000/docs")
    logger.info("📄 PDF Support: " + ("✓ Active" if PDF_SUPPORT else "✗ Install pymupdf"))
    logger.info("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)