"""
app.py
────────────────────────────────────────────────────────────────────
UoH AI Quiz Engine — Deep Learning Edition
FastAPI backend. Auth, quiz generation (text or PDF), history,
and ML pipeline introspection endpoints.

Run:
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
    python app.py

First run will download the pretrained sentence-transformer and T5
question-generation models from HuggingFace (needs internet once;
cached locally afterwards).

To enable the fine-tuned difficulty classifier instead of the
heuristic fallback, see train_difficulty.py.
────────────────────────────────────────────────────────────────────
"""

import re
import time
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd

from ml_engine import get_engine
from difficulty_model import load_training_meta

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("UOH_QUIZ_DL")

# ════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════
DATABASE_URL = "sqlite:///./uoh_quiz_dl.db"
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
    ml_pipeline = Column(String(200), default="SentenceTransformer|T5-QG|KMeans|PyTorchNN|NumPy")
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
    question_type = Column(String(30), default="neural_qg")
    session_parent = relationship("QuizSession", back_populates="questions")


Base.metadata.create_all(bind=engine_db)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ════════════════════════════════════════════════════════════════
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if not PDF_SUPPORT:
        raise HTTPException(500, "PDF support not available. Install pymupdf: pip install pymupdf")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            t = page.get_text("text")
            t = re.sub(r"\n{3,}", "\n\n", t)
            t = re.sub(r"[ \t]+", " ", t)
            chunks.append(t.strip())
        doc.close()
        combined = "\n\n".join(chunks)
        combined = re.sub(r"\n([a-z])", r" \1", combined)
        combined = re.sub(r"-\n(\w)", r"\1", combined)
        combined = re.sub(r"\n{2,}", ". ", combined)
        combined = re.sub(r"\s+", " ", combined)
        return combined.strip()
    except Exception as e:
        raise HTTPException(500, f"PDF extraction failed: {str(e)}")


# ════════════════════════════════════════════════════════════════
# ML ENGINE — loaded once at startup (downloads models on first run)
# ════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("🚀 Loading deep learning quiz engine (this may take a while on first run)…")
quiz_engine = get_engine()
logger.info("=" * 70)


# ════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════
class SignupSchema(BaseModel):
    username: str
    email: str
    password: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        v = v.strip()
        if len(v) < 3 or len(v) > 30:
            raise ValueError("Username: 3-30 characters")
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Only letters, numbers, underscores")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        v = v.strip().lower()
        if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", v):
            raise ValueError("Invalid email format")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        errors = []
        if len(v) < 8: errors.append("8+ chars")
        if not re.search(r"[A-Z]", v): errors.append("uppercase")
        if not re.search(r"[a-z]", v): errors.append("lowercase")
        if not re.search(r"\d", v): errors.append("digit")
        if not re.search(r"[!@#$%^&*(),.?]", v): errors.append("special char")
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

    @field_validator("count")
    @classmethod
    def validate_count(cls, v):
        if not (1 <= v <= 100):
            raise ValueError("Count must be 1-100")
        return v

    @field_validator("text_content")
    @classmethod
    def validate_text(cls, v):
        v = v.strip()
        if len(v) < 50:
            raise ValueError("Minimum 50 characters of academic text required")
        return v


# ════════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════════
app = FastAPI(
    title="UoH AI Quiz Engine — Deep Learning Edition",
    description="SentenceTransformers + T5 question generation + PyTorch difficulty net + KMeans + NumPy",
    version="7.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── AUTH ──
@app.post("/auth/signup")
def signup(data: SignupSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(400, "Username already taken")
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "Email already registered")
    user = User(
        username=data.username,
        email=data.email,
        password=generate_password_hash(data.password),
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


# ── HELPERS ──
def _quiz_type_for(count: int) -> str:
    return {10: "quick", 25: "standard", 50: "extended", 100: "full"}.get(
        count, "standard" if count <= 25 else "extended"
    )


def _persist_session_and_questions(db: Session, user_id: int, title: str, content_summary: str,
                                    questions: list, elapsed: float, source_type: str) -> QuizSession:
    quiz_type = _quiz_type_for(len(questions))
    session = QuizSession(
        user_id=user_id,
        title=title,
        content_summary=content_summary[:300],
        total_questions=len(questions),
        quiz_type=quiz_type,
        processing_time=elapsed,
        source_type=source_type,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    for q in questions:
        db.add(QuestionBank(
            session_id=session.id,
            question_body=q["question"],
            correct_ans=q["correct"],
            distractors_json="|".join(q["options"]),
            difficulty=q["difficulty"],
            topic_cluster=q["topic_cluster"],
            quality_score=q["quality_score"],
            question_type=q.get("question_type", "neural_qg"),
        ))
    db.commit()
    return session


def _build_stats(questions: list) -> dict:
    df = pd.DataFrame(questions)
    return {
        "easy": int((df["difficulty"] == "easy").sum()),
        "medium": int((df["difficulty"] == "medium").sum()),
        "hard": int((df["difficulty"] == "hard").sum()),
        "avg_quality": round(float(df["quality_score"].mean()), 3),
        "clusters": int(df["topic_cluster"].nunique()),
        "neural_qg": int((df["question_type"] == "neural_qg").sum()),
        "fill_blank": int((df["question_type"] == "fill_blank").sum()),
    }


# ── QUIZ GENERATION (TEXT) ──
@app.post("/api/v1/generate-quiz")
async def generate_quiz(req: QuizRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        raise HTTPException(404, "User not found")

    t0 = time.time()
    try:
        questions = quiz_engine.generate(req.text_content, req.count)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    elapsed = round(time.time() - t0, 3)

    if not questions:
        raise HTTPException(
            422,
            "Could not generate questions. Please provide more detailed academic text "
            "(at least 3-5 sentences with clear concepts, definitions, or facts).",
        )

    session = _persist_session_and_questions(
        db, req.user_id, f"Quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        req.text_content, questions, elapsed, "text",
    )
    stats = _build_stats(questions)

    return {
        "session_id": session.id,
        "time": f"{elapsed}s",
        "total": len(questions),
        "quiz_type": session.quiz_type,
        "source_type": "text",
        "ml_pipeline": "SentenceTransformer → CosineCentrality → KMeans → T5-QG → WordNet+Semantic Distractors → PyTorch DifficultyNet → NumPy Quality Scorer",
        "quiz": questions,
        "stats": stats,
    }


# ── QUIZ GENERATION (PDF) ──
@app.post("/api/v1/generate-quiz-pdf")
async def generate_quiz_from_pdf(
    user_id: int = Form(...),
    count: int = Form(10),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not PDF_SUPPORT:
        raise HTTPException(500, "PDF support not installed. Run: pip install pymupdf")
    if not file.filename.lower().endswith(".pdf"):
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
            "Make sure the PDF contains readable text (not scanned images).",
        )
    logger.info(f"✓ Extracted {len(extracted_text)} characters from PDF")

    t0 = time.time()
    try:
        questions = quiz_engine.generate(extracted_text, count)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    elapsed = round(time.time() - t0, 3)

    if not questions:
        raise HTTPException(
            422,
            "Could not generate questions from PDF content. "
            "Ensure the PDF has academic text with definitions and facts.",
        )

    session = _persist_session_and_questions(
        db, user_id, f"PDF_{file.filename[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        extracted_text, questions, elapsed, "pdf",
    )
    stats = _build_stats(questions)
    stats["pdf_chars_extracted"] = len(extracted_text)
    stats["pdf_filename"] = file.filename

    return {
        "session_id": session.id,
        "time": f"{elapsed}s",
        "total": len(questions),
        "quiz_type": session.quiz_type,
        "source_type": "pdf",
        "pdf_filename": file.filename,
        "pdf_text_length": len(extracted_text),
        "ml_pipeline": "PyMuPDF → SentenceTransformer → CosineCentrality → KMeans → T5-QG → Distractor Mining → PyTorch DifficultyNet → NumPy Quality Scorer",
        "quiz": questions,
        "stats": stats,
    }


# ── HISTORY ──
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
                "total_questions": s.total_questions,
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
                "options": q.distractors_json.split("|") if q.distractors_json else [],
                "difficulty": q.difficulty,
                "topic_cluster": q.topic_cluster,
                "quality_score": q.quality_score,
                "question_type": q.question_type,
            }
            for q in questions
        ],
    }


# ── ML INTROSPECTION ──
@app.get("/api/v1/ml-info")
def ml_info():
    train_meta = load_training_meta()
    return {
        "engine": "UoH AI Quiz Engine — Deep Learning Edition v7.0",
        "question_generation_mode": quiz_engine.qgen.mode,
        "embedding_model": "all-MiniLM-L6-v2 (sentence-transformers, 384-dim)" if quiz_engine.embedder.available() else "unavailable",
        "difficulty_model_status": "fine-tuned (loaded from difficulty_weights.pt)" if quiz_engine.classifier.is_trained() else "heuristic fallback — run train_difficulty.py to enable the neural classifier",
        "difficulty_training_run": train_meta,
        "pipeline_steps": [
            {"step": 1, "name": "PyMuPDF Extractor", "purpose": "Extract text from uploaded PDF files"},
            {"step": 2, "name": "NLTK Sentence Tokenizer", "purpose": "Split text into candidate sentences"},
            {"step": 3, "name": "Sentence-Transformer (MiniLM)", "purpose": "Dense semantic embeddings per sentence"},
            {"step": 4, "name": "Cosine Centrality Ranking", "purpose": "Rank sentences by semantic importance"},
            {"step": 5, "name": "KMeans on Embeddings", "purpose": "Group sentences into topic clusters"},
            {"step": 6, "name": "POS Tagging (NLTK)", "purpose": "Identify candidate answer spans (nouns)"},
            {"step": 7, "name": "T5 Question-Generation Transformer", "purpose": "Neural generation of fluent questions"},
            {"step": 8, "name": "Semantic + WordNet Distractor Mining", "purpose": "Plausible wrong answers"},
            {"step": 9, "name": "PyTorch DifficultyNet", "purpose": "Fine-tunable neural difficulty classifier"},
            {"step": 10, "name": "NumPy Quality Scorer", "purpose": "5-signal weighted dot-product scoring"},
            {"step": 11, "name": "Pandas Analytics", "purpose": "Per-quiz statistics & reporting"},
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "7.0",
        "database": "sqlite",
        "pdf_support": PDF_SUPPORT,
        "embedding_model_ready": quiz_engine.embedder.available(),
        "question_gen_mode": quiz_engine.qgen.mode,
        "difficulty_model_trained": quiz_engine.classifier.is_trained(),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("🚀 UoH AI Quiz Engine — Deep Learning Edition")
    logger.info("📍 http://127.0.0.1:8000")
    logger.info("📚 Docs: http://127.0.0.1:8000/docs")
    logger.info("📄 PDF Support: " + ("✓ Active" if PDF_SUPPORT else "✗ Install pymupdf"))
    logger.info("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
