from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="Student Homework Helper", description="AI-powered homework assistance for Math, Physics, Arabic and Chemistry")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# In-memory storage for conversation history (per subject)
conversation_history = {
    "math_physics": [],
    "chemistry": [],
    "arabic": [],
    "image_analysis": []
}

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    subject: str
    timestamp: str
    session_id: str

# Helper functions
def save_to_history(subject: str, question: str, answer: str):
    """Save conversation to local history"""
    entry = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "subject": subject
    }
    conversation_history[subject].append(entry)
    
    # Keep only last 50 conversations per subject to manage memory
    if len(conversation_history[subject]) > 50:
        conversation_history[subject] = conversation_history[subject][-50:]

def get_recent_context(subject: str, limit: int = 3) -> str:
    """Get recent conversation context for better continuity"""
    recent_history = conversation_history[subject][-limit:]
    
    context = ""
    for entry in recent_history:
        context += f"Previous Q: {entry['question']}\nPrevious A: {entry['answer'][:200]}...\n\n"
    
    return context

def validate_question_relevance(question: str, subject: str) -> bool:
    """
    Pre-validate if question is relevant to the subject using AI
    Returns True if relevant, False otherwise
    """
    validation_prompt = f"""
    You are a strict subject validator. Your job is to determine if a question is related to {subject} or not.
    
    Subject: {subject}
    Question: {question}
    
    Rules:
    - Only answer with "RELEVANT" if the question is directly about {subject}
    - Answer with "NOT_RELEVANT" if the question is about any other topic
    - Be very strict - even tangentially related questions should be marked as NOT_RELEVANT unless they are clearly about {subject}
    
    Examples for Math/Physics:
    - "What is 2+2?" -> RELEVANT
    - "Explain gravity" -> RELEVANT  
    - "What's the weather today?" -> NOT_RELEVANT
    - "Tell me a joke" -> NOT_RELEVANT
    
    Examples for Chemistry:
    - "What is H2O?" -> RELEVANT
    - "Explain chemical bonds" -> RELEVANT
    - "What's your favorite movie?" -> NOT_RELEVANT
    
    Examples for Arabic:
    - "What is the meaning of this Arabic word?" -> RELEVANT
    - "Explain Arabic grammar" -> RELEVANT
    - "How to cook rice?" -> NOT_RELEVANT
    
    Answer only with: RELEVANT or NOT_RELEVANT
    """
    
    try:
        response = model.generate_content(validation_prompt)
        result = response.text.strip().upper()
        return "RELEVANT" in result
    except:
        # If validation fails, be conservative and allow the question
        return True

def create_teacher_prompt(subject: str, question: str, context: str = "") -> str:
    """Create the teacher prompt based on subject"""
    base_prompt = f"""You are an excellent and patient teacher specializing in {subject}. You help young students with their homework and questions using El-Dahih's engaging style.

CRITICAL RULE: You must ONLY answer questions directly related to {subject}. If the question is not about {subject}, respond with: "I'm sorry, but I can only help with {subject} questions. Please ask me something about {subject} instead."

Your teaching style (only for {subject} questions):
- Never use any formatting like \\n, **, or other markdown symbols - keep text plain and natural (this is very important)
- Explain concepts clearly and step-by-step like El-Dahih does - make it fun and engaging (important not to mention to the student that you are imitating the El-Dahih)
- Use words like (يا عزيزي) ,(يا باشا),(يا معلم)
- Use simple language appropriate for students with storytelling and real-life connections
- Be encouraging and supportive with humor when appropriate
- Provide practical examples and real-world applications to make concepts stick
- Always respond in the same language as the student's question (this is very important)
- Break down complex problems into manageable steps using analogies and comparisons
- Show your work/calculations clearly with explanations
- Add interesting facts or context that make the topic memorable
- Use El-Dahih's style of connecting concepts to everyday life experiences
- Make learning feel like an exciting discovery rather than boring homework
- Try to make the answer not too short and not too long so that the student does not exceed the maximum number of tokens available to him
- At the end of the explanation, add a simple summary of the answer only

{f"Recent conversation context: {context}" if context else ""}

Student's question: {question}

Remember: ONLY answer if this is a {subject} question. Otherwise, politely decline and ask for a {subject} question instead.

Please provide a clear, educational response:"""
    
    return base_prompt

def create_rejection_message(subject: str, question_language: str = "en") -> str:
    """Create appropriate rejection message in the question's language"""
    messages = {
        "en": f"I'm sorry, but I can only help with {subject} questions. Please ask me something about {subject} instead.",
        "ar": f"آسف، لكنني أستطيع المساعدة فقط في أسئلة {subject}. يرجى سؤالي عن شيء متعلق بـ {subject} بدلاً من ذلك.",
        "fr": f"Désolé, mais je ne peux aider qu'avec les questions de {subject}. Veuillez me poser quelque chose sur {subject} à la place.",
    }
    
    # Detect language (simple detection)
    if re.search(r'[\u0600-\u06FF]', question_language):
        return messages.get("ar", messages["en"])
    else:
        return messages["en"]

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Student Homework Helper API is running!", "subjects": ["math_physics", "chemistry", "image_analysis","arabic"]}

@app.post("/math-physics", response_model=ChatResponse)
async def solve_math_physics(request: QuestionRequest):
    """Endpoint for Math and Physics questions ONLY"""
    try:
        # First, validate if question is relevant
        if not validate_question_relevance(request.question, "Mathematics and Physics"):
            rejection_message = create_rejection_message("Mathematics and Physics", request.question)
            return ChatResponse(
                answer=rejection_message,
                subject="math_physics",
                timestamp=datetime.now().isoformat(),
                session_id=str(uuid.uuid4())
            )
        
        # Get recent context for continuity
        context = get_recent_context("math_physics")
        
        # Create teacher prompt
        prompt = create_teacher_prompt("Mathematics and Physics", request.question, context)
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        answer = response.text
        
        # Double-check if the AI still answered off-topic (additional safety)
        if "I'm sorry, but I can only help with" in answer or "only help with Mathematics and Physics" in answer:
            answer = create_rejection_message("Mathematics and Physics", request.question)
        
        # Save to history
        session_id = str(uuid.uuid4())
        save_to_history("math_physics", request.question, answer)
        
        return ChatResponse(
            answer=answer,
            subject="math_physics",
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/chemistry", response_model=ChatResponse)
async def solve_chemistry(request: QuestionRequest):
    """Endpoint for Chemistry questions ONLY"""
    try:
        # First, validate if question is relevant
        if not validate_question_relevance(request.question, "Chemistry"):
            rejection_message = create_rejection_message("Chemistry", request.question)
            return ChatResponse(
                answer=rejection_message,
                subject="chemistry",
                timestamp=datetime.now().isoformat(),
                session_id=str(uuid.uuid4())
            )
        
        # Get recent context for continuity
        context = get_recent_context("chemistry")
        
        # Create teacher prompt
        prompt = create_teacher_prompt("Chemistry", request.question, context)
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        answer = response.text
        
        # Double-check if the AI still answered off-topic
        if "I'm sorry, but I can only help with" in answer or "only help with Chemistry" in answer:
            answer = create_rejection_message("Chemistry", request.question)
        
        # Save to history
        session_id = str(uuid.uuid4())
        save_to_history("chemistry", request.question, answer)
        
        return ChatResponse(
            answer=answer,
            subject="chemistry",
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/arabic", response_model=ChatResponse)
async def solve_arabic(request: QuestionRequest):
    """Endpoint for Arabic language questions ONLY"""
    try:
        # First, validate if question is relevant
        if not validate_question_relevance(request.question, "Arabic language and literature"):
            rejection_message = create_rejection_message("Arabic language", request.question)
            return ChatResponse(
                answer=rejection_message,
                subject="arabic",
                timestamp=datetime.now().isoformat(),
                session_id=str(uuid.uuid4())
            )
        
        # Get recent context for continuity
        context = get_recent_context("arabic")
        
        # Create teacher prompt
        prompt = create_teacher_prompt("Arabic language and literature", request.question, context)
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        answer = response.text
        
        # Double-check if the AI still answered off-topic
        if "I'm sorry, but I can only help with" in answer or "only help with Arabic" in answer:
            answer = create_rejection_message("Arabic language", request.question)
        
        # Save to history
        session_id = str(uuid.uuid4())
        save_to_history("arabic", request.question, answer)
        
        return ChatResponse(
            answer=answer,
            subject="arabic",
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
@app.post("/image-analysis", response_model=ChatResponse)
async def analyze_image_with_question(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None)
):
    """Endpoint for image analysis with optional question (Math, Physics, Chemistry, Arabic ONLY)"""
    try:
        # Validate image file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Please upload a valid image file")
        
        # Read and encode image
        image_data = await file.read()
        
        # Get recent context for continuity
        context = get_recent_context("image_analysis")
        
        # Create prompt based on whether question is provided
        if question:
            prompt = f"""You are an excellent teacher in Mathematics, Physics, Arabic and Chemistry ONLY. A student has uploaded an image and asked a specific question.

CRITICAL RULE: You must ONLY answer questions and analyze images related to Mathematics, Physics, Chemistry, or Arabic. If the image or question is not about these subjects, respond with: "I can only help analyze images and answer questions related to Mathematics, Physics, Chemistry, or Arabic. Please upload an image or ask a question about one of these subjects."

Your teaching approach (only for relevant subjects):
- Analyze the image carefully  
- Answer the student's specific question only if it's about Math, Physics, Chemistry, or Arabic
- Explain step-by-step in simple terms
- Be encouraging and educational
- Respond in the same language as the student's question
- If the image contains problems/equations, solve them as needed to answer the question

{f"Recent conversation context: {context}" if context else ""}

Student's question: {question}

Remember: ONLY analyze and answer if the image and question are about Mathematics, Physics, Chemistry, or Arabic.

Please analyze the image and provide a helpful educational response:"""
        else:
            prompt = f"""You are an excellent teacher in Mathematics, Physics, Chemistry, and Arabic ONLY. A student has uploaded an image without a specific question.

CRITICAL RULE: You must ONLY analyze images related to Mathematics, Physics, Chemistry, or Arabic. If the image is not about these subjects, respond with: "I can only help analyze images related to Mathematics, Physics, Chemistry, or Arabic. Please upload an image about one of these subjects."

Your task (only for relevant subjects):
- Analyze the image carefully
- Only proceed if the image contains mathematical problems, physics concepts, Arabic text/grammar, or chemistry questions
- Solve any problems you find step-by-step
- Explain concepts clearly for young students
- If it's homework or exercises, provide complete solutions with explanations
- Use simple, educational language
- If you detect the language in the image, respond in that language, otherwise use English

{f"Recent conversation context: {context}" if context else ""}

Remember: ONLY analyze if the image is about Mathematics, Physics, Chemistry, or Arabic.

Please analyze this image:"""
        
        # Prepare image for Gemini
        image_part = {
            "mime_type": file.content_type,
            "data": image_data
        }
        
        # Generate response using Gemini Vision
        response = model.generate_content([prompt, image_part])
        answer = response.text
        
        # Check if AI declined to answer off-topic image
        if "I can only help analyze images related to" in answer:
            answer = "I can only help analyze images and answer questions related to Mathematics, Physics, Chemistry, or Arabic. Please upload an image or ask a question about one of these subjects."
        
        # Save to history
        session_id = str(uuid.uuid4())
        question_text = question or "Image analysis (no specific question)"
        save_to_history("image_analysis", question_text, answer)
        
        return ChatResponse(
            answer=answer,
            subject="image_analysis",
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.get("/history/{subject}")
async def get_conversation_history(subject: str, limit: int = 10):
    """Get conversation history for a specific subject"""
    if subject not in conversation_history:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    history = conversation_history[subject]
    
    # Return most recent entries
    return {"subject": subject, "history": history[-limit:]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
