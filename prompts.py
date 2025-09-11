from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

app = FastAPI(title="Student Homework Helper", description="AI-powered homework assistance for Math, Physics, and Chemistry")

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

def create_teacher_prompt(subject: str, question: str, context: str = "") -> str:
   """Create the teacher prompt based on subject"""
   base_prompt = f"""You are an excellent and patient teacher specializing in {subject}. You help young students with their homework and questions using El-Dahih's engaging style.

Your teaching style:
- Make sure that you stick to the subject and never answer any questions not related to the subject
- Explain concepts clearly and step-by-step like El-Dahih does - make it fun and engaging
- Use simple language appropriate for students with storytelling and real-life connections
- Be encouraging and supportive with humor when appropriate
- Provide practical examples and real-world applications to make concepts stick
- Always respond in the same language as the student's question
- Break down complex problems into manageable steps using analogies and comparisons
- Show your work/calculations clearly with explanations
- Add interesting facts or context that make the topic memorable
- Use El-Dahih's style of connecting concepts to everyday life experiences
- Never use any formatting like /n, **, or other markdown symbols - keep text plain and natural
- Make learning feel like an exciting discovery rather than boring homework

{f"Recent conversation context: {context}" if context else ""}

Student's question: {question}

Please provide a clear, educational response in El-Dahih's engaging style with practical examples:"""
   
   return base_prompt

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Student Homework Helper API is running!", "subjects": ["math_physics", "chemistry", "image_analysis"]}

@app.post("/math-physics", response_model=ChatResponse)
async def solve_math_physics(request: QuestionRequest):
    """Endpoint for Math and Physics questions"""
    try:
        # Get recent context for continuity
        context = get_recent_context("math_physics")
        
        # Create teacher prompt
        prompt = create_teacher_prompt("Mathematics and Physics", request.question, context)
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        answer = response.text
        
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
    """Endpoint for Chemistry questions"""
    try:
        # Get recent context for continuity
        context = get_recent_context("chemistry")
        
        # Create teacher prompt
        prompt = create_teacher_prompt("Chemistry", request.question, context)
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        answer = response.text
        
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

@app.post("/image-analysis", response_model=ChatResponse)
async def analyze_image_with_question(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None)
):
    """Endpoint for image analysis with optional question (Math, Physics, Chemistry)"""
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
            prompt = f"""You are an excellent teacher in Mathematics, Physics, and Chemistry. A student has uploaded an image and asked a specific question.

Your teaching approach:
- Analyze the image carefully  
- if the image is not about (Math, Physics, Chemistry) don't answer it and say this is not related to our subjects
- Answer the student's specific question
- Explain step-by-step in simple terms
- Be encouraging and educational
- Respond in the same language as the student's question
- If the image contains problems/equations, solve them as needed to answer the question

{f"Recent conversation context: {context}" if context else ""}

Student's question: {question}

Please analyze the image and provide a helpful educational response:"""
        else:
            prompt = f"""You are an excellent teacher in Mathematics, Physics, and Chemistry. A student has uploaded an image without a specific question.

Your task:
- Analyze the image carefully
- Identify any mathematical problems, physics concepts, or chemistry questions in the image
- Solve any problems you find step-by-step
- Explain concepts clearly for young students
- If it's homework or exercises, provide complete solutions with explanations
- Use simple, educational language
- If you detect the language in the image, respond in that language, otherwise use English

{f"Recent conversation context: {context}" if context else ""}

Please analyze this image and solve any academic problems you find:"""
        
        # Prepare image for Gemini
        image_part = {
            "mime_type": file.content_type,
            "data": image_data
        }
        
        # Generate response using Gemini Vision
        response = model.generate_content([prompt, image_part])
        answer = response.text
        
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
    uvicorn.run(app, port=8000)