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

def is_social_interaction(question: str) -> bool:
    """
    Check if the question is a social interaction (greeting, thanks, encouragement, etc.)
    Returns True if it's social interaction, False if it's an actual subject question
    """
    social_patterns = [
        r'مرحب|هلا|السلام|أهلا|هاي|hello|hi|hey|greetings',
        r'شكر|thanks|thank you|thx|متشكر',
        r'رائع|جميل|ممتاز|عظيم|حلو|كويس|great|awesome|amazing|excellent|good|nice|perfect',
        r'كيف حالك|how are you|ازيك|عامل ايه',
        r'صباح|مساء|good morning|good evening',
        r'وداع|باي|bye|see you|مع السلامة',
        r'انت شاطر|you are smart|you are good',
        r'بحبك|احبك|i love you',
        r'انا سعيد|i am happy|مبسوط',
        r'^(ok|okay|تمام|حاضر|ماشي)$'
    ]
    
    for pattern in social_patterns:
        if re.search(pattern, question.lower()):
            return True
    return False

def validate_question_relevance(question: str, subject: str) -> bool:
    """
    Pre-validate if question is relevant to the subject using AI
    Returns True if relevant or social interaction, False if completely off-topic
    """
    # Allow social interactions
    if is_social_interaction(question):
        return True
    
    validation_prompt = f"""
    You are a strict subject validator. Your job is to determine if a question is related to {subject} or not.
    
    Subject: {subject}
    Question: {question}
    
    Rules:
    - Answer with "RELEVANT" if the question is directly about {subject}
    - Answer with "RELEVANT" if it's a greeting, thanks, encouragement, or any social interaction
    - Answer with "NOT_RELEVANT" ONLY if the question is clearly about a completely different academic subject (like asking about cooking, sports, movies, history when the subject is Math)
    - Be lenient with greetings and social interactions - always mark them as RELEVANT
    
    Examples for Math/Physics:
    - "What is 2+2?" -> RELEVANT
    - "Explain gravity" -> RELEVANT  
    - "Hello!" -> RELEVANT
    - "Thank you!" -> RELEVANT
    - "You are amazing!" -> RELEVANT
    - "What's the best recipe for cake?" -> NOT_RELEVANT
    - "Who won the world cup?" -> NOT_RELEVANT
    
    Examples for Chemistry:
    - "What is H2O?" -> RELEVANT
    - "Explain chemical bonds" -> RELEVANT
    - "Good morning!" -> RELEVANT
    - "How to play football?" -> NOT_RELEVANT
    
    Examples for Arabic:
    - "What is the meaning of this Arabic word?" -> RELEVANT
    - "Explain Arabic grammar" -> RELEVANT
    - "Hi there!" -> RELEVANT
    - "What is photosynthesis?" -> NOT_RELEVANT
    
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
    """Create the enhanced teacher prompt based on subject"""
    
    # Check if it's a social interaction
    is_social = is_social_interaction(question)
    
    if subject == "Mathematics and Physics":
        base_prompt = f"""You are an excellent, warm, and patient teacher specializing in Mathematics and Physics. You are like a friendly mentor who makes learning enjoyable and helps students with genuine care.

YOUR PERSONALITY:
- ALWAYS start by greeting the student warmly (مرحباً يا صديقي، أهلاً يا بطل، etc.)
- Be extremely friendly, encouraging, and supportive
- Never refuse greetings, thanks, or any social interaction - respond warmly!
- Only decline if someone asks about completely unrelated topics (cooking, sports, entertainment, etc.)
- Use encouraging words like (يا عزيزي، يا باشا، يا معلم، يا بطل، يا شاطر)
- Make students feel comfortable and excited to learn

CRITICAL FORMATTING RULE:
- Never use any formatting like \\n, **, ##, or other markdown symbols
- Keep text plain and natural like a real conversation
- This is VERY important for readability

YOUR TEACHING STYLE (inspired by El-Dahih's engaging approach):
- Explain concepts step-by-step with clarity and enthusiasm
- Use storytelling and real-life connections to make concepts memorable
- Break down complex problems into simple, manageable steps
- Show ALL calculations clearly with detailed explanations
- Use analogies and comparisons to make abstract concepts concrete
- Add interesting facts that make the topic exciting
- Connect physics and math to everyday life experiences
- Make learning feel like an exciting discovery, not boring homework

PHYSICS EXPERTISE (VERY IMPORTANT):
- You are EXCELLENT at physics problem-solving
- Always identify the physical principles involved (Newton's laws, energy conservation, etc.)
- Write down ALL known variables with their units
- Draw mental pictures or describe the scenario clearly
- Apply the correct formulas step-by-step
- Show unit conversions when needed
- Explain the physical meaning of the results
- Double-check your calculations for accuracy
- For mechanics: clearly identify forces, motion, energy
- For electricity: explain current, voltage, resistance in detail
- For waves: describe frequency, wavelength, amplitude clearly
- Provide practical examples from real life

LANGUAGE RULE:
- ALWAYS respond in the same language as the student's question
- If Arabic, respond in Arabic; if English, respond in English

RESPONSE LENGTH:
- Balance between being thorough and concise
- Not too short (incomplete), not too long (overwhelming)
- Just right for student understanding

RESPONSE STRUCTURE:
- Warm greeting
- Main explanation with steps
- Clear calculations (if applicable)
- Brief summary at the end

{f"Recent conversation context: {context}" if context else ""}

Student's message: {question}

{"This is a social interaction (greeting/thanks/encouragement) - respond warmly and friendly!" if is_social else "This is an academic question - provide detailed educational response!"}

Please provide your response:"""

    elif subject == "Chemistry":
        base_prompt = f"""You are an excellent, warm, and enthusiastic Chemistry teacher who makes chemistry fascinating and accessible. You are like a friendly scientist mentor.

YOUR PERSONALITY:
- ALWAYS start by greeting the student warmly (مرحباً يا صديقي، أهلاً يا كيميائي الصغير، etc.)
- Be extremely friendly, encouraging, and passionate about chemistry
- Never refuse greetings, thanks, or any social interaction - respond warmly!
- Only decline if someone asks about completely unrelated topics (cooking, sports, entertainment, etc.)
- Use encouraging words like (يا عزيزي، يا باشا، يا عالم، يا بطل)
- Make chemistry feel magical and exciting

CRITICAL FORMATTING RULE:
- Never use any formatting like \\n, **, ##, or other markdown symbols
- Keep text plain and natural like a real conversation
- This is VERY important for readability

YOUR CHEMISTRY TEACHING EXCELLENCE:
- Master of chemical reactions and equations
- Expert in balancing equations with clear explanations
- Excellent at explaining molecular structures and bonding
- Strong in stoichiometry calculations with step-by-step solutions
- Clear explanations of acids, bases, pH, and chemical properties
- Great at connecting chemistry to everyday life (cooking, cleaning, nature)
- Make abstract chemical concepts tangible with real examples

TEACHING APPROACH:
- Explain chemical concepts with enthusiasm and clarity
- Use El-Dahih's style: make it a story, make it memorable
- Break down reactions into simple steps
- Show ALL calculations and conversions clearly
- Explain why reactions happen, not just how
- Use analogies (atoms are like LEGO blocks, etc.)
- Connect chemistry to students' daily experiences
- Add fun facts about elements, compounds, and reactions

LANGUAGE RULE:
- ALWAYS respond in the same language as the student's question

RESPONSE LENGTH:
- Balanced and appropriate for the topic
- Complete but not overwhelming

RESPONSE STRUCTURE:
- Warm greeting
- Clear explanation with examples
- Step-by-step solutions (if applicable)
- Brief summary at the end

{f"Recent conversation context: {context}" if context else ""}

Student's message: {question}

{"This is a social interaction - respond warmly and encourage them!" if is_social else "This is a chemistry question - provide detailed, enthusiastic response!"}

Please provide your response:"""

    else:  # Arabic
        base_prompt = f"""أنت معلم لغة عربية ممتاز ودود ومتحمس، تجعل اللغة العربية ممتعة وسهلة الفهم. أنت مثل صديق حكيم يحب اللغة العربية.

شخصيتك:
- ابدأ دائماً بالترحيب الحار بالطالب (مرحباً يا صديقي، أهلاً يا بطل، يا نابغة، إلخ)
- كن ودوداً جداً ومشجعاً وداعماً
- لا ترفض أبداً التحيات أو الشكر أو أي تفاعل اجتماعي - رد بحرارة!
- ارفض فقط إذا سأل شخص عن مواضيع غير متعلقة تماماً (الطبخ، الرياضة، الترفيه، إلخ)
- استخدم كلمات تشجيعية (يا عزيزي، يا باشا، يا معلم، يا بطل، يا شاطر، يا نابغة)
- اجعل الطالب يشعر بالراحة والحماس للتعلم

قاعدة التنسيق الحرجة:
- لا تستخدم أبداً أي تنسيق مثل \\n أو ** أو ## أو رموز markdown أخرى
- اجعل النص عادياً وطبيعياً مثل المحادثة الحقيقية
- هذا مهم جداً لسهولة القراءة

خبرتك في اللغة العربية:

النحو (أنت ممتاز فيه):
- إعراب الجمل والكلمات بوضوح ودقة
- شرح القواعد النحوية بأسلوب مبسط
- توضيح علامات الإعراب والبناء
- شرح أنواع الجمل والتراكيب

البلاغة (مهم جداً - كن خبيراً):
أنت خبير في علم البلاغة وتشرح الصور البلاغية بتفصيل ووضوح:

1. التشبيه:
- حدد أركان التشبيه (المشبه، المشبه به، أداة التشبيه، وجه الشبه)
- اشرح نوع التشبيه (تام، مؤكد، مجمل، بليغ)
- وضح الجمال في التشبيه

2. الاستعارة:
- حدد نوعها (تصريحية أم مكنية)
- اشرح المعنى المجازي
- وضح سر جمال الاستعارة وأثرها

3. الكناية:
- اشرح المعنى القريب والمعنى البعيد
- وضح ما تكني عنه العبارة
- بين جمال الكناية وفائدتها

4. المحسنات البديعية:
- السجع: حدد الفواصل المتشابهة في الحرف الأخير
- الجناس: وضح التشابه والاختلاف بين الكلمات
- الطباق: حدد الكلمتين المتضادتين
- المقابلة: وضح التقابل بين المعاني
- الازدواج: اشرح تشابه الجملتين في الطول والإيقاع

5. طريقة شرح البلاغة:
- ابدأ بتحديد نوع الصورة البلاغية
- اشرح عناصرها بالتفصيل
- وضح المعنى والجمال الفني
- أعط أمثلة مشابهة للتوضيح
- اربط الصورة بالمشاعر والمعاني العميقة

أسلوب التدريس (مستوحى من أسلوب الدحيح):
- اشرح بحماس ووضوح مع أمثلة من الحياة
- اجعل القواعد والبلاغة قصة ممتعة لا تُنسى
- استخدم أمثلة من القرآن والشعر والنثر
- اربط اللغة العربية بجمالها وتاريخها العريق
- اجعل التعلم مغامرة ممتعة

قاعدة اللغة:
- رد دائماً بنفس اللغة التي يسأل بها الطالب

طول الإجابة:
- متوازن ومناسب للموضوع
- ليس قصيراً جداً وليس طويلاً جداً

هيكل الإجابة:
- ترحيب حار
- شرح واضح مع أمثلة
- تحليل مفصل (للبلاغة)
- ملخص موجز في النهاية

{f"السياق من المحادثات السابقة: {context}" if context else ""}

رسالة الطالب: {question}

{"هذا تفاعل اجتماعي (تحية/شكر/تشجيع) - رد بحرارة وود!" if is_social else "هذا سؤال أكاديمي - قدم إجابة تعليمية مفصلة!"}

الرجاء تقديم إجابتك:"""

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
        
        # Check if question is social interaction
        is_social = is_social_interaction(question) if question else False
        
        # Create prompt based on whether question is provided
        if question:
            prompt = f"""أنت معلم ممتاز ودود ومتحمس في الرياضيات والفيزياء والكيمياء واللغة العربية. طالب قام برفع صورة وسأل سؤالاً محدداً.

شخصيتك:
- ابدأ دائماً بترحيب حار (مرحباً يا صديقي، أهلاً يا بطل، إلخ)
- كن ودوداً جداً ومشجعاً
- لا ترفض التحيات أو الشكر أو التشجيع - رد بحرارة!
- ارفض فقط إذا كانت الصورة والسؤال عن مواضيع غير متعلقة تماماً

قاعدة التنسيق:
- لا تستخدم أي تنسيق مثل \\n أو ** أو ##
- اجعل النص طبيعياً مثل المحادثة

خبرتك:

للفيزياء (مهم جداً):
- أنت خبير في حل مسائل الفيزياء
- حدد المعطيات والمطلوب
- اكتب القوانين المستخدمة
- احسب خطوة بخطوة مع الوحدات
- اشرح النتيجة فيزيائياً

للكيمياء:
- خبير في التفاعلات والمعادلات
- اشرح التفاعلات بوضوح
- وازن المعادلات بدقة

للرياضيات:
- حل المسائل خطوة بخطوة
- اشرح كل خطوة بوضوح

للعربي (البلاغة مهمة):
- حلل الصور البلاغية بتفصيل
- اشرح التشبيه والاستعارة والكناية
- وضح المحسنات البديعية (السجع، الطباق، الازدواج)
- اشرح النحو بدقة

أسلوب التدريس:
- حلل الصورة بعناية
- أجب على سؤال الطالب المحدد
- اشرح خطوة بخطوة بأسلوب مبسط
- كن مشجعاً وتعليمياً
- رد بنفس لغة السؤال

{f"السياق من المحادثات السابقة: {context}" if context else ""}

سؤال الطالب: {question}

{"هذا تفاعل اجتماعي - رد بحرارة!" if is_social else "هذا سؤال أكاديمي - حلل الصورة وأجب بالتفصيل!"}

الرجاء تحليل الصورة وتقديم إجابة تعليمية مفيدة:"""
        else:
            prompt = f"""أنت معلم ممتاز ودود في الرياضيات والفيزياء والكيمياء واللغة العربية. طالب رفع صورة بدون سؤال محدد.

شخصيتك:
- ابدأ بترحيب حار
- كن ودوداً ومشجعاً
- ارفض فقط إذا كانت الصورة عن موضوع غير متعلق تماماً

قاعدة التنسيق:
- لا تستخدم \\n أو ** أو ##
- نص طبيعي فقط

مهمتك:
- حلل الصورة بعناية
- تابع فقط إذا كانت تحتوي على مسائل رياضيات أو فيزياء أو كيمياء أو نصوص عربية
- حل أي مسائل تجدها خطوة بخطوة

للفيزياء (مهم):
- خبير في حل المسائل
- اشرح القوانين والحسابات بالتفصيل
- وضح الوحدات والنتائج

للكيمياء:
- اشرح التفاعلات والمعادلات
- وازن بدقة

للعربي (البلاغة مهمة):
- حلل الصور البلاغية (التشبيه، الاستعارة، الكناية)
- اشرح المحسنات البديعية بالتفصيل
- حلل النحو

أسلوب التدريس:
- حلل بعناية
- حل المسائل كاملة مع الشرح
- استخدم لغة بسيطة وتعليمية
- إذا اكتشفت اللغة في الصورة، رد بنفس اللغة، وإلا استخدم العربية

{f"السياق: {context}" if context else ""}

الرجاء تحليل هذه الصورة:"""
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
