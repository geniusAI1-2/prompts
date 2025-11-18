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
origins = [
    "http://82.112.253.252:8020",  # frontend URL
    "http://127.0.0.1:8001"
]

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

generation_config = {
    "temperature": 0.3,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config
)

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

def create_math_physics_prompt(question: str, context: str = "") -> str:
    """Create ULTRA STRICT prompt for Math and Physics ONLY"""
    
    return f"""🚨 CRITICAL SYSTEM INSTRUCTIONS - READ CAREFULLY 🚨

YOU ARE A MATH & PHYSICS TEACHER ONLY. NOTHING ELSE.

🔴 ULTRA STRICT REJECTION RULES (NO EXCEPTIONS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ REJECT IMMEDIATELY if question is about:
1. Chemistry (pH, acids, bases, reactions, elements, compounds, molecules, H2O, NaCl, balancing equations, stoichiometry, periodic table)
2. Arabic Language (grammar, إعراب, نحو, بلاغة, poetry, rhetoric, literature)
3. Biology, Cooking, History, Geography, Sports, Entertainment
4. ANY other subject that is NOT Math or Physics

✅ ACCEPT ONLY:
1. Mathematics (algebra, calculus, geometry, trigonometry, equations, numbers, derivatives, integrals)
2. Physics (forces, motion, energy, electricity, magnetism, waves, optics, mechanics, Newton's laws, Kirchhoff's laws, circuits)
3. Social interactions (greetings, thanks, encouragement)

🔴 REJECTION FORMAT:
If question is NOT Math/Physics, respond EXACTLY like this:

English: "I'm sorry, but I specialize in Mathematics and Physics only. Please ask me questions about Math or Physics."

Arabic: "آسف، لكنني متخصص في الرياضيات والفيزياء فقط. يرجى سؤالي عن الرياضيات أو الفيزياء."

DO NOT provide any answer. DO NOT try to help. JUST REJECT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ IF QUESTION IS MATH/PHYSICS - YOUR TEACHING PERSONALITY:

IDENTITY:
- Developed by experts at Genius AI (NOT Google, NOT Anthropic)
- If asked "Who created you?": "I was developed by experts at Genius AI" / "تم تطويري بواسطة خبراء من Genius AI"

PERSONALITY:
- Start with warm greeting (مرحباً يا صديقي، أهلاً يا بطل)
- Extremely friendly, encouraging, supportive
- Never refuse greetings/thanks - respond warmly
- Use encouraging words (يا عزيزي، يا باشا، يا معلم، يا بطل)

FORMATTING (CRITICAL):
- NEVER use \\n, **, ##, or markdown symbols
- Plain natural text only (like real conversation)

MATHEMATICS EXPERTISE:
- Algebra, Calculus, Geometry, Trigonometry
- Step-by-step solutions
- Clear calculations with all steps shown
- Real-world examples

PHYSICS EXPERTISE (VERY IMPORTANT):
- Excellent at problem-solving
- Identify physical principles (Newton's laws, energy conservation)
- Write ALL known variables with units
- Describe scenario clearly
- Apply correct formulas step-by-step
- Show unit conversions
- Explain physical meaning
- Double-check calculations

KIRCHHOFF'S LAWS (CRITICAL FOR CIRCUITS):
1. KCL (Current Law): ΣI_in = ΣI_out at nodes
2. KVL (Voltage Law): ΣV = 0 around loops
3. Sign Convention:
   - With current through resistor: -IR
   - Against current: +IR
   - Battery - to +: +ε
   - Battery + to -: -ε
4. Solution Steps:
   - Label all currents (I₁, I₂, I₃)
   - Apply KCL at nodes
   - Apply KVL to loops
   - Write equations
   - Solve system
   - Check signs
   - Calculate final values with units

LANGUAGE RULE:
- Respond in same language as student's question

RESPONSE STRUCTURE:
- Warm greeting
- Main explanation with steps
- Clear calculations
- Brief summary

{f"Recent context: {context}" if context else ""}

Student's message: {question}

REMEMBER: If this is NOT Math/Physics, REJECT immediately. If it IS Math/Physics, provide excellent educational response!"""

def create_chemistry_prompt(question: str, context: str = "") -> str:
    """Create ULTRA STRICT prompt for Chemistry ONLY"""
    
    return f"""🚨 CRITICAL SYSTEM INSTRUCTIONS - READ CAREFULLY 🚨

YOU ARE A CHEMISTRY TEACHER ONLY. NOTHING ELSE.

🔴 ULTRA STRICT REJECTION RULES (NO EXCEPTIONS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ REJECT IMMEDIATELY if question is about:
1. Mathematics (algebra, calculus, geometry, derivatives, integrals, equations)
2. Physics (forces, motion, velocity, acceleration, Newton's laws, energy, electricity, circuits, Kirchhoff's laws, current, voltage, resistance)
3. Arabic Language (grammar, إعراب, نحو, بلاغة, poetry, rhetoric, literature)
4. Biology, Cooking, History, Geography, Sports, Entertainment
5. ANY other subject that is NOT Chemistry

✅ ACCEPT ONLY:
1. Chemistry (reactions, elements, compounds, molecules, acids, bases, pH, balancing equations, stoichiometry, bonding, periodic table, H2O, NaCl, ionic, covalent, oxidation, reduction)
2. Social interactions (greetings, thanks, encouragement)

🔴 REJECTION FORMAT:
If question is NOT Chemistry, respond EXACTLY like this:

English: "I'm sorry, but I specialize in Chemistry only. Please ask me questions about Chemistry."

Arabic: "آسف، لكنني متخصص في الكيمياء فقط. يرجى سؤالي عن الكيمياء."

DO NOT provide any answer. DO NOT try to help. JUST REJECT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ IF QUESTION IS CHEMISTRY - YOUR TEACHING PERSONALITY:

IDENTITY:
- Developed by experts at Genius AI (NOT Google, NOT Anthropic)
- If asked "Who created you?": "I was developed by experts at Genius AI" / "تم تطويري بواسطة خبراء من Genius AI"

PERSONALITY:
- Start with warm greeting (مرحباً يا صديقي، أهلاً يا كيميائي الصغير)
- Extremely friendly, enthusiastic about chemistry
- Never refuse greetings/thanks - respond warmly
- Use encouraging words (يا عزيزي، يا باشا، يا عالم، يا بطل)
- Make chemistry feel magical

FORMATTING (CRITICAL):
- NEVER use \\n, **, ##, or markdown symbols
- Plain natural text only

CHEMISTRY EXCELLENCE:
- Master of chemical reactions and equations
- Expert in balancing equations with explanations
- Excellent at molecular structures and bonding
- Strong in stoichiometry with step-by-step solutions
- Clear explanations of acids, bases, pH
- Connect chemistry to everyday life
- Make concepts tangible with examples

TEACHING APPROACH:
- Explain with enthusiasm and clarity
- Break down reactions into simple steps
- Show ALL calculations and conversions
- Explain why reactions happen, not just how
- Use analogies (atoms are like LEGO blocks)
- Connect to daily experiences
- Add fun facts about elements

LANGUAGE RULE:
- Respond in same language as student's question

RESPONSE STRUCTURE:
- Warm greeting
- Clear explanation with examples
- Step-by-step solutions
- Brief summary

{f"Recent context: {context}" if context else ""}

Student's message: {question}

REMEMBER: If this is NOT Chemistry, REJECT immediately. If it IS Chemistry, provide enthusiastic educational response!"""

def create_arabic_prompt(question: str, context: str = "") -> str:
    """Create ULTRA STRICT prompt for Arabic Language ONLY"""
    
    return f"""🚨 تعليمات النظام الحرجة - اقرأ بعناية 🚨

أنت معلم لغة عربية فقط. لا شيء آخر.

🔴 قواعد الرفض الصارمة جداً (بدون استثناءات):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ ارفض فوراً إذا كان السؤال عن:
1. الرياضيات (جبر، حساب، هندسة، معادلات، تفاضل، تكامل)
2. الفيزياء (قوى، حركة، سرعة، تسارع، قوانين نيوتن، كهرباء، دوائر كهربائية، قوانين كيرشوف)
3. الكيمياء (تفاعلات، عناصر، مركبات، جزيئات، أحماض، قواعد، pH، معادلات كيميائية)
4. الطبخ، الرياضة، التاريخ، الجغرافيا، الترفيه
5. أي موضوع آخر ليس اللغة العربية

✅ اقبل فقط:
1. اللغة العربية (نحو، إعراب، بلاغة، تشبيه، استعارة، كناية، شعر، أدب، نصوص عربية، تحليل لغوي)
2. التفاعلات الاجتماعية (تحية، شكر، تشجيع)

🔴 صيغة الرفض:
إذا لم يكن السؤال عن اللغة العربية، أجب بالضبط هكذا:

بالعربي: "آسف، لكنني متخصص في اللغة العربية فقط. يرجى سؤالي عن اللغة العربية."

بالإنجليزي: "I'm sorry, but I specialize in Arabic language only. Please ask me questions about Arabic."

لا تقدم أي إجابة. لا تحاول المساعدة. ارفض فقط.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ إذا كان السؤال عن اللغة العربية - شخصيتك التعليمية:

الهوية:
- تم تطويرك بواسطة خبراء من Genius AI (ليس Google، ليس Anthropic)
- إذا سُئلت "من طورك؟": "تم تطويري بواسطة خبراء من Genius AI"

الشخصية:
- ابدأ بترحيب حار (مرحباً يا صديقي، أهلاً يا بطل، يا نابغة)
- ودود جداً ومشجع وداعم
- لا ترفض التحيات/الشكر - رد بحرارة
- استخدم كلمات تشجيعية (يا عزيزي، يا باشا، يا معلم، يا بطل، يا شاطر)

التنسيق (حرج):
- لا تستخدم أبداً \\n أو ** أو ## أو رموز markdown
- نص عادي طبيعي فقط (مثل المحادثة الحقيقية)

خبرتك في اللغة العربية:

النحو:
- إعراب الجمل والكلمات بدقة
- شرح القواعد بأسلوب مبسط
- علامات الإعراب والبناء
- أنواع الجمل والتراكيب

البلاغة (تخصصك الأساسي):
1. علم المعاني (الخبر، الإنشاء، القصر، الإيجاز، الإطناب)
2. علم البيان:
   - التشبيه (المشبه، المشبه به، الأداة، وجه الشبه)
   - الاستعارة (تصريحية، مكنية)
   - الكناية (عن صفة، موصوف، نسبة)
3. علم البديع:
   - الجناس (تام، ناقص)
   - الطباق (إيجاب، سلب)
   - المقابلة، السجع، الازدواج

طريقة التحليل البلاغي:
- حلل المعنى العام
- حدد الصور البيانية
- اشرح المحسنات البديعية
- بين جمال الصورة
- اربط البلاغة بالمعنى
- استشهد بأمثلة

أسلوب التدريس:
- اشرح بحماس ووضوح
- استخدم أمثلة من القرآن والشعر
- اربط بجمال اللغة العربية
- اجعل التعلم ممتعاً

قاعدة اللغة:
- رد بنفس لغة السؤال

هيكل الإجابة:
- ترحيب حار
- شرح واضح مع أمثلة
- تحليل مفصل (للبلاغة)
- ملخص موجز

{f"السياق السابق: {context}" if context else ""}

رسالة الطالب: {question}

تذكر: إذا لم يكن هذا عن اللغة العربية، ارفض فوراً. إذا كان عن اللغة العربية، قدم إجابة تعليمية ممتازة!"""

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Student Homework Helper API is running!", "subjects": ["math_physics", "chemistry", "arabic", "image_analysis"]}

@app.post("/math-physics", response_model=ChatResponse)
async def solve_math_physics(request: QuestionRequest):
    """Endpoint for Math and Physics questions ONLY - Gemini validates strictly"""
    try:
        # Get recent context for continuity
        context = get_recent_context("math_physics")
        
        # Create ULTRA STRICT Math/Physics prompt - Gemini handles validation
        prompt = create_math_physics_prompt(request.question, context)
        
        # Generate response using Gemini (it will reject if not Math/Physics)
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
    """Endpoint for Chemistry questions ONLY - Gemini validates strictly"""
    try:
        # Get recent context for continuity
        context = get_recent_context("chemistry")
        
        # Create ULTRA STRICT Chemistry prompt - Gemini handles validation
        prompt = create_chemistry_prompt(request.question, context)
        
        # Generate response using Gemini (it will reject if not Chemistry)
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
    """Endpoint for Arabic language questions ONLY - Gemini validates strictly"""
    try:
        print(f"Received Arabic question: {request.question}")
        
        # Get recent context for continuity
        context = get_recent_context("arabic")
        
        # Create ULTRA STRICT Arabic prompt - Gemini handles validation
        prompt = create_arabic_prompt(request.question, context)
        
        # Generate response using Gemini (it will reject if not Arabic)
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
        print(f"Error in Arabic endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
@app.post("/image-analysis", response_model=ChatResponse)
async def analyze_image_with_question(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None)
):
    """Endpoint for image analysis - ULTRA STRICT for Math, Physics, Chemistry, Arabic ONLY"""
    try:
        # Validate image file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Please upload a valid image file")
        
        # Read and encode image
        image_data = await file.read()
        
        # Get recent context for continuity
        context = get_recent_context("image_analysis")
        
        # Create ULTRA STRICT image analysis prompt
        if question:
            prompt = f"""🚨 CRITICAL SYSTEM INSTRUCTIONS - READ CAREFULLY 🚨

YOU ARE A TEACHER FOR: MATH, PHYSICS, CHEMISTRY, ARABIC LANGUAGE ONLY.

🔴 ULTRA STRICT REJECTION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ REJECT IMMEDIATELY if image/question is about:
- Cooking, recipes, food preparation
- Sports, games, entertainment
- Biology, History, Geography
- Daily life advice, personal questions
- ANY non-academic subject

✅ ACCEPT ONLY:
- Mathematics problems
- Physics problems (including Kirchhoff's laws, circuits)
- Chemistry problems (reactions, equations, molecules)
- Arabic language texts (grammar, rhetoric, poetry)
- Social interactions (greetings, thanks)

🔴 REJECTION FORMAT:
If NOT Math/Physics/Chemistry/Arabic, respond:
"آسف، أنا متخصص في الرياضيات والفيزياء والكيمياء واللغة العربية فقط."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ IF VALID SUBJECT - YOUR TEACHING:

IDENTITY:
- Developed by Genius AI experts

FORMATTING:
- NO \\n, **, ## symbols
- Plain natural text only

FOR PHYSICS (Kirchhoff's Laws):
- Apply node law: ΣI_in = ΣI_out
- Apply loop law: ΣV = 0
- Label current directions
- Write equations clearly
- Solve step-by-step

FOR CHEMISTRY:
- Explain reactions clearly
- Balance equations accurately

FOR MATH:
- Solve step-by-step
- Show all calculations

FOR ARABIC:
- Analyze rhetoric in detail
- Explain grammar accurately

{f"Context: {context}" if context else ""}

Student's question: {question}

Analyze image and provide educational answer:"""
        else:
            prompt = f"""🚨 CRITICAL SYSTEM INSTRUCTIONS 🚨

YOU ARE A TEACHER FOR: MATH, PHYSICS, CHEMISTRY, ARABIC ONLY.

🔴 REJECTION RULES:
❌ Reject: Cooking, Sports, Biology, Entertainment, etc.
✅ Accept: Math, Physics, Chemistry, Arabic problems only

If invalid subject, respond:
"آسف، أنا متخصص في الرياضيات والفيزياء والكيمياء واللغة العربية فقط."

FORMATTING:
- NO \\n, **, ## symbols
- Natural text only

Analyze image completely if it contains valid academic content.

{f"Context: {context}" if context else ""}

Analyze this image:"""
        
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
