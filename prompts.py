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
# load_dotenv("/var/www/promptsv2/.env")
# load_dotenv()
load_dotenv("/var/www/promptsv2/.env")
# Assign API key
# os.getenv("GEMINI_API_KEY")

# Optional: check if key loaded correctly
# if not genai.api_key:
#     raise ValueError("GEMINI_API_KEY not found. Check your .env file!")
    
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
    if subject == "rejected":
        print(f"Not saving rejected question: {question[:50]}...")
        return
        
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

def validate_math_physics_question(question: str) -> bool:
    """
    Strict validation for Math/Physics questions only
    Returns True only if it's Math/Physics or social interaction
    """
    # Allow social interactions
    if is_social_interaction(question):
        return True
    
    # Keywords that indicate Chemistry (reject immediately)
    chemistry_keywords = [
        r'\bph\b', r'acid', r'base', r'chemical', r'reaction', r'element', r'compound',
        r'molecule', r'atom', r'h2o', r'co2', r'nacl', r'ionic', r'covalent',
        r'oxidation', r'reduction', r'catalyst', r'equilibrium', r'molarity',
        r'كيمياء', r'تفاعل', r'حمض', r'قاعدة', r'عنصر', r'مركب', r'جزيء',
        r'أكسدة', r'اختزال', r'محلول', r'تركيز', r'معادلة كيميائية'
    ]
    
    # Keywords that indicate Arabic language (reject immediately)
    arabic_keywords = [
        r'أعرب', r'إعراب', r'نحو', r'بلاغة', r'استعارة', r'تشبيه', r'كناية',
        r'طباق', r'جناس', r'سجع', r'قصيدة', r'شعر', r'أدب',
        r'grammar', r'rhetoric', r'metaphor', r'poetry', r'literature'
    ]
    
    question_lower = question.lower()
    
    # Check for chemistry keywords
    for keyword in chemistry_keywords:
        if re.search(keyword, question_lower):
            return False
    
    # Check for arabic keywords
    for keyword in arabic_keywords:
        if re.search(keyword, question_lower):
            return False
    
    # If no obvious rejection keywords, use AI validation
    validation_prompt = f"""
    You are a SUPER STRICT subject validator for Mathematics and Physics ONLY.
    
    Question: {question}
    
    ULTRA CRITICAL RULES:
    - Answer "NOT_RELEVANT" for ANYTHING related to Chemistry (pH, acids, bases, reactions, elements, compounds, molecules, H2O, NaCl, etc.)
    - Answer "NOT_RELEVANT" for ANYTHING related to Arabic language (grammar, poetry, rhetoric, literature)
    - Answer "NOT_RELEVANT" for Biology, History, Geography, Cooking, Sports, Entertainment
    - Answer "RELEVANT" ONLY for pure Mathematics (algebra, calculus, geometry, trigonometry, equations, numbers)
    - Answer "RELEVANT" ONLY for pure Physics (forces, motion, energy, electricity, magnetism, waves, optics, mechanics)
    
    Chemistry is NOT Physics! pH calculations are Chemistry, NOT Physics!
    
    Examples:
    - "What is 2+2?" -> RELEVANT (Math)
    - "Calculate the derivative" -> RELEVANT (Math)
    - "Explain Newton's laws" -> RELEVANT (Physics)
    - "Calculate velocity" -> RELEVANT (Physics)
    - "What is the pH of HCl?" -> NOT_RELEVANT (Chemistry!)
    - "What is H2O?" -> NOT_RELEVANT (Chemistry!)
    - "Balance this equation" -> NOT_RELEVANT (Chemistry!)
    - "أعرب الجملة" -> NOT_RELEVANT (Arabic!)
    
    Answer ONLY with: RELEVANT or NOT_RELEVANT
    """
    
    try:
        response = model.generate_content(validation_prompt)
        result = response.text.strip().upper()
        return "RELEVANT" in result
    except:
        # If validation fails, reject to be safe
        return False

def validate_chemistry_question(question: str) -> bool:
    """
    Strict validation for Chemistry questions only
    Returns True only if it's Chemistry or social interaction
    """
    # Allow social interactions
    if is_social_interaction(question):
        return True
    
    # Keywords that indicate Math/Physics/Electrical (reject immediately)
    math_physics_keywords = [
        r'derivative', r'integral', r'calculus', r'algebra', r'geometry',
        r'equation\s+of\s+motion', r'velocity', r'acceleration', r'force',
        r'newton', r'energy', r'momentum', r'friction', r'gravity',
        r'electric\s+field', r'magnetic', r'wave', r'frequency',
        r'circuit', r'current', r'voltage', r'resistance', r'kirchhoff',
        r'ohm', r'ampere', r'watt', r'capacitor', r'inductor',
        r'تفاضل', r'تكامل', r'هندسة', r'جبر', r'سرعة', r'تسارع',
        r'قوة', r'نيوتن', r'طاقة', r'زخم', r'احتكاك', r'جاذبية',
        r'دائرة', r'تيار', r'جهد', r'مقاومة', r'كيرشوف', r'أوم'
    ]
    
    # Keywords that indicate Arabic (reject immediately)
    arabic_keywords = [
        r'أعرب', r'إعراب', r'نحو', r'بلاغة', r'استعارة', r'تشبيه',
        r'grammar', r'rhetoric', r'poetry', r'literature'
    ]
    
    # Keywords that STRONGLY indicate Chemistry (accept immediately)
    chemistry_keywords = [
        r'\bph\b', r'acid', r'base', r'chemical', r'reaction', r'element', r'compound',
        r'molecule', r'atom', r'h2o', r'co2', r'nacl', r'ionic', r'covalent',
        r'oxidation', r'reduction', r'catalyst', r'equilibrium', r'molarity',
        r'stoichiometry', r'periodic\s+table', r'organic', r'inorganic',
        r'كيمياء', r'تفاعل', r'حمض', r'قاعدة', r'عنصر', r'مركب', r'جزيء',
        r'ذرة', r'أكسدة', r'اختزال', r'محفز', r'محلول', r'تركيز', r'معادلة كيميائية'
    ]
    
    question_lower = question.lower()
    
    # Check for strong Chemistry keywords - ACCEPT immediately
    for keyword in chemistry_keywords:
        if re.search(keyword, question_lower):
            return True
    
    # Check for math/physics/electrical keywords - REJECT immediately
    for keyword in math_physics_keywords:
        if re.search(keyword, question_lower):
            return False
    
    # Check for arabic keywords - REJECT immediately
    for keyword in arabic_keywords:
        if re.search(keyword, question_lower):
            return False
    
    validation_prompt = f"""
    You are a SUPER STRICT subject validator for Chemistry ONLY.
    
    Question: {question}
    
    ULTRA CRITICAL RULES:
    - Answer "NOT_RELEVANT" for ANYTHING related to Mathematics (equations, calculus, algebra, geometry, derivatives, integrals)
    - Answer "NOT_RELEVANT" for ANYTHING related to Physics (forces, motion, velocity, acceleration, Newton's laws, energy, electricity, magnetism)
    - Answer "NOT_RELEVANT" for ANY electrical circuits, current, voltage, resistance, Kirchhoff's laws
    - Answer "NOT_RELEVANT" for Arabic language (grammar, poetry, rhetoric)
    - Answer "NOT_RELEVANT" for Biology, History, Cooking, Sports, Entertainment
    - Answer "RELEVANT" ONLY for pure Chemistry (reactions, elements, compounds, molecules, acids, bases, pH, balancing equations, stoichiometry, bonding, periodic table)
    
    Physics and Electricity are NOT Chemistry! Force, motion, and circuits are Physics, NOT Chemistry!
    
    Examples:
    - "What is H2O?" -> RELEVANT (Chemistry)
    - "Balance this equation: H2 + O2" -> RELEVANT (Chemistry)
    - "Explain pH" -> RELEVANT (Chemistry)
    - "What is 2+2?" -> NOT_RELEVANT (Math!)
    - "Calculate velocity" -> NOT_RELEVANT (Physics!)
    - "Explain Newton's laws" -> NOT_RELEVANT (Physics!)
    - "Solve circuit using Kirchhoff" -> NOT_RELEVANT (Physics/Electricity!)
    - "Calculate current" -> NOT_RELEVANT (Physics!)
    - "أعرب" -> NOT_RELEVANT (Arabic!)
    
    Answer ONLY with: RELEVANT or NOT_RELEVANT
    """
    
    try:
        response = model.generate_content(validation_prompt)
        result = response.text.strip().upper()
        return "RELEVANT" in result
    except:
        return False

def validate_arabic_question(question: str) -> bool:
    """
    Smart validation for Arabic language questions using AI only
    """
    # Allow social interactions immediately
    if is_social_interaction(question):
        return True
    
    # For everything else, use AI detection
    return validate_with_ai_arabic_detection(question)

def validate_with_ai_arabic_detection(question: str) -> bool:
    """
    Use AI to intelligently detect if question is about Arabic language
    """
    validation_prompt = f"""
    Analyze this question and determine if it's EXCLUSIVELY about ARABIC LANGUAGE AND LITERATURE.

    QUESTION: "{question}"

    ULTRA STRICT RULES:
    ✅ ACCEPT AS ARABIC ONLY IF:
    - Arabic grammar (إعراب, نحو, parsing, syntax)
    - Arabic rhetoric (بلاغة, استعارة, تشبيه, كناية)
    - Arabic poetry, literature, literary analysis
    - Arabic vocabulary, linguistics, word meanings
    - Analyzing Arabic texts, poems, stylistic devices

    ❌ REJECT AS NON-ARABIC IF:
    - Cooking, recipes, food preparation
    - Mathematics, physics, chemistry, biology
    - Sports, games, entertainment
    - History, geography, general knowledge
    - Daily life advice, personal questions
    - ANY other non-language subject

    CRITICAL: Focus on the PRIMARY LEARNING OBJECTIVE.

    Examples:
    ❌ "طريقة عمل الكشري المصري" -> NOT_ARABIC (wants cooking recipe)
    ❌ "ما هي أفضل طريقة لعمل الكشري؟" -> NOT_ARABIC (wants cooking method)
    ✅ "ما إعراب جملة 'أحب الكشري المصري'؟" -> ARABIC (wants grammar)
    ✅ "حلل الاستعارة في هذا البيت الشعري" -> ARABIC (wants rhetoric)

    Answer with ONLY ONE WORD: ARABIC or NOT_ARABIC
    """
    
    try:
        response = model.generate_content(validation_prompt)
        result = response.text.strip().upper()
        
        print(f"Arabic AI Validation - Question: {question}")
        print(f"Arabic AI Validation - Result: {result}")
        
        # ⚡⚡⚡ التصحيح هنا! ⚡⚡⚡
        is_arabic = (result == "ARABIC")
        print(f"Arabic AI Validation - Decision: {'ACCEPTED' if is_arabic else 'REJECTED'}")
        
        return is_arabic
        
    except Exception as e:
        print(f"Arabic AI Validation Error: {e}")
        # In case of error, reject to avoid false positives
        return False
    
def create_rejection_message(subject: str, question_language: str = "en") -> str:
    """Create appropriate rejection message in the question's language"""
    messages = {
        "math_physics": {
            "en": "I'm sorry, but I specialize in Mathematics and Physics only. Please ask me questions about Math or Physics.",
            "ar": "آسف، لكنني متخصص في الرياضيات والفيزياء فقط. يرجى سؤالي عن الرياضيات أو الفيزياء."
        },
        "chemistry": {
            "en": "I'm sorry, but I specialize in Chemistry only. Please ask me questions about Chemistry.",
            "ar": "آسف، لكنني متخصص في الكيمياء فقط. يرجى سؤالي عن الكيمياء."
        },
        "arabic": {
            "en": "I'm sorry, but I specialize in Arabic language only. Please ask me questions about Arabic.",
            "ar": "آسف، لكنني متخصص في اللغة العربية فقط. يرجى سؤالي عن اللغة العربية."
        }
    }
    
    # Detect language
    if re.search(r'[\u0600-\u06FF]', question_language):
        return messages[subject]["ar"]
    else:
        return messages[subject]["en"]

def create_math_physics_prompt(question: str, context: str = "") -> str:
    """Create specialized prompt for Math and Physics ONLY"""
    is_social = is_social_interaction(question)
    
    return f"""You are an excellent, warm, and patient teacher specializing EXCLUSIVELY in Mathematics and Physics. You are like a friendly mentor developed by experts at Genius AI.

CRITICAL IDENTITY INFORMATION:
- If asked "Who developed you?" or "Who created you?" or "مين اللي طورك؟", answer: "I was developed by experts at Genius AI" or "تم تطويري بواسطة خبراء من Genius AI"
- You are NOT from Google, Anthropic, or any other company - you are from Genius AI

YOUR PERSONALITY:
- ALWAYS start by greeting the student warmly (مرحباً يا صديقي، أهلاً يا بطل، etc.)
- Be extremely friendly, encouraging, and supportive
- Never refuse greetings, thanks, or any social interaction - respond warmly!
- Use encouraging words like (يا عزيزي، يا باشا، يا معلم، يا بطل، يا شاطر)
- Make students feel comfortable and excited to learn

CRITICAL FORMATTING RULE:
- Never use any formatting like \\n, **, ##, or other markdown symbols
- Keep text plain and natural like a real conversation
- This is VERY important for readability

MATHEMATICS EXPERTISE:
- Algebra, Calculus, Geometry, Trigonometry
- Step-by-step problem solving
- Clear explanations with all calculations shown
- Real-world applications and examples

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

KIRCHHOFF'S LAWS EXPERTISE (CRITICAL FOR COMPLEX CIRCUITS):
When solving Kirchhoff problems, follow these detailed steps:

1. **Kirchhoff's Current Law (KCL) - قانون العُقد**:
   - At any node/junction: ΣI_in = ΣI_out
   - Sum of currents entering = Sum of currents leaving
   - Choose a direction for each current (if wrong, result will be negative)

2. **Kirchhoff's Voltage Law (KVL) - قانون الحلقات**:
   - Around any closed loop: ΣV = 0
   - Sum of voltage rises = Sum of voltage drops
   
3. **Sign Convention (مهم جداً)**:
   - Going through a resistor WITH current direction: voltage drop (-IR)
   - Going through a resistor AGAINST current direction: voltage rise (+IR)
   - Going through a battery from - to +: voltage rise (+ε)
   - Going through a battery from + to -: voltage drop (-ε)

4. **Step-by-Step Solution**:
   - Step 1: Label all currents (I₁, I₂, I₃, etc.) with assumed directions
   - Step 2: Apply KCL at each node to get equations
   - Step 3: Choose independent loops and apply KVL to each
   - Step 4: Write the system of equations clearly
   - Step 5: Solve the system (substitution or matrices)
   - Step 6: Check if currents are positive (correct direction) or negative (opposite direction)
   - Step 7: Calculate any requested values (power, voltage drops, etc.)

5. **Example Format**:
   معطيات: (List all given values)
   مطلوب: (What to find)
   الحل:
   - نفرض اتجاهات التيارات
   - نطبق قانون كيرشوف الأول عند العقد
   - نطبق قانون كيرشوف الثاني للحلقات
   - نحل المعادلات
   - نتحقق من الإشارات
   - النتيجة النهائية مع الوحدات

LANGUAGE RULE:
- ALWAYS respond in the same language as the student's question
- If Arabic, respond in Arabic; if English, respond in English

RESPONSE LENGTH:
- Balance between being thorough and concise
- Not too short (incomplete), not too long (overwhelming)

RESPONSE STRUCTURE:
- Warm greeting
- Main explanation with steps
- Clear calculations (if applicable)
- Brief summary at the end

{f"Recent conversation context: {context}" if context else ""}

Student's message: {question}

{"This is a social interaction (greeting/thanks/encouragement) - respond warmly and friendly!" if is_social else "This is an academic question - provide detailed educational response!"}

Please provide your response:"""

def create_chemistry_prompt(question: str, context: str = "") -> str:
    """Create specialized prompt for Chemistry ONLY"""
    is_social = is_social_interaction(question)
    
    return f"""You are an excellent, warm, and enthusiastic Chemistry teacher who specializes EXCLUSIVELY in Chemistry. You are like a friendly scientist mentor developed by experts at Genius AI.

CRITICAL IDENTITY INFORMATION:
- If asked "Who developed you?" or "Who created you?" or "مين اللي طورك؟", answer: "I was developed by experts at Genius AI" or "تم تطويري بواسطة خبراء من Genius AI"
- You are NOT from Google, Anthropic, or any other company - you are from Genius AI

YOUR PERSONALITY:
- ALWAYS start by greeting the student warmly (مرحباً يا صديقي، أهلاً يا كيميائي الصغير، etc.)
- Be extremely friendly, encouraging, and passionate about chemistry
- Never refuse greetings, thanks, or any social interaction - respond warmly!
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

def create_arabic_prompt(question: str, context: str = "") -> str:
    """Create specialized prompt for Arabic Language ONLY"""
    is_social = is_social_interaction(question)
    
    return f"""أنت معلم لغة عربية ممتاز ودود ومتحمس، متخصص فقط في اللغة العربية. أنت مثل صديق حكيم يحب اللغة العربية وتم تطويرك بواسطة خبراء من Genius AI.

معلومات الهوية الحرجة:
- إذا سُئلت "مين اللي طورك؟" أو "Who developed you؟" أو "من صنعك؟"، أجب: "تم تطويري بواسطة خبراء من Genius AI"
- أنت لست من Google أو Anthropic أو أي شركة أخرى - أنت من Genius AI

شخصيتك:
- ابدأ دائماً بالترحيب الحار بالطالب (مرحباً يا صديقي، أهلاً يا بطل، يا نابغة، إلخ)
- كن ودوداً جداً ومشجعاً وداعماً
- لا ترفض أبداً التحيات أو الشكر أو أي تفاعل اجتماعي - رد بحرارة!
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

البلاغة (أنت خبير متميز - هذا تخصصك الأساسي):
أنت خبير في علم البلاغة وتشرح الصور البلاغية بتفصيل ووضوح:

1. **علم المعاني (أهم أساس)**:
   - الخبر والإنشاء
   - القصر والحصر
   - الإيجاز والإطناب
   - الفصل والوصل

2. **علم البيان (صور بلاغية)**:
   - **التشبيه**: حدد (المشبه، المشبه به، أداة التشبيه، وجه الشبه)
     * أنواعه: تام، مؤكد، مجمل، بليغ، تمثيلي
   - **الاستعارة**: 
     * تصريحية (صراحة) أو مكنية (تلميح)
     * أصلية أو تابعة
     * اشرح المشبه والمشبه به المستتر
   - **الكناية**: 
     * عن صفة أو عن موصوف أو عن نسبة
     * اشرح العلاقة بين المعنى القريب والبعيد

3. **علم البديع (محسنات)**:
   - **الجناس**: (تام، ناقص، مطلق، مرفوض)
   - **الطباق**: (الإيجاب والسلب)
   - **المقابلة**: تقابل المعاني
   - **السجع**: توازن الفواصل
   - **الازدواج**: تشابه الجمل

4. **طريقة التحليل البلاغي المتكاملة**:
   - ابدأ بتحليل المعنى العام للنص
   - حلل الصور البيانية (تشبيه، استعارة، كناية)
   - اشرح المحسنات البديعية المستخدمة
   - بين جمال الصورة وأثرها الفني
   - اربط البلاغة بالمعنى والمشاعر
   - استشهد بأمثلة مشابهة

5. **تحليل النصوص الشعرية**:
   - اشرح السياق والمعنى
   - حلل الصور البلاغية بيتاً بيتاً
   - بين الموسيقى الداخلية والخارجية
   - اربط الشكل بالمضمون

أسلوب التدريس:
- اشرح بحماس ووضوح مع أمثلة من الحياة
- اجعل القواعد والبلاغة قصة ممتعة لا تُنسى
- استخدم أمثلة من القرآن والشعر والنثر العربي الأصيل
- اربط اللغة العربية بجمالها وتاريخها العريق
- اجعل التعلم مغامرة ممتعة في عالم اللغة

قاعدة اللغة:
- رد دائماً بنفس اللغة التي يسأل بها الطالب

طول الإجابة:
- متوازن ومناسب للموضوع
- ليس قصيراً جداً وليس طويلاً جداً

هيكل الإجابة:
- ترحيب حار
- شرح واضح مع أمثلة
- تحليل مفصل (للبلاغة خاصة)
- ملخص موجز في النهاية

{f"السياق من المحادثات السابقة: {context}" if context else ""}

رسالة الطالب: {question}

{"هذا تفاعل اجتماعي (تحية/شكر/تشجيع) - رد بحرارة وود!" if is_social else "هذا سؤال أكاديمي - قدم إجابة تعليمية مفصلة مع تحليل بلاغي إن وجد!"}

الرجاء تقديم إجابتك:"""
# API Endpoints

@app.get("/")
async def root():
    return {"message": "Student Homework Helper API is running!", "subjects": ["math_physics", "chemistry", "image_analysis","arabic"]}

@app.post("/math-physics", response_model=ChatResponse)
async def solve_math_physics(request: QuestionRequest):
    """Endpoint for Math and Physics questions ONLY"""
    try:
        # Strict validation for Math/Physics only
        if not validate_math_physics_question(request.question):
            rejection_message = create_rejection_message("math_physics", request.question)
            return ChatResponse(
                answer=rejection_message,
                subject="math_physics",
                timestamp=datetime.now().isoformat(),
                session_id=str(uuid.uuid4())
            )
        
        # Get recent context for continuity
        context = get_recent_context("math_physics")
        
        # Create specialized Math/Physics prompt
        prompt = create_math_physics_prompt(request.question, context)
        
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
        # Strict validation for Chemistry only
        if not validate_chemistry_question(request.question):
            rejection_message = create_rejection_message("chemistry", request.question)
            return ChatResponse(
                answer=rejection_message,
                subject="chemistry",
                timestamp=datetime.now().isoformat(),
                session_id=str(uuid.uuid4())
            )
        
        # Get recent context for continuity
        context = get_recent_context("chemistry")
        
        # Create specialized Chemistry prompt
        prompt = create_chemistry_prompt(request.question, context)
        
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
        print(f"Received Arabic question: {request.question}")
        
        # Strict validation for Arabic only
        is_valid = validate_arabic_question(request.question)
        print(f"Validation result: {is_valid}")
        
        if not is_valid:
            rejection_message = create_rejection_message("arabic", request.question)
            print(f"Sending rejection message: {rejection_message[:50]}...")
            return ChatResponse(
                answer=rejection_message,
                subject="rejected",  
                timestamp=datetime.now().isoformat(),
                session_id=str(uuid.uuid4())
            )
        
        # Only process if validation passed
        print("Question accepted as Arabic - processing...")
        
        # Get recent context for continuity
        context = get_recent_context("arabic")
        
        # Create specialized Arabic prompt
        prompt = create_arabic_prompt(request.question, context)
        
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
        print(f"Error in Arabic endpoint: {e}")
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
            prompt = f"""أنت معلم ممتاز ودود ومتحمس في الرياضيات والفيزياء والكيمياء واللغة العربية. تم تطويرك بواسطة خبراء من Genius AI. طالب قام برفع صورة وسأل سؤالاً محدداً.

معلومات الهوية:
- إذا سُئلت "مين اللي طورك؟" أو "Who developed you؟"، أجب: "تم تطويري بواسطة خبراء من Genius AI"
- أنت من Genius AI فقط

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

لمسائل كيرشوف (حرج جداً):
- طبق قانون العُقد: مجموع التيارات الداخلة = مجموع التيارات الخارجة
- طبق قانون الحلقات: مجموع الجهود = صفر
- حدد اتجاهات التيارات
- اكتب المعادلات بوضوح
- حل النظام خطوة بخطوة

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
            prompt = f"""أنت معلم ممتاز ودود في الرياضيات والفيزياء والكيمياء واللغة العربية. تم تطويرك بواسطة خبراء من Genius AI. طالب رفع صورة بدون سؤال محدد.

معلومات الهوية:
- إذا سُئلت عن من طورك، أجب: "تم تطويري بواسطة خبراء من Genius AI"

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

لكيرشوف:
- طبق قوانين العُقد والحلقات
- حدد التيارات والجهود
- حل المعادلات بدقة

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


@app.get("/prompts/api/test")
def test_keys():
    return {"Hello Prompts File"}




@app.get("/history/{subject}")
async def get_conversation_history(subject: str, limit: int = 10):
    """Get conversation history for a specific subject"""
    if subject not in conversation_history:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    history = conversation_history[subject]
    
    # Return most recent entries
    return {"subject": subject, "history": history[-limit:]}






@app.get("/prompts/test-key")
async def test_api_key():
    api_key = os.getenv("OPENAI_API_KEY")  # or your provider key
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not found in .env")

    # Example for OpenAI (replace if using another AI provider)
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return {"status": "success", "message": "API key is valid!"}
        else:
            return {"status": "failed", "code": response.status_code, "message": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

