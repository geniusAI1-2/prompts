# 🎓 Student Homework Helper

An AI-powered FastAPI application that helps students with their Math, Physics, and Chemistry homework using Google's Gemini AI. The assistant uses an engaging teaching style inspired by popular science educators to make learning fun and memorable.

## ✨ Features

- **Subject-Specific Help**: Specialized assistance for Math, Physics, and Chemistry
- **Image Analysis**: Upload images of homework problems for instant solutions
- **Engaging Teaching Style**: Uses El-Dahih's entertaining approach to explain concepts
- **Conversation History**: Maintains context across conversations for better continuity
- **Multi-language Support**: Responds in the same language as the student's question
- **Step-by-Step Solutions**: Breaks down complex problems into manageable steps
- **Real-world Examples**: Connects academic concepts to everyday life

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/prompts.git
   cd prompts
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn google-generativeai python-dotenv python-multipart
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the application**
   ```bash
   uvicorn prompts:app --host 0.0.0.0 --port 8001 --reload
   ```

6. **Access the API**
   - API Documentation: `http://localhost:8001/docs`
   - Interactive API: `http://localhost:8001/redoc`

## 📚 API Endpoints

### 🔢 Math & Physics Questions
```http
POST /math-physics
```

**Request Body:**
```json
{
  "question": "What is the derivative of x²?"
}
```

### 🧪 Chemistry Questions
```http
POST /chemistry
```

**Request Body:**
```json
{
  "question": "Balance this equation: H2 + O2 → H2O"
}
```

### 📷 Image Analysis
```http
POST /image-analysis
```

**Form Data:**
- `file`: Image file (JPEG, PNG, etc.)
- `question`: Optional specific question about the image

### 📖 Conversation History
```http
GET /history/{subject}?limit=10
```

Get recent conversation history for a specific subject (math_physics, chemistry, image_analysis).

## 🎯 Usage Examples

### Text-based Questions

**Math Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/math-physics",
    json={"question": "Solve for x: 2x + 5 = 15"}
)
print(response.json())
```

**Chemistry Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/chemistry",
    json={"question": "What happens when sodium reacts with water?"}
)
print(response.json())
```

### Image Analysis

```python
import requests

with open("homework_problem.jpg", "rb") as f:
    files = {"file": f}
    data = {"question": "Solve this physics problem"}
    
    response = requests.post(
        "http://localhost:8000/image-analysis",
        files=files,
        data=data
    )
print(response.json())
```

