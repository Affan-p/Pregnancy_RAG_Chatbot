import gradio as gr
import os
import sys
from datetime import datetime
import traceback


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from backend.rag_functions import get_direct_answer, get_answer_with_query_engine
from backend.utils import get_index
print("✅ Successfully imported RAG functions")

class PregnancyRiskAgent:
    def __init__(self):
        self.conversation_history = []  
        self.current_symptoms = {}
        self.risk_assessment_done = False
        self.user_context = {}  
        self.last_user_query = ""  
        
        
        self.symptom_questions = [
            "Are you currently experiencing any unusual bleeding or discharge?",
            "How would you describe your baby's movements today compared to yesterday?",
            "Have you had any headaches that won't go away or that affect your vision?",
            "Do you feel any pressure or pain in your pelvis or lower back?",
            "Are you experiencing any other symptoms? (If yes, please describe briefly)"
        ]
        
        self.current_question_index = 0
        self.waiting_for_first_response = True
        
    def add_to_conversation_history(self, role, message):
        self.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_conversation_context(self):
        context_parts = []
        
        recent_history = self.conversation_history[-10:]
        
        for entry in recent_history:
            if entry["role"] == "user":
                context_parts.append(f"User: {entry['message']}")
            else:
                context_parts.append(f"Assistant: {entry['message'][:200]}...")
        
        return "\n".join(context_parts)
    
    def is_follow_up_question(self, user_input):
        follow_up_indicators = [
            "what about", "can you explain", "what does", "why", "how", 
            "tell me more", "what should i", "is it normal", "should i be worried",
            "what if", "when should", "how long", "what causes", "is this"
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in follow_up_indicators)
    
    def process_user_input(self, user_input, chat_history):
        try:
            self.last_user_query = user_input
            self.add_to_conversation_history("user", user_input)
            
            
            if self.waiting_for_first_response:
                self.current_symptoms[f"question_0"] = user_input
                self.waiting_for_first_response = False
                self.current_question_index = 1
                
                if self.current_question_index < len(self.symptom_questions):
                    bot_response = f"{self.symptom_questions[self.current_question_index]}"
                else:
                    bot_response = self.provide_risk_assessment()
                    self.risk_assessment_done = True
                
                self.add_to_conversation_history("assistant", bot_response)
                return bot_response
            
            
            elif self.current_question_index < len(self.symptom_questions) and not self.risk_assessment_done:
                self.current_symptoms[f"question_{self.current_question_index}"] = user_input
                self.current_question_index += 1
                
                if self.current_question_index < len(self.symptom_questions):
                    bot_response = f"{self.symptom_questions[self.current_question_index]}"
                else:
                    bot_response = self.provide_risk_assessment()
                    self.risk_assessment_done = True
                
                self.add_to_conversation_history("assistant", bot_response)
                return bot_response
            
            
            else:
                bot_response = self.handle_follow_up_conversation(user_input)
                self.add_to_conversation_history("assistant", bot_response)
                return bot_response
                
        except Exception as e:
            print(f"❌ Error in process_user_input: {e}")
            traceback.print_exc()
            error_response = "I encountered an error. Please try again or consult your healthcare provider."
            self.add_to_conversation_history("assistant", error_response)
            return error_response
    
    def handle_follow_up_conversation(self, user_input):
        try:
            print(f"🔍 Processing follow-up question: {user_input}")
            
            symptom_summary = self.create_symptom_summary()
            conversation_context = self.get_conversation_context()
            
            if any(word in user_input.lower() for word in ["last", "previous", "what did i ask", "my question"]):
                if self.last_user_query:
                    return f"Your last question was: \"{self.last_user_query}\"\n\nWould you like me to elaborate on that topic or do you have a different question?"
                else:
                    return "I don't have a record of your previous question. Could you please rephrase what you'd like to know?"
            
            rag_response = get_direct_answer(user_input, symptom_summary, conversation_context=conversation_context, is_risk_assessment=False)
            
            if "Error" in rag_response or len(rag_response) < 50:
                print("🔄 Trying alternative method...")
                rag_response = get_answer_with_query_engine(user_input)
            
            bot_response = f"""Based on your symptoms and medical literature:

{rag_response}"""
            
            return bot_response
            
        except Exception as e:
            print(f"❌ Error in follow-up conversation: {e}")
            return "I encountered an error processing your question. Could you please rephrase it or consult your healthcare provider?"
        
    def create_symptom_summary(self):
        if not self.current_symptoms:
            return "No specific symptoms reported yet"
            
        summary_parts = []
        for i, (key, response) in enumerate(self.current_symptoms.items()):
            if i < len(self.symptom_questions):
                question = self.symptom_questions[i]
                summary_parts.append(f"{question}: {response}")
        return "\n".join(summary_parts)

    def parse_risk_level(self, text):
        import re
        
        patterns = [
            r'\*\*Risk Level:\*\*\s*(Low|Medium|High)',  
            r'Risk Level:\s*\*\*(Low|Medium|High)\*\*',  
            r'Risk Level:\s*(Low|Medium|High)',          
            r'\*\*Risk Level:\*\*\s*<(Low|Medium|High)>', 
            r'Risk Level.*?<(Low|Medium|High)>',          
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                risk_level = match.group(1).capitalize()
                print(f"✅ Successfully parsed risk level: {risk_level}")
                return risk_level
        
        print(f"❌ Could not parse risk level from: {text[:200]}...")
        return None

    def provide_risk_assessment(self):
        all_symptoms = self.create_symptom_summary()
        
        rag_query = f"Analyze these pregnancy symptoms for risk assessment:\n{all_symptoms}\n\nProvide risk level and medical recommendations."
        detailed_analysis = get_direct_answer(rag_query, all_symptoms, is_risk_assessment=True)

        print(f"🔍 RAG Response: {detailed_analysis[:300]}...")
        
        llm_risk_level = self.parse_risk_level(detailed_analysis)
        
        if llm_risk_level:
            risk_level = llm_risk_level
            
            if risk_level == "Low":
                action = "✅ Continue routine prenatal care and self-monitoring"
            elif risk_level == "Medium":
                action = "⚠️ Contact your doctor within 24 hours"
            elif risk_level == "High":
                action = "🚨 Immediate visit to ER or OB emergency care required"
        else:
            print("⚠️ RAG assessment failed, using fallback")
            risk_level = "Medium"
            action = "⚠️ Contact your doctor within 24 hours"

        symptom_list = []
        for i, (key, symptom) in enumerate(self.current_symptoms.items()):
            question = self.symptom_questions[i] if i < len(self.symptom_questions) else f"Question {i+1}"
            symptom_list.append(f"• **{question}**: {symptom}")
        
        assessment = f"""
## 🏥 **Risk Assessment Complete**

**Risk Level: {risk_level}**
**Recommended Action: {action}**

### 📋 **Your Reported Symptoms:**
{chr(10).join(symptom_list)}

### 🔬 **Medical Analysis:**
{detailed_analysis}

### 💡 **Next Steps:**
- Follow the recommended action above
- Keep monitoring your symptoms
- Contact your healthcare provider if symptoms worsen
- Feel free to ask me any follow-up questions about pregnancy health

"""
        return assessment
    
    def reset_conversation(self):
        self.conversation_history = []
        self.current_symptoms = {}
        self.current_question_index = 0
        self.risk_assessment_done = False
        self.waiting_for_first_response = True
        self.user_context = {}
        self.last_user_query = ""
        return get_welcome_message()

def get_welcome_message():
    return """Hello! I'm here to help assess pregnancy-related symptoms and provide risk insights based on medical literature.

I'll ask you a few important questions about your current symptoms, then provide a risk assessment and recommendations. After that, feel free to ask any follow-up questions!

**To get started, please tell me:**
Are you currently experiencing any unusual bleeding or discharge?

---
⚠️ **Important**: This tool is for informational purposes only and should not replace professional medical care. In case of emergency, contact your healthcare provider immediately."""


def create_new_agent():
    
    return PregnancyRiskAgent()


agent = create_new_agent()

def chat_interface_with_reset(user_input, history):
    global agent
    
    if user_input.lower() in ["reset", "restart", "new assessment"]:
        agent = create_new_agent()
        return get_welcome_message()
    
    response = agent.process_user_input(user_input, history)
    return response

def reset_chat():
    global agent
    agent = create_new_agent()
    return [{"role": "assistant", "content": get_welcome_message()}], ""



custom_css = """
body, .gradio-container {
    color: yellow !important;
}

.header {
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.header h1 {
    color: black !important;
    margin-bottom: 0.5rem;
    font-size: 2.5rem;
}

.header p {
    color: black !important;
    font-size: 1.1rem;
    margin: 0.5rem 0;
}

.warning {
    background-color: #fff4e6;
    border-left: 6px solid #ff7f00;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}

.warning h3 {
    color: black !important;
    margin-top: 0;
}

.warning p {
    color: black !important;
    line-height: 1.6;
}

div[style*="background-color: #e8f5e8"] {
    color: black !important;
}

div[style*="background-color: #e8f5e8"] h3 {
    color: black !important;
}

div[style*="background-color: #e8f5e8"] li {
    color: black !important;
}

.chatbot {
    color: black !important;
}

.message {
    color: black !important;
}

/* Hide Gradio footer elements */
.footer {
    display: none !important;
}

.gradio-container .footer {
    display: none !important;
}

footer {
    display: none !important;
}

.api-docs {
    display: none !important;
}

.built-with {
    display: none !important;
}

.gradio-container > .built-with {
    display: none !important;
}

.settings {
    display: none !important;
}

div[class*="footer"] {
    display: none !important;
}

div[class*="built"] {
    display: none !important;
}

*:contains("Built with Gradio") {
    display: none !important;
}

*:contains("Use via API") {
    display: none !important;
}

*:contains("Settings") {
    display: none !important;
}
"""


with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
    <div class="header">
        <h1>🤱 Pregnancy RAG Chatbot</h1>
        <p><strong style="color: black !important;">Proactive RAG-powered pregnancy risk management</strong></p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="warning">
                <h3>⚠️ Medical Disclaimer</h3>
                <p>This AI assistant provides information based on medical literature but is NOT a substitute for professional medical advice, diagnosis, or treatment.</p>
                <p><strong style="color: black !important;">In emergencies, call emergency services immediately.</strong></p>
            </div>
            """)
    
    
    chatbot = gr.ChatInterface(
        fn=chat_interface_with_reset,
        chatbot=gr.Chatbot(
            value=[{"role": "assistant", "content": get_welcome_message()}],
            show_label=False,
            type='messages'
        ),
        textbox=gr.Textbox(
            placeholder="Type your response here...", 
            show_label=False,
            max_length=1000,
            submit_btn=True
        )
    )

    with gr.Row():
        reset_btn = gr.Button("🔄 Start New Assessment", variant="secondary")

        reset_btn.click(
            fn=reset_chat,
            outputs=[chatbot.chatbot, chatbot.textbox],
            show_progress=False
        )


def check_groq_connection():
    try:
        from backend.utils import llm
        test_response = llm.complete("Hello")
        print("✅ Groq connection successful")
        return True
    except Exception as e:
        print(f"❌ Groq connection failed: {e}")
        return False


def refresh_page():
    """Force a complete page refresh"""
    return None



if __name__ == "__main__":
    print("🚀 Starting GraviLog Pregnancy Risk Assessment Agent...")
    check_groq_connection()
    
    
    is_hf_space = os.getenv('SPACE_ID') is not None
    
    if is_hf_space:
        print("📍 Running on Hugging Face Spaces")
        print("📍 Each page refresh will start a new conversation")
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  
            debug=False   
        )
    else:
        print("📍 Running locally")
        print("📍 Using Groq API for LLM processing")
        print("📍 Make sure your GROQ_API_KEY is set in environment variables")
        print("📍 Make sure your Pinecone index is set up and populated")
        
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=True,
            show_error=True
        )