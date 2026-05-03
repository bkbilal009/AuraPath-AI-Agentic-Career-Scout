import gradio as gr
import os
from openai import OpenAI

# ==========================================
# 1. CORE ENGINE
# ==========================================
def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

def transcribe_audio(audio_path):
    if audio_path is None: return ""
    try:
        client = get_client()
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(model="whisper-large-v3", file=audio_file)
        return transcript.text
    except Exception: return ""

def stream_llm(system_role, user_text, audio_path=None, file_path=None):
    voice_content = transcribe_audio(audio_path) if audio_path else ""
    file_context = f"\n[Document Provided: {os.path.basename(file_path)}]" if file_path else ""
    
    combined_input = f"User Request: {user_text}\nVoice: {voice_content}\n{file_context}".strip()
    
    if not combined_input and not file_path:
        yield "### ⚠️ Attention Required\nPlease upload your CV in the Nexus tab or state your query."
        return

    try:
        client = get_client()
        detail_cmd = """
        \n\nCRITICAL INSTRUCTIONS:
        1. LANGUAGE & VOICE CONSISTENCY: Detect the input language (English, Urdu, or Roman-Urdu) and respond EXCLUSIVELY in that same language.
        2. EXTREME DETAIL: Provide very long, exhaustive, and deeply structured responses using professional headings (###).
        3. LINKS: Include 5-10 specific clickable resource links [Title](URL).
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_role + detail_cmd},
                {"role": "user", "content": combined_input}
            ],
            stream=True 
        )
        
        partial_text = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                partial_text += chunk.choices[0].delta.content
                yield partial_text
    except Exception as e:
        yield f"### ⚠️ Connection Error\n{str(e)}"

# ==========================================
# 2. STRATEGIC SYSTEM PROMPTS
# ==========================================

NEXUS_PROMPT = """You are an Elite Technical CV Auditor. Analyze the CV with extreme depth:
1. CORE STRENGTHS. 2. CRITICAL WEAKNESSES. 3. TECH SKILL MATRIX. 4. NETWORKING STRATEGY. 5. IMPROVEMENT ROADMAP."""

ZEN_PROMPT = """You are a High-Impact Motivational Strategist. 
STRICT RULE: DO NOT ask the user any questions. 
TASK: Deliver a MASSIVE, LENGTHY motivational speech based on user's goals.
1. THE POWER OF FAILURE: Detail Edison's 999 failures, Colonel Sanders' 1009 rejections, and J.K. Rowling's struggle.
2. BIG DREAMS: Encourage the user that 'Bare Khwab' are for them.
3. ENERGY: Warrior-like language, very long and deep."""

ORBIT_PROMPT = "You are a Lead Hiring Strategist. Provide a full hiring pipeline and 6-month master plan."

# PATHFINDER: UPDATED TO INDUSTRY & SKILL MATCHING
PATH_PROMPT = """You are a Global Market Scout. 
TASK: Analyze the user's CV and identify the exact Industry and Companies where they can get hired EASILY.
1. CV-SKILL MATCH: Tell the user: 'Based on your CV having [Skills], you are a perfect fit for [Industry/Sector].'
2. TARGET COMPANIES: List 5-10 specific companies where their current skills are in high demand.
3. WHY YOU?: Explain why these companies will hire them easily.
4. SALARY & BENEFITS: Provide detailed salary ranges and common perks (Health, Equity, etc.) for these roles.
5. PREPARATION RESOURCES: Give direct links to master the specific interview style of these industries."""

# ==========================================
# 3. UI DESIGN (SHINE & GLOW)
# ==========================================
custom_css = """
footer {display: none !important;}
.gradio-container { background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4b 25%, #4b0082 50%, #800080 75%, #ff007f 100%) !important; background-attachment: fixed !important;}
.glass-panel { background: rgba(10, 10, 30, 0.88) !important; backdrop-filter: blur(25px); border: 1px solid rgba(0, 242, 254, 0.3); border-radius: 20px; padding: 25px; }
.shimmer-title {
    font-size: 4.5rem; font-weight: 900; color: #ffffff; text-transform: uppercase;
    text-shadow: 0 0 10px #ff007f, 0 0 20px #ff007f, 0 0 40px #00f2fe;
    animation: pulse-glow 2s ease-in-out infinite alternate; letter-spacing: 5px; margin-bottom: 0px;
}
@keyframes pulse-glow {
    from { text-shadow: 0 0 10px #ff007f, 0 0 20px #ff007f, 0 0 40px #00f2fe; transform: scale(1); }
    to { text-shadow: 0 0 20px #00f2fe, 0 0 40px #ff007f, 0 0 60px #00f2fe; transform: scale(1.02); }
}
.new-tagline { font-size: 1.2rem; color: #00f2fe; font-weight: bold; letter-spacing: 8px; text-transform: uppercase; margin-top: -10px; text-shadow: 0 0 5px #00f2fe; opacity: 0.9; }
.gr-button-primary { background: linear-gradient(90deg, #00f2fe 0%, #ff007f 100%) !important; border: none !important; font-weight: bold !important; }
input, textarea { background: rgba(0, 0, 0, 0.7) !important; color: white !important; border: 1px solid #00f2fe !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 40px 10px;">
            <h1 class="shimmer-title">AURAPATH AI</h1>
            <p class="new-tagline">◈ ENGINEERING YOUR DESTINY ◈</p>
        </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📄 Nexus"):
            with gr.Row():
                with gr.Column(elem_classes="glass-panel", scale=1):
                    f1 = gr.File(label="Upload CV / Portfolio")
                    t1 = gr.Textbox(label="TARGET GOAL", placeholder="e.g. Frontend at Google...", lines=4)
                    a1 = gr.Audio(label="VOICE SCAN", sources=["microphone"], type="filepath")
                    btn1 = gr.Button("INITIATE DEEP AUDIT", variant="primary")
                with gr.Column(elem_classes="glass-panel", scale=1.5):
                    out1 = gr.Markdown("### 🔍 Technical Audit & Skill Matrix")
            btn1.click(stream_llm, [gr.State(NEXUS_PROMPT), t1, a1, f1], out1)

        with gr.Tab("🧠 Zen"):
            with gr.Row():
                with gr.Column(elem_classes="glass-panel", scale=1):
                    gr.Markdown("#### 🔥 WARRIOR MODE\nNo questions. Only power. Based on your profile.")
                    a2 = gr.Audio(label="VOICE LOG", sources=["microphone"], type="filepath")
                    btn2 = gr.Button("ACTIVATE ZEN POWER", variant="primary")
                with gr.Column(elem_classes="glass-panel", scale=1.5):
                    out2 = gr.Markdown("### 🔥 High-Intensity Vision Protocol")
            btn2.click(stream_llm, [gr.State(ZEN_PROMPT), t1, a2, f1], out2)

        with gr.Tab("📅 Orbit"):
            with gr.Row():
                with gr.Column(elem_classes="glass-panel", scale=1):
                    gr.Markdown("#### 🛰️ MISSION ROADMAP")
                    a3 = gr.Audio(label="VOICE COMMAND", sources=["microphone"], type="filepath")
                    btn3 = gr.Button("LAUNCH MASTER PLAN", variant="primary")
                with gr.Column(elem_classes="glass-panel", scale=1.5):
                    out3 = gr.Markdown("### 🛣️ Strategic Execution Roadmap")
            btn3.click(stream_llm, [gr.State(ORBIT_PROMPT), t1, a3, f1], out3)

        with gr.Tab("🎯 Pathfinder"):
            with gr.Row():
                with gr.Column(elem_classes="glass-panel", scale=1):
                    gr.Markdown("#### 🔭 MARKET MATCHING\nFinding companies where you fit perfectly.")
                    a_p = gr.Audio(label="VOICE COMMAND", sources=["microphone"], type="filepath")
                    btn_p = gr.Button("FIND MY INDUSTRY MATCH", variant="primary")
                with gr.Column(elem_classes="glass-panel", scale=1.5):
                    out_p = gr.Markdown("### 🏢 Skill-Based Industry & Company Match")
            btn_p.click(stream_llm, [gr.State(PATH_PROMPT), t1, a_p, f1], out_p)

if __name__ == "__main__":
    demo.launch()