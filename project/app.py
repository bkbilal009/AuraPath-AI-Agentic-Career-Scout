import gradio as gr
import os
from openai import OpenAI

# ==========================================
# 1. CORE ENGINE (WITH AUTO-FALLBACK & LANGUAGE LOCK)
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

    # LIST OF MODELS TO PREVENT 429 ERRORS[cite: 1, 3]
    models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    
    # UNIVERSAL LANGUAGE MIRRORING GUARD[cite: 1]
    detail_cmd = """
    \n\nCRITICAL SYSTEM RULES:
    1. STRICT LANGUAGE MATCH: Identify the input language (English, Urdu, or Roman-Urdu) and respond ONLY in that exact language. NO SWITCHING.
    2. HIGH-QUALITY VOICE: Professional, human, and empathetic tone.
    3. EXTREME DETAIL: Structure with bold headings (###).
    4. CLICKABLE RESOURCES: Include 5-10 specific clickable links.
    """

    client = get_client()
    success = False
    
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_role + detail_cmd},
                    {"role": "user", "content": combined_input}
                ],
                stream=True,
                temperature=0.75 
            )
            
            partial_text = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    partial_text += chunk.choices[0].delta.content
                    yield partial_text
            success = True
            break 
        except Exception as e:
            if "429" in str(e): # Switch to next model if rate limit reached[cite: 3]
                continue
            else:
                yield f"### ⚠️ Connection Error\n{str(e)}"
                return

    if not success:
        yield "### ⚠️ Servers Overloaded\nPlease try again in a few minutes."

# ==========================================
# 2. STRATEGIC SYSTEM PROMPTS
# ==========================================

NEXUS_PROMPT = """You are an Elite Technical CV Auditor. Analyze the CV with extreme depth:
1. CORE STRENGTHS. 2. CRITICAL WEAKNESSES. 3. TECH SKILL MATRIX. 4. NETWORKING STRATEGY. 5. IMPROVEMENT ROADMAP."""

ZEN_PROMPT = """You are a Master Motivational Strategist. 
STRICT LIMITATION: Your ONLY role is to encourage and motivate. Do not perform any other task.
CORE MESSAGE: Remind the user that failure is natural. 
MANDATORY EXAMPLES: Always include real-life examples of leaders like Steve Jobs, Elon Musk, and Jack Ma. Respond ONLY in user's input language."""

ORBIT_PROMPT = """You are a Lead Hiring Strategist. Create a 6-Month Roadmap (Month 1-2: Skill Gaps, Month 3-4: Projects, Month 5: Branding, Month 6: Applications). Respond ONLY in user's input language."""

# PATHFINDER: STRICT LANGUAGE REINFORCEMENT[cite: 1, 3]
PATH_PROMPT = """You are a Global Market Scout. 
STRICT RULE: Detect the user's language and respond EXCLUSIVELY in that language. If the user asks in English, respond in English. If they use Roman-Urdu, respond in Roman-Urdu. 
TASK: Identify Industry and Companies where the user fits perfectly based on their skills."""

# ==========================================
# 3. UI DESIGN (REMAINING STABLE)
# ==========================================
custom_css = """
footer {display: none !important;}
body { margin: 0; padding: 0; }
.gradio-container { 
    background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4b 25%, #4b0082 50%, #800080 75%, #ff007f 100%) !important; 
    background-attachment: fixed !important;
    background-size: cover !important;
    min-height: 100vh !important;
}
.glass-panel { background: rgba(10, 10, 30, 0.88) !important; backdrop-filter: blur(25px); border: 1px solid rgba(0, 242, 254, 0.3); border-radius: 20px; padding: 25px; height: fit-content; }
.shimmer-title {
    font-size: 4rem; font-weight: 900; color: #ffffff; text-transform: uppercase;
    text-shadow: 0 0 10px #ff007f, 0 0 20px #ff007f, 0 0 40px #00f2fe;
    animation: pulse-glow 2s ease-in-out infinite alternate; letter-spacing: 5px; margin-bottom: 0px;
}
@keyframes pulse-glow {
    from { text-shadow: 0 0 10px #ff007f, 0 0 20px #ff007f, 0 0 40px #00f2fe; transform: scale(1); }
    to { text-shadow: 0 0 20px #00f2fe, 0 0 40px #ff007f, 0 0 60px #00f2fe; transform: scale(1.02); }
}
.new-tagline { font-size: 1.1rem; color: #00f2fe; font-weight: bold; letter-spacing: 8px; text-transform: uppercase; margin-top: -10px; opacity: 0.9; }
.gr-button-primary { background: linear-gradient(90deg, #00f2fe 0%, #ff007f 100%) !important; border: none !important; font-weight: bold !important; }
input, textarea { background: rgba(0, 0, 0, 0.7) !important; color: white !important; border: 1px solid #00f2fe !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 30px 10px;">
            <h1 class="shimmer-title">AURAPATH AI</h1>
            <p class="new-tagline">◈ ENGINEERING YOUR DESTINY ◈</p>
        </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📄 Nexus"):
            with gr.Row():
                with gr.Column(elem_classes="glass-panel", scale=1):
                    f1 = gr.File(label="Upload CV / Portfolio")
                    t1 = gr.Textbox(label="TARGET GOAL", placeholder="e.g. AI Developer...", lines=4)
                    a1 = gr.Audio(label="VOICE SCAN", sources=["microphone"], type="filepath")
                    btn1 = gr.Button("INITIATE DEEP AUDIT", variant="primary")
                with gr.Column(elem_classes="glass-panel", scale=1.5):
                    out1 = gr.Markdown("### 🔍 Technical Audit & Skill Matrix")
            btn1.click(stream_llm, [gr.State(NEXUS_PROMPT), t1, a1, f1], out1)

        with gr.Tab("🧠 Zen"):
            with gr.Row():
                with gr.Column(elem_classes="glass-panel", scale=1):
                    gr.Markdown("#### 🔥 WARRIOR MODE\nPure motivation in your own language.")
                    a2 = gr.Audio(label="VOICE LOG", sources=["microphone"], type="filepath")
                    btn2 = gr.Button("ACTIVATE ZEN POWER", variant="primary")
                with gr.Column(elem_classes="glass-panel", scale=1.5):
                    out2 = gr.Markdown("### 🔥 High-Intensity Vision Protocol")
            btn2.click(stream_llm, [gr.State(ZEN_PROMPT), t1, a2, f1], out2)

        with gr.Tab("📅 Orbit"):
            with gr.Row():
                with gr.Column(elem_classes="glass-panel", scale=1):
                    gr.Markdown("#### 🛰️ MISSION ROADMAP\nRoadmap strictly matching your input language.")
                    a3 = gr.Audio(label="VOICE COMMAND", sources=["microphone"], type="filepath")
                    btn3 = gr.Button("LAUNCH MASTER PLAN", variant="primary")
                with gr.Column(elem_classes="glass-panel", scale=1.5):
                    out3 = gr.Markdown("### 🛣️ 6-Month Strategic Execution Roadmap")
            btn3.click(stream_llm, [gr.State(ORBIT_PROMPT), t1, a3, f1], out3)

        with gr.Tab("🎯 Pathfinder"):
            with gr.Row():
                with gr.Column(elem_classes="glass-panel", scale=1):
                    gr.Markdown("#### 🔭 MARKET MATCHING\nIndustry match in your exact language.")
                    a_p = gr.Audio(label="VOICE COMMAND", sources=["microphone"], type="filepath")
                    btn_p = gr.Button("FIND MY INDUSTRY MATCH", variant="primary")
                with gr.Column(elem_classes="glass-panel", scale=1.5):
                    out_p = gr.Markdown("### 🏢 Skill-Based Industry & Company Match")
            btn_p.click(stream_llm, [gr.State(PATH_PROMPT), t1, a_p, f1], out_p)

if __name__ == "__main__":
    demo.launch()