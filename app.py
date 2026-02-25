import streamlit as st
import google.generativeai as genai
import fitz
from streamlit_mic_recorder import mic_recorder
import time

# --- CONFIGURAÇÃO DE UI SENIOR ---
st.set_page_config(page_title="TechnoBolt Co-Pilot v3", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .answer-card {
        background: #161b22;
        border-radius: 12px;
        padding: 25px;
        border-left: 6px solid #238636;
        color: #e6edf3;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    .experience-tag {
        color: #58a6ff;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.85rem;
        display: block;
        margin-bottom: 8px;
    }
    .history-item {
        background: #0d1117;
        padding: 10px;
        border-bottom: 1px solid #30363d;
        font-size: 0.9rem;
        color: #8b949e;
    }
    </style>
    """, unsafe_allow_html=True)

MOTORES = [
    "models/gemini-3-flash-preview",
    "models/gemini-2.5-flash", 
    "models/gemini-2.0-flash", 
    "models/gemini-flash-latest"
]

def extract_cv_content(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def generate_expert_response(api_key, cv_context, audio_bytes):
    genai.configure(api_key=api_key)
    
    # Configuração de Resposta Cirúrgica
    gen_config = {
        "temperature": 0.3,
        "top_p": 0.8,
        "max_output_tokens": 180
    }
    
    # PROMPT DE REFORÇO DE CONTEXTO
    system_instruction = f"""
    PERSONA: Você é o Italo, Arquiteto de Dados/DBA Sênior. 
    CURRÍCULO BASE: {cv_context}

    REGRAS OBRIGATÓRIAS:
    1. CITAÇÃO DE EXPERIÊNCIA: Para cada afirmação técnica, você DEVE citar em qual empresa ou projeto do currículo você aplicou aquilo (Ex: 'Na TechnoBolt...', 'Durante meu tempo na CI&T...').
    2. ESTILO: Responda em 1ª pessoa. Seja assertivo, sênior e direto.
    3. TAMANHO: No máximo 4 frases.
    4. FOCO: Governança, Escalabilidade e Performance.
    """

    for model_name in MOTORES:
        try:
            model = genai.GenerativeModel(model_name=model_name, generation_config=gen_config)
            response = model.generate_content([
                system_instruction, 
                {"mime_type": "audio/wav", "data": audio_bytes}
            ])
            return response.text, model_name
        except Exception as e:
            if "429" in str(e): continue
            return f"Erro no motor {model_name}: {e}", "Erro"
    
    return "Falha crítica: Sem cota em todos os motores.", "Nenhum"

# --- INTERFACE ---

st.title("🛡️ TechnoBolt Interview Co-Pilot")
st.caption("Especialista Senior em Arquitetura de Dados & DBA")

with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Gemini API Key", type="password")
    cv_file = st.file_uploader("Currículo (PDF)", type="pdf")
    
    if cv_file and 'cv_text' not in st.session_state:
        st.session_state.cv_text = extract_cv_content(cv_file)
        st.success("Perfil Indexado!")
    
    st.divider()
    st.markdown("### 🎧 Dica de Áudio")
    st.warning("Para ouvir o som do sistema, use o **VB-Audio Cable** e selecione 'Cable Output' quando o navegador pedir permissão de microfone.")

# --- CORE ---
if 'history' not in st.session_state:
    st.session_state.history = []

col_main, col_hist = st.columns([2, 1])

with col_main:
    st.subheader("🎙️ Escuta Ativa")
    # O navegador pedirá permissão automaticamente ao clicar
    audio_data = mic_recorder(
        start_prompt="🔴 Iniciar Escuta",
        stop_prompt="⏹️ Analisar Pergunta",
        key='interview_recorder'
    )

    if audio_data:
        if not api_key or 'cv_text' not in st.session_state:
            st.error("Chave API ou Currículo ausentes.")
        else:
            with st.spinner("IA Processando (Cotejando experiências)..."):
                start_t = time.time()
                resposta, motor = generate_expert_response(api_key, st.session_state.cv_text, audio_data['bytes'])
                end_t = time.time()
                
                st.markdown(f"""
                <div class="answer-card">
                    <span class="experience-tag">Estratégia Recomendada</span>
                    {resposta}
                    <hr style="border:0; border-top:1px solid #30363d; margin:15px 0;">
                    <small style="color:#8b949e">Motor: {motor} | Latência: {end_t - start_t:.1f}s</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Salva no histórico (Pergunta e Resposta)
                st.session_state.history.append(resposta)

with col_hist:
    st.subheader("📚 Histórico")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Resposta {len(st.session_state.history)-i}", expanded=(i==0)):
                st.write(h)
    else:
        st.write("Aguardando interações...")
