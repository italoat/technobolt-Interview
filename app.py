import streamlit as st
import google.generativeai as genai
import fitz
from streamlit_mic_recorder import mic_recorder
import time

# --- CONFIGURAÇÃO VISUAL SENIOR ---
st.set_page_config(
    page_title="TechnoBolt Co-Pilot | High-Performance Edition",
    page_icon="⚡",
    layout="wide"
)

# Estilização customizada (Dark Mode Pro)
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stAlert { border-radius: 10px; }
    .answer-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-left: 5px solid #238636;
        padding: 25px;
        border-radius: 12px;
        color: #e6edf3;
        font-size: 1.15rem;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
    }
    .model-info {
        color: #8b949e;
        font-size: 0.8rem;
        margin-bottom: 15px;
        font-family: monospace;
    }
    .status-badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        background: #238636;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTORES CONFIGURADOS (LISTA DO USUÁRIO) ---
MOTORES = [
    "models/gemini-3-flash-preview",
    "models/gemini-2.5-flash", 
    "models/gemini-2.0-flash", 
    "models/gemini-flash-latest"
]

def extract_pdf_context(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def call_gemini_with_fallback(api_key, cv_content, audio_bytes):
    genai.configure(api_key=api_key)
    audio_data = {"mime_type": "audio/wav", "data": audio_bytes}
    
    # Prompt de Engenharia: Focado em Senioridade e Postura
    system_prompt = f"""
    PERSONA: Você é o candidato Italo, um Arquiteto de Dados e DBA Sênior de alto nível.
    CONTEXTO PROFISSIONAL: {cv_content}

    SUA MISSÃO:
    Responda à pergunta do entrevistador (áudio) de forma natural, humana e estratégica.
    
    DIRETRIZES DE RESPOSTA:
    1. Seja direto: Máximo 3 frases curtas e potentes.
    2. Linguagem: Use "eu", "minha equipe", "implementamos". Foque em arquitetura, governança e ROI.
    3. Humanidade: Use conectores naturais como "Na verdade...", "Um ponto crítico que resolvi foi...", "Exatamente, e inclusive...".
    4. Proibido: Não diga "conforme o currículo" ou "sou uma IA". Você É o profissional.
    """

    for model_name in MOTORES:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([system_prompt, audio_data])
            return response.text, model_name
        except Exception as e:
            if "429" in str(e):
                st.sidebar.warning(f"⚠️ {model_name} sem cota. Tentando fallback...")
                continue
            return f"Erro técnico no motor {model_name}: {str(e)}", "Falha"
    
    return "⚠️ Limite global de API atingido. Aguarde 60 segundos para nova tentativa.", "Nenhum"

# --- INTERFACE PRINCIPAL ---

st.title("🛡️ TechnoBolt Interview Co-Pilot")
st.markdown("<span class='status-badge'>MODO ESPECIALISTA ATIVO</span>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Gemini API Key", type="password", help="Pegue sua chave em aistudio.google.com")
    cv_file = st.file_uploader("Seu Currículo (PDF)", type="pdf")
    
    if cv_file and 'cv_text' not in st.session_state:
        st.session_state.cv_text = extract_pdf_context(cv_file)
        st.success("Perfil carregado com sucesso!")

st.divider()

col_mic, col_ans = st.columns([1, 1.8])

with col_mic:
    st.markdown("### 🎙️ Captura")
    st.info("O sistema ouvirá a pergunta do entrevistador e usará o motor Gemini para formular sua fala.")
    
    # Gravação via navegador
    recorded_audio = mic_recorder(
        start_prompt="🔴 Iniciar Escuta",
        stop_prompt="⏹️ Analisar Pergunta",
        key='interview_mic'
    )

with col_ans:
    st.markdown("### 🧠 Resposta do Candidato")
    if recorded_audio:
        if not api_key or 'cv_text' not in st.session_state:
            st.error("Configure a API Key e o CV antes de começar.")
        else:
            with st.spinner("IA processando áudio..."):
                start_time = time.time()
                resposta, motor_usado = call_gemini_with_fallback(
                    api_key, 
                    st.session_state.cv_text, 
                    recorded_audio['bytes']
                )
                end_time = time.time()
                
                st.markdown(f"""
                <div class="answer-card">
                    <div class="model-info">MOTOR: {motor_usado} | LATÊNCIA: {end_time - start_time:.2f}s</div>
                    {resposta}
                </div>
                """, unsafe_allow_html=True)
                
                # Adiciona ao histórico da sessão
                if 'history' not in st.session_state: st.session_state.history = []
                st.session_state.history.append(resposta)
    else:
        st.write("Aguardando áudio para processar...")

# Histórico discreto no rodapé
if 'history' in st.session_state and st.session_state.history:
    with st.expander("Timeline de Respostas"):
        for r in reversed(st.session_state.history):
            st.markdown(f"- {r}")
