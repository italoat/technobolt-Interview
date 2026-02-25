import streamlit as st
import google.generativeai as genai
import fitz
from streamlit_mic_recorder import mic_recorder
import time

# --- CONFIGURAÇÃO DE UI ---
st.set_page_config(page_title="TechnoBolt Co-Pilot v4", page_icon="🛡️", layout="wide")

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
    .meta-tag {
        color: #8b949e;
        font-size: 0.75rem;
        font-family: monospace;
        display: block;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

MOTORES = [
    "models/gemini-2.5-flash", 
    "models/gemini-3-flash-preview",
    "models/gemini-2.0-flash", 
    "models/gemini-flash-latest"
]

def extract_cv_content(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def generate_expert_response(api_key, cv_context, audio_bytes):
    genai.configure(api_key=api_key)
    
    # AJUSTE: Aumentamos para 500 para evitar que a resposta seja cortada
    gen_config = {
        "temperature": 0.3,
        "top_p": 0.8,
        "max_output_tokens": 500 
    }
    
    system_instruction = f"""
    PERSONA: Você é o Italo, Arquiteto de Dados e DBA Sênior. 
    CONTEÚDO DO CURRÍCULO: {cv_context}

    REGRAS DE OURO:
    1. CITE A EXPERIÊNCIA: Você DEVE mencionar obrigatoriamente em qual empresa ou projeto do currículo você viveu o que está dizendo.
       Exemplo: "Na TechnoBolt, implementamos..." ou "Durante minha atuação na CI&T, resolvi isso..."
    2. ESTILO: Responda em 1ª pessoa. Seja assertivo e mostre autoridade técnica.
    3. TAMANHO: Mantenha a resposta entre 3 a 5 frases. Não se alongue demais, mas não corte o raciocínio.
    4. FOCO: Governança, Performance de Dados e Escalabilidade.
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
    
    return "Falha crítica: Sem cota disponível no momento.", "Nenhum"

# --- INTERFACE ---
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Gemini API Key", type="password")
    cv_file = st.file_uploader("Currículo (PDF)", type="pdf")
    if cv_file and 'cv_text' not in st.session_state:
        st.session_state.cv_text = extract_cv_content(cv_file)
        st.success("Perfil Indexado!")
    
    st.divider()
    st.markdown("### 🛠️ Ajuste de Áudio")
    st.info("Para que o sistema 'ouça' o som do computador:")
    st.write("1. No Windows, mude a saída de som para **CABLE Input (VB-Cable)**.")
    st.write("2. No Navegador, quando clicar em gravar, selecione **CABLE Output** como seu microfone.")

st.title("🛡️ TechnoBolt Interview Co-Pilot")

col_main, col_hist = st.columns([2, 1])

with col_main:
    st.subheader("🎙️ Escuta Ativa")
    audio_data = mic_recorder(
        start_prompt="🔴 Iniciar Escuta do Entrevistador",
        stop_prompt="⏹️ Gerar Resposta",
        key='interview_recorder'
    )

    if audio_data:
        if not api_key or 'cv_text' not in st.session_state:
            st.error("Chave API ou Currículo ausentes.")
        else:
            with st.spinner("Analisando áudio e cruzando com experiências..."):
                t_start = time.time()
                resposta, motor = generate_expert_response(api_key, st.session_state.cv_text, audio_data['bytes'])
                t_end = time.time()
                
                st.markdown(f"""
                <div class="answer-card">
                    <span class="meta-tag">MOTOR: {motor} | LATÊNCIA: {t_end - t_start:.1f}s</span>
                    {resposta}
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.history.append({"text": resposta, "time": time.strftime("%H:%M:%S")})

with col_hist:
    st.subheader("📚 Histórico")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.expander(f"Turno - {item['time']}", expanded=False):
                st.write(item['text'])
    else:
        st.write("Nenhuma interação registrada.")
