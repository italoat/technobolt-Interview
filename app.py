import streamlit as st
import google.generativeai as genai
import fitz
from streamlit_mic_recorder import mic_recorder
import io

# --- CONFIGURAÇÃO DE UI MODERNA ---
st.set_page_config(
    page_title="TechnoBolt Co-Pilot | Interview AI",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS para um visual futurista e limpo
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMarkdown { font-family: 'Inter', sans-serif; }
    .answer-card {
        background-color: #161b22;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #30363d;
        color: #c9d1d9;
        margin-top: 20px;
    }
    .status-online {
        color: #238636;
        font-weight: bold;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGICA DE BACKEND ---

def extract_cv_data(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def generate_human_response(api_key, model_name, cv_context, audio_bytes):
    genai.configure(api_key=api_key)
    # Usando o 2.0 Flash para latência mínima e compreensão nativa de áudio
    model = genai.GenerativeModel(model_name)
    
    audio_part = {"mime_type": "audio/wav", "data": audio_bytes}
    
    # Prompt System: O "Cérebro" do entrevistado senior
    prompt = f"""
    VOCÊ É O CANDIDATO. Sua personalidade é de um profissional Sênior, confiante e direto.
    Use o seguinte contexto de currículo para basear suas experiências:
    {cv_context}

    INSTRUÇÕES DE RESPOSTA (ESTRITO):
    1. PERSONA: Responda em 1ª pessoa. Você não está "ajudando" o candidato, você É o candidato.
    2. TOM: Profissional, mas coloquial. Use termos como "Na prática...", "O grande desafio foi...", "Eu liderei...".
    3. ESTRUTURA: Máximo de 3 a 4 frases. Foque em RESULTADO e TECNOLOGIA (ex: SQL Server, Arquitetura de Dados, Automação).
    4. HUMANIDADE: Comece a resposta de forma natural, ex: "Boa pergunta. No meu tempo na [Empresa]...", ou "Sim, eu já lidei com isso da seguinte forma...".
    5. NÃO seja um robô. Se a pergunta for técnica, descreva a solução. Se for comportamental, mostre maturidade.
    
    Ouça o áudio anexo e responda agora.
    """
    
    response = model.generate_content([prompt, audio_part])
    return response.text

# --- INTERFACE ---

st.title("🎙️ TechnoBolt Interview Co-Pilot")
st.markdown("<p class='status-online'>● AI ENGINE READY</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Core Settings")
    key = st.text_input("Gemini API Key", type="password")
    model_choice = st.selectbox("Intelligence Level", 
                                ["gemini-2.0-flash", "gemini-3-flash-preview", "gemini-1.5-flash"])
    file = st.file_uploader("Upload Profile (PDF)", type="pdf")
    
    if file:
        if 'cv_data' not in st.session_state:
            with st.spinner("Indexing profile..."):
                st.session_state.cv_data = extract_cv_data(file)
            st.success("Profile Loaded.")

# --- ZONA DE INTERAÇÃO ---
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Listening")
    st.info("Clique e faça a pergunta como se fosse o recrutador.")
    audio = mic_recorder(
        start_prompt="Start Listening",
        stop_prompt="Stop & Process",
        key='recorder'
    )

with col2:
    st.subheader("Suggested Answer")
    if audio:
        if not key or 'cv_data' not in st.session_state:
            st.warning("Aguardando Configurações (Chave API ou CV).")
        else:
            with st.spinner("Analyzing intent..."):
                try:
                    answer = generate_human_response(key, model_choice, st.session_state.cv_data, audio['bytes'])
                    st.markdown(f"""
                    <div class="answer-card">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Log de histórico simples
                    if 'history' not in st.session_state: st.session_state.history = []
                    st.session_state.history.append(answer)
                except Exception as e:
                    st.error(f"Engine Error: {e}")
    else:
        st.write("Aguardando entrada de áudio...")

if 'history' in st.session_state and st.session_state.history:
    with st.expander("Timeline de Respostas"):
        for i, h in enumerate(reversed(st.session_state.history)):
            st.text(f"Turno {len(st.session_state.history)-i}: {h[:100]}...")
