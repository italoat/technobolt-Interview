import streamlit as st
import google.generativeai as genai
import fitz
from streamlit_mic_recorder import mic_recorder
import time

# --- CONFIGURAÇÃO DE UI ---
st.set_page_config(page_title="Interview Co-Pilot | Streaming Edition", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #e6edf3; }
    .streaming-card {
        background: #161b22;
        border-radius: 12px;
        padding: 25px;
        border-left: 6px solid #58a6ff;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        font-size: 1.15rem;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .status-badge { color: #8b949e; font-size: 0.8rem; font-family: monospace; display: block; margin-bottom: 15px; }
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

def generate_streaming_response(api_key, cv_context, audio_bytes):
    genai.configure(api_key=api_key)
    
    # REMOVIDO o max_output_tokens para evitar cortes abruptos.
    gen_config = {
        "temperature": 0.2, # Reduzido para evitar alucinações de currículo
        "top_p": 0.8
    }
    
    # PROMPT DE IDENTIDADE DINÂMICA (Ajustado para Perfil Completo)
    system_instruction = f"""
    PERSONA DINÂMICA: Assuma a identidade, o nome, a senioridade, as formações acadêmicas, as certificações e o histórico profissional exato contido no documento abaixo. Você é o candidato sendo entrevistado.
    
    CONTEXTO DO CURRÍCULO (Exportação do LinkedIn):
    {cv_context}

    COMO LER ESTE DOCUMENTO:
    1. O documento é linear. Identifique as seções principais: "Resumo", "Experiência", "Certifications" e "Formação acadêmica".
    2. Na seção "Experiência", o nome da empresa geralmente aparece primeiro, seguido pelo tempo total, seu cargo, o período e suas atribuições.
    3. Incorpore ativamente suas certificações (ex: OCI, AWS, ITIL, Scrum) e formações acadêmicas quando for pertinente para validar sua senioridade.

    DIRETRIZES DE RESPOSTA (ESTRITO):
    1. IDENTIDADE: Você é o entrevistado. Fale sempre na primeira pessoa do singular ("Eu desenvolvi", "Minha formação em..."). Nunca diga que é uma IA ou um assistente.
    2. PROVA DE EXPERIÊNCIA: OBRIGATORIAMENTE cite a empresa, projeto, certificado ou formação do currículo onde você adquiriu a experiência que está narrando.
    3. FLUÍDEZ: Seja conversacional e direto. Responda à pergunta do áudio de forma pragmática, como em uma entrevista real.
    4. TAMANHO: O raciocínio deve ser conciso (cerca de 3 a 5 frases completas), mas NUNCA deixe a frase pela metade. Conclua seu pensamento de forma coesa.
    """

    for model_name in MOTORES:
        try:
            model = genai.GenerativeModel(model_name=model_name, generation_config=gen_config)
            
            # ATIVADO STREAM=TRUE para fluidez de tempo real
            response = model.generate_content(
                [system_instruction, {"mime_type": "audio/wav", "data": audio_bytes}], 
                stream=True
            )
            return response, model_name
        except Exception as e:
            if "429" in str(e): 
                continue
            return None, f"Erro: {str(e)}"
    
    return None, "Limite Global Excedido"

# --- INTERFACE E LÓGICA ---
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Setup do Candidato")
    api_key = st.text_input("Gemini API Key", type="password")
    cv_file = st.file_uploader("Upload do Currículo (PDF)", type="pdf")
    
    if cv_file and 'cv_text' not in st.session_state:
        st.session_state.cv_text = extract_cv_content(cv_file)
        st.success("Identidade Extraída com Sucesso!")
    
    st.divider()
    st.info("💡 **Dica de Fluxo:** Clique em iniciar quando o entrevistador começar a falar. Clique em parar assim que ele terminar a pergunta para a resposta fluir imediatamente.")

st.title("🎙️ Interview Co-Pilot")

col_main, col_hist = st.columns([2, 1])

with col_main:
    st.subheader("Captura de Áudio")
    audio_data = mic_recorder(
        start_prompt="🔴 Iniciar Escuta",
        stop_prompt="⏹️ Finalizar Pergunta e Responder",
        key='interview_recorder'
    )

    if audio_data:
        if not api_key or 'cv_text' not in st.session_state:
            st.error("Chave API ou Currículo ausentes.")
        else:
            st.markdown("### 🧠 Resposta em Tempo Real")
            
            # Cria um container vazio que será preenchido aos poucos (Efeito Máquina de Escrever)
            response_container = st.empty()
            
            try:
                # Chama a função que agora retorna um iterador de stream
                stream_response, motor = generate_streaming_response(api_key, st.session_state.cv_text, audio_data['bytes'])
                
                if stream_response:
                    full_text = ""
                    
                    # Layout customizado para a resposta em streaming
                    with response_container.container():
                        st.markdown(f"<span class='status-badge'>MOTOR EM USO: {motor} | PROCESSANDO...</span>", unsafe_allow_html=True)
                        
                        # Função geradora para o st.write_stream do Streamlit
                        def stream_parser():
                            for chunk in stream_response:
                                try:
                                    if chunk.text:
                                        yield chunk.text
                                except ValueError:
                                    pass
                        
                        # Escreve o texto fluentemente na tela
                        st.markdown("<div class='streaming-card'>", unsafe_allow_html=True)
                        full_text = st.write_stream(stream_parser)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Salva a resposta completa no histórico
                    st.session_state.history.append({"text": full_text, "time": time.strftime("%H:%M:%S")})
                else:
                    st.error(f"Falha na API: {motor}")
                    
            except Exception as e:
                st.error(f"Erro no processamento: {e}")

with col_hist:
    st.subheader("📚 Histórico Rápido")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.expander(f"Turno - {item['time']}", expanded=False):
                st.write(item['text'])
    else:
        st.write("Aguardando interações...")
