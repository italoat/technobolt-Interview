import streamlit as st
import google.generativeai as genai
import fitz
import time
import speech_recognition as sr

# --- CONFIGURAÇÃO DE UI ---
st.set_page_config(page_title="Interview Co-Pilot | Autonomous Edition", page_icon="⚡", layout="wide")

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
    .listening-pulse { color: #3fb950; font-weight: bold; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
    """, unsafe_allow_html=True)

MOTORES = [
    "models/gemini-flash-latest",
    "models/gemini-2.0-flash", 
    "models/gemini-2.5-flash", 
    "models/gemini-3-flash-preview"

]

# Palavras-gatilho que você vai usar para avisar a IA que a pergunta acabou
GATILHOS = ["certo", "ok", "vamos lá", "entendi", "perfeito", "pode mandar"]

def extract_cv_content(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def generate_streaming_response(api_key, cv_context, question_text):
    genai.configure(api_key=api_key)
    
    gen_config = {
        "temperature": 0.2, 
        "top_p": 0.8
    }
    
    system_instruction = f"""
    PERSONA DINÂMICA: Assuma a identidade, o nome, a senioridade, as formações acadêmicas, as certificações e o histórico profissional exato contido no documento abaixo. Você é o candidato sendo entrevistado.
    
    CONTEXTO DO CURRÍCULO:
    {cv_context}

    DIRETRIZES DE RESPOSTA (ESTRITO):
    1. IDENTIDADE: Você é o entrevistado. Fale sempre na primeira pessoa do singular ("Eu desenvolvi", "Minha formação em..."). Nunca diga que é uma IA.
    2. PROVA DE EXPERIÊNCIA: OBRIGATORIAMENTE cite a empresa, projeto, certificado ou stack (Python, AWS, Oracle, NiFi) do currículo onde adquiriu a experiência.
    3. FLUÍDEZ: Seja conversacional, direto e pragmático. 
    4. TAMANHO: Raciocínio conciso (3 a 5 frases completas), indo direto ao ponto da pergunta.
    
    PERGUNTA DO ENTREVISTADOR (Transcrita):
    "{question_text}"
    """

    for model_name in MOTORES:
        try:
            model = genai.GenerativeModel(model_name=model_name, generation_config=gen_config)
            # Como a transcrição já foi feita localmente, passamos apenas o texto (muito mais rápido)
            response = model.generate_content(system_instruction, stream=True)
            return response, model_name
        except Exception as e:
            if "429" in str(e): 
                continue
            return None, f"Erro: {str(e)}"
    
    return None, "Limite Global Excedido"

# --- INTERFACE E LÓGICA DE ESTADO ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False

with st.sidebar:
    st.header("⚙️ Setup do Candidato")
    api_key = st.text_input("Gemini API Key", type="password")
    cv_file = st.file_uploader("Upload do Currículo (PDF)", type="pdf")
    
    if cv_file and 'cv_text' not in st.session_state:
        st.session_state.cv_text = extract_cv_content(cv_file)
        st.success("Identidade Extraída com Sucesso!")
    
    st.divider()
    st.info("💡 **Como funciona agora:**\nA máquina escutará em loop. Tudo que o recrutador disser será acumulado num buffer. Quando você disser apenas **'Certo'**, **'Ok'** ou **'Vamos lá'**, ela pega o buffer, gera a resposta e limpa a memória para a próxima pergunta.")

st.title("🎙️ Interview Co-Pilot (Autonomous)")

col_main, col_hist = st.columns([2, 1])

with col_main:
    # Botão de controle de estado (Start/Stop)
    if not st.session_state.is_listening:
        if st.button("▶️ Iniciar Entrevista", use_container_width=True, type="primary"):
            st.session_state.is_listening = True
            st.rerun()
    else:
        if st.button("⏹️ Pausar Entrevista", use_container_width=True):
            st.session_state.is_listening = False
            st.rerun()

    status_ui = st.empty()
    transcript_ui = st.empty()
    response_ui = st.empty()

    # O LOOP DE ESCUTA AUTÔNOMA
    if st.session_state.is_listening:
        if not api_key or 'cv_text' not in st.session_state:
            st.error("⚠️ Chave API ou Currículo ausentes.")
            st.session_state.is_listening = False
            st.rerun()

        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        # Ajusta o ruído ambiente inicial (crucial para o VAD funcionar bem)
        with microphone as source:
            status_ui.markdown("<span class='listening-pulse'>🔄 Calibrando ruído ambiente... aguarde 2s.</span>", unsafe_allow_html=True)
            recognizer.adjust_for_ambient_noise(source, duration=2)
        
        # Buffer para acumular as partes da pergunta do recrutador
        question_buffer = []

        while st.session_state.is_listening:
            status_ui.markdown("<span class='listening-pulse'>🎙️ Escutando ativamente... (Diga 'Ok' ou 'Certo' para responder)</span>", unsafe_allow_html=True)
            
            try:
                with microphone as source:
                    # Escuta até detectar um período de silêncio
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
                
                status_ui.markdown("<span class='status-badge'>Processando fala...</span>", unsafe_allow_html=True)
                # Usa a API gratuita do Google para transcrever rápido
                text = recognizer.recognize_google(audio, language="pt-BR").lower().strip()
                
                # Verifica se a frase dita é um dos nossos gatilhos
                is_trigger = any(trigger == text for trigger in GATILHOS)
                
                if is_trigger:
                    if not question_buffer:
                        transcript_ui.warning("Gatilho detectado, mas nenhuma pergunta foi ouvida antes.")
                        continue
                    
                    status_ui.markdown("<span class='status-badge'>Gatilho acionado! Gerando resposta...</span>", unsafe_allow_html=True)
                    full_question = " ".join(question_buffer)
                    transcript_ui.info(f"**Pergunta capturada:** {full_question}")
                    
                    # Chama o Gemini
                    response_container = response_ui.container()
                    stream_response, motor = generate_streaming_response(api_key, st.session_state.cv_text, full_question)
                    
                    if stream_response:
                        with response_container:
                            st.markdown(f"<span class='status-badge'>MOTOR EM USO: {motor}</span>", unsafe_allow_html=True)
                            
                            def stream_parser():
                                for chunk in stream_response:
                                    try:
                                        if chunk.text: yield chunk.text
                                    except ValueError: pass
                            
                            st.markdown("<div class='streaming-card'>", unsafe_allow_html=True)
                            full_text = st.write_stream(stream_parser)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Salva histórico e limpa buffer
                            st.session_state.history.append({"q": full_question, "a": full_text, "time": time.strftime("%H:%M:%S")})
                            question_buffer = [] # Zera para a próxima pergunta
                    else:
                        st.error("Falha ao contatar a API do Gemini.")
                
                else:
                    # Se não é gatilho, assume que é o recrutador falando e acumula no buffer
                    question_buffer.append(text)
                    transcript_ui.success(f"**Ouvido até agora:** {' '.join(question_buffer)}")

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                # Silêncio ou ruído irreconhecível, apenas ignora
                continue
            except Exception as e:
                status_ui.error(f"Erro no áudio: {e}")
                time.sleep(2)

with col_hist:
    st.subheader("📚 Histórico Rápido")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.expander(f"Turno - {item['time']}", expanded=False):
                st.markdown(f"**Q:** {item['q']}")
                st.markdown(f"**R:** {item['a']}")
    else:
        st.write("Aguardando interações...")
