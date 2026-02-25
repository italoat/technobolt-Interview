import streamlit as st
import google.generativeai as genai
import fitz
from streamlit_mic_recorder import mic_recorder
import io

st.set_page_config(page_title="AI Interview Assistant", layout="wide")

# Configurações na barra lateral
st.sidebar.header("Configurações")
api_key = st.sidebar.text_input("Google API Key", type="password")
selected_model = st.sidebar.selectbox("Modelo", ["gemini-2.0-flash", "gemini-1.5-flash"])
uploaded_cv = st.sidebar.file_uploader("Anexe seu Currículo (PDF)", type="pdf")

if "cv_text" not in st.session_state:
    st.session_state.cv_text = None

if uploaded_cv and not st.session_state.cv_text:
    doc = fitz.open(stream=uploaded_cv.read(), filetype="pdf")
    st.session_state.cv_text = "".join([page.get_text() for page in doc])
    st.sidebar.success("Currículo carregado!")

st.title("🎙️ Assistente de Entrevista (Web Version)")

# Componente de gravação de áudio do navegador
st.write("Clique no botão para gravar a pergunta do entrevistador:")
audio_record = mic_recorder(
    start_prompt="🔴 Iniciar Gravação",
    stop_prompt="⏹️ Parar e Analisar",
    key='recorder'
)

if audio_record:
    if not api_key or not st.session_state.cv_text:
        st.error("Certifique-se de que a API Key e o Currículo foram inseridos.")
    else:
        st.audio(audio_record['bytes'])
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(selected_model)
            
            # Preparando o áudio para o Gemini
            audio_data = {
                "mime_type": "audio/wav",
                "data": audio_record['bytes']
            }
            
            prompt = f"""
            Você é um candidato sendo entrevistado. Responda à pergunta contida no áudio 
            baseando-se estritamente nas experiências deste currículo:
            
            {st.session_state.cv_text}
            
            REGRAS:
            1. Responda em primeira pessoa.
            2. Seja conciso (máximo 3 frases).
            3. Não mencione "baseado no currículo", apenas responda naturalmente.
            """
            
            with st.spinner("Analisando áudio e gerando resposta..."):
                response = model.generate_content([prompt, audio_data])
                
                st.subheader("Sugestão de Resposta:")
                st.info(response.text)
                
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
