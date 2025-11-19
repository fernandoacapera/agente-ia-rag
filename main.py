# !pip install streamlit langchain-groq langchain-core langchain-community langchain-text-splitters langchain-huggingface langchain-classic faiss-cpu python-dotenv PyMuPDF

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Atendimento Leilões", page_icon="🤖")
st.title("Chatbot - Grupo Carvalho")

id_model = "llama-3.1-8b-instant"
temperature = 0.7
path = None  # use busca automática por padrão (procura a pasta 'faqs' na raiz do projeto)

### Carregamento da LLM
def load_llm(id_model, temperature):
  llm = ChatGroq(
    model=id_model,
    temperature=temperature,
    max_tokens=None,
    timeout=None,
    max_retries=2,
  )
  return llm

llm = load_llm(id_model, temperature)

### Exibição do resultado
def show_res(res):
  from IPython.display import Markdown
  if "</think>" in res:
    res = res.split("</think>")[-1].strip()
  else:
    res = res.strip()  # fallback se não houver tag
    Markdown(res)

### Extração do conteúdo
def extract_text_pdf(file_path):
  loader = PyMuPDFLoader(file_path)
  doc = loader.load()
  content = "\n".join([page.page_content for page in doc])
  return content

### Indexação e recuperação
def _find_faqs_dir(start: Path| None):
  start = start.resolve()
  for candidate in [start] + list(start.parents):
    faqs = candidate / "faqs"
    if faqs.is_dir():
      return faqs
  return None


def config_retriever(folder_path: str | None = None):
  # Carregar documentos
  project_root = Path(__file__).parent

  # Resolver folder_path: aceita absoluto, relativo ao cwd ou relativo à raiz do projeto
  if folder_path:
    candidate = Path(folder_path)
    if not candidate.exists():
      # tentar relativo à raiz do projeto
      candidate = project_root / folder_path.lstrip("/\\")
  else:
    candidate = None

  if candidate and candidate.exists() and candidate.is_dir():
    docs_path = candidate
  else:
    # procura por uma pasta 'faqs' subindo a árvore a partir do projeto
    found = _find_faqs_dir(project_root)
    docs_path = found if found is not None else (project_root / "faqs")

  pdf_files = sorted(docs_path.glob("*.pdf"))

  if len(pdf_files) < 1:
    st.error(f"Nenhum arquivo PDF encontrado em: {docs_path}")
    st.stop()

  loaded_documents = [extract_text_pdf(pdf) for pdf in pdf_files]

  # Divisão em pedaços de texto / Split
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200
  )
  chunks = []
  for doc in loaded_documents:
      chunks.extend(text_splitter.split_text(doc))

  # Embeddings
  embedding_model = "BAAI/bge-m3" #sentence-transformers/all-mpnet-base-v2

  embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

  # Armazenamento
  if Path("index_faiss").exists():
    vectorstore = FAISS.load_local("index_faiss", embeddings, allow_dangerous_deserialization=True)
  else:
    vectorstore = FAISS.from_texts(chunks,  embedding=embeddings)
    vectorstore.save_local("index_faiss")

  # Configurando o recuperador de texto / Retriever
  retriever = vectorstore.as_retriever(
      search_type='mmr',
      search_kwargs={'k':3, 'fetch_k':4}
  )

  return retriever

### Chain da RAG
def config_rag_chain(llm, retriever):
  # Prompt para perguntas e respostas (Q&A)
  system_prompt = """Você é um assistente virtual prestativo e está respondendo perguntas gerais sobre os serviços de uma empresa.
  Use os seguintes pedaços de contexto recuperado para responder à pergunta.
  Se você não sabe a resposta, apenas comente que não sabe dizer com certeza.
  Mas caso seja uma dúvida muito comum, pode sugerir como alternativa uma solução possível.
  Mantenha a resposta concisa.
  Responda em português. \n\n"""

  qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "Pergunta: {input}\n\n Contexto: {context}"),
  ])

  # Configurar LLM e Chain para perguntas e respostas (Q&A)
  qa_chain = create_stuff_documents_chain(llm, qa_prompt)

  # Implementação manual da lógica RAG para evitar dependências de composição LCEL
  class SimpleRAGChain:
    def __init__(self, retriever, qa_chain):
      self.retriever = retriever
      self.qa_chain = qa_chain

    def invoke(self, inputs: dict):
      # Espera-se: inputs contain 'input' and optional 'chat_history'
      query = inputs.get("input")
      if query is None:
        raise ValueError("Missing 'input' for RAG chain")
      # Recuperar documentos: tentativa determinística
      import streamlit as _st
      docs = []
      last_exc = None

      if not self.retriever:
        _st.error("Retriever não configurado")
        return {"answer": "Erro interno: retriever não configurado."}

      # 1) tentar a API preferencial: retriever.invoke(query)
      try:
        docs = self.retriever.invoke(query)
      except Exception as e:
        last_exc = e
        docs = []

      # 2) se falhar, tentar operações diretas no vectorstore (se disponível)
      if not docs:
        try:
          vs = getattr(self.retriever, "vectorstore", None)
          search_kwargs = getattr(self.retriever, "search_kwargs", {}) or {}
          search_type = getattr(self.retriever, "search_type", None)
          if vs is not None:
            if search_type == "mmr" and hasattr(vs, "max_marginal_relevance_search"):
              docs = vs.max_marginal_relevance_search(query, **search_kwargs)
            elif hasattr(vs, "similarity_search"):
              docs = vs.similarity_search(query, **search_kwargs)
            else:
              # tentar ambos por segurança
              if hasattr(vs, "max_marginal_relevance_search"):
                docs = vs.max_marginal_relevance_search(query, **search_kwargs)
              elif hasattr(vs, "similarity_search"):
                docs = vs.similarity_search(query, **search_kwargs)
        except Exception as e2:
          last_exc = e2
          docs = []

      if not docs:
        _st.error(f"Falha ao recuperar documentos. Erro: {last_exc}")

      # Chamar o chain de QA com os documentos recuperados
      try:
        out = self.qa_chain.invoke({"context": docs, "input": query, "chat_history": inputs.get("chat_history", [])})
      except Exception as e:
        return {"answer": f"Erro ao gerar resposta: {e}"}

      # Normalizar saída para dicionário com chave 'answer' para compatibilidade
      return {"answer": out}

  return SimpleRAGChain(retriever, qa_chain)

### Interação com chat
def chat_llm(rag_chain, input):

  st.session_state.chat_history.append(HumanMessage(content=input))

  response = rag_chain.invoke({
      "input": input,
      "chat_history": st.session_state.chat_history
  })

  res = response["answer"]
  res = res.split("</think>")[-1].strip() if "</think>" in res else res.strip()

  st.session_state.chat_history.append(AIMessage(content=res))

  return res
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "retriever" not in st.session_state:
  st.session_state.retriever = None
  
  if not st.session_state.chat_started:
    st.markdown("Clique abaixo para iniciar o atendimento:")
    if st.button("Iniciar atendimento"):
        with st.spinner("Aguarde enquanto te transferimos para um atendente..."):
            st.session_state.retriever = config_retriever(path)
            st.session_state.chat_started = True
            st.rerun()
    st.stop()

input = st.chat_input("Escreva sua pergunta")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [
      AIMessage(content = "Olá, sou o assistente virtual do Grupo Carvalho! Como posso te ajudar?"),
  ]


for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)

if input is not None:
  with st.chat_message("Human"):
    st.markdown(input)

  with st.chat_message("AI"):
    # Garantir que o retriever exista antes de criar o RAG chain
    if st.session_state.retriever is None:
      try:
        st.session_state.retriever = config_retriever(path)
      except Exception as e:
        st.error(f"Não foi possível inicializar o retriever: {e}")
        st.stop()

    rag_chain = config_rag_chain(llm, st.session_state.retriever)
    res = chat_llm(rag_chain, input)
    st.write(res)