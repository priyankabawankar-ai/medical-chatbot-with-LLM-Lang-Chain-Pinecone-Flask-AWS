import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *

app = Flask(__name__)

# Load env variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings
embeddings = download_embeddings()

# Connect to Pinecone index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# LLM
chatModel = ChatOpenAI(model="gpt-4o-mini")

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Store chat history per session
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Chains
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    try:
        user_msg = request.form.get("msg")

        if not user_msg:
            return {"answer": "Please enter a question."}, 400

        session_id = "user_1"

        response = conversational_rag_chain.invoke(
            {"input": user_msg},
            config={"configurable": {"session_id": session_id}}
        )

        if isinstance(response, dict):
            answer = response.get("answer") or response.get("output") or str(response)
        else:
            answer = str(response)

        return {"answer": answer}

    except Exception as e:
        print("Error:", str(e))
        return {"answer": str(e)}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)