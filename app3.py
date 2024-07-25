import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
import os
import pickle
import hashlib


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def precompute_and_save_embeddings(text_chunks, embeddings_model, file_path):
    text_embeddings = list(
        zip(text_chunks, embeddings_model.embed_documents(text_chunks))
    )
    with open(file_path, "wb") as f:
        pickle.dump(text_embeddings, f)


def load_precomputed_embeddings(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_vectorstore(vectorstore, file_path):
    vectorstore.save_local(file_path)


def load_vectorstore(file_path, embeddings):
    return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)


def save_embeddings_and_chunks(pdf_metadata, folder_path="data"):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
        pickle.dump(pdf_metadata, f)


def load_embeddings_and_chunks(folder_path="data"):
    if os.path.exists(os.path.join(folder_path, "metadata.pkl")):
        with open(os.path.join(folder_path, "metadata.pkl"), "rb") as f:
            pdf_metadata = pickle.load(f)
        return pdf_metadata
    return {}


def hash_file(file):
    hasher = hashlib.md5()
    buf = file.read(8192)
    while len(buf) > 0:
        hasher.update(buf)
        buf = file.read(8192)
    file.seek(0)  # Reset file pointer to the beginning
    return hasher.hexdigest()


def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama2-uncensored")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.write("Please process the documents first.")
        return

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    pdf_metadata = load_embeddings_and_chunks()
    combined_text_chunks = []
    combined_text_embeddings = []
    embeddings_model = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                new_metadata = {}
                for pdf in pdf_docs:
                    pdf_hash = hash_file(pdf)
                    if pdf_hash not in pdf_metadata:
                        # get pdf text
                        raw_text = get_pdf_text([pdf])

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)
                        combined_text_chunks.extend(text_chunks)

                        # precompute embeddings and save them
                        embeddings_file_path = f"data/{pdf_hash}_embeddings.pkl"
                        if not os.path.exists(embeddings_file_path):
                            device = "cpu"
                            embeddings_model = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2",
                                model_kwargs={"device": device},
                                encode_kwargs={"normalize_embeddings": True},
                            )
                            precompute_and_save_embeddings(
                                text_chunks, embeddings_model, embeddings_file_path
                            )

                        # save metadata
                        new_metadata[pdf_hash] = {
                            "filename": pdf.name,
                            "chunks": text_chunks,
                            "embeddings_path": embeddings_file_path,
                        }
                        pdf_metadata[pdf_hash] = new_metadata[pdf_hash]

                # save embeddings and chunks
                save_embeddings_and_chunks(pdf_metadata)

                # load all precomputed embeddings
                for metadata in pdf_metadata.values():
                    text_embeddings = load_precomputed_embeddings(
                        metadata["embeddings_path"]
                    )
                    combined_text_embeddings.extend(text_embeddings)

                # create combined vector store
                combined_vectorstore = FAISS.from_embeddings(
                    text_embeddings=combined_text_embeddings, embedding=embeddings_model
                )
                save_vectorstore(combined_vectorstore, "data/combined_vectorstore")

                st.session_state.conversation = get_conversation_chain(
                    combined_vectorstore
                )

        elif os.path.exists("data/combined_vectorstore"):
            device = "cpu"
            embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            combined_vectorstore = load_vectorstore(
                "data/combined_vectorstore", embeddings_model
            )
            st.session_state.conversation = get_conversation_chain(combined_vectorstore)

        if pdf_metadata:
            st.subheader("PDFs in the database")
            for data in pdf_metadata.values():
                st.write(data["filename"])

    # Container at the bottom for user input
    # TODO Remove this
    container = st.container()
    with container:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)


if __name__ == "__main__":
    main()
