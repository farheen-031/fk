import os
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from difflib import get_close_matches
from fuzzywuzzy import fuzz
import chainlit as cl

# Environment setup for Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAKH5eqM8_D_zGKaay8fEbNpgbuhAwXkb4"

class PDFQnAPipeline:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.vector_store = None
        self.retrieval_qa_chain = None
        self.chat_model = None
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.current_entity = None  # Tracks the main entity in the conversation

    def load_and_chunk_pdf(self):
        print("Loading and chunking the PDF...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)

    def store_embeddings_in_chroma(self, docs):
        print("Storing embeddings in Chroma vector database...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.vector_store = Chroma.from_documents(docs, embeddings)

    def create_retriever_and_llm(self):
        print("Setting up retriever and LLM...")
        if self.vector_store is None:
            raise ValueError("Vector store has not been initialized yet.")
        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=0.1
        )
        self.retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            retriever=retriever,
            return_source_documents=True
        )

    def correct_query_typo(self, query, possible_questions):
        matches = get_close_matches(query, possible_questions, n=1, cutoff=0.7)
        return matches[0] if matches else query

    def extract_key_entity(self, query, answer):
        if "nurse call" in query.lower() or "nurse call" in answer.lower():
            return "nurse call system"
        return None

    def refine_follow_up_query(self, query):
        if self.current_entity and "it" in query.lower():
            return query.replace("it", f"the {self.current_entity}")
        return query

# Initialize the Chainlit chatbot interface for the PDF QnA
@cl.on_chat_start
async def on_chat_start():
    # Wait for the user to upload a PDF file
    pdf_file = None
    while pdf_file is None:
        pdf_file = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    pdf_path = pdf_file[0].path
    pipeline = PDFQnAPipeline(pdf_path)
    
    # Load, chunk, and store embeddings in Chroma DB
    split_docs = pipeline.load_and_chunk_pdf()
    pipeline.store_embeddings_in_chroma(split_docs)
    
    # Ensure that retrieval and LLM are initialized
    try:
        pipeline.create_retriever_and_llm()
    except Exception as e:
        await cl.Message(content=f"Error during pipeline setup: {e}").send()
        return
    
    # Inform the user that the system is ready
    await cl.Message(content=f"Processing `{pdf_file[0].name}` complete. You can now ask questions!").send()
    
    # Store the pipeline and chain for later use
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("retrieval_qa_chain", pipeline.retrieval_qa_chain)

@cl.on_message
async def on_message(message: cl.Message):
    pipeline = cl.user_session.get("pipeline")  # type: PDFQnAPipeline
    chain = cl.user_session.get("retrieval_qa_chain")  # type: RetrievalQA
    
    # Ensure the pipeline and vector_store are initialized
    if pipeline is None or pipeline.vector_store is None:
        await cl.Message(content="Error: The system is not ready. Please upload a PDF first and wait for processing to complete.").send()
        return

    query = message.content
    possible_questions = []
    try:
        # Retrieve possible questions based on the vector store
        possible_questions = [doc.metadata["source"][:100] for doc in pipeline.vector_store.similarity_search(query, k=10)]
    except Exception as e:
        await cl.Message(content=f"Error accessing documents for typo correction: {e}").send()
    
    corrected_query = pipeline.correct_query_typo(query, possible_questions)
    refined_query = pipeline.refine_follow_up_query(corrected_query)

    if corrected_query != query:
        await cl.Message(content=f"Did you mean: {corrected_query}?").send()

    try:
        # Perform query using the retriever chain
        result = await chain.acall({"query": refined_query, "memory": pipeline.memory})
        answer = result['result']
        source_documents = result.get('source_documents', [])
        
        # Prepare the answer and source documents
        text_elements = []
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                text_elements.append(
                    cl.Text(
                        content=source_doc.page_content, name=source_name, display="side"
                    )
                )
            source_names = [text_el.name for text_el in text_elements]
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
        
        # Send the answer and documents back to the user
        await cl.Message(content=answer, elements=text_elements).send()

        # Optionally set the current entity if needed (e.g., if a "key entity" is mentioned in the answer)
        pipeline.current_entity = pipeline.extract_key_entity(refined_query, answer)

    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()

# Main entry point for running the app (for testing purposes, not required in Chainlit interface)
if __name__ == "__main__":
    pdf_path = r"C:\Users\farheen.khanam\Downloads\Mindray.pdf"  # Replace with your actual PDF path
    pipeline = PDFQnAPipeline(pdf_path)
    # Step 1 & 2: Load and Chunk the PDF
    split_docs = pipeline.load_and_chunk_pdf()
    # Step 3: Store Chunks and Embeddings in Chroma DB
    pipeline.store_embeddings_in_chroma(split_docs)
    # Step 4: Create Retriever and LLM
    pipeline.create_retriever_and_llm()
    # Step 5: Ask Questions (for testing purposes, user interaction in Chainlit replaces this)
    pipeline.ask_questions()
    # Step 6: Generate Synthetic Q&A Pairs (optional, for testing)
    generate_qna = input("Do you want to generate synthetic Q&A pairs? (yes/no): ").lower()
    if generate_qna == "yes":
        synthetic_pairs = pipeline.generate_synthetic_qna(split_docs, num_pairs=30)
        # Step 7: Evaluate Performance on the Test Set
        pipeline.evaluate_performance(synthetic_pairs)
