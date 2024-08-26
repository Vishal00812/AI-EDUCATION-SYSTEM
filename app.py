import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import os
import time
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import Cohere
import asyncio
import nest_asyncio
import pathlib
import textwrap
from PIL import Image
import pyttsx3
from langchain_groq import ChatGroq


import google.generativeai as genai
nest_asyncio.apply()
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load environment variables


st.markdown(
    """
    <style>
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        padding: 10px 0;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True
)
option = st.sidebar.selectbox(
        "Choose an option:",
        ["Get Solution from Image", "Chat with Your Book"]
    )
if option=="Chat with Your Book":
    load_dotenv()

# Ensure the Google API key is retrieved correctly
    #Cohere_API_KEY = os.getenv("COHERE_API_KEY")
    Groq_API_KEY = os.getenv("GROQ_API_KEY")
    llm=ChatGroq(groq_api_key=Groq_API_KEY,model_name="Llama3-8b-8192")

# Initialize LLM with synchronous method
    #llm =Cohere(model="command", temperature=0, cohere_api_key=Cohere_API_KEY)
    class_options = [9, 10, 11, 12]
    selected_class = st.selectbox("Select your class:", class_options)
    if selected_class==11:
    # Function to create vector embeddings
        def vector_embedding():
            if "vectors" not in st.session_state:
                index_file = "faiss_index_11"
                if os.path.exists(index_file):
                    st.session_state.vectors = FAISS.load_local(index_file, CohereEmbeddings(model="multilingual-22-12"),allow_dangerous_deserialization=True)
    

        # Define prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """)

        # Initialize session state for storing chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Streamlit UI
        st.title("CHAT WITH YOUR BOOK")

        # Button to embed documents
        if st.button("Documents Embedding"):
            vector_embedding()
            st.success("Vector Store DB Is Ready")

        # Display chat history
        for message in st.session_state.messages:
            st.write(message)

        # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

        # Input box for user to type in at the bottom
        prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")

        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Handle user input and AI response
        if prompt1:
            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                
                # Display response and chat history
                st.session_state.messages.append(f"You: {prompt1}")
                st.session_state.messages.append(f"AI: {response['answer']}")
                st.write("Response Time:", response_time)
                st.write(response['answer'])

                # Clear the input box by resetting the text input
                st.session_state.input = ""

            else:
                st.error("Please embed the documents first by clicking 'Documents Embedding'.")

        # Reset chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []

    if selected_class==10:
        load_dotenv()

        # Ensure the Google API key is retrieved correctly
        #Cohere_API_KEY = os.getenv("COHERE_API_KEY")
        Groq_API_KEY = os.getenv("GROQ_API_KEY")
        llm=ChatGroq(groq_api_key=Groq_API_KEY,model_name="Llama3-8b-8192")

    # Initialize LLM with synchronous method
        #llm =Cohere(model="command", temperature=0, cohere_api_key=Cohere_API_KEY)
        # Function to create vector embeddings
        def vector_embedding():
            if "vectors" not in st.session_state:
                index_file = "faiss_index_10"
                if os.path.exists(index_file):
                    st.session_state.vectors = FAISS.load_local(index_file, CohereEmbeddings(model="multilingual-22-12"),allow_dangerous_deserialization=True)
                
            # Define prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """)

            # Initialize session state for storing chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

            # Streamlit UI
        st.title("CHAT WITH YOUR BOOK")

            # Button to embed documents
        if st.button("Documents Embedding"):
            vector_embedding()
            st.success("Vector Store DB Is Ready")

            # Display chat history
        for message in st.session_state.messages :
            st.write(message)

            # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

            # Input box for user to type in at the bottom
        prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")

        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

            # Handle user input and AI response
        if prompt1:
            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                    
                    # Display response and chat history
                st.session_state.messages.append(f"You: {prompt1}")
                st.session_state.messages.append(f"AI: {response['answer']}")
                st.write("Response Time:", response_time)
                st.write(response['answer'])

                    # Clear the input box by resetting the text input
                st.session_state.input = ""

            else:
                st.error("Please embed the documents first by clicking 'Documents Embedding'.")

            # Reset chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []

    if selected_class==9:
        load_dotenv()

        # Ensure the Google API key is retrieved correctly
        #Cohere_API_KEY = os.getenv("COHERE_API_KEY")
        Groq_API_KEY = os.getenv("GROQ_API_KEY")
        llm=ChatGroq(groq_api_key=Groq_API_KEY,model_name="Llama3-8b-8192")

    # Initialize LLM with synchronous method
        #llm =Cohere(model="command", temperature=0, cohere_api_key=Cohere_API_KEY)
        # Function to create vector embeddings
        def vector_embedding():
            if "vectors" not in st.session_state:
                index_file = "faiss_index_9"
                if os.path.exists(index_file):
                    st.session_state.vectors = FAISS.load_local(index_file, CohereEmbeddings(model="multilingual-22-12"),allow_dangerous_deserialization=True)
                
            # Define prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """)

            # Initialize session state for storing chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

            # Streamlit UI
        st.title("CHAT WITH YOUR BOOK")

            # Button to embed documents
        if st.button("Documents Embedding"):
            vector_embedding()
            st.success("Vector Store DB Is Ready")

            # Display chat history
        for message in st.session_state.messages :
            st.write(message)

            # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

            # Input box for user to type in at the bottom
        prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")

        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

            # Handle user input and AI response
        if prompt1:
            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                    
                    # Display response and chat history
                st.session_state.messages.append(f"You: {prompt1}")
                st.session_state.messages.append(f"AI: {response['answer']}")
                st.write("Response Time:", response_time)
                st.write(response['answer'])

                    # Clear the input box by resetting the text input
                st.session_state.input = ""

            else:
                st.error("Please embed the documents first by clicking 'Documents Embedding'.")

            # Reset chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []


    if selected_class==12:
        load_dotenv()

        # Ensure the Google API key is retrieved correctly
        #Cohere_API_KEY = os.getenv("COHERE_API_KEY")
        Groq_API_KEY = os.getenv("GROQ_API_KEY")
        llm=ChatGroq(groq_api_key=Groq_API_KEY,model_name="Llama3-8b-8192")

    # Initialize LLM with synchronous method
        #llm =Cohere(model="command", temperature=0, cohere_api_key=Cohere_API_KEY)
        # Function to create vector embeddings
        def vector_embedding():
            if "vectors" not in st.session_state:
                index_file = "faiss_index_12"
                if os.path.exists(index_file):
                    st.session_state.vectors = FAISS.load_local(index_file, CohereEmbeddings(model="multilingual-22-12"),allow_dangerous_deserialization=True)
                
            # Define prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """)

            # Initialize session state for storing chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

            # Streamlit UI
        st.title("CHAT WITH YOUR BOOK")

            # Button to embed documents
        if st.button("Documents Embedding"):
            vector_embedding()
            st.success("Vector Store DB Is Ready")

            # Display chat history
        for message in st.session_state.messages :
            st.write(message)

            # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

            # Input box for user to type in at the bottom
        prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")

        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

            # Handle user input and AI response
        if prompt1:
            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                    
                    # Display response and chat history
                st.session_state.messages.append(f"You: {prompt1}")
                st.session_state.messages.append(f"AI: {response['answer']}")
                st.write("Response Time:", response_time)
                st.write(response['answer'])

                    # Clear the input box by resetting the text input
                st.session_state.input = ""

            else:
                st.error("Please embed the documents first by clicking 'Documents Embedding'.")

            # Reset chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []





elif option=="Get Solution from Image":
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    ## Function to load OpenAI model and get respones

    def get_gemini_response(input,image):
        model = genai.GenerativeModel('gemini-1.5-flash')
        if input!="":
            response = model.generate_content([input,image])
        else:
            response = model.generate_content(image)
        return response.text

    ##initialize our streamlit app

  

    st.header("GET SOLUTION FROM IMAGE")
    input="provide the solutiion of the question in the image"
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)


    submit=st.button("Provide Me A Solution")

    ## If ask button is clicked

    if submit:
        
        response=get_gemini_response(input,image)
        st.subheader("The Response is")
        st.write(response)



