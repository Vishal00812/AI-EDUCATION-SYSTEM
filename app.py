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
from PIL import Image
from langchain_groq import ChatGroq
from langchain import LLMChain, PromptTemplate
from utils import recognize_speech
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
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
st.markdown(
    """
    <style>
    /* Background settings */
    body {
        background-color: #f7f9fc;
        font-family: 'Open Sans', sans-serif;
    }

    /* Center the title */
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: Blue;
        margin-bottom: 30px;
    }

    /* Style for header sections */
    .header {
        color: #264653;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
    }

    /* Custom buttons */
    .stButton button {
        background-color: #2A9D8F;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 1.2rem;
        cursor: pointer;
    }

    .stButton button:hover {
        background-color: #21867a;
    }

    /* Styling file uploader */
    .stFileUploader button {
        background-color: #e76f51;
        color: white;
        border-radius: 8px;
        font-size: 1rem;
        padding: 10px 15px;
        cursor: pointer;
    }

    /* Custom inputs */
    .stTextInput input {
        border: 2px solid #2A9D8F;
        border-radius: 5px;
        padding: 10px;
        font-size: 1.2rem;
        width: 100%;
        margin-top: 10px;
    }

    /* Footer styles */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #264653;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 1rem;
        display: flex;
        justify-content: center; /* Centers horizontally */
        align-items: center; /* Centers vertically */
    }

    /* Image styling */
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Transcript text area */
    .transcript {
        background-color: #ffffff;
        border: 1px solid #2A9D8F;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        font-size: 1rem;
        color: #333;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True
)
option = st.sidebar.selectbox(
        "Choose an option:",
        ["Get Solution from Image", "Chat with Your Book","Transcript Youtube Video ","Genrate Practice MCQ"]
    )
if option=="Chat with Your Book":
    st.markdown('<h1 class="title">Chat with Your Book</h1>', unsafe_allow_html=True)
    load_dotenv()

# Ensure the Google API key is retrieved correctly
    #Cohere_API_KEY = os.getenv("COHERE_API_KEY")
    Groq_API_KEY = os.getenv("GROQ_API_KEY")
    llm=ChatGroq(groq_api_key=Groq_API_KEY,model_name="Llama3-8b-8192")

# Initialize LLM with synchronous method
    #llm =Cohere(model="command", temperature=0, cohere_api_key=Cohere_API_KEY)
    class_options = [9, 10, 11, 12]
    selected_class = st.selectbox("Select Your Class", class_options)
    if selected_class==11:
        st.markdown('<h2 class="header">Chatting with Class 11</h2>', unsafe_allow_html=True)
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

        
        # Button to embed documents
        if st.button("Read My Book "):
            vector_embedding()
            st.success("Read Your Book Successfully")

        # Display chat history
        for message in st.session_state.messages:
            st.write(message)

        # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

        # Input box for user to type in at the bottom
        prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")

        if st.button("Speak 🎙️"):
            st.write("please speak....")
            prompt1=recognize_speech()
            st.write(f"You said: {prompt1}")
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
                st.error("Please let me read the book by clicking  'Read My Book'.")


        # Reset chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []

    if selected_class==10:
        st.markdown('<h2 class="header">Chatting with Class 10</h2>', unsafe_allow_html=True)
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
        

            # Button to embed documents
        if st.button("Read My Book "):
            vector_embedding()
            st.success("Read Your Book Successfully")

            # Display chat history
        for message in st.session_state.messages :
            st.write(message)

            # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

            # Input box for user to type in at the bottom
        prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")
        if st.button("Speak 🎙️"):
            st.write("please speak....")
            prompt1=recognize_speech()
            st.write(f"You said: {prompt1}")

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
                st.error("Please let me read the book by clicking  'Read My Book'.")


            # Reset chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []

    if selected_class==9:
        st.markdown('<h2 class="header">Chatting with Class 9</h2>', unsafe_allow_html=True)
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
        

            # Button to embed documents
        if st.button("Read My Book "):
            vector_embedding()
            st.success("Read Your Book Successfully")

            # Display chat history
        for message in st.session_state.messages :
            st.write(message)

            # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

            # Input box for user to type in at the bottom
        prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")
        if st.button("Speak 🎙️"):
            st.write("please speak....")
            prompt1=recognize_speech()
            st.write(f"You said: {prompt1}")

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
                st.error("Please let me read the book by clicking  'Read My Book'.")

            # Reset chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []


    if selected_class==12:
        st.markdown('<h2 class="header">Chatting with Class 12</h2>', unsafe_allow_html=True)

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


            # Button to embed documents
        if st.button("Read My Book "):
            vector_embedding()
            st.success("Read Your Book Successfully")

            # Display chat history
        for message in st.session_state.messages :
            st.write(message)

            # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

            # Input box for user to type in at the bottom
        prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")
        if st.button("Speak 🎙️"):
            st.write("please speak....")
            prompt1=recognize_speech()
            st.write(f"You said: {prompt1}")

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
                st.error("Please let me read the book by clicking  'Read My Book'.")

            # Reset chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []





elif option=="Get Solution from Image":
    st.markdown('<h1 class="title">Get Solution from Image</h1>', unsafe_allow_html=True)
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


elif option=="Transcript Youtube Video ":
    st.markdown('<h1 class="title">Transcript Youtube Video </h1>', unsafe_allow_html=True)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    prompt="""You are Yotube video summarizer. You will be taking the transcript text
    and summarizing the entire video and providing the important summary in points
    within 250 words. Please provide the summary of the text given here:  """


## getting the transcript data from yt videos
    def extract_transcript_details(youtube_video_url):
        try:
            video_id=youtube_video_url.split("=")[1]
            
            transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

            transcript = ""
            for i in transcript_text:
                transcript += " " + i["text"]

            return transcript

        except Exception as e:
            raise e

    def generate_gemini_content(transcript_text,prompt):

        model=genai.GenerativeModel("gemini-pro")
        response=model.generate_content(prompt+transcript_text)
        return response.text

    st.title("YouTube Transcript ")
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        print(video_id)
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get Detailed Notes"):
        transcript_text=extract_transcript_details(youtube_link)

        if transcript_text:
            summary=generate_gemini_content(transcript_text,prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)

elif option=="Genrate Practice MCQ":
    st.markdown('<h1 class="title">Genrate Practice MCQ</h1>', unsafe_allow_html=True)
    load_dotenv()
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")
    llm=ChatGroq(groq_api_key=GROQ_API_KEY ,model_name="Llama3-8b-8192")

    PROMPT_TEMPLATE_STRING = """
    You are a highly knowledgeable AI specialized in educational content creation.
    Generate {number} multiple-choice question(s) on the topic: {topic} with the following details:
    - Difficulty level: {difficulty}
    - Each question should be challenging and provide four answer choices, with only one correct answer.

    Format each question as follows:
    Question: [The generated question text]

    A) [Answer choice 1]
    B) [Answer choice 2]
    C) [Answer choice 3]
    D) [Answer choice 4]

    Correct Answer: [The letter corresponding to the correct answer]

    Make sure that the correct answer is clearly indicated.
    """




    # Create a PromptTemplate instance with the string template
    prompt_template = PromptTemplate(input_variables=["number","topic","difficulty"],template=PROMPT_TEMPLATE_STRING)

    # Input for the topic of the MCQ
    topic = st.text_input("Enter the topic for the MCQ:")

    # Select difficulty level
    difficulty = st.selectbox("Select difficulty level:", ["easy", "medium", "hard"])
    
    number = st.selectbox("Select number of question :", [5, 10, 15 , 20])
    

    if st.button("Generate MCQ"):
        if topic:
            with st.spinner("Generating MCQ..."):
                # Prepare the formatted prompt

                # Initialize LLMChain with the prompt and LLM
                mcq_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
                mcq = mcq_chain.run({"number":number,"topic":topic,"difficulty":difficulty})
                if mcq:
                    st.write(mcq)
                else:
                    st.error("Failed to generate MCQ. Please try again.")
        else:
            st.error("Please enter a topic.")

st.markdown('<div class="footer">AI Education</div>', unsafe_allow_html=True)
