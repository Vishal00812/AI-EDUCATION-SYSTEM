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
from utils import speak
import nest_asyncio
from PIL import Image
from langchain_groq import ChatGroq
from langchain import LLMChain, PromptTemplate
import google.generativeai as genai
from utils import takeCommand
from youtube_transcript_api import YouTubeTranscriptApi
from utils import class_9_subjects , class_10_subjects , class_11_subjects , class_12_subjects

nest_asyncio.apply()
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)



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
        color: #2A9D8F;
        margin-bottom: 30px;
    }

    /* Style for header sections */
    .header {
        color: #ffffff;
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

st.sidebar.markdown(
    "<h2 style='text-align: center; color: white; font-weight: bold;'>WELCOME TO EDUAI</h2>",
    unsafe_allow_html=True
)

st.sidebar.image("logo.png", width=160, use_column_width=False, output_format="auto", caption="Let AI Educate You")

option = st.sidebar.selectbox(
        "Choose an option:",
        ["Get Solution from Image", "Chat with Your Book","Transcript Youtube Video ","Genrate Practice MCQ","Self Assesment"]
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
    selected_class = st.selectbox("Select Your Class",["Select Class"]+class_options)
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
        Provide only the answer, without any additional text.
        <context>
        {context}
        <context>
        Questions: {input}
        """)

        # Initialize session state for storing chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        Sucess=''
        
        if "button_clicked_c11" not in st.session_state:
            st.session_state.button_clicked_c11 = False

        # Conditionally display the button only if it hasn't been clicked
        if not st.session_state.button_clicked_c11:
            # Button updates the session state directly in the same run
            if st.button("Read My Book"):
                vector_embedding()
                Sucess="Book Read Successfully"
                st.success(Sucess)
                st.session_state.button_clicked_c11 = True

        if st.session_state.button_clicked_c11:
            Sucess=''
            st.write(Sucess)


        # Button to embed documents
        #if st.button("Read My Book "):
            #vector_embedding()
            #st.success("Read Your Book Successfully")

        
        # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

        col1, col2 = st.columns([4, 1])
        with col1:
            prompt1 = st.chat_input("What you want to know?")
        with col2:
            Speak=st.button("Speak 🎙️")
        if Speak:
            st.write("please speak....")
            prompt1=takeCommand()
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
                
                st.chat_message("user").markdown(prompt1)
    # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt1})
                st.write("Response Time:", response_time)
                answer=response['answer']
                response = f"Assistant: {answer}"
    # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.audio(speak(answer), format="audio/wav")
            else:
                st.error("Please let me read the book by clicking  'Read My Book'.")
                
       


    if selected_class==10:
        st.markdown('<h2 class="header">Chatting with Class 10</h2>', unsafe_allow_html=True)
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
        Provide only the answer, without any additional text.
        <context>
        {context}
        <context>
        Questions: {input}
        """)


        # Initialize session state for storing chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        Sucess=''
        if "button_clicked_c10" not in st.session_state:
            st.session_state.button_clicked_c10 = False

        # Conditionally display the button only if it hasn't been clicked
        if not st.session_state.button_clicked_c10:
            # Button updates the session state directly in the same run
            if st.button("Read My Book"):
                vector_embedding()
                Sucess="Book Read Successfully"
                st.success(Sucess)
                st.session_state.button_clicked_c10 = True

        if st.session_state.button_clicked_c10:
            Sucess=''
            st.write(Sucess)
       

 
        st.write("-----")  


        col1, col2 = st.columns([4, 1])
        with col1:
            prompt1 = st.chat_input("What you want to know?")
        with col2:
            Speak=st.button("Speak 🎙️")
        if Speak:
            st.write("please speak....")
            prompt1=takeCommand()
            st.write(f"You said: {prompt1}")
        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
      

        if prompt1:
            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                
                st.chat_message("user").markdown(prompt1)

                st.session_state.messages.append({"role": "user", "content": prompt1})
                st.write("Response Time:", response_time)
                answer=response['answer']
                response = f"Assistant: {answer}"

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.audio(speak(answer), format="audio/wav")

            else:
                st.error("Please let me read the book by clicking  'Read My Book'.")

    if selected_class==12:
        st.markdown('<h2 class="header">Chatting with Class 12</h2>', unsafe_allow_html=True)

        def vector_embedding():
            if "vectors" not in st.session_state:
                index_file = "faiss_index_12"
                if os.path.exists(index_file):
                    st.session_state.vectors = FAISS.load_local(index_file, CohereEmbeddings(model="multilingual-22-12"),allow_dangerous_deserialization=True)
    

  
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        Provide only the answer, without any additional text.
        <context>
        {context}
        <context>
        Questions: {input}
        """)

        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if "button_clicked_c12" not in st.session_state:
            st.session_state.button_clicked_c12 = False

       
        Sucess=''
        if not st.session_state.button_clicked_c12:
         
            if st.button("Read My Book"):
                vector_embedding()
                Sucess="Book Read Successfully"
                st.success(Sucess)
                st.session_state.button_clicked_c12 = True

        if st.session_state.button_clicked_c12:
            Sucess=''
            st.write(Sucess)
        st.write("-----") 

        col1, col2 = st.columns([4, 1])
        with col1:
            prompt1 = st.chat_input("What you want to know?")
        with col2:
            Speak=st.button("Speak 🎙️")
        if Speak:
            st.write("please speak....")
            prompt1=takeCommand()
            st.write(f"You said: {prompt1}")
        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if prompt1:
            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                
                st.chat_message("user").markdown(prompt1)
    # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt1})
                st.write("Response Time:", response_time)
                answer=response['answer']
                response = f"Assistant: {answer}"
    # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.audio(speak(answer), format="audio/wav")

            else:
                st.error("Please let me read the book by clicking  'Read My Book'.")

    if selected_class==9:
        st.markdown('<h2 class="header">Chatting with Class 9</h2>', unsafe_allow_html=True)
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
        Provide only the answer, without any additional text.
        <context>
        {context}
        <context>
        Questions: {input}
        """)

        # Initialize session state for storing chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        
        if "button_clicked_c9" not in st.session_state:
            st.session_state.button_clicked_c9 = False

        # Conditionally display the button only if it hasn't been clicked
        sucess=''
        if not st.session_state.button_clicked_c9:
            # Button updates the session state directly in the same run
            if st.button("Read My Book"):
                vector_embedding()
                Sucess="Book Read Successfully"
                st.success(Sucess)
                st.session_state.button_clicked_c9 = True

        if st.session_state.button_clicked_c9:
            Sucess=''
            st.write(Sucess)


        # Move the input box to the bottom of the page
        st.write("-----")  # Add a separator

        # Input box for user to type in at the bottom
        col1, col2 = st.columns([4, 1])
        with col1:
            prompt1 = st.chat_input("What you want to know?")
        with col2:
            Speak=st.button("Speak 🎙️")
        if Speak:
            st.write("please speak....")
            prompt1=takeCommand()
            st.write(f"You said: {prompt1}")
        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
      
        # Handle user input and AI response
        if prompt1:
            if "vectors" in st.session_state:
                st.chat_message("user").markdown(prompt1)
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start

    # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt1})
                st.write("Response Time:", response_time)
                answer=response['answer']
                response = f"Assistant: {answer}"
    # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.audio(speak(answer), format="audio/wav")

            else:
                st.error("Please let me read the book by clicking  'Read My Book'.")




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

  
  
    input = st.text_input("What you want to know?")
    uploaded_file=None
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

    prompt="""You are a highly knowledgeable AI specialized in text summarization for educational content creation.
        Generate a summary of exactly {number} words based on the following text: "{text}".
        Ensure the summary captures the key points and is concise and informative.
        """

    prompt_template = PromptTemplate(input_variables=["number","text"],template=prompt)
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

    def generate_gemini_content(prompt_template,number,text):

        #model=genai.GenerativeModel("gemini-pro")
        Groq_API_KEY = os.getenv("GROQ_API_KEY")
        llm=ChatGroq(groq_api_key=Groq_API_KEY,model_name="Llama3-8b-8192")
        chain =LLMChain(llm=llm, prompt=prompt_template, verbose=True)
        response = chain.run({"number":number,"text":text})
        return response


    youtube_link = st.text_input("Enter YouTube Video Link:")
    number = st.slider("Select the number of lines for the summary", 50, 1000, 50)
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        print(video_id)
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get Detailed Notes"):
        transcript_text=extract_transcript_details(youtube_link)

        if transcript_text:
            summary=generate_gemini_content(prompt_template,number,transcript_text)
            st.markdown("## Detailed Notes:")
            st.write(summary)
            st.audio(speak(summary), format="audio/wav")
   


elif option=="Genrate Practice MCQ":
    st.markdown('<h1 class="title">Genrate Practice MCQ</h1>', unsafe_allow_html=True)
    load_dotenv()
    Groq_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=Groq_API_KEY, model_name="Llama3-8b-8192")
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


elif option=="Self Assesment" :
    st.markdown('<h1 class="title">Self Assesment</h1>', unsafe_allow_html=True)
    class_options = ['Class 9', 'Class 10', 'Class 11', 'Class 12']
    selected_class = st.selectbox("Select Your Class", ['Select Class']+class_options)
    if selected_class == "Class 9":
            st.markdown('<h2 class="header">Self Assesment 9</h2>', unsafe_allow_html=True)
            subject_option = st.selectbox("Select Subject", ["Select"] + list(class_9_subjects.keys()))
            if subject_option != "Select":
                chapters = class_9_subjects.get(subject_option, [])
                chapter_option = st.selectbox("Select Chapter", ["Select"] + chapters)
                if chapter_option != "Select":
                    st.write(f"You have selected: **{chapter_option}**")
                    def vector_embedding():
                        if "vectors" not in st.session_state:
                            index_file = "faiss_index_9"
                            if os.path.exists(index_file):
                                st.session_state.vectors = FAISS.load_local(
                                    index_file, CohereEmbeddings(model="multilingual-22-12"), allow_dangerous_deserialization=True
                                )
                            else:
                                st.error("Index file not found. Please check the path.")
                    vector_embedding()
                    PROMPT_TEMPLATE_STRING = """
                    Based on the CBSE Class 9 , generate a question in the form of a complete sentence:
                                        Create a question about the following chapter: {Chapter}
                                        
                                        Provide only the question as the output, with no additional text. 
                        """
                    prompt_template = PromptTemplate(input_variables=["Chapter"],template=PROMPT_TEMPLATE_STRING)
                    prompt2 = ChatPromptTemplate.from_template("""\
                        Evaluate the provided answer strictly based on the given context only, and assign marks out of 1.
                        Consider even minor inaccuracies when deducting marks. If the given marks are equal or greter than 0.6, round up to 1.
                        Provide only the marks as the output.

                        <context>
                        {context}
                        <context>

                        Question: {input}
                        Answer: {answer}

                        Output the marks out of 1.
                    """)
                    load_dotenv()
                    Groq_API_KEY = os.getenv("GROQ_API_KEY")
                    llm = ChatGroq(groq_api_key=Groq_API_KEY, model_name="Llama3-8b-8192")
                    prompt1 = chapter_option 
                    num_questions = st.number_input("Enter the number of questions you want", min_value=1,max_value=15, step=1)
                    if "marks" not in st.session_state:
                        st.session_state.marks = []
                    if "questions" not in st.session_state:
                        st.session_state.questions = []
                    if "answers" not in st.session_state:
                        st.session_state.answers = {}
                    if "generated_questions" not in st.session_state:
                        st.session_state.generated_questions = set()
                    if prompt1:
                        question_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
                        question = question_chain.run({"Chapter":prompt1})
                        while len(st.session_state.questions) < num_questions:
                            if question not in st.session_state.generated_questions:
                                st.session_state.questions.append(question)
                                st.session_state.generated_questions.add(question)
                            if len(st.session_state.questions) >= num_questions:
                                break
                    for i in range(num_questions):
                        if i < len(st.session_state.questions):
                            st.write(f"### Question {i + 1}: {st.session_state.questions[i]}")
                            answer_key = f"answer_{i}" 
                            answer = st.text_input(f"Enter your answer for Question {i + 1}", 
                                                key=answer_key, 
                                                value=st.session_state.answers.get(answer_key, ""))
                            if st.button(f"Submit Answer for Question {i + 1}", key=f"submit_{i}"):
                                if answer:
                                    st.session_state.answers[answer_key] = answer
                                    document_chain = create_stuff_documents_chain(llm, prompt2)
                                    retriever = st.session_state.vectors.as_retriever()
                                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                                    response = retrieval_chain.invoke({
                                        'input': st.session_state.questions[i], 
                                        'answer': answer
                                    })
                                    marks = float(response['answer'])
                                    if(marks>=0.6):
                                        marks=1
                                    else:
                                        marks=0 
                                    st.session_state.marks.append(marks)
                                    st.write(f"Marks for Question {i + 1}: {marks}/1")
                    if st.session_state.marks:
                        total_marks = sum(st.session_state.marks)
                        st.write(f"### Total Marks: {total_marks} out of {num_questions * 1}")

    elif selected_class == "Class 10":
            st.markdown('<h2 class="header">Self Assesment 10</h2>', unsafe_allow_html=True)
            subject_option = st.selectbox("Select Subject", ["Select"] + list(class_10_subjects.keys()))
            if subject_option != "Select":
                chapters = class_10_subjects.get(subject_option, [])
                chapter_option = st.selectbox("Select Chapter", ["Select"] + chapters)
                if chapter_option != "Select":
                    st.write(f"You have selected: **{chapter_option}**")
                    def vector_embedding():
                        if "vectors" not in st.session_state:
                            index_file = "faiss_index_10"
                            if os.path.exists(index_file):
                                st.session_state.vectors = FAISS.load_local(
                                    index_file, CohereEmbeddings(model="multilingual-22-12"), allow_dangerous_deserialization=True
                                )
                            else:
                                st.error("Index file not found. Please check the path.")
                    vector_embedding()
                    PROMPT_TEMPLATE_STRING = """
                    Based on the CBSE Class 10 , generate a question in the form of a complete sentence:
                                        Create a question about the following chapter: {Chapter}
                                        
                                        Provide only the question as the output, with no additional text. 
                        """
                    prompt_template = PromptTemplate(input_variables=["Chapter"],template=PROMPT_TEMPLATE_STRING)
                    prompt2 = ChatPromptTemplate.from_template("""\
                        Evaluate the provided answer strictly based on the given context only, and assign marks out of 1.
                        Consider even minor inaccuracies when deducting marks. If the given marks are equal or greter than 0.6, round up to 1.
                        Provide only the marks as the output.

                        <context>
                        {context}
                        <context>

                        Question: {input}
                        Answer: {answer}

                        Output the marks out of 1.
                    """)
                    load_dotenv()
                    Groq_API_KEY = os.getenv("GROQ_API_KEY")
                    llm = ChatGroq(groq_api_key=Groq_API_KEY, model_name="Llama3-8b-8192")
                    prompt1 = chapter_option 
                    num_questions = st.number_input("Enter the number of questions you want", min_value=1,max_value=15, step=1)
                    if "marks" not in st.session_state:
                        st.session_state.marks = []
                    if "questions" not in st.session_state:
                        st.session_state.questions = []
                    if "answers" not in st.session_state:
                        st.session_state.answers = {}
                    if "generated_questions" not in st.session_state:
                        st.session_state.generated_questions = set()
                    if prompt1:
                        question_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
                        question = question_chain.run({"Chapter":prompt1})
                        while len(st.session_state.questions) < num_questions:
                            if question not in st.session_state.generated_questions:
                                st.session_state.questions.append(question)
                                st.session_state.generated_questions.add(question)
                            if len(st.session_state.questions) >= num_questions:
                                break
                    for i in range(num_questions):
                        if i < len(st.session_state.questions):
                            st.write(f"### Question {i + 1}: {st.session_state.questions[i]}")
                            answer_key = f"answer_{i}" 
                            answer = st.text_input(f"Enter your answer for Question {i + 1}", 
                                                key=answer_key, 
                                                value=st.session_state.answers.get(answer_key, ""))
                            if st.button(f"Submit Answer for Question {i + 1}", key=f"submit_{i}"):
                                if answer:
                                    st.session_state.answers[answer_key] = answer
                                    document_chain = create_stuff_documents_chain(llm, prompt2)
                                    retriever = st.session_state.vectors.as_retriever()
                                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                                    response = retrieval_chain.invoke({
                                        'input': st.session_state.questions[i], 
                                        'answer': answer
                                    })
                                    marks = float(response['answer'])
                                    if(marks>=0.6):
                                        marks=1
                                    else:
                                        marks=0 
                                    st.session_state.marks.append(marks)
                                    st.write(f"Marks for Question {i + 1}: {marks}/1")
                    if st.session_state.marks:
                        total_marks = sum(st.session_state.marks)
                        st.write(f"### Total Marks: {total_marks} out of {num_questions * 1}")

    elif selected_class == "Class 11":
            st.markdown('<h2 class="header">Self Assesment 11</h2>', unsafe_allow_html=True)
            subject_option = st.selectbox("Select Subject", ["Select"] + list(class_11_subjects.keys()))
            if subject_option != "Select":
                chapters = class_11_subjects.get(subject_option, [])
                chapter_option = st.selectbox("Select Chapter", ["Select"] + chapters)
                if chapter_option != "Select":
                    st.write(f"You have selected: **{chapter_option}**")
                    def vector_embedding():
                        if "vectors" not in st.session_state:
                            index_file = "faiss_index_11"
                            if os.path.exists(index_file):
                                st.session_state.vectors = FAISS.load_local(
                                    index_file, CohereEmbeddings(model="multilingual-22-12"), allow_dangerous_deserialization=True
                                )
                            else:
                                st.error("Index file not found. Please check the path.")
                    vector_embedding()
                    PROMPT_TEMPLATE_STRING = """
                    Based on the CBSE Class 11 , generate a question in the form of a complete sentence:
                                        Create a question about the following chapter: {Chapter}
                                        
                                        Provide only the question as the output, with no additional text. 
                        """
                    prompt_template = PromptTemplate(input_variables=["Chapter"],template=PROMPT_TEMPLATE_STRING)
                    prompt2 = ChatPromptTemplate.from_template("""\
                        Evaluate the provided answer strictly based on the given context only, and assign marks out of 1.
                        Consider even minor inaccuracies when deducting marks. If the given marks are equal or greter than 0.6, round up to 1.
                        Provide only the marks as the output.

                        <context>
                        {context}
                        <context>

                        Question: {input}
                        Answer: {answer}

                        Output the marks out of 1.
                    """)
                    load_dotenv()
                    Groq_API_KEY = os.getenv("GROQ_API_KEY")
                    llm = ChatGroq(groq_api_key=Groq_API_KEY, model_name="Llama3-8b-8192")
                    prompt1 = chapter_option 
                    num_questions = st.number_input("Enter the number of questions you want", min_value=1,max_value=15, step=1)
                    if "marks" not in st.session_state:
                        st.session_state.marks = []
                    if "questions" not in st.session_state:
                        st.session_state.questions = []
                    if "answers" not in st.session_state:
                        st.session_state.answers = {}
                    if "generated_questions" not in st.session_state:
                        st.session_state.generated_questions = set()
                    if prompt1:
                        question_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
                        question = question_chain.run({"Chapter":prompt1})
                        while len(st.session_state.questions) < num_questions:
                            if question not in st.session_state.generated_questions:
                                st.session_state.questions.append(question)
                                st.session_state.generated_questions.add(question)
                            if len(st.session_state.questions) >= num_questions:
                                break
                    for i in range(num_questions):
                        if i < len(st.session_state.questions):
                            st.write(f"### Question {i + 1}: {st.session_state.questions[i]}")
                            answer_key = f"answer_{i}" 
                            answer = st.text_input(f"Enter your answer for Question {i + 1}", 
                                                key=answer_key, 
                                                value=st.session_state.answers.get(answer_key, ""))
                            if st.button(f"Submit Answer for Question {i + 1}", key=f"submit_{i}"):
                                if answer:
                                    st.session_state.answers[answer_key] = answer
                                    document_chain = create_stuff_documents_chain(llm, prompt2)
                                    retriever = st.session_state.vectors.as_retriever()
                                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                                    response = retrieval_chain.invoke({
                                        'input': st.session_state.questions[i], 
                                        'answer': answer
                                    })
                                    marks = float(response['answer'])
                                    if(marks>=0.6):
                                        marks=1
                                    else:
                                        marks=0 
                                    st.session_state.marks.append(marks)
                                    st.write(f"Marks for Question {i + 1}: {marks}/1")
                    if st.session_state.marks:
                        total_marks = sum(st.session_state.marks)
                        st.write(f"### Total Marks: {total_marks} out of {num_questions * 1}")

    elif selected_class == "Class 12":
            st.markdown('<h2 class="header">Self Assesment 12</h2>', unsafe_allow_html=True)
            subject_option = st.selectbox("Select Subject", ["Select"] + list(class_12_subjects.keys()))
            if subject_option != "Select":
                chapters = class_12_subjects.get(subject_option, [])
                chapter_option = st.selectbox("Select Chapter", ["Select"] + chapters)
                if chapter_option != "Select":
                    st.write(f"You have selected: **{chapter_option}**")
                    def vector_embedding():
                        if "vectors" not in st.session_state:
                            index_file = "faiss_index_12"
                            if os.path.exists(index_file):
                                st.session_state.vectors = FAISS.load_local(
                                    index_file, CohereEmbeddings(model="multilingual-22-12"), allow_dangerous_deserialization=True
                                )
                            else:
                                st.error("Index file not found. Please check the path.")
                    vector_embedding()
                    PROMPT_TEMPLATE_STRING = """
                    Based on the CBSE Class 12 , generate a question in the form of a complete sentence:
                                        Create a question about the following chapter: {Chapter}
                                        
                                        Provide only the question as the output, with no additional text. 
                        """
                    prompt_template = PromptTemplate(input_variables=["Chapter"],template=PROMPT_TEMPLATE_STRING)
                    prompt2 = ChatPromptTemplate.from_template("""\
                        Evaluate the provided answer strictly based on the given context only, and assign marks out of 1.
                        Consider even minor inaccuracies when deducting marks. If the given marks are equal or greter than 0.6, round up to 1.
                        Provide only the marks as the output.

                        <context>
                        {context}
                        <context>

                        Question: {input}
                        Answer: {answer}

                        Output the marks out of 1.
                    """)
                    load_dotenv()
                    Groq_API_KEY = os.getenv("GROQ_API_KEY")
                    llm = ChatGroq(groq_api_key=Groq_API_KEY, model_name="Llama3-8b-8192")
                    prompt1 = chapter_option 
                    num_questions = st.number_input("Enter the number of questions you want", min_value=1,max_value=15, step=1)
                    if "marks" not in st.session_state:
                        st.session_state.marks = []
                    if "questions" not in st.session_state:
                        st.session_state.questions = []
                    if "answers" not in st.session_state:
                        st.session_state.answers = {}
                    if "generated_questions" not in st.session_state:
                        st.session_state.generated_questions = set()
                    if prompt1:
                        question_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
                        question = question_chain.run({"Chapter":prompt1})
                        while len(st.session_state.questions) < num_questions:
                            if question not in st.session_state.generated_questions:
                                st.session_state.questions.append(question)
                                st.session_state.generated_questions.add(question)
                            if len(st.session_state.questions) >= num_questions:
                                break
                    for i in range(num_questions):
                        if i < len(st.session_state.questions):
                            st.write(f"### Question {i + 1}: {st.session_state.questions[i]}")
                            answer_key = f"answer_{i}" 
                            answer = st.text_input(f"Enter your answer for Question {i + 1}", 
                                                key=answer_key, 
                                                value=st.session_state.answers.get(answer_key, ""))
                            if st.button(f"Submit Answer for Question {i + 1}", key=f"submit_{i}"):
                                if answer:
                                    st.session_state.answers[answer_key] = answer
                                    document_chain = create_stuff_documents_chain(llm, prompt2)
                                    retriever = st.session_state.vectors.as_retriever()
                                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                                    response = retrieval_chain.invoke({
                                        'input': st.session_state.questions[i], 
                                        'answer': answer
                                    })
                                    marks = float(response['answer'])
                                    if(marks>=0.6):
                                        marks=1
                                    else:
                                        marks=0 
                                    st.session_state.marks.append(marks)
                                    st.write(f"Marks for Question {i + 1}: {marks}/1")
                    if st.session_state.marks:
                        total_marks = sum(st.session_state.marks)
                        st.write(f"### Total Marks: {total_marks} out of {num_questions * 1}")