from pathlib import Path
from flask import Flask, render_template, request,flash, redirect, url_for
import shutil
import sys
from flask import Flask, json, jsonify, render_template, request
import os
import openai
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from werkzeug.utils import secure_filename
import constants
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import cx_Oracle

OPENAI_API_KEY = constants.MSNOE_API_KEY
OPENAI_DEPLOYMENT_ENDPOINT = constants.MSNOE_ENDPOINT
OPENAI_DEPLOYMENT_NAME = constants.MODEL_GPT35
OPENAI_MODEL_NAME =  constants.MODEL_NAME
OPENAI_DEPLOYMENT_VERSION =  constants.API_VERSION
 
OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME =  constants.MODEL_LARGE
OPENAI_ADA_EMBEDDING_MODEL_NAME =  constants.EMBEDDING_MODEL_NAME
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "chatbot_with_langchain"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDw-x2n9ertBiisOSpHWeilBLoyO07bzH0"
     
 
store_vector_index_path="./VectorStore"
pdf_folder_path = "\documents"
 
 
openai.api_type = "azure"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = os.getenv('OPENAI_API_VERSION')
 
 
model = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)
 
embeddings=AzureOpenAIEmbeddings(azure_deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                            model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                            azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                            openai_api_type="azure",
                            openai_api_key=OPENAI_API_KEY,
                            chunk_size=1)            
def get_db_connection():
    dsn_tns = cx_Oracle.makedsn('host', 1521, service_name='service_name')  # Update with your DB details
    connection = cx_Oracle.connect(user='u121137', password='Devideepu13$', dsn=dsn_tns)  # Update with your credentials
    return connection
def log_chat_to_db(user_message, bot_response):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO chat_log (user_message, bot_response) VALUES (:1, :2)",
                       (user_message, bot_response))
        connection.commit()
    except Exception as e:
        print(f"Error logging chat to DB: {e}")
    finally:
        cursor.close()
        connection.close()
def ask_question(qa, question):
    result = qa({"query": question})
    print("Question:", question)
    print("Answer:", result["result"])
 
def ask_question_with_context(qa, question, chat_history):
    query = "Hi"
    result = qa({"question": question, "chat_history": chat_history})
    print("Assistant:", result["answer"])
    chat_history = [(query, result["answer"])]
    return result["answer"]
 
def  initialise_faiss_index(embeddings):
    texts = ["Universal Tester", "Virtual Assitant"]
    faiss = FAISS.from_texts(texts, embeddings)
    #save the embeddings into FAISS vector store
    faiss.save_local(store_vector_index_path)
 
def load_and_process_pdfs(pdf_folder_path):
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits
 
 
def CheckDirectory(path):
    folderpath = Path(__file__).parent / path
    if os.path.isdir(folderpath):
        try:
            shutil.rmtree(path)                  
            os.makedirs(path)
            print(f"Directory '{folderpath}' has been removed and created folder.")
        except FileNotFoundError:
            print(f"Directory '{folderpath}' does not exist.")
        except OSError as e:
            print(f"Error: {e}")
    else:
        try:
            os.makedirs(path)
            print(f"Directory '{folderpath}' has been created.")
        except Exception as e:
            print(f"Error: {e}")
           
def Ingestdata(index_filepath):
 
    CheckDirectory(store_vector_index_path)
 
    #Create FAISS Index file      
    initialise_faiss_index(embeddings)
 
   # load the faiss vector store we saved into memory    
    vectorStore = FAISS.load_local(store_vector_index_path, embeddings,allow_dangerous_deserialization=True)
    splits = load_and_process_pdfs(index_filepath)
    print(f"Directory '{index_filepath }'.")
    vectorStore.add_documents(splits)
 
    vectorStore.save_local(store_vector_index_path)
    print(f"Directory '{store_vector_index_path }' has been created.")
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}  # Modify as needed
 
 
def allowed_file(filename):
    try:
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    except:
        return False
 
@app.route("/api/v1.0/csharp_python_restfulapi_json", methods=["POST"])
def csharp_python_restfulapi_json():
    """
    simple c# test to call python restful api web service
    """
    try:                
        request_json = request.get_json()  
        name = request_json.get("content")
 
        print(request_json)  
        print(name)      
        # response = jsonify({"message": "Received data successfully!"})
        response=  jsonify({"Assistant": get_chat_response_new(name)})      
        response.status_code = 200  
    except:
        exception_message = sys.exc_info()[1]
        response = json.dumps({"content":exception_message})
        response.status_code = 400
    return response
 
@app.route("/")
def index():
    return render_template('chat.html')
 
 
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input=msg
    response =  get_chat_response_new(input)
    return response
   
 
def get_Chat_response(text):
    return 'Yes'
 
 
           
def get_chat_response_new(text):
 
    vectorStore = FAISS.load_local(store_vector_index_path, embeddings,allow_dangerous_deserialization=True)
 
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":5})
 
    review_template_str = """Your job is to answer questions only from the Embeddigs. Be Precise and limit your response within 5 to 6 sentence
    ... as possible, but don't make up any information that's not
    ... from the context. If you don't know an answer, say you don't know.
    ...
    ...{chat_history}
    ...
    ... {question}
    ... """
 
 
    QUESTION_PROMPT = PromptTemplate.from_template(review_template_str)
 
    # QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
 
    # Chat History:
    # {chat_history}  
    # Follow Up Input: {question}
    # Standalone question:""")
 
    qa = ConversationalRetrievalChain.from_llm(llm=model,
                                            retriever=retriever,
                                            condense_question_prompt=QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)
 
    chat_history = []
    return ask_question_with_context(qa, text, chat_history)
 
@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files[]')  
 
    for file in files:
        if file.filename == '':
            print('No selected file')
            continue
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            current_dir = os.getcwd()+pdf_folder_path
            print (f'{current_dir} here it is uploaded')
            current_dir="C:\chatbot main1\chatbot starter_Main\documents"
            filepath = os.path.join(current_dir, filename)
            file.save(filepath)
            print(f'File {filename} uploaded successfully')
        else:
            print(f'File {filename} has an invalid extension')
    Ingestdata(current_dir)
    return 'success'
@app.route('/ingest', methods=['POST'])
def process_folder_path():
   
    folder_path = request.form.get('folderPath')
 
    folder_path =   os.path(folder_path)
    print(f'FullPath: {folder_path}')
    if folder_path:
    # Validate and sanitize path
    # ...
      try:
      # Use os.scandir or similar for directory scanning
        entries = os.scandir(folder_path)
        print(f' {entries} here' )
      # Process directory entries
      # ...
      except FileNotFoundError:
      # Handle file not found error
        return 'Error: Folder not found.'
      else:
    # Handle missing folder path
       return 'Error: Folder path not provided.'
    # Process the folder path (e.g., save to a database, perform actions, etc.)
   #print(folder_path)
   # return folder_path
   # return Ingestdata(folder_path)
    #return f"Received folder path: {folder_path}"
 
@app.route('/users')
def users():
    # Render your ingest page template (e.g., ingest.html)
    return render_template('users.html')
 
if __name__ == '__main__':
    app.run(host='0.0.0.0')  # Use a different port (e.g., 8080)