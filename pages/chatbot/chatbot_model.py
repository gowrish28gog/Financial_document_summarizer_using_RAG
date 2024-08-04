from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI


from transformers import LlamaForCausalLM, LlamaTokenizer

# Transformers packages
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
# from transformers import AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Loading Model
print('Loading the model...')

# define model to use
MODEL_NAME = "GPT"

if MODEL_NAME == "GPT":
	from langchain.chat_models import ChatOpenAI
	from langchain.prompts import ChatPromptTemplate
	from langchain.schema.runnable import RunnablePassthrough
	from langchain.schema.output_parser import StrOutputParser
	import os
	import openai
	# Set your OpenAI API key
	openai.api_key = os.getenv("OPENAI_API_KEY")

	# Define LLM
	llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
elif MODEL_NAME == "LLAMA2":

	model_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
	model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
	                                         trust_remote_code = True, config = model_config, device_map = 'auto')
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

elif MODEL_NAME == "FLANT5":
	model_name = "google/flan-t5-large"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)

if MODEL_NAME != "GPT":
	print('Creating Pipeline...')
	# Creating Pipeline
	query_pipeline = transformers.pipeline(
			"text-generation",
			model=model,
			tokenizer=tokenizer,
			torch_dtype=torch.float16,
			device_map="auto",)
	llm = HuggingFacePipeline(pipeline=query_pipeline)


# Loading PDF
print('Loading the corpus for TESLA...')
from langchain.document_loaders import PyPDFLoader
# create a loader
loader = PyPDFLoader(r"data/tsla-20230930.pdf")

# load your data
data = loader.load()

# Text splitter
print('Instantiating Text Splitter...')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
all_splits = text_splitter.split_documents(data)

# Creating Embeddings
print('Preparing Embeddings...')
model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {"device": device}

embeddings = HuggingFaceEmbeddings(model_name=model_name)#, model_kwargs=model_kwargs)

print('Preparing Vector Embeddings...')
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

# Preparing Chain
print('Preparing chain...')

if MODEL_NAME == "GPT":

	# Define prompt template
	template = """You are an assistant for question-answering tasks for Retrieval Augmented Generation system for the financial reports such as 10Q and 10K.
	Use the following pieces of retrieved context to answer the question. 
	If you don't know the answer, just say that you don't know. 
	Use two sentences maximum and keep the answer concise.
	Question: {question} 
	Context: {context} 
	Answer:
	"""

	prompt = ChatPromptTemplate.from_template(template)
	retriever = vectordb.as_retriever()

	# Setup RAG pipeline
	conversation_chain = (
		{"context": retriever,  "question": RunnablePassthrough()} 
		| prompt 
		| llm
		| StrOutputParser() 
	)
	# chat = OpenAI(temperature=0)



	# conversation = ConversationChain(
	#     llm=chat, 
	#     verbose=True,
	#     memory=ConversationBufferMemory()
	# )
else:
	conversation_chain = load_qa_chain(llm, chain_type="map_reduce")

print('Chain Prepared...')