from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Loading Model
print('Loading the model...')
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


from datasets import Dataset

# sample 3 questions
questions = ["What’s the total assets?",
             "How much cash was provided by or used in operating activities during the quarter?",
             "What are the biggest risks for Tesla as a business?",
            ]
ground_truth = [["The total assets of the company is $93,941."],
                ["The amount of cash provided by or used in operating activities during the quarter was $8.89 billion."],
                ["The biggest risks for Tesla as a business are its ability to continue as a going concern and its inability to raise additional capital to fund its operations and growth."]]
answers = []
contexts = []

# Inference
for query in questions:
  answers.append(conversation_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

### LLama 2 evaluation
## copied from LLama 2 notebook
# llama2_answers = [
#   "The total assets of Tesla as of September 30, 2023, are $93,941 million.",
#   "According to the provided documents, Tesla provided $8,886 million in cash from operating activities during the quarter.",
#   "The biggest risks for Tesla as a business are legal proceedings, risks, and claims that arise from the normal course of business activities, including a recent data breach that potentially affected individuals (current and former employees) and regulatory authorities. Tesla has made notifications to potentially affected individuals and is working with law enforcement and other authorities to address the issue. Additionally, the company faces risks related to its energy generation and storage business, as well as its financial performance and stockholders' equity."
# ]
# llama2_contexts = [
#   ['September 30,\n2023December 31,\n2022\nAssets\nCurrent assets\nCash and cash equivalents $ 15,932  $ 16,253  \nShort-term investments 10,145  5,932  \nAccounts receivable, net 2,520  2,952  \nInventory 13,721  12,839  \nPrepaid expenses and other current assets 2,708  2,941  \nTotal current assets 45,026  40,917  \nOperating lease vehicles, net 6,119 5,035  \nSolar ener gy systems, net 5,293  5,489  \nProperty , plant and equipment, net 27,744  23,548  \nOperating lease right-of-use assets 3,637  2,563',
#  'Property , plant and equipment, net 27,744  23,548  \nOperating lease right-of-use assets 3,637  2,563  \nDigital assets, net 184 184 \nIntangible assets, net 191 215 \nGoodwill 250 194 \nOther non-current assets 5,497  4,193  \nTotal assets $ 93,941  $ 82,338  \nLiabilities\nCurrent liabilities\nAccounts payable $ 13,937  $ 15,255  \nAccrued liabilities and other 7,636  7,142  \nDeferred revenue 2,206  1,747  \nCustomer deposits 894 1,063  \nCurrent portion of debt and finance leases 1,967  1,502',
#  'Prepaid expenses and other current assets 2,708  2,941  \nTotal current assets 45,026  40,917  \nOperating lease vehicles, net 6,119 5,035  \nSolar ener gy systems, net 5,293  5,489  \nProperty , plant and equipment, net 27,744  23,548  \nOperating lease right-of-use assets 3,637  2,563  \nDigital assets, net 184 184 \nIntangible assets, net 191 215 \nGoodwill 250 194 \nOther non-current assets 5,497  4,193  \nTotal assets $ 93,941  $ 82,338  \nLiabilities\nCurrent liabilities',
#  'investments $ 26,089 $ 1 $ (13)$ 26,077 $ 15,932 $ 10,145 \n December 31, 2022\n Adjusted CostGross\nUnrealized\nGainsGross\nUnrealized\nLosses Fair ValueCash and Cash\nEquivalentsShort-Term\nInvestments\nCash $ 13,965 $ — $ — $ 13,965 $ 13,965 $ — \nMoney market funds 2,188 — — 2,188 2,188 — \nU.S. government securities 897 — (3) 894 — 894 \nCorporate debt securities 907 — (22) 885 — 885 \nCertificates of deposit and time deposits 4,252 1 — 4,253 100 4,153 \nTotal cash, cash equivalents and short-term'],
#  ['Net cash provided by operating activities 8,886 11,446 \nCash Flows from Investing Activities\nPurchases of property and equipment excluding finance leases, net of sales (6,592) (5,300)\nPurchases of solar energy systems, net of sales — (5)\nProceeds from sales of digital assets — 936 \nPurchase of intangible assets — (9)\nPurchases of investments (13,221) (1,467)\nProceeds from maturities of investments 8,959 3 \nProceeds from sales of investments 138 —',
#  'Cash and cash equivalents and restricted cash, end of period $ 16,590 $ 20,149 \nSupplemental Non-Cash Investing and Financing Activities\nAcquisitions of property and equipment included in liabilities $ 1,717 $ 1,877 \nLeased assets obtained in exchange for finance lease liabilities $ 1 $ 36 \nLeased assets obtained in exchange for operating lease liabilities $ 1,548 $ 691',
#  'Total as presented in the consolidated statements of cash\nflows $ 16,590 $ 16,924 $ 20,149 $ 18,144 \nAccounts Receivable and Allowance for Doubtful Accounts\nDepending on the day of the week on which the end of a fiscal quarter falls, our accounts receivable balance may fluctuate as\nwe are waiting for certain customer payments to clear through our banking institutions and receipts of payments from our financing',
#  'Summary of Cash Flows\n Nine Months Ended September 30,\n(Dollars in millions) 2023 2022\nNet cash provided by operating activities $ 8,886 $ 11,446 \nNet cash used in investing activities $ (10,780)$ (5,842)\nNet cash provided by (used in) financing activities $ 1,702 $ (3,032)\nCash Flows from Operating Activities\nNet cash provided by operating activities decreased by $2.56 billion to $8.89 billion during the nine months ended'],
#  ['position or brand.\nWe are also subject to various other legal proceedings, risks and claims that arise from the normal course of business\nactivities. For example, during the second quarter of 2023, a foreign news outlet reported that it obtained certain misappropriated data\nincluding, purportedly, among other things, non-public Tesla business and personal information. Tesla has made notifications to',
#  'Table of Contents\nTesla, Inc.\nConsolidated Statements of Operations\n(in millions, except per share data)\n(unaudited)\n Three Months Ended September 30, Nine Months Ended September 30,\n 2023 2022 2023 2022\nRevenues\nAutomotive sales $ 18,582  $ 17,785  $ 57,879  $ 46,969  \nAutomotive regulatory credits 554 286 1,357  1,309  \nAutomotive leasing 489 621 1,620  1,877  \nTotal automotive revenues 19,625  18,692  60,856  50,155  \nEnergy generation and storage 1,559  1,117 4,597  2,599',
#  'including, purportedly, among other things, non-public Tesla business and personal information. Tesla has made notifications to\npotentially affected individuals (current and former employees) and regulatory authorities and we are working with certain law\nenforcement and other authorities. On August 5, 2023, a putative class action was filed in the United States District Court for the',
#  'Table of Contents\nTesla, Inc.\nConsolidated Statements of Redeemable Noncontrolling Interests and Equity\n(in millions, except per share data)\n(unaudited)\nThree Months Ended September\n30, 2023Redeemable\nNoncontrolling\nInterestsCommon StockAdditional\nPaid-In\nCapitalAccumulated\nOther\nComprehensive\nLossRetained\nEarningsTotal\nStockholders’\nEquityNoncontrolling\nInterests in\nSubsidiariesTotal\nEquity Shares Amount\nBalance as of June 30, 2023$ 288 3,174$ 3 $33,436 $ (410)$18,101 $ 51,130 $ 764 $51,894']
# ]

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truth
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
df.to_csv('rag_gpt35_eval.csv')
print(df)