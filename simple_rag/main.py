import os

from llama_index.core import (
	Settings,
	SimpleDirectoryReader,
	VectorStoreIndex,
	get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

from simple_rag.utils import get_project_path

google_api_key = os.getenv("GOOGLE_API_KEY")

# Setting global parameter
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Gemini(model_name="models/gemini-pro")

documents = SimpleDirectoryReader(
	input_files=[get_project_path("documents/2307.09288v2.pdf")]
).load_data()

# Transformations
# SentenceSplitter attempts to split text while respecting the boundaries of sentences
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
Settings.text_splitter = text_splitter

# Generating Index
index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])

# Storing
# index.storage_context.persist(persist_dir="/index")

# Querying
template = """
You are a knowledgeable and precise assistant specialized in question-answering tasks, 
particularly from academic and research-based sources. 
Your goal is to provide accurate, concise, and contextually relevant answers based on the given information.

Instructions:

Comprehension and Accuracy: Carefully read and comprehend the provided context from the research paper to ensure accuracy in your response.
Conciseness: Deliver the answer in no more than three sentences, ensuring it is concise and directly addresses the question.
Truthfulness: If the context does not provide enough information to answer the question, clearly state, "I don't know."
Contextual Relevance: Ensure your answer is well-supported by the retrieved context and does not include any information beyond what is provided.

Remember if no context is provided please say you don't know the answer
Here is the question and context for you to work with:

\nQuestion: {question} \nContext: {context} \nAnswer:"""

prompt_tmpl = PromptTemplate(
	template=template,
	template_var_mappings={"query_str": "question", "context_str": "context"},
)

# configure retriever
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

# configure response sythesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
	retriever=retriever, response_synthesizer=response_synthesizer
)

query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_tmpl})

response = query_engine.query("What are different variants of LLama?")
print(response)
