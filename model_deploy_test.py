from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

import modal


with open('data/state_of_the_union.txt') as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])

query = "What did the president say about Justice Breyer"
docs = docsearch.similarity_search(query)


chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")

base_img = modal.Image.debian_slim().pip_install(["faiss-cpu", "langchain", "openai"])


stub = modal.Stub(name="qa", image=base_img)


@stub.webhook(
	label='sotu',
	secret=modal.Secret.from_name("my-openai-secret"),
	mounts=[modal.Mount(local_dir="/Users/shreyarajpal/tiffin/data", remote_dir="/root/data")]
)
def query_sotu(query: str):
	chain_output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
	return chain_output


def query_sotu_local(query: str):
	chain_output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
	return chain_output


if __name__ == '__main__':
	print(query_sotu_local("What did the president say about Justice Breyer?"))
