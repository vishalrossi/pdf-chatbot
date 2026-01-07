from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from databricks.vector_search.client import VectorSearchClient

def load_chain(env):
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name="pdf-vector-search",
        index_name=f"pdf_chatbot_{env}"
    )

    retriever = index.as_langchain_retriever(k=3)
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )