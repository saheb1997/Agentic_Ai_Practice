from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader =PyPDFLoader('DSA Lab Assignment - 4.pdf')
doc = loader.load()

spiter  = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    separators='',
    chunk_overlap =0
)


result =  spiter.split_documents(doc)

print(result[0])