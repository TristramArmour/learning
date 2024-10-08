from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI,OpenAI
from langchain.document_loaders import WebBaseLoader,PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
import shutil
import os

class RAG:
    def __init__(self,project_name="default_name",config=None,chromadb_folder="./chroma_db",chunkSize=2000,overlap=200) -> None:
        # save config settings to connect to Azure 4o-mini
        self.config = config
        # save project_name
        self.project_name = project_name
        # chromadb path
        self.chromadb_folder = chromadb_folder
        ###### some constants
        # chunksize
        self.chunksize = chunkSize
        # overlap
        self.overlap = overlap
        # embedding model
        # self.model_id = "BAAI/bge-base-en-v1.5" - although this is meant to be really good, doesn't really perform very well!
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        # Set USER_AGENT environment variable
        os.environ["USER_AGENT"] = "myRAGClass/1.0 - " + self.project_name
        # set the openai api keys
        os.environ["OPENAI_API_KEY"] = self.config["api_key"]
        os.environ["OPENAI_API_VERSION"] = self.config["api_version"]
        os.environ["AZURE_OPENAI_ENDPOINT"] = self.config["api_base"]

        # create embedding function
        self.createEmbeddingFunction()

        # create reference to ChromaDB
        self.vectorstore = Chroma(persist_directory=self.chromadb_folder, embedding_function=self.embedding_function)

        # function to split the text into chuncks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunksize, chunk_overlap=self.overlap)

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def add_split_text(self,split_doc):
        # Create a unique index on the chunks
        last_page_id = None
        current_chunk_index = 0

        for chunk in split_doc:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            if page == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            last_page_id = page

            chunk_id = f"{source}:{page}:{current_chunk_index}"
            # add chunk id
            chunk.metadata["id"] = chunk_id
        
        # add chunk if it does not exist
        existing_items = self.vectorstore.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # only add documents that don't exist
        new_chunks = []
        for chunk in split_doc:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        if len(new_chunks) > 0:
            self.vectorstore.add_documents(new_chunks,ids=new_chunk_ids)
        else:
            print("Document was in the database already")


    def add_documents(self,docs):
        # loop through references, check whether it is a website, pdf or docx
        for doc in docs:
            if doc.lower().endswith(".pdf"):
                self.add_pdf(doc)
            elif doc.lower().endswith(".docx"):
                self.add_docx(doc)
            elif doc.startswith("http"):
                self.add_website(doc)
            else:
                # print out message that we couldn't add the document
                print(f"Could not add {doc} to the database")
        
    def add_website(self,webpath):
        # load website
        loader = WebBaseLoader(web_paths=(webpath,))
        loader.requests_kwargs = {"verify": False}
        docs = loader.load()
        # split the text into chuncks
        splits = self.text_splitter.split_documents(docs)
        # add the split text to the database
        self.add_split_text(splits)
        print(f"Added {webpath} to the database")

    def add_pdf(self,pdfpath):
        # load pdf
        loader = PyPDFLoader(pdfpath)
        doc = loader.load()
        # split the text into chuncks
        splits = self.text_splitter.split_documents(doc)
        # add the split text to the database
        self.add_split_text(splits)
        print(f"Added {pdfpath} to the database")

    def add_docx(self,filepath):
        doc = Docx2txtLoader(filepath).load()
        # split the text into chuncks
        splits = self.text_splitter.split_documents(doc)
        # add the split text to the database
        self.add_split_text(splits)
        print(f"Added {filepath} to the database")


    def createEmbeddingFunction(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.model_id, 
                                                        model_kwargs={'device': 'cpu',
                                                                      "trust_remote_code":True},
                                                        encode_kwargs = {"max_seq_length": 512})


    def clearDatabase(self):
        # remove the existing RAG database
        shutil.rmtree(self.chromadb_folder)

    def setupLLM(self,local=False,k=10):
        self.prompt = hub.pull("rlm/rag-prompt",api_key=self.config["langsmith_key"])
        self.k = k
        # get retriever
        self.retriever = self.vectorstore.as_retriever(k=self.k)

        if local == True:
            self.llm = OpenAI(base_url="http://localhost:1234/v1", 
                              api_key="not-needed",
                              max_tokens=6000,
                              temperature=0,
                              max_retries=2)
        else:
            self.llm = AzureChatOpenAI(
                deployment_name=self.config["deployment"],
                model_name=self.config["deployment"], 
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2)

        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser())
    
    def invoke(self,question):
        print("invoking...")
        result = self.rag_chain.invoke(question)
        print(result)
        print("Complete")
        return result
