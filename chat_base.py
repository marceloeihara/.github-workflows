from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

class ChatPDFBase:
    def __init__(self):
        self.vector_db_path = "chroma_db"
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            #"Você é um assistente que responde perguntas sobre o documento PDF enviado pelo usuário.\n\n"
            "Você é um especialista que ajuda na posologia e indica a dosagem correta sobre o medicamento no documento PDF enviado pelo usuário,\n\n"
            "Sempre forneça a dosagem e o nome do remédio está no documento PDF, se não encontrar informe o usuário,\n\n"          
            "Aqui estão os trechos relevantes do documento, responda sempre em pt-br: {context}\nPergunta: {question}"
        )
        # Inicialize o vector store Chroma com o diretório de persistência
        self.vector_store = Chroma(
                persist_directory=self.vector_db_path, embedding_function=FastEmbedEmbeddings()
            )
        self.retriever = None
        self.chain = None

    def retrieve_relevant_chunks(self, query_text):

        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path, embedding_function=FastEmbedEmbeddings()
            )

        # Busca apenas os trechos mais relevantes (k=3, por exemplo)
        results = self.vector_store.similarity_search_with_relevance_scores(query_text, k=3)
        relevant_chunks = [doc.page_content for doc, score in results if score > 0.5]
        return "\n\n---\n\n".join(relevant_chunks)

    def ingest(self, pdf_file_path: str):

        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path, embedding_function=FastEmbedEmbeddings()
            )

        # Antes da transação
        estado_atual_embeddings = len(self.vector_store.get()['documents'])
        print('Total de embeddings registrados no ChromaDB:', estado_atual_embeddings)

        unique_id = f"{pdf_file_path}"  # Cria um ID único baseado no nome do arquivo e na posição do chunk
        '''
        CUIDADO: Se um arquivo, mesmo que diferente, tiver o mesmo nome (pdf_file_path), os embeddings não serão persistidos.
        Da mesma forma, se um mesmo arquivo tiver nomes distintos (em diretórios distintos), ele irá armazenar os embeddings.
        Entenda que o uso do ID é um mecanismo que visa reduzir a redundancia de registros, mas que precisa avaliar os riscos. 
        '''
        # Inserção de novos embeddings
        self.vectorstore = Chroma.from_documents(
            documents=chunks, embedding=FastEmbedEmbeddings(),
            persist_directory=self.vector_db_path, ids=unique_id
        )

        # Depois da transação
        estado_novo_embeddings = len(self.vector_store.get()['documents'])

        # Embeddings adicionados na transação
        adicionados = estado_novo_embeddings - estado_atual_embeddings

        print('Novos embeddings adicionados ao ChromaDB:',adicionados)

    def ask_com_db(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path, embedding_function=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        context = self.retrieve_relevant_chunks(query)
        prompt = self.prompt.format(context=context, question=query)
        return self.generate_response(prompt)

    def generate_response(self, prompt):
        """Método abstrato para gerar resposta. Deve ser implementado pelas subclasses."""
        raise NotImplementedError("Subclasses devem implementar este método.")

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None