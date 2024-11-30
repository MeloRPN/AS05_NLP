import os
import requests
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
import torch
import streamlit as st
import PyPDF2

# VETORIZAÇÃO, MODELAGEM, Q.A
class VETORIZADORPDF:
    def __init__(self, pinecone_api_key, huggingface_api_key):
        # PINECONE - INDEX
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "embeddings-docs-pdf"

        # CASO NÃO EXISTA
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=384, 
                metric="cosine", 
                spec=ServerlessSpec(#SPECS PINECONE
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # PUXAR O ÍNDICE
        self.index = self.pc.Index(self.index_name)

        # MODELO PARA TAREFA DE EMBEDDING - TOKENIZER E MODEL
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # MODELO PARA Q.A
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=self.qa_model, 
            tokenizer=self.qa_tokenizer
        )
        
        # API KEY PARA O MODELO Q.A
        self.huggingface_api_key = huggingface_api_key
    
    # EMBEDDING - PARAMS/POOLING
    def embedder(self, text):
        with torch.no_grad():
            tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**tokens)
            embeddings = self.poolmedia(outputs, tokens['attention_mask'])#POOLING + ATTENTION
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze().numpy().tolist()
    
    # POOLING - MÉDIA / ATTENTION
    def poolmedia(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # EXTRATOR PDF
    def index_pdf(self, pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)#LEITOR
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        # CHUNKS - VELOCIDADE
        chunks = self._split_text(full_text)

        # INDEXADOR DOS CHUNKS
        upserts = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedder(chunk)
            upserts.append({
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {'text': chunk}
            })
        
        #UPSERT PINECONE
        self.index.upsert(upserts)
        return len(chunks)
    
    # SPLIT - CHUNKS = 500
    def _split_text(self, text, chunk_size=500):
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    # MÉTODO DE RESULTADO MAIS PRÓXIMOS (cosine) - top_k=2
    def query_documents(self, query, top_k=2):
        query_embedding = self.embedder(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match['metadata']['text'] for match in results['matches']]

    # MODELO DE Q.A
    def hf_qa_model(self, question, context, max_answer_length=100):
        try:
            # CHECK SE O CONTEXTO ESTÁ VAZIO
            if not context.strip():
                return "Sem contexto relevante para responder a pergunta"
            
            # FINALIZAR CONTEXTO - EVITAR ERRO
            max_context_length = 4000
            context = context[:max_context_length]
            
            # PIPELINE - QUESTION & CONTEXT
            result = self.qa_pipeline({
                'question': question,
                'context': context
            })
            #RESPOSTA
            return result['answer']

        
        except Exception as e:
            return f"ERRO: {str(e)}"

# APP NO STREAMLIT
def main():
    st.title("Assistente Conversacional Baseado em LLM")
    
    #APIS
    pinecone_api_key = "pcsk_6Zhx1S_UP7QtT6K1FRdi8uKiMuwwuEuX68BoovDBRqoHVzHQQNG3UMEjaBKTdQkW5Peyk6"
    huggingface_api_key = "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"

    if pinecone_api_key and huggingface_api_key:
        try:
            assistant = VETORIZADORPDF(pinecone_api_key, huggingface_api_key)
            
            uploaded_file = st.sidebar.file_uploader("Adicione o PDF para carregar", type=['pdf'])
            
            if uploaded_file:
                if st.sidebar.button("Indexação de PDF"):
                    try:
                        num_chunks = assistant.index_pdf(uploaded_file)
                        st.sidebar.success(f"Indexados {num_chunks} chunks")
                    except Exception as e:
                        st.sidebar.error(f"ERRO NA INDEXAÇÃO: {e}")
            
            query = st.text_input("Faça a sua pergunta sobre o documento enviado")

            if query:
                try:
                    # TRECHO MAIS PRÓXIMO EM RELEVÂNCIA
                    relevant_texts = assistant.query_documents(query)
                    context = " ".join(relevant_texts)

                    # HF MODEL
                    answer = assistant.hf_qa_model(query, context)
                    st.write(f"**Resposta produzida pelo modelo:** {answer}")
                    
                    # AVALIAR RELEVÂNCIA - CHECK
                    with st.expander("Contextos apresentados como relevantes"):
                        for i, text in enumerate(relevant_texts, 1):
                            st.text(f"Trechos {i}: {text[:500]}...")

                except Exception as e:
                    st.error(f"ERRO NA CONSULTA: {e}")
        
        except Exception as e:
            st.error(f"ERRO NA INICIALIZAÇÃO: {e}")

if __name__ == "__main__":
    main()