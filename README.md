# AS05: Implementação de Assistente Conversacional Baseado em LLM

Problema
Implementar um assistente conversacional baseado em LLM que seja capaz de indexar vetores (embeddings textuais) de uma coleção de documentos PDF para posteriormente responder a perguntas feitas através de uma interfce de conversação.

### Descrição Geral
Sistema de assistente conversacional baseado em LLM, utilizando bibliotecas como transformers, torch, e pinecone. Processa documentos PDF (já um conjunto pré-processado), extrai informações relevantes e responde a perguntas contextuais. Interface construída utilizando Streamlit.

### PDFs utilizados tem de origem da base de documentos das Nações Unidas. Temas de Mudança climática, Gênero e Comércio respectivamente:<br>
  
  PDF 1: Gender and climate change. Draft decision -/CP.29-Proposal by the President<br>  
  
  PDF 2: Commission on the Status of Women - 2019<br>
  
  PDF 3: Arab trade in 2023: trends and highlights<br>

O usuário pode adicionar pdfs da sua preferência, mas é necessário atentar-se que o projeto é baseado em disponibilidade gratuita do Pinecone e Hugging Faces.

Requerimentos estão citados em requirements.txt


