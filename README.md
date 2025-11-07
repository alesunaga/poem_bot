📜 PoemBot: 
  Chatbot de Estilo Poético com Processamento de Linguagem NaturalEste projeto demonstra a criação de um chatbot simples, baseado em recuperação de informação (retrieval-based), treinado em um corpus de texto não-estruturado (simulando um livro de poemas).O objetivo é fazer com que o bot absorva o tom e o vocabulário do texto de treinamento, gerando respostas que parecem uma continuação estilística do conteúdo original.
  
  🧠 Conceitos de NLP AplicadosPré-processamento de Texto: O texto do livro é limpo, padronizado e dividido em frases.Segmentação de Frases (Q/A Pairing): O texto é transformado em pares de diálogo, onde a Frase N é considerada a Query (pergunta do usuário) e a Frase N+1 é a Response (resposta do bot).Word Vectorization: O TF-IDF (Term Frequency-Inverse Document Frequency) é usado para transformar as queries em vetores numéricos, permitindo que a máquina entenda o significado semântico das palavras.Checking Similarity: A Similaridade de Cosseno é calculada para encontrar a query de treinamento mais próxima da entrada do usuário.⚙️ RequisitosPara executar o projeto, você precisará ter o Python 3.x e as seguintes bibliotecas instaladas:pip install pandas scikit-learn numpy

📂 Estrutura do Projeto
  O projeto é composto por dois arquivos essenciais:poem_chatbot.py: O código principal do chatbot, responsável por carregar os dados, treinar o modelo e gerenciar o loop de conversação.book_of_poems.txt: O corpus de texto para treinamento. Você pode substituir este arquivo por qualquer livro, roteiro ou documento extenso (.txt) que queira usar para dar personalidade ao seu bot.🚀 Como Executar o ChatbotClone o Repositório:git clone [SEU_REPOSITÓRIO]
cd [SEU_REPOSITÓRIO]

  Execute o Script:python poem_chatbot.py
  Interaja: O terminal iniciará a sessão de chat. Tente digitar uma frase ou uma palavra-chave relacionada ao conteúdo do book_of_poems.txt para ver a resposta estilística.Exemplo de saída:==================================================

📜 POEM CHATBOT (Modelo de Estilo e Tom)
  Tente iniciar uma frase do livro. Diga 'sair' para encerrar.
  
==================================================

  Você: The wind whispers secrets
💬 Resposta: it tells tales of forgotten journeys and empty ships


🛠️ Detalhes de Implementação (poem_chatbot.py)split_and_pair_text(text): Esta função crucial manipula o texto contínuo, dividindo-o em sentenças e criando os pares de (Query, Response) com base na sequência de ocorrência no arquivo.train_vectorizer(df): Utiliza TfidfVectorizer com stopwords em inglês para gerar uma matriz de recursos de 5000 dimensões que representa todas as queries possíveis.get_response(query, ...): Calcula a Similaridade de Cosseno e usa um similarity_threshold de 0.15 para garantir que apenas respostas contextualmente relevantes sejam retornadas.Desenvolvido por: [Seu Nome]Inspiração: Projeto Pessoal de NLP (Processamento de Linguagem Natural)
