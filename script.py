import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys

# --- CONFIGURAÇÃO ---
POEM_BOOK_PATH = 'poems.txt' # Novo caminho para o arquivo TXT
MAX_FEATURES = 5000 

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO ---

def basic_clean(text):
    """Aplica limpeza básica: minúsculas, remove não-alfanuméricos (exceto espaço) e espaços extras."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Mantemos pontuação por enquanto para ajudar na tokenização de frases
    text = re.sub(r'[^a-z\s.,!?;:]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_and_pair_text(text):
    """
    Divide o texto em frases e cria pares de Query (Frase N) e Response (Frase N+1).
    Usamos regex para dividir por pontuação final (. ! ?) seguido de espaço.
    """
    # Adiciona um espaço para garantir que a pontuação não fique grudada na palavra
    text = text.replace('.', '. ').replace('!', '! ').replace('?', '? ')
    
    # Remove espaços duplicados após a correção
    text = re.sub(r'\s+', ' ', text)

    # Tokeniza em frases usando pontuações finais como delimitadores
    # O re.split usa a pontuação como delimitador, e o re.sub antes garante o espaço
    # Filtra entradas vazias
    sentences = [s.strip() for s in re.split(r'[.!?]\s*', text) if s.strip()]

    data = []
    # Cria pares de (Frase N, Frase N+1)
    for i in range(len(sentences) - 1):
        data.append({
            'Query': sentences[i],
            'Response': sentences[i+1]
        })
    return pd.DataFrame(data)

def load_and_preprocess_data(path):
    """Carrega o arquivo TXT e prepara o DataFrame de Q/A."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        print(f"Erro: Arquivo {path} não encontrado. Certifique-se de que o arquivo TXT está no mesmo diretório.")
        sys.exit(1)

    # 1. Limpeza básica do texto
    cleaned_full_text = basic_clean(full_text)
    
    # 2. Split e criação dos pares de Q/A
    df = split_and_pair_text(cleaned_full_text)
    
    # 3. Criar a coluna de Query limpa (para o TF-IDF)
    # Aqui removemos a pontuação para a vetorização ser mais eficaz
    df['Clean_Query'] = df['Query'].apply(lambda t: re.sub(r'[^\w\s]', '', t))
    
    print(f"Texto carregado e transformado em {len(df)} pares de Query/Response.")
    return df

def train_vectorizer(df):
    """Treina o vetorizador TF-IDF e transforma as queries."""
    print("Treinando o vetorizador TF-IDF...")
    # Usando a lista de stopwords em inglês do scikit-learn
    vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
    
    # Treinar e transformar o dataset de queries limpas
    tfidf_matrix = vectorizer.fit_transform(df['Clean_Query'])
    
    print(f"Matriz TF-IDF treinada. Dimensões: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix

# --- LÓGICA DO CHATBOT ---

def get_response(query, vectorizer, tfidf_matrix, df):
    """Busca a resposta mais adequada com base na Similaridade de Cosseno."""
    
    # 1. Pré-processar a query do usuário (primeira limpeza e depois remover pontuação para vetorização)
    clean_query_basic = basic_clean(query)
    clean_query_vectorized = re.sub(r'[^\w\s]', '', clean_query_basic)
    
    if not clean_query_vectorized:
        return "O eco das palavras não me alcança. Fale mais claramente."

    # 2. Transformar a query em vetor TF-IDF
    query_vector = vectorizer.transform([clean_query_vectorized])

    # 3. Calcular a Similaridade de Cosseno
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # 4. Encontrar o índice da query mais similar
    best_match_index = np.argmax(cosine_similarities)
    
    # 5. Definir um limite mínimo de similaridade
    similarity_threshold = 0.15 
    
    best_similarity = cosine_similarities[best_match_index]

    if best_similarity < similarity_threshold:
        return "Sua pergunta paira em mistério. O livro não me deu palavras para isso."
    
    # 6. Extrair a resposta (Frase N+1)
    # matched_query = df['Query'].iloc[best_match_index] # Para debug
    chatbot_response = df['Response'].iloc[best_match_index]
    
    return chatbot_response

# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO ---

def run_chatbot():
    """Executa o loop principal de conversação."""
    df = load_and_preprocess_data(POEM_BOOK_PATH)
    vectorizer, tfidf_matrix = train_vectorizer(df)
    
    print("\n" + "="*50)
    print("📜 POEM CHATBOT (Modelo de Estilo e Tom)")
    print("Tente iniciar uma frase do livro. Diga 'sair' para encerrar.")
    print("="*50 + "\n")

    # Iniciar o loop de chat
    while True:
        try:
            user_input = input("Você: ")
            
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("\n🤖 Chatbot: As palavras silenciam. Adeus. Fim da sessão.")
                break
                
            response = get_response(user_input, vectorizer, tfidf_matrix, df)
            
            # Formatar a resposta
            print(f"💬 Resposta: {response}")

        except EOFError:
            print("\n🤖 Chatbot: Fim da sessão.")
            break
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            break

# 7. Execução do Chatbot (chamada principal)
print("O novo Chatbot de Poemas está pronto. Execute o arquivo 'poem_chatbot.py' no seu ambiente local!")
# run_chatbot() # Descomente para executar localmente
