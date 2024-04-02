from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dados de exemplo
textos = [
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",
    "Campeonato de futebol",
    "Política internacional",
    "Novo smartphone da Samsung",
    "Final do campeonato de tênis",
    "Reunião de cúpula sobre mudanças climáticas",
    "Rumores sobre novo console de videogame"
]
categorias = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política",
              "tecnologia", "esportes", "política", "tecnologia"]

# Convertendo textos em uma matriz de contagens de tokens
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.3, random_state=42)

# Treinando o classificador
clf = MultinomialNB(alpha=0.1)  # Adicionando um pequeno valor de alfa para suavização de Laplace
clf.fit(X_train, y_train)

# Predição e Avaliação
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")


"""
Existem várias técnicas de pré-processamento de texto que podem melhorar a performance do algoritmo. Aqui estão algumas delas:

1. **Remoção de pontuação e caracteres especiais:** Muitas vezes, a pontuação e caracteres especiais não são relevantes para 
a análise de texto e podem ser removidos. Isso pode ajudar a reduzir o tamanho do vocabulário e melhorar a eficiência do algoritmo.

2. **Tokenização:** Dividir o texto em tokens (palavras, frases, ou outros elementos significativos) pode ajudar a capturar melhor 
o significado do texto. Existem diferentes métodos de tokenização, como tokenização baseada em espaços em branco, tokenização baseada 
em expressões regulares, ou o uso de bibliotecas específicas para essa tarefa.

3. **Remoção de stop words:** Stop words são palavras comuns que não contribuem muito para o significado de uma frase e podem 
ser removidas com segurança. Exemplos de stop words incluem "a", "o", "em", "para", etc. A remoção de stop words pode reduzir 
a dimensionalidade dos dados e melhorar o desempenho do modelo.

4. **Stemming e lematização:** Ambas as técnicas visam reduzir as palavras às suas formas básicas. Stemming corta as palavras 
para remover sufixos e prefixos, enquanto a lematização utiliza regras gramaticais para reduzir as palavras à sua forma básica, 
chamada de "lema". Isso ajuda a tratar variações de palavras e reduzir a dimensionalidade do vocabulário.

5. **Normalização de texto:** Isso inclui a conversão de texto para minúsculas (lowercasing) para garantir que palavras escritas 
de forma diferente, mas com o mesmo significado, sejam tratadas da mesma forma. Também pode incluir a remoção de números ou 
substituição de números por uma representação comum, como "NÚMERO".

6. **Feature engineering:** Além das técnicas de pré-processamento mencionadas acima, você pode extrair recursos específicos do
 texto que podem ser úteis para o modelo. Isso pode incluir a contagem de palavras-chave relevantes, a extração de n-grams 
 (sequências de n palavras adjacentes), ou o uso de técnicas mais avançadas, como word embeddings.

7. **Tratamento de texto não estruturado:** Dependendo do tipo de texto que você está lidando, pode ser útil aplicar técnicas 
específicas de pré-processamento. Por exemplo, se estiver trabalhando com texto em redes sociais, pode ser necessário lidar com 
emojis, hashtags e menções de usuário de maneira adequada.

Ao aplicar essas técnicas de pré-processamento de texto de forma adequada, você pode melhorar a qualidade dos dados de entrada e,
 consequentemente, a performance do algoritmo de classificação de texto.

"""