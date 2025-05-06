from flask import Flask, request, jsonify, render_template
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample news articles DataFrame
articles = pd.DataFrame({
    "title": [
        "Zimbabwe's Economic Growth: A New Dawn",
        "Sports Update: Zimbabwe Wins Regional Championship",
        "Technology in Zimbabwe: Innovations and Startups",
        "Politics in Zimbabwe: A Year of Change",
        "Wildlife Conservation Efforts in Zimbabwe"
    ],
    "text": [
        "Zimbabwe's economy shows signs of recovery with new policies.",
        "The national football team clinched the championship title last night.",
        "Tech startups are emerging in Harare, driving innovation.",
        "Political shifts are reshaping the landscape in Zimbabwe.",
        "Conservationists are working hard to protect endangered species."
    ]
})

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

# Preprocess articles
articles["text"] = articles["text"].apply(preprocess_text)

# Create TF-IDF vectorizer and compute similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(articles["text"])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Get recommendations
def get_recommendations(article_index, num_recommendations):
    similarity_scores = list(enumerate(similarity_matrix[article_index]))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return similarity_scores[1:num_recommendations + 1]

@app.route('/')
def home():
    return render_template('index.html', articles=articles.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    article_index = data['article_index']
    num_recommendations = data.get('num_recommendations', 2)
    recommendations = get_recommendations(article_index, num_recommendations)
    
    results = [{'title': articles.iloc[index]['title'], 'similarity_score': score} for index, score in recommendations]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)