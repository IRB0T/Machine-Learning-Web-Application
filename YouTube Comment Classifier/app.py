import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from flask import Flask,render_template,url_for,request
data= pd.read_csv("Data/Combined_data.csv")
data_y = data["CLASS"]
corpus = []
Lemmatizer = WordNetLemmatizer()
for i in range(0,len(data)):
    text = re.sub('[^a-zA-Z]', ' ', data["CONTENT"][i])
    text = text.lower()
    text = text.split()
    text = [Lemmatizer.lemmatize(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
c = TfidfVectorizer(ngram_range=(2,2))
X = c.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(X, data_y, test_size=0.20, random_state=20)
model = MultinomialNB()
model.fit(X_train,y_train)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	ans=''
	if request.method == 'POST':
		msg = request.form['msg']
		text = re.sub('[^a-zA-Z]', ' ', msg)
		text = text.lower()
		text = text.split()
		text = [Lemmatizer.lemmatize(word) for word in text if not word in stopwords.words('english')]
		text = ' '.join(text)
		vect = c.transform([text])	
		if model.predict(vect) == 1:
			ans = "Spam"
		else:
			ans = "Ham (Not Spam)"
	print(ans)
	return render_template('result.html',answer=ans)

if __name__ == '__main__':
	app.run(debug=True)