import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from flask import Flask,render_template,url_for,request

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	data= pd.read_csv("Data/Combined_data.csv")
	data_x = data["CONTENT"]
	data_y = data["CLASS"]
	corpus = data_x
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(corpus)
	X_train, X_test, y_train, y_test = train_test_split(X, data_y, test_size=0.20, random_state=20)
	model = MultinomialNB()
	model.fit(X_train,y_train)
	model.score(X_test,y_test)
	ans=''
	if request.method == 'POST':
		msg = request.form['msg']
		data = [msg]
		vect = vectorizer.transform(data).toarray()
		if model.predict(vect) == 1:
			ans = "Spam"
		else:
			ans = "Ham (Not Spam)"
	print(ans)
	return render_template('result.html',answer=ans)

if __name__ == '__main__':
	app.run(debug=True)