from flask import Flask, render_template, request
import pickle

# Load the Random Forest model and CountVectorizer object from disk
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv_transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = model.predict(vect)[0]
    	return render_template('index.html', msg=my_prediction)

if __name__ == '__main__':
	app.run(debug = True)