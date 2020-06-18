from flask import Flask, render_template, request, jsonify
import Home.flask_helper_predictor as tp
app = Flask(__name__)

@app.route('/')
def home():
    print('rendering home')
    return render_template('input.html')
    print('rendering home finished')

@app.route('/predict', methods=['GET', 'POST'])
def my_form_post():
    text = request.form['text1']
    processed_text = predictor.predict(text)
    print(processed_text)
    result = {
        "output": processed_text
    }
    result = {str(key): value for key, value in result.items()}
    return(jsonify(result= result))


if __name__ == '__main__':
    print('started main')
    predictor = tp.sentiment_analysis()
    print('created our model successfully')
    app.run(debug = True, host='0.0.0.0', port=5000)

