from flask import Flask,request,render_template,jsonify
from src.sentimentAnalysis.pipeline.predict import PredictionPipeline



app=Flask(__name__)



@app.route('/home')
def home_page():
    return render_template('index.html')


@app.route('/check_sentiment',methods=['GET','POST'])

def sentiment_check():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        
        
        text = request.form.get('text_input')
        raw_text = text
        text = " ".join(text.split())

        pred_obj = PredictionPipeline(text)


        predicted_sentiment = pred_obj.predict()

        rounded_sentiment_value = round(predicted_sentiment)
        color = ""

        if rounded_sentiment_value<=40:
            color = 'red'
        elif rounded_sentiment_value>40 and rounded_sentiment_value<=60:
            color = 'blue'
        else:
            color = 'green'

        percent_bar = str(rounded_sentiment_value)+", 100"
        print(color)
        
        return render_template('index.html',output=rounded_sentiment_value, percent_bar = percent_bar, color = color, raw_text=raw_text)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)