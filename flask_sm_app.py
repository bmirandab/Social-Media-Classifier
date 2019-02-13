from flask import Flask
from flask import render_template, request
import pickle
import numpy as np

feature_order = ['sex', 'party', 'emplnw', 'age', 'books1', 'pial12', 'pial11', 'pial5c', 'pial5b', 'snsint2',
                 'device1b', 'device1c', 'smart2', 'intfreq', 'educ2', 'marital', 'books2a', 'books2b', 'books2c']

pkl_models = ('twitter_pkl', 'instagram_pkl', 'facebook_pkl', 'youtube_pkl')
models = {}

for pkl in pkl_models:
    with open(pkl, "rb") as f:
        models[pkl] = pickle.load(f)
        models = {k.replace('_pkl', ''): v  
         for k, v in models.items()}

with open('scaler', "rb") as f:
    scaler = pickle.load(f)


app = Flask(__name__)

def order_answers(d):
    dict = np.float_([d[name] for name in feature_order])
    return dict

def score_format(pred_answer):
    conversion=[]
    for n in pred_answer[0]:
        conversion.append("{:6.2f}".format(100*n).strip())
    return [conversion]

@app.route('/')
def home() -> str:
    return render_template("home.html")

@app.route('/survey')
def survey() -> str:
    return render_template("survey_form.html")

@app.route("/predict" , methods=["POST", "GET"])
def predict():

    data = request.form.to_dict()
    unscaled_answers = order_answers(data)
    answers = scaler.transform(unscaled_answers.reshape(1, -1))

    twt_prediction = models['twitter'].predict_proba(answers)
    inst_prediction = models['instagram'].predict_proba(answers)
    fb_prediction = models['facebook'].predict_proba(answers)
    yt_prediction = models['youtube'].predict_proba(answers)

    # yes_answers = (twt_prediction, inst_prediction, fb_prediction, yt_prediction)

    # score_format(pred_answer)

    return render_template('result.html', twitter=score_format(twt_prediction), instagram=score_format(inst_prediction), facebook=score_format(fb_prediction), youtube=score_format(yt_prediction))


@app.route('/about')
def about() -> str:
    return render_template("about.html")


    

if __name__ == '__main__':
    app.run(debug=True)

