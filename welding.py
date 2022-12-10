import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor

#app
app = Flask(__name__, template_folder='templates')

# load scaler and model
try:
    scf = open('scaler_X.pkl', 'rb')
except IOError:
    print('No "scaler_X.pkl" file')
    exit(1)
scaler = pickle.load(scf)
try:
    mf = open('erfr_model.pkl', 'rb')
except IOError:
    print('No "erfr_model.pkl" file')
    exit (1)
model = pickle.load(mf)



# frontend
@app.route('/', methods=['POST', 'GET'])
def main():
    res = ""
    if request.method == 'GET':
        return render_template('main.html',result=res)

    if request.method == 'POST':
        # get welding parameters
        x = []
        IW = request.form['IW']
        IF = request.form['IF']
        WV = request.form['WV']
        FP = request.form['FP']

        # build vector for predict
        x.append(float(IW))
        x.append(float(IF))
        x.append(float(WV))
        x.append(float(FP))
        #print(x)

        # scaling vector for predict
        x_scaled = scaler.transform([x])

        prediction = model.predict(x_scaled)
        #print(prediction)
        res = 'Глубина: {}\nШирина: {}'.format(prediction[0,0].round(4), prediction[0,1].round(4))

        return render_template('main.html', result=res)

#backend
@app.route('/api/msg/<uuid>', methods=['GET', 'POST'])
def msg(uuid):
    x = request.json
    #print(x['params'])

    x_scaled = scaler.transform([x['params']])
    prediction = model.predict(x_scaled)
    #print(prediction, type(prediction))
    return jsonify({uuid:str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=False, host = '0.0.0.0')