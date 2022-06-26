import pickle
import numpy as np
import sklearn
import pandas as ss
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
app=Flask(__name__)
@app.route('/')
def home():
    ddd = ss.read_csv('https://www.stats.govt.nz/assets/Uploads/Serious-injury-outcome-indicators/Serious-injury-outcome-indicators-2000-20/Download-data/serious-injury-outcome-indicators-2000-2020-CSV.csv')
    ddd['Validated'] = ddd['Validation'] == 'Validated'
    ddd['Provisional'] = ddd['Validation'] == 'Provisional'
    k = ddd[['Data_value', 'Lower_CI', 'Upper_CI', 'Validated', 'Provisional']].values
    l = ddd['Severity'].values
    model1 = RandomForestClassifier(n_estimators=20, random_state=111)
    model1.fit(k,l)
    print(model1.score(k,l))
    with open('model1.pkl', 'wb') as file:
        pickle.dump(model1, file)
    return render_template('home.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    Data_value=request.form['Data_value']
    Lower_CI=request.form['Lower_CI']
    Upper_CI=request.form['Upper_CI']
    Validated=request.form['Validated']
    #nparry1=np.array([[Data_value,Lower_CI,Upper_CI]],dtype='O')
    #nparry2=np.array([[Validated,Provisional]],dtype='U4')
    #form_array=np.hstack([nparry1, nparry2])
    model1=pickle.load(open('model1.pkl','rb'))
    #prediction=model1.predict([[form_array]])
    if Validated == 'True':
        (f) = True
        (ff) = False
        prediction=model1.predict([[Data_value,Lower_CI,Upper_CI,f,ff]])
    else:
        (f) = False
        (ff) = True
        prediction = model1.predict([[Data_value, Lower_CI, Upper_CI, f, ff]])
    if prediction =='Fatal':
        result='Fatal'
        mssg='Go to floor 2 room no 003'
    elif prediction =='Serious':
        result='Serious'
        mssg='Go to floor 3 room no 102'
    else:
        result='Serious but non Fatal'
        mssg='Go to floor 4 room no 112'
    return render_template('result.html',result=result,mssg=mssg)

if __name__=='__main__':
    app.run(debug=True)