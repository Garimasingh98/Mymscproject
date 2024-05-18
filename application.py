from flask import Flask,render_template,request,redirect
from flask_cors import CORS, cross_origin
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel_new.pkl','rb'))
car1=pd.read_csv("D:\FDS_PYTHON\car_price_predictor\pythonProject\Cleaned_cardetails_data.csv")
@app.route('/')
def index():
    name=sorted(car1['name'].unique())
    year=sorted(car1['year'].unique(),reverse=True)
    km_driven=sorted(car1['km_driven'].unique())
    fuel=sorted(car1['fuel'].unique())
    seats = sorted(car1['seats'].unique())
    seller_type=sorted(car1['seller_type'].unique())
    owner = sorted(car1['owner'].unique())
    transmission=sorted(car1['transmission'].unique())
    mileage=sorted(car1['mileage'].astype(str).unique())
    engine=sorted(car1['engine'].astype(str).unique())

    return render_template('index.html',
                           name=name,years=year,km_driven=km_driven,
                           fuel=fuel,seats=seats,seller_type=seller_type,
                           owner=owner,transmission=transmission,
                           mileage=mileage, engine=engine)


@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict():

    name=request.form.get('name')
    year = request.form.get('year')
    km_driven = request.form.get('km_driven')
    fuel = request.form.get('fuel')
    seats = request.form.get('seats')
    seller_type = request.form.get('seller_type')
    owner = request.form.get('owner')
    transmission = request.form.get('transmission')
    mileage = request.form.get('mileage')
    engine=request.form.get('engine')

    prediction=model.predict(pd.DataFrame(columns=['name', 'year', 'km_driven', 'fuel','seats','seller_type','owner','transmission','mileage','engine'],
                              data=np.array([name,year,km_driven,fuel,seats,seller_type,
                                             owner,transmission,mileage,engine]).reshape(1,10)))
    print(prediction)
    return str(np.round(prediction[0],2))
if __name__=="__main__":
    app.run(debug=True)