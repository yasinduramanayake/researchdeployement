from pycaret.classification import *

model = load_model('Naive_Bayes')

Temperature = '35.00C'
Humidity = '0.91'

data = np.array([['Temperature', 'Humidity'], [Temperature, Humidity]])

result = predict_model(model, data=pd.DataFrame(data=data[0:, 0:], index=data[0:, 0], columns=data[0, 0:])).iat[1, 2]
print('Predicted result ' + str(result))
