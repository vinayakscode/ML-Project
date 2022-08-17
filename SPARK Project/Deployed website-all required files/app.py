import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model/Landslide_predict1.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Cummulative_rainfall = flask.request.form['Cummulative_rainfall']
        Intensity = flask.request.form['Intensity']
        Product = flask.request.form['Product']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[Cummulative_rainfall, Intensity, Product]],
                                       columns=['Cummulative_rainfall', 'Intensity', 'Product'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Cummulative_rainfall':Cummulative_rainfall,
                                                     'Intensity':Intensity,
                                                     'Product':Product},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
