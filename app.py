from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
import io
import pickle
import base64
from io import StringIO
import base64
import seaborn as sns
from io import BytesIO



app = Flask(__name__, template_folder='C:\\Users\\lizpa\\OneDrive\\Desktop\\Crime-Rate-Prediction-main\\template')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'mysecretkey'

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('signup'))
        else:
            new_user = User(username=username, email=email, password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))


# Loading the model and data
model = pickle.load(open('./model/crime_model.pkl', 'rb'))
data = pd.read_csv('./static/crime_data.csv')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if request.method == 'POST':
            file = request.files['csv_file']
            if file:
                stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
                csv_input = csv.reader(stream)
                data = []
                for row in csv_input:
                    data.append(row)
                
                # Convert data to a pandas dataframe and drop null values
                df = pd.DataFrame(data[1:], columns=data[0])
                # df_cleaned = df.dropna(inplace=True)
                # print(df_cleaned)
                
                # Get the selected target feature and plot a line graph against other features
                target_feature = request.form.get('target_feature')
                if target_feature:
                    # Filter out non-numeric columns
                    numeric_cols = df.select_dtypes(include='number').columns
                    
                    # Filter out the target feature
                    numeric_cols = numeric_cols.drop(target_feature)
                    
                    # Plot a line graph for each numeric column
                    fig, ax = plt.subplots()
                    for col in numeric_cols:
                        ax.plot(df[target_feature], df[col], label=col)
                    
                    ax.set_xlabel(target_feature)
                    ax.set_ylabel('Values')
                    ax.set_title(f'Line graph for {target_feature}')
                    ax.legend()

                    # Save the plot as a base64 string
                    img = io.BytesIO()
                    fig.savefig(img, format='png')
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

                    

                    return render_template('dashboard.html', user=user, data=data, plot_url=plot_url)
                
                return render_template('dashboard.html', user=user, data=data)
        
        return render_template('dashboard.html', user=user)
    
    else:
        return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    # Preprocessing user input
    features = [float(x) for x in request.form.values()]
    input_data = np.array([features])

    # Predicting the cluster
    cluster = model.predict(input_data)[0]

    # Filtering the data for the predicted cluster
    filtered_data = data[model.labels_ == cluster]

    # Calculating the mean crime rates for the predicted cluster
    mean_murder_rate = filtered_data['Murder'].mean()
    mean_year_rate = filtered_data['Year'].mean()
    mean_location_rate = filtered_data['Location'].mean()
    mean_theft_rate = filtered_data['Theft'].mean()
    mean_kidnap_rate = filtered_data['Kidnap'].mean()

    # Generating the histogram
    fig, ax = plt.subplots()
    sns.histplot(data=model.labels_, ax=ax)
    image_histogram = plot_to_img(fig)

    # Create a list of mean rates
    mean_rates = [mean_murder_rate, mean_year_rate, mean_location_rate, mean_theft_rate, mean_kidnap_rate]

    fig, ax = plt.subplots()
    ax.pie([mean_murder_rate, mean_year_rate, mean_location_rate, mean_theft_rate, mean_kidnap_rate], labels=['Murder', 'Year', 'Location', 'Theft', 'Kidnap'])
    ax.set_title('Mean Rates')
    image_piechart = plot_to_img(fig)

    # Converting images to base64 strings
    encoded_histogram = base64.b64encode(image_histogram).decode('utf-8')
    encoded_piechart = base64.b64encode(image_piechart).decode('utf-8')

    # Rendering the result
    return render_template('predict.html', prediction_text='The predicted cluster is {}. \nThe murder rate is {:.2f} \n followed by the Cybercrime rate which is {:.2f}, \nThen the probable Assualt rate which is {:.2f} \n followed by the Theft rate which is {:.2f}. \nThen lastly, the Kidnapping rate which is {:.2f}'.format(cluster, mean_murder_rate, mean_year_rate, mean_location_rate, mean_theft_rate, mean_kidnap_rate), 
                           image_histogram=encoded_histogram, image_piechart=encoded_piechart)

def plot_to_img(fig):
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    return img_bytes.getvalue()

# load crime data and train the linear regression models
crime_data = pd.read_csv("crime.csv")
X = crime_data["year"].values.reshape(-1, 1)
y_theft = crime_data["theft "].values.reshape(-1, 1)
y_assault = crime_data["assault"].values.reshape(-1, 1)
y_murder = crime_data["murder"].values.reshape(-1, 1)

X_train, _, y_theft_train, _, y_assault_train, _, y_murder_train, _ = train_test_split(X, y_theft, y_assault, y_murder, test_size=0.2, random_state=42)
model_theft = LinearRegression()
model_assault = LinearRegression()
model_murder = LinearRegression()

model_theft.fit(X_train, y_theft_train)
model_assault.fit(X_train, y_assault_train)
model_murder.fit(X_train, y_murder_train)


# define a Flask route for the homepage
@app.route("/predictc", methods=["GET", "POST"])
def predictc():
    theft_pred = None
    assault_pred = None
    murder_pred = None
    plot = None
    if request.method == "POST":
        year = int(request.form["year"])
        theft_pred = int(model_theft.predict([[year]])[0][0])
        assault_pred = int(model_assault.predict([[year]])[0][0])
        murder_pred = int(model_assault.predict([[year]])[0][0])

    plot = plot_distribution()
    return render_template("predictc.html", theft_pred=theft_pred, assault_pred=assault_pred, murder_pred=murder_pred, year=request.form["year"] if request.form else None, plot=plot)

# define a function to generate a plot of the distribution of theft and assault across years
def plot_distribution():
    fig, ax = plt.subplots()
    ax.plot(crime_data["year"], crime_data["theft "], label="theft ")
    ax.plot(crime_data["year"], crime_data["assault"], label="assault")
    ax.plot(crime_data["year"], crime_data["murder"], label="murder")

    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Crimes")
    ax.set_title("Distribution of Theft, Assault and Murder Across Years")
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8").replace("\n", "")
    return image_base64

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
