import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
from datetime import datetime

selectDataset = st.sidebar.selectbox ("Select Dataset", options = ["Home", "Forex", "Stock","Commodity","Cryptocurrency","Futures"])

if selectDataset == "Home":

    st.title("Group Project CSC649")
    st.header("Market Price Prediction")
    st.header("Looking to make smart investment decisions in the ever-changing financial landscape? \n")
    st.write("Discover the ultimate tool that can take your investment strategies to the next level. \n Introducing our state-of-the-art prediction model designed to help you navigate the complexities of forex, stocks, commodities, cryptocurrencies, and futures markets with confidence.")
    image = Image.open('image/banner.png')

    st.image(image, caption='Our cutting-edge prediction model leverages the latest advancements in artificial intelligence and machine learning. Powered by robust algorithms and historical market data, it analyzes trends, patterns, and price movements to make data-driven forecasts with impressive accuracy.')
    

    image = Image.open('image/team.png')

    st.image(image, caption='our team')

    st.header("The key to successful investing lies in being one step ahead. Our prediction model empowers you with the knowledge and confidence to execute trades strategically, minimizing risks, and maximizing potential gains.")

#FOREX PRICE
if selectDataset == "Forex":
    
    st.subheader("Full dataset for Forex")
    # Load the full dataset
    forex_dataset = pd.read_csv('eurusd_hour.csv')

    # Determine the fraction of the data you want to keep
    fraction_to_keep = 8000 / len(forex_dataset)

    # Randomly sample a subset of the data
    sampled_df = forex_dataset.sample(frac=fraction_to_keep, random_state=42)

    # Split the data into input features (X) and target variable (y)
    X = sampled_df[['BH', 'BO', 'BL', 'AO', 'AH', 'AL', 'YEAR', 'MONTH', 'DAY', 'HOUR']]
    y = sampled_df['BC']


    st.subheader("Training and testing data will be divided using Train_Test_Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    X_train
    st.write("Training Data Target")
    y_train

    st.subheader("Testing data for input and target")
    st.write("Training Data Input")
    X_test
    st.write("Training Data Target")
    y_test

#Algorithm Selection
    selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest","Prediction"])

#RANDOM FOREST
    if selectModel == "Random Forest":

        st.subheader("Random Forest age estimation model")
        #List of number estimators
        estimators = [15, 25, 50, 100]
        for n in estimators:
            st.subheader("- - - - -")
            st.write("N Estimator =",n)            

            rf = RandomForestRegressor (n_estimators = n, random_state = 0)
            st.write("Training the Model...")
            rf.fit (X_train, y_train)

            st.write("Successfully Train the model")
            outputPredictedRF = rf.predict(X_test)
            st.write("Predicted result for Testing Dataset: ")
            outputPredictedRF

            MSERF = mean_squared_error (y_test,outputPredictedRF)
            st.write("The mean Squared Error Produced by n_estimator:",n,"=", MSERF)
            rc= np.round(rf.score(X_test, y_test),2)*100
            st.write("Accuracy score:",n,"=", rc)
            from sklearn.metrics import r2_score
            r2=np.round(r2_score(y_test,outputPredictedRF),2)
            st.write("R2 score:",n,"=", r2)

    elif selectModel == "Prediction":
        # Split the data into input features (X) and target variable (y)
        X = sampled_df[['BH', 'BO', 'BL', 'AO', 'AH', 'AL', 'YEAR', 'MONTH', 'DAY', 'HOUR']]
        y = sampled_df['BC']

        # Train the Random Forest model
        rf = RandomForestRegressor(n_estimators=50, random_state=0)
        rf.fit(X, y)
        st.write("Successfully Train the model")
        outputPredictedRF = rf.predict(X_test)
        st.write("Predicted result for Testing Dataset: ")
        outputPredictedRF
        MSERF = mean_squared_error (y_test,outputPredictedRF)
        st.write("The mean Squared Error Produced by n_estimator:=", MSERF)
        rc= np.round(rf.score(X_test, y_test),2)*100
        st.write("Accuracy score:=", rc)
        from sklearn.metrics import r2_score
        r2=np.round(r2_score(y_test,outputPredictedRF),2)
        st.write("R2 score:=", r2)

        # Create X_test using the same columns as X (for user input)
        X_test = sampled_df[['BH', 'BO', 'BL', 'AO', 'AH', 'AL','YEAR', 'MONTH', 'DAY', 'HOUR']]

        # Streamlit app
        st.subheader("Forex Close Price Prediction")
        st.write("Enter the details below to predict the close price:")

        year = st.text_input("Year (integer)", value=str(int(X_test['YEAR'].mean())))
        month = st.text_input("Month (integer)", value=str(int(X_test['MONTH'].mean())))
        day = st.text_input("Day (integer)", value=str(int(X_test['DAY'].mean())))
        hour = st.text_input("Hour (integer)", value=str(int(X_test['HOUR'].mean())))
        bh = st.slider("Highest Bid Price in that one hour period (BH)", min_value=float(X_test['BH'].min()), max_value=float(X_test['BH'].max()), key="bh_slider")
        bo = st.slider("Opening Bid Price (BO)", min_value=float(X_test['BO'].min()), max_value=float(X_test['BO'].max()), key="bo_slider")
        bl = st.slider("Lowest Bid Price in that one hour period (BL)", min_value=float(X_test['BL'].min()), max_value=float(X_test['BL'].max()), key="bl_slider")
        ao = st.slider("Opening ask price (AO)", min_value=float(X_test['AO'].min()), max_value=float(X_test['AO'].max()), key="ao_slider")
        ah = st.slider("Highest ask price in that one hour period (AH)", min_value=float(X_test['AH'].min()), max_value=float(X_test['AH'].max()), key="ah_slider")
        al = st.slider("Lowest ask price in that one hour period (AL)", min_value=float(X_test['AL'].min()), max_value=float(X_test['AL'].max()), key="al_slider")



        # Create a new input data point with user input
        new_data = pd.DataFrame({
            'BH': bh,
            'BO': bo,
            'BL': bl,
            'AO': ao,
            'AH': ah,
            'AL': al,
            'YEAR': int(year),
            'MONTH': int(month),
            'DAY': int(day),
            'HOUR': int(hour)
        }, index=[0])

        # Predict function
        def predict_close_price():
            # Make predictions using the trained model
            predicted_close = rf.predict(new_data)
            return predicted_close

        # Predict button
        if st.button("Predict"):
            predicted_close_price = predict_close_price()
            st.subheader("Predicted Close Price")
            st.write(predicted_close_price)

#KNN
    elif selectModel == "K-Nearest Neighbors":
        
    
        n_neighbors_list = [10, 15, 20, 100]

        for n_neighbors in n_neighbors_list:
            st.subheader(f"K-Nearest Neighbors age estimation model (n_neighbors = {n_neighbors})")
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            st.write("Training the Model...")
            knn.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            output_predicted_knn = knn.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            st.write(output_predicted_knn)

            MSE_knn = mean_squared_error(y_test,output_predicted_knn)
            st.write(f"The Mean Squared Error produced by KNN with number of nearest neighbors {n_neighbors}: ", MSE_knn)
            sc = np.round(knn.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
            from sklearn.metrics import r2_score
            knnr2=np.round(r2_score(y_test,output_predicted_knn),2)
            st.write("R2 score:",n_neighbors,"=", knnr2)

#SVM
    elif selectModel == "Support Vector Machine":

        selectKernel = st.sidebar.selectbox ("Select Kernel", options = ["RBF", "Linear", "Sigmoid", "Poly"])

        if selectKernel == "RBF":

            st.subheader("Support Vector Machine age estimation model")
            st.write(" ")
            st.subheader("RBF")
            svm_model = SVR(kernel="rbf")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "rbf", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for RBF Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "rbf" , svm)
            
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

            st.write(" ")

        elif selectKernel == "Linear":
            st.subheader("Linear")
            svm_model = SVR(kernel="linear")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "linear", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for linear Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "linear" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)


            st.write(" ")

        elif selectKernel == "Poly":
            st.subheader("Poly")
            svm_model = SVR(kernel="poly")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "poly", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for poly Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "poly" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

            st.write(" ")
        elif selectKernel == "Sigmoid":
            st.subheader("Sigmoid")
            svm_model = SVR(kernel="sigmoid")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "sigmoid", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for sigmoid Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "sigmoid", svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
        


#STOCK PRICE
elif selectDataset == "Stock":
    st.subheader("Description of the dataset")
    with st.columns(3)[1]:
        st.image("image/amazon.jpg", width=400)
    text = """
    <style>
    .justify-text {
        text-align: justify;
    }
    </style>

    <div class="justify-text">
    There are 7 total columns and 851264 total rows in the dataset. The data was gathered for attributes including the date,symbol,open,close,low,high and volume. As the dataset consist of many stocks,this system will only sample an Amazon stock based on the symbol AMZN that takes 1762 total rows of data. This dataset has been meticulously curated, ensuring data accuracy, and can be utilized for a wide range of applications, including stock market analysis, time-series forecasting, trend identification, and machine learning models for predicting Amazon's stock prices in the future. The closing price of a stock is determined by its price at the end of the trading day, as opposed to the Close. The closing price, on the other hand, determines value by taking into account variables like dividends, stock splits, and new stock offerings. Therefore, Close is the outcome variable, and its value needs to be predicted.
    </div>
    """
    
    st.write(text, unsafe_allow_html=True)

    st.subheader("Full dataset for Stock")
    #your dataset
    stock = pd.read_csv('prices.csv',na_values=['null'],index_col='date',parse_dates=True,infer_datetime_format=True)
    stock

    #sampling amazon dataset
    st.write("Sample amazon dataset")
    stock1 = stock[stock['symbol']=='AMZN']
    stock1

    st.subheader("Data input for stock")
    data_input_training = stock1.drop(columns = ["symbol", "close"])
    data_input_training

    st.subheader("Data target for stock")
    data_target_training = stock1['close']
    data_target_training

    st.subheader("Training and testing data will be divided using Train_Test_Split")
    X = data_input_training
    y = data_target_training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =scaler.transform(X_test)

    st.subheader("Training data for input and target")
    st.write("Training Data Input")

    X_train
    st.write("Training Data Target")
    y_train

    st.subheader("Testing data for input and target")
    st.write("Testing Data Input")
    X_test
    st.write("Testing Data Target")
    y_test

#Algorithm selection
    selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])

#RANDOM FOREST
    if selectModel == "Random Forest":

        st.subheader("Random Forest age estimation model")
        #List of number estimators
        estimators = [15, 25, 50, 100]
        for n in estimators:
            st.subheader("- - - - -")
            st.write("N Estimator =",n)            

            rf = RandomForestRegressor (n_estimators = n, random_state = 0)
            st.write("Training the Model...")
            rf.fit (X_train, y_train)

            st.write("Successfully Train the model")
            outputPredictedRF = rf.predict(X_test)
            st.write("Predicted result for Testing Dataset: ")
            outputPredictedRF

            MSERF = mean_squared_error (y_test,outputPredictedRF)
            st.write("The mean Squared Error Produced by n_estimator:",n,"=", MSERF)
            rc= np.round(rf.score(X_test, y_test),2)*100
            st.write("Accuracy score:",n,"=", rc)
            from sklearn.metrics import r2_score
            r2=np.round(r2_score(y_test,outputPredictedRF),2)
            st.write("R2 score:",n,"=", r2)

#KNN
    elif selectModel == "K-Nearest Neighbors":
        
        st.subheader("K-Nearest Neighbors age estimation model")
        # List of number of neighbors
        neighbors = [15, 55, 100, 200]
        # knn = KNeighborsRegressor (n_neighbors = 10)
        # st.write("Training the Model...")
        # knn.fit (X_train, y_train)

        # st.write("Successfully Train the model")

        # outputPredictedKNN = knn.predict(X_test)
        # st.write("Predicted result for Testing Dataset: ")
        # outputPredictedKNN

        # MSEKNN = mean_squared_error (outputPredictedKNN, y_test)
        # st.write("The mean Squared Error Produced by KNN with number of nearest neighbors 10: ", MSEKNN)
        
        for n in neighbors:
            st.subheader("- - - - -")
            st.write("Neighbor =",n)            
            knn = KNeighborsRegressor(n_neighbors=n)
            st.write("Training the Model...")

            knn.fit(X_train, y_train)
            st.write("Successfully Train the model")

            outputPredictedKNN = knn.predict(X_test)
            st.write("Predicted result for Testing Dataset: ")
            outputPredictedKNN

            MSEKNN = mean_squared_error (y_test,outputPredictedKNN)
            st.write("The mean Squared Error Produced by KNN with number of nearest neighbors:",n,"=", MSEKNN)
            kc= np.round(knn.score(X_test, y_test),2)*100
            st.write("Accuracy score:",n,"=", kc)

            # accuracy = accuracy_score(y_test,outputPredictedKNN)
            # print("Neighbors=", n,"Accuracy:", accuracy)
            from sklearn.metrics import r2_score
            knnr2=np.round(r2_score(y_test,outputPredictedKNN),2)
            st.write("R2 score:",n,"=", knnr2)


#SVM
    elif selectModel == "Support Vector Machine":

        selectKernel = st.sidebar.selectbox ("Select Kernel", options = ["RBF", "Linear", "Sigmoid", "Poly"])
       
        if selectKernel == "RBF":
            st.subheader("Support Vector Machine age estimation model")
            st.write(" ")
            st.subheader("RBF")
            svm_model = SVR(kernel="rbf")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "rbf", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for RBF Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "rbf" , svm)
            
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
            rbfr2=np.round(r2_score(y_test,prediction),2)
            st.write("R2 score:", rbfr2)

            st.write(" ")

        elif selectKernel == "Linear":
            st.subheader("Linear")
            svm_model = SVR(kernel="linear")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "linear", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for linear Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "linear" , svm)
            linearr2=np.round(r2_score(y_test,prediction),2)
            st.write("R2 score:", linearr2)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)


            st.write(" ")

        elif selectKernel == "Poly":
            st.subheader("Poly")
            svm_model = SVR(kernel="poly")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "poly", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for poly Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "poly" , svm)
            polyr2=np.round(r2_score(y_test,prediction),2)
            st.write("R2 score:", polyr2)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

            st.write(" ")
        elif selectKernel == "Sigmoid":
            st.subheader("Sigmoid")
            svm_model = SVR(kernel="sigmoid")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "sigmoid", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for sigmoid Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "sigmoid", svm)
            sigmoidr2=np.round(r2_score(y_test,prediction),2)
            st.write("R2 score:", sigmoidr2)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)



#COMMODITY PRICE
elif selectDataset == "Commodity":
    st.subheader("Description of the dataset")
    with st.columns(3)[1]:
        st.image("image/gold.png", width=300)
    text = """
    <style>
    .justify-text {
        text-align: justify;
    }
    </style>

    <div class="justify-text">
    There are 80 total columns and 1718 total rows in the dataset. The data was gathered for attributes including the price of oil, the Standard and Poor's 500 index, the Dow Jones Index US Bond rates (10 years), the exchange rate between the euro and the dollar, the price of precious metals such as silver and platinum as well as other metals like palladium and rhodium, the price of the US Dollar Index, and the Eldorado Gold Corporation and Gold Miners ETF. The historical data for the Gold ETF is available in seven columns: Date, Open, High, Low, Close, Adjusted Close, and Volume. The closing price of a stock is determined by its price at the end of the trading day, as opposed to the Adjusted Close. The adjusted closing price, on the other hand, determines value by taking into account variables like dividends, stock splits, and new stock offerings. Therefore, Adjusted Close is the outcome variable, and its value needs to be predicted.
    </div>
    """
    
    st.write(text, unsafe_allow_html=True)

    st.subheader("Full dataset for Commodity")
    #your dataset
    #your dataset

    commodity_dataset = pd.read_csv('final_USO.csv',na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
    commodity_dataset

    st.subheader("Data input for Commodity")
    data_input_training = commodity_dataset.drop(columns = ["Adj Close"])
    data_input_training

    st.subheader("Data target for Commodity")
    data_target_training = commodity_dataset['Adj Close']
    data_target_training

    st.subheader("Training and testing data will be divided using Train_Test_Split")
    X = data_input_training
    y = data_target_training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =scaler.transform(X_test)

    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    X_train
    st.write("Training Data Target")
    y_train

    st.subheader("Testing data for input and target")
    st.write("Testing Data Input")
    X_test
    st.write("Testing Data Target")
    y_test

#Algorithm selection
    selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])

#RANDOM FOREST
    if selectModel == "Random Forest":
        n_estimators_list = [15, 25, 50, 100]
        st.subheader(f"Random Forest age estimation model")
        
        for n_estimators in n_estimators_list:
            st.subheader("- - - - -")
            st.write("N Estimator =",n_estimators)   
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
            st.write("Training the Model...")
            rf.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            output_predicted = rf.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            st.write(output_predicted)

            MSE = mean_squared_error(y_test,output_predicted)
            st.write(f"The Mean Squared Error produced by n_estimators:",n_estimators,"=", MSE)
            from sklearn.metrics import r2_score
            r2=np.round(r2_score(y_test,output_predicted),2)
            st.write("R2 score:",n_estimators,"=", r2)
    

        model = rf
        selectPredict = st.sidebar.selectbox ("Select Prediction", options = ["Predict This"])
        def predict_target_value(selected_date):
            # Convert user input date to string and then to numeric representation
            selected_date_str = selected_date.strftime("%Y-%m-%d")
            user_numeric_date = datetime.strptime(selected_date_str, "%Y-%m-%d").toordinal()

            # Prepare features for prediction (fill other features with default value, e.g., 0)
            default_features = [0] * (X_train.shape[1] - 1)  # Fill with zeros except for the date feature
            user_features = [user_numeric_date] + default_features

            # Scale the user input features
            user_scaled = scaler.transform([user_features])

            # Make prediction using the trained model
            prediction = rf.predict(user_scaled)

            return prediction[0]

        # Set the title of the app
        st.title("Predict Adjusted Close from Date")

        # Add a date input widget
        selected_date = st.date_input("Select a date", help="Choose a date")

        if selected_date:
            predicted_value = predict_target_value(selected_date)
            st.write("Predicted Adjusted Close Value:", predicted_value)
    
#KNN
    elif selectModel == "K-Nearest Neighbors":
        n_neighbors_list = [10, 15, 20, 100]

        for n_neighbors in n_neighbors_list:
            st.subheader(f"K-Nearest Neighbors age estimation model (n_neighbors = {n_neighbors})")
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            st.write("Training the Model...")
            knn.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            output_predicted_knn = knn.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            st.write(output_predicted_knn)

            MSE_knn = mean_squared_error(y_test,output_predicted_knn)
            st.write(f"The Mean Squared Error produced by KNN with number of nearest neighbors {n_neighbors}: ", MSE_knn)
            from sklearn.metrics import r2_score
            knnr2=np.round(r2_score(y_test,output_predicted_knn),2)
            st.write("R2 score:",n_neighbors,"=", knnr2)

#SVM
    elif selectModel == "Support Vector Machine":

        selectKernel = st.sidebar.selectbox ("Select Kernel", options = ["RBF", "Linear", "Sigmoid", "Poly"])
       
        if selectKernel == "RBF":
            st.subheader("Support Vector Machine age estimation model")
            st.write(" ")
            st.subheader("RBF")
            svm_model = SVR(kernel="rbf")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "rbf", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for RBF Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "rbf" , svm)

            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
            from sklearn.metrics import r2_score
            rbfr2=np.round(r2_score(y_test,prediction),2)
            st.write("R2 score:", rbfr2)

            st.write(" ")

        elif selectKernel == "Linear":
            st.subheader("Linear")
            svm_model = SVR(kernel="linear")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "linear", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for linear Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "linear" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
            from sklearn.metrics import r2_score
            linearr2=np.round(r2_score(y_test,prediction),2)
            st.write("R2 score:", linearr2)


            st.write(" ")

        elif selectKernel == "Poly":
            st.subheader("Poly")
            svm_model = SVR(kernel="poly")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "poly", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for poly Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "poly" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
            polyr2=np.round(r2_score(y_test,prediction),2)
            st.write("R2 score:", polyr2)

        


            st.write(" ")
        elif selectKernel == "Sigmoid":
            st.subheader("Sigmoid")
            svm_model = SVR(kernel="sigmoid")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "sigmoid", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for sigmoid Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "sigmoid", svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
            sigmoidr2=np.round(r2_score(y_test,prediction),2)
            st.write("R2 score:", sigmoidr2)

#CRYPTOCURRENCY PRICE
elif selectDataset == "Cryptocurrency":

    st.subheader("Full dataset for Cryptocurrency")
    #your dataset

    coin_dataset = pd.read_csv('coin_XRP.csv')
    coin_dataset

    st.subheader("Data input for Cryptocurrency")
    data_input_training = coin_dataset.drop(columns = ["unix","symbol","date","close","Volume XRP","Volume USDT"])
    data_input_training

    missing_values = data_input_training.isnull().sum()

    st.subheader("Missing Values:")
    st.write(missing_values)

    st.subheader("Data target for Cryprocurrency")
    data_target_training = coin_dataset['close']
    data_target_training

    st.subheader("Training and testing data will be divided using Train_Test_Split")
    X = data_input_training
    y = data_target_training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    X_train
    st.write("Training Data Target")
    y_train

    st.subheader("Testing data for input and target")
    st.write("Training Data Input")
    X_test
    st.write("Training Data Target")
    y_test

#Algorithm selection
    selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])

#RANDOM FOREST
    if selectModel == "Random Forest":

        st.subheader("Random Forest age estimation model")
        #List of number estimators
        estimators = [15, 25, 50, 100]
        for n in estimators:
            st.subheader("- - - - -")
            st.write("N Estimator =",n)            

            rf = RandomForestRegressor (n_estimators = n, random_state = 0)
            st.write("Training the Model...")
            rf.fit (X_train, y_train)

            st.write("Successfully Train the model")
            outputPredictedRF = rf.predict(X_test)
            st.write("Predicted result for Testing Dataset: ")
            outputPredictedRF

            MSERF = mean_squared_error (y_test,outputPredictedRF)
            st.write("The mean Squared Error Produced by n_estimator:",n,"=", MSERF)
            rc= np.round(rf.score(X_test, y_test),2)*100
            st.write("Accuracy score:",n,"=", rc)
            from sklearn.metrics import r2_score
            r2=np.round(r2_score(y_test,outputPredictedRF),2)
            st.write("R2 score:",n,"=", r2)

#KNN
    elif selectModel == "K-Nearest Neighbors":
        
        n_neighbors_list = [10, 15, 20, 100]

        for n_neighbors in n_neighbors_list:
            st.subheader(f"K-Nearest Neighbors age estimation model (n_neighbors = {n_neighbors})")
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            st.write("Training the Model...")
            knn.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            output_predicted_knn = knn.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            st.write(output_predicted_knn)

            MSE_knn = mean_squared_error(y_test,output_predicted_knn)
            st.write(f"The Mean Squared Error produced by KNN with number of nearest neighbors {n_neighbors}: ", MSE_knn)
            sc = np.round(knn.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
            from sklearn.metrics import r2_score
            knnr2=np.round(r2_score(y_test,output_predicted_knn),2)
            st.write("R2 score:",n_neighbors,"=", knnr2)
#SVM
    elif selectModel == "Support Vector Machine":

        selectKernel = st.sidebar.selectbox ("Select Kernel", options = ["RBF", "Linear", "Sigmoid", "Poly"])
       
        if selectKernel == "RBF":
            st.subheader("Support Vector Machine age estimation model")
            st.write(" ")
            st.subheader("RBF")
            svm_model = SVR(kernel="rbf")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "rbf", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for RBF Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "rbf" , svm)
            
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

            st.write(" ")

        elif selectKernel == "Linear":
            st.subheader("Linear")
            svm_model = SVR(kernel="linear")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "linear", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for linear Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "linear" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)


            st.write(" ")

        elif selectKernel == "Poly":
            st.subheader("Poly")
            svm_model = SVR(kernel="poly")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "poly", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for poly Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "poly" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

            st.write(" ")
        elif selectKernel == "Sigmoid":
            st.subheader("Sigmoid")
            svm_model = SVR(kernel="sigmoid")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "sigmoid", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for sigmoid Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "sigmoid", svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)



#REAL ESTATE PRICE
elif selectDataset == "Futures":
    from sklearn.impute import SimpleImputer
    
    st.subheader("Full dataset for Futures")
    #your dataset

    re_dataset = pd.read_csv('FUTURES\Crude Oil.csv',nrows=1500)
    df = pd.DataFrame(re_dataset)

    # Replace '?' with NaN
    re_dataset = re_dataset.replace('?', float('nan'))


    st.subheader("Data input for Future")
    data_input_training = re_dataset.drop(columns = ["Adj Close","Close","Date","Volume"])
    data_input_training


    st.subheader("Data target for Commodity")
    data_target_training = re_dataset['Close']
    data_target_training

    missing_values = data_input_training.isnull().sum()

    st.subheader("Missing Values:")
    st.write(missing_values)

    # Drop rows with missing values from both X and y
    data_input_training = data_input_training.dropna()
    data_target_training = data_target_training.dropna()

    # Handle missing data with SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(data_input_training)

    

    st.subheader("Training and testing data will be divided using Train_Test_Split")
    y = data_target_training
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.33, random_state=42)

    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    st.write(X_train)
    st.write("Training Data Target")
    st.write(y_train)

    st.subheader("Testing data for input and target")
    st.write("Training Data Input")
    st.write(X_test)
    st.write("Training Data Target")
    st.write(y_test)


#Algorithm selection
    selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])

#RANDOM FOREST
    if selectModel == "Random Forest":

        st.subheader("Random Forest age estimation model")
        #List of number estimators
        estimators = [15, 25, 50, 100]
        for n in estimators:
            st.subheader("- - - - -")
            st.write("N Estimator =",n)            

            rf = RandomForestRegressor (n_estimators = n, random_state = 0)
            st.write("Training the Model...")
            rf.fit (X_train, y_train)

            st.write("Successfully Train the model")
            outputPredictedRF = rf.predict(X_test)
            st.write("Predicted result for Testing Dataset: ")
            outputPredictedRF

            MSERF = mean_squared_error (y_test,outputPredictedRF)
            st.write("The mean Squared Error Produced by n_estimator:",n,"=", MSERF)
            rc= np.round(rf.score(X_test, y_test),2)*100
            st.write("Accuracy score:",n,"=", rc)
            from sklearn.metrics import r2_score
            r2=np.round(r2_score(y_test,outputPredictedRF),2)
            st.write("R2 score:",n,"=", r2)

#KNN
    elif selectModel == "K-Nearest Neighbors":
        
        n_neighbors_list = [10, 15, 20, 100]

        for n_neighbors in n_neighbors_list:
            st.subheader(f"K-Nearest Neighbors age estimation model (n_neighbors = {n_neighbors})")
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            st.write("Training the Model...")
            knn.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            output_predicted_knn = knn.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            st.write(output_predicted_knn)

            MSE_knn = mean_squared_error(y_test,output_predicted_knn)
            st.write(f"The Mean Squared Error produced by KNN with number of nearest neighbors {n_neighbors}: ", MSE_knn)
            sc = np.round(knn.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)
            from sklearn.metrics import r2_score
            knnr2=np.round(r2_score(y_test,output_predicted_knn),2)
            st.write("R2 score:",n_neighbors,"=", knnr2)

#SVM
    elif selectModel == "Support Vector Machine":

        selectKernel = st.sidebar.selectbox ("Select Kernel", options = ["RBF", "Linear", "Sigmoid", "Poly"])
       
        if selectKernel == "RBF":
            st.subheader("Support Vector Machine age estimation model")
            st.write(" ")
            st.subheader("RBF")
            svm_model = SVR(kernel="rbf")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "rbf", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for RBF Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "poly" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

            st.write(" ")

        elif selectKernel == "Linear":
            st.subheader("Linear")
            svm_model = SVR(kernel="linear")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "linear", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for linear Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "linear" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)


            st.write(" ")

        elif selectKernel == "Poly":
            st.subheader("Poly")
            svm_model = SVR(kernel="poly")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "poly", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for poly Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "poly" , svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

            st.write(" ")
        elif selectKernel == "Sigmoid":
            st.subheader("Sigmoid")
            svm_model = SVR(kernel="sigmoid")
            st.write("Training the Model...")
            svm_model.fit(X_train, y_train)

            st.write("Successfully Train the model")

            st.write("SVM", "sigmoid", "Prediction")
            prediction = svm_model.predict(X_test)
            st.write("Predicted result for sigmoid Testing Dataset: ")
            prediction

            svm = mean_squared_error(y_test,prediction)
            st.write("mean squared error: for kernel", "sigmoid", svm)
            sc= np.round(svm_model.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)