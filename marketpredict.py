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

st.title("Group Project CSC649")
st.header("Market Price Prediction")
st.write("UiTM Tapah Perak")
st.write("MUHAMMAD AFIF FAHMI BIN RAFPI 2022764801")
st.write("MUHAMMAD SYAFIQ BIN KHERUDDIN 2022949945")
st.write("MUHAMMAD IFFAT HAIKAL BIN SAABAN HAFIZHI 2022978265")
st.write("NUR ANIS KHAIRINA BINTI SHAHRUL ANUAR 2022780117")
st.write("NURIN ADLINA BINTI MOHD NAZRI 2022947239")

selectDataset = st.sidebar.selectbox ("Select Dataset", options = ["Home", "Forex", "Stock","Commodity","Crytocurrrency","Real Estate"])

#FOREX PRICE
if selectDataset == "Forex":
    
    st.subheader("Full dataset for Forex")
    #your dataset
    forex_dataset = pd.read_csv('EURUSD.csv')
    forex_dataset

    df = pd.DataFrame(forex_dataset)
    df['time'] = pd.to_datetime(df['time'])
    df['Year'] = df['time'].dt.year
    df['Month'] = df['time'].dt.month
    df['Day'] = df['time'].dt.day
    df.drop(columns=['time'], inplace=True)
    st.write(df.dtypes)

     

    st.subheader("Data input for Forex")
    data_input_training = df.drop(columns=['close'])  
    data_input_training

    st.subheader("Data target for Forex")
    data_target_training = df['close']
    data_target_training

    st.subheader("Training and testing data will be divided using Train_Test_Split")
    X = data_input_training
    y = data_target_training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])

#RANDOM FOREST
    if selectModel == "Random Forest":

        st.subheader("Random Forest age estimation model")
        rf = RandomForestRegressor (n_estimators = 50, random_state = 0)
        st.write("Training the Model...")
        rf.fit (X_train, y_train)

        st.write("Successfully Train the model")
        outputPredicted50 = rf.predict(X_test)
        st.write("Predicted result for Testing Dataset: ")
        outputPredicted50

        MSE50 = mean_squared_error (outputPredicted50, y_test)
        st.write("The mean Squared Error Produced by n_estimator = 50: ", MSE50)

#KNN
    elif selectModel == "K-Nearest Neighbors":
        
        st.subheader("K-Nearest Neighbors age estimation model")
        knn = KNeighborsRegressor (n_neighbors = 10)
        st.write("Training the Model...")
        knn.fit (X_train, y_train)

        st.write("Successfully Train the model")

        outputPredictedKNN = knn.predict(X_test)
        st.write("Predicted result for Testing Dataset: ")
        outputPredictedKNN

        MSEKNN = mean_squared_error (outputPredictedKNN, y_test)
        st.write("The mean Squared Error Produced by KNN with number of nearest neighbors 10: ", MSEKNN)

#SVM
    elif selectModel == "Support Vector Machine":

       

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

        mse = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "rbf" , mse)

        r2 = r2_score(y_test, prediction)
        st.write("R-squared: for kernel", "rbf" , r2)

        st.write(" ")
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
        st.write("Accuracy score:", svm)


        st.write(" ")
        st.subheader("Poly")
        svm_model = SVR(kernel="poly")
        st.write("Training the Model...")
        svm_model.fit(X_train, y_train)

        st.write("Successfully Train the model")

        st.write("SVM", "poly", "Prediction")
        prediction = svm_model.predict(X_test)
        st.write("Predicted result for poly Testing Dataset: ")
        prediction

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "poly" , svm)


        st.write(" ")
        st.subheader("Sigmoid")
        svm_model = SVR(kernel="sigmoid")
        st.write("Training the Model...")
        svm_model.fit(X_train, y_train)

        st.write("Successfully Train the model")

        st.write("SVM", "sigmoid", "Prediction")
        prediction = svm_model.predict(X_test)
        st.write("Predicted result for sigmoid Testing Dataset: ")
        prediction

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "sigmoid" , svm)
        


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
    There are 7 total columns and 851264 total rows in the dataset. The data was gathered for attributes including the date,symbol,open,close,low,high and volume. As the dataset consist of many stocks,this system will only sample an Amazon stock based on the symbol AMZN. The closing price of a stock is determined by its price at the end of the trading day, as opposed to the Close. The closing price, on the other hand, determines value by taking into account variables like dividends, stock splits, and new stock offerings. Therefore, Close is the outcome variable, and its value needs to be predicted.
    </div>
    """
    
    st.write(text, unsafe_allow_html=True)

    st.subheader("Full dataset for Stock")
    #your dataset
    stock = pd.read_csv('prices.csv')
    stock

    #sampling amazon dataset
    st.write("Sample amazon dataset")
    stock1 = stock[stock['symbol']=='AMZN']
    stock1

    st.subheader("Data input for stock")
    stock1.drop(columns = ["symbol"])
    data_input_training = stock1[['volume','open']]
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
        from sklearn.metrics import r2_score
        polyr2=np.round(r2_score(y_test,prediction),2)
        st.write("R2 score:", polyr2)


        st.write(" ")
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
        from sklearn.metrics import r2_score
        sigmoidr2=np.round(r2_score(y_test,prediction),2)
        st.write("R2 score:", sigmoidr2)



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

    commodity_dataset = pd.read_csv('final_USO.csv')
    commodity_dataset

    st.subheader("Data input for Commodity")
    data_input_training = commodity_dataset.drop(columns = ["Adj Close","Date"])
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
        n_estimators_list = [50, 100, 150, 200]

        for n_estimators in n_estimators_list:
            st.subheader(f"Random Forest age estimation model (n_estimators = {n_estimators})")
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
            st.write("Training the Model...")
            rf.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            output_predicted = rf.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            st.write(output_predicted)

            MSE = mean_squared_error(output_predicted, y_test)
            st.write(f"The Mean Squared Error produced by n_estimators = {n_estimators}: ", MSE)
            sc = np.round(rf.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

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

            MSE_knn = mean_squared_error(output_predicted_knn, y_test)
            st.write(f"The Mean Squared Error produced by KNN with number of nearest neighbors {n_neighbors}: ", MSE_knn)
            sc = np.round(knn.score(X_test, y_test),2)*100
            st.write("Accuracy score:", sc)

#SVM
    elif selectModel == "Support Vector Machine":

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

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "rbf" , svm)

        sc= np.round(svm_model.score(X_test, y_test),2)*100
        st.write("Accuracy score:", sc)

        st.write(" ")
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



#CRYPTOCURRENCY PRICE
elif selectDataset == "Cryptocurrency":

    st.subheader("Full dataset for Cryptocurrency")
    #your dataset

    female_dataset = pd.read_csv('xray_image_dataset_female.csv')
    female_dataset

    st.subheader("Data input for female")
    data_input_training = female_dataset.drop(columns = ["No", "Race", "Gender", "DOB", "Exam Date", "Tanner", "Trunk HT (cm)"])
    data_input_training

    st.subheader("Data target for female")
    data_target_training = female_dataset['ChrAge']
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
        rf = RandomForestRegressor (n_estimators = 50, random_state = 0)
        st.write("Training the Model...")
        rf.fit (X_train, y_train)

        st.write("Successfully Train the model")
        outputPredicted50 = rf.predict(X_test)
        st.write("Predicted result for Testing Dataset: ")
        outputPredicted50

        MSE50 = mean_squared_error (outputPredicted50, y_test)
        st.write("The mean Squared Error Produced by n_estimator = 50: ", MSE50)

#KNN
    elif selectModel == "K-Nearest Neighbors":
        
        st.subheader("K-Nearest Neighbors age estimation model")
        knn = KNeighborsRegressor (n_neighbors = 10)
        st.write("Training the Model...")
        knn.fit (X_train, y_train)

        st.write("Successfully Train the model")

        outputPredictedKNN = knn.predict(X_test)
        st.write("Predicted result for Testing Dataset: ")
        outputPredictedKNN

        MSEKNN = mean_squared_error (outputPredictedKNN, y_test)
        st.write("The mean Squared Error Produced by KNN with number of nearest neighbors 10: ", MSEKNN)

#SVM
    elif selectModel == "Support Vector Machine":

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

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "rbf" , svm)

        st.write(" ")
        st.subheader("Linear")
        svm_model = SVR(kernel="linear")
        st.write("Training the Model...")
        svm_model.fit(X_train, y_train)

        st.write("Successfully Train the model")

        st.write("SVM", "linear", "Prediction")
        prediction = svm_model.predict(X_test)
        st.write("Predicted result for linear Testing Dataset: ")
        prediction

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "linear" , svm)


        st.write(" ")
        st.subheader("Poly")
        svm_model = SVR(kernel="poly")
        st.write("Training the Model...")
        svm_model.fit(X_train, y_train)

        st.write("Successfully Train the model")

        st.write("SVM", "poly", "Prediction")
        prediction = svm_model.predict(X_test)
        st.write("Predicted result for poly Testing Dataset: ")
        prediction

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "poly" , svm)


        st.write(" ")
        st.subheader("Sigmoid")
        svm_model = SVR(kernel="sigmoid")
        st.write("Training the Model...")
        svm_model.fit(X_train, y_train)

        st.write("Successfully Train the model")

        st.write("SVM", "sigmoid", "Prediction")
        prediction = svm_model.predict(X_test)
        st.write("Predicted result for sigmoid Testing Dataset: ")
        prediction

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "sigmoid", svm)



#REAL ESTATE PRICE
elif selectDataset == "Real Estate":

    
    st.subheader("Full dataset for Real Estate")
    #your dataset

    female_dataset = pd.read_csv('xray_image_dataset_female.csv')
    female_dataset

    st.subheader("Data input for female")
    data_input_training = female_dataset.drop(columns = ["No", "Race", "Gender", "DOB", "Exam Date", "Tanner", "Trunk HT (cm)"])
    data_input_training

    st.subheader("Data target for female")
    data_target_training = female_dataset['ChrAge']
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
        rf = RandomForestRegressor (n_estimators = 50, random_state = 0)
        st.write("Training the Model...")
        rf.fit (X_train, y_train)

        st.write("Successfully Train the model")
        outputPredicted50 = rf.predict(X_test)
        st.write("Predicted result for Testing Dataset: ")
        outputPredicted50

        MSE50 = mean_squared_error (outputPredicted50, y_test)
        st.write("The mean Squared Error Produced by n_estimator = 50: ", MSE50)

#KNN
    elif selectModel == "K-Nearest Neighbors":
        
        st.subheader("K-Nearest Neighbors age estimation model")
        knn = KNeighborsRegressor (n_neighbors = 10)
        st.write("Training the Model...")
        knn.fit (X_train, y_train)

        st.write("Successfully Train the model")

        outputPredictedKNN = knn.predict(X_test)
        st.write("Predicted result for Testing Dataset: ")
        outputPredictedKNN

        MSEKNN = mean_squared_error (outputPredictedKNN, y_test)
        st.write("The mean Squared Error Produced by KNN with number of nearest neighbors 10: ", MSEKNN)

#SVM
    elif selectModel == "Support Vector Machine":

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

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "rbf" , svm)

        st.write(" ")
        st.subheader("Linear")
        svm_model = SVR(kernel="linear")
        st.write("Training the Model...")
        svm_model.fit(X_train, y_train)

        st.write("Successfully Train the model")

        st.write("SVM", "linear", "Prediction")
        prediction = svm_model.predict(X_test)
        st.write("Predicted result for linear Testing Dataset: ")
        prediction

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "linear" , svm)


        st.write(" ")
        st.subheader("Poly")
        svm_model = SVR(kernel="poly")
        st.write("Training the Model...")
        svm_model.fit(X_train, y_train)

        st.write("Successfully Train the model")

        st.write("SVM", "poly", "Prediction")
        prediction = svm_model.predict(X_test)
        st.write("Predicted result for poly Testing Dataset: ")
        prediction

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "poly" , svm)


        st.write(" ")
        st.subheader("Sigmoid")
        svm_model = SVR(kernel="sigmoid")
        st.write("Training the Model...")
        svm_model.fit(X_train, y_train)

        st.write("Successfully Train the model")

        st.write("SVM", "sigmoid", "Prediction")
        prediction = svm_model.predict(X_test)
        st.write("Predicted result for sigmoid Testing Dataset: ")
        prediction

        svm = mean_squared_error(prediction,y_test)
        st.write("mean squared error: for kernel", "sigmoid", svm)
