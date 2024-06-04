import streamlit as st
import pickle
from PIL import Image
import os

# Function to load pickle model
def load_model(model_filename):
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the models from pickle files
logreg_model = load_model('LogReg_model.pkl')
decisiontree_model = load_model('DecisionTree_model.pkl')
naivebayes_model = load_model('NaiveBayes_model.pkl')
rf_model = load_model('RF_model.pkl')

# Function to classify the input
def classify(answer):
    return answer[0] + " is the best crop for cultivation here."

# Streamlit app main function
def main():
    st.title("AgroGuide (Crop Recommender)")

    import os

    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, 'cc.jpg')
    image = Image.open(image_path)
    st.image(image)
    html_temp = """
    <div style="background-color:teal; padding:10px">
    <h2 style="color:white;text-align:center;">Find The Most Suitable Crop</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Sidebar to select model
    activities = ['Naive Bayes (The Best Model)', 'Logistic Regression', 'Decision Tree', 'Random Forest']
    option = st.sidebar.selectbox("Which model would you like to use?", activities)
    st.subheader(option)

    # Sliders for input features
    sn = st.slider('NITROGEN (N)', 0.0, 150.0)
    sp = st.slider('PHOSPHOROUS (P)', 0.0, 150.0)
    pk = st.slider('POTASSIUM (K)', 0.0, 210.0)
    pt = st.slider('TEMPERATURE', 0.0, 50.0)
    phu = st.slider('HUMIDITY', 0.0, 100.0)
    pPh = st.slider('Ph', 0.0, 14.0)
    pr = st.slider('RAINFALL', 0.0, 300.0)
    inputs = [[sn, sp, pk, pt, phu, pPh, pr]]

    # Classify button
    if st.button('Classify'):
        if option == 'Logistic Regression':
            st.success(classify(logreg_model.predict(inputs)))
        elif option == 'Decision Tree':
            st.success(classify(decisiontree_model.predict(inputs)))
        elif option == 'Naive Bayes (The Best Model)':  # Corrected model name here
            st.success(classify(naivebayes_model.predict(inputs)))
        else:
            st.success(classify(rf_model.predict(inputs)))

# Run the app
if __name__ == '__main__':
    main()
