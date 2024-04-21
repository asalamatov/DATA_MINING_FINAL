import pickle
from utils import Preprocessor
import pandas as pd
import streamlit as st

with open('gb_p.pkl', 'rb') as f:
    reloaded_model = pickle.load(f)
item = {"TransactionDate":"2020.12","HouseAge":17.0,"DistanceToStation":467.6447748,"NumberOfPubs":4.0,"PostCode":"5222.0"}


def main():
    st.subheader('Abdulkerim Karasoy')
    st.title("House Price Prediction")

    item = {
        "TransactionDate": st.text_input("Transaction Date", "2020.12"),
        "HouseAge": st.number_input("House Age", value=17.0),
        "DistanceToStation": st.number_input("Distance To Station", value=467.6447748),
        "NumberOfPubs": st.number_input("Number Of Pubs", value=4.0),
        "PostCode": st.text_input("Post Code", "5222.0")
    }

    formatted_output = reloaded_model.predict(pd.DataFrame([item.values()], columns=item.keys()))

    st.success("The House Price Prediction: %.2f" % formatted_output[0])

if __name__ == "__main__":
    main()