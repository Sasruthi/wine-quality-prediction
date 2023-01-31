# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:27:34 2023

@author: SasruthiSri
"""

import numpy as np
import pickle
import Streamlit as st

loaded_model=pickle.load(open('C:/Users/SSGPRANAV/Downloads/wine_quality_model.sav','rb'))

def wine_quality_prediction(input_data):
    
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    return prediction

def main():
    
    st.title(' WINE QUALITY PREDICTION ')
    fixed_acidity=st.text_input('ENTER VALUE OF FIXED_ACIDITY:')
    volatile_acidity=st.text_input('ENTER VALUE OF VOLATILE ACIDITY:')
    citric_acid=st.text_input('ENTER VALUE OF CITRIC ACID:')
    residual_sugar=st.text_input('ENTER VALUE OF RESIDUAL_SUGAR:')
    chlorides=st.text_input('ENTER VALUE OF CHLORIDES:')
    free_sulfur_dioxide=st.text_input('ENTER VALUE OF FREE SULFUR DIOXIDE:')
    total_sulfur_dioxide=st.text_input('ENTER VALUE OF SULFUR DIOXIDE(in total):')
    density=st.text_input('ENTER VALUE OF DENSITY:')
    pH=st.text_input('ENTER VALUE OF PH:')
    sulphates=st.text_input('ENTER VALUE OF SULPHATES:')
    alcohol=st.text_input('ENTER VALUE OF ALCOHOL:')
    

    
    winequalitypredictions = ''
    
    if st.button('PREDICT THE WINE QUALITY:'):
        winequalitypredictions=wine_quality_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
        
    st.success(winequalitypredictions)
    
    
if __name__=='__main__':
    main()