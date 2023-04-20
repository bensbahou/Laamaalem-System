# streamlit app recieve the data from the user and send it to the model and show the result

import streamlit as st 

import numpy as np

from main import *

st.title('multi-cteteria optimization model') 

# create three columns for the user to enter the data

c1Inov_colum, c2Cost_colum, c3Deriv_colum = st.columns(3)

# create a editable table for the user to enter the c1Inov data

c1Inov_colum.write('Inovation Matrix of the suppliers')
_c1Inov = c1Inov_colum.experimental_data_editor(c1Inov)
 

# create a editable table for the user to enter the c2Cost data
c2Cost_colum.write('Cost Matrix of the suppliers') 
_c2Cost = c2Cost_colum.experimental_data_editor(c2Cost)

 

# create a editable table for the user to enter the c3Deriv data
c3Deriv_colum.write('Derivability Matrix of the suppliers') 
_c3Deriv = c3Deriv_colum.experimental_data_editor(c3Deriv)

F01_vector, F01_optimal = run_with(F01Inov)
F02_vector, F02_optimal = run_with(F02Cost)
F02_optimal = 1 / F02_optimal
F03_vector, F03_optimal = run_with(F03Deriv)

# create a button to send the data to the model
if st.button('Send Data'):
    # create thre columns for the result 
    # send the data to the model
    c1Inov_colum.write('Orders repartiotion for inovation optimization')
    c1Inov_colum.write(F01_vector.reshape(4,4))
    c1Inov_colum.write('Optimal value for innovation function')
    c1Inov_colum.write(F01_optimal[0][0])


    c2Cost_colum.write('Orders repartiotion for cost optimization')
    c2Cost_colum.write(F02_vector.reshape(4,4))
    c2Cost_colum.write('Optimal value for cost function')
    c2Cost_colum.write(F02_optimal[0][0])

    c3Deriv_colum.write('Orders repartiotion for derivability optimization')
    c3Deriv_colum.write(F03_vector.reshape(4,4))
    c3Deriv_colum.write('Optimal value for derivability function')
    c3Deriv_colum.write(F03_optimal[0][0])



 

