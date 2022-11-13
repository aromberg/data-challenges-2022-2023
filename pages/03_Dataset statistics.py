import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dataset Statistics",
                   page_icon="📊")

df = pd.read_csv('data/heart_disease/heart_disease_combined.csv')


st.markdown(
    """
    # Dataset Statistics 📊
    ## Histogram
    """)

option = st.selectbox(
    'Which column would you like to plot?',
    df.columns[1:])

hist = px.histogram(df, x=option)
st.plotly_chart(hist, use_container_width=True)

st.write("## Scatter Plot")

scatter_x = st.selectbox(
    'Which column would you like to plot on the x axis?',
    df.columns[1:])

scatter_y = st.selectbox(
    'Which column would you like to plot on the y axis?',
    df.columns[1:],
    index=7)


scatter = px.scatter(df, x=scatter_x, y=scatter_y)
st.plotly_chart(scatter, use_container_width=True)

st.markdown(
    """
    ## What are these columns about?

    1. **age**: age in years
    2. **sex**: sex (1 = male; 0 = female)      
    3. **cp**: chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)       
    4. **trestbps**: resting blood pressure (in mm Hg on admission to the hospital)
    5. **chol**: serum cholestoral in mg/dl
    6. **fbs**: fasting blood sugar > 120 mg/dl  (1 = yes; 0 = no)
    7. **restecg**: resting electrocardiographic results (0 = normal; 1 = having ST-T wave abnormality; 2 = showing probable or definite left ventricular hypertrophy)
    8. **thalach**: maximum heart rate achieved
    9. **exang**: exercise induced angina (1 = yes; 0 = no)
    10. **oldpeak**: ST depression induced by exercise relative to rest
    11. **slope**: slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
    12. **ca**: number of major vessels (0-3) colored by flourosopy   
    13. **thal**: 3 = normal; 6 = fixed defect; 7 = reversable defect
    14. **num**: diagnosis of heart disease/angiographic disease status (0 = < 50% diameter narrowing; 1 = > 50% diameter narrowing)
    
    ---
    """
)

with st.expander("See complete dataset"):
    st.write(df)