import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Dataset Selection",
                   page_icon="📲")

st.markdown(
    """
    # Dataset Selection 📲
    
    ## Dataset Sources
    - [UC Irvine Machine Learning Repository](https://archive-beta.ics.uci.edu/datasets) (Categories “Life Sciences” and “Other”) -- a collection of publicly available datasets for machine learning purposes
    - [Moody PhysioNet Challenges](https://physionet.org/about/challenge/moody-challenge) -- data challenges in cardiology
    - [Digital Health Data Repository](https://github.com/DigitalBiomarkerDiscoveryPipeline/Digital_Health_Data_Repository) -- collection of sample data for use with the Digital Biomarker Discovery Pipeline
   
   
   ## Selection Criteria
   Datasets were evaluated using the following criteria, in this order:

    1. 🩺 **Medical data**: Dataset relates to (human) illness and injuries and to their treatment or prevention. Does not include distantly related topics such as signaling pathways
    2. ⏳ **Chronic diseases**: Dataset relates to chronic diseases, which “are defined broadly as conditions that last 1 year or more and require ongoing medical attention or limit activities of daily living or both. Chronic diseases such as heart disease, cancer, and diabetes are the leading causes of death and disability in the United States. They are also leading drivers of the nation’s $4.1 trillion in annual health care costs.” — [cdc.gov](https://www.cdc.gov/chronicdisease/about/index.htm)
    3. 🤯 **Not too specific**: Dataset does not contain data that is considered too specific, e. g. EEG data, voice samples or xray images
    4. ⚥ **Gender info**: Dataset includes data on biological sex of participants

    ## Selection Funnel
   """
)


df = pd.read_csv("data/datasets.csv")

data = dict(
            Stages=['All',
                    'Medical', 
                    'Chronic Diseases',
                    'Not too specific',
                    'Gender info'],
            Datasets=[l := len(df),
                      l := l - len(df[df.Status == "No medical data - rejected"]),
                      l := l - len(df[df.Status == "No chronic disease - rejected"]),
                      l := l - len(df[df.Status == "Too specific - rejected"]),
                      l := l - len(df[df.Status == "No gender info  - rejected"])])

fig = px.funnel(data, y='Stages', x='Datasets')

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    The selection process resulted in {} datasets possibly fit for analysis.

    ## Sample sizes

    Since the size of a dataset is of vital importance when choosing a dataset, we decided to plot them using a histogram:
    """.format(str(l))
)


df_hist = df[(df.Status == "Possible fit") | (df.Status == "Selected")]
hist = px.histogram(df_hist, x="Sample size", nbins = 200)
st.plotly_chart(hist, use_container_width=True)

st.write("Zooming in on possibly fitting datasets with sample sizes < 5000:")

df_zoom = df_hist.loc[df["Sample size"] < 5000]
hist_zoomed = px.histogram(df_zoom, x="Sample size", nbins=25)
st.plotly_chart(hist_zoomed, use_container_width=True)

st.markdown(
    """
    ## Final selection
    
    We finally decided to select the **"Heart Disease" dataset**. The data was published by David Aha in 1988 and consists of data from four different clinics: Cleveland, Hungary, Switzerland, and Long Beach V. While the original version consists of 76 attributes, only **14 attributes** are available and are commonly used for analysis. It includes **920 samples**. The "num" column refers to the predicted variable (0 if no disease was found and != 0 if a disease was found).

    Our rationale for selecting it was not only that it appeared the **most interesting for our team**, but also its **good documentation**, how well it is known, and lastly, its relevance given that heart disease is the **leading cause of death worldwide** (except Africa) and that it is estimated that **90% of all cases are preventable**.
    
    ---
   """
)


with st.expander("See all datasets"):
    st.write(df)