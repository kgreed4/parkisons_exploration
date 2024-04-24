# Parkinson's Progression AI
This repository contains the source code for the Parkinson's Progression AI, a tool designed to predict the progression of Parkinson's disease using three machine learning models through sit-to-stand transitions, audio, and handwriting analyses. The application is built with Python and utilizes the Streamlit framework for an interactive web interface.The deployed application can be found at: https://parkinsons-progression-ai.streamlit.app.


## Installation

To run this project locally, you will need to install the necessary Python packages. Follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-repository/parkinsons_exploration.git
   ```
2. Navigate to the project directory:
   ```
   cd parkinsons_explorations
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

After installing the requirements, you can run the application locally using Streamlit:
```
streamlit run app.py
```

This process will run `app.py` which calls each of the three models and their processing scripts.

### Structure
There are three categoires analyzed in this application: `sts, voice, and handwriting`. Each of these has their own folder, respectively. Within each folder includes the inference script, `sts.py, handwriting.py, voice.py`, the training script, `train_sts.py, train_handwriting.py, train_voice.py`, and the trained output model(s) that are being used during inferece.