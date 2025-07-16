

# 🏏 IPL Score Prediction Using Deep Learning

This project analyzes Indian Premier League (IPL) match data and uses a deep learning model to predict the total score using match context and player information.

## 📌 Project Overview

The project performs the following key tasks:

1. **Data Visualization**:

   * Matches played per venue.
   * Top 10 batsmen based on runs.
   * Top 10 bowlers based on wickets.

2. **Preprocessing**:

   * Label Encoding of categorical features.
   * Feature scaling using `MinMaxScaler`.

3. **Model Building**:

   * A deep learning regression model is built using Keras and TensorFlow.
   * Huber loss is used to improve robustness to outliers.


## 📁 Dataset

The dataset used is `ipl_dataset.csv`, which includes ball-by-ball details of IPL matches like:

* Batting and bowling teams
* Venue
* Batsman, bowler
* Runs scored
* Wickets taken
* Overs bowled
* Total match score



## 🛠️ Libraries Used

* `pandas`, `numpy` – Data manipulation
* `matplotlib`, `seaborn` – Visualization
* `scikit-learn` – Preprocessing and train-test splitting
* `keras`, `tensorflow` – Deep learning model


## 📊 Exploratory Data Analysis (EDA)

* **Matches per Venue**: Barplot showing the number of matches played in each stadium.
* **Top Batsmen**: Batsmen with the highest individual scores.
* **Top Bowlers**: Bowlers with the most wickets.



## 🧹 Data Preprocessing

* **Categorical Encoding**: Label encoding for `bat_team`, `bowl_team`, `venue`, `batsman`, `bowler`.
* **Scaling**: Features scaled between 0 and 1 using `MinMaxScaler`.
* **Correlation Heatmap**: Visual insight into feature relationships.



## 🤖 Model Details

A Sequential model is built with the following architecture:

* Dense layer with 512 neurons, ReLU activation
* Dense layer with 216 neurons, ReLU activation
* Output layer with 1 neuron (linear activation)

Loss function: **Huber Loss**
Optimizer: **Adam**
Epochs: **10**
Batch size: **64**



## 🔍 Results

* The model is trained on 70% of the data and validated on the remaining 30%.
* Evaluation metrics such as MAE/RMSE can be added for future improvements.



## 🧠 Future Improvements

* Add model evaluation metrics (e.g., MAE, RMSE, R²).
* Experiment with different model architectures (e.g., LSTM for sequence modeling).
* Add player and team historical stats as features.
* Optimize hyperparameters using techniques like Grid Search.



## 🚀 How to Run

1. Clone this repository.
2. Place `ipl_dataset.csv` in the project directory.
3. Run the Python script (e.g., `python ipl_score_prediction.py`) or use a Jupyter Notebook.
4. Ensure required libraries are installed using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow
```


## 📬 Contact

For any questions or feedback, feel free to reach out.


