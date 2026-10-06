Solar Power Plant Output Prediction from Weather
AI4003 -- Applied Machine Learning
Fall 2026 | Assignment No. 01
A supervised machine learning project for predicting the hourly AC
power output of a solar photovoltaic (PV) plant using on-site sensor
data and publicly available weather data.

1. Project Overview
Solar power generation depends strongly on solar irradiation and
temperature. In a real deployment, an operator may have access to
on-site sensors, while a rooftop installer or grid planner may only have
access to free public weather data.
This project investigates the following practical question:
How much prediction accuracy is lost when on-site irradiation and
temperature sensors are replaced by free public weather data?
The project uses the Plant 1 data from the Solar Power Generation
Data dataset and compares two feature sets:
•	Set A: On-site sensor measurements
•	Set B: Public weather data from Open-Meteo
Three linear regression solvers are implemented manually using NumPy:
1.	Normal Equation
2.	Batch Gradient Descent
3.	Stochastic Gradient Descent
No machine-learning library is used for fitting the regression models or
calculating the regression evaluation metric.
2. Project Objectives
The main objectives are to:
•	Clean and prepare the raw solar power data.
•	Aggregate inverter-level generation into plant-level power.
•	Merge generation data with plant weather sensor data.
•	Resample the data to hourly resolution.
•	Perform exploratory data analysis.
•	Download historical public weather data using the Open-Meteo API.
•	Align public weather data with the local plant data.
•	Implement linear regression using the Normal Equation.
•	Implement Batch Gradient Descent from scratch.
•	Implement Stochastic Gradient Descent from scratch.
•	Compare the performance of on-site and public weather features.
•	Evaluate models using RMSE.
•	Analyze daytime prediction accuracy.
•	Build a simple Streamlit prediction interface.

3. Dataset
The project uses the Solar Power Generation Data dataset containing
measurements from two grid-connected photovoltaic plants in India.
This assignment works specifically with Plant 1.
The original dataset contains:
•	Plant 1 Generation Data
•	Plant 1 Weather Sensor Data
•	Plant 2 Generation Data
•	Plant 2 Weather Sensor Data
The raw CSV files are stored in the data/ directory.
Plant 1 Generation Data
Important columns include:
•	DATE_TIME
•	SOURCE_KEY
•	DC_POWER
•	AC_POWER
•	DAILY_YIELD
•	TOTAL_YIELD
Generation is recorded at the inverter level. Plant-level AC and DC
power are obtained by summing the inverter measurements at each
timestamp.
Plant 1 Weather Sensor Data
Important columns include:
•	DATE_TIME
•	AMBIENT_TEMPERATURE
•	MODULE_TEMPERATURE
•	IRRADIATION

4. Public Weather Data
Historical public weather data is downloaded from Open-Meteo using
an HTTP request.
Plant 1 starting coordinates specified in the assignment are:
•	Latitude: 14.82
•	Longitude: 78.28
•	Time zone: Asia/Kolkata
The requested public weather variables are:
•	Shortwave radiation
•	Temperature at 2 m
•	Cloud cover
Open-Meteo radiation is provided in W/m², while the on-site
irradiation data is in kW/m². Therefore, the public radiation is
converted when comparing the two sources:
W/m² ÷ 1000 = kW/m²

5. Machine Learning Problem
Input
The model receives weather/time features.
Output
The target variable is:
AC Power (kW)
The project uses two different feature sets.
Set A -- On-site Sensors
Set A uses:
Irradiation
Module Temperature
Ambient Temperature
sin(hour)
cos(hour)
Set B -- Public Weather
Set B uses:
Shortwave Radiation
Temperature 2m
Cloud Cover
sin(hour)
cos(hour)
The time features represent the daily cycle:
sin(2πh/24)
cos(2πh/24)
where h is the hour of the day.

6. Regression Model
The hypothesis used in this project is:
hθ(x) = θᵀx
The design matrix contains an intercept column:
X = [1, x1, x2, x3, x4, x5]
Therefore, the learned parameter vector contains six values:
θ0  Intercept
θ1  Feature 1
θ2  Feature 2
θ3  Feature 3
θ4  sin(hour)
θ5  cos(hour)

7. Data Split
The data is split chronologically, without shuffling.
Training
15 May 2020 – 10 June 2020
Testing
11 June 2020 – 17 June 2020
The test data is not used for:
•	Fitting the model
•	Calculating training scaling statistics
•	Selecting the learning rate


8. Feature Scaling
The features are standardized using statistics calculated only from
the training data.
The scaling formula is:
x_scaled = (x - training_mean) / training_std
The same training mean and standard deviation are then applied to the
test data.
This is particularly important for gradient descent because the features
have very different numerical scales.

9. Implemented Algorithms
9.1 Normal Equation
The closed-form solution is:
θ = (XᵀX)⁻¹Xᵀy
The implementation uses NumPy matrix operations.
9.2 Batch Gradient Descent
The model starts with:
θ = 0
and updates the parameters using all training examples at every
iteration.
The cost is recorded after every iteration.
9.3 Stochastic Gradient Descent
SGD updates the parameters one training example at a time.
The cost is recorded after every epoch.

10. Cost Function
The implemented training cost is:
J(θ) = 1/2 Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
The cost history is used to analyze the convergence of Batch Gradient
Descent and Stochastic Gradient Descent.

11. Evaluation Metric
Model performance is evaluated using Root Mean Square Error (RMSE):
RMSE = √(1/m Σ(y - ŷ)²)
RMSE is reported for:
1.	All test hours
2.	Daytime test hours only
Daytime evaluation uses:
irradiation > 0
Negative predictions are clipped to zero because a solar plant cannot
produce negative AC power:
prediction = max(prediction, 0)

12. Exploratory Data Analysis
The project generates four main exploratory plots:
1. AC Power vs Irradiation
Used to investigate the relationship between solar irradiation and
generated AC power.
2. Module Temperature vs Ambient Temperature
The points are colored according to irradiation to investigate the
effect of solar radiation on module temperature.
3. AC Power vs DC Power
Used to examine the relationship between the power entering and leaving
the inverter.
4. Average AC Power by Hour
Shows the average plant AC power for each hour of the day and helps
visualize the daily solar-generation pattern.

13. Public Weather Location Verification
On-site irradiation and Open-Meteo shortwave radiation are plotted
together for three days.
The peak hour of both curves is checked.
This step is important because an incorrect coordinate or time-zone
setting can shift the weather data relative to the plant measurements
and produce incorrect model results.

14. Repository Structure
AML_junaid_Assignment_no_01/
│
├── .venv/
│
├── app/
│   └── app.py
│
├── data/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   ├── Plant_2_Weather_Sensor_Data.csv
│   ├── plant1_hourly.csv
│   ├── plant1_merged.csv
│   └── plant1_openmeteo.csv
│
├── results/
│   ├── figures/
│   ├── analysis.md
│   
│ 
│
├── src/
│   ├── load_data.py
│   ├── prepare.py
│   ├── eda.py
│   ├── fetch_weather.py
│   ├── regression.py
│   └── train_eval.py
│
├── README.md
└── requirements.txt
The exact contents of results/ depend on the outputs generated
during the experiment.

15. Source Files
src/load_data.py
Loads the four original CSV files into pandas DataFrames.
src/prepare.py
Performs Task 1:
•	Parses timestamps.
•	Aggregates inverter-level AC/DC power.
•	Merges generation and sensor data.
•	Resamples the data to hourly means.
•	Handles missing values.
•	Saves plant1_hourly.csv.
src/eda.py
Performs Task 2 and generates the four exploratory plots.
src/fetch_weather.py
Performs Task 3:
•	Calls the Open-Meteo historical API.
•	Parses the JSON response.
•	Saves public weather data.
•	Merges it with the hourly plant data.
•	Performs the three-day location/time-zone verification.
src/regression.py
Contains the regression functions:
hypothesis()
cost()
fit_normal()
fit_batch_gd()
fit_sgd()
rmse()
clip_predictions()
src/train_eval.py
Responsible for:
•	Preparing Set A and Set B.
•	Training/testing split.
•	Feature scaling.
•	Learning-rate experiments.
•	Model fitting.
•	Test prediction.
•	RMSE calculation.
•	Comparison of the six regression fits.
•	Saving model results/weights and figures as required.
app/app.py
Provides the front-end interface for entering public weather features
and receiving predicted AC power.

16. Installation
Create and activate a Python virtual environment if required.
Install the required packages:
pip install numpy pandas matplotlib requests streamlit
The core assignment intentionally avoids machine-learning libraries such
as:
scikit-learn
statsmodels
scipy.stats
for model fitting and regression evaluation.

17. How to Run the Project
Run the scripts from the project root directory.
Step 1 -- Check the raw data
python src/load_data.py
Step 2 -- Prepare Plant 1 hourly data
python src/prepare.py
Step 3 -- Generate exploratory plots
python src/eda.py
Step 4 -- Download and merge Open-Meteo data
python src/fetch_weather.py
Step 5 -- Train and evaluate the models
python src/train_eval.py
Step 6 -- Launch the front end
streamlit run app/app.py
The Streamlit application normally opens at:
http://localhost:8501


18. Front-End
The application provides a simple interface where the user enters the
public weather features required by Set B.
The model then returns the predicted:
AC Power (kW)
The front end demonstrates how the trained regression model can be used
through a simple user-facing interface.

19. Expected Results and Analysis
The final analysis compares:
•	Normal Equation vs Batch Gradient Descent vs SGD
•	Set A vs Set B
•	All-hours RMSE vs daytime RMSE
•	Batch GD learning-rate behavior
•	SGD learning-rate behavior
•	Convergence of the optimization algorithms
•	Residual behavior by hour of day
A key focus is the difference between the daytime RMSE of Set A and Set
B.
The project is intended to determine whether publicly available weather
data can provide sufficiently accurate solar-power predictions when
on-site sensors are unavailable.
Final numerical results should be taken from the actual experiment
outputs rather than assumed in advance.

20. Main Figure
The completed repository should contain the figures required by the
assignment, including:
1.	AC power vs irradiation
2.	Module temperature vs ambient temperature
3.	AC power vs DC power
4.	Average AC power by hour
5.	Sensor irradiation vs Open-Meteo radiation
6.	Batch GD learning-rate comparison
7.	SGD learning-rate comparison
8.	Actual vs predicted AC power
9.	Residuals vs hour of day
10.	Front-end screenshot

21. Practical Question
The central question of this project is not only whether the regression
model can predict AC power.
It is:
How much accuracy is lost when on-site solar sensors are replaced by
free public weather data?
The final comparison between Set A and Set B provides the evidence
needed to answer whether public weather information is sufficiently
accurate for a practical rooftop installation or planning scenario.

22. Academic Note
This project was developed as part of AI4003 -- Applied Machine
Learning, Fall 2026.
The implementation follows the assignment requirements and focuses on
understanding the mathematical and engineering principles behind linear
regression, feature scaling, optimization, evaluation, and deployment.

23. Authors
Team Member 1: Muhammad Junaid
Team Member 2: Ali Hassan
Course: AI4003 -- Applied Machine Learning
Semester: Fall 2026
Institution: FAST-NUCES

24. License / Usage
This repository is intended for academic coursework and demonstration
purposes.
