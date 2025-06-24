# NBA Defensive Performance Predictor (Capstone Project for WGU) -- Undergrad Level Project

🏀 Predicting NBA players’ Defensive Box Plus/Minus (DBPM) using physical attributes and machine learning.

This project was developed as a capstone for my B.S. in Computer Science at Western Governors University.

## 📌 Overview

This app uses Random Forest regression to estimate a player’s defensive impact (DBPM) from NBA Draft Combine stats like height, weight, sprint time, lane agility, and verticals.

It includes:

- Full ML pipeline using **SEMMA methodology**
- Custom feature engineering for each position (PG–C)
- SQLite-powered data joins (draft combine + advanced stats)
- Model explainability using **SHAP**
- GUI built in **Tkinter** for interactive predictions
- Packaged as a `.exe` — no setup required

## 📊 Features

- 🔁 Auto-fetches data from Kaggle
- 🧠 Position-specific ML models (Guards, Forwards, Bigs)
- 🧮 R² and MSE metrics for each position
- 📈 Visualizations: SHAP bar, SHAP beeswarm, feature importance, scatterplots
- 💾 Auto-saves all plots to folders
- 🎛️ GUI interface to enter player stats and run predictions

## 🛠️ Technologies Used

- Python, scikit-learn, SHAP, pandas, matplotlib
- SQLite for structured joins
- Tkinter for GUI
- Packaged with PyInstaller for `.exe` distribution

## 📷 Screenshots

<img width="953" alt="image" src="https://github.com/user-attachments/assets/e2f1c0c6-45c8-4731-99ff-7f8eb4bca8e0" />
*Figure 1: launching main.exe and GUI*

<img width="497" alt="image" src="https://github.com/user-attachments/assets/0f32556c-3ce9-4568-9d0d-a1d57dd96c05" />
*Figure 2: selecting R^2 for position and the submit button to get the predictive DBPM*

<img width="731" alt="image" src="https://github.com/user-attachments/assets/e4b64744-25f5-4faf-aacb-0ff6e2580634" />
*Figure 3: generating Feautre Importance Graph*

<img width="434" alt="image" src="https://github.com/user-attachments/assets/663b4a22-2918-4b38-b991-6c70d4f58b05" />
*Figure 4: generating Beeswarm and Bar Graphs using SHAP- saved to program directory*

![image](https://github.com/user-attachments/assets/3f859ff5-49f7-4aef-b75c-42cf21acd206)
*Figure 5: generating scatterplots comparing actual vs predictive data points*


## 🚀 How to Run

1. Get your `kaggle.json` from https://www.kaggle.com/account
2. Place it in `C:\Users\<YourName>\.kaggle\`
3. Download or clone this repo
4. Run `main.exe` (or run `main.py` if you have Python installed)

## 🧪 Example Input

- Height (in): `80`
- Weight (lbs): `190`
- Wingspan (in): `80`
- Standing Vert (in): `35`
- Lane Agility (s): `3.5`
- Sprint (s): `3.5`
- Max Vert (in): `42`
- Bench Press (reps): `0`

## 📄 Capstone Paper

See the full write-up here: [C964 Writeup- Micah Lin.pdf](https://github.com/user-attachments/files/20886357/C964.Writeup-.Micah.Lin.pdf)


## 📬 Contact

Created by [Micah Lin](https://github.com/bob-feracho)  
Open to research collaboration and further development.
