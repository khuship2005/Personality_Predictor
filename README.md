# 🧠 Personality Predictor
AI-powered personality predictor built using the Random Forest algorithm to classify personality types based on behavioral patterns.

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/khuship2005/Personality_Predictor.git
cd Personality_Predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python train_model.py
```

4. **Run the application**
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 🎯 Problem Statement

Personality assessment is important for recruitment, team building, career guidance, and personal growth. However, traditional methods are often time-consuming, costly, subjective, and not easily accessible. This project uses Random Forest algorithm to automate personality prediction based on input features like **time_spent_alone**, **social_events_attended**, **going_outside**, **friend_circle**, **post_frequency**, **drained_after_socialising**, **stage_fear**.
Based on these inputs, the system classifies personality into two categories: **Extrovert** and **Introvert**.

## 🛠️ Tech Stack

- **Python 3.8+** - Programming language
- **Random Forest** - Machine learning algorithm
- **scikit-learn** - ML library
- **pandas & numpy** - Data processing
- **Flask/Streamlit** - Web interface

## 📂 Project Files

```
├── app.py                      # Web application
├── train_model.py              # Model training script
├── personality_dataset1.csv    # Training dataset
└── requirements.txt            # Dependencies
```

## 🎯 How It Works

1. Load personality dataset from CSV file
2. Preprocess and clean the data
3. Train Random Forest classifier on the dataset
4. Generates and saves the trained model file (`model_data.pkl`)
5. Creates a data model(`personality_model.pkl`) file for feature mapping and preprocessing
6. Make predictions on new personality data
7. Display results with confidence scores

## 📊 Features

- ✅ Accurate personality classification into Introvert or Extrovert using Random Forest
- ✅ Easy-to-use web interface
- ✅ Fast predictions with confidence scores
- ✅ Trained on behavioral personality data
