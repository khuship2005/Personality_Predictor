import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
# Page configuration
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --background-dark: #0f172a;
        --card-background: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
    }
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-top: 0.5rem;
    }
    /* Card styling */
    .stMetric {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    /* Input styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }
    /* Prediction result styling */
    .prediction-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 2px solid;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .prediction-introvert {
        border-color: #8b5cf6;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);
    }
    .prediction-extrovert {
        border-color: #ec4899;
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.2) 0%, rgba(251, 146, 60, 0.2) 100%);
    }
    .prediction-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.3rem;
        color: #cbd5e1;
        margin-top: 1rem;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background: #1e293b;
    }
    /* Info box styling */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    /* Feature importance styling */
    .feature-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        font-weight: 600;
        padding: 1rem 2rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)
# LOAD PRE-TRAINED MODEL AND DATA
@st.cache_resource
def load_model():
    """Load the pre-trained model and associated data"""
    try:
        # Check if model file exists
        if not os.path.exists('model_data.pkl'):
            st.error("""
            ⚠️ **Model not found!** 
            Please run `train_model.py` first
            """)
            st.stop()
        # Load model data
        with open('model_data.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return (
            model_data['model'],
            model_data['accuracy'],
            model_data['conf_matrix'],
            model_data['classification_report'],
            model_data['feature_importance'],
            model_data['feature_names']
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
# Initialize model
model, accuracy, conf_matrix, classification_rep, feature_importance, feature_names = load_model()
# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧠 Personality Predictor</h1>
    <p class="header-subtitle">A MAchine Learning application for predicting personality</p>
</div>
""", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🎯 Predict Personality", "📊 Model Performance", "📈 Feature Analysis"])
# TAB 1: PREDICTION
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Enter your Information")
        # Create two columns for inputs
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            time_alone = st.slider(
                "⏰ How many hours do you spend alone per day?",
                min_value=0,
                max_value=10,
                value=0,
                )
            social_events = st.slider(
                "🎉 How many social events do you attend monthly?",
                min_value=0,
                max_value=10,
                value=0,
                )
            going_outside = st.slider(
                "🚶 How often do you go outside per week?",
                min_value=0,
                max_value=10,
                value=0,
            )
            friends_circle = st.slider(
                "👥 How many close friends do you have?",
                min_value=0,
                max_value=10,
                value=0,
            )
            post_frequency = st.slider(
                "📱 How many times do you post on social media weekly?",
                min_value=0,
                max_value=15,
                value=0,
            )
        with input_col2:
            drained = st.selectbox(
                "😴 Do you feel drained or tired after social interactions?",
                options=["No", "Yes"],
            )
            stage_fear = st.selectbox(
                "🎭 Do you experience stage fear?",
                options=["No", "Yes"],
            )
    with col2:
        st.markdown("### ℹ️ About This Predictor")
        st.info("""
        This AI model analyzes your behavioral patterns 
        to predict your personality type.
        **How it works:**
        1. Adjust the sliders based on your behavior
        2. Select Yes/No for binary questions
        3. Click the Predict button
        4. View your personality type and confidence score
        The model uses a Random Forest algorithm 
        trained on personality data with {:.2f}% accuracy.
        """.format(accuracy * 100))
    st.markdown("<br>", unsafe_allow_html=True)
    # Predict button
    predict_btn = st.button("🔮 Predict Personality", use_container_width=True)
    if predict_btn:
        # Prepare input data
        stage_fear_val = 1 if stage_fear == "Yes" else 0
        drained_val = 1 if drained == "Yes" else 0
        input_data = np.array([[
            time_alone,
            stage_fear_val,
            social_events,
            going_outside,
            drained_val,
            friends_circle,
            post_frequency
        ]])
        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        personality = "Introvert" if prediction == 0 else "Extrovert"
        confidence = prediction_proba[prediction] * 100
        # Display result with animation
        st.markdown("<br>", unsafe_allow_html=True)
        prediction_class = "prediction-introvert" if personality == "Introvert" else "prediction-extrovert"
        color = "#8b5cf6" if personality == "Introvert" else "#ec4899"
        
        st.markdown(f"""
        <div class="prediction-box {prediction_class}">
            <div class="prediction-label">Your Personality Type</div>
            <div class="prediction-value" style="color: {color};">{personality}</div>
            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Input Summary")
        summary_df = pd.DataFrame({
            'Feature': ['Time Alone', 'Stage Fear', 'Social Events', 'Going Outside', 
                'Drained After Socializing', 'Friends Circle', 'Post Frequency'],
                'Value': [time_alone, stage_fear, social_events, going_outside, 
                 drained, friends_circle, post_frequency]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
# TAB 2: MODEL PERFORMANCE
with tab2:
    st.markdown("### 📊 Model Performance Metrics")
    # Metrics row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with metric_col2:
        precision = classification_rep['weighted avg']['precision']
        st.metric("Precision", f"{precision*100:.2f}%")
    with metric_col3:
        recall = classification_rep['weighted avg']['recall']
        st.metric("Recall", f"{recall*100:.2f}%")
    with metric_col4:
        f1 = classification_rep['weighted avg']['f1-score']
        st.metric("F1-Score", f"{f1*100:.2f}%")
    st.markdown("---")
    # Confusion Matrix
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Introvert', 'Extrovert'],
            yticklabels=['Introvert', 'Extrovert'],
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        # Confusion matrix explanation
        st.info(f"""
        **Confusion Matrix Breakdown:**
        - Correct introvert guesses: {conf_matrix[0][0]}
        - Introverts wrongly labeled as extroverts: {conf_matrix[0][1]}
        - Extroverts wrongly labeled as introverts: {conf_matrix[1][0]}
        - Correct extrovert guesses: {conf_matrix[1][1]}
        """)
    with col2:
        st.markdown("### Classification Report")
        # Create classification report dataframe
        report_df = pd.DataFrame(classification_rep).transpose()
        report_df = report_df.round(4)
        # Display with custom styling
        st.dataframe(
            report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
            use_container_width=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
# TAB 3: FEATURE ANALYSIS
with tab3:
    st.markdown("### 📈 Feature Importance Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        # Create feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Sort feature importance
        feature_importance_sorted = feature_importance.sort_values(by='Weight', ascending=True)
        colors = ['#3d4f5e', '#4a5d6f', '#566b7f', '#637990', '#7087a1', '#7d95b2', '#8aa3c3']
        bar_colors = [colors[i % len(colors)] for i in range(len(feature_importance_sorted))]
        bars = ax.barh(feature_importance_sorted['Input'], feature_importance_sorted['Weight'], color=bar_colors)
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, feature_importance_sorted['Weight'])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.4f}',
                   ha='left', va='center', fontweight='bold', 
                   fontsize=10, color='#2c3e50',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8ecf0', alpha=0.95, edgecolor='#3d4f5e'))
        ax.set_xlabel('Weighted Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown("### Feature Ranking")
        # Display feature importance table
        ranked_features = feature_importance.sort_values(by='Weight', ascending=False).copy()
        ranked_features['Rank'] = range(1, len(ranked_features) + 1)
        ranked_features['Importance %'] = (ranked_features['Weight'] * 100).round(2)
        ranked_features = ranked_features.rename(columns={'Input': 'Feature'})
        st.dataframe(
            ranked_features[['Rank', 'Feature', 'Importance %']].style.background_gradient(
                cmap='YlGnBu', subset=['Importance %']
            ),
            use_container_width=True,
            hide_index=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 2rem 0;'>
    <p style='font-size: 0.9rem;'>
        <strong>Powered by Random Forest Algorithm | Model Accuracy: {:.2f}%</strong>
    </p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
        Built with Streamlit • scikit-learn • Seaborn
    </p>
</div>
""".format(accuracy*100), unsafe_allow_html=True)