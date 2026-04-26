import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------
# Page setup
# -----------------------------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("📚 Student Performance Analysis & Prediction")
st.write("A simple tool to explore student data and predict marks based on study hours.")

# -----------------------------------------------
# Load dataset
# -----------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("student_data.csv")
    # Fill any missing values just in case
    df["StudyHours"].fillna(df["StudyHours"].mean(), inplace=True)
    df["Attendance"].fillna(df["Attendance"].mean(), inplace=True)
    df["Marks"].fillna(df["Marks"].mean(), inplace=True)
    return df

df = load_data()

# -----------------------------------------------
# Train the model once
# -----------------------------------------------
@st.cache_resource
def train_model(data):
    X = data[["StudyHours"]]
    y = data["Marks"]
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(df)

# -----------------------------------------------
# Sidebar
# -----------------------------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Dataset", "EDA & Charts", "Predict Marks"])

# -----------------------------------------------
# Section 1: Dataset
# -----------------------------------------------
if section == "Dataset":
    st.header("📋 Student Dataset")
    st.write(f"Total students: **{len(df)}**")
    st.dataframe(df)

    st.subheader("Basic Statistics")
    st.write(df[["StudyHours", "Attendance", "Marks"]].describe().round(2))

# -----------------------------------------------
# Section 2: EDA & Charts
# -----------------------------------------------
elif section == "EDA & Charts":
    st.header("📊 Exploratory Data Analysis")

    # Scatter: Study Hours vs Marks
    st.subheader("Study Hours vs Marks")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.scatter(df["StudyHours"], df["Marks"], color="steelblue", alpha=0.7, edgecolors="white")
    ax1.set_xlabel("Study Hours")
    ax1.set_ylabel("Marks")
    ax1.set_title("Study Hours vs Marks")
    st.pyplot(fig1)

    # Bar: Attendance vs Marks (bucketed)
    st.subheader("Attendance vs Average Marks")
    df["AttendanceBucket"] = pd.cut(df["Attendance"], bins=[50, 70, 80, 90, 100],
                                     labels=["50-70%", "70-80%", "80-90%", "90-100%"])
    bucket_avg = df.groupby("AttendanceBucket", observed=True)["Marks"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(bucket_avg["AttendanceBucket"].astype(str), bucket_avg["Marks"],
            color=["#f28b82", "#fbbc04", "#34a853", "#4285f4"])
    ax2.set_xlabel("Attendance Range")
    ax2.set_ylabel("Average Marks")
    ax2.set_title("Attendance Range vs Average Marks")
    st.pyplot(fig2)

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    corr = df[["StudyHours", "Attendance", "Marks"]].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax3)
    ax3.set_title("Correlation Between Features")
    st.pyplot(fig3)

# -----------------------------------------------
# Section 3: Predict Marks
# -----------------------------------------------
elif section == "Predict Marks":
    st.header("🎯 Predict Student Marks")
    st.write("Enter the number of hours a student studies per day to predict their marks.")

    study_hours = st.number_input(
        "Study Hours (per day)",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.5
    )

    if st.button("Predict Marks"):
        predicted = model.predict([[study_hours]])[0]
        predicted = max(0, min(100, round(predicted, 1)))  # keep between 0-100

        st.success(f"📝 Predicted Marks: **{predicted} / 100**")

        # Give a simple grade label
        if predicted >= 80:
            grade = "A (Excellent)"
            color = "green"
        elif predicted >= 60:
            grade = "B (Good)"
            color = "blue"
        elif predicted >= 45:
            grade = "C (Average)"
            color = "orange"
        else:
            grade = "F (Needs Improvement)"
            color = "red"

        st.markdown(f"**Estimated Grade:** :{color}[{grade}]")

        # Show a small chart with prediction point highlighted
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df["StudyHours"], df["Marks"], color="steelblue", alpha=0.5,
                   label="Existing Students")
        ax.scatter([study_hours], [predicted], color="red", s=120, zorder=5,
                   label=f"Your Prediction ({study_hours} hrs → {predicted} marks)")

        # Regression line
        x_line = np.linspace(0, 12, 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        ax.plot(x_line, np.clip(y_line, 0, 100), color="orange", linewidth=2,
                linestyle="--", label="Regression Line")

        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Marks")
        ax.set_title("Prediction vs Existing Data")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.caption("Model used: Linear Regression | Dataset: student_data.csv")
