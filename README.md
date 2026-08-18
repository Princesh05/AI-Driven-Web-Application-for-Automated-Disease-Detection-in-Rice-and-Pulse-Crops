
# 🌱 AI-Driven Web Application for Plant Disease Detection

## 📌 Overview

**AI Plant Disease Detection** is a deep-learning-powered web application that automatically identifies plant diseases from leaf images.

The application allows users to create an account, upload a leaf image, obtain an AI-based disease prediction with confidence scores, view alternative predictions, receive AI-generated treatment recommendations, and maintain a history of previous disease detections.

The system integrates a trained TensorFlow/Keras image-classification model with a Streamlit web interface, SQLite database, and Google Gemini API for generating plant-care recommendations.

---

## 🎯 Problem Statement

Plant diseases can significantly affect crop productivity and quality. Identifying diseases manually from leaf symptoms can be difficult, time-consuming, and dependent on agricultural expertise.

This project aims to provide an automated image-based disease detection system that can assist users in identifying possible plant diseases from leaf images using deep learning.

---

## 🚀 Objectives

* Develop an AI-based system for automated plant disease detection.
* Classify plant diseases from leaf images using deep learning.
* Provide confidence scores and the top predicted disease classes.
* Integrate the trained model into an interactive web application.
* Store user accounts and prediction history using SQLite.
* Generate AI-based treatment and prevention recommendations.
* Provide downloadable PDF reports for disease analysis.

---

## ✨ Key Features

### 🔬 AI-Based Disease Detection

Users can upload a leaf image through the Streamlit application. The image is preprocessed according to the trained model's input requirements and passed to the TensorFlow/Keras model for prediction.

### 📊 Top-3 Predictions

Instead of showing only one prediction, the application calculates and displays the **top three predicted disease classes with their confidence scores**.

### 🌿 Multiple Plant Classes

The application supports 44 defined classes covering healthy and diseased plants across multiple crops, including:

* Rice
* Tomato
* Potato
* Apple
* Corn/Maize
* Grape
* Peach
* Pepper
* Strawberry
* Orange
* Cassava
* Cherry
* Squash

The complete class mapping is defined in the application.

### 🤖 AI Treatment Recommendations

After detecting a disease, the application sends the predicted disease name to the **Google Gemini API** and generates a concise treatment recommendation covering:

* Prevention
* Immediate Treatment
* Best Practices

The Gemini API key is loaded through an environment variable rather than being hard-coded in the application.

### 👤 User Authentication

The application provides:

* User registration
* Login
* Profile information
* Profile picture upload
* Account settings
* Logout

User information is stored in an SQLite database.

### 📜 Prediction History

The application stores previous predictions for each user, including:

* Disease name
* Confidence score
* Leaf image
* Top-3 predictions
* Prediction date and time

Users can view their previous detections and delete individual history records.

### 📄 PDF Report Generation

After disease detection and AI treatment recommendation, the application generates a downloadable PDF report containing the AI treatment recommendation.

---

## 🧠 Machine Learning Workflow

The overall workflow of the application is:

```text
                Leaf Image
                    │
                    ▼
             Image Upload
                    │
                    ▼
          Image Preprocessing
                    │
                    ▼
        Trained TensorFlow/Keras
              Deep Learning Model
                    │
                    ▼
            Disease Prediction
                    │
             ┌──────┴──────┐
             ▼             ▼
       Top Prediction    Top-3 Results
             │
             ▼
      Gemini AI Treatment
          Recommendation
             │
             ▼
       SQLite Prediction
           History
             │
             ▼
        PDF Report
```

The application loads the trained Keras model and determines its expected input dimensions before performing prediction.

The uploaded image is resized, converted to an array, adjusted for the required number of channels, preprocessed, and passed to the model.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* Trained `.keras` model

### Image Processing

* Pillow (PIL)
* NumPy

### Web Application

* Streamlit

### Database

* SQLite

### Generative AI

* Google Gemini API

### PDF Generation

* FPDF

### Environment & Configuration

* Python-dotenv
* Google Colab
* Visual Studio Code

The main application imports and uses these components directly in `app.py`.

---

## 🗄️ Database Design

The application uses SQLite to manage user and prediction information.

### Users Table

Stores:

* User ID
* First name
* Last name
* Email
* Password
* Profile picture
* Account creation date

### Predictions Table

Stores:

* Prediction ID
* User email
* Leaf image path
* Predicted disease
* Confidence score
* Top-3 alternatives
* Plant type
* Prediction date and time

The database tables are initialized automatically when the application starts.

---

---

## 🔐 Environment Variables

The application uses the Gemini API through an environment variable:

```text
GEMINI_API_KEY=your_api_key_here
```

The actual `.env` file should **never be uploaded to GitHub**.

Use `.env.example` instead:

```text
GEMINI_API_KEY=
```

The application loads the environment variables using `python-dotenv`.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd AI-Plant-Disease-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Gemini API

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key
```

### 6. Update local paths

Before running the application, update the model and database paths in `app.py` to match your local project structure.

The current version of `app.py` contains Windows-specific absolute paths, so these should be changed before publishing the project for other users.

### 7. Run the application

```bash
streamlit run app.py
```

---

## 🔄 Application Workflow

1. User opens the application.
2. User creates an account or logs in.
3. User enters the disease detection dashboard.
4. User uploads a plant leaf image.
5. The image is preprocessed.
6. The trained deep learning model analyzes the image.
7. The application displays the predicted disease and confidence score.
8. The top three predictions are displayed.
9. The prediction is saved to the user's history.
10. Gemini generates treatment and prevention recommendations.
11. The user can download a PDF treatment report.
12. Previous predictions can be viewed from the prediction history section.

---

## 📈 Results

The application provides an end-to-end workflow from **leaf image upload → AI disease prediction → confidence analysis → AI-generated treatment recommendation → prediction history → downloadable report**.

The application's user interface currently presents an 88% accuracy figure; if this figure is included in the final README, it should be backed by the model's actual evaluation results from the training notebook rather than only the UI text.

---

## 🔮 Future Improvements

* Improve model accuracy using additional training data.
* Add more crop and disease classes.
* Improve password security using password hashing.
* Replace absolute local file paths with configurable project-relative paths.
* Add model explainability using techniques such as Grad-CAM.
* Deploy the application to a cloud platform.
* Add multilingual support for farmers.
* Improve mobile responsiveness.
* Add more detailed agricultural recommendations.
* Add model performance monitoring.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes. AI-generated disease predictions and treatment recommendations should not be considered a substitute for professional agricultural or plant-pathology advice.

---

## 👨‍💻 Project

**AI-Driven Web Application for Automated Plant Disease Detection**

Developed as an AI/ML project involving deep learning, computer vision, generative AI, database integration, and web application development.
