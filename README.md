# Aspect-Based Sentiment Analysis on Student Feedback

This project performs Aspect-Based Sentiment Analysis (ABSA) on student feedback, extracting sentiments for various course aspects to support educational improvements. Using both traditional machine learning algorithms and deep learning models (including BERT), this analysis classifies sentiments and visualizes insights into student satisfaction with course elements such as content, delivery, and instructors.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

## Project Overview
The objective of this project is to leverage aspect-based sentiment analysis to provide a nuanced understanding of student feedback. By analyzing sentiments across specific course aspects, insights into student satisfaction and areas needing improvement can be highlighted. This project uses a range of machine learning and natural language processing techniques for sentiment classification, word frequency analysis, and sentiment visualization.

## Features
- **Text Preprocessing**: Cleaning and preparing text data by removing URLs, special characters, stop words, and performing tokenization and lemmatization.
- **Sentiment Classification**: Classifies feedback as positive, negative, or neutral using machine learning models.
- **Aspect-Specific Analysis**: Sentiment scores by course aspect and feedback trends.
- **Data Visualization**: Displays sentiment distributions, word clouds, and sentiment heatmaps.

## Dataset
The dataset consists of student feedback on various courses, with entries containing textual feedback and sentiment labels. The feedback data is preprocessed, and aspects of the data (such as course satisfaction and study hours) are analyzed for additional insights.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vaibhavzambre/Aspect-Based-Sentiment-Analysis-on-Student-Feedback.git
2.Navigate to the project directory:
   cd Aspect-Based-Sentiment-Analysis-on-Student-Feedback
3.Ensure you have Python 3.x installed.

## Usage
1. **Preprocess the Data**: The notebook and script contain preprocessing functions to clean the feedback data.
2. **Train Models**: Execute cells in the notebook to train models on the preprocessed data.
3. **Analyze Results**: Visualize sentiment distribution across different course aspects, analyze top words, and view model performance metrics.

To run the notebook interactively:
```bash
jupyter notebook Aspect_Based_Sentiment_Analysis_on_Student_Feedback.ipynb


## Models Used
The project applies several machine learning algorithms, comparing their performance:
- **XGBoost Classifier**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Naive Bayes Classifier**
- **BERT with KerasNLP** for deep learning-based sentiment classification

Each model's accuracy, precision, recall, and F1 scores are evaluated. Cross-validation techniques are applied to optimize the XGBoost and SVM models.

## Results
The sentiment analysis provides insights into:
- Distribution of positive, neutral, and negative sentiments across course aspects
- Keyword frequency analysis for identifying commonly used feedback terms
- Correlations between sentiment scores and overall course satisfaction
- Visualizations including word clouds, sentiment heatmaps, and distribution charts by course code

### Sample Output Visualizations
- **Word clouds** for positive and negative sentiments
- **Correlation heatmap** between study hours and satisfaction
- **Bar plots** of sentiment counts by course difficulty

### Sample Output Visualizations
- **Word clouds** for positive and negative sentiments
- **Correlation heatmap** between study hours and satisfaction
- **Bar plots** of sentiment counts by course difficulty

## Technologies Used
- **Languages**: Python
- **Libraries**: NLTK, Scikit-learn, Seaborn, Matplotlib, WordCloud, TextBlob, XGBoost
- **Deep Learning**: KerasNLP, Transformers, TensorFlow
- **Tools**: Jupyter Notebook, Google Colab

## Contributors
- **Vaibhav Zambre**
  - Email: vaibhavzambre15@gmail.com
  - [GitHub Profile](https://github.com/vaibhavzambre)
