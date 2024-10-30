import os
import requests
from tqdm import tqdm
import tarfile
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_files
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
 
 
nltk.download("stopwords")
 
 
# Download path and dataset variables
DATA_URL = "https://ndownloader.figshare.com/files/5975967"
DATA_DIR = "20_newsgroups"
ARCHIVE_FILE = "20_newsgroups.tar.gz"
DATA_DIR_TRAIN = os.path.join(DATA_DIR, "20news-bydate-train")
 
 
# Download dataset with progress bar
def download_dataset(url, output_path):
   response = requests.get(url, stream=True)
   total_size = int(response.headers.get('content-length', 0))
   with open(output_path, "wb") as file, tqdm(
       desc="Downloading 20 Newsgroups dataset",
       total=total_size,
       unit="B",
       unit_scale=True,
       unit_divisor=1024,
   ) as bar:
       for data in response.iter_content(chunk_size=1024):
           file.write(data)
           bar.update(len(data))
 
 
# Extract the dataset if not done already
def extract_dataset(archive_path, extract_to):
   if not os.path.isdir(extract_to):
       print("Extracting dataset...")
       with tarfile.open(archive_path, "r:gz") as tar:
           tar.extractall(path=extract_to)
 
 
# Check and prepare dataset
if not os.path.isdir(DATA_DIR):
   if not os.path.isfile(ARCHIVE_FILE):
       download_dataset(DATA_URL, ARCHIVE_FILE)
   extract_dataset(ARCHIVE_FILE, DATA_DIR)
 
 
# Verify dataset directory content
if os.path.isdir(DATA_DIR):
   print(f"Contents of '{DATA_DIR}':", os.listdir(DATA_DIR))
 
 
# Load dataset with selective categories
try:
   newsgroups = load_files(DATA_DIR_TRAIN, categories=["talk.politics.misc", "rec.sport.baseball", "sci.med", "comp.graphics"])
   if len(newsgroups.data) == 0:
       raise ValueError("Dataset is empty. Check if the files were extracted correctly.")
except Exception as e:
   print(f"Error loading dataset: {e}")
   exit()
 
 
# Decode documents and apply preprocessing
def preprocess_documents(documents):
    ps = PorterStemmer()
     
    # Define custom stop words for the categories
    custom_stop_words = stopwords.words("english") + [
        "politics", "political", "gun", "guns", "sports", "baseball", 
        "graphics", "graphic", "medicine", "medical", "health", "sci", 
        "science", "discussion", "topic", "newsgroup", "forum"
    ]
     
    processed_docs = []
    for doc in documents:
        try:
            decoded = doc.decode("utf-8", errors="ignore")
            stemmed = " ".join([ps.stem(word) for word in decoded.split() if word.lower() not in custom_stop_words])
            processed_docs.append(stemmed)
        except UnicodeDecodeError:
            continue
    return processed_docs
 
 
 
 
# Preprocess and split the dataset
X_data = preprocess_documents(newsgroups.data)
X_train, X_test, y_train, y_test = train_test_split(X_data, newsgroups.target, test_size=0.2, random_state=42)
 
 
# Vectorizer with custom stop words
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
 
 
# Parameter tuning for Logistic Regression
logreg = LogisticRegression(max_iter=1000)
param_grid = {'C': [0.1, 1, 10, 100]}
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X_train_tfidf, y_train)
 
 
# Models with tuned parameters
models = {
   "Naive Bayes": MultinomialNB(),
   "Logistic Regression": logreg_cv.best_estimator_,
   "SVM": SVC(probability=True)
}
 
 
# Training models and storing accuracies
model_pipelines = {}
accuracies = {}
for model_name, model in models.items():
   pipeline = make_pipeline(vectorizer, model)
   pipeline.fit(X_train, y_train)
   model_pipelines[model_name] = pipeline
   y_pred = pipeline.predict(X_test)
   accuracies[model_name] = accuracy_score(y_test, y_pred)
 
 
# Function to classify text based on selected model
def classify_text():
   user_text = input_text.get("1.0", "end-1c").strip()
   if not user_text:
       messagebox.showwarning("Input Required", "Please enter some text to classify.")
       return
 
 
   selected_model = model_choice.get()
   pipeline = model_pipelines[selected_model]
   prediction = pipeline.predict([user_text])[0]
   probabilities = pipeline.predict_proba([user_text])[0]
 
 
   # Display results
   result_text.set(f"Predicted Category: {newsgroups.target_names[prediction]}")
   prob_text = "\n".join([f"{newsgroups.target_names[i]}: {prob:.2%}" for i, prob in enumerate(probabilities)])
   prob_label.config(text=f"Prediction Probabilities:\n{prob_text}")
 
 
# Function to clear input and output fields
def clear_text():
   input_text.delete("1.0", tk.END)
   result_text.set("")
   prob_label.config(text="")
 
 
# Tkinter UI setup
app = tk.Tk()
app.title("Text Classifier - The Pycodes")
app.geometry("600x600")
 
 
# Input Text Label
tk.Label(app, text="Enter text to classify:", font=("Arial", 12)).pack(pady=10)
 
 
# Text Box for Input
input_text = tk.Text(app, height=8, width=60, font=("Arial", 10))
input_text.pack(pady=10)
 
 
# Classifier Choice Dropdown
model_choice = ttk.Combobox(app, values=list(models.keys()), font=("Arial", 10))
model_choice.set("Naive Bayes")
model_choice.pack(pady=10)
 
 
# Show training accuracy of selected model
accuracy_text = tk.StringVar()
accuracy_text.set(f"Training Accuracies:\n" + "\n".join([f"{model}: {acc:.2%}" for model, acc in accuracies.items()]))
accuracy_label = tk.Label(app, textvariable=accuracy_text, font=("Arial", 10), fg="blue")
accuracy_label.pack(pady=5)
 
 
# Classify Button
classify_button = tk.Button(app, text="Classify Text", command=classify_text, font=("Arial", 12), bg="lightgreen")
classify_button.pack(pady=5)
 
 
# Result Label
result_text = tk.StringVar()
result_label = tk.Label(app, textvariable=result_text, font=("Arial", 14), fg="green")
result_label.pack(pady=10)
 
 
# Prediction Probabilities Label
prob_label = tk.Label(app, text="", font=("Arial", 10), fg="purple")
prob_label.pack(pady=10)
 
 
# Clear Button
clear_button = tk.Button(app, text="Clear Text", command=clear_text, font=("Arial", 12), bg="lightcoral")
clear_button.pack(pady=5)
 
 
# Run the Tkinter main loop
app.mainloop()
