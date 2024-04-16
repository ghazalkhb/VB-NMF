import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the dataset from CSV
df = pd.read_csv("C:/Users/acer/Desktop/5P77/Project/VBNMF/train.csv")

# Combine the title and abstract columns into a single text column
df['text'] = df['TITLE'] + ' ' + df['ABSTRACT']

# Function to preprocess text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove all numbers (attached or standalone) and underscores
    text = re.sub(r'\d+', '', text)  # Removes all digits
    text = re.sub(r'_', ' ', text)   # Replace underscores with spaces to ensure they are not part of tokens
    # Tokenize and remove words with less than 5 characters
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 4]
    return ' '.join(tokens)

# Apply preprocessing to the text column
df['text'] = df['text'].apply(preprocess_text)

# Extract the text and topic labels
texts = df['text'].tolist()
labels = df.iloc[:, 3:-1].values.astype(int)
label_names = df.columns[3:-1]  # Extract label names from the DataFrame

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert text to TF-IDF features with sparse matrices
vectorizer = TfidfVectorizer(max_features=50000, stop_words='english', token_pattern=r'\b[a-zA-Z]{5,}\b', min_df=0.01)
tfidf_matrix_train = vectorizer.fit_transform(X_train)
tfidf_matrix_test = vectorizer.transform(X_test)


# Convert the TF-IDF matrices to PyTorch sparse tensors
X_train_tensor_sparse = torch.tensor(tfidf_matrix_train.toarray(), dtype=torch.float32)
X_test_tensor_sparse = torch.tensor(tfidf_matrix_test.toarray(), dtype=torch.float32)

# Define a simple classifier model
class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))  # Sigmoid activation for multi-label classification
        return x

# Instantiate the classifier model
input_size = X_train_tensor_sparse.shape[1]
output_size = y_train.shape[1]  # Number of topics
model = Classifier(input_size, output_size)

# Define the loss function for multi-label classification
loss_fn = nn.BCELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train the classifier model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor_sparse)
    loss = loss_fn(outputs, torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Extract the top words for each topic
top_words_per_topic = []
with torch.no_grad():
    for i in range(output_size):
        topic_weights = model.fc.weight[i].numpy()
        top_word_indices = np.argsort(topic_weights)[::-1][:10]  # Indices of top 10 words
        top_words = [vectorizer.get_feature_names_out()[idx] for idx in top_word_indices]
        top_words_per_topic.append(top_words)

# Print the top words for each topic along with the name of the topic
for i, (label_name, top_words) in enumerate(zip(label_names, top_words_per_topic)):
    print(f"{label_name}: {', '.join(top_words)}")

# Evaluate the model on the testing set
with torch.no_grad():
    outputs = model(X_test_tensor_sparse)
    # Round the predictions to 0 or 1
    predicted_labels = torch.round(outputs).numpy()
    # Calculate accuracy for each label
    label_accuracy = (predicted_labels == y_test).mean(axis=0)
    # Calculate overall accuracy
    overall_accuracy = np.mean(predicted_labels == y_test)
    print(f'Overall Accuracy: {overall_accuracy:.4f}')
    print('Label-wise Accuracy:')
    for i, acc in enumerate(label_accuracy):
        print(f'{label_names[i]}: {acc:.4f}')
