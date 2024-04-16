import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Load data
df = pd.read_csv("dataset path")
df['text'] = df['TITLE'] + ' ' + df['ABSTRACT']
df['text'] = df['text'].apply(lambda x: x.lower().replace('_', ' '))

# Prepare the labels
labels = df.iloc[:, 3:9].values  # Assuming labels are in columns 4-9
label_tensor = torch.tensor(labels, dtype=torch.float)

# Text vectorization
tfidf = TfidfVectorizer(max_features=50000, stop_words='english', min_df=0.01, max_df=0.3)
tfidf_matrix = tfidf.fit_transform(df['text']).toarray()
tfidf_tensor = torch.tensor(tfidf_matrix, dtype=torch.float)

# Parameters
n_samples, n_features = tfidf_tensor.shape
n_components = label_tensor.shape[1]  # Number of topics equals number of labels

# Define the VB-NMF model
class VBNMF(nn.Module):
    def __init__(self, n_samples, n_features, n_components):
        super(VBNMF, self).__init__()
        # Variational parameters for W
        self.mean_W = nn.Parameter(torch.randn(n_samples, n_components))
        self.log_var_W = nn.Parameter(torch.randn(n_samples, n_components))

        # Variational parameters for H
        self.mean_H = nn.Parameter(torch.randn(n_components, n_features))
        self.log_var_H = nn.Parameter(torch.randn(n_components, n_features))

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self):
        W = self.reparameterize(self.mean_W, self.log_var_W)
        H = self.reparameterize(self.mean_H, self.log_var_H)
        return torch.relu(W @ H)  # Ensure non-negativity

# Model, Loss, and Optimizer
model = VBNMF(n_samples, n_features, n_components)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Loss function (ELBO)
def loss_function(recon_x, x, labels, mean_W, log_var_W, mean_H, log_var_H):
    MSE = torch.mean((recon_x - x) ** 2)
    Label_Loss = torch.mean((mean_W - labels) ** 2)  # Encourage alignment with labels
    KLD_W = -0.5 * torch.sum(1 + log_var_W - mean_W.pow(2) - log_var_W.exp())
    KLD_H = -0.5 * torch.sum(1 + log_var_H - mean_H.pow(2) - log_var_H.exp())
    return MSE + Label_Loss + KLD_W + KLD_H

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    recon_batch = model()
    loss = loss_function(recon_batch, tfidf_tensor, label_tensor, model.mean_W, model.log_var_W, model.mean_H, model.log_var_H)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Extract top 10 words for each topic
feature_names = tfidf.get_feature_names_out()
with torch.no_grad():
    H_matrix = model.mean_H.detach().numpy()
top_words_per_topic = []
for i, topic_weights in enumerate(H_matrix):
    top_indices = topic_weights.argsort()[-10:][::-1]
    top_words = [feature_names[j] for j in top_indices]
    top_words_per_topic.append(top_words)
    print(f'Topic {df.columns[3+i]}: {", ".join(top_words)}')

