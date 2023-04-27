from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (classification_report, accuracy_score, 
                            f1_score, roc_auc_score, balanced_accuracy_score)
from sklearn.model_selection import train_test_split


fname="./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
all_locuteur_df = load_pres(fname)

fname_test="./datasets/AFDpresidentutf8/corpus.tache1.test.utf8"
all_locuteur_test_df = load_pres_test(fname_test)

learning_rate = 0.001
num_epochs = 10
batch_size = 32

X_text_train, X_text_test, y_train, y_test = train_test_split(all_locuteur_df['text'], 
    all_locuteur_df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_text_train)
X_test = vectorizer.transform(X_text_test)

y_train = torch.tensor(list(y_train))
y_test = torch.tensor(list(y_test))


class BoWDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = BoWDataset(X_train, y_train)
test_dataset = BoWDataset(X_test, y_test)


class BoWModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BoWModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = self.fc(x)
        return x


model = BoWModel(input_size=X_train.shape[1], output_size=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
predictions = []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        predictions.extend(predicted.tolist())

print(classification_report(y_test.tolist(), predictions, target_names=[-1, 1]))
print('Accuracy: {:.4f}'.format(accuracy_score(y_test.tolist(), predictions)))

unique, counts = torch.unique(y_train, return_counts=True)
class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)
class_weights = class_weights / torch.sum(class_weights)

predictions_minority = [predictions[i] for i in range(len(predictions)) if y_test[i] == torch.argmin(class_weights)]
y_test_minority = [y_test[i] for i in range(len(y_test)) if y_test[i] == torch.argmin(class_weights)]

print('F1 score (minority class): {:.4f}'.format(f1_score(y_test_minority, predictions_minority)))
print('AUC: {:.4f}'.format(roc_auc_score(y_test.tolist(), predictions)))
print('Balanced Accuracy: {:.4f}'.format(balanced_accuracy_score(y_test.tolist(), predictions)))
print('F1 score (micro): {:.4f}'.format(f1_score(y_test.tolist(), predictions, average='micro')))
print('F1 score (macro): {:.4f}'.format(f1_score(y_test.tolist(), predictions, average='macro')))