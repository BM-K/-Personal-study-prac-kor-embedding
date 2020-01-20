import time
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from konlpy.tag import Mecab
from torchtext.data import TabularDataset
from torchtext.data import Iterator
from torchtext.vocab import Vectors
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

    # input_dim = len(TEXT.vocab)
    # embedding_dim = 160 # kr-data 벡터 길이
    # hidden_dim = 256
    # output_dim = 1 # sentiment analysis

def tokenizer1(text):
    result_text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》;]', '', text)
    a = Mecab().morphs(result_text)
    return( [a[i] for i in range(len(a))] )

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 batch_size, dropout, pad_idx):
        super().__init__()
        vocab = TEXT.vocab

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)

        self.fc1 = nn.Linear(hidden_dim*2, 128)
        self.fc2 = nn.Linear(128, output_dim)

        self.fc = nn.Linear(hidden_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, hidden):
        # text = [sent len, batch size]
        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.gru(embedded, hidden)
        # output = [sent len, batch size, hid dim*2(numlayer)]
        # hidden = [4, batch size, hid dim]

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # batch_dim x hid_dim*2     히든사이즈 4개의 의미를 알아야 이것을 푼다. 논문 세미나 끝나고ㄱ

        return torch.sigmoid(self.fc(hidden))

    def init_hidden(self):
        result = Variable(torch.rand(4, self.batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        hidden = model.init_hidden()
        hidden = hidden.data

        predictions = model(batch.text, hidden).squeeze(1)

        batch.label = batch.label.float()
        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            hidden = model.init_hidden()
            hidden = hidden.data

            predictions = model(batch.text, hidden).squeeze(1)

            batch.label = batch.label.float()
            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

train_df = pd.read_table('ratings_train.txt')
test_df = pd.read_table('ratings_test.txt')

ID = data.Field(sequential=False,
                use_vocab=False)

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer1,
                  lower=True,
                  batch_first=False,
                  fix_length=20,
                  )

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True,
                   )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data = TabularDataset.splits(
    path='.', train='ratings_train.txt', test='ratings_test.txt', format='tsv',
    fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True
) # train과 text파일이 현재 디렉토리에 있어햐함. 여기서 tsv로 구분된 것 사용. 첫 번째 header는 무시

batch_size = 16
train_loader = Iterator(dataset=train_data, batch_size=batch_size, device=device)
test_loader = Iterator(dataset=test_data, batch_size=batch_size, device=device)

vectors = Vectors(name="kr-projected.txt")

TEXT.build_vocab(train_data, vectors=vectors, min_freq=5, max_size=15000)

if __name__ == '__main__':
    print("good")
    input_dim = len(TEXT.vocab)
    embedding_dim = 160 # kr-data 벡터 길이
    hidden_dim = 256
    output_dim = 1
    dropout = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, batch_size, dropout, PAD_IDX)

    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.cuda()
    criterion = nn.BCELoss()

    N_EPOCHS = 4

    best_valid_loss = float('inf')
    x_epoch = []
    drow_loss = []
    drow_acc = []
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion)
        x_epoch.append(epoch)
        drow_loss.append(valid_loss)
        drow_acc.append(valid_acc)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    # import matplotlib.pyplot as plt
    # plt.plot(x_epoch, drow_acc)  # x측 = x_epoch, y축 = y_loss
    # plt.ylabel('acc')  # y축 라벨을 붙여준다.
    # plt.xlabel('Epoch')  # x축 라벨을 붙여준다.
    # plt.draw()
    # fig = plt.gcf()
    # fig.savefig('one_model_acc.png', dpi=fig.dpi)
    #
    # plt.plot(x_epoch, drow_loss)  # x측 = x_epoch, y축 = y_loss
    # plt.ylabel('loss')  # y축 라벨을 붙여준다.
    # plt.xlabel('Epoch')  # x축 라벨을 붙여준다.
    # plt.draw()
    # fig = plt.gcf()
    # fig.savefig('one_model_loss.png', dpi=fig.dpi)