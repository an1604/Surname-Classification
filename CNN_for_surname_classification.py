# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:22:31 2023

@author: adina
"""
# imports :
import string

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

"""The Vocabulary:"""
class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""
    def __init__(self , token_to_idx = None , add_unk = True , unk_token ='<UNK>'):
        """Args :
           token_to_idx(dict) :  a pre-existing map of tokens to indices
           add_unk(bool) :a flag that indicates whether to add the UNK token
           unk_token(str) : the UNK token to add into the Vocabulary"""
        
        # checking for the token_to_idx dictionary 
        if token_to_idx is None :
               token_to_idx ={}
        self._token_to_idx  = token_to_idx 
        
        # create the mapping 
        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_index = -1 
        if add_unk:
            self.unk_index = self.add_token(unk_token)
        
    """to_serializable and from_serializable gonna be unused!"""
    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx':self._token_to_idx,
                'add_unk':self._add_unk,
                'unk_token': self._unk_token}
    
    classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self,token):
        """Update mapping dicts based on the token.
        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token"""
        try:
            index = self._token_to_idx[token]
        except KeyError:
            # adding another index to the dictionary
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            # updating this new place in the dictionary
            self._idx_to_token[index] = token
        # returning the right index at the end 
        return index 

    def lookup_token(self, token):
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]
    
    def __str__(self):
       return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
       return len(self._token_to_idx)

"""The SurnameDataset Class :
    the dataset is composed of a matrix of one­hot vectors rather than a collapsed one­hot vector
"""
class SurnameDataset(Dataset):
    def __init__(self, surname_df,vectorizer):
        """Args:
            surname_df (pandas.DataFrame): the dataset
        """
        self.surname_df = surname_df
        # from the df generate the vectorizer
        self._vectorizer = vectorizer
        # splitting the data to 3 - train , validation and test 
        np.random.seed(0)
        self.surname_df['split'] = np.random.choice(['train', 'val', 'test'], size=len(self.surname_df), p=[0.7, 0.15, 0.15])
        # Split the data into separate DataFrames
        self.train_df = self.surname_df[self.surname_df['split'] == 'train']
        self.val_df = self.surname_df[self.surname_df['split'] == 'val']
        self.test_df = self.surname_df[self.surname_df['split'] == 'test']
        # Kepping the sizes
        self.train_size = len(self.train_df)
        self.validation_size = len(self.val_df)
        self.test_size = len(self.test_df)
        
        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')
        
        # generating class weights :
        class_counts = surname_df.nationality.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.nationality_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)
       
    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        """Load dataset and make a new vectorizer from scratch
        
        Args:
            surname_csv (str): location of the dataset
        Returns:
            an instance of SurnameDataset
        """
        surname_df = pd.read_csv(surname_csv)
        surname_df['split'] = np.random.choice(['train', 'val', 'test'], size=len(surname_df), p=[0.7, 0.15, 0.15])
        train_surname_df = surname_df[surname_df.split=='train']
        return cls(surname_df, SurnameVectorizer.from_dataframe(train_surname_df))


    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer
    
    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
        
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        surname_matrix = \
            self._vectorizer.vectorize(row.surname)

        nationality_index = \
            self._vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {'x_surname': surname_matrix,
                'y_nationality': nationality_index}

"""The Vectorizer class"""
class SurnameVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self , surname_vocab , nationality_vocab,max_surname_length):
        """Args:
            surname_vocab (Vocabulary): maps characters to integers
            nationality_vocab (Vocabulary): maps nationalities to integers
            max_surname_length (int): the length of the longest surname
        """
        self.surname_vocab= surname_vocab
        self.nationality_vocab = nationality_vocab
        self._max_surname_length = max_surname_length

    
    def vectorize(self, surname):
        """Args: 
            surname (str): the surname
        Returns :
            one_hot_matrix (np.ndarray): a matrix of one­hot vectors
        """
        one_hot_matrix_size = (len(self.surname_vocab),self._max_surname_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size,dtype = np.float32)
        
        for position_index , character in enumerate(surname):
            # compute the index of the character from the token dictionary
            character_index = self.surname_vocab.lookup_token(character)
            # updating in the right place in the matrix 
            one_hot_matrix[character_index][position_index] =1 
        return one_hot_matrix
    
    @classmethod
    def from_dataframe(cls, surname_df):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            surname_df (pandas.DataFrame): the surnames dataset
        Returns:
            an instance of the SurnameVectorizer
        """
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0

        for index, row in surname_df.iterrows():
            max_surname_length = max(max_surname_length, len(row.surname))
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab, max_surname_length)

    @classmethod
    def from_serializable(cls, contents):
        surname_vocab = Vocabulary.from_serializable(contents['surname_vocab'])
        nationality_vocab =  Vocabulary.from_serializable(contents['nationality_vocab'])
        return cls(surname_vocab=surname_vocab, nationality_vocab=nationality_vocab, 
                   max_surname_length=contents['max_surname_length'])

    def to_serializable(self):
        return {'surname_vocab': self.surname_vocab.to_serializable(),
                'nationality_vocab': self.nationality_vocab.to_serializable(), 
                'max_surname_length': self._max_surname_length}
        
        
        
"""The classifier class"""    
class SurnameClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        """Args : 
            initial_num_channels (int): size of the incoming feature vector
            num_classes (int): size of the output prediction vector
            num_channels (int): constant channel size to use throughout network
        """
        # calling the super constructor 
        super(SurnameClassifier, self).__init__()
        # Creating the neural net - 3 CONV1D layers and foreach 1 ELU (activation function) layer
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels, 
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3),
            nn.ELU()
        )
        # The fully-connected layer 
        self.fc = nn.Linear(num_channels, num_classes) 
        
    def forward(self, x_surname, apply_softmax=False):
        """ The forward pass of the classifier.
        Args:
            x_surname (torch.Tensor): an input data tensor.
             x_surname.shape should be (batch, initial_num_channels, max_surname_length)
           apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
            """
        features = self.convnet(x_surname).squeeze(dim=2)
        
        # Going throught the fully connected layer 
        prediction_vector = self.fc(features)
        # Checking for the softmax 
        if apply_softmax:
          prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector

    

"""The Training routine:"""
def compute_accuracy(y_pred, y_target):
    y_pred_indices = y_pred.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


# initializing
dataset = SurnameDataset.load_dataset_and_make_vectorizer('surnames.csv')        
vectorizer = dataset.get_vectorizer()
classifier = SurnameClassifier(initial_num_channels=len(vectorizer.surname_vocab), 
                               num_classes=len(vectorizer.nationality_vocab),
                               num_channels= 256)
if classifier:
    print('classifier succesfullt created!')


"""Training loop"""
#We will train the model on the cpu 
classifier = classifier.to('cpu')
dataset.class_weights = dataset.class_weights.to('cpu')    
# Properties
hidden_dim=100
learning_rate=0.001
batch_size=128
num_epochs=100
dropout_p=0.1  
    
loss_func = nn.CrossEntropyLoss(dataset.class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

# Splitting the data for the train & validation
dataset.set_split('train')   
dataset.set_split('val')    
try:
    for epoch_index in range(num_epochs):
            dataset.set_split('train')
            batch_generator = generate_batches(dataset, 
                                              batch_size=batch_size, 
                                              device='cpu')
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()
        
    print('Done!')
    
    print('starting the training loop...')
    for batch_index, batch_dict in enumerate(batch_generator):
        # the training routine is these 5 steps:

            # --------------------------------------
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            y_pred = classifier(batch_dict['x_surname'])

            # step 3. compute the loss
            y_nationality = batch_dict['y_nationality']
            # reduce the dimentionality of y_nationality 
            loss = loss_func(y_pred,y_nationality)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            
            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()
            
            # compute the accuracy
            acc_t = compute_accuracy(y_pred,y_nationality)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
    print("Test loss: {};".format(running_loss))
    print("Test Accuracy: {}".format(running_acc))
        
    # Iterate over val dataset

        # setup: batch generator, set loss and acc to 0; set eval mode on
    print('Iterate over val dataset...')
    dataset.set_split('val')
    batch_generator = generate_batches(dataset, 
                                      batch_size=batch_size, 
                                      device='cpu')
    running_loss = 0.
    running_acc = 0.
    classifier.eval()
    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
            y_pred =  classifier(batch_dict['x_surname'])
            # step 3. compute the loss
            y_nationality = batch_dict['y_nationality']
            # reduce the dimentionality of y_nationality 
            loss = loss_func(y_pred,y_nationality)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # compute the accuracy
            acc_t = compute_accuracy(y_pred,y_nationality)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
    print("Test loss: {};".format(running_loss))
    print("Test Accuracy: {}".format(running_acc))
            
except KeyboardInterrupt:
    print("Exiting loop")

 
        
"""Test set"""

print("Tseting the model on the test set ...")
classifier = classifier.to('cpu')
dataset.class_weights = dataset.class_weights.to('cpu')
loss_func = nn.CrossEntropyLoss(dataset.class_weights)

dataset.set_split('test')
batch_generator = generate_batches(dataset, 
                                   batch_size=batch_size, 
                                   device='cpu')
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # compute the output
    y_pred =  classifier(batch_dict['x_surname'])
    # step 3. compute the loss
    y_nationality = batch_dict['y_nationality']
    # reduce the dimentionality of y_nationality 
    loss = loss_func(y_pred,y_nationality)
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)
    # compute the accuracy
    acc_t = compute_accuracy(y_pred,y_nationality)
    running_acc += (acc_t - running_acc) / (batch_index + 1)

print("Test loss: {};".format(running_loss))
print("Test Accuracy: {}".format(running_acc))


"""Prediction"""
def predict_nationality(surname, classifier, vectorizer):
    """Predict the nationality from a new surname.
    Args:
        surname (str): the surname to classifier
        classifier (SurnameClassifer): an instance of the classifier
        vectorizer (SurnameVectorizer): the corresponding vectorizer
    Returns: 
        a dictionary with the most likely nationality and its probability
    """
    vectorized_surname = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0)
    result = classifier(vectorized_surname, apply_softmax=True)

    probability_values, indices = result.max(dim=1)
    index = indices.item()

    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
    probability_value = probability_values.item()

    return {'nationality': predicted_nationality, 'probability': probability_value}


new_surname = input("Enter a surname to classify: ")
classifier = classifier.cpu()
prediction = predict_nationality(new_surname, classifier, vectorizer)
print("{} -> {} (p={:0.2f})".format(new_surname,
                                    prediction['nationality'],
                                    prediction['probability']))



"""Predction in top k"""

def predict_topk_nationality(surname, classifier, vectorizer, k=5):
    """Predict the top K nationalities from a new surname
    
    Args:
        surname (str): the surname to classifier
        classifier (SurnameClassifer): an instance of the classifier
        vectorizer (SurnameVectorizer): the corresponding vectorizer
        k (int): the number of top nationalities to return
    Returns:
        list of dictionaries, each dictionary is a nationality and a probability
    """
    vectorized_surname = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(dim=0)
    prediction_vector = classifier(vectorized_surname, apply_softmax=True)
    probability_values, indices = torch.topk(prediction_vector, k=k)
    
    # returned size is 1,k
    probability_values = probability_values[0].detach().numpy()
    indices = indices[0].detach().numpy()
    
    results = []
    for kth_index in range(k):
        nationality = vectorizer.nationality_vocab.lookup_index(indices[kth_index])
        probability_value = probability_values[kth_index]
        results.append({'nationality': nationality, 
                        'probability': probability_value})
    return results


new_surname = input("Enter a surname to classify: ")

k = int(input("How many of the top predictions to see? "))
if k > len(vectorizer.nationality_vocab):
    print("Sorry! That's more than the # of nationalities we have.. defaulting you to max size :)")
    k = len(vectorizer.nationality_vocab)
    
predictions = predict_topk_nationality(new_surname, classifier, vectorizer, k=k)

print("Top {} predictions:".format(k))
print("===================")
for prediction in predictions:
    print("{} -> {} (p={:0.2f})".format(new_surname,
                                        prediction['nationality'],
                                        prediction['probability']))








