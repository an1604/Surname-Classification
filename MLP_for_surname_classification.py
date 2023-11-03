"""An example instantiation of an MLP"""
# imports :
import string

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

"""Data Vectorization classes:"""

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



"""The Vectorizer class"""
class SurnameVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self , surname_vocab , nationality_vocab):
        """Args:
            surname_vocab (Vocabulary): maps characters to integers
            nationality_vocab (Vocabulary): maps nationalities to integers
        """
        self.surname_vocab= surname_vocab
        self.nationality_vocab = nationality_vocab
    
    def vectorize(self,surname):
        """Args:
            surname (str): the surname
        Returns:
            one_hot (np.ndarray): a collapsed one-hot encoding 
        """
        # saving the mappinf of the surnames in a variable
        vocab = self.surname_vocab
        # creating the one_hot in the vocabulary len
        one_hot = np.zeros(len(vocab) , dtype=np.float32)
        # iterating throught the surname and fill the place in one_hot from the vocabulary
        for token in surname:
            # filling 1 in the index associated with the token 
            one_hot[vocab.lookup_token(token)] = 1 
        return one_hot
    
    @classmethod
    def from_dataframe(cls, surname_df):
        """Generate the vectorizer from the dataset dataframe.
        Args:
            surname_df (pandas.DataFrame): the surnames dataset
        Returns:
            an instance of the SurnameVectorizer
        """
        # creating vocabularies for the df 
        # in surname_vocab we will get unknown tokens (names) so we set the unk_token
        surname_vocab = Vocabulary(unk_token='@')
        # in the nationality_vocab we will not get unknown tokens so we set it to false
        nationality_vocab = Vocabulary(add_unk=False)
        # Iterate over the rows using iterrows
        for index,row in surname_df.iterrows():
            # # Iterate over the letters in each row
            for letter in row.surname:
                # adding the letter as token to the vocabulary
                surname_vocab.add_token(letter)
            # addingt he nationality as token too
            nationality_vocab.add_token(row.nationality)
        return cls(surname_vocab,nationality_vocab)
    
    """Again these 2 functions are unused!"""
    @classmethod
    def from_serializable(cls, contents):
        surname_vocab = Vocabulary.from_serializable(contents['surname_vocab'])
        nationality_vocab =  Vocabulary.from_serializable(contents['nationality_vocab'])
        return cls(surname_vocab=surname_vocab, nationality_vocab=nationality_vocab)

    def to_serializable(self):
        return {'surname_vocab': self.surname_vocab.to_serializable(),
                'nationality_vocab': self.nationality_vocab.to_serializable()}
    
"""The Dataset class:"""
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
        """the primary entry point method for PyTorch datasets.
        Args:
            index (int): the index to the data point
        Returns :
            a dictionary holding the data point's:
                features (x_surname)
                label (y_nationality)
        """
        row = self._target_df.iloc[index]
        surname_vector = self._vectorizer.vectorize(row.surname)
        nationality_vector = self._vectorizer.vectorize(row.nationality)
        return {'x_surname' : surname_vector,
                'y_nationality' : nationality_vector}
    
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
    
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
        """ A generator function which wraps the PyTorch DataLoader. It will 
            ensure each tensor is on the write device location. 
        """
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
        
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict
            
"""The Model: SurnameClassifier"""
class SurnameClassifier(nn.Module):
    """ A 2-layer Multilayer Perceptron for classifying surnames """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        # calling the super class init first
        super(SurnameClassifier, self).__init__()
        # initialize the fully-connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    # the forward propagation progress
    def forward(self ,x_in , apply_softmax = False):
        """The forward pass of the classifier.
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
           apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_softmax:
           prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector
    
    
"""Training Routine"""

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
}

def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

# initializing
dataset = SurnameDataset.load_dataset_and_make_vectorizer('surnames.csv')        
vectorizer = dataset.get_vectorizer()
classifier = SurnameClassifier(input_dim = len(vectorizer.surname_vocab),
                               hidden_dim =300
                               , output_dim = len(vectorizer.nationality_vocab))
if classifier:
    print('classifier succesfullt created!')

"""Training loop"""
#We will train the model on the cpu 
classifier = classifier.to('cpu')
dataset.class_weights = dataset.class_weights.to('cpu')

nb_epochs = 100
learning_rate=0.001
batch_size=64

loss_func = nn.CrossEntropyLoss(dataset.class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    
try:
    print('Iterate over training dataset and generate the batches...')
    for epoch_index in range(nb_epochs):
        # Iterate over training dataset
        # setup: batch generator, set loss and acc to 0, set train mode on
        dataset.set_split('train')
        batch_generator = generate_batches(dataset,
                                           batch_size = batch_size,
                                           device ='cpu')
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
    
    print('Done!')
    
    print('Starting the training foreacg batch...')
    for batch_index, batch_dict in enumerate(batch_generator):
        # the training routine is these 5 steps:
        
        # step 1. zero the gradients
            optimizer.zero_grad()
        
        # step 2. compute the output
            y_pred = classifier(batch_dict['x_surname'])
            
        # step 3. compute the loss
            y_nationality = batch_dict['y_nationality']
            # reduce the dimentionality of y_nationality 
            y_nationality  = torch.argmax(y_nationality, dim=1)
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
            
            print('batch : ' +str(batch_index) + ' loss: ' + str(running_loss)+
                  ' running accuracy : ' + str(running_acc) )

except KeyboardInterrupt:
    print("Exiting loop")


"""Test set :"""
print('Testing the model on the Test set...')
classifier = classifier.to('cpu')
dataset.class_weights = dataset.class_weights.to('cpu')
loss_func = nn.CrossEntropyLoss(dataset.class_weights)

# split the dataset to the test 
dataset.set_split('test')
# generate the bacthes according to the test set 
batch_generator = generate_batches(dataset, batch_size=batch_size, shuffle=False, drop_last=False, device='cpu')

running_loss = 0.
running_acc = 0.
# get rid of the SGD progress in the test set 
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # Compute the output
    y_pred = classifier(batch_dict['x_surname'])
   
    # Compute the loss
    y_nationality = batch_dict['y_nationality']
    # Reduce the dimensionality of y_nationality 
    y_nationality = torch.argmax(y_nationality, dim=1)
    loss = loss_func(y_pred, y_nationality)
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)
    
    # Compute the accuracy
    acc_t = compute_accuracy(y_pred, y_nationality)
    running_acc += (acc_t - running_acc) / (batch_index + 1)
    
    print('batch : ' + str(batch_index) + ' loss: ' + str(running_loss) + ' running accuracy: ' + str(running_acc))




"""Prediction: """
def predict_nationality(surname, classifier, vectorizer):
    """Predict the nationality from a new surname.
    Args: 
        surname (str): the surname to classifier
        classifier (SurnameClassifer): an instance of the classifier
        vectorizer (SurnameVectorizer): the corresponding vectorizer
    Returns:
        dictionary with the most likely nationality and its probability
    """
    # vectorize the new surname 
    vectorized_surname =vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).view(1, -1)
    result = classifier(vectorized_surname, apply_softmax=True)
    
    # getting the max index of the predicted nationality
    probability_values, indices = result.max(dim=1)
    index = indices.item()
    
    # getting the right nationality according the index
    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
    probability_value = probability_values.item()
    
    return {'nationality': predicted_nationality , 
            'probability' : probability_value}


new_surname = input("Enter a surname to classify: ")
classifier = classifier .to('cpu')
prediction = predict_nationality(new_surname, classifier, vectorizer)
print("{} -> {} (p={:0.2f})".format(new_surname,
                                    prediction['nationality'],
                                    prediction['probability']))


"""Top k predictions :"""

def predict_topk_nationality(name, classifier, vectorizer, k=5):
    vectorized_name = vectorizer.vectorize(name)
    vectorized_name = torch.tensor(vectorized_name).view(1, -1)
    prediction_vector = classifier(vectorized_name, apply_softmax=True)
    # The addition for the top k prediction
    probability_values, indices = torch.topk(prediction_vector, k=k)
    # returned size is 1,k
    probability_values = probability_values.detach().numpy()[0]
    indices = indices.detach().numpy()[0]
    
    results = []
    for prob_value, index in zip(probability_values, indices):
        nationality = vectorizer.nationality_vocab.lookup_index(index)
        results.append({'nationality': nationality, 
                        'probability': prob_value})
    
    return results

new_surname = input("Enter a surname to classify: ")
classifier = classifier.to("cpu")

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