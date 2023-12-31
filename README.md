﻿# Surname-Classification
## Overview
This repository provides examples of surname classification using two different approaches: a Multilayer Perceptron (MLP) and a Convolutional Neural Network (CNN). Both models are trained to predict the nationality of a surname based on its characters.

## Project Structure

### Multilayer Perceptron (MLP)
1. Data Vectorization classes:
   - `Vocabulary`: Class to process text and extract vocabulary for mapping.
   - `SurnameVectorizer`: Vectorizes surnames and nationalities using the vocabularies.
2. Dataset class:
   - `SurnameDataset`: Loads the dataset, splits it into training, validation, and test sets, and generates batches.
3. Model:
   - `SurnameClassifier`: A 2-layer MLP for classifying surnames into nationalities.

### Convolutional Neural Network (CNN)
1. Data Vectorization classes (similar to the MLP):
   - `Vocabulary`: Class to process text and extract vocabulary for mapping.
   - `SurnameVectorizer`: Vectorizes surnames and nationalities using the vocabularies.
2. Dataset class (similar to the MLP):
   - `SurnameDataset`: Loads the dataset, splits it into training, validation, and test sets, and generates batches.
3. Model:
   - `SurnameCNNClassifier`: A CNN model for surname classification.
