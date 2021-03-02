# Caption-Generation
Neural Network Arhitecutre for automatically generate caption from images

## Content

This repository will have 3 main python scripts:

1. The CNN-LSTM arhitecture
2. Training
3. Visualisation

## CNN-LSTM arhitecture

There is an encoder CNN(ResNet50), which has the final fully-conected layer removed, with a linear layer at the end. The decoder is an LSTM. In my case the Xavier initialization was used fot the weights initialization. The hidden_size and embed_size are set to 512 however I have tried with 256 and 128 and my results were not better.

## Training

For training the actual model the most important part is to finaly tune the training parameters

- batch_size - the batch size of each training batch
- vocab_threshold - the minimum word count threshold
- vocab_from_file - A Boolean that decideds wheter to load the vocabulary from file
- embed_size - the dimensionality of the image and word embeddings
- hidden_size - the number of features in the hidden state
- num_epochs - the number of epochs to train the model
- save_every - determines how often to save the model weights

## Visualization and results

| Description                                                                                    | Image  |
|  --------------------------------------------------------------------------------------------- | ------ |
| This is an image of a Hydrant which the Model predicted as expected | ![alt Hydrant](https://github.com/claudiup423/Caption-Generation/blob/main/result_images/Hydrant.png) |
| This is an image of a Fruit basket which the Model predicted as expected | ![alt Fruit](https://github.com/claudiup423/Caption-Generation/blob/main/result_images/Fruits.png) |
| This is one of the failed prediction the model predicts it as a train traveling however this is a construction sight| ![alt Faild Prediction](https://github.com/claudiup423/Caption-Generation/blob/main/result_images/Track.png) |