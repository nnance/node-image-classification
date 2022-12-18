# TensorFlow.js Example: Training MNIST with Node.js

This example shows how to build a TensorFlow.js model to recognize handwritten digits with a convolutional neural network. First, we'll train the classifier by having it "look" at thousands of handwritten digit images and their labels. The classifier's accuracy can be evaluated using test data that the model has never seen.

This is considered a classification task as we are training the model to assign a category (the digit that appears in the image) to the input image. We will train the model by showing it many examples of inputs along with the correct output. This is referred to as supervised learning.

This example shows you how to train MNIST (using the layers API) under Node.js.

This model will compute accuracy after one pass through the training dataset
(60,000 samples) and evaluate 50 images from the test data set for accuracy after each epoch.

Prepare the node environment:
```sh
$ npm install
$ npm run build
```

Run the training script:
```sh
$ node ./dist/index.js
```
