Code Neuron
===========

Recurrent neural network to detect code blocks. Runs on Tensorflow. It is trained in two stages.

First stage is pre-training the character level RNN with two branches - before and after:

![CharRNN Architecture](doc/char_rnn_arch.png)

```
my code :  FooBar
------> x <------
```

I set 512 LSTM neurons and reach 89% validation accuracy over 200 most frequent character classes:

![CharRNN Validation](doc/char_rnn_validation.png)

The second stage is training the same network but with the different dense layer which predicts
only 3 classes: code block begins, code block ends and no-op.

![Code Neuron Validation](doc/code_neuron_validation.png)

It is fast to train and reaches ~98% accuracy.

Training set
------------

[StackSample questions and answers](https://www.kaggle.com/stackoverflow/stacksample), processed with

```
unzip -p Answers(Questions).csv.zip | ./dataset | sed  '/^$/d' | gzip >> Dataset.txt.gz
```

Baked model
-----------

[model_LSTM_512_0.9790.pb](model_LSTM_512_0.9790.pb) - reaches 97.9% accuracy on validation
(and 99.9% on train). Tensorflow graph format.

Pretraining was performed with 20% validation on first 8000000 bytes of the uncompressed questions.
Training was performed with 20% validation on first 128000000 bytes of the uncompressed questions.
This means I was lazy to wait a week for it to train on the whole dataset - you are encouraged
to experiment.

License
-------

MIT, see [LICENSE](LICENSE).