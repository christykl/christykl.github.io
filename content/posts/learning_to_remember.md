+++
title = 'Learning to Remember (and Forget)'
date = 2024-07-31T01:05:21-04:00
draft = false
tags = ["learningdeeplearning"]
+++

## The Evolution of Memory in Neural Networks
### Recurrent Neural Networks

A major limitation of vanilla neural networks is that they are unable to process variable-length structures: they map fixed-length inputs to fixed-length outputs by learning features over a fixed number of layers. Further, traditional neural networks have no "memory"—when presented with sequential data such as this sentence, there is no easy way to use context learned from previous inputs to inform future outputs. 

{{< figure src="/images/RNN_unroll.png" alt="unrolled RNN" caption="Unrolling a sequence-to-sequence RNN ([*Deep Learning*. Goodfellow *et al.*, 2016](https://www.deeplearningbook.org/))." >}}

Recurrent neural networks (RNNs) are a class of neural networks specialized for sequential data. They can learn dependencies from previous inputs in the sequence though a self-loop connection which maintains a hidden state. As a result, RNNs also have the advantage of sharing the same parameters over time steps, much like a 1D temporal convolution. At each time step $t$, the network updates the hidden state $h_t$ using the learned parameters $\theta$, the previous hidden state $h_{t-1}$, and an input $x_t$ if applicable. The hidden step at each time step can be written $h_t = f(h_{t-1},\,x_t;\, \theta_h)$, where $f$ is the transformation of the hidden layer. The hidden state effectively encodes a limited summary of relevant features from all previous input sequence vectors. The output can be written $o_t = g(h_{t};\, \theta_t)$, where $g$ is the transformation of the output layer.

{{< figure src="/images/RNN_models.png" alt="RNN architectures" caption="Various basic RNN architectures ([Karpathy, 2015](https://karpathy.github.io/2015/05/21/rnn-effectiveness/))." >}}

A helpful way to visualize RNNs is to unroll them into a chain of repeated feedforward networks that share the same parameters at each timestep. There are several basic RNN structures including one-to-one, one-to-many, many-to-one, and many-to-many. Each is used for different applications. For example, one-to-one would be appropriate for vanilla neural network tasks such as image classification. Unsynced many-to-many could be used for translation tasks, taking in a sentence in one language and producing a corresponding sentence in the target language, while synced many-to-many could be used for frame-by-frame video classification ([Karpathy, 2015](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)). 

Using the unrolled view of the RNN, we can use backpropagation to learn the model parameters just as we do with feedforward networks. Since this backpropagation is performed over the states at different time steps, the algorithm is called backpropagation through time (BPTT). 

A well-known problem with RNNs is that they struggle to capture long-term dependencies, or remembering information over longer sequences. Consider the following sentence:

> "The dog, which was a golden retriever with a friendly disposition and a penchant for chasing squirrels in the park near the old oak tree where children often played on sunny afternoons, wagged its tail."

There is a long-term dependency between "The author" at the beginning of the sentence and "revealed" near the end. A model trying to predict that "revealed" comes after "finally" in the sentence must keep "author" as the subject of the sentence despite all of the added context. Since there is a large distance between the two relevant phrases, the gradients calculated by BPTT will tend to vanish (or, less commonly, explode) as a result of repeated matrix multiplication, and the model will not be able to learn their connection. 
### Long Short-Term Memory

To address the vanishing/exploding gradients problem, Long Short-Term Memory (LSTMs) were proposed. LSTMs are a type of gated RNN that are able to learn long-term dependencies. Individual LSTM cells replace the hidden state of the RNN with a more complicated recurrent network that includes an additional linear self-loop, allowing gradients to flow over long periods, subject to gating units. These gating units are dependent on the previous cell's state as well as the input sequence, allowing the LSTM to dynamically forget or add new information depending on context.

{{< figure src="/images/LSTM.png" alt="LSTM" caption="The inner workings of a LSTM cell ([Olah, 2015](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))." >}}

The first gate is the forget gate, which weights the input vector and previous cell state and applies a sigmoid layer to squash the values between 0 and 1. The resulting vector is multiplied pointwise to the current cell state. This gate decides how much of the old information should be thrown out, with 0 representing deleting everything and 1 representing keeping everything. The next gate is the input gate, which decides how much and which new information is let in. A tanh and sigmoid layers are separately applied to the input to normalize the vector and determine each new feature input's importance, respectively. The two resulting vectors are multiplied together pointwise before being added to the cell state. Finally, the cell state is put through a tanh layer before being multiplied with the output gate sigmoid, controlling what information is returned and passed to the next cell state.

### Attention

### It's All You Need??