 **Audio Classification Using Neural Networks** 

 Sai Divya Sivani Pragadaraju, Simrin Shah 



University of Colorado Boulder



 **1 Abstract** 

  Deep neural networks are quite prominent in modeling visual and textual data.
  In this project we wanted to see how audio data can be handled with the help of 
  various neural networks architectures and testify to its diversity and efficiency. The main goal of this project is to develop a model that would be able to classify a given audio sample into its respective class using few specialized deep neural network architectures such as convolutional neural networks(CNN), simple Recurrent neural networks(RNN) and Long Short Term memory(LSTM) neural networks. CNNs are a type of neural networks that are known for their efficiency in dealing with visual representations such as images. RNNs and LSTMs 
  are well known to deal with temporal data. In this project we have converted the 
  Audio sample into Spectrograms so that they can be fed to CNNs and RNNs. 
  We have also converted the audio to its corresponding text and then fed it to 
  LSTMs and have compared the accuracy of all these models to analyze which 
  ones have performed well. From these experiments it can be noted that CNN 
  with spectrograms yielded an accuracy of 88% , RNN with spectrograms has 
  yielded an accuracy of 69% and LSTM with textual data has yielded 12%. 025 026 026

 **2 Introduction**
 This project is an attempt to perform classification task on Audio inputs with the 
 help of neural network architectures like Convolutional Neural networks (CNN) 
 and Recurrent Neural Networks (RNN) i.e given an audio clip as an input, we 
 build various neural models that will be able to classify the category of the 
 respective audio sample. Lately, There has been more focus on handling images 
 and text data in the machine learning community that led to significant 
 developments in the sub-domains of Machine Learning, Computer Vision and 
 Natural Language Processing compared to that of Audio processing which is 
 still in the emerging phase. We believe that Audio processing, like Image and 
 Text processing play a vital role in the overall development of Artificial systems 
 as Sound(Audio) is one of the most crucial components of interpretation 
 of and interaction with things around. This sub-domain would imitate the ability 
 of listening in human beings. Audio classification is one of the basic tasks which 
 enables the machine to distinguish and interpret different types of sounds. 
 Audio classification finds its applications in the development of important recommendation algorithms and also in better human - robot interaction systems. 


Audio based applications will make things easier for illiterate and blind people 
or pretty much any human who can simply get their work done by speaking to 
the system. For instance, your system will be able to find a song you would like 
to hear when you simply hum a part of that song. Your system would be able 
to take your instructions, process it and respond to you .This diverse scope of
audio Processing motivated us to choose this task for this project. 050 051 
By referring to the papers titled “Deep Recurrent Neural Networks for Audio 
Classification in Construction Sites” and “CNN architectures for Large-scale 
Audio Classification” we could infer that audio signals could be converted to 
various Spectrograms and can be fed into CNN Architectures or RNN architectures 
in order to classify these audios. Though these approaches yielded better results, we would like to experiment if these audio clips can be converted to 
various other forms like for instance text or numbers and then feed them to the RNN / LSTM model to analyze how they perform. We have not come across 
any literature that converts audio to such textual or numerical forms and would 
like to experiment with it. We intend to execute all three approaches and draw 
a comparison in terms of their performance in classifying audios. 

For this project, We would like to perform a classification task of audio based inputs using the Audio MNIST dataset. As per our exploration, we understand
that audio clips are converted to spectrograms which are nothing but visualizaions of signal strength and are then fed into various neural architectures
for classification. As part of this project we would also like to experiment by
converting audio inputs to textual forms or numerical data instead of converting
them to spectrograms and then feeding them to the RNN architecture to analyze
how they perform. This attempt to project audio as textual representations and                    
feeding them into a neural network will enable us to explore a new method to
process audio signals and analyze its performance in comparison to spectrograms
fed to different neural network architectures. It will also help us determine the best conversion technique and architecture for this audio classification problem. 

 **3 Related Work**

Neural Network Architecture 
As per the existing work in the paper “CNN Architectures for Large-Scale Audio 
 Classification” , audio classification is performed using pretrained CNN models like AlexNet, Inception V3 and VGG models. Another paper “Deep Recurrent 
 Neural Networks for Audio Classifcation in Construction Sites” [2] makes use of Deep Recurrent Neural Networks to perform the same task. We implemented 
 this task using vanilla CNN, simple RNN and LSTM models by building each model from scratch.Spectrogram 
A spectrogram is a visual way of representing the signal strength of a signal over 
time at various frequencies present in a particular waveform. In order to execute the task of audio classification, these spectrograms are created for each audio 
sample to train the model. These images are then fed into the Neural Networks that processes these images and based on the features extracted, it classifies the 
audio. This technique is generally found in papers [1], [2] to perform audio classification.
We converted this audio to a textual representation and fed it into the neural network and analyzed its performance over the already established 
method.

**Dataset** 

The work done in [1], [2] have used datasets like YouTube-100M dataset and 
equipment operations in construction site dataset respectively.We used the Audio 
MNIST dataset for this experiment. This dataset has been chosen as this 
enabled us to experiment with textual representations of the audio because it 
contained samples of spoken digits.It also provided good insights into the difference between the performance of this task using different input representations 
and neural network architectures.This dataset has 3000 samples of spoken digits 
from 0-9 by 6 different speakers with 50 of each digit per speaker. 

 **4 Methods** 

For this project we split our dataset of audio samples into 80% of train data and
20% test data. We executed the task of audio classification with 3 different approaches: One way is to convert each of the audio samples into their respective 
spectrograms which are nothing but pictorial representations of audio signals 
with the help of mfccs method from Librosa Library and created a list of them along
with their corresponding labels and then fed them into our self-built convolutional neural network (CNN). This neural network has 5 layers and we have 
chosen this number of layers as it is generalizing well, further the number of 
layers is leading to high variance and overfitting. The first layer has 256 neurons 
 and a filter of size (4\*4). Since the spectrogram in the form of image is of 256\*256 
pixels so we have decided to use 256 neurons in this layer so that all pixels can 
be thoroughly processed. The second layer has 128 neurons followed by the third 
layer with 64 neurons.Each of these layers is followed by a max pooling layer 
and Batch normalization regularization of each layer. We used max pooling so 
that most prominent features can be retained with lower resolution and batch 
normalization helps with easy flow of information between hidden layers and 
makes training easy and quick. For all the layers except for the last but one 
we have used Adam activation function as it is very adaptable with respect to 
learning from sparse data of the images. Following which we flattened the next 
layer so that it can be served for classification of these extracted features from 
the feature map and finally in the output layer we have 10 neurons as we have 
10 classes of digits for classification. The activation used in this final layer is 
the softmax function as it outputs a probability distribution of which the one 
with maximum probability will be our chosen class.We have iterated this for 50 
epochs and with a batch size of 32.

 Also, we have also experimented with these spectrograms with RNN. This neural network also has 5 layers and has all the parameters and hyperparameters 
 similar to that of CNN above except that we have used dropout regularization 
 with a drop out rate of 0.2 instead of batch normalization as it does not consider 
 the recurrent part of the network while using batches.
 In the third approach we have converted each of the audio samples into 
 corresponding text using pyAudio library and speechRecognition API. This text is 
 then converted into numerical data and then fed into our self built LSTM neural 
 network. We have used the same set of parameters and hyper-parameters as for 
 the other neural networks in this project to maintain the uniformity of models 
 while comparing and analyzing the performance of all the models.

 **5 Experimental Design** 

Architecture comparison:



We conducted an analysis of comparing the different architectures and input
types to perform the task of audio classification. We performed the task of audio
classification on the Audio MNIST dataset consisting of 3000 audio samples of
digits from 0-9 by building models based on the CNN and RNN architectures.
The idea was to implement audio classification in 3 different forms:

 1).Converting audio clips to spectrograms and processing it using the CNN architecture 
2).Converting audio clips to spectrograms and processing it using RNN architecture 
 3).Converting audio clips to textual form and processing it using RNN-LSTM 
 architecture. After implementing these, we drew an analysis based on the performance of these 
 three methods and thus compared different architectures and input types. 165 166 166

 **Evaluation Metrics:** 

 The performance of each model was evaluated using the confusion matrix to 
 identify the number of audio samples correctly classified. Since this is a classification 
 task, the confusion matrix is the most remarkable approach for evaluating 
 a classification model. It helps in providing accurate insights into how correctly 
 the model has classified the classes depending on the data passed to the model 
 and how the classes are misclassified. It provides a very convenient way to display the performance of the model by pictorially showing the number of correctly 
 classified and misclassified classes. 
 We will also compare the accuracy of different models and represent the loss 
 curve graph for the same - Epochs vs Accuracy for train and test data. Accuracy is the most intuitive form of performance measure as it is simply a ratio 
 of the correctly predicted observations to the total observations. It works well 
 in case of a balanced dataset and since our dataset is a balanced one we chose 
 to determine the accuracy of the individual models.In order to compare it for 
 the train and test set, we chose to show the loss curves and visualize how the 
 accuracy improves over the epochs. This comparison between the train and test 
 set shows whether the model is overfitting or has been trained well in order to 
 predict unseen samples. Hence these evaluation metrics will help us determine 
 how to improve the model’s performance. 

**6 Experimental Results **

The experimental results for each of the models are: 

|Sr. No.|Model|Train Accuracy|Test Accuracy|
| - | - | :- | - |
|1|CNN + Spectrogram|98\.56%|88\.40%|
|2|RNN + Spectrogram|97\.16%|69\.0%|
|3|RNN + Textual Representation|14\.62%|14\.62%|
![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.001.png)



Given below is the Confusion Matrix for each model:



![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.002.png) 

 Fig.2: Model 2 

![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.003.png)  
Fig.3: Model 3  Given below is the Epoch vs Accuracy loss curve graph for each model: 

![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.004.png) 
Fig.4: Model 1 
![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.005.png) 

Fig.5: Model 2 
![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.006.png) 
Fig.6: Model 3

The general trends observed are: 
From the loss curves for the first model, it is observed that the train accuracy 
increases initially as the number of epochs increases and then stabilizes for the 
remaining epochs. In the case of the test model, the test accuracy also increases 
sharply in the first few epochs and then fluctuates a little till it stabilizes to a test 
accuracy of nearly 88%. The train accuracy definitely improves and increases as 
compared to the previous models but the test accuracy is much less compared 
to the train accuracy. This actually implies overfitting the model which defeats 
the purpose of using Batch Normalization which is actually designed to reduce 
overfitting. This could happen because batch normalization is generally suitable 
for very large datasets and our dataset might not be large enough to use this 
technique. It does improve the training accuracy as it normalizes all the data 
and maintains a better test accuracy than other models. From the confusion 
matrix it is observed that compared to the misclassified classes, the number of 328 classes correctly classified are more. This could be possible because CNN works 
very well with images as they reduce the number of parameters without affecting 
the quality of the images and this model could extract essential features more 
efficiently as compared to any other neural network architecture. 
 The loss curve of the second model implies that just like the first model, the 334 train accuracy increases significantly in the first few epochs and then stabilize,
to a 97% accuracy with a few fluctuations. The test accuracy increases a little 
in the first few epochs and then stagnates at an accuracy of 69%. This indicates
a huge difference in the train and test accuracy suggesting that the model is
overfitting. This could be because the RNN model has less feature compatibility
as compared to CNN and it has the ability to take arbitrary input/output length                      340 340
which affects the computational efficiency. Hence, RNN is not able to extract 
features efficiently because of which it has learnt a lot of noise and is unable to
identify test audios effectively. The confusion matrix for this model also reflects that the number of misclassified classes are nearly at par with the number of 
correctly classified classes. 
The third model was an experiment to convert the audio files to text and feed
 this data into the RNN. The loss curves show that its train accuracy was very low overall but it increased over the epochs whereas the test accuracy increased 
a little initially and then remained almost constant at around 12%. The reason 
for the low performance of this model could be that the audio was not converted 
to text efficiently. A lot of audios were identifiable by the library and they were 
termed as ‘Not Recognizable’. This could have been due to different accents of 
the speakers. Also a lot of identifiable audio files were being converted to different text,
for e.g. zero was being identified as ‘little’, nine as ‘time’ and so on. 
So that led to inefficient training of the model. The confusion matrix also shows 
majority of the misclassified samples thus implying that this method was not at 
all efficient. 

**Conclusions** 

From this project, it can be clearly noted that audio based data can be effectively 
classified by converting them to spectrograms and then feeding them convolutional neural networks as it yielded a good test accuracy of around 88.5% as 
indicated in the results section. This classification can also be performed with 375 RNN the similar way but is not as effective as CNN as it is just yielding a 
moderate test accuracy of 69%. Although, converting the audio samples to textual
data and feeding them to LSTMs did not give satisfactory results, it can be 
improved by making use of even better audio to text converters wherein most information is recognized and retained.





 **References**

392  [1] Shawn H, Sourish Ch, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, R. 392
392  Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, 393
392  Malcolm Slaney, Ron J. Weiss, Kevin Wilson. “CNN ARCHITECTURES FOR 394
392  LARGE-SCALE AUDIO CLASSIFICATION” in arXiv:1609.09430v2 [cs.SD] 395
392  2017 396
392  [2] Michele Scarpiniti, Danilo Comminiello, Aurelio Uncini, Yong-Cheol Lee, 397
392  “Deep Recurrent Neural Networks for Audio Classification in Construction Sites“ 398
392  in 2020 28th European Signal Processing Conference (EUSIPCO), IEEE, 2020 399
392  [3] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with 400
392  deep convolutional neural networks,” in Advances in neural information process- 401
392  ing systems, 2012, pp. 1097–1105. 402
392  [4] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large- 403
392  scale image recognition,” arXiv preprint arXiv:1409.1556, 2014. 404
405  [5] J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. 405
405  Moore, M. Plakal, and M. Ritter, “Audio Set: An ontology and human-labeled 406
405  dartaset for audio events,” in IEEE ICASSP 2017, New Orleans, 2017. 407 408 408

