000 Audio Classification Using Neural Networks 000 001 001

002 002 003 Sai Divya Sivani Pragadaraju, Simrin Shah 003

004 004

University of Colorado Boulder

005 005 006 006 007 007

008 1 Abstract 008 009 009

10  Deep neural networks are quite prominent in modeling visual and textual data. 010
10  In this project we wanted to see how audio data can be handled with the help of 011
10  various neural networks architectures and testify to its diversity and efficiency. 012
10  The main goal of this project is to develop a model that would be able to classify 013
10  a given audio sample into its respective class using few specialized deep neural 014
10  network architectures such as convolutional neural networks(CNN), simple Re- 015
10  current neural networks(RNN) and Long Short Term memory(LSTM) neural 016
10  networks. CNNs are a type of neural networks that are known for their effi- 017
10  ciency in dealing with visual representations such as images. RNNs and LSTMs 018
10  are well known to deal with temporal data. In this project we have converted the 019
10  Audio sample into Spectrograms so that they can be fed to CNNs and RNNs. 020
10  We have also converted the audio to its corresponding text and then fed it to 021
10  LSTMs and have compared the accuracy of all these models to analyze which 022
10  ones have performed well. From these experiments it can be noted that CNN 023
10  with spectrograms yielded an accuracy of 88% , RNN with spectrograms has 024
10  yielded an accuracy of 69% and LSTM with textual data has yielded 12%. 025 026 026

027 2 Introduction 027 028 028

29  This project is an attempt to perform classification task on Audio inputs with the 029
29  help of neural network architectures like Convolutional Neural networks (CNN) 030
29  and Recurrent Neural Networks (RNN) i.e given an audio clip as an input, we 031
29  build various neural models that will be able to classify the category of the 032
29  respective audio sample. Lately, There has been more focus on handling im- 033
29  ages and text data in the machine learning community that led to significant 034
29  developments in the sub-domains of Machine Learning, Computer Vision and 035
29  Natural Language Processing compared to that of Audio processing which is 036
29  still in the emerging phase. We believe that Audio processing, like Image and 037
29  Text processing play a vital role in the overall development of Artificial sys- 038
29  tems as Sound(Audio) is one of the most crucial components of interpretation 039
29  of and interaction with things around. This sub-domain would imitate the abil- 040
29  ity of listening in human beings. Audio classification is one of the basic tasks 041
29  which enables the machine to distinguish and interpret different types of sounds. 042
29  Audio classification finds its applications in the development of important rec- 043
29  ommendation algorithms and also in better human - robot interaction systems. 044

ECCV-16 submission ID 1 11

45  Audio based applications will make things easier for illiterate and blind people 045
45  or pretty much any human who can simply get their work done by speaking to 046
45  the system. For instance, your system will be able to find a song you would like 047
45  to hear when you simply hum a part of that song. Your system would be able 048
45  to take your instructions, process it and respond to you .This diverse scope of 049
45  audio Processing motivated us to choose this task for this project. 050 051 051
52  By referring to the papers titled “Deep Recurrent Neural Networks for Au- 052
52  dio Classification in Construction Sites” and “CNN architectures for Large-scale 053
52  Audio Classification” we could infer that audio signals could be converted to 054
52  various Spectrograms and can be fed into CNN Architectures or RNN architec- 055
52  tures in order to classify these audios. Though these approaches yielded better 056
52  results, we would like to experiment if these audio clips can be converted to 057
52  various other forms like for instance text or numbers and then feed them to the 058
52  RNN / LSTM model to analyze how they perform. We have not come across 059
52  any literature that converts audio to such textual or numerical forms and would 060
52  like to experiment with it. We intend to execute all three approaches and draw 061
52  a comparison in terms of their performance in classifying audios. 062 063 063

064 For this project, We would like to perform a classification task of audio based 064 inputs using the Audio MNIST dataset. As per our exploration, we understand

065 065

that audio clips are converted to spectrograms which are nothing but visual-

066 066

izations of signal strength and are then fed into various neural architectures

067 067 for classification. As part of this project we would also like to experiment by

068 068

converting audio inputs to textual forms or numerical data instead of converting

069 069

them to spectrograms and then feeding them to the RNN architecture to analyze

070 070

how they perform. This attempt to project audio as textual representations and                    071 071

feeding them into a neural network will enable us to explore a new method to

072 072

process audio signals and analyze its performance in comparison to spectrograms

73  fed to different neural network architectures. It will also help us determine the 073
73  best conversion technique and architecture for this audio classification problem. 074 075 075 076 076

077 3 Related Work 077 078 078

79  Neural Network Architecture 079
79  As per the existing work in the paper “CNN Architectures for Large-Scale Audio 080
79  Classification” [1], audio classification is performed using pretrained CNN mod- 081
79  els like AlexNet, Inception V3 and VGG models. Another paper “Deep Recurrent 082
79  Neural Networks for Audio Classifcation in Construction Sites” [2] makes use 083
79  of Deep Recurrent Neural Networks to perform the same task. We implemented 084
79  this task using vanilla CNN, simple RNN and LSTM models by building each 085
79  model from scratch. 086 087 087
88  Spectrogram 088
88  A spectrogram is a visual way of representing the signal strength of a signal over 089
90  time at various frequencies present in a particular waveform. In order to execute 090
90  the task of audio classification, these spectrograms are created for each audio 091
90  sample to train the model. These images are then fed into the Neural Networks 092
90  that processes these images and based on the features extracted, it classifies the 093
90  audio. This technique is generally found in papers [1], [2] to perform audio clas- 094
90  sification. We converted this audio to a textual representation and fed it into 095
90  the neural network and analyzed its performance over the already established 096
90  method. 097 098 098
99  Dataset 099
99  The work done in [1], [2] have used datasets like YouTube-100M dataset and 100
99  equipment operations in construction site dataset respectively.We used the Au- 101
99  dio MNIST dataset for this experiment. This dataset has been chosen as this 102
99  enabled us to experiment with textual representations of the audio because it 103
99  contained samples of spoken digits.It also provided good insights into the differ- 104
99  ence between the performance of this task using different input representations 105
99  and neural network architectures.This dataset has 3000 samples of spoken digits 106
99  from 0-9 by 6 different speakers with 50 of each digit per speaker. 107 108 108

109 4 Methods 109 110 110 111 111

For this project we split our dataset of audio samples into 80% of train data and

112 112

20% test data. We executed the task of audio classification with 3 different ap-

113 113

proaches: One way is to convert each of the audio samples into their respective

114  spectrograms which are nothing but pictorial representations of audio signals 114
114  with the help of mfccs method from Librosa Library and created a list of them 115
114  along with their corresponding labels and then fed them into our self-built con- 116
114  volutional neural network (CNN). This neural network has 5 layers and we have 117
114  chosen this number of layers as it is generalizing well, further the number of 118
114  layers is leading to high variance and overfitting. The first layer has 256 neurons 119
114  and a filter of size (4\*4). Since the spectrogram in the form of image is of 256\*256 120
114  pixels so we have decided to use 256 neurons in this layer so that all pixels can 121
114  be thoroughly processed. The second layer has 128 neurons followed by the third 122
114  layer with 64 neurons.Each of these layers is followed by a max pooling layer 123
114  and Batch normalization regularization of each layer. We used max pooling so 124
114  that most prominent features can be retained with lower resolution and batch 125
114  normalization helps with easy flow of information between hidden layers and 126
114  makes training easy and quick. For all the layers except for the last but one 127
114  we have used Adam activation function as it is very adaptable with respect to 128
114  learning from sparse data of the images. Following which we flattened the next 129
114  layer so that it can be served for classification of these extracted features from 130
114  the feature map and finally in the output layer we have 10 neurons as we have 131
114  10 classes of digits for classification. The activation used in this final layer is 132
114  the softmax function as it outputs a probability distribution of which the one 133
114  with maximum probability will be our chosen class.We have iterated this for 50 134
135  epochs and with a batch size of 32. 135 136 136
137  Also, we have also experimented with these spectrograms with RNN. This neu- 137
137  ral network also has 5 layers and has all the parameters and hyperparameters 138
137  similar to that of CNN above except that we have used dropout regularization 139
137  with a drop out rate of 0.2 instead of batch normalization as it does not consider 140
137  the recurrent part of the network while using batches. 141 142 142
143  In the third approach we have converted each of the audio samples into cor- 143
143  responding text using pyAudio library and speechRecognition API. This text is 144
143  then converted into numerical data and then fed into our self built LSTM neural 145
143  network. We have used the same set of parameters and hyper-parameters as for 146
143  the other neural networks in this project to maintain the uniformity of models 147
143  while comparing and analyzing the performance of all the models. 148 149 149

150 5 Experimental Design 150 151 151 152 152

Architecture comparison:

153 153

We conducted an analysis of comparing the different architectures and input

154 154

types to perform the task of audio classification. We performed the task of audio

155 155 classification on the Audio MNIST dataset consisting of 3000 audio samples of

156 156

digits from 0-9 by building models based on the CNN and RNN architectures.

157 157

The idea was to implement audio classification in 3 different forms:

158  1).Converting audio clips to spectrograms and processing it using the CNN ar- 158
158  chitecture 159
158  2).Converting audio clips to spectrograms and processing it using RNN archi- 160
158  tecture 161
158  3).Converting audio clips to textual form and processing it using RNN-LSTM 162
158  architecture. 163
158  After implementing these, we drew an analysis based on the performance of these 164
158  three methods and thus compared different architectures and input types. 165 166 166
167  Evaluation Metrics: 167
167  The performance of each model was evaluated using the confusion matrix to 168
167  identify the number of audio samples correctly classified. Since this is a classifi- 169
167  cation task, the confusion matrix is the most remarkable approach for evaluating 170
167  a classification model. It helps in providing accurate insights into how correctly 171
167  the model has classified the classes depending on the data passed to the model 172
167  and how the classes are misclassified. It provides a very convenient way to dis- 173
167  play the performance of the model by pictorially showing the number of correctly 174
167  classified and misclassified classes. 175
167  We will also compare the accuracy of different models and represent the loss 176
167  curve graph for the same - Epochs vs Accuracy for train and test data. Accu- 177
167  racy is the most intuitive form of performance measure as it is simply a ratio 178
167  of the correctly predicted observations to the total observations. It works well 179
180  in case of a balanced dataset and since our dataset is a balanced one we chose 180
180  to determine the accuracy of the individual models.In order to compare it for 181
180  the train and test set, we chose to show the loss curves and visualize how the 182
180  accuracy improves over the epochs. This comparison between the train and test 183
180  set shows whether the model is overfitting or has been trained well in order to 184
180  predict unseen samples. Hence these evaluation metrics will help us determine 185
180  how to improve the model’s performance. 186 187 187

188 188 189 189

190 6 Experimental Results 190 191 191 192 192 193 193

194 The experimental results for each of the models are: 194 195 195

|Sr. No.|Model|Train Accuracy|Test Accuracy|
| - | - | :- | - |
|1|CNN + Spectrogram|98\.56%|88\.40%|
|2|RNN + Spectrogram|97\.16%|69\.0%|
|3|RNN + Textual Representation|14\.62%|14\.62%|

196 196 197 197 198

198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.001.png)

199

200

201

202

203

204

Given below is the Confusion Matrix for each model:

205

206 207 208 209 210 211 212 213

214 214 215 215 216 216 217 217 218 218 219 219 220 220 221 221 222 222 223 223

224 Fig.1: Model 1 224

225 225 226![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.002.png) 226 227 227 228 228 229 229 230 230 231 231 232 232 233 233 234 234 235 235 236 236 237 237 238 238

239 Fig.2: Model 2 239 240 240

241 241 242 242 243 243 244 244 245 245 246 246 247 247 248![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.003.png) 248 249 249 250 250 251 251 252 252 253 253 254 254 255 255 256 256 257 257 258 258 259 259 260 260 261 Fig.3: Model 3 261 262 262 263 263 264 264 265 265 266 266 267 Given below is the Epoch vs Accuracy loss curve graph for each model: 267 268 268 269 269

270 270 271![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.004.png) 271 272 272 273 273 274 274 275 275 276 276 277 277 278 278 279 279 280 280 281 Fig.4: Model 1 281 282 282 283 283 284 284 285 285 286 286 287![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.005.png) 287 288 288 289 289 290 290 291 291 292 292 293 293 294 294 295 295 296 296 297 297

Fig.5: Model 2

298 298 299 299 300 300 301 301 302 302 303![](Aspose.Words.fd15adee-f0ee-43d5-af62-e75947c9c5b3.006.png) 303 304 304 305 305 306 306 307 307 308 308 309 309 310 310 311 311 312 312 313 313

314 Fig.6: Model 3 314

315  The general trends observed are: 315
315  From the loss curves for the first model, it is observed that the train accuracy 316
315  increases initially as the number of epochs increases and then stabilizes for the 317
315  remaining epochs. In the case of the test model, the test accuracy also increases 318
315  sharply in the first few epochs and then fluctuates a little till it stabilizes to a test 319
315  accuracy of nearly 88%. The train accuracy definitely improves and increases as 320
315  compared to the previous models but the test accuracy is much less compared 321
315  to the train accuracy. This actually implies overfitting the model which defeats 322
315  the purpose of using Batch Normalization which is actually designed to reduce 323
315  overfitting. This could happen because batch normalization is generally suitable 324
315  for very large datasets and our dataset might not be large enough to use this 325
315  technique. It does improve the training accuracy as it normalizes all the data 326
315  and maintains a better test accuracy than other models. From the confusion 327
315  matrix it is observed that compared to the misclassified classes, the number of 328 classes correctly classified are more. This could be possible because CNN works
315  very well with images as they reduce the number of parameters without affecting 329330
315  the quality of the images and this model could extract essential features more 331
315  efficiently as compared to any other neural network architecture. 332 332

333 333

334 The loss curve of the second model implies that just like the first model, the 334 train accuracy increases significantly in the first few epochs and then stabilizes

335 335

to a 97% accuracy with a few fluctuations. The test accuracy increases a little

336 336

in the first few epochs and then stagnates at an accuracy of 69%. This indicates

337 337

a huge difference in the train and test accuracy suggesting that the model is

338 338

overfitting. This could be because the RNN model has less feature compatibility

339 339

as compared to CNN and it has the ability to take arbitrary input/output length                      340 340

which affects the computational efficiency. Hence, RNN is not able to extract

341 341 features efficiently because of which it has learnt a lot of noise and is unable to

342 342 identify test audios effectively. The confusion matrix for this model also reflects

343  that the number of misclassified classes are nearly at par with the number of 343
343  correctly classified classes. 344 345 345
346  The third model was an experiment to convert the audio files to text and feed 346
346  this data into the RNN. The loss curves show that its train accuracy was very 347
346  low overall but it increased over the epochs whereas the test accuracy increased 348
346  a little initially and then remained almost constant at around 12%. The reason 349
346  for the low performance of this model could be that the audio was not converted 350
346  to text efficiently. A lot of audios were identifiable by the library and they were 351
346  termed as ‘Not Recognizable’. This could have been due to different accents of 352
346  the speakers. Also a lot of identifiable audio files were being converted to differ- 353
346  ent text, for e.g. zero was being identified as ‘little’, nine as ‘time’ and so on. 354
346  So that led to inefficient training of the model. The confusion matrix also shows 355
346  majority of the misclassified samples thus implying that this method was not at 356
346  all efficient. 357 358 358 359 359

ECCV-16 submission ID 1 15

360  Most of our questions were answered by this experiment but we believe that 360
360  the next step of research should be trying to identify the emotions from the 361
360  audio files and classify that emotion or converting audio files to text and under- 362
360  standing the emotion from that. A combination of CNN and RNN can be used 363
360  to achieve this task but more research is needed in this area. One more direction 364
360  of research could be trying to classify the audios by processing them in their raw 365
360  form without converting them into any other form like image or text and feeding 366
360  them into the specialized neural network architectures like CNN and RNN. 367 368 368

369 7 Conclusions 369 370 370 371 371

372  From this project, it can be clearly noted that audio based data can be effectively 372
372  classified by converting them to spectrograms and then feeding them convolu- 373
372  tional neural networks as it yielded a good test accuracy of around 88.5% as 374
372  indicated in the results section. This classification can also be performed with 375 RNN the similar way but is not as effective as CNN as it is just yielding a mod-

376 376

erate test accuracy of 69%. Although, converting the audio samples to textual

377 377

data and feeding them to LSTMs did not give satisfactory results, it can be

378 378

improved by making use of even better audio to text converters wherein most

379 379

information is recognized and retained.

380 380 381 381

Further improvements on this project can help in significantly developing human-

382 382

machine interaction as this classification task will help the system to understand

383 383

and bifurcate the audio information communicated to it in the form of audio sig-

384  nals. Most of the work can simply be done by talking to the machine like booking 384
384  tickets or filling a form online. Applications developed on this principle can be of 385
384  great aid to the blind and to the illiterate as most of work gets done by simply 386
384  speaking. 387 388 388 389 389

390 References 390 391 391

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

409 409 410 410 411 411 412 412 413 413 414 414 415 415 416 416 417 417 418 418 419 419 420 420 421 421 422 422 423 423 424 424 425 425 426 426 427 427 428 428 429 429 430 430 431 431 432 432 433 433 434 434 435 435 436 436 437 437 438 438 439 439 440 440 441 441 442 442 443 443 444 444 445 445 446 446 447 447 448 448 449 449
