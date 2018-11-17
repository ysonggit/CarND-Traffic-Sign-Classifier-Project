# **Traffic Sign Recognition**

## Writeup

---

**Project Outlines**

The goals / steps of this project are the following:
1. Load the data set (see below for links to the project dataset)
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./images/dataset_distribution.png "Dataset Visualization"
[image2]: ./images/lenet_cnn.jpg "LeNet Architecture"
[image3]: ./images/extra_10.png "10 Extra German Traffic Signs"
[image4]: ./images/extra_10_2.png "Top 5 Guess of 10 Extra Images"
[image5]: ./images/rgb.png "Traffic Sign RGB"
[image6]: ./images/gray.png "Traffic Sign Grayscale"
[image7]: ./images/gray_5.png "Traffic Sign Grayscale 2"
[image8]: ./images/gray_norm_5.png "Traffic Sign Grayscale 2 Normalized"
[image9]: ./images/modifiedLeNet.jpeg "Modified LeNet Architecture"
[image10]: ./images/43_signs.png "All Signs"
[image11]: ./images/softmax_prob.png "Softmax Probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Data Set Summary & Exploration

#### 1. Summary of the data set

The 1st code block of my Ipython notebook loads the image sets from the `../data/` directory of the workspace. The 3rd code block prints the basic information of the datasets with numpy:

* The size of training set is 34769.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The figure below is plotted by the codes in the 4th block. It shows the bar charts representing the data distributions in the train set (blue color), the test set (red), and the validation set (green), respectively.

The conclusion is that the data distributions of the train, test, and validation sets are mostly consistent with each other. Therefore, our learning algorithm runs under a good premise of datasets.

![Dataset distributions][image1]

Furthermore, I would like to know what the distinct traffic signs look like in the train set. So I define codes in the 6th code block to display all 43 distinct signs:

![All 43 Signs][image10]

### Design and Test a Model Architecture

#### 1.Preprocessed the image data.

I used two techniques to preprocess the input images.
1. Convert RGB images to grayscale.
2. Normalized the grayscale images.

Recall in the LeNet Lab, the LeNet classifier takes grayscale MNIST data, which is a set of `32x32x1` images as one of its inputs. Hence, in order to re-use the LeNet classifier in this project with the **minimum** architecture changes, I want to keep the input's format as the same as the MNIST dataset and decide to first convert the RGB images (`32x32x3`) to grayscale ones.

When converting the image, I referred to the codes of Jeremy Shannon [1] because I got the runtime exceptions while invoking the openCV `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` functions. It turns out Jeremy's method is more straightforward and efficient using Numpy's vector operation against the image array.

Here is an example of a traffic sign image before and after grayscaling.

![rgb image][image5] ![grayscale image][image6]

Because the image data should be normalized so that the data has mean zero and equal variance. I follow the suggested way, namely, `(pixel - 128)/ 128`, to approximately normalize the data and can be used in this project.

The codes are in the 8th code block of the notebook. The pictures below show the grayscale image on the left and normalized grayscale image on the right.

![grayscale image][image7] ![normalized grayscale image][image8]

#### 2. Final model architecture

My idea of solving this project is simpler: just re-use the LeNet architecture in the Lab solution as much as possible. First, because I believe that the architecture introduced by Yann LeCun ([LeNet paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)) should work well with the traffic sign image datasets. The intuition behind my belief is that the traffic sign dataset can be treated as just a special type of "MNIST" dataset. Secondly, I am curious to know how would the original LeNet architecture perform with the traffic sign dataset.

In my 12th code block of the notebook, I keep the most of the codes previously tested in the LeNet Lab. The only change I made is to change the dimension of the output layer from 10 to 43, which is the actual number of classes of signs, as the figure shows below:

![LeNet Architecture][image2]

My final model consisted of the following layers:

| Layer                 |     Description                                           |
|:-----------------:|:-------------------------------------------:|
| Input                 | 32x32x1 Gray image                                         |
| Convolution          | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                        |                                                                      |
| Max Pooling            | 2x2 stride, outputs 14x14x6                            |
| Convolution         | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU                   |                                                             |
| Max Pooling             | 2x2 stride, outputs 5x5x16                                     |
|    Flatten                     |    outputs 400                                                        |
| Fully Connected        |    outputs    120                                                      |
| RELU                   |                                                             |
| Fully Connected        |    outputs    84                                                       |
| RELU                   |                                                             |
| Fully Connected        |    outputs    43                                                       |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model,  I used mostly the same components and parameters used in the LeNet Lab, such as the Adam optimizer, valid padding, max pooling, etc. The final settings used were:

- epochs: 60
- batch size: 100
- learning rate: 0.001
- mu: 0
- sigma: 0.1
- dropout: N/A

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.989
* test set accuracy of 0.931

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  - LeNet. It is always to start with an architecture that I am familiar with.
* What were some problems with the initial architecture?
  - I observe the performance of this CNN architecture fluctuates in a wide range with different hyperparameters. For example, with `EPOCH=60, BATCH_SIZE=100, rate=0.001`, the test accuracy is over 90%. However, with `EPOCH=50, BATCH_SIZE=128, rate=0.001`, the test accuracy is below 80%.
  - Even I use exactly the same set of parameters, running the training process could finally output slightly different accuracy. For example, with `EPOCH=60, BATCH_SIZE=100, rate=0.001`, the highest validation accuracy I ever achieved is 0.992, and the lowest accuracy I got is around 0.960.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  - Not much. I only adjusted the size of the last fully connected layer by changing it from 10 to 43.
* Which parameters were tuned? How were they adjusted and why?
  - I adjusted parameters of `EPOCH`, `BATCH_SIZE`, and `rate` (learning rate).
  - The way I adjusted these values is to try the different combination of EPOCH, BATCH_SIZE, and learning rate values iteratively.
    - The EPOCH values are: `{30, 40, 50, 55, 60, 80}`.
    - The BATCH_SIZE values are: `{90, 100, 128, 200}`.
    - The learning rate values are: `{0.0005, 0.0009, 0.001, 0.0011, 0.0015, 0.002}`.
    - The dropout (keep probability) values are: `{1.0, 0.5}`.
    - For the other parameters I use the same values as the ones in the LeNet Lab solution: `mu=0, sigma=0.1`.

    For each combination, I had to rerun the training process to see the results and compare the accuracy.

    I finally choose the combination of `EPOCH=60, BATCH_SIZE=100, rate=0.001` among all the possible combinations because their results are the best! You may raise a question that there are hundreds of combinations and trying one by one is very time-consuming. Indeed, this process costs a lot of times and efforts. I would like to share three observations learned from this process:
    1. A useful trick I find to accelerate this process is: **look at the first validation accuracy closely!** I find when a first validation accuracy is a low number, such as 0.107, 0.431, etc., the final validation accuracy will always be very poor. Whenever I see such a scenario, that is, the first one or two accuracy values are under 0.500, I immediately stop the training process and re-start it with another parameter combination.
      ```
      Training...

      EPOCH 1 ...
      Validation Accuracy = 0.782

      EPOCH 2 ...
      Validation Accuracy = 0.911

      EPOCH 3 ...
      Validation Accuracy = 0.945

      EPOCH 4 ...
      Validation Accuracy = 0.949

      ...

      EPOCH 58 ...
      Validation Accuracy = 0.989

      EPOCH 59 ...
      Validation Accuracy = 0.989

      EPOCH 60 ...
      Validation Accuracy = 0.989
      ```
    2. The higher initial validation accuracy, the higher the possibility to achieve a good final accuracy. From my experiments, a very good final validation accuracy always starts from 0.7xx or above at the first iteration.
    3. To choose a good `EPOCH` value, how to decide if the number is good enough? My experience is that the validation accuracy should increase from low to high and finally converge to a number. For example, when I used `EPOCH=40`, I saw the validation accuracy kept increasing from 0.7xx to 0.96x without staying around 0.96x, so I increased the epoch value. In contrast, when I tried `EPOCH=80`, I saw the validation accuracy went to the highest number 0.99x and then decreased to 0.98x. Then I reduced the epoch value. You can see from above logs, when `EPOCH=60`, the validation accuracy stayed at 0.989 but no longer increased or decreased.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  - I also tried adding a dropout layer in the architecture. I run the training process with the keep probability values of 1.0 and 0.5, respectively. Meanwhile, I keep other parameters no changed. As a result, I find the final accuracies of the two cases are similar. So I just do not add the dropout layer consider its minor performance impact in my architecture.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web, the images are available in Alex Staravoitau's Github [2].

![Extra 10 images][image3]

- The 1st image might be difficult to classify because it looks similar to the sign of "road work".
- The 4th image may be difficult to recognize by the classifier because the sign does not face the camera directly, the plate's round white edge may confuse the classifier.
- The 8th and 9th images of the "roundabout mandatory" sign are difficult to classify because they both look very similar to the sign of the "Priority road".
- The last left turn ahead sign may be difficult to be classified because its pose makes it look like to the sign of "keeping left".

I refer to the Wikipedia [3] and the `signnames.csv` file to manually label each sign with its sign name and saved the label in a list called `my_labels` in the 20th cell.

![Top 5 Guesses of Each Image][image4]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| Double-curve                | Double-curve                                        |
| Keep Left                | Keep Left                                        |
| No entry             | No entry                                     |
| Roundabout Mandatory           | Priority Road                                     |
| Turn Left Ahead        | Turn Left Ahead                                      |


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 93.1%. Therefore, the model performs consistently well with the images in the test set and the images I found on the web.

#### 3. Describe how certain the model is when predicting each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22nd code block of the Ipython notebook. I referred to Jeremy Shannon's codes [1] and my contributions are:
1. Refactor the original codes to one function `showTopKSoftmax()` to improve the readability of the codes.
2. Jeremy's solution includes the keep probability for the dropout parameter tuning. I do not use this parameter because I tried with different keep probability values and finally observed there were NO better outputs than not using it.

For all the 10 images, the model always gives a certain belief (probability of 1.0) of the image to the first guess, and the probabilities of other guesses are 0.

For each image, the figure on its right side plots its top 5 softmax probabilities. The Y-axis is the possibility (belief) of the guess: the higher value indicates the stronger belief. The X-axis is the number of the label, ranging from 0 to 42.

For example, for the first sign, the figure on its right has a bar at 21st position in the X-axis, and the bar length is 1. This means the model considers that the sign has 100% possibility to be the 21st sign, namely, Double-curve, in terms of the corresponding name in the `signnames.csv` file.

![Softmax Probabilities][image11]

For the 1st image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | Double-curve (21)                                    |
| 0.0                     | Road work (25)                                        |
| 0.0                        | Beware of ice/snow (30)                                            |
| 0.0                      | Right-of-way at the next intersection (11)                |
| 0.0                      | Wild animals crossing (31)                                 |

For the 2nd image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | Keep left (39)                                    |
| 0.0                     | Turn right ahead (33)                                        |
| 0.0                        | Yield (13)                                            |
| 0.0                      | Go straight or left (37)                |
| 0.0                      | Ahead only (35)                                 |

For the 3rd image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | No entry (17)                                    |
| 0.0                     | Stop (14)                                        |
| 0.0                        | Bicycles crossing (29)                                            |
| 0.0                      | Traffic signals (26)                |
| 0.0                      | Turn right ahead (33)                                 |

For the 4th image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | No entry (17)                                    |
| 0.0                     | Speed limit (20km/h) (0)                                        |
| 0.0                        | Roundabout mandatory (40)                                            |
| 0.0                      |    Stop (14)                |
| 0.0                      | No passing for vehicles over 3.5 metric tons (10)    |

For the 5th image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | No entry (17)                                    |
| 0.0                     | Stop (14)                                            |
| 0.0                        | No passing for vehicles over 3.5 metric tons (10)            |
| 0.0                      |    Bicycles crossing (29)                            |
| 0.0                      | Speed limit (70km/h) (4)    |

An observation is that consider images 3 ~ 5, they represent the same sign and the model made the predictions correctly for all of them. But the top 5 guesses between any of the two images could not be the same due to the differences between the lightning conditions, view of angles, etc.

For the 6th image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | Keep left (39)                                    |
| 0.0                     | Yield (13)                                            |
| 0.0                        | Ahead only (35)            |
| 0.0                      |    Turn right ahead (33)                            |
| 0.0                      | Go straight or left (37)    |

For the 7th image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | Keep left (39)                                    |
| 0.0                     | Yield (13)                                            |
| 0.0                        | Ahead only (35)            |
| 0.0                      |    Turn right ahead (33)                            |
| 0.0                      | Go straight or left (37)    |

For the 8th image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | Priority road (12)                                    |
| 0.0                     | End of no passing (41)                                            |
| 0.0                        | Roundabout mandatory (40)            |
| 0.0                      |    Yield (13)                            |
| 0.0                      | End of no passing by vehicles over 3.5 metric tons (42)    |

For the 9th image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | Roundabout mandatory (40)                                        |
| 0.0                     | Right-of-way at the next intersection (11)                    |
| 0.0                        | Speed limit (100km/h) (7)        |
| 0.0                      |    Priority road (12)                            |
| 0.0                      | End of no passing by vehicles over 3.5 metric tons (42)    |

For the 10th image, the top five softmax probabilities are:

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                 | Turn left ahead (34)                                        |
| 0.0                     | Ahead only (35)                    |
| 0.0                        | No vehicles (15)        |
| 0.0                      |    Speed limit (120km/h) (8)                            |
| 0.0                      | No passing (9)    |


## References
- [1] [Jeremy Shannon's Github](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project)
- [2] [Alex Staravoitau's Github](https://github.com/navoshta/traffic-signs)
- [3] [Wikipedia: Road Signs in Germany](https://en.wikipedia.org/wiki/Road_signs_in_Germany)
- [4] [Pierre Sermanet and Yann LeCun's IJCNN Paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
