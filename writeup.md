# **Traffic Sign Recognition**

## Writeup

---

**Project Outlines**

The goals / steps of this project are the following:
1. Load the data set (see below for links to the project data set)
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./images/dataset_distribution.png "Dataset Visualization"
[image2]: ./images/lenet_cnn.jpg "LeNet Architecture"
[image3]: ./images/extra_10.png "10 Extra German Traffic Signs"
[image4]: ./images/top_k.png "Top 5 Guess of 10 Extra Images"
[image5]: ./images/rgb.png "Traffic Sign RGB"
[image6]: ./images/gray.png "Traffic Sign Grayscale"
[image7]: ./images/gray_5.png "Traffic Sign Grayscale 2"
[image8]: ./images/gray_norm_5.png "Traffic Sign Grayscale 2 Normalized"
[image9]: ./images/modifiedLeNet.jpeg "Modified LeNet Architecture"

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

Here is an exploratory visualization of the data set. The figure below shows the bar charts representing the data distributions in the train set (blue color), the test set (red), and the validation set (green), respectively.

The conclusion is that the data distributions of the train, test, and validation sets are mostly the consistent with each other. Therefore, our learning algorithm runs under a good premise of datasets.

![Dataset distributions][image1]

### Design and Test a Model Architecture

#### 1.Preprocessed the image data.

I used two techniques to preprocess the input images.
1. Convert RGB images to grayscale.
2. Normalized the grayscale images.

Recall in the LeNet Lab, the LeNet classifier takes grayscale MNIST data, which is as set of `32x32x1` images as one of its inputs. Hence, in order to re-use the LeNet classifier in this project with the **minimum** architecture changes, I want to keep the inputs format as the same as the MNIST dataset and decide to first convert the rgb images (`32x32x3`) to gray scale ones.

When converting the image, I referred to the codes of Jeremy Shannon [1] because I got the runtime exceptions while invoking the openCV `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` functions. It turns out Jeremy's method is more straightforward and efficient using numpy's vector operation against the image array.

Here is an example of a traffic sign image before and after grayscaling.

![rgb image][image5] ![grayscale image][image6]

Because the image data should be normalized so that the data has mean zero and equal variance. I follow the suggested way, namely, `(pixel - 128)/ 128`, to approximately normalize the data and can be used in this project.

The codes are in the 8th code block of the notebook. The pictures below show the grayscale image on the left and normalized grayscale image on the right.

![grayscale image][image7] ![normalized grayscale image][image8]

#### 2. Final model architecture

Jeremy Shannon obtained impressive results with a quite high accuracy on the train and test datasets in his work [1]. However, he used a modified CNN architecture (shown below) by splitting the results of the 1st convolution layer into two parts, then apply a convolution layer to just one set and join with the other set with the results together.

![Modified LeNet Architecture][image9]

(The above architecture figure is from the [IJCNN paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) of Pierre Sermanet and Yann LeCun.)

My idea of solving this project is simpler: just re-use the LeNet architecture in the Lab solution as much as possible. First because I believe that the architecture introduced by Yann LeCun ([LeNet paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)) should work well with the traffic sign image datasets. Secondly, I am curious to know how would the original LeNet architecture perform with the traffic sign dataset.

In my 12th code block of the notebook, I keep the most of the codes in the LeNet Lab. The only change I made is to change the dimension of the output layer from 10 to 43, which is the actual number of classes of signs, as the figure shows below:

![LeNet Architecture][image2]

My final model consisted of the following layers:

| Layer         		|     Description	                   					|
|:-----------------:|:-------------------------------------------:|
| Input         		| 32x32x1 Gray image   							          |
| Convolution    	  | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU				    	|												                      |
| Max Pooling	    	| 2x2 stride, outputs 14x14x6	 	   		        |
| Convolution 	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU	           	|                             								|
| Max Pooling 			| 2x2 stride, outputs 5x5x16 									|
|	Flatten 					|	outputs 400            											|
| Fully Connected		|	outputs	120 	             									|
| RELU	           	|                             								|
| Fully Connected		|	outputs	84  	             									|
| RELU	           	|                             								|
| Fully Connected		|	outputs	43  	             									|

#### 3. Steps to train my model.

To train the model, I used an different combination of EPOCH, BATCH_SIZE, and learn_rate values. I tried EPOCH values in `{30, 40, 50, 60}` and choose the BATCH_SIZE in `{90, 100, 128, 200}`. I also tried to tune the learn_rate use larger or smaller value than the default `0.001`, such as `0.0005, 0.0009, 0.0015, 0.002`. For the other parameters I use the same values as the ones in the LeNet Lab solution: `mu=0, sigma=0.1`. For each combination, I had to rerun the training process to see the results and compare the accuracy.

I finally choose `EPOCH=50, BATCH_SIZE=100, learn_rate=0.001` this combination of the parameters because their results are the best!

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.992
* test set accuracy of 0.933

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web, the source images are in Alex Staravoitau's Github [2].

![Extra 10 images][image3]

The first image might be difficult to classify because it looks similar to the sign of children crossing the road. The forth image may be difficult to recognize by the classifier because the sign does not face the camera directly, the plate's round white edge may confuse the classifier. The last left turn ahead sign may be difficult to be classified because its pose makes it look like to the sign of "keeping left".

![alt text][image4]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

## References
- [1] [Jeremy Shannon's Github](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project)
- [2] [Alex Staravoitau's Github](https://github.com/navoshta/traffic-signs)
- [3] [Wikipedia: Road Signs in Germany](https://en.wikipedia.org/wiki/Road_signs_in_Germany)
- [4] [Pierre Sermanet and Yann LeCun's IJCNN Paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
