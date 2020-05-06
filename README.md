# Introduction of a new Dataset and Method for Detecting and Counting the Pistachios based on Deep Learning

Pistachio is a nutritious nut that has many uses in the food industry. Iran is one of its largest producers, and pistachio is considered as a strategic export product for this country. This product has a great variety, most of which are cultivated in Iran and taken to other countries. Pistachios are sorted based on the shape of their shell into two categories: Open-mouth and Closed-mouth. The open-mouth pistachios are higher in price, value, and demand than the closed-mouth pistachios.
In the countries that are famous in pistachio production and exporting, there are companies that pack the picked pistachios from the trees and make them ready for exporting. As there are differences between the price and the demand of the open-mouth and closed-mouth pistachios, it is considerable for these companies to know precisely how much of these two kinds of pistachios exist in each packed package. In this paper, we have introduced and shared a new dataset of pistachios, which is called Pesteh-Set. This dataset is prepared by a company in Iran and has been recorded from a cell-phone camera above the line that transported the pistachios.
Pesteh-Set includes 6 videos with a total length of 164 seconds and 561 moving pistachios. It also contains 423 labeled images that totally include 3927 labeled pistachios. At the first stage, we have used RetinaNet, the deep fully convolutional object detector for detecting the pistachios in the video frames. In the second stage, we introduce our method for counting the open-mouth and closed-mouth pistachios in the videos. Pistachios that move and roll on the transportation line may appear as closed-mouth in some frames and as open-mouth in other frames. With this circumstance, the main challenge of our work is to count these two kinds of pistachios correctly and fast.
Our introduced method performs very fast with no need for GPU, and it also achieves good counting results. The computed accuracy of our counting method is 94.75%.
Our introduced methods can be remotely performed by using the videos taken from the implemented cameras that could monitor the pistachios.

Pesteh-set is available on https://github.com/mr7495/Pesteh-Set

The main purpose in our paper was to count the open-mouth and closed-mouth pistachios in videos. At the first stage, we have to generate the frames of the video and detect the pistachios in them with RetinaNet.

IWe have separated the dataset into five-folds and allocated 20 percent of the dataset for validation and the rest for the training. After the detection phase, we present the method we used for counting the open-mouth and closed-mouth pistachios. This counting algorithm runs very fast with good accuracy. The general schematic of our work is presented in next fig.

<p align="center">
	<img src="images/graphical_abstract.jpg" alt="photo not available" width="100%" height="70%">
	<br>
	<em>The General View of our proposed method for counting the pistachios</em>
</p>

The images of the dataset were preprocessed and then resized to 1070×600 pixels.

We trained and validated RetinaNet on 3 different backbones: ResNet50, ResNet152, and VGG16. Transfer learning from the ImageNet pre-trained weights was utilized at the beginning of the training to speed up the network convergence. We also used data augmentation methods to improve the learning efficiency and stop the network from overfitting. The details of each fold are present is the next table.

Fold  | Training Images | Validation Images | Open-mouth Pistachios in Training Set | Closed-mouth Pistachios in Training Set | Open-mouth Pistachios in Validation Set | Closed-mouth Pistachios in Validation Set
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
Fold1 | 339 | 84 | 1600 | 1550 | 393 | 384
Fold2 | 339 | 84 | 1610 | 1572 | 383 | 362
Fold3 | 339 | 84 | 1553 | 1506 | 440 | 428
Fold4 | 339 | 84 | 1641 | 1575 | 352 | 359
Fold5 | 336 | 87 | 1568 | 1533 | 425 | 401
