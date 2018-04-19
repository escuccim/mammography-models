# ConvNets for Detection of Abnormalities in Mammograms
Eric Scuccimarra (skooch@gmail.com)

## Abstract
Mammography is the most common method of detecting breast cancer. Early detection significantly improves survival rates, and between 8% and 25% of abnormalities go undetected. We trained ConvNets on the DDSM dataset to detect the presence of lesions and predict the class and pathology of the lesions.

We were able to achieve an accuracy of 99% on determining whether scans were normal or abnormal.

## Introduction
Breast cancer is the second most common cancer in women worldwide. About 1 in 8 U.S. women (about 12.4%) will develop invasive breast cancer over the course of her lifetime. The five year survival rates for stage 0 or stage 1 breast cancers are close to 100%, but the rates go down dramatically for later stages: 93% for stage II, 72% for stage III and 22% for stage IV. Human recall for identifying lesions is estimated to be between 0.75 and 0.92 [1], which means that as many as 25% of abnormalities may go undetected. 

The DDSM is a dataset of normal and abnormal scans; however the size is relatively small. To increase the size of the dataset we extract the Regions of Interest (ROI) from each image, perform data augmentation and then train ConvNets on the augmented data. The ConvNets were trained to predict both whether a scan was normal or abnormal, and to predict whether abnormalities were calcifications or masses and benign or malignant.

## Related Work
There exists a great deal of research into applying deep learning to medical diagnosis, but privacy concerns make the availability of training data a limiting factor, and thus there is not much research into applying ConvNets to mammography. [1, 4] use ConvNets to classify pre-detected breast masses by pathology and type, but do not attempt to detect masses from scans. [2,3] detect abnormalities using combinations of region-based CNNs and random forests. 

## Datasets
The DDSM [6] is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information. The CBIS-DDSM [8] collection includes a subset of the DDSM data selected and curated by a trained mammographer. The CBIS-DDSM images are better quality than the DDSM images, but this dataset does not contain normal images. Normal images were taken from the DDSM dataset and combined with the abnormal images from the CBIS-DDSM in order to have all classes represented.

Data from the University of California Irvine Machine Learning Repository [5] was also used for exploratory data analysis to gain insight into the characteristics of abnormalities.

## Methods
The DDSM and CBIS-DDSM datasets are relatively small. The images were pre-processed with data augmentation to create a dataset of reasonable size. ConvNets were then constructed and trained on the data using multiple labelling methods.

### Data Augmentation
The CBIS-DDSM scans were of relatively large size, with a mean height of 5295 pixels and a mean width of 3131 pixels. Masks highlighting the ROIs were provided. In order to create training data, the ROIs were extracted from the abnormal images using the masks and sized down to 299x299 pixels. 

The ROIs had a mean size of 450 pixels and a standard deviation of 396. In the interest of representing each ROI as well as possible, each ROI was extracted in multiple ways:
1.	The ROI was extracted at 598x598 at its original size.
2.	The ROI was zoomed to 598x598, with padding to provide context.
3.	If the ROI had the size of one dimension more than 1.5 times the other dimension it was extracted as two tiles centered in the center of each half of the ROI along it's largest dimension.

The 598x598 images were then resized to 299x299. In order to increase the size of the dataset data augmentation was used, including randomly positioning the ROI within the image, random horizontal flipping, random vertical flipping and random rotation. The ROIs were extracted using two systems of margins which will be detailed below.

The normal scans from the DDSM dataset did not have ROIs so were processed differently. As these images had not been pre-processed they contained artifacts such as white borders and white patches of pixels used to cover up personal identifying information. To remove the borders each image was cropped by 7% on each side. In order to attempt to keep these images on the same scale as the CBIS-DDSM images, the DDSM images were sized down by a random factor between 1.8 and 3.2, then segmented into 299x299 tiles with a variable stride between 150 and 200. Each tile was then randomly rotated and flipped.
 
To avoid the inclusion of images which contained overlay text, or were mostly black background, each tile was then added to the dataset only if it met upper and lower thresholds on mean and variance. The thresholds were determined through random sampling of tiles and tuning of the thresholds to eliminate images which did not contain usable data.

Multiple datasets were created using different data augmentation techniques. The datasets ranged in size from 27,000 training images to 62,000 training images. The datasets had differing amounts of data augmentation, and different techniques were used to extract the ROIs for each. 

### ROI Context
The analysis of the UCI data indicated that the edges of an abnormality were important as to determining its pathology and type, and this was confirmed by a radiologist. Levy et al [1] also report that the inclusion of context was an important contributor to the accuracy of the classification.

Of the training datasets created, two were used for training:

1. Dataset 5 consisted of 39,316 images. The dataset was created using padding around each ROI which was randomly set between 30 and 50 pixels. 
2. Dataset 8 consisted of 40,559 images. This dataset used the extraction methodology described above to provide greater context for each ROI.  


### Data Balance
Only about 10% of mammograms are abnormal, in order to maximize recall we weighted our training data more heavily towards abnormal scan, with a target of 85% normal. As each ROI was extracted to multiple images, in order to prevent different images of the same ROI appearing in both the training and test data, the existing divisions of the CBIS-DDSM data were maintained. The test data was divided evenly between test and validation data with no shuffling to avoid overlap.

The normal images had no overlap, so were shuffled and divided among the training, test and validation data. The final divisions were 80% training, 10% test and 10% validation.

### Labels
In the DDSM dataset the scans are grouped into the following categories:
1.	Normal
2.	Benign Calcification
3.	Malignant Calcification
4.	Benign Mass
5.	Malignant Mass

As previous work [1] has already dealt with classifying pre-identified abnormalities, we focused on classifying images as normal or abnormal.

When classifying into all five categories, the predictions were also "collapsed" into binary normal/abnormal in order to measure the precision and recall. 

### ConvNet Architecture
Our first thought was to train existing ConvNets, such as VGG or Inception, on this dataset. However a lack of computational resources made this impractical. 

The features of medical scans are very different from the features of ImageNet images used to train these ConvNets and we were concerned that the large size of these models might lead to overfitting. For these reasons we decided to design our own architecture specifically for this task, attempting to keep the models as simple as possible. 

We started with a simple model based on VGG, consisting of stacked 3x3 convolutional layers alternating with max pools followed by fully connected layers. This architecture was iteratively improved, with each iteration changing only one aspect in the architecture and then being evaluated. Techniques evaluated include Inception-style branches [16, 17, 18] and residual connections [19]. 

The architecture was designed so that the same model could be used for both binary classification and multi-class classification by retraining the fully connected layers. In order to maximize recall a weighted cross entropy loss function was used giving abnormal scans double the weight of normal scans.

The models were constructed using TensorFlow and metrics were logged to TensorBoard. Batch normalization [15] was used for every layer, with dropout applied to the fully connected and pooling layers, and L2 regularization applied to all layers, with different lambdas for convolutional layers and fully connected layers.

The best performing architecture will be detailed below.

### Training
For the model selection phase models were trained on Dataset 5 through 50 epochs with binary labels. Accuracy, precision, recall and f1 score were used to evaluate the models. 

The models which performed well on Dataset 5 were retrained from scratch on Dataset 8 classifying to all 5 categories, on the assumption that this would cause the convolutional filters to extract the most important features.

Once the models had been trained on Dataset 8 with all classes, the convolutional layers were frozen and the fully connected layers were then retrained for the normal/abnormal binary classification.

We had considered using transfer learning from VGG or Inception, but decided that the features of the ImageNet data were different enough from those of radiological scans that it made more sense to learn the features from scratch on this dataset. However, the use of transfer learning between models greatly sped up the training process saving weeks of training time.

## Results
### Architecture
The best two performing models were 1.0.0.28 and 1.0.1.39. 

Model 1.0.0.28 consists of stacked 3x3 convolutions alternating with max pools followed by two fully connected layers with 2048 units each.
<img src="model_1.0.0.29f.png" alt="Model 1.0.0.28" align="right" style="max-width: 50%;">

Model 1.0.0.39 is very similar to 1.0.0.28, but with one extra fully connected layer, and two extra convolutional layers. As the abnormal mammograms had abnormalities ranging in size from several pixels to several hundred pixels a branch was inserted after layer 1 to attempt to detect very small features. The branches are concatenated together before max pool layer 1. This model also has an extra fully connected layer.
<img src="model_1.0.1.41a.png" alt="Model 1.0.1.39" style="max-width: 50%;" align="right" >

### Performance

Both models performed better than expected on Dataset 5, but when retrained from scratch on Dataset 6 the simpler model did not perform well.

|Model      |Dataset    |Accuracy    |Recall      |
|-----------|-----------|------------|------------| 
|1.0.1.39n  |          5|.9935       |.9590       |
|1.0.1.39n  |          6|     |     |
|1.0.0.28n  |          5|.9903       |.9431       |
|1.0.0.28n  |          6|     |     |
_Table 1: Performance on Test Set_

Model 1.0.0.28 performed excellent on both the training and validation data, as seen in Figure 1.

When retrained on Dataset 6 model 1.0.0.28 scored similar on the training data as it did on the smaller dataset, but the model did not generalize to the validation data. On the validation data the model seemed to predict everything as negative, which kept the accuracy near the most frequent baseline. The precision was excellent, but the recall approached zero. This is shown in Figure 2. 

_figures 1 and 2 - model 1.0.0.28_

Model 1.0.1.39 also performed remarkably well on both the training and validation data on Dataset 5, but also performed well on Dataset 6. This model was more complicated than model 1.0.0.28, and included a branch to attempt to detect smaller abnormalities. 

_figures 3 and 4 - model 1.0.1.39_

The use of multiple branches was evaluated on Set 5, and while they did provide better results on the training data they seemed to make the model generalize to the validation data more poorly so were not included.

### Decision Thresholds
These results were obtained using a threshold of 0.50. The precision and recall could be drastically altered by changing the decision threshold. It was suprisingly easy to achieve a precision of close to 1.0, however we were focused on improving recall. Adjusting the threshold from between 0.20 to 0.50 allowed us to improve recall by a few percentage points while decreasing precision dramatically. 

This could be very useful for radiologists, allowing them to screen out scans which are either definitely negative or definitely positive and allowing them to focus on the more ambiguous scans.

## Conclusion
We have demonstrated that Convolutional Neural Networks can be trained to determine whether a section of a mammogram contains an abnormality with recall of 95%, substantially above human performance. Adjusting the decision threshold would further improve the recall. These methods could be used to pre-screen mammograms allowing radiologists to focus on scans which are likely to contain abnormalities.

Future work would include creating a system which would take a full mammogram as input and analyse it, possibly with a sliding window, to determine whether the image contains abnormalities. Levy et al [1] have shown that ConvNets can be used to classify pre-identified ROIs, those techniques can be combined with the techniques demonstrated here to create a complete end-to-end system for classifying mammograms. Another option would be to use algorithms such as YOLO [20] or R-CNN [21] on an entire scan to attempt to identify and locate potential abnormalities. 

## References
[1]	D. Levy, A. Jain, Breast Mass Classification from Mammograms using Deep Convolutional Neural Networks, arXiv:1612.00542v1, 2016

[2]	N. Dhungel, G. Carneiro, and A. P. Bradley. Automated mass detection in mammograms using cascaded deep learning and random forests. In Digital Image Computing: Techniques and Applications (DICTA), 2015 International Conference on, pages 1–8. IEEE, 2015.

[3]	N.Dhungel, G.Carneiro, and A.P.Bradley. Deep learning and structured prediction for the segmentation of mass in mammograms. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 605–612. Springer International Publishing, 2015.

[4]	J.Arevalo, F.A.González, R.Ramos-Pollán,J.L.Oliveira,andM.A.G.Lopez. Representation learning for mammography mass lesion classiﬁcation with convolutional neural networks. Computer methods and programs in biomedicine, 127:248–257, 2016.

[5]	Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

[6]	The Digital Database for Screening Mammography, Michael Heath, Kevin Bowyer, Daniel Kopans, Richard Moore and W. Philip Kegelmeyer, in Proceedings of the Fifth International Workshop on Digital Mammography, M.J. Yaffe, ed., 212-218, Medical Physics Publishing, 2001. ISBN 1-930524-00-5.

[7]	Current status of the Digital Database for Screening Mammography, Michael Heath, Kevin Bowyer, Daniel Kopans, W. Philip Kegelmeyer, Richard Moore, Kyong Chang, and S. Munish Kumaran, in Digital Mammography, 457-460, Kluwer Academic Publishers, 1998; Proceedings of the Fourth International Workshop on Digital Mammography.

[8]	Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016). Curated Breast Imaging Subset of DDSM. The Cancer Imaging Archive.

[9]	Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057.

[10]	O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.

[11]	William H. Wolberg and O.L. Mangasarian: "Multisurface method of pattern separation for medical diagnosis applied to breast cytology", Proceedings of the National Academy of Sciences, U.S.A., Volume 87, December 1990, pp 9193-9196.

[12]	O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition via linear programming: Theory and application to medical diagnosis", in: "Large-scale numerical optimization", Thomas F. Coleman and YuyingLi, editors, SIAM Publications, Philadelphia 1990, pp 22-30.

[13]	K. P. Bennett & O. L. Mangasarian: "Robust linear programming discrimination of two linearly inseparable sets", Optimization Methods and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).

[14]	K. Simonyan, A. Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, arXiv:1409.1556, 2014

[15]	S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of The 32nd International Conference on Machine Learning, pages 448–456, 2015

[16]	C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015.

[17]	C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. arXiv preprint arXiv:1512.00567, 2015.

[18]	C. Szegedy, S. Ioffe, V. Vanhoucke, Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, arXiv:1602.07261v2, 2016

[19]	K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, arXiv:1512.03385, 2015

[20]  J. Redmon, S. Divvala, R. Girshick, A. Farhadi, You Only Look One: Unified, Real-Time Object Detection, arXiv:1506.02640, 2015

[21] R. Girshick, J. Donahue, T. Darrell, J. Malik, Rich feature hierarchies for accurate object detection and semantic segmentation, arXiv:1311.2524, 2013