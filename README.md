# csc413-2516-assignment-2--convolutional-neural-networks-solved
**TO GET THIS SOLUTION VISIT:** [CSC413-2516 Assignment 2- Convolutional Neural Networks Solved](https://www.ankitcodinghub.com/product/csc413-2516-programming-assignment-2-convolutional-neural-networks-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;118067&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC413-2516 Assignment 2- Convolutional Neural Networks Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Version: 1.4

Changes by Version:

• (v1.4) Removed BatchNorm parameters from Part A question 3.

• (v1.3) Added BatchNorm references.

• (v1.2) Fixed a few typos.

Based on an assignment by Lisa Zhang

Submission: You must submit 2 files through MarkUs : a PDF file containing your writeup, titled a2-writeup.pdf, and your code file a2-code.ipynb. Your writeup must be typed.

The programming assignments are individual work. See the Course Information handout for detailed policies.

Introduction

Image Colourization as Classification

In this section, we will perform image colourization using three convolutional neural networks (Figure 1). Given a grayscale image, we wish to predict the color of each pixel. We have provided a subset of 24 output colours, selected using k-means clustering . The colourization task will be framed as a pixel-wise classification problem, where we will label each pixel with one of the 24 colours. For simplicity, we measure distance in RGB space. This is not ideal but reduces the software dependencies for this assignment.

We will use the CIFAR-10 data set, which consists of images of size 32×32 pixels. For most of the questions we will use a subset of the dataset. The data loading script is included with the notebooks, and should download automatically the first time it is loaded.

Helper code for Part A is provided in a2-code.ipynb, which will define the main training loop as well as utilities for data manipulation. Run the helper code to setup for this question and answer the following questions.

Part A: Pooling and Upsampling (2 pts)

1. Complete the model PoolUpsampleNet, following the diagram in Figure 1a. Use the Py-

Torch layers nn.Conv2d, nn.ReLU, nn.BatchNorm2d , nn.Upsample, and nn.MaxPool2d. Your CNN should be configurable by parameters kernel, num in channels, num filters, and num colours. In the diagram, num in channels, num filters and num colours are denoted NIC, NF and NC respectively. Use the following parameterizations (if not specified, assume default parameters):

• nn.Conv2d: The number of input filters should match the second dimension of the input tensor (e.g. the first nn.Conv2d layer has NIC input filters). The number of output filters should match the second dimension of the output tensor (e.g. the first nn.Conv2d layer has NF output filters). Set kernel size to parameter kernel. Set padding to the padding variable included in the starter code.

• nn.BatchNorm2d: The number of features should match the second dimension of the output tensor (e.g. the first nn.BatchNorm2d layer has NF features).

• nn.Upsample: Use scaling factor = 2.

• nn.MaxPool2d: Use kernel size = 2.

(a) PoolUpsampleNet (b) ConvTransposeNet (c) UNet

Figure 1: Three network architectures that we will be using for image colourization. Numbers inside square brackets denote the shape of the tensor produced by each layer: BS: batch size, NIC: num in channels, NF: num filters, NC: num colours.

Note: grouping layers according to the diagram (those not separated by white space) using the nn.Sequential containers will aid implementation of the forward method.

2. Run main training loop of PoolUpsampleNet. This will train the CNN for a few epochs using the cross-entropy objective. It will generate some images showing the trained result at the end. Do these results look good to you? Why or why not?

3. Compute the number of weights, outputs, and connections in the model, as a function of NIC, NF and NC. Compute these values when each input dimension (width/height) is doubled. Report all 6 values. Note: Please remove BatchNorm parameters when answering this question. Thanks!

Part B: Strided and Transposed Dilated Convolutions (3 pts)

1. Complete the model ConvTransposeNet, following the diagram in Figure 1b. Use the PyTorch layers nn.Conv2d, nn.ReLU, nn.BatchNorm2d and nn.ConvTranspose2d. As before, your CNN should be configurable by parameters kernel, dilation, num in channels, num filters, and num colours. Use the following parameterizations (if not specified, assume default parameters):

• nn.Conv2d: The number of input and output filters, and the kernel size, should be set in the same way as Part A. For the first two nn.Conv2d layers, set stride to 2 and set padding to 1.

• nn.BatchNorm2d: The number of features should be specified in the same way as for Part A.

• nn.ConvTranspose2d: The number of input filters should match the second dimension of the input tensor. The number of output filters should match the second dimension of the output tensor. Set kernel size to parameter kernel. Set stride to 2, set dilation to 1, and set both padding and output padding to 1.

2. Train the model for at least 25 epochs using a batch size of 100 and a kernel size of 3. Plot the training curve, and include this plot in your write-up.

The validation accuracy should be higher and validation loss should be lower. Some possible reasons include:

•

4. How would the padding parameter passed to the first two nn.Conv2d layers, and the padding and output padding parameters passed to the nn.ConvTranspose2d layers, need to be modified if we were to use a kernel size of 4 or 5 (assuming we want to maintain the shapes of all tensors shown in Figure 1b)?

Note: PyTorch documentation for nn.Conv2d and nn.ConvTranspose2d includes equations that can be used to calculate the shape of the output tensors given the parameters.

• kernelsize = 4, padding = 1, output padding = 0

• kernelsize = 5, padding = 2, output padding = 1

• kernel size = x, padding = (x – 1) // 2, output padding = int(x % 2 == 1)

5. Re-train a few more ConvTransposeNet models using different batch sizes (e.g., 32, 64, 128, 256, 512) with a fixed number of epochs. Describe the effect of batch sizes on the training/validation loss, and the final image output quality. You do not need to attach the final output images.

As batch size increases, validation loss increases (given the same number of epochs).

Part C: Skip Connections (1 pts)

A skip connection in a neural network is a connection which skips one or more layer and connects to a later layer. We will introduce skip connections to the model we implemented in Part B.

1. Add a skip connection from the first layer to the last, second layer to the second last, etc. That is, the final convolution should have both the output of the previous layer and the initial greyscale input as input (see Figure 1c). This type of skip-connection is introduced by Ronneberger et al. [2015], and is called a “UNet”. Following the ConvTransposeNet class that you have completed, complete the init and forward methods of the UNet class in Part C of the notebook.

Hint: You will need to use the function torch.cat.

2. Train the model for at least 25 epochs using a batch size of 100 and a kernel size of 3. Plot the training curve, and include this plot in your write-up.

3. How does the result compare to the previous model? Did skip connections improve the validation loss and accuracy? Did the skip connections improve the output qualitatively? How? Give at least two reasons why skip connections might improve the performance of our CNN models.

• Skip connections improve the vanishing gradient problem.

• Skip connections allow some of the information that was lost in the pooling or strided convolution layers to be reintroduced to the final output layers.

Object Detection

In the previous two parts, we worked on training models for image colourization. Now we will switch gears and perform object detection by fine-tuning a pre-trained model.

Object detection is a task of detecting instances of semantic objects of a certain class in images. Fine-tuning is often used when you only have limited labeled data.

Part D.1: Fine-tuning from pre-trained models for object detection (2 pts)

1. A common practice in computer vision tasks is to take a pre-trained model trained on a large datset and finetune only parts of the model for a specific usecase. This can be helpful, for example, for preventing overfitting if the dataset we fine-tune on is small.

To keep track of which weights we want to update, we use the PyTorch utility

Model.named parameters() , which returns an iterator over all the weight matrices of the model.

Complete the model parameter freezing part in train.py by adding 3-4 lines of code where indicated. See also the notebook for further instruction.

Part D.2: Implement the classification loss (2 pts)

1. See the notebook for instructions for this part. You will fill in the BCEcls loss used for the classification loss.

2. Visualize the predictions on 2 sample images by running the helper code provided.

What you have to submit

For reference, here is everything you need to hand in. See the top of this handout for submission directions.

• A PDF file titled a2-writeup.pdf containing only the following:

– Answers to questions from Part A

∗ Q1 code for model PoolUpsampleNet (screenshot or text)

∗ Q2 visualizations and your commentary

∗ Q3 answer (6 values as function of NIC, NF, NC) – Answers to questions from Part B

∗ Q1 code for model ConvTransposeNet (screenshot or text)

∗ Q2 answer: 1 plot figure (training/validation curves)

∗ Q3 answer

∗ Q4 answer

∗ Q5 answer

– Answers to questions from Part C

∗ Q1 code for model UNet (screenshot or text)

∗ Q2 answer: 1 plot figure (training/validation curves)

∗ Q3 answer

– Answers to questions from Part D.1

– Answers to questions from Part D.2

∗ Q2 answer: visualization of predictions (screenshot or image files)

• Your code file a2-code.ipynb

References

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computerassisted intervention, pages 234–241. Springer, 2015.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.
