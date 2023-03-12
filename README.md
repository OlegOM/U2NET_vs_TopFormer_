# U2NET vs TopFormer #

# Solution plan:

* **1** : Split labeled data to train and test set for fine-tuning and evaluation purposes.
Added script: **train_test_splitter.py**
  _Finished on **Thursday**._ 

* **2** : Train U2Net on training set data, evaluate on test set using IoU, record the results.
  Added script: **infer_and_measure.py**.  
  Produced file **metrics.csv** 
  Added script **calculate_iou.py** to callculate final **iOU** with **JaccardIndex** method: **0.45947448796658585** 
  _Finished on **Thursday**._

* **3** : Cut TopFormer NN from its repo towards cloth-segmentation pipeline of training and evaluatio.
  Added module: **TopFormer** to networks module  
  _Finished on **Sunday**._
  
* **4** : Train TopFormer NN on training set data, evaluate on test set using IoU, record the results.
  **STUCK**
  
* **5** : Compare the results and produce report
  **STUCK**

# STUCK
Stuck on part 3 due to complexity of disentangling of all the parts related to NN (backbone, head, optimizer, loss, configurations) from the codebase.
The main issue with TopFormer repo was difficulty in transferring Topformer neural network to cloth-segmentation repo. There are already useful scripts in cloth-segmentation repo related to dataset preprocessing (loading annotations from csv into tensor format, image preprocessing), training loop and inference as well as IoU calculations. The plan was to leverage those scripts with the TopFormer model instead of transferring the lot to the training and testing routine of Topformer library. There were 4 places of train script that needed to be modified: model definition, optimizer, loss and weights initialization. Model definition took a lot of effort, as in the TopFormer library the model is built via Registry from its backbone, head, configs, optimizer and loss. I ran out of time as I was refactoring the model to enable using the training script of cloth-segmentation repo. However, given several more days, I am confident that I would have been able to transfer the model and quickly finish points 4-5 of the plan.

# Part 2 
Check the “Dress Code” dataset. Based on the annotations offered by this dataset which include landmarks, segments and skeleton - do you believe training with that dataset would give better results than the one you’ve had with the TopFormer model? Explain why.

I believe that more data is better than less data. Sensor fusion approaches are successfully and extensively used in autonomous vehicles and there are experiments in the field of transformers and multi-modal learning (e.g. https://arxiv.org/pdf/2206.06488.pdf). With extra annotations containing information on human dense pose, keypoints and skeleton, we could leverage those as inputs to a neural network combining the information from annotations to arrive at a prediction regarding the segmentation to hopefully achieve better quality than from raw image alone. My intuition is that by doing so we would augment the perception of neural networks with the things that we humans know about the person on an image like pose or keypoints.

_Finished on **Thursday**._

# Bonus Question
1. Quantization
   Quantization is primarily a technique to speed up inference and only the forward pass is supported for quantized operators. PyTorch supports multiple approaches to quantizing a deep learning model. In most cases the model is trained in FP32 and then the model is converted to INT8.
   https://pytorch.org/docs/stable/quantization.html

2. Knowledge distillation
   Knowledge distillation refers to the process of transferring knowledge from a large model to a smaller one.

3. Pruning:
   Neural network pruning is a method of compressing a model by removing some of the parameters. 
   Pruning, like quantization, can be used before, after and during training.
   The easiest way to prune a neural network is under the hood of PyTorch, and it is implemented with just one line of code.

    torch.nn.utils.prune.random_unstructed(module , name = ‘weight’ , amount = 0.3)

4. Combination of 1-3


5. iOS Convert PyTorch to CoreML
   iOS feature, converts PyTorch to CoreMLL.
   https://coremltools.readme.io/docs/pytorch-conversion


6. TFLight on Android


7. PyTorch Mobile
   Converts PyTorch models directly for mobile using:
   https://pytorch.org/mobile/home/


_Finished on **Thursday**._