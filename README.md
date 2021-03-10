# Mask_Detector_CNN
White Mask Detector with a Convolutional Neural Network 

Inspiration and Data Set: https://github.com/prajnasb/observations

Used for a TU Berlin Rasperry PI Master Project - WS 2020/21

Next Steps:
- expand Data Set for different mask colors, light and position conditions 
- train model again - Testing for the accuracy that works out best for real life apllication - enhance the robustness
- train model with GPU instead of CPU
- Visualize statistical Measurments 


Usage:

You first have to train a model, which due to capacity space was not possible to upload here. For that, use the MaskDetectorTraining file first. The Training data you can get from the source posted above. 
Afterwards you can use the slim MaskDetector-Script to use your model or if you for example want to use it on a device with low processor power the MaskDetetcor-Script with the associated lite-model will work out for you. 

&ensp;

![Mutation](https://user-images.githubusercontent.com/79472608/110202702-bc8e2f00-7e6a-11eb-84ec-f092296670e0.png)
