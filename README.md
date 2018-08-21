# autoencoder_for_background_extraction
Background extraction is an import step for background subtraction. This small project shows how to model background using convolutional autoencoder.   
The codes base on the project[1] which provides pure tensorflow code to build a convolutional autoencoder.
The ssim index is used as the loss function. The tensorflow code of ssim can be found in [2].

Reference:  
[1] https://github.com/Seratna/TensorFlow-Convolutional-AutoEncoder  
[2] https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow  
