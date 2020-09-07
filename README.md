# Speckle Noise Reduction and Image Enhancement of Echocardiographic Images for Early Detection of Heart Diseases #

*Abstract :*
>The use of ultrasound imaging in medical diagnosis is well established due to its
>non-invasiveness, low cost, capability of forming real time imaging, and continuing
>improved image quality. However, in taking ultrasound images there is still a noise
>that causes the decreases of the ultrasound image quality, one of the most disturbing
>one is speckle noise. Speckle noise has long been known as a factor that limits the
>detection of ultrasound images, especially in low contrast images. There are several
>methods that can be used in reducing noise produced by ultrasound images, in this
>paper the method used is Hybrid Oriented Speckle noise Reducing Anisotropic
>Diffusion (HOSRAD) which is a combination of the Relaxed Median Filter method
>and the Oriented Speckle Reducing Anisotropic Diffusion (OSRAD) method. And
>if the doctor wants to obtain the edges of heart boundaries and heart valves
>movement, the method used is Canny Edge Detection. There are several parameters
>used as quantitative statistical analysis in order to see the differences in ultrasound
>images that has been restored with the original ultrasound images used as a
>reference, namely Signal to Noise Ratio (SNR), Peak Signal to Noise Ratio (PSNR),
>and Mean Square Error (MSE), Figure of Merit (FOM), and Structural Similarity
>quality Index (SSIM). After various experiments, the results for the best image
>restoration with a gamma value of 0.5 at the gamma correction process, a time step
>of 0.1 at the OSRAD process, and upper and lower boundaries of 5 and 9 at the
>Relaxed Median Filter. So, we get the results of the HOSRAD method with MSE
>and PSNR values of 70.13 and 29.67 with 5 iterations. The results of the proposed
>method are proven to be able to improve the quality of heart images which from the
>beginning had MSE and PSNR of 76,123 and 29.31.

## Introduction ##

This final project was created in order to completed my study courses. In this project, i am using 3 methods of image processing :

* Power Law Transform
* Oriented Speckle Reducing Anisotropic Diffusion (OSRAD)
* Relaxed Median Filter

For OSRAD and Relaxed Median Filter are combined to be Hybrid Oriented Speckle Reducing Anisotropic Diffusion (HOSRAD). 

I made python file for each methods to be used as a library. I'm not yet build a GUI for this project.
