## Sample Generarion Job
This job aims to generate samples of sysnthetic data based on template document and bounding boxes. The template document will be filled with handwritten text using bounding boxes information.
The generated data is versioned and subdivided into train and test datasets.

### Jobs Input
- <b>document_id</b>: the id of the document, this is used to retrieve the document template and buonding boxes via API
- <b>num_samples</b>: this parameter specify the number of samples to generate

### Degradations
During the job execution a set of degradations will be applied randomly to the document image to make it appear more real.
The degradations are defined as classes in `image_degradations.py`.
While `ImageDegradator` class allows to use them in combo to generate a realistic document image.
Here is the list of degradations currently available with some examples:
> #### GaussianBlurDegradation
> ![GaussianBlurDegradation](../../images/gaussian_blur.png)

> #### MotionBlurDegradation
>![MotionBlurDegradation](../../images/motion_blur.png)

> #### GaussianNoiseDegradation
>![GaussianNoiseDegradation](../../images/gaussian_noise.png)

> #### SaltPepperNoiseDegradation
>![SaltPepperNoiseDegradation](../../images/salt_and_pepper.png)

> #### BrightnessContrastDegradation
>![BrightnessContrastDegradation](../../images/brightness_and_contrast.png)

> #### WaveDistortionDegradation
>![WaveDistortionDegradation](../../images/wave_distortion.png)

> #### ShadowDegradation
>![ShadowDegradation](../../images/shadow.png)

> #### ColorFilterDegradation
>![ColorFilterDegradation](../../images/color_filter.png)