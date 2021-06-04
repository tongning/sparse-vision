# sparse-vision

Train MobileNetV2 and ResNet18 for image classification on [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), with
support for a variety of transformations to train on sparse representations of the images. Written for CS230 project, [Evaluation of Sparse Image Representations for Visual Recognition](https://drive.google.com/file/d/1UhQ6rov98XyQSB3haUhGNpiidMboEKNf/view?usp=sharing).

`train_general.py` is the main training script, run with `-h` to see available options. See this [Colab Notebook](https://colab.research.google.com/drive/1bH49PIHsJKYdH6AcCFoasvEN6BMjphtR?usp=sharing) for sample usage.

`sift` and `hog` sparse representations cannot be generated within the training script, use `utils/transform_image_folder.py` to generate these representations.

Use `utils/reformat_tinyimagenet_val_test.py` to reorganize the Tiny ImageNet validation images before training.
