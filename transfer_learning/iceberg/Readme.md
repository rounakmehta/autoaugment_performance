This is our first example of how to "transfer" policies from AutoAugment.

To do this, we used the augmentation_transforms.py from the original paper github repo, which implements all the transformations.  We had to remove some hardcodings of image sizes from there to accomadate the 75x75 images from the iceberg dataset.

Then, in the Kaggle Kernel my-best-single-model-simple-cnn-lb-0-1541.py, we substitute the "get_more_images" function with one that calls "apply_policy" from augmentation_transforms.py. 

In order to change policies used, just change the following import in my-best-single-model-simple-cnn-lb-0-1541.py:

from shvn_policies import good_policies
or
from cifar10_policies import good_policies

The results are detailed in the paper.
