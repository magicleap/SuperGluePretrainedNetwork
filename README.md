# SuperGlue inference library

This is a very thin python library on top of awesome research work by Magic Leap. 
Some pieces of original code have been cutout to simplify usage for my own needs.

## Disclaimer 

The library copies the original work [license](license). The library is provided as is and is not supposed to be used anywhere but personal experimental and educational purposes.
 
Please follow [the original repo](https://github.com/magicleap/SuperGluePretrainedNetwork) for any questions, issues, original code etc.


## Installation
`pip install git+https://github.com/arsenyinfo/SuperGluePretrainedNetwork`  

## Usage

Just extract SuperPoint descriptors: 
```
from superglue import SuperPoint
from superglue.utils import read_image

model = SuperPoint(config={}).train(False)
img_path = os.path.join(os.path.dirname(__file__), 'lena_color.png')
img = read_image(img_path)

with torch.no_grad():
    res = model({'image': img})
```

Full pipeline: 
```
from superglue import Matching
from superglue.utils import read_image

model = Matching(config={}).train(False)
img1 = read_image(img_path1)
img2 = read_image(img_path2)
with torch.no_grad():
    res = model({'image0': img1, 'image1': img2})
```