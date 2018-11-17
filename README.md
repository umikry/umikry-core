# umikry-core

umikry-core ships different detection, transformation and data generation methods to detect, generate and replace faces

## Setup

```zsh
git clone https://github.com/umikry/umikry-core.git
cd umikry-core
pip(3) install -r requirements.txt
```

## Usage

```python
from core import umikry
import cv2
import matplotlib.pyplot as plt

print('photo (public domain) downloaded from https://www.flickr.com/photos/presidioofmonterey/15482602317')
image = cv2.imread('/home/whoami/photo.jpg')
b,g,r = cv2.split(image)

plt.figure(figsize=(16, 16))
plt.subplot(121)
plt.imshow(cv2.merge([r,g,b]))
plt.axis('off')

image = umikry(image, detection='caffe', transformation='blur')
b,g,r = cv2.split(image)

plt.subplot(122)
plt.imshow(cv2.merge([r,g,b]))
plt.axis('off')
plt.show()
```