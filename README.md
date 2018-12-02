# umikry-core

umikry-core ships different detection, transformation and data generation methods to detect, generate and replace faces.
It also allows you to recognize familiar persons to let them unaffected (see [umikry examples](https://github.com/umikry/example_applications) for more information)

![image](https://user-images.githubusercontent.com/1525818/48979609-1c0f0900-f0be-11e8-9f1b-f7e0c493ea66.png)

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

fig, axis = plt.subplots(2, 2)
fig.set_figheight(10)
fig.set_figwidth(15)
axis[0, 0].imshow(cv2.merge([r,g,b]))
axis[0, 0].axis('off')
axis[0, 0].set_title('ORIGINAL')

methods = [([0, 1], 'GAN'), ([1, 0], 'AUTOENCODER'), ([1, 1], 'BLUR')]

for position, method in methods:    
    b,g,r = cv2.split(umikry(image, detection='CAFFE', transformation=method))

    axis[position[0], position[1]].imshow(cv2.merge([r,g,b]))
    axis[position[0], position[1]].axis('off')
    axis[position[0], position[1]].set_title(method)

plt.show()
```
