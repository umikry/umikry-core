# umikry-core

umikry-core ships the detector, generator, data_loader and feature_extractor methods to
detect, generate and replace faces by different algorithms

## Setup

```zsh
git clone https://github.com/umikry/umikry-core.git
cd umikry-core
pip(3) install -r requirements.txt
```

## Usage

```python
from detection import UmikryFaceDetector

# use a simple and fast haar classifier 
# (output: bounding box)

umikryFaceDetector = UmikryFaceDetector(method='haar')
image = cv2.imread('/path/to/my/image.jpg')
faces = umikryFaceDetector.detect(image)

# or even a more sophisticated convnet method 
# (output: semantic segmentation)

config = configparser.ConfigParser()
config.read('umikry.ini')
if 'DATA' in config:
	data_dir = config['DATA']['Location']
else:
	data_dir = input('Where is your train data located:')
	config['DATA'] = {'Location': data_dir}
	with open('umikry.ini', 'w') as configfile:
		config.write(configfile)

umikryFaceDetector = UmikryFaceDetector(method='cnn')

generateTrainTestSet(data_dir, datasets=['OpenImages'])

train_sequence = ImageSequence(data_dir, batch_size=16, patch_size=(384, 384), is_training=True)
test_sequence = ImageSequence(data_dir, batch_size=16, patch_size=(384, 384))

checkpoint_dir = 'models'
model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'community_facedetector_weights.{epoch:02d}-{val_loss:.2f}.h5'), 
                                 save_weights_only=True, save_best_only=True, monitor='val_acc')
early_stopping = EarlyStopping(patience=5)

umikryFaceDetector.train(train_sequence, epochs=20,
                       test_generator=test_sequence,
                       callbacks=[model_checkpoint, early_stopping])
umikryFaceDetector.model.save_weights(os.path.join(checkpoint_dir, 'community_facedetector_weights.h5'))
image = cv2.imread('/path/to/my/image.jpg')
face_segmentation = umikryFaceDetector.detect(image)
```