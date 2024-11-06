# Purpose

This project is meant to be a submission to SBU AI Community competition

## Model Architecture

The model is built on MobileNetV2 as a means for transfer learning. After that, data augmentation is used in a anti-curriculum manner in combination to fine tuning in phases

### Phase 1: High Augment (~15 epochs)

In the first phase, the model using stronger augmentation factors. The data is augmented with flip, zoom, translation and rotation. By training on more difficult data first, it helps the grow in a more general manner

### Phase 2: Low Augment (~15 epochs)

In the second phase, the model uses weaker augmentation factors. There are also less augmentations to the data. The data is augmented with flip, zooms and rotations. By training it now on less difficult data, it should start to train more specifically on the training data, but still stay more general

### Phase 3: No Augment (~20 epochs)

In the third phase, the model does not use augmentation factors. Now that there are no augmentations, the model will train on the data most similar to what it will be used to predict on. This training phase will go on until it plateaus

### Phase 4: Fine Tune (~20 epochs)

In the fourth and final phase, the layers of MobileNetV2 unfreeze, allowing for the model to fine tune to the training data
