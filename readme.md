# Purpose

This project is meant to be a submission to SBU AI Community competition

## Model Architecture

The model is built on MobileNetV2 as a means for transfer learning. Furthermore, data augmentation is used in a anti-curriculum manner in combination to fine tuning in phases. The fine-tuning phase is computationally expensive, and can cause overfitting, so it is made to be the final phase. All of the phases before it attempt to maximize the final performance before plateauing by first training on more generalized, augmented data before reaching the true dataset. The fine-tuning phase benefits from this, since it is able to train the model past its first plateau.

### Phase 1: High Augment (~15 epochs)

In the first phase, the model using stronger augmentation factors. The data is augmented with flip, zoom, translation and rotation. By training on more difficult data first, it helps the grow in a more general manner

### Phase 2: Low Augment (~15 epochs)

In the second phase, the model uses weaker augmentation factors. There are also less augmentations to the data. The data is augmented with flip, zooms and rotations. By training it now on less difficult data, it should start to train more specifically on the training data, but still stay more general

### Phase 3: No Augment (~20 epochs)

In the third phase, the model does not use augmentation factors. Now that there are no augmentations, the model will train on the data most similar to what it will be used to predict on. This training phase will go on until it plateaus

### Phase 4: Fine Tune (~20 epochs)

In the fourth and final phase, the layers of MobileNetV2 unfreeze, allowing for the model to fine tune to the training data.

# Results
![Control-V(1)](https://github.com/user-attachments/assets/5d390c4f-c7a4-4aab-94df-cbc9f5910a08)

Here, we see how the model is able to learn to failure through the use of decreasingly augmented data before finetuning to reach a final accuracy of approximately 90%

