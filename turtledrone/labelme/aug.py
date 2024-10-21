from ultralytics.data.augment import CopyPaste
dataset = '/media/mor582/Storage/Nick/surveys/train/yolo/'  # Your image dataset
copypaste = CopyPaste(dataset, p=0.5)
augmented_labels = copypaste(original_labels)