from ultralytics.data.converter import yolo_bbox2segment

input_path = "/media/mor582/Storage/Nick/surveys/train/yolo/val_best/"

yolo_bbox2segment(  
    im_dir=input_path,
    save_dir=None,  # saved to "labels-segment" in images directory
    sam_model="sam_b.pt",
)