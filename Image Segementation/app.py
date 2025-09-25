import pixellib
from pixellib.instance import instance_segmentation
import numpy as np
if not hasattr(np, "bool"):
    np.bool = np.bool_
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5") 
segment_image.segmentImage("cycle.jpg", show_bboxes = True, output_image_name = "output.jpg")



