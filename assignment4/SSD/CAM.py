import warnings

from ssd.data.transforms import gpu_transforms
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import click
from ssd import utils
from ssd.data.tdt4265 import TDT4265Dataset
from ssd.modeling import AnchorEncoder
import tops
from tops import checkpointer
from tops import logger
from tops.config import instantiate

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import AblationCAM, EigenCAM
# from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
# from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image



coco_names = TDT4265Dataset.class_names
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def predict(input_tensor, model, detection_threshold, anchor_encoder, imshape):
    bbox_delta, confs = model(input_tensor)

    # Decode. bbox_delta -> ltrb-boxes. Applies softmax on confs.
    pred_bboxes, confs = anchor_encoder.decode_output(bbox_delta, confs)

    # We use CPU from here
    pred_bboxes = pred_bboxes.cpu().detach().numpy()[0]
    confs = confs.cpu().detach().numpy()[0]

    # Remove anchors where confidence is largest for background class
    not_background_mask = confs.argmax(axis=1) != 0
    pred_bboxes = pred_bboxes[not_background_mask]
    confs = confs[not_background_mask]

    # Scale pictres to pixel values
    (im_h, im_w) = imshape
    pred_bboxes[:,[0,2]] *= im_w
    pred_bboxes[:,[1,3]] *= im_h

    pred_labels = confs.argmax(axis=1)
    pred_classes = np.array([coco_names[i] for i in pred_labels])
    pred_scores = confs.max(axis=1)

    # Filter removing all boxes with pred_score lower than threshold
    threshold_mask = pred_scores >= detection_threshold
    classes = pred_classes[threshold_mask]
    labels = pred_labels[threshold_mask]
    indices = np.arange(len(threshold_mask))[threshold_mask]
    boxes = pred_bboxes[threshold_mask].astype(np.int32)

    return boxes, classes, labels,indices

def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 1
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1,
                    lineType=cv2.LINE_AA)
    return image


def fasterrcnn_reshape_transform(x):
    target_size = x['feat4'].size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations


def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        images = []
        for x1, y1, x2, y2 in boxes:
            img = renormalized_cam * 0
            img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
            images.append(img)
        
        renormalized_cam = np.max(np.float32(images), axis = 0)
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized



@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--image", "-i", default='data/tdt4265/dataset/images/train/trip007_glos_Video00005_52.png', type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--out", "-o", default='cam_image.png', type=click.Path(exists=False, dir_okay=False, path_type=str))
@click.option("--renormalize", '-r', default=False, is_flag=True, help="Normalize the CAM to be in the range [0, 1] inside every bounding boxes, and zero outside of the bounding boxes.")
def main(config_path, image, out, renormalize):
    cfg = utils.load_config(config_path)
    tops.init(cfg.output_dir)

    model = tops.to_cuda(instantiate(cfg.model))

    checkpointer.register_models(
    dict(model=model))
    if checkpointer.has_checkpoint():
        train_state = checkpointer.load_registered_models(load_best=False)
        logger.log(f"Resuming train from: epoch: {logger.epoch()}, global step: {logger.global_step()}")

    image = np.array(Image.open(image))
    image_float_np = np.float32(image) / 255

    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    input_tensor = transform(image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)

    gpu_transform = torchvision.transforms.Normalize(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
    gpu_transform(input_tensor)

    anchors = instantiate(cfg.anchors)
    anchor_encoder = AnchorEncoder(anchors)
    imshape = instantiate(cfg.train).imshape

    # Run the model and display the detections
    boxes, classes, labels, indices = predict(input_tensor, model, 0.95, anchor_encoder, imshape)

    target_layers = [model.feature_extractor.fpn]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    print(targets)
    cam = EigenCAM(model,
                target_layers, 
                use_cuda=torch.cuda.is_available(),
                reshape_transform=fasterrcnn_reshape_transform)
    cam.uses_gradients = False

    grayscale_cam = cam(input_tensor, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    if renormalize:
        cam_image = renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam)
    else:
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    out_image = Image.fromarray(image_with_bounding_boxes)
    out_image.save(out)

if __name__ == '__main__':
    main()
