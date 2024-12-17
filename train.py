import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path

class TomatoDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        num_objs = len(coco_annotation)
        
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        masks = np.zeros((num_objs, *coco.annToMask(coco_annotation[0]).shape), dtype=np.uint8)
        labels = np.zeros(num_objs, dtype=np.int64)
        
        for idx, ann in enumerate(coco_annotation):
            bbox = ann['bbox']
            boxes[idx] = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            masks[idx] = coco.annToMask(ann)
            labels[idx] = ann['category_id']

        boxes = torch.from_numpy(boxes)
        masks = torch.from_numpy(masks)
        labels = torch.from_numpy(labels)
        
        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)

class Transform:
    """
    Transform class for data augmentation
    """
    def __call__(self, image):
        # Convert to tensor and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image

def get_transform():
    """
    Returns transform object
    """
    return Transform()

def get_model_instance_segmentation(num_classes, pretrained=True):
    """
    Enhanced model initialization with proper weights
    """
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None)
    
    # Modify the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Modify the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
    """
    Enhanced training loop with improved logging and gradient clipping
    """
    model.train()
    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_mask = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0
    
    pbar = tqdm(data_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}', ncols=120)
    
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        
        # Calculate individual losses
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_mask += loss_dict['loss_mask'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

        # Backward pass with gradient clipping
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update progress bar with detailed losses
        pbar.set_postfix({
            'loss': f'{losses.item():.3f}',
            'cls_loss': f'{loss_dict["loss_classifier"].item():.3f}',
            'box_loss': f'{loss_dict["loss_box_reg"].item():.3f}',
            'mask_loss': f'{loss_dict["loss_mask"].item():.3f}'
        })

    # Calculate average losses
    num_batches = len(data_loader)
    avg_losses = {
        'total_loss': total_loss / num_batches,
        'classifier_loss': total_loss_classifier / num_batches,
        'box_reg_loss': total_loss_box_reg / num_batches,
        'mask_loss': total_loss_mask / num_batches,
        'objectness_loss': total_loss_objectness / num_batches,
        'rpn_box_reg_loss': total_loss_rpn_box_reg / num_batches
    }
    
    return avg_losses

@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Enhanced evaluation function with proper error handling
    """
    model.eval()
    coco_results = []
    
    pbar = tqdm(data_loader, desc='Evaluating...', ncols=120)
    
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        
        outputs = model(images)
        
        for target, output in zip(targets, outputs):
            image_id = target['image_id'].item()
            
            if len(output['boxes']) > 0:
                keep_mask = output['scores'] > 0.5
                
                boxes = output['boxes'][keep_mask].cpu().numpy()
                scores = output['scores'][keep_mask].cpu().numpy()
                labels = output['labels'][keep_mask].cpu().numpy()
                masks = output['masks'][keep_mask].cpu().numpy()
                
                for box, score, label, mask in zip(boxes, scores, labels, masks):
                    result = {
                        'image_id': image_id,
                        'category_id': label.item(),
                        'bbox': [
                            float(box[0]), float(box[1]),
                            float(box[2] - box[0]), float(box[3] - box[1])
                        ],
                        'score': float(score),
                        'segmentation': mask_to_rle(mask[0])
                    }
                    coco_results.append(result)
                
                del boxes, scores, labels, masks
            else:
                # Add a dummy result for empty detections
                result = {
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': [0, 0, 1, 1],
                    'score': 0.0,
                    'segmentation': {'counts': [0], 'size': [1, 1]}
                }
                coco_results.append(result)
    
    return coco_results

def mask_to_rle(binary_mask):
    """
    Convert binary mask to RLE format
    """
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()
    
    mask_shape = binary_mask.shape
    mask_flat = binary_mask.ravel(order='F')
    diff = np.diff(np.concatenate([[0], mask_flat, [0]]))
    runs_starts = np.where(diff != 0)[0]
    runs_lengths = np.diff(runs_starts)
    
    return {
        'counts': runs_lengths.tolist(),
        'size': list(mask_shape)
    }

def save_results(results, epoch, save_dir):
    """
    Save evaluation results to file
    """
    results_file = os.path.join(save_dir, f'results_epoch_{epoch}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)
    return results_file

def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    This needs to be at module level (not inside main()) for multiprocessing to work.
    """
    return tuple(zip(*batch))

def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Dataset paths
    data_path = "laboro-tomato2"
    train_data_dir = os.path.join(data_path, "train")
    train_coco = os.path.join(data_path, "annotations", "instances_train.json")
    val_data_dir = os.path.join(data_path, "val")
    val_coco = os.path.join(data_path, "annotations", "instances_val.json")

    # Create datasets
    dataset = TomatoDataset(train_data_dir, train_coco, get_transform())
    dataset_val = TomatoDataset(val_data_dir, val_coco, get_transform())

    # Create data loaders with more conservative settings
    data_loader = DataLoader(
        dataset, 
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,  # Using the module-level collate_fn
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,  # Using the module-level collate_fn
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Model setup
    num_classes = 6 + 1  # background + 6 classes
    model = get_model_instance_segmentation(num_classes, pretrained=True)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=0.0001,
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=20,
        steps_per_epoch=len(data_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )

    # Create directories for saving
    checkpoints_dir = Path('checkpoints')
    results_dir = Path('results')
    checkpoints_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Training parameters
    num_epochs = 50
    best_map = 0.0
    patience = 5
    patience_counter = 0

    # Print training info
    print(f"Number of training images: {len(dataset)}")
    print(f"Number of validation images: {len(dataset_val)}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {data_loader.batch_size}")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        losses = train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs)
        
        # Print detailed losses
        print("\nTraining Losses:")
        for loss_name, loss_value in losses.items():
            print(f"{loss_name}: {loss_value:.4f}")

        # Validation
        print("\nStarting validation...")
        try:
            results = evaluate(model, data_loader_val, device)
            results_file = save_results(results, epoch + 1, results_dir)
            
            coco_gt = dataset_val.coco
            coco_dt = coco_gt.loadRes(results_file)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            map_score = coco_eval.stats[0]
            
            # Early stopping logic
            if map_score > best_map:
                best_map = map_score
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'map': map_score,
                    'losses': losses
                }, checkpoints_dir / 'best_model.pth')
                print(f"New best model saved with mAP: {map_score:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                    break
        
            # Save current epoch model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': map_score,
                'losses': losses
            }, checkpoints_dir / f'model_epoch_{epoch + 1}.pth')
            
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            map_score = 0.0
        
        # Update learning rate
        lr_scheduler.step()
        
        # Print epoch summary
        print(f'\nEpoch {epoch + 1} Results:')
        print(f'Average Loss: {losses["total_loss"]:.4f}')
        print(f'Validation mAP: {map_score:.4f}')
        print(f'Best mAP: {best_map:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')

    print("Training completed!")
    
    # Load best model
    best_model_path = checkpoints_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        print(f"\nBest model was achieved at epoch {checkpoint['epoch']}")
        print(f"Best mAP: {checkpoint['map']:.4f}")
    
    return model, best_map

if __name__ == '__main__':
    try:
        model, best_map = main()
        print(f"\nTraining successfully completed with best mAP: {best_map:.4f}")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise
    