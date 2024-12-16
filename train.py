import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_to_tensor(image):
    """
    将PIL图像转换为张量
    """
    return F.to_tensor(image)

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
        
        # 预先分配numpy数组
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        masks = np.zeros((num_objs, *coco.annToMask(coco_annotation[0]).shape), dtype=np.uint8)
        labels = np.zeros(num_objs, dtype=np.int64)
        
        # 填充数组
        for idx, ann in enumerate(coco_annotation):
            bbox = ann['bbox']
            boxes[idx] = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            masks[idx] = coco.annToMask(ann)
            labels[idx] = ann['category_id']

        # 转换为张量
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

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    
    # 创建进度条
    pbar = tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=120)
    
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 更新进度条显示的损失值
        pbar.set_postfix({
            'loss': f'{losses.item():.3f}',
            'avg_loss': f'{total_loss / (pbar.n + 1):.3f}'
        })

    return total_loss / len(data_loader)

def get_transform():
    """
    定义数据转换
    """
    return convert_to_tensor

def get_model_instance_segmentation(num_classes):
    """
    加载预训练的模型并修改输出层
    """
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")

    # 设置数据集路径
    data_path = "laboro-tomato"
    train_data_dir = os.path.join(data_path, "train")
    train_coco = os.path.join(data_path, "annotations", "instances_train.json")
    val_data_dir = os.path.join(data_path, "val")
    val_coco = os.path.join(data_path, "annotations", "instances_val.json")

    # 创建数据集
    dataset = TomatoDataset(train_data_dir, train_coco, get_transform())
    dataset_val = TomatoDataset(val_data_dir, val_coco, get_transform())

    # 创建数据加载器
    data_loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=1, 
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 获取类别数（包括背景）
    num_classes = 6 + 1
    print(f"类别数量（包含背景）: {num_classes}")
    print(f"训练集大小: {len(dataset)}")
    print(f"验证集大小: {len(dataset_val)}")

    # 获取模型
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # 定义学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练轮数
    num_epochs = 10

    # 创建保存模型的目录
    os.makedirs('checkpoints', exist_ok=True)

    # 训练循环
    for epoch in range(num_epochs):
        # 训练一个epoch并获取平均损失
        epoch_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 保存模型
        checkpoint_path = os.path.join('checkpoints', f'maskrcnn_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        
        # 打印当前学习率
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')