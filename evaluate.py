import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from train import (TomatoDataset, get_transform, get_model_instance_segmentation, 
                  collate_fn, mask_to_rle)

def evaluate_model_comprehensive(model, data_loader, device, dataset_val, save_dir):
    """全面评估模型性能"""
    model.eval()
    results = []
    all_predictions = []
    all_targets = []
    
    # 设置较大的初始列表大小以避免频繁的内存重新分配
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 使用with torch.cuda.amp.autocast()来启用自动混合精度
    from torch.cuda.amp import autocast
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("开始评估...")
    with torch.no_grad(), autocast():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # 批量处理前清理不必要的缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 优化数据传输到GPU的过程
            images = [image.to(device, non_blocking=True) for image in images]
            outputs = model(images)
            
            # 立即将不需要的张量移到CPU
            outputs = [{k: v.cpu() if isinstance(v, torch.Tensor) else v 
                       for k, v in output.items()} 
                      for output in outputs]
            
            # 收集每个图像的预测和真实标签
            for target, output in zip(targets, outputs):
                image_id = target['image_id'].item()
                
                # 获取高置信度的预测
                keep_mask = output['scores'] > 0.5
                boxes = output['boxes'][keep_mask].cpu()
                scores = output['scores'][keep_mask].cpu()
                labels = output['labels'][keep_mask].cpu()
                masks = output['masks'][keep_mask].cpu()
                
                # 记录预测结果
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
                    results.append(result)
                
                # 收集用于混淆矩阵的标签
                pred_labels = labels.numpy()
                true_labels = target['labels'].cpu().numpy()
                
                all_predictions.extend(pred_labels)
                all_targets.extend(true_labels)
    
    # 1. 计算COCO指标
    results_file = save_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        import json
        json.dump(results, f)
    
    coco_gt = dataset_val.coco
    coco_dt = coco_gt.loadRes(str(results_file))
    
    # 评估边界框
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    
    # 评估分割
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
    
    # 2. 计算并绘制混淆矩阵
    classes = list(range(1, 7))  # 假设有6个类别（不包括背景）
    cm = confusion_matrix(all_targets, all_predictions, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()
    
    # 3. 计算每个类别的精确度和召回率
    class_metrics = {}
    for i in range(len(classes)):
        true_pos = cm[i, i]
        false_pos = cm[:, i].sum() - true_pos
        false_neg = cm[i, :].sum() - true_pos
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[f'Class_{i+1}'] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    
    # 保存详细评估结果
    metrics_df = pd.DataFrame(class_metrics).T
    metrics_df.to_csv(save_dir / 'class_metrics.csv')
    
    # 返回评估结果摘要
    return {
        'bbox_map': coco_eval_bbox.stats[0],
        'segm_map': coco_eval_segm.stats[0],
        'class_metrics': class_metrics
    }

def main():
    # 设置设备和内存优化选项
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        # 设置cudnn基准模式和其他优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    print(f"使用设备: {device}")
    
    # 如果是GPU，打印显存信息
    if torch.cuda.is_available():
        print(f"GPU显存使用情况:")
        print(f"分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # 数据集路径
    data_path = "laboro-tomato"
    val_data_dir = os.path.join(data_path, "val")
    val_coco = os.path.join(data_path, "annotations", "instances_val.json")
    
    # 创建验证数据集和数据加载器
    dataset_val = TomatoDataset(val_data_dir, val_coco, get_transform())
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 加载模型
    num_classes = 6 + 1  # 背景 + 6个类别
    model = get_model_instance_segmentation(num_classes, pretrained=False)
    
    # 加载保存的模型权重
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"加载了在epoch {checkpoint['epoch']}保存的最佳模型")
    print(f"保存时的mAP: {checkpoint['map']:.4f}")
    
    # 创建评估结果保存目录
    eval_save_dir = Path('evaluation_results')
    
    # 进行全面评估
    evaluation_results = evaluate_model_comprehensive(
        model, data_loader_val, device, dataset_val, eval_save_dir
    )
    
    # 打印评估结果
    print("\n详细评估结果:")
    print(f"Bbox mAP: {evaluation_results['bbox_map']:.4f}")
    print(f"Segmentation mAP: {evaluation_results['segm_map']:.4f}")
    print("\n每个类别的指标:")
    for class_name, metrics in evaluation_results['class_metrics'].items():
        print(f"\n{class_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        raise