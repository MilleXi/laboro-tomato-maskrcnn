import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont
import os
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

def get_tomato_categories():
    """
    返回番茄数据集的类别映射
    """
    categories = {
        1: 'b_fully_ripened', #correct
        2: 'b_green', # correct
        3: 'b_half_ripened',
        4: 'l_fully_ripened',
        5: 'l_green',
        6: 'l_half_ripened'
    }
    return categories

def load_model(checkpoint_path, num_classes):
    """
    加载训练好的模型
    """
    model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')  # 修改这里
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_color_list():
    """
    为每个类别生成固定的颜色（使用RGB字符串格式）
    """
    colors = {
        'b_green': "green",
        'l_green': "lightgreen",
        'l_fully_ripened': "red",
        'b_half_ripened': "orange",
        'l_half_ripened': "pink",
        'b_fully_ripened': "darkred"
    }
    return colors

def visualize_prediction(image_path, model, device, score_threshold=0.5, mask_threshold=0.5):
    """
    对单张图片进行推理和可视化，同时显示原图和检测结果
    """
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    # 转换为张量
    image_tensor = F.to_tensor(image)
    # 将图片转换为uint8类型
    image_uint8 = (image_tensor * 255).byte()
    
    # 将模型设置为评估模式
    model.eval()
    model.to(device)
    
    # 获取类别映射和颜色映射
    categories = get_tomato_categories()
    color_map = get_color_list()
    
    # 进行推理
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])
        prediction = prediction[0]
    
    # 获取高于阈值的预测
    keep = prediction['scores'] > score_threshold
    boxes = prediction['boxes'][keep].cpu()
    labels = prediction['labels'][keep].cpu()
    scores = prediction['scores'][keep].cpu()
    masks = prediction['masks'][keep].cpu()
    
    # 将掩码转换为布尔类型
    masks = masks.squeeze(1) > mask_threshold
    
    # 准备标签文本和颜色
    label_texts = []
    box_colors = []
    mask_colors = []
    
    for label, score in zip(labels, scores):
        category_name = categories[label.item()]
        label_texts.append(f"{category_name} {score:.2f}")
        color = color_map[category_name]
        box_colors.append(color)
        # 为mask准备RGB颜色值
        if color == "green":
            mask_colors.append([0, 255, 0])
        elif color == "lightgreen":
            mask_colors.append([144, 238, 144])
        elif color == "red":
            mask_colors.append([255, 0, 0])
        elif color == "orange":
            mask_colors.append([255, 165, 0])
        elif color == "pink":
            mask_colors.append([255, 192, 203])
        else:  # darkred
            mask_colors.append([139, 0, 0])
    
    # 绘制边界框
    result_with_boxes = draw_bounding_boxes(
        image_uint8,
        boxes=boxes,
        labels=label_texts,
        colors=box_colors,
        width=2
    )
    
    # 绘制掩码
    if len(masks) > 0:
        result_with_masks = draw_segmentation_masks(
            result_with_boxes,
            masks=masks,
            alpha=0.5,
            colors=mask_colors
        )
    else:
        result_with_masks = result_with_boxes
    
    # 创建一个包含两个子图的图像
    plt.figure(figsize=(20, 8))
    
    # 显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Original Image', fontsize=15)
    
    # 显示检测结果
    plt.subplot(1, 2, 2)
    plt.imshow(result_with_masks.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Detection Result', fontsize=15)
    
    # 添加图例
    legend_elements = []
    for cat_name, color in color_map.items():
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                        markerfacecolor=color,
                                        markersize=10, label=cat_name))
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存结果
    plt.savefig('prediction_comparison.png', bbox_inches='tight', dpi=300)
    # 显示结果
    plt.show()
    
    return prediction

def print_prediction_results(prediction, score_threshold=0.5):
    """
    打印预测结果
    """
    categories = get_tomato_categories()
    print("\n预测结果:")
    for i in range(len(prediction['labels'])):
        if prediction['scores'][i] > score_threshold:
            label = prediction['labels'][i].item()
            score = prediction['scores'][i].item()
            box = prediction['boxes'][i].cpu().numpy()
            class_name = categories[label]
            print(f"类别: {class_name}, 置信度: {score:.3f}")
            print(f"边界框: {box}\n")

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 设置路径
    val_dir = "laboro-tomato/train"
    num_classes = 7  # 6个类别 + 1个背景类
    
    try:
        # 加载训练好的模型
        model = load_model('checkpoints/best_model.pth', num_classes)
        model.to(device)
        
        # 获取验证集中的所有图片
        image_files = os.listdir(val_dir)
        
        # 如果没有图片，抛出错误
        if not image_files:
            raise FileNotFoundError("验证集目录为空")
            
        # 获取第一张图片
        test_image = os.path.join(val_dir, image_files[150])
        print(f"处理图片: {test_image}")
        
        # 进行推理和可视化
        prediction = visualize_prediction(
            test_image,
            model,
            device,
            score_threshold=0.5,
            mask_threshold=0.5
        )
        
        # 打印预测结果
        print_prediction_results(prediction)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise