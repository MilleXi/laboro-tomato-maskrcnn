import os
import json
import shutil
from glob import glob
from pathlib import Path

def convert_and_organize(src_dir, coco_root):
    """
    将自定义格式的数据集转换为COCO格式
    
    Args:
        src_dir: 源数据集目录
        coco_root: 输出的COCO格式数据集目录
    """
    directories = ["annotations", 'train', 'val']

    # 创建COCO数据集目录结构
    for dir in directories:
        os.makedirs(os.path.join(coco_root, dir), exist_ok=True)

    # 初始化COCO格式数据
    def init_coco_format():
        return {
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "b_fully_ripened",
                    "supercategory": "tomato"
                },{
                    "id": 2,
                    "name": "b_green",
                    "supercategory": "tomato"
                },{
                    "id": 3,
                    "name": "b_half_ripened",
                    "supercategory": "tomato"
                },{
                    "id": 4,
                    "name": "l_fully_ripened",
                    "supercategory": "tomato"
                },{
                    "id": 5,
                    "name": "l_green",
                    "supercategory": "tomato"
                },{
                    "id": 6,
                    "name": "l_half_ripened",
                    "supercategory": "tomato"
                },
            ]
        }

    train_coco = init_coco_format()
    val_coco = init_coco_format()
    
    ann_id = 1
    
    # 处理训练集和验证集
    splits = {
        'Train': (directories[1], train_coco),
        'Test': (directories[2], val_coco)
    }

    # 定义类别映射
    class_to_id = {
        "b_fully_ripened": 1,
        "b_green": 2,
        "b_half_ripened": 3,
        "l_fully_ripened": 4,
        "l_green": 5,
        "l_half_ripened": 6
    }
    
    for src_split, (target_split, coco_data) in splits.items():
        img_dir = os.path.join(src_dir, src_split, 'img')
        ann_dir = os.path.join(src_dir, src_split, 'ann')
        target_img_dir = os.path.join(coco_root, target_split)
        
        # 确保目标图片目录存在
        os.makedirs(target_img_dir, exist_ok=True)
        
        img_files = sorted(glob(os.path.join(img_dir, '*.jpg')))
        
        for img_id, src_img_path in enumerate(img_files, 1):
            # 生成12位数字的文件名
            new_img_name = f"{img_id:012d}.jpg"
            dst_img_path = os.path.join(target_img_dir, new_img_name)
            
            # 复制并重命名图片
            shutil.copy2(src_img_path, dst_img_path)
            
            # 读取原始标注
            ann_path = os.path.join(ann_dir, f"{os.path.basename(src_img_path)}.json")
            if not os.path.exists(ann_path):
                print(f"Warning: Annotation file not found for {src_img_path}")
                continue
                
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            
            # 添加图片信息
            image_info = {
                "id": img_id,
                "file_name": new_img_name,
                "width": ann_data["size"]["width"],
                "height": ann_data["size"]["height"]
            }
            coco_data["images"].append(image_info)
            
            # 处理标注信息
            for obj in ann_data["objects"]:
                # 获取类别ID
                category_id = class_to_id.get(obj["classTitle"])
                if category_id is None:
                    print(f"Warning: Unknown category {obj['classTitle']}")
                    continue

                # 处理分割点
                points = obj["points"]["exterior"]
                segmentation = []
                for point in points:
                    segmentation.extend(point)
                
                # 计算边界框
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                bbox = [
                    min(x_coords),          # x
                    min(y_coords),          # y
                    max(x_coords) - min(x_coords),  # width
                    max(y_coords) - min(y_coords)   # height
                ]
                
                # 创建标注
                annotation = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": bbox[2] * bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0
                }
                
                coco_data["annotations"].append(annotation)
                ann_id += 1
    
    # 保存标注文件
    ann_train_path = os.path.join(coco_root, 'annotations', 'instances_train.json')
    ann_val_path = os.path.join(coco_root, 'annotations', 'instances_val.json')
    
    with open(ann_train_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    print(f"Saved training annotations to {ann_train_path}")
    
    with open(ann_val_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"Saved validation annotations to {ann_val_path}")

if __name__ == "__main__":
    src_dir = "laborotomato-DatasetNinja"  # 源数据集目录
    coco_root = "laboro-tomato"  # 输出的COCO格式数据集目录
    convert_and_organize(src_dir, coco_root)
    print("转换完成！")