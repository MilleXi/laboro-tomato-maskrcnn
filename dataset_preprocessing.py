import os
import json
import shutil
from glob import glob
from pathlib import Path

def convert_and_organize(src_dir, coco_root):
    directories = ["annotations",'train','val']

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
    
    for src_split, (target_split, coco_data) in splits.items():
        img_dir = os.path.join(src_dir, src_split, 'img')
        ann_dir = os.path.join(src_dir, src_split, 'ann')
        target_img_dir = os.path.join(coco_root, target_split)
        
        img_files = sorted(glob(os.path.join(img_dir, '*.jpg')))
        
        for img_id, src_img_path in enumerate(img_files, 1):
            # 生成6位数字的文件名
            new_img_name = f"{img_id:012d}.jpg"
            dst_img_path = os.path.join(target_img_dir, new_img_name)
            
            # 复制并重命名图片
            shutil.copy2(src_img_path, dst_img_path)
            
            # 读取原始标注
            ann_path = os.path.join(ann_dir, f"{os.path.basename(src_img_path)}.json")
            if not os.path.exists(ann_path):
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
                
                annotation = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": bbox[2] * bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0
                }
                
                coco_data["annotations"].append(annotation)
                ann_id += 1
    
    # 保存标注文件
    with open(os.path.join(coco_root, 'annotations', 'instances_train.json'), 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(os.path.join(coco_root, 'annotations', 'instances_val.json'), 'w') as f:
        json.dump(val_coco, f, indent=2)

if __name__ == "__main__":
    src_dir = "dataset/laboro-tomato-DatasetNinja"  # LaboroTomato数据集目录
    coco_root = "dataset/laboro-tomato-DatasetNinja-Coco"  # 输出的COCO格式数据集目录
    convert_and_organize(src_dir, coco_root)
    print("转换完成！")