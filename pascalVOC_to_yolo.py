import os
import argparse
import xml.etree.ElementTree as ET

# Class name to class ID mapping
class_name_to_id = {
    "person": 0,
    "hard-hat": 1,
    "gloves": 2,
    "mask": 3,
    "glasses": 4,
    "boots": 5,
    "vest": 6,
    "ppe-suit": 7,
    "ear-protector": 8,
    "safety-harness": 9
}

def convert_voc_to_yolo(voc_dir, yolo_dir):
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)
        
    for filename in os.listdir(voc_dir):
        if filename.endswith('.xml'):
            tree = ET.parse(os.path.join(voc_dir, filename))
            root = tree.getroot()
            
            image_file = root.find('filename').text
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            
            yolo_annotation_file = os.path.join(yolo_dir, os.path.splitext(image_file)[0] + '.txt')
            with open(yolo_annotation_file, 'w') as yolo_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in class_name_to_id:
                        print(f"Warning: '{class_name}' not found in class mapping.")
                        continue
                    
                    class_id = class_name_to_id[class_name]
                    xmin = int(obj.find('bndbox').find('xmin').text)
                    ymin = int(obj.find('bndbox').find('ymin').text)
                    xmax = int(obj.find('bndbox').find('xmax').text)
                    ymax = int(obj.find('bndbox').find('ymax').text)
                    
                    # Normalizing coordinates
                    x_center = (xmin + xmax) / (2.0 * width)
                    y_center = (ymin + ymax) / (2.0 * height)
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height

                    # if class_id == 0: # for person detection preprocessing
                    # if class_id in (1, 2, 3, 4, 5, 6, 7, 8, 9): # for ppe detection preprocessing
                    yolo_file.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC to YOLOv8 format with class ID mapping")
    parser.add_argument('voc_dir', type=str, help="Path to PascalVOC annotations directory")
    parser.add_argument('yolo_dir', type=str, help="Path to output YOLOv8 annotations directory")
    
    args = parser.parse_args()
    convert_voc_to_yolo(args.voc_dir, args.yolo_dir)
