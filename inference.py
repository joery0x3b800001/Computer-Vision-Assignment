import argparse
import os
import cv2
from ultralytics import YOLO

def load_models(person_model_path, ppe_model_path):
    person_model = YOLO(person_model_path, task='detect')
    ppe_model = YOLO(ppe_model_path, task='detect')
    return person_model, ppe_model

def draw_boxes(image, boxes, color, labels):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box[:4]  # Get coordinates
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main(input_dir, output_dir, person_model_path, ppe_model_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    person_model, ppe_model = load_models(person_model_path, ppe_model_path)

    # Define the labels for the models
    person_label = 'person'
    ppe_labels = [
        'hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest',
        'ppe-suit', 'ear-protector', 'safety-harness'
    ]

    for image_file in os.listdir(input_dir):
        if not image_file.endswith('.jpg'):
            continue

        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        # Perform person detection
        person_results = person_model(image)
        person_boxes = person_results[0].boxes.xyxy.cpu().numpy()  # Get boxes in [x1, y1, x2, y2, conf, cls] format
        person_labels = [person_label] * len(person_boxes)

        for person_box in person_boxes:
            x1, y1, x2, y2 = person_box[:4]
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

            # Perform PPE detection on the cropped person image
            ppe_results = ppe_model(cropped_image)
            ppe_boxes = ppe_results[0].boxes.xyxy.cpu().numpy()  # Get boxes in [x1, y1, x2, y2, conf, cls] format
            ppe_cls = ppe_results[0].boxes.cls.cpu().numpy()  # Get class IDs

            ppe_labels_detected = [ppe_labels[int(cls)] for cls in ppe_cls]

            # Adjust PPE bounding boxes to the original image coordinates
            for ppe_box in ppe_boxes:
                ppe_box[0] += x1
                ppe_box[1] += y1
                ppe_box[2] += x1
                ppe_box[3] += y1

            # Draw PPE bounding boxes
            image = draw_boxes(image, ppe_boxes, (0, 255, 0), ppe_labels_detected)

        # Draw person bounding boxes
        image = draw_boxes(image, person_boxes, (255, 0, 0), person_labels)

        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for YOLOv8 models")
    parser.add_argument("input_dir", type=str, help="Input directory path containing images")
    parser.add_argument("output_dir", type=str, help="Output directory path to save results")
    parser.add_argument("person_model", type=str, help="Path to the trained person detection model")
    parser.add_argument("ppe_model", type=str, help="Path to the trained PPE detection model")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.person_model, args.ppe_model)
