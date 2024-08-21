# Object Detection with YOLOv8

This repository contains scripts for object detection using YOLOv8 models. The `inference.py` script is used to detect persons and PPE items in images. The `pascalVOC_to_yolo.py` script converts PascalVOC format annotations to YOLOv8 format.

## Prerequisites

- Python 3.8 or higher
- `virtualenv` package (for creating a virtual environment)

## Setup

### Create and Activate Virtual Environment

1. **Create Virtual Environment**:

   ```sh
   python3 -m venv venv
   ```
2. **Activate Virtual Environment**:

   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```

### Install Dependencies

Make sure to activate the virtual environment before installing the dependencies.

```sh
pip install -r requirements.txt
```

## Scripts

### Inference Script

The `inference.py` script performs person and PPE detection on images.

#### Usage

```sh
python inference.py <input_dir> <output_dir> <person_model> <ppe_model>
```

- `input_dir`: Path to the directory containing input images.
- `output_dir`: Path to the directory where output images with detections will be saved.
- `person_model`: Path to the trained person detection model (e.g., `person_detection.pt`).
- `ppe_model`: Path to the trained PPE detection model (e.g., `ppe_detection.pt`).

#### Example (Paste this line)

```sh
python inference.py datasets/images_and_annotation output weights/person_detection.pt weights/ppe_detection.pt
```

### PascalVOC to YOLOv8 Format Converter

The `pascalVOC_to_yolo.py` script converts PascalVOC annotations to YOLOv8 format with specified class ID mappings.

#### Usage

```sh
python pascalVOC_to_yolo.py <voc_dir> <yolo_dir>
```

- `voc_dir`: Path to the directory containing PascalVOC XML annotation files.
- `yolo_dir`: Path to the directory where YOLOv8 annotation files will be saved.

#### Example (Paste this line)

```sh
python pascalVOC_to_yolo.py datasets/labels datasets/yolov8-format
```

## Deactivate Virtual Environment

When you're done working in the virtual environment, you can deactivate it by running:

```sh
deactivate
```

## Notes

- Make sure your `requirements.txt` includes all necessary dependencies, such as `ultralytics` and `opencv-python`.

```bash
  ultralytics
  opencv-python
```

## Links and References

[HuggingFace Ultralytics pretrained model](https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8n.pt)

[Model Training with Ultralytics YOLO](https://docs.ultralytics.com/modes/train/#__tabbed_1_2)

[How to convert PASCAL VOC to YOLO](https://stackoverflow.com/questions/64581692/how-to-convert-pascal-voc-to-yolo)

[YOLO Object Detection Explained: Evolution, Algorithm, and Applications](https://encord.com/blog/yolo-object-detection-guide/)

[Google Colab](https://colab.research.google.com/)

## Contact

For any questions or issues, please contact [@ShaunJoe](shaunjoeroy1234@gmail.com).
