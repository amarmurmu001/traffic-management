# Traffic Management System

This project implements a traffic management system using computer vision techniques. The system can detects vehicles in a video stream or integrated camera , calculates their density in each lane, and controls traffic lights based on the density of vehicles in each lane.

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- YOLOv5

## Installation

1. Clone this repository: https://github.com/amarmurmu001/traffic-management.git

2. Install the required dependencies: OpenCV and PyTorch

3. Download the YOLOv5 model weights from [here](https://github.com/ultralytics/yolov5/releases), and place them in the project directory.

## Usage

1. Place the video file you want to process in the project directory.

2. Modify the `video_path` variable in the script to point to your video file.

3. Run the script: python traffic_management.py

4. Press 'q' to exit the application.

## Features

- Detects vehicles in a video stream using YOLOv5.
- Calculates the density of vehicles in each lane.
- Controls traffic lights based on vehicle density.
- Displays vehicle counts and traffic light status in real-time.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the [MIT License](LICENSE).
