import cv2
import torch

# Function to check if a detection is within an ROI
def is_within_roi(box, roi):
    box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return (roi[0][0] <= box_center[0] <= roi[1][0]) and (roi[0][1] <= box_center[1] <= roi[1][1])

# Function to calculate density (to be fully implemented)
def calculate_density(two_wheelers, four_wheelers):
    # Placeholder logic for calculating density
    density = two_wheelers + four_wheelers  # Replace with real calculation if needed
    return density

# Function to manage traffic signals (to be fully implemented)
def manage_traffic_signals(densities):
    # Placeholder logic for managing traffic signals based on lane densities
    max_density_lane = max(densities, key=densities.get)
    print(f"Lane with max density: {max_density_lane}")
    # In a real application, you would interface with traffic control systems here

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load your video
video_path = '4_traffic.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Define the display window's width and height
display_width = 1000
display_height = 600

# Calculate the width of each lane assuming the view is perpendicular and lanes are equally wide
lane_width = display_width // 4

# Define the ROIs for each lane based on the display dimensions
lane_rois = {
    'lane1': ((0, 0), (lane_width, display_height)),
    'lane2': ((lane_width, 0), (lane_width * 2, display_height)),
    'lane3': ((lane_width * 2, 0), (lane_width * 3, display_height)),
    'lane4': ((lane_width * 3, 0), (display_width, display_height))
}

# Process video frames
while True:
    # Reset vehicle counts for each lane at the start of processing each frame
    lane_vehicle_counts = {lane: {'two_wheelers': 0, 'four_wheelers': 0} for lane in lane_rois.keys()}
    
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for display
    display_frame = cv2.resize(frame, (display_width, display_height))

    # Perform inference on the original frame
    results = model(frame)

    # Results show all classes by default, you can filter for 'car', 'truck', 'bus', etc.
    results_data = results.pandas().xyxy[0]  # Results as a Pandas DataFrame

    # Iterate over detections and classify vehicles
    for index, row in results_data.iterrows():
        box = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        # Classify as 'two_wheelers' or 'four_wheelers' based on the 'name' field (this is an assumption, adjust as needed)
        vehicle_type = 'two_wheelers' if row['name'] in ['motorcycle','person', 'bicycle'] else 'four_wheelers'

        for lane, roi in lane_rois.items():
            if is_within_roi(box, roi):
                lane_vehicle_counts[lane][vehicle_type] += 1
                break  # Assuming a vehicle can only be in one lane at a time

        # Draw bounding boxes and labels on the display frame
        x1, y1, x2, y2 = int(row['xmin'] * display_width / frame.shape[1]), \
                         int(row['ymin'] * display_height / frame.shape[0]), \
                         int(row['xmax'] * display_width / frame.shape[1]), \
                         int(row['ymax'] * display_height / frame.shape[0])
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{row['name']} {vehicle_type}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Draw white lines for lanes
    for i in range(1, 4):  # Draw lines between lanes
        cv2.line(display_frame, (i * lane_width, 0), (i * lane_width, display_height), (255, 255, 255), 2)

    # Calculate density and manage traffic signals
    densities = {lane: calculate_density(counts['two_wheelers'], counts['four_wheelers']) 
                 for lane, counts in lane_vehicle_counts.items()}
    manage_traffic_signals(densities)

    # Display the current vehicle counts on the frame
    for lane, counts in lane_vehicle_counts.items():
        cv2.putText(display_frame, f"{lane} TW: {counts['two_wheelers']} FW: {counts['four_wheelers']}", 
                    (10, 30 + 30*list(lane_rois.keys()).index(lane)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # Display the resized frame
    cv2.imshow('Vehicle Detection', display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
