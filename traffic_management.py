import cv2
import torch

# Function to check if a detected object's center is within a Region Of Interest (ROI)
def is_within_roi(box, roi):
    # Calculate the center point of the bounding box
    box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    # Check if the center point is within the specified ROI
    return (roi[0][0] <= box_center[0] <= roi[1][0]) and (roi[0][1] <= box_center[1] <= roi[1][1])

# Function to calculate the density of vehicles in a lane
def calculate_density(two_wheelers, four_wheelers):
    density = two_wheelers + four_wheelers  
    return density

# Setup display dimensions
display_width = 1000
display_height = 600

# Calculate lane width based on the display width
lane_width = display_width // 4

# Define Regions Of Interest (ROIs) for each lane
lane_rois = {
    'lane1': ((0, 0), (lane_width, display_height)),
    'lane2': ((lane_width, 0), (lane_width * 2, display_height)),
    'lane3': ((lane_width * 2, 0), (lane_width * 3, display_height)),
    'lane4': ((lane_width * 3, 0), (display_width, display_height))
}

# Initialize traffic lights state for each lane
traffic_lights = {lane: 'off' for lane in lane_rois.keys()}

# Update traffic lights based on the density of vehicles in each lane
def update_traffic_lights(densities, traffic_lights):
    # Find the lane with the maximum density of vehicles
    max_density_lane = max(densities, key=densities.get)
    for lane in traffic_lights.keys():
        # Only the lane with the highest density will show traffic lights (red, green, yellow)
        if lane == max_density_lane:
            traffic_lights[lane] = ['red', 'green', 'yellow']
        else:
            # Other lanes will not show any traffic lights
            traffic_lights[lane] = 'off'

# Draw traffic lights on the display frame
def draw_traffic_lights(display_frame, traffic_lights, lane_width, display_height):
    # Define positions for the traffic lights in each lane
    light_positions = {
        'lane1': (lane_width//2, 50),
        'lane2': (lane_width + lane_width//2, 50),
        'lane3': (lane_width*2 + lane_width//2, 50),
        'lane4': (lane_width*3 + lane_width//2, 50)
    }
    for lane, states in traffic_lights.items():
        if states == 'off':
            continue  # Skip drawing for lanes with no lights
        else:
            # Draw red, green, and yellow lights for the lane with the highest density
            for idx, color in enumerate([(0, 0, 255), (0, 255, 0), (0, 255, 255)]):  # Red, Green, Yellow
                cv2.circle(display_frame, (light_positions[lane][0], light_positions[lane][1] + idx*30), 10, color, -1)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Setup video capture
video_path = 'sourav.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Main loop to process the video frames
while True:
    # Initialize vehicle counts for each lane
    lane_vehicle_counts = {lane: {'two_wheelers': 0, 'four_wheelers': 0} for lane in lane_rois.keys()}
    
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is captured

    # Resize the frame to fit the display dimensions
    display_frame = cv2.resize(frame, (display_width, display_height))

    # Detect objects in the frame using YOLOv5
    results = model(frame)

    # Process the detection results
    results_data = results.pandas().xyxy[0]

    for index, row in results_data.iterrows():
        # Get the bounding box of the detected object
        box = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        # Classify the vehicle type
        vehicle_type = 'two_wheelers' if row['name'] in ['motorcycle', 'bicycle'] else 'four_wheelers'

        # Check which lane the detected vehicle belongs to and update the count
        for lane, roi in lane_rois.items():
            if is_within_roi(box, roi):
                lane_vehicle_counts[lane][vehicle_type] += 1
                break

        # Draw bounding boxes and labels on the display frame
        x1, y1, x2, y2 = int(row['xmin'] * display_width / frame.shape[1]), \
                         int(row['ymin'] * display_height / frame.shape[0]), \
                         int(row['xmax'] * display_width / frame.shape[1]), \
                         int(row['ymax'] * display_height / frame.shape[0])
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{row['name']} {vehicle_type}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Draw lane separation lines
    for i in range(1, 4):
        cv2.line(display_frame, (i * lane_width, 0), (i * lane_width, display_height), (255, 255, 255), 2)

    # Calculate the density of vehicles in each lane
    densities = {lane: calculate_density(counts['two_wheelers'], counts['four_wheelers']) for lane, counts in lane_vehicle_counts.items()}
    # Update traffic light states based on densities
    update_traffic_lights(densities, traffic_lights)
    # Draw traffic lights on the display frame
    draw_traffic_lights(display_frame, traffic_lights, lane_width, display_height)

    # Display vehicle counts for each lane
    for lane, counts in lane_vehicle_counts.items():
        cv2.putText(display_frame, f"{lane} TW: {counts['two_wheelers']} FW: {counts['four_wheelers']}", 
                    (10, 30 + 30*list(lane_rois.keys()).index(lane)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # Show the frame with detections and annotations
    cv2.imshow('traffic management', display_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
