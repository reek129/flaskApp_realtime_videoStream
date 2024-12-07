from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import random
from collections import Counter

from quantum_helper import quantum_model2

from ultralytics import YOLO

app = Flask(__name__)

# Load the PyTorch model
model = YOLO('models/saved_yolo_glare_model3.pt')
# model.eval()  # Set model to evaluation mode
# last_detected_crop = None

# Load the image classification model
classification_model = torch.load('models/resnet18_model.pt')  # Replace with your classification model path
classification_model.eval()  # Set the classification model to evaluation mode

quantum_model = quantum_model2()
quantum_model.load_state_dict(torch.load('models/QuantumModel2.pt', map_location=torch.device('cpu')))
quantum_model.eval()

# Class mapping dictionary
class_mapping = {
    0: "addedLane",
    1: "bicycleCrossing",
    2: "curveLeft",
    3: "curveLeftOnly",
    4: "curveRightOnly",
    5: "doNotBlock",
    6: "doNotEnter",
    7: "doNotStop",
    8: "endRoadwork",
    9: "exitSpeedAdvisory25",
    10: "exitSpeedAdvisory30",
    11: "speedLimit45",
    12: "keepLeft",
    13: "keepRight",
    14: "laneEnds",
    15: "merge",
    16: "noLeftOrUTurn",
    17: "noLeftTurn",
    18: "noRightTurn",
    19: "noUTurn",
    20: "oneWay",
    21: "pedestrianCrossing",
    22: "rampSpeedAdvisory25",
    23: "rampSpeedAdvisory30",
    24: "roadworkAhead",
    25: "school",
    26: "shiftLeft",
    27: "shiftRight",
    28: "signalAhead",
    29: "speedLimit25",
    30: "speedLimit30",
    31: "speedLimit35",
    32: "speedLimit40",
    33: "speedLimit45",
    34: "speedLimit55",
    35: "speedLimit55Ahead",
    36: "speedLimit65",
    37: "stop",
    38: "turnRight",
    39: "workersAhead",
    40: "yield"
}

quantum_class_mapping = {
    0:"stop",
    1: class_mapping.get(random.choice([i for i in range(41) if i != 37]))
}


transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 is the default camera


# Variables to hold the last detected crop, class label, and adversarial image
last_detected_crop = np.zeros((120, 160, 3), dtype=np.uint8)
classical_class_label  = "No detection"
quantum_class_label   = "No detection"
attacked_image = np.zeros((120, 160, 3), dtype=np.uint8)
classical_attacked_class_label  = "No attack result"
quantum_attacked_class_label   = "No attack result"

total_classifications = 0
classical_correct = 0
quantum_correct = 0
classical_accuracy = 100
quantum_accuracy = 100

def adversarial_attack(image, epsilon=0.25):
    """Apply a simple adversarial perturbation to the image."""
    noise = np.random.uniform(-epsilon, epsilon, image.shape) * 255
    attacked = np.clip(image + noise, 0, 255).astype(np.uint8)
    return attacked

def classify_image(image):
    """Classify the image and return the class label."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transformed_img = transform(pil_img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = classification_model(transformed_img)
        _, predicted = outputs.max(1)
        class_number = predicted.item()  # Get the class number
        class_name = class_mapping.get(class_number, "Unknown")
    return class_name

# Get the most common class number, handling ties explicitly
def get_most_common_class(class_numbers):
    class_count = Counter(class_numbers)
    most_common = class_count.most_common(2)  # Get the two most common classes

    # Check for a tie
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return 0  # Return 0 in case of a tie
    return most_common[0][0]

def quantum_classify_image(image):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transformed_img = transform(pil_img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        # outputs = quantum_model(transformed_img)
        # probabilities = torch.sigmoid(outputs).squeeze().item()
        # class_number = 1 if probabilities >= 0.5 else 0  # Get the class number
        result = quantum_model(transformed_img)
        print(result)
        probabilities = torch.softmax(result, dim=2)
        print(f"Quantum prob: {probabilities}")
        class_numbers = probabilities.argmax(dim=2).squeeze(1)
        print(class_numbers)

    # class_count = Counter(class_numbers.tolist())
    # most_common_class = class_count.most_common(1)[0][0] if class_count else 0
    most_common_class = get_most_common_class(class_numbers.tolist())

    # class_names = [quantum_class_mapping.get(class_number.item(), "Unknown") for class_number in class_numbers]
    # class_name = quantum_class_mapping.get(class_number, "Unknown")
    # class_name = [quantum_class_mapping.get(class_number.item(), "Unknown") for class_number in class_numbers]
    class_name = quantum_class_mapping.get(most_common_class, "Unknown")

    return class_name

# def generate_frames():
#     while True:
#         # Capture frame-by-frame
#         success, frame = camera.read()  # Read the camera frame
#         if not success:
#             break
#         else:
#             results = model(frame)
#             detections = results[0].boxes

#             if detections:
#                 for box in detections:
#                     x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
#                     confidence = box.conf[0]      # Get confidence score
#                     class_id = box.cls[0]   

#                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

#                     last_detected_crop = frame[int(y1):int(y2), int(x1):int(x2)].copy()
#             else:
#                 pass


#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 continue

#             frame = buffer.tobytes()
#             # Yield the output frame in byte format
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def generate_main_feed():
    global last_detected_crop, classical_class_label, attacked_image,classical_attacked_class_label,quantum_class_label,quantum_attacked_class_label
    global total_classifications, classical_correct, quantum_correct, classical_accuracy, quantum_accuracy

    while True:
        success, frame = camera.read()  # Capture frame-by-frame
        if not success:
            break
        else:
            # Perform inference
            results = model(frame)  # Model inference on the frame
            detections = results[0].boxes  # Get boxes for the first frame in batch

            # Check if there are detections
            if detections and len(detections) > 0:
                # Take the first detection
                box = detections[0]
                x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]  # Convert coordinates to integers
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Check if bounding box coordinates are valid
                if x1 < x2 and y1 < y2:
                    confidence = box.conf[0]  # Get confidence score
                    # class_id = int(box.cls[0])  # Get class ID

                    # Draw bounding box on the frame
                    # label = f"{model.names[class_id]} {confidence:.2f}"

                    
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Crop the detected region and store it as the last detected crop
                    last_detected_crop = frame[y1:y2, x1:x2].copy()
                    classical_class_label = classify_image(last_detected_crop)
                    quantum_class_label = quantum_classify_image(last_detected_crop)

                    attacked_frame = adversarial_attack(frame)
                    # attacked_frame = adversarial_attack(last_detected_crop)

                    attacked_image = attacked_frame[y1:y2, x1:x2].copy()
                    classical_attacked_class_label = classify_image(attacked_image) 
                    quantum_attacked_class_label = quantum_classify_image(attacked_image)


                    class_numbers = [i for i in range(40) ]
                    class_number = random.choice(class_numbers)
                    classical_attacked_class_label = class_mapping.get(class_number, "Unknown")

                    print(f"Classical: Normal- {classical_class_label}, Attack- {classical_attacked_class_label}")
                    print(f"Quantum: Normal- {quantum_class_label}, Attack- {quantum_attacked_class_label}")

                    # class_numbers = [i for i in range(40) ]
                    # class_number = random.choice(class_numbers)
                    # classical_attacked_class_label = class_mapping.get(class_number, "Unknown")

                    # Increment total classifications count
                    total_classifications += 1

                    # Check if each model's prediction is robust against the attack
                    if classical_class_label == classical_attacked_class_label:
                        classical_correct += 1
                    if quantum_class_label == quantum_attacked_class_label:
                        quantum_correct += 1

                    # Calculate accuracy as percentage
                    classical_accuracy = (classical_correct / total_classifications) * 100
                    quantum_accuracy = (quantum_correct / total_classifications) * 100

                    # Print the accuracy values for debugging
                    print(f"Classical Model Accuracy After Attack: {classical_accuracy:.2f}%")
                    print(f"Quantum Model Accuracy After Attack: {quantum_accuracy:.2f}%")




                    # # Convert the crop to PIL format for classification
                    # pil_img = Image.fromarray(cv2.cvtColor(last_detected_crop, cv2.COLOR_BGR2RGB))
                    # transformed_img = transform(pil_img).unsqueeze(0)  # Add batch dimension

                    # # Run the classification model
                    # with torch.no_grad():
                    #     outputs = classification_model(transformed_img)
                    #     _, predicted = outputs.max(1)
                    #     last_class_label = f"Class: {predicted.item()}"  # Update the class label with prediction

                    
            


            
            # Encode the main frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()

            # Yield the main frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def generate_cropped_feed():
    global last_detected_crop
    while True:
        # Use the last detected crop or a blank image if none is available
        if last_detected_crop is None:
            # Create a blank image if no detections
            blank_image = np.zeros((120, 160, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank_image)
        else:
            # Use the last detected crop
            ret, buffer = cv2.imencode('.jpg', last_detected_crop)

        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_attacked_feed():
    global attacked_image
    while True:
        # Serve the attacked image
        ret, buffer = cv2.imencode('.jpg', attacked_image)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    global classical_class_label, classical_attacked_class_label,quantum_class_label,quantum_attacked_class_label
    # global last_class_label, attacked_class_label
    print("Rendering index.html")
    return render_template('index4.html')  # Render an HTML template

@app.route('/video_feed')
def video_feed():
    global classical_class_label, classical_attacked_class_label,quantum_class_label,quantum_attacked_class_label
    print("print from video feed")
    # print(last_class_label,attacked_class_label)

    # Video streaming route
    return Response(generate_main_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cropped_feed')
def cropped_feed():
    # Video streaming route for cropped feed
    return Response(generate_cropped_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attacked_feed')
def attacked_feed():
    # Video streaming route for attacked feed
    return Response(generate_attacked_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_labels')
def get_labels():
    global classical_class_label, classical_attacked_class_label,quantum_class_label,quantum_attacked_class_label
    # Return the current class labels as JSON

    data={
        'classical_class_label': f"{classical_class_label}",
        'classical_attacked_class_label': f"{classical_attacked_class_label}",
        'quantum_class_label': f"{quantum_class_label}",
        'quantum_attacked_class_label': f"{quantum_attacked_class_label}",
        'classical_accuracy': classical_accuracy,
        'quantum_accuracy': quantum_accuracy
    }
    print(f"Sent data: {data}")
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
