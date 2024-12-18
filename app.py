from flask import Flask, render_template, Response, jsonify , send_from_directory , request
import cv2
import threading
from ultralytics import YOLO
import os
from datetime import datetime

# Import your detection logic and model classes here
from utils import *
from models import LSTMModel , EnhancedLSTMModel
from config import *
import shutil
from send_mail import EmailSender

# Initialize YOLOv8
yolo_model = YOLO('knife_mask_model.pt')  # or your specific model path

label_lock = threading.Lock()
label = "Neutral"

# Instantiate the model and load weights
# model = LSTMModel(input_size=INPUT_SHAPE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
model = EnhancedLSTMModel(input_size=INPUT_SHAPE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('punch_kick_attention_lstm_model2.pth'))
model.eval() 


# sending EMAIL Creds
# Directory where screenshots are stored
SCREENSHOT_DIR = "E:\Violence\static\screenshots"  # Replace with your screenshot directory path

# Email credentials
EMAIL_ADDRESS = "sarthak.samantaray50@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "ztve nlpc hiac bspq"  # Replace with your email password

# Initialize EmailSender
email_sender = EmailSender(EMAIL_ADDRESS, EMAIL_PASSWORD)
app = Flask(__name__)


## Directory to save screenshots
SCREENSHOT_DIR = "static/screenshots"
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

screenshot_data = []

# Global variables for camera access
cap = None
label = "neutral"
weapon = None
def detect(model, lm_list):
    global label
    with torch.no_grad():  # Disable gradient computation for inference
        lm_list = np.array(lm_list, dtype=np.float32)
        
        # Ensure the input shape is [1, 20, 98] (batch_size, seq_length, input_size)
        if len(lm_list) == 20:  # Ensure we have 20 frames of data
            lm_list = torch.tensor(lm_list).unsqueeze(0).to(DEVICE)  # Add batch dimension
            print("SHAPE OF INPUT:", lm_list.shape)  # Should print torch.Size([1, 20, 98])
            
            result = model(lm_list)
            probabilities = torch.softmax(result, dim=1).cpu().numpy()[0]  # Move to CPU before converting to NumPy
            print(f"Model prediction probabilities: {probabilities}")
            
            if probabilities[0] > 0.5:
                label = "Punch"
            elif probabilities[1] > 0.5:
                label = "Neutral"
            # elif probabilities[2] > 0.5:
            #     label = "Neutral"
            # else:
            #     label = "Neutral"
    return str(label)

def draw_yolo_predictions(frame, results):
    """Draw YOLO predictions on frame"""
    weapons_detected = set()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get confidence
            conf = float(box.conf[0])
            # Get class name
            cls = int(box.cls[0])
            class_name = yolo_model.names[cls]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{class_name} {conf:.2f}'
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            weapons_detected.add(class_name)
    return frame , weapons_detected

def gen_frames():
    global cap, label
    cap = cv2.VideoCapture(0)
    lm_list = []
    warm_up_frames = 30
    i = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run YOLO detection
        yolo_results = yolo_model(frame, conf=0.5)  # Adjust confidence threshold as needed
        frame ,weapons_detected= draw_yolo_predictions(frame, yolo_results)
        
        
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        i += 1

        if i > warm_up_frames and results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []

            frame = draw_landmark_on_image(mp_draw, results, frame)
            frame , status = draw_bounding_box_and_label(frame, results, label,weapons_detected)

            # Capture screenshot if status is "Danger"
            if status == "Danger":
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                screenshot_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                screenshot_data.append({"path": screenshot_path, "timestamp": timestamp})

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_current_label')
def get_current_label():
    with label_lock:
        current_label = label
    return jsonify({'label': current_label})

@app.route('/stop_detection')
def stop_detection():
    global cap
    if cap is not None:
        cap.release()
    return jsonify({'status': 'success'})



@app.route('/save_screenshot', methods=['POST'])
def save_screenshot():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Remove old screenshots
    for filename in os.listdir(SCREENSHOT_DIR):
        os.remove(os.path.join(SCREENSHOT_DIR, filename))

    # Save the new screenshot
    filepath = os.path.join(SCREENSHOT_DIR, 'latest_screenshot.png')
    file.save(filepath)

    return jsonify({'message': 'Screenshot saved', 'path': filepath})

@app.route('/get_screenshot', methods=['GET'])
def get_screenshot():
    files = os.listdir(SCREENSHOT_DIR)
    if files:
        latest_file = files[0]  # Only one file will exist
        return jsonify({
            'path': f'{SCREENSHOT_DIR}/{latest_file}',
            'timestamp': os.path.getmtime(os.path.join(SCREENSHOT_DIR, latest_file))
        })
    else:
        return jsonify({'error': 'No screenshot available'}), 404
# Add this function to delete all contents of the screenshots directory
def clear_screenshots_directory():
    if os.path.exists(SCREENSHOT_DIR):
        shutil.rmtree(SCREENSHOT_DIR)  # Remove the directory and its contents
        os.makedirs(SCREENSHOT_DIR)   # Recreate the directory
    else:
        os.makedirs(SCREENSHOT_DIR)   # Create the directory if it doesn't exist





@app.route('/send_screenshot', methods=['POST'])
def send_screenshot():
    # Get the latest screenshot
    files = sorted(os.listdir(SCREENSHOT_DIR), reverse=True)
    if not files:
        return jsonify({'error': 'No screenshot available'}), 404

    latest_file = os.path.join(SCREENSHOT_DIR, files[-1])

    # Email parameters
    recipient_email = "sarthak.samantaray50@gmail.com"  # Replace with recipient email
    subject = "DANGER ALERT!"
    body = f"Attached is the latest screenshot captured from the app. TIMESTAMP {datetime.now().date(),datetime.now().time()}"

    try:
        # Send the email
        email_sender.send_email(
            recipient_email=recipient_email,
            subject=subject,
            body=body,
            attachment_path=latest_file,
        )
        return jsonify({'message': 'Email sent successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_screenshots', methods=['POST'])
def clear_screenshots():
    try:
        clear_screenshots_directory()
        return jsonify({'message': 'Screenshots cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


