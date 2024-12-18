- Why did I choose the scenario?
    
    - I chose Human Activity Recognition with weapon detection because it addresses a critical need for proactive public safety solutions in a world facing increasing violence. 
    Traditional surveillance systems are reactive and prone to human error, while AI-powered solutions can instantly detect and alert authorities to threats, reducing response times and saving lives.


- What problem you are solving?

    - The problem I am solving is the lack of real-time, proactive security systems capable of accurately identifying violent actions combined with potential threats like weapons in public spaces. Current surveillance systems rely heavily on human monitoring, which is prone to errors, delays, and inefficiencies, especially in high-stress or high-traffic environments.

    This solution addresses the following key challenges:

    Delayed Threat Response: Traditional systems only react after an incident occurs, missing opportunities for prevention.
    Manual Monitoring Fatigue: Security personnel often miss critical events due to the overwhelming volume of surveillance data.
    False Positives/Negatives: Existing systems may detect isolated events (e.g., weapon detection) but fail to assess contextual behaviors like aggression.
    By combining weapon detection with activity recognition, this AI-powered system provides context-aware alerts, ensuring timely and accurate responses to potential threats, ultimately enhancing public safety and reducing the likelihood of violent incidents escalating.

- Which industry is it applicable to?
    - Applicable Industries: 
        This solution is highly versatile and can be applied to various industries where public safety and proactive security are priorities:

        Public Safety and Law Enforcement:

        Monitoring public spaces like parks, streets, and government buildings for potential threats.
        Airports and Transportation Hubs:

        Enhancing security checks and surveillance to detect suspicious behavior and weapons in real-time.
        Retail and Hospitality:

        Protecting customers and staff in malls, hotels, and restaurants by identifying violent actions early.
        Event Management and Stadiums:

        Ensuring crowd safety at concerts, sports events, and other large gatherings.
        Education:

        Securing schools and universities to prevent violent incidents or weapon-related threats.
        Smart Cities and Infrastructure:

        Integrating into city-wide surveillance networks as part of smart city initiatives for enhanced urban safety.
        Military and Defense:

        Real-time threat detection for sensitive locations like military bases or border areas.


# Design Architecture

        - Simplified Architecture for Human Activity Recognition System
        Input and Preprocessing
        Video Input: Captures live video feed from cameras.
        Stream Splitting: Video feed is split into two parallel processing streams:
        Weapon Detection
        Action Recognition
        Weapon Detection
        YOLOv8 Object Detection:
        Processes video frames to identify weapons (e.g., guns, knives).
        Outputs: Bounding boxes and labels for detected weapons.
        Action Recognition
        Mediapipe Pose Estimation:
        Extracts skeletal movements from video to track body poses.
        Enhanced LSTM Model with Attention:
        Input: Pose sequences from Mediapipe.
        Action classification into categories such as "Punch" or "Neutral."
        Outputs: Action labels (e.g., Aggressive Action or Neutral).
        Decision Module
        Integration of Outputs: Combines results from Weapon Detection and Action Recognition:
        If a weapon and an aggressive action (e.g., punch) are detected simultaneously:
        Triggers an alert.
        Alert System
        Alert Generation:
        Captures a screenshot of the relevant frame with timestamp.
        Sends an email with the screenshot and incident details to designated authorities or supervisors.
# AI models
YOLO v8 for object detection (Weapons)
Mediapipe for keypoint detection

# Training and Tuning
- What was the data input?
1. The data for Human action recognition was created using , mediapipe. 
    - We capture the data points of the body parts , while doing a specific action i.e., Puching / Neutral
    - Then the landmarks have 5 features for each point i.e., x-axis , y-axis , z-axis (Depth) , Visibilitu , Presence
    - Then these are saved in the dataframe ,  33 landmarks each with 5 features i.e., 165 features in total.
    - Then we assign a lable to each action in our case , 0 is Punch and 1 is Neutral.
    - This dataframe is trained using a LSTM - pseudo attention layer , for more insights on the architecture diagram you can go to this link : https://lucid.app/lucidchart/eb605497-5ca4-4692-b960-50bc4f258006/edit?viewport_loc=-8065%2C-6880%2C15416%2C7587%2C0_0&invitationId=inv_f8b63bdd-83f4-41ef-a21d-45b39780a433

2. The data for Object detection of Knife and Mask :
    - First we collected the images of me holding a knife at different angles and wearing mask.
    - We capture those images and save it in a directory names as "datasets"
    - Now its time for some annotation , I used LableImg for annotating the images in YOLO format and save the labels in label folder and images in images folder.
    - For training the yolov8l.pt model , I used the google colab pro and trained the images for 100 epochs.

- What data is the output?
    - The output of the data is a probability score of Punch + weapon or Punch or Neutral.
    Output Scenarios : 
        - If only NEUTRAL is detected nothing happens.
        - If only PUNCH is detected there is warning sign in YELLOW.
        - If PUNCH with WEAPON(mask / knife) is detected then it's a danger in RED and send a automatic screenshot to a specific mail id.

- Model Optimization
1. The first draft of the LSTM model , was a simple LSTM MODEL with dropout layers. 
Here is the optimized version. 
1. Architecture Optimization
The EnhancedLSTMModel improves upon the original LSTM model by reducing complexity and introducing critical features like Batch Normalization and Attention Mechanism.
This results in better learning of temporal patterns, reduced overfitting, and improved focus on relevant time steps in the sequence.
2. Dropout Layers for Regularization
Both models use dropout layers (0.2–0.3) to prevent overfitting by randomly deactivating neurons during training, ensuring the model generalizes well on unseen data.
3. Attention Mechanism
The EnhancedLSTMModel incorporates an attention mechanism, allowing the model to focus on important features within the sequence data. This improves interpretability and enhances performance, especially for complex temporal dependencies.
4. Hyperparameter Optimization
Learning Rate: A smaller learning rate (0.0003) was used for more stable and gradual convergence.
Batch Size: Increased to 32 to balance computational efficiency and generalization.
5. Advanced Optimizer
AdamW Optimizer: Used for better weight regularization, reducing overfitting, and achieving smoother convergence compared to standard Adam.
Weight decay of 0.01 was introduced to control overfitting further.
6. Learning Rate Scheduler
A ReduceLROnPlateau scheduler adjusts the learning rate dynamically based on the model’s validation loss, allowing the model to fine-tune parameters effectively when improvement slows down.
7. Reduced Model Complexity
The EnhancedLSTMModel uses only two LSTM layers instead of four, reducing computational requirements while retaining performance.
8. Criterion for Optimization
The CrossEntropyLoss function was chosen as it is well-suited for classification problems like this one. It ensures the model predicts accurate class probabilities.
9. Increased Hidden Size
Hidden state size increased to 128, enhancing the model's ability to capture complex temporal relationships without unnecessary computational overhead.
10. Iterative Training and Validation
Models were trained over 100 epochs, with regular validation and adjustments guided by the scheduler to prevent overfitting and ensure robustness.
11. Efficiency Gains
The model balances accuracy and latency, ensuring it is both performant and deployable in real-time environments, critical for applications like Human Activity Recognition.

FUTURE OPTIMIZATION : 
Hybrid Models: Combine LSTM with CNNs for spatial-temporal feature extraction, particularly useful for video or image-based temporal data.


# Market Research
USA CRIME STATS : 
<a href='https://www.macrotrends.net/global-metrics/countries/USA/united-states/crime-rate-statistics'>U.S. Crime Rate & Statistics 1990-2024</a>
CANADA CRIME STATS
<a href='https://www.macrotrends.net/global-metrics/countries/CAN/canada/crime-rate-statistics'>Canada Crime Rate & Statistics 1990-2024</a>

Sectors like smart surveillance and autonomous systems require models with rapid response capabilities.

Adoption of AI in Public Safety:

Increasing interest in weapon detection and behavioral analytics in public spaces.
Governments and private sectors investing in predictive AI for crime prevention.


ideo Analytics Market Growth:

Estimated at $7 billion in 2021, with a projected CAGR of over 20% by 2030.
Use cases like retail analytics, traffic monitoring, and security systems are major drivers.

The video analytics market was estimated at approximately $7 billion in 2021 and is projected to grow at a compound annual growth rate (CAGR) exceeding 20% by 2030. This growth is driven by increasing adoption in sectors like retail analytics, traffic monitoring, and security systems. These applications leverage advanced technologies, including AI and machine learning, to provide actionable insights from video data.
SOURCE : https://www.reportsanddata.com/report-detail/video-analytics-market

# Scalability

1. Data Scalability
Handling Diverse Input Features: Incorporate datasets from various sources and environments (e.g., diverse lighting conditions, angles, and object types) to enhance robustness.
Expanding Dataset Volume: Ensure the model can process larger datasets efficiently by leveraging advanced preprocessing pipelines (e.g., distributed data pipelines like Apache Kafka).
Dynamic Data Inclusion: Design the architecture to adapt to new, real-time data streams without full retraining, using transfer learning or online learning techniques.
2. Model Scalability
Architecture Modularity: The modular design of your Enhanced LSTM allows for the replacement or addition of layers, enabling it to adapt to larger and more complex datasets.
Parallel Processing: Incorporate distributed training strategies, such as data or model parallelism, to handle large datasets or high-resolution video inputs.
Model Compression: Use techniques like pruning, quantization, or distillation to reduce computational overhead for edge or mobile deployments.
3. Infrastructure Scalability
Cloud and Edge Integration: Deploy the model on cloud platforms (e.g., AWS SageMaker, Google Vertex AI) for high scalability, or optimize for edge devices to handle localized tasks.
Hardware Acceleration: Utilize GPUs, TPUs, or specialized hardware (e.g., NVIDIA Jetson) to scale inference for real-time applications.
4. Operational Scalability
Real-Time Processing: Optimize pipelines for lower latency by integrating frameworks like TensorRT or ONNX for faster inference.
API and Microservices: Package the model into containerized microservices (e.g., Docker) to support scaling across multiple nodes.
5. Use Case Expansion
Multi-Task Learning: Adapt the model for related tasks like object detection, behavior prediction, or activity recognition.
Cross-Domain Applications: Apply the model to new domains, such as healthcare (patient monitoring) or retail (customer analytics).
6. Future Enhancements
Improved Attention Mechanisms: Explore advanced attention modules (e.g., transformers) to enhance performance and flexibility.
Federated Learning: Enable training on decentralized datasets while preserving privacy, expanding adoption in sensitive domains like healthcare and finance.

# Cost / Employment Analysis

----------------------------------------------------------	                    |
       Category         |    Estimated Costs
----------------------------------------------------------
Cloud Deployment	    |    $200–$500/month (CPU)
                        |    $500–$2000/month (GPU)
Edge Devices	        |    $99–$600/device (one-time)
Development Team	    |    $600k–$900k/year (core team)
Additional Staff	    |    $200k–$300k/year
Advertising/Marketing	|    $50k–$100k/year
Training/Optimization	|    $150–$1500/session (on cloud)
Operational Maintenance	|    $150k–$200k/year
Development Time	    |    12–18 months
----------------------------------------------------------