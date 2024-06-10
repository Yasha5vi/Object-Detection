from ultralytics import YOLO
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt

model=YOLO("yolov8n.pt")

def videoStream():
    cap = cv2.VideoCapture(0)
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    while True:
        success, img = cap.read()
        if not success:
            break

        # Doing detections using YOLOv8 frame by frame
        results = model(img, stream=True)

        # Process the detected objects and draw bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf:.2f}'

                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness=1)
                c2_x = x1 + text_size[0] + 10
                c2_y = y1 - text_size[1] - 5
                cv2.rectangle(img, (x1, y1), (c2_x, c2_y), [0, 255, 0], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 0.7, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break

        # Convert the frame buffer to bytes
        frame_bytes = buffer.tobytes()

        # Yield the frame as byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# def videoStream():
#     cap=cv2.VideoCapture(0)

#     frame_width=int(cap.get(3))
#     frame_height = int(cap.get(4))

#     #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

#     classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#                 "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#                 "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#                 "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#                 "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#                 "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#                 "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#                 "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#                 "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#                 "teddy bear", "hair drier", "toothbrush"
#                 ]
#     while True:
#         success, img = cap.read()
#         # Doing detections using YOLOv8 frame by frame
#         #stream = True will use the generator and it is more efficient than normal
#         results=model(img,stream=True)
#         #Once we have the results we can check for individual bounding boxes and see how well it performs
#         # Once we have have the results we will loop through them and we will have the bouning boxes for each of the result
#         # we will loop through each of the bouning box
#         for r in results:
#             boxes=r.boxes
#             for box in boxes:

#                 x1,y1,x2,y2=box.xyxy[0]
#                 #print(x1, y1, x2, y2)
#                 x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
#                 print(x1,y1,x2,y2)
#                 # creating a rectangle around each of the detected object
#                 cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),3)

#                 #confidence value in terms of tensor
#                 print(box.conf[0])
#                 #converting in int
#                 conf=math.ceil((box.conf[0]*100))/100
#                 # getting the class if
#                 cls=int(box.cls[0])
#                 # searching in our class
#                 class_name=classNames[cls]
#                 # label at boxes
#                 label=f'{class_name}{conf}'
#                 # label size
#                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                
#                 # #print(t_size)
#                 # c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                 # Calculate the size of the text
#                 text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness=1)

#                 # Calculate the coordinates of the bottom-right corner of the bounding box
#                 c2_x = x1 + text_size[0] + 10   # here 10 is padding_x
#                 c2_y = y1 - text_size[1] - 5    # here 5 is padding_y
#                 cv2.rectangle(img, (x1,y1), (c2_x,c2_y), [0,255,0], -1, cv2.LINE_AA)  # filled
#                 #add label to the rectangle 
#                 cv2.putText(img, label, (x1,y1-2),0, 0.7,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

#         #out.write(img)
#         cv2.imshow("Image", img)
#         if cv2.waitKey(1) & 0xFF==ord('1'):
#             break
#     #out.release()

# def imageCapture():
#     im1 = "./Images/parking.jpg"
#     results = model.predict(source=im1,show = True)  # save plotted images
#     cv2.waitKey(0)

def imageCapture(img_path):
    results = model.predict(source=img_path, show=False)  # Perform detection on the image
    img = results[0].orig_img
    
    # Draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = model.names[cls]
            label = f'{class_name} {conf:.2f}'
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness=1)

            c2_x = x1 + text_size[0] + 10
            c2_y = y1 - text_size[1] - 5
            cv2.rectangle(img, (x1, y1), (c2_x, c2_y), [0, 255, 0], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 0.7, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    cv2.imwrite("static/result.jpg", img)  # Save the processed image
