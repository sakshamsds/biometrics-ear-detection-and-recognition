import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
detector_model = YOLO('yolo_detector.pt')
recognition_model = YOLO('yolo_classifier.pt')

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
    fps = vid.get(cv2.CAP_PROP_FPS)
    print('fps:', fps)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Capture the video frame by frame
    _, frame = vid.read()

    # Run YOLOv8 inference on the frame
    detected_roi = detector_model(frame)

    # Visualize the results on the frame
    annotated_frame = detected_roi[0].plot()

    recognition_result = recognition_model(frame)  # predict on an image
    recognized_subject = recognition_result[0].names[recognition_result[0].probs.argmax(
    ).item()]

    print(recognized_subject)

    cv2.putText(img=annotated_frame, 
                text=recognized_subject,
                org=(250, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=2)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # the 'q' button is set as the quitting button you may use any desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
