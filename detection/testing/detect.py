import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('ear_model_5_subjects.pt')
# model = YOLO('ear_model_2_subjects.pt')

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):

    fps = vid.get(cv2.CAP_PROP_FPS)
    print('fps:', fps)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Capture the video frame by frame
    ret, frame = vid.read()

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # # Display the resulting frame
    # cv2.imshow('frame', frame)

    # the 'q' button is set as the quitting button you may use any desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
