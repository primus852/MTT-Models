import cv2
import time
import numpy as np
from ..utils import Helper


class OpenCVStream:
    """Object Detection on OpenCV Capture"""

    def __init__(self, model_folder, video_input=0):
        """

        :param video_input: 0 is the webcam. Can be a path to a VideoFile
        """

        self.video_input = video_input
        self.model_folder = model_folder
        self.model = self._load_model()

        with open(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.MODEL_FOLDER / 'classes.txt', 'r') as f:
            self.class_names = f.read().split('\n')

        self.label_border_colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

        # Threshold to for confidence to detect an Object
        self.threshold = .4

        # Set the Video Input
        self.capture, self.output = self._grab_input()

        # Stream the Video
        self._stream_video()

    def _load_model(self):

        # Load the DNN model
        return cv2.dnn.readNet(model=str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.MODEL_FOLDER / self.model_folder / 'frozen_inference_graph.pb'),
                               config=str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.MODEL_FOLDER / self.model_folder / 'config.pbtxt'),
                               framework='TensorFlow')

    def _grab_input(self):
        cap = cv2.VideoCapture(self.video_input)

        if not cap.isOpened():
            print("Cannot open Video")
            exit()

        # get the video frames' width and height for proper saving of videos
        # @TODO: Understand why 16/9 does not save the Video properly (same with 30 FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.OPENCV_FOLDER / '{}.mp4'.format(self.model_folder)),
                              cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        return cap, out

    def _stream_video(self):
        # Object Detection in every single Frame
        while self.capture.isOpened():
            ret, frame = self.capture.read()

            if ret:

                # Extract Image
                image = frame
                image_height, image_width, _ = image.shape

                # Create Blob from Image
                blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)

                # Start Timer for FPS
                start = time.time()

                # Detect Objects
                self.model.setInput(blob)
                output = self.model.forward()

                # End Timer after Detection
                end = time.time()

                # Calculate the FPS for Current Detection
                # @TODO: Save for Metrics in Paper?
                fps = 1 / (end - start)

                # Loop through the Detections
                for detection in output[0, 0, :, :]:

                    # Extract Confidence
                    confidence = detection[2]

                    # Draw Box if Confidence is above Threshold
                    if confidence > self.threshold:
                        # get the class id
                        class_id = detection[1]

                        # map the class id to the class
                        class_name = self.class_names[int(class_id) - 1]
                        color = self.label_border_colors[int(class_id)]

                        # get the bounding box coordinates
                        box_x = detection[3] * image_width
                        box_y = detection[4] * image_height

                        # get the bounding box width and height
                        box_width = detection[5] * image_width
                        box_height = detection[6] * image_height

                        # draw a rectangle around each detected object
                        cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color,
                                      thickness=2)

                        # put the class name text on the detected object
                        cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                                    2)
                        # put the FPS text on top of the frame
                        cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('image', image)
                self.output.write(image)

                # Properly end the Stream
                # @TODO: End by Sign?
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

        self.capture.release()
        cv2.destroyAllWindows()
