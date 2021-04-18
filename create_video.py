import cv2
import os


def generate_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


generate_video(image_folder="C:/Users/Administrator/Desktop/Linh tinh/Dataset/yolov5/Evaluate_video/",
               video_name="Evaluate_video.avi")
