import cv2
import os
import glob

def make_video(image_files, output_path, fps=10):
    if not image_files:
        return
    # Read the first image to get frame size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)

        if frame is not None:
            print(f"Writing frame {i+1}/{len(image_files)}: {img_path}")
            video_writer.write(frame)
    video_writer.release()

if __name__ == "__main__":

    image_folder = "/media/tyler/hummingbird/wire_tracking_05-07_40fov/ransac_results_3d/"
    output_path = "/media/tyler/hummingbird/wire_tracking_05-07_40fov/ransac_results_3d_video.mp4"
    fps = 30

    rgb_image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    make_video(rgb_image_files, output_path, fps)