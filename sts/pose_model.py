import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import tensorflow_io as tfio
import subprocess
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# from html import HTML
import joblib
from tensorflow.keras.models import load_model

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.2

# Helper Functions
def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

# def progress(value, max=100):
#   return HTML("""
#       <progress
#           value='{value}'
#           max='{max}',
#           style='width: 100%'
#       >
#           {value}
#       </progress>
#   """.format(value=value, max=max))

def movenet(input_image, interpreter):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

# Cropping Algorithm
def init_crop_region(image_height, image_width):
  """Defines the default crop region.

  The function provides the initial crop region (pads the full image from both
  sides to make it a square image) when the algorithm cannot reliably determine
  the crop region from the previous frame.
  """
  if image_width > image_height:
    box_height = image_width / image_height
    box_width = 1.0
    y_min = (image_height / 2 - image_width / 2) / image_height
    x_min = 0.0
  else:
    box_height = 1.0
    box_width = image_height / image_width
    y_min = 0.0
    x_min = (image_width / 2 - image_height / 2) / image_width

  return {
    'y_min': y_min,
    'x_min': x_min,
    'y_max': y_min + box_height,
    'x_max': x_min + box_width,
    'height': box_height,
    'width': box_width
  }

def torso_visible(keypoints):
  """Checks whether there are enough torso keypoints.

  This function checks whether the model is confident at predicting one of the
  shoulders/hips which is required to determine a good crop region.
  """
  return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE) and
          (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(
    keypoints, target_keypoints, center_y, center_x):
  """Calculates the maximum distance from each keypoints to the center location.

  The function returns the maximum distances from the two sets of keypoints:
  full 17 keypoints and 4 torso keypoints. The returned information will be
  used to determine the crop size. See determineCropRegion for more detail.
  """
  torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
  max_torso_yrange = 0.0
  max_torso_xrange = 0.0
  for joint in torso_joints:
    dist_y = abs(center_y - target_keypoints[joint][0])
    dist_x = abs(center_x - target_keypoints[joint][1])
    if dist_y > max_torso_yrange:
      max_torso_yrange = dist_y
    if dist_x > max_torso_xrange:
      max_torso_xrange = dist_x

  max_body_yrange = 0.0
  max_body_xrange = 0.0
  for joint in KEYPOINT_DICT.keys():
    if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
      continue
    dist_y = abs(center_y - target_keypoints[joint][0]);
    dist_x = abs(center_x - target_keypoints[joint][1]);
    if dist_y > max_body_yrange:
      max_body_yrange = dist_y

    if dist_x > max_body_xrange:
      max_body_xrange = dist_x

  return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(
      keypoints, image_height,
      image_width):
  """Determines the region to crop the image for the model to run inference on.

  The algorithm uses the detected joints from the previous frame to estimate
  the square region that encloses the full body of the target person and
  centers at the midpoint of two hip joints. The crop size is determined by
  the distances between each joints and the center point.
  When the model is not confident with the four torso joint predictions, the
  function returns a default crop which is the full image padded to square.
  """
  target_keypoints = {}
  for joint in KEYPOINT_DICT.keys():
    target_keypoints[joint] = [
      keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
      keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
    ]

  if torso_visible(keypoints):
    center_y = (target_keypoints['left_hip'][0] +
                target_keypoints['right_hip'][0]) / 2;
    center_x = (target_keypoints['left_hip'][1] +
                target_keypoints['right_hip'][1]) / 2;

    (max_torso_yrange, max_torso_xrange,
      max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
          keypoints, target_keypoints, center_y, center_x)

    crop_length_half = np.amax(
        [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
          max_body_yrange * 1.2, max_body_xrange * 1.2])

    tmp = np.array(
        [center_x, image_width - center_x, center_y, image_height - center_y])
    crop_length_half = np.amin(
        [crop_length_half, np.amax(tmp)]);

    crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

    if crop_length_half > max(image_width, image_height) / 2:
      return init_crop_region(image_height, image_width)
    else:
      crop_length = crop_length_half * 2;
      return {
        'y_min': crop_corner[0] / image_height,
        'x_min': crop_corner[1] / image_width,
        'y_max': (crop_corner[0] + crop_length) / image_height,
        'x_max': (crop_corner[1] + crop_length) / image_width,
        'height': (crop_corner[0] + crop_length) / image_height -
            crop_corner[0] / image_height,
        'width': (crop_corner[1] + crop_length) / image_width -
            crop_corner[1] / image_width
      }
  else:
    return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size):
  """Crops and resize the image to prepare for the model input."""
  boxes=[[crop_region['y_min'], crop_region['x_min'],
          crop_region['y_max'], crop_region['x_max']]]
  output_image = tf.image.crop_and_resize(
      image, box_indices=[0], boxes=boxes, crop_size=crop_size)
  return output_image

def run_inference(movenet, image, crop_region, crop_size, interpreter):
  """Runs model inference on the cropped region.

  The function runs the model inference on the cropped region and updates the
  model output to the original image coordinate system.
  """
  image_height, image_width, _ = image.shape
  input_image = crop_and_resize(
    tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
  # Run model inference.
  keypoints_with_scores = movenet(input_image, interpreter)
  # Update the coordinates.
  for idx in range(17):
    keypoints_with_scores[0, 0, idx, 0] = (
        crop_region['y_min'] * image_height +
        crop_region['height'] * image_height *
        keypoints_with_scores[0, 0, idx, 0]) / image_height
    keypoints_with_scores[0, 0, idx, 1] = (
        crop_region['x_min'] * image_width +
        crop_region['width'] * image_width *
        keypoints_with_scores[0, 0, idx, 1]) / image_width
  return keypoints_with_scores

'''
This function extracts the video data from the input video.

Parameters:
    image (numpy.ndarray): The input image
    video_path (str): The path to the video
    input_size (int): The input size of the model
    interpreter (tf.lite.Interpreter): The TFLite interpreter
    
Returns:
    df (pandas.DataFrame): The dataframe containing the extracted video data
'''
def extract_video_data(image, video_path, input_size, interpreter):

    label_list = []
    label_list = ['frame #', 'time']

    # Create x and y values for each pose estimate point
    for i in KEYPOINT_DICT.keys():
        label_list.append(i+'_x')
        label_list.append(i+'_y')

    # Create df to store output for each point at each frame
    df = pd.DataFrame(columns=label_list)

    # Load the input image.
    num_frames, image_height, image_width, _ = image.shape
    crop_region = init_crop_region(image_height, image_width)

    # Run ffprobe to get video metadata
    ffprobe_command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=avg_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    video_output = subprocess.check_output(ffprobe_command).decode('utf-8').strip()

    # Parse the output to get the frame rate
    numerator, denominator = map(int, video_output.split('/'))
    frame_rate = numerator / denominator
    
    # Calculate the time interval between frames.
    time_interval = timedelta(seconds=1 / frame_rate)

    # Loop through the frames
    for frame_idx in range(num_frames):

        # Get keypoints
        keypoints_with_scores = run_inference(
            movenet, image[frame_idx, :, :, :], crop_region,
            crop_size=[input_size, input_size], interpreter=interpreter)

        # Calculate the timestamp for the current frame.
        frame_time = time_interval * frame_idx

        # Extract keypoints
        keypoints = []
        for i in range(17):
            keypoints.append(keypoints_with_scores[0, 0, i, 0])
            keypoints.append(keypoints_with_scores[0, 0, i, 1])

        df.loc[frame_idx] = [frame_idx, frame_time] + keypoints

        # Crop the frame to the region of interest.
        crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width)

    return df

'''
This function ges the input from the user and sets the attributes of the output_df.

Parameters:
    output_df (pandas.DataFrame): The output dataframe

Returns:
    output_df (pandas.DataFrame): The output dataframe with the user input
'''
def get_user_input(output_df):
  # Get input from the user
  on_off_medication = input("On or Off medication: ") #either On medication, Off medication
  dbs_state = input("DBS state: ") # If healthy control participant: always "Control". If participant with Parkinson's disease: either "On DBS" (deep brain stimulator switched on or within 1 hour of it being switched off), "Off DBS" (1 hour or longer after deep brain stimulator switched off until it is switched back on again) or "-" (no deep brain stimulator in situ).

  # Set attributes inputted to output_df
  output_df.loc[0, 'On_or_Off_medication'] = on_off_medication
  output_df.loc[0, 'DBS_state'] = dbs_state
  return output_df

'''
This function calculates the sts_whole_episode_duration.

Parameters:
    df (pandas.DataFrame): The dataframe containing the extracted video data
    output_df (pandas.DataFrame): The output dataframe

Returns:
    output_df (pandas.DataFrame): The output dataframe with the sts_whole_episode_duration
'''
def calculate_sts_whole_episode_duration(df, output_df):
  # Get frame 3 time
  start_time = df.loc[df['frame #'] == 3, 'time'].iloc[0]

  # Find the max frame
  max_frame = df['frame #'].max()
  end_time = df.loc[df['frame #'] == max_frame-3, 'time'].iloc[0]

  sts_whole_episode_duration = end_time - start_time

  # Save in first entry of output_df
  output_df.loc[0, 'sts_whole_episode_duration'] = sts_whole_episode_duration.total_seconds()

  return output_df

'''
This function calculates the sts_final_attempt_duration.

Parameters:
    df (pandas.DataFrame): The dataframe containing the extracted video data
    output_df (pandas.DataFrame): The output dataframe

Returns:
    output_df (pandas.DataFrame): The output dataframe with the sts_final_attempt_duration
'''
def calculate_sts_final_attempt_duration(df, output_df):
  # Duration in seconds of "final attempt duration" label in milliseconds, comprising their impression of the duration between the lowest point of the head (start)
  # and when the person was fully upright/the maximum vertical position of the vertex of the head (end).

  # Create a head column that is the average y values of nose_y, left_eye_y, right_eye_y, left_ear_y, right_ear_y
  df['head'] = (df['nose_y'] + df['left_eye_y'] + df['right_eye_y'] + df['left_ear_y'] + df['right_ear_y']) / 5

  # Get min and max values and their rows from head column
  # REMEMBER: Pose estimation y-axis starts at top left corner and moves down (reverse)
  min_head = df['head'].min()
  max_head = df['head'].max()

  # Get highest head point
  highest_head = df[df['head'] == min_head]

  # Get lowest head point
  lowest_head = df[df['head'] == max_head]

  # Get time entry at highest_head and lowest_head
  highest_head_time = highest_head['time'].iloc[0]
  lowest_head_time = lowest_head['time'].iloc[0]
  print('Highest head time: ', highest_head_time, 'Lowest head time: ', lowest_head_time)

  # Calculate final attempt duration
  final_attempt_duration = highest_head_time.total_seconds() - lowest_head_time.total_seconds()
  print('Total duration: ', final_attempt_duration)

  # Save in second entry of output_df
  output_df.loc[0, 'sts_final_attempt_duration'] = final_attempt_duration

  return output_df

'''
This function calculates the MDS-UPDRS score 3.9 arising from chair and STS additional features.

Parameters:
    df (pandas.DataFrame): The dataframe containing the extracted video data
    output_df (pandas.DataFrame): The output dataframe

Returns:
    output_df (pandas.DataFrame): The output dataframe with the MDS-UPDRS score 3.9 arising from chair and STS additional features
  '''
def calculate_MDS_UPDRS_score_3_9_and_STS_additional_features(df, output_df):
  mds_score = 0
  STS_additional_features = []

  # Compute the differene between min and max values of left_shoulder_x, right_shoudler_x, right_elbow_x, left_elbow_x, left_hip_x, right_hip_x from df
  left_shoulder_x_diff = df['left_shoulder_x'].max() - df['left_shoulder_x'].min()
  right_shoulder_x_diff = df['right_shoulder_x'].max() - df['right_shoulder_x'].min()
  left_elbow_x_diff = df['left_elbow_x'].max() - df['left_elbow_x'].min()
  right_elbow_x_diff = df['right_elbow_x'].max() - df['right_elbow_x'].min()
  left_hip_x_diff = df['left_hip_x'].max() - df['left_hip_x'].min()
  right_hip_x_diff = df['right_hip_x'].max() - df['right_hip_x'].min()

  # Compare differences to threshold value to determine swaying
  threshold = 0.25
  if left_shoulder_x_diff > threshold or right_shoulder_x_diff > threshold or left_elbow_x_diff > threshold or right_elbow_x_diff > threshold or left_hip_x_diff > threshold or right_hip_x_diff > threshold:
      mds_score = 2
      STS_additional_features.append("Uses arms of chair")
  else:
      print("No sway detected.")

  # If whole duration > ~5.5 seconds then MDS score = 3
  if output_df.loc[0, 'sts_whole_episode_duration'] > 5.5:
    mds_score = 3

  # If final duration is greater than 2.1 seconds, then STS additioanl features include "Slow"
  if output_df.loc[0, 'sts_final_attempt_duration'] > 2.1:
      STS_additional_features.append("Slow")

  # Set mds_score and STS additional features in output_df
  output_df.loc[0, 'MDS-UPDRS_score_3.9 _arising_from_chair'] = mds_score
  output_df.loc[0, 'STS_additional_features'] = ','.join(STS_additional_features)

  return output_df

'''
This function analyzes the video to obtain the data needed for the classification model.

Parameters:
    df (pandas.DataFrame): The dataframe containing the extracted video data

Returns:
    output_df (pandas.DataFrame): The output dataframe containing the data needed for the classification model
'''
def analyze_video(df):
  # Create output dataframe
  output_df = pd.DataFrame(columns=['sts_whole_episode_duration','sts_final_attempt_duration','On_or_Off_medication','DBS_state','STS_additional_features','MDS-UPDRS_score_3.9 _arising_from_chair'])

  # Get user input
  output_df = get_user_input(output_df)

  # Calculate sts_whole_episode_duration
  output_df = calculate_sts_whole_episode_duration(df, output_df)

  # Calculate sts_final_attempt_duration 
  output_df = calculate_sts_final_attempt_duration(df, output_df)

  # Calculate MDS and STS additional features
  output_df = calculate_MDS_UPDRS_score_3_9_and_STS_additional_features(df, output_df)

  return output_df

'''
This function processes the video frames.

Parameters:
    video_path (str): The path to the input video

Returns:
    frames (numpy.ndarray): The processed video frames
'''
def process_video_frames(video_path):
      cap = cv2.VideoCapture(video_path)
      frames = []
      while True:
          ret, frame = cap.read()
          if not ret:
              break

          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
          frame = cv2.resize(frame, (224, 224))  # Resize the frame
          frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32) / 255.0
          frames.append(frame_tensor)

      cap.release()
      return tf.stack(frames)  # Stack into one tensor

'''
This function performs pose estimation on the input video.

Parameters:
    video_path (str): The path to the input video
    
Returns:
    output_df (pandas.DataFrame): The output dataframe containing the extracted video data
'''
def pose_estimation(video_path):

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path="sts/model.tflite")
  interpreter.allocate_tensors()

  # Load and preproces the video
  video = tf.io.read_file(video_path)

  # TFIO.ffmpeg.decode_video does not work locally on MacOS
  # image = tfio.experimental.ffmpeg.decode_video(video)
  image = process_video_frames(video_path)

  # Set input size based upon model, default 192
  input_size = 192

  # Extract video data
  df = extract_video_data(image, video_path, input_size, interpreter)

  # Analyze video to obtain data needed for classification model
  output_df = analyze_video(df)

  return output_df

'''
This function organizes the data for the classification model.

Parameters:
    df (pandas.DataFrame): The dataframe containing the extracted video data

Returns:
    X (numpy.ndarray): The standardized features
'''
def organize_data_for_classification(df):
  # Load the label encoder
  label_encoder = joblib.load('sts/label_encoder.pkl')
  
  # Encode categorical features
  df['On_or_Off_medication'] = label_encoder.fit_transform(df['On_or_Off_medication'])
  df['STS_additional_features'] = label_encoder.fit_transform(df['STS_additional_features'])
  df['DBS_state'] = label_encoder.fit_transform(df['DBS_state'])

  # Standardize features
  scaler = StandardScaler()
  X = scaler.fit_transform(df)
  return X

'''
This function loads the classification model.

Parameters:
    model_path (str): Path to the saved model

Returns:
    sts_model (tensorflow.keras.Model): The saved model
'''
def load_classification_model(model_path='sts/sts_model.h5'):
  # Load the saved model
  return load_model(model_path)

'''
This function makes predictions using the classification model.

Parameters:
    sts_input (numpy.ndarray): The standardized features

Returns:
    sts_predictions (numpy.ndarray): Prediction probabilities
'''
def make_predicitions(sts_input):
  # Load the classification model
  sts_model = load_classification_model()

  # Make a prediction
  sts_predictions = sts_model.predict(sts_input)

  return sts_predictions

'''
This function performs the entire sts score prediction process.

Parameters:
    video_path (str): The path to the input video
    
Returns:
    sts_predictions (numpy.ndarray): Prediction probabilities
'''
def main(video_path='../test-data/sts-test.MOV'):
  # Perform pose estimation
  output_df = pose_estimation(video_path)

  # Analyze video to obtain data needed for classification model
  sts_input = organize_data_for_classification(output_df)

  # Make predictions
  sts_predictions = make_predicitions(sts_input)

  return sts_predictions

if __name__ == "__main__":
  main()