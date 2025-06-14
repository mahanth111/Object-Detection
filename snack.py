import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Load the TensorFlow model and labels
graph_def = tf.compat.v1.GraphDef()
labels = []

filename = r"C:/Users/mahan/Downloads/SNACK-DETECTION1.TensorFlow/model.pb"
labels_filename = r"C:/Users/mahan/Downloads/SNACK-DETECTION1.TensorFlow/labels.txt"

with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

# Function to update image orientation based on EXIF tags
def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

# Function to convert image to OpenCV format
def convert_to_opencv(image):
    # Your code for conversion here
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image


# Function to crop center of image
def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

# Function to resize image to maximum dimension of 1600
def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

# Function to resize image to 256x256
def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

# Streamlit app title
st.title("Snack Item Detection")

# Checkbox to start/stop object detection
run = st.checkbox('Start/Stop the Detection')
FRAME_WINDOW = st.image([])

# Load the camera
cap = cv2.VideoCapture(0)

# Main Streamlit app loop
while run:
    # Capture a frame
    ret, frame = cap.read()

    # Convert frame to PIL format
    pil_frame = Image.fromarray(frame)

    # Update orientation based on EXIF tags
    pil_frame = update_orientation(pil_frame)

    # Convert to OpenCV format
    cv2_frame = convert_to_opencv(pil_frame)

    # Resize the image
    resized_image = resize_down_to_1600_max_dim(cv2_frame)

    # Crop to square
    cropped_image = crop_center(resized_image, 1600, 1600)

    # Resize to 256x256
    augmented_image = resize_to_256_square(cropped_image)

    # Perform prediction
    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
        network_input_size = input_tensor_shape[1]
        augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

        prob_tensor = sess.graph.get_tensor_by_name('loss:0')
        predictions = sess.run(prob_tensor, {'Placeholder:0': [augmented_image]})
        highest_probability_index = np.argmax(predictions)
        st.write('Object detected as:', labels[highest_probability_index])  

        for box in labels[highest_probability_index]:
        # # Retrieve bounding box tensor (assuming it's named "detection_boxes")
            # bbox_tensor = sess.graph.get_tensor_by_name('Placeholder:0')

            # # Run the model and get both predictions and bounding boxes
            # predictions, bboxes = sess.run([prob_tensor, bbox_tensor], {'Placeholder:0': [augmented_image]})

            # # Get the index with highest probability
            # highest_probability_index = np.argmax(predictions)
            # for i in range(highest_probability_index):  # Iterate through each image in the batch
                # offset = i * 4
            # # Extract x1, y1, x2, y2 from bounding box for the highest probability class
                # if len(bboxes.shape) > 1:  # Check if bboxes has multiple detections
                    # bbox = [0, highest_probability_index]  # Assuming one image and selecting by index
                # else:
                    # bbox = bboxes[offset]  # Assuming single detection output
                # x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0], bbox[1]
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.rectangle(frame, (80, 80), (80, 80), (255, 0, 255), 3)

            # object details
            org = [80, 80]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(frame, labels[highest_probability_index], org, font, fontScale, color, thickness)

    # Display the frame with Streamlit
    cv2.imshow('Live Camera', frame)
    FRAME_WINDOW.image(frame, channels="BGR")

    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == 27:  # ESC key0
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()