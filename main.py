import cv2
import argparse
from sklearn.cluster import DBSCAN
import numpy


def find_faces(image, args):

    face_cascade = cv2.CascadeClassifier()

    if not face_cascade.load(cv2.samples.findFile(args.face_cascade)):
        raise ValueError('--(!)Error loading face cascade')

    #-- Detect faces
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=10, minSize=[30,30])
    centers = []
    for (x,y,w,h) in faces:
        center = [x + w // 2, y + h // 2, w, h]
        centers.append(center)
        # image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)#0, 0, 360, (0, 255, 0), 4)
    if len(centers) > 0:
        centers = numpy.array(centers)
    else:
        centers = None
    return centers


def find_eyes(image, args):
    eyes_cascade = cv2.CascadeClassifier()

    if not eyes_cascade.load(cv2.samples.findFile(args.eyes_cascade)):
        raise ValueError('--(!)Error loading eyes cascade')

    # -- Detect eyes
    eyes = eyes_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=10, minSize=[30, 30])

    centers = []
    for (x, y, w, h) in eyes:
        center = [x + w // 2, y + h // 2, w, h]
        centers.append(center)
        # image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)#0, 0, 360, (0, 255, 0), 4)
    if len(centers) > 0:
        centers = numpy.array(centers)
    else:
        centers = None
    return centers


# def facial_detection_preprocessing(image):
#     # image = cv2.equalizeHist(image)
#     # Create a CLAHE object (with optional parameters)
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     #
#     # # Apply CLAHE
#     # hist_gray = clahe.apply(image)
#     # hist_gray = custom_equalize(image)
#
#
#     # # Define the brightness value you want to add
#     # brightness_value = 50
#     #
#     # # Add the brightness value
#     # brightened_image = cv2.add(hist_gray, numpy.array([brightness_value]))
#
#     return image


# def custom_equalize(image):
#     # Calculate histogram and CDF
#     hist, bins = numpy.histogram(image.flatten(), 256, [0, 256])
#     cdf = hist.cumsum()
#     cdf_normalized = cdf * float(hist.max()) / cdf.max()
#
#     # Apply a more aggressive transformation to the darker side of the spectrum
#     # For instance, use a cubic root transformation
#     cdf_half = numpy.cbrt(cdf_normalized[:128])
#
#     # Scale the transformed CDF more aggressively
#     # The scaling factor can be adjusted to control the stretch
#     scale_factor = cdf_normalized[127] / cdf_half[-1]
#     cdf_modified = numpy.copy(cdf_normalized)
#     cdf_modified[:128] = cdf_half * scale_factor * 1.5  # Adjust the 1.5 as needed
#
#     # Use the modified CDF to map the original pixel values to the new ones
#     cdf_m = numpy.ma.masked_equal(cdf_modified, 0)
#     cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
#     cdf_final = numpy.ma.filled(cdf_m, 0).astype('uint8')
#     image_equalized = cdf_final[image]
#
#     return image_equalized
# def invert(image):
#     # Ensure all values are non-negative
#     image = numpy.clip(image, 1, None)
#
#     # Log Transformation
#     c = 45
#     log_of_image = numpy.log(image)
#     log_transformed = c * (log_of_image)
#
#     # Convert to 8-bit unsigned integer format
#     log_transformed = numpy.uint8(log_transformed)
#     cv2.imshow("log_transformed", log_transformed)
#
#
#     # Inverse-Log Transformation
#     inverse_log_transformed = c * (numpy.exp(log_transformed / c) - 1)
#
#     # Convert to 8-bit unsigned integer format again
#     inverse_log_transformed = numpy.uint8(inverse_log_transformed)
#     cv2.imshow("inverse_log_transformed", inverse_log_transformed)
#
#     return inverse_log_transformed


# def detectAndDisplay(image, face_cascade, eyes_cascade):
#     frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.equalizeHist(frame_gray)
#
#     #-- Detect faces
#     faces = face_cascade.detectMultiScale(frame_gray)
#
#     for (x,y,w,h) in faces:
#         center = (x + w//2, y + h//2)
#         image = cv2.ellipse(image, center, (w // 2, int(h * 0.75)), 0, 0, 360, (0, 255, 0), 4)
#         face_region_of_interest = frame_gray[y:y+h,x:x+w]
#
#         #-- In each face, detect eyes
#         eyes = eyes_cascade.detectMultiScale(face_region_of_interest)
#
#         top_left = None
#         bottom_right = None
#         for (x2,y2,w2,h2) in eyes:
#             x_absolute = x + x2
#             y_absolute = y + y2
#             if top_left is None or top_left[0] > x_absolute:
#                 top_left = (x_absolute, y_absolute)
#             if bottom_right is None or bottom_right[0] < x_absolute:
#                 bottom_right = (x_absolute + w2, y_absolute + h2)
#         image = cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 4)
#
#     return image


def rotate(image, degrees, original_dimensions=None, detected_objects=None):
    # Image dimensions
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)

    # Absolute cosine and sine of the rotation angle
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # New dimensions after rotation
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to consider translation
    rotation_matrix[0, 2] += new_w / 2 - center[0]
    rotation_matrix[1, 2] += new_h / 2 - center[1]

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    if detected_objects is not None:
        center_points = detected_objects[:, :2]  # Selects all rows and the first two columns (x and y)

        transformed_points = cv2.transform(numpy.array([center_points]), rotation_matrix)
        reshaped_points = transformed_points.reshape(-1, 2)
        for (index, transformed_point) in enumerate(reshaped_points):
            detected_objects[index, 0] = transformed_point[0]  # Update x-coordinate of the center point
            detected_objects[index, 1] = transformed_point[1]  # Update y-coordinate of the center point

            # cv2.circle(rotated_image, center=face_centers[index, :2], radius=5, thickness=2, color=(0,0,255))

    if original_dimensions is not None and (original_w := original_dimensions[0]) and (original_h := original_dimensions[1]):
        x = (new_w - original_w) // 2
        y = (new_h - original_h) // 2

        # Crop the image
        rotated_image = rotated_image[y:y+original_h, x:x+original_w]

        if detected_objects is not None:
            # Adjust the centers
            for index, row in enumerate(detected_objects):
                detected_objects[index, 1] -= y
                detected_objects[index, 0] -= x

    return rotated_image, detected_objects


def preprocess_for_face_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=None, fx=2, fy=2)
    image = cv2.GaussianBlur(image,ksize=(3,3), sigmaX=20, sigmaY=20)
    return image


def preprocess_for_eye_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=None, fx=2, fy=2)
    image = cv2.GaussianBlur(image,ksize=(3,3), sigmaX=20, sigmaY=20)
    return image


def detect(detector, image, args):
    all_objects =[]

    for angle in [i * 15 + -45 for i in range(6)]:
        image, detected_objects = detect_at(detector, angle, image, args)

        if detected_objects is not None:
            for index in range(detected_objects.shape[0]):
                detected_objects[index, 0] /= 2  # Update x-coordinate of the center point
                detected_objects[index, 1] /= 2  # Update y-coordinate of the center point

            all_objects += detected_objects.tolist()

    return cluster_objects(all_objects, args)


def cluster_objects(objects, args):
    objects = numpy.array(objects)

    # eps determines cluster affinity, min_samples required for a cluster
    dbscan = DBSCAN(eps=args.cluster_eps_eyes, min_samples=args.cluster_min_eyes)
    clusters = dbscan.fit_predict(objects)

    # Averaging points in each cluster
    averages = []
    for cluster in set(clusters):
        avg = objects[clusters == cluster].mean(axis=0).round().astype(int)
        averages.append(avg)

    return averages


def detect_at(detector, angle, image, args):
    original_height = image.shape[0]
    original_width = image.shape[1]

    image, _ = rotate(image, angle)
    objects = detector(image, args)
    rotated, objects = rotate(image, -1 * angle, (original_width, original_height), objects)

    return rotated, objects


def extract_face_regions(original):

    image_processed = preprocess_for_face_detection(original)
    faces = detect(find_faces, image_processed, args)

    face_data = []
    if faces is not None:
        for row in faces:
            center_x, center_y, width, height = row.tolist()
            origin_x = center_x - width // 2
            origin_y = center_y - height // 2

            # Extract the region of interest
            face_image = original[origin_y:origin_y + height, origin_x:origin_x + width]
            origin = (origin_x, origin_y)
            data = (face_image, origin, width, height)
            face_data.append(data)

    return face_data
    #         # cv2.imshow(f'face_image {face_image.shape}', face_image)
    #         # cv2.imwrite("closeup.png", face_image)
    #
    #         image_processed = preprocess_for_eye_detection(face_image)
    #         eyes = detect(find_eyes, image_processed, args)
    #         print(eyes)
    #
    #         cv2.imshow(f'eyes {image_processed.shape}', image_processed)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #
    #         cv2.circle(original, center=(center_x, center_y), radius=5, thickness=2, color=(0, 0, 255))
    #         cv2.rectangle(original, (origin_x, origin_y), (origin_x + width, origin_y + height), color=(0, 0, 255))
    #
    # # Process the clustering results...
    #
    # cv2.imshow(f'original {original.shape}', original)
    # # cv2.imshow(f'image {image.shape}', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", type=str, default=None, help="path to the input image")
    # ap.add_argument("-c", "--clip", type=float, default=2.0, help="threshold for contrast limiting")
    # ap.add_argument("-t", "--tile", type=int, default=8, help="tile grid size -- divides image into tile x time cells")

    # ap.add_argument('--face_cascade',  default='classifiers/haarcascades/haarcascade_frontalface_alt2.xml')
    # ap.add_argument('--face_cascade', default='classifiers/haarcascades/haarcascade_frontalface_alt.xml')
    ap.add_argument('--face_cascade', default='classifiers/lbpcascades/lbpcascade_frontalface_improved.xml')

    # ap.add_argument('--face_cascade', default='classifiers/lbpcascades/lbpcascade_frontalface.xml')
    # ap.add_argument('--face_cascade',  default='classifiers/haarcascades/haarcascade_frontalface_alt_tree.xml')
    # ap.add_argument('--face_cascade',  default='classifiers/haarcascades/haarcascade_frontalface_default.xml')

    ap.add_argument('--eyes_cascade', default='classifiers/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    # ap.add_argument('--eyes_cascade', default='classifiers/haarcascades/haarcascade_eye.xml')
    # ap.add_argument('--eyes_cascade', default='classifiers/haarcascades/lefteye_2splits.xml')
    # ap.add_argument('--eyes_cascade', default='classifiers/haarcascades/righteye_2splits.xml')
    # ap.add_argument('--camera', type=int, default=0)
    # args = parser.parse_args()

    ap.add_argument('--cluster_min_face', type=int, default=1,  help="number of neighbors necessary to be considered a detected face group")
    ap.add_argument('--cluster_eps_face', type=int, default=50, help="eps to be considered a detected face group")

    ap.add_argument('--cluster_min_eyes', type=int, default=1, help="number of neighbors necessary to be considered a detected eye group")
    ap.add_argument('--cluster_eps_eyes', type=int, default=5, help="eps to be considered a detected eye group")
    args = ap.parse_args()

    image_name = "closeup.png"
    original = cv2.imread(image_name)

    image_processed = preprocess_for_eye_detection(original)
    eyes = detect(find_eyes, image_processed, args)
    print(eyes)

    # if eyes is not None:
    #     for row in eyes:
    #         center_x, center_y, width, height = row.tolist()
    #         cv2.circle(original, center=(center_x, center_y), radius=5, thickness=2, color=(0, 255, 0))
    #
    # cv2.imshow(f'eyes {original.shape}', original)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




    image_name = "dogpark.jpeg"  # "dogs.jpeg"#"classroom.jpeg"#
    original = cv2.imread(image_name)
    face_data = extract_face_regions(original)

    for data in face_data:
        face_image, (origin_x, origin_y), width, height = data
        # cv2.circle(original, center=(center_x, center_y), radius=5, thickness=2, color=(0, 0, 255))
        cv2.rectangle(original, (origin_x, origin_y), (origin_x + width, origin_y + height), color=(0, 0, 255))

    cv2.imshow(f'original {original.shape}', original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # print("[INFO] applying CLAHE...")
    # clahe = cv2.createCLAHE(clipLimit=args.clip,  tileGridSize=(args.tile, args.tile))
    # equalized = clahe.apply(image)
    # cv2.imshow('equalized', equalized)

    # # Constants for finding range of skin color in YCrCb
    # min_YCrCb = numpy.array([0, 133, 77], numpy.uint8)
    # max_YCrCb = numpy.array([255, 173, 127], numpy.uint8)
    #
    # # Convert image to YCrCb
    # imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    #
    # # Find region with skin tone in YCrCb image
    # skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    #
    # # Do contour detection on skin region
    # contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # sourceImage = image.copy()
    #
    # # Draw the contour on the source image
    # for i, c in enumerate(contours):
    #     area = cv2.contourArea(c)
    #     if area > 1000:
    #         cv2.drawContours(sourceImage, contours, i, (0, 255, 0), 3)
    #
    # cv2.imshow('sourceImage', sourceImage)


    # output = cv2.cvtColor(dogpark, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # three_channel_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # cv2.imshow("gray", gray_image)
    # three_channel_image = invert(image)


    # canny = cv2.Canny(image, threshold1=100, threshold2=200)
    # cv2.imshow('canny', canny)

    # Define the brightness value you want to add
    # brightness_value = 150

    # Add the brightness value
    # brightened_image = cv2.add(gray_image, numpy.array([brightness_value]))
    # cv2.imshow("brightened_image", brightened_image)