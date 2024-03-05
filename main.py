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

    image_width = image.shape[1]
    image_center = image_width // 2, image.shape[0] // 2

    centers = []
    distances = []
    for (x, y, w, h) in eyes:
        center = [x + w // 2, y + h // 2, w, h]

        # remove points that are close and far from center
        distance_to_center = distance_between(image_center, center)
        percent_of_width = distance_to_center / image_width
        distances.append(percent_of_width)
        if 0.1 < percent_of_width < 0.25:
            centers.append(center)

    if len(centers) > 0:
        centers = numpy.array(centers)
    else:
        centers = None
    return centers


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


# def preprocess_for_face_detection(image, resize_factor):
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(image, dsize=None, fx=resize_factor, fy=resize_factor)
#     image = cv2.GaussianBlur(image,ksize=(3,3), sigmaX=5, sigmaY=5)
#     return image


def preprocess(image, resize_factor):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=None, fx=resize_factor, fy=resize_factor)
    image = cv2.GaussianBlur(image,ksize=(3,3), sigmaX=5, sigmaY=5)
    return image


def detect(detector, image, args, resize_factor):
    all_objects =[]

    # run the image through the detector at 15 degree intervals from -45 to 45
    for angle in [i * 15 + -45 for i in range(6)]:
        image, detected_objects = detect_at(detector, angle, image, args)

        if detected_objects is not None:
            for index in range(detected_objects.shape[0]):
                detected_objects[index, 0] /= resize_factor # Update x-coordinate of the center point
                detected_objects[index, 1] /= resize_factor # Update y-coordinate of the center point

            all_objects += detected_objects.tolist()

    return cluster_objects(detector, all_objects, args)


def cluster_objects(detector, objects, args):
    objects = numpy.array(objects)

    if detector.__name__ == "find_faces":
        # eps determines cluster affinity, min_samples required for a cluster
        dbscan = DBSCAN(eps=args.cluster_eps_faces, min_samples=args.cluster_min_faces)
    elif detector.__name__ == "find_eyes":
        dbscan = DBSCAN(eps=args.cluster_eps_eyes, min_samples=args.cluster_min_eyes)
    else:
        raise ValueError("Unsupported detector type")

    clusters = dbscan.fit_predict(objects) if len(objects) > 0 else objects

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


def blur_eye_regions(image, args, resize_factor):
    image_processed = preprocess(image, resize_factor)
    eyes = detect(find_eyes, image_processed, args, resize_factor)
    print(eyes)

    # image_width = image.shape[1]
    # image_center = image.shape[1] // 2, image.shape[0] // 2
    #
    # distances = []
    if eyes is not None:
        # for i in range(len(eyes)):
        #     row = eyes[i]
        #     x1, y1, _, _ = row.tolist()
        #     for j, other in enumerate(eyes):
        #         x2, y2, _, _ = other.tolist()
        #         distance = distance_between((x1, y1), (x2, y2))
        #         ratio = distance / image_width
        #         distances.append(ratio)

        # print(f"distances: {distances}")
        # distances.sort()
        # print(f"distances: {distances}")

        for row in eyes:
        #     center_x, center_y, width, height = row.tolist()
        #     distance_from_center = distance_between(image_center, (center_x, center_y))
        #     distances.append(distance_from_center)
        #
        # eyes_with_pairs = []
        # diffs = []
        # for i in range(len(distances)):
        #     distance_from_center = distances[i]
        #     for j, other in enumerate(distances):
        #         if j == i:
        #             continue
        #         diff = abs(distance_from_center - other)
        #         diffs.append(diff)
        #         if 0.5 < diff < 5:
        #             eyes_with_pairs.append(eyes[i])
        # print(f"diffs: {diffs}")

        # diffs.sort()
        # print(f"diffs-: {diffs}")
        # for row in eyes_with_pairs:
            center_x, center_y, width, height = row.tolist()
            # image = circular_blur(image, center=(center_x, center_y), radius=15)
            cv2.circle(image, center=(center_x, center_y), radius=5, thickness=2, color=(0, 255, 0))

    return image


def distance_between(point_1, point_2):
    return ((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2) ** 0.5

def extract_face_regions(image):

    resize_factor = 2
    image_processed = preprocess(image, resize_factor)
    faces = detect(find_faces, image_processed, args, resize_factor)

    face_data = []
    if faces is not None:
        for row in faces:
            center_x, center_y, width, height = row.tolist()
            origin_x = center_x - width // 2
            origin_y = center_y - height // 2

            # Extract the region of interest
            face_image = image[origin_y:origin_y + height, origin_x:origin_x + width]
            origin = (origin_x, origin_y)
            data = (face_image, origin, width, height)
            face_data.append(data)

    return face_data

def circular_blur(image, center, radius):
    # thanks to https://stackoverflow.com/a/60911696

    # create white circle mask
    mask_img = numpy.zeros(image.shape, dtype='uint8')
    cv2.circle(mask_img, center, radius, (255, 255, 255), -1)

    # make a blurred copy of the entire image
    img_all_blurred = cv2.GaussianBlur(image, (7,7), sigmaX=5)

    # copy blurred version to the original image where your mask is > 0.
    return numpy.where(mask_img > 0, img_all_blurred, image)


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

    ap.add_argument('--cluster_min_faces', type=int, default=1,  help="number of neighbors necessary to be considered a detected face group")
    ap.add_argument('--cluster_eps_faces', type=int, default=50, help="eps to be considered a detected face group")

    ap.add_argument('--cluster_min_eyes', type=int, default=1, help="number of neighbors necessary to be considered a detected eye group")
    ap.add_argument('--cluster_eps_eyes', type=int, default=15, help="eps to be considered a detected eye group")
    args = ap.parse_args()


    image_name = "dogs.jpeg"#"dogpark.jpeg"#"classroom.jpeg"#
    original = cv2.imread(image_name)

    face_data = extract_face_regions(original)

    for data in face_data:
        face_image, (origin_x, origin_y), width, height = data
        cv2.rectangle(original, (origin_x, origin_y), (origin_x + width, origin_y + height), thickness=4, color=(0, 0, 255))

        resize_factor = 1500 // original.shape[0]
        eyes_blurred_image = blur_eye_regions(face_image, args, resize_factor)

        end_x = origin_x + width
        end_y = origin_y + height
        original[origin_y:end_y, origin_x:end_x] = eyes_blurred_image

    cv2.imshow(f'{image_name} {original.shape}', original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()