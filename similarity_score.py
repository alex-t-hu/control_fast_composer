from PIL import Image
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet

# extract a single face from a given photograph
def extract_faces(detector, filename, required_size=(160, 160)):
    image = Image.open(filename) # load image from file
    image = image.convert('RGB') # convert to RGB, if needed
    pixels = np.asarray(image) # convert to array
    results = detector.detect_faces(pixels) # detect faces in the image
    ret = []
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box'] # extract the bounding box from the first face
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        ret.append(face_array)
    return ret

def similarity(x1, x2, mode='cosine'):
    if mode == 'cosine':
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    assert mode == 'l2'
    return np.linalg.norm(x1 - x2)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    # mean, std = face_pixels.mean(), face_pixels.std()
    # face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]

def similarity_score(image_path_1, image_path_2):
    detector = MTCNN() # create the detector, using default weights
    model = FaceNet()
    faces1 = extract_faces(detector, image_path_1)
    faces2 = extract_faces(detector, image_path_2)
    assert len(faces1) == len(faces2), 'Number of faces in both images should be same'

    if len(faces1) == 1:
        embedding1 = get_embedding(model, faces1[0])
        embedding2 = get_embedding(model, faces2[0])
        score = similarity(embedding1, embedding2)
        return score
    else:
        assert len(faces1) == 2

        embeddings1 = [get_embedding(model, faces1[0]), get_embedding(model, faces1[1])]
        embeddings2 = [get_embedding(model, faces2[0]), get_embedding(model, faces2[1])]

        score1 = similarity(embeddings1[0], embeddings2[0]) + similarity(embeddings1[1], embeddings2[1])
        score2 = similarity(embeddings1[0], embeddings2[1]) + similarity(embeddings1[1], embeddings2[0])

        if score1 > score2:
            return score1 / 2
        else:
            return score2 / 2

if __name__ == "__main__":
    generated_image_path = './outs/hypertune_single_reference_dancing1/poses/alpha0.0/output_image.png'
    reference_image_path = './data/reference/einstein/einstein.jpeg'
    score = similarity_score(generated_image_path, reference_image_path)
    print(score)