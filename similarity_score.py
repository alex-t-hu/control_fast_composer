from PIL import Image
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet
import clip
import torch

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
        cos = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return (cos + 1) / 2
    assert mode == 'l2'
    return np.linalg.norm(x1 - x2)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    # mean, std = face_pixels.mean(), face_pixels.std()
    # face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]

def similarity_score(detector, model, image_path_1, image_path_2, score_mode='l2'):
    faces1 = extract_faces(detector, image_path_1)
    faces2 = extract_faces(detector, image_path_2)
    if len(faces1) == 0 or len(faces2) == 0:
        return 1
    # assert len(faces1) == len(faces2), f'Number of faces in both images should be same but faces1 has {len(faces1)} and faces2 has {len(faces2)}. image paths {image_path_1}, {image_path_2}'
    faces1 = faces1[0]
    faces2 = faces2[0]

    embedding1 = get_embedding(model, faces1)
    embedding2 = get_embedding(model, faces2)
    score = similarity(embedding1, embedding2, score_mode)
    return score

def similarity_score_double(detector, model, image_path_1, ref_path_1, ref_path_2, score_mode='l2'):
    faces1 = extract_faces(detector, image_path_1)
    ref1 = extract_faces(detector, ref_path_1)
    ref2 = extract_faces(detector, ref_path_2)
    if len(faces1) < 2 or len(ref1) == 0 or len(ref2) == 0:
        return 1
    # assert len(faces1) == len(faces2), f'Number of faces in both images should be same but faces1 has {len(faces1)} and faces2 has {len(faces2)}. image paths {image_path_1}, {image_path_2}'
    faces1 = faces1[:2]
    faces2 = [ref1[0], ref2[0]]


    embeddings1 = [get_embedding(model, faces1[0]), get_embedding(model, faces1[1])]
    embeddings2 = [get_embedding(model, faces2[0]), get_embedding(model, faces2[1])]

    score1 = similarity(embeddings1[0], embeddings2[0], score_mode) + similarity(embeddings1[1], embeddings2[1], score_mode)
    score2 = similarity(embeddings1[0], embeddings2[1], score_mode) + similarity(embeddings1[1], embeddings2[0], score_mode)

    if score_mode == 'cosine':
        return max(score1, score2) / 2
    else:
        return min(score1, score2) / 2


def get_clip_score(image_path, text):
    # Load the pre-trained CLIP model and the image
    model, preprocess = clip.load('ViT-B/32')
    image = Image.open(image_path)

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score

if __name__ == "__main__":
    detector = MTCNN() # create the detector, using default weights
    model = FaceNet()
    generated_image_path = './outs/hypertune_single_reference_dancing1/poses/alpha0.0/output_image.png'
    reference_image_path = './data/reference/einstein/einstein.jpeg'
    score = similarity_score(detector, model, generated_image_path, reference_image_path)
    print(score)