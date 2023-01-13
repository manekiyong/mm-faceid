import numpy as np
import torch
from typing import List

from mtcnn.manager import MTCNNManager
from triton.manager import TritonManager

class EmbeddingGenerator():
    def __init__(self):
        self.mtcnn = MTCNNManager()
        self.inference = TritonManager()

    def generate_embedding(self, img_b64_list: List):
        """
        Takes in a list of images parsed as base64 format and generate the average embedding of all the faces
        The assumption is that the image provided has only 1 distinct face within image to establish the identity of an individual

        Args:
            img_b64_list (list): Mapping to be checked

        Returns:
            int: 0 if there is invalid types, 1 otherwise

        """
        avg_emb = torch.zeros(512)
        img_count = 0  # Reset Image Count
        for i in img_b64_list:
            face_data = self.mtcnn.crop_faces_from_b64(i['image'])
            if len(face_data['img']) != 0:
                img_count += 1
                max_val = np.argmax(face_data['prob'])
                top_img = [face_data['img'][max_val]]
                img_embedding = self.inference.infer_with_triton(top_img)
                img_embedding = torch.from_numpy(img_embedding)
                avg_emb = avg_emb.add(img_embedding[max_val])
        if img_count == 0:
            return []
        avg_emb = avg_emb.div(img_count)
        return avg_emb.tolist()