from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from embeddings.embedding import EmbeddingGenerator

emb_generator = EmbeddingGenerator()


class Image(BaseModel):
    image: str


class ImageFolder(BaseModel):
    id: int
    images: List[Image]


api = FastAPI()

@api.post("/embedding")
def embedding(img_list: ImageFolder):
    """
    Takes in an ImageFolder object, generates an embedding and saves it locally.
    """
    img_dict = img_list.dict()
    emb = emb_generator.generate_embedding(img_dict['images'])
    if emb == []:
        return {"response": "No embedding generated"}
    else:
        return {"response": "200", "emb": emb}
