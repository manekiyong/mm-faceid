import requests
import io
import base64
import json
import yaml

from mtcnn.model.mtcnn import MTCNN
from PIL import Image

def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()
URL = config['MTCNN']['URL']


class MTCNNManager():

    def __init__(self):
        # self.path = URL+'/crop'
        # self.header = {'Content-type': 'application/json',
        #                'Accept': 'text/plain'}
        self.mtcnn = MTCNN(image_size=160, margin=0, device='cuda:0', keep_all=True)

    def parse_results(self, im_bytes):
        image = Image.open(io.BytesIO(im_bytes)).convert('RGB')
        img_tensor, prob, box, = self.mtcnn(image, return_prob=True)
        res_dict = {}
        if prob[0] == None:
            res_dict['img'] = []
            res_dict['prob'] = []
            res_dict['box'] = []
        else:
            img_list = img_tensor.tolist()
            prob_list = prob.tolist()
            box_list = box.tolist()
            pred_len = len(prob_list)
            for i, conf in enumerate(reversed(prob_list)):
                # Clear faces usually easily obtain conf of 0.99++
                if conf < 0.9:  # Eliminate low confidence faces detected.
                    pop_index = pred_len-i-1
                    img_list.pop(pop_index)
                    box_list.pop(pop_index)
                    prob_list.pop(pop_index)
            res_dict['img'] = img_list
            res_dict['prob'] = prob_list
            res_dict['box'] = box_list
        return res_dict


    def crop_faces_from_path(self, img_path):
        """        
        INPUT:
        ------------------------------------
        img_path:   Path to .jpg file

        RETURNS:
        ------------------------------------
        res:    Result dict comprising of:
                - res['img']:   normalised image in the form of NCHW or NCWH i dont really know lol
                                example shape: [N, 3, 224, 224]
                - res['box']:   bounding box of faces
                                example shape: [N, 4]
                - res['prob']:  confidence score of each face cropped
                                example shape: [N, 1]
                where N is the number of faces found
        """
        with open(img_path, "rb") as f:
            im_bytes = f.read()
        res_dict = self.parse_results(im_bytes)
        # im_b64 = base64.b64encode(im_bytes).decode("utf8")
        # payload = json.dumps({"image": im_b64})
        # r = requests.post(self.path, data=payload, headers=self.header)
        
        return res_dict

    def crop_faces_from_b64(self, img_bytes):
        """        
        This method is for FastAPI to directly pass the dict forward to MTCNN, instead of reading and re-wrapping the image. 
        INPUT:
        ------------------------------------
        img_bytes:  Image dict in the format of:
                - img_bytes['image']:   Original image in b64 format

        RETURNS:
        ------------------------------------
        res:    Result dict comprising of:
                - res['img']:   normalised image in the form of NCHW or NCWH i dont really know lol
                                example shape: [N, 3, 224, 224]
                - res['box']:   bounding box of faces
                                example shape: [N, 4]
                - res['prob']:  confidence score of each face cropped
                                example shape: [N, 1]
                where N is the number of faces found
        """
        res = self.parse_results(img_bytes)
        return res
