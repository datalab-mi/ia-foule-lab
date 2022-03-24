import os
import time

import numpy as np
import onnxruntime
from PIL import Image


class Mobilecount:
    """Class for MobileCount Inference (Crowd Counting Model Inference)"""

    authorized_extension = ['onnx']

    def __init__(self, model_path=None):

        self.model_filename = os.path.basename(model_path)
        self.model_extension = os.path.splitext(os.path.basename(model_path))[-1][1:]
        self.model_type = os.path.splitext(os.path.basename(model_path))[0].split('_')[0]

        self.initialize_parameters()

        self.model_session = None
        self.error_message = 'OK'

        if self.model_type != 'mobilecount':

            self.error_message = "The model is not a 'MobileCount' model : {}".format(self.model_filename)

        elif self.model_extension not in Mobilecount.authorized_extension:

            self.error_message = "'Mobilecount' extension not yet implemented : {}".format(self.model_extension)

        elif self.model_extension == 'onnx':

            try:
                self.model_session = onnxruntime.InferenceSession(str(model_path))
            except Exception as e:
                self.error_message = 'Onnx runtime session cannot be initialize : {}'.format(e)

    def initialize_parameters(self):
        """Inialization of parameters used to preprocess image before prediction"""

        self.mean = np.array([0.452016860247, 0.447249650955, 0.431981861591])
        self.mean = np.float64(self.mean.reshape(1, -1))
        self.std = np.array([0.23242045939, 0.224925786257, 0.221840232611])
        self.stdinv = 1 / np.float64(self.std.reshape(1, -1))

    def predict(self, img: Image):
        """Crowd Couting Inference/Prediction
        , give nb of person and density map for a given image

        :param img: PIL Image
        :return (tuple) : - int(nb_person) : nb of person predicted on the image
                          - density_map : density map associated to the prediction
                          - prediction_time : Inference time in milliseconds
        """
        start_time = time.time()

        if self.model_session is None:
            print("You cannot use the 'predict' function - Error : " + self.error_message)
            return None, None, None

        if (img.mode in ['BGR','RGBA','P']):
            img = img.convert('RGB')

        img = np.asarray(img)
        img = (img / 255. - self.mean) * self.stdinv
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)  # batch size

        ort_inputs = {self.model_session.get_inputs()[0].name: img}
        ort_outs = self.model_session.run(None, ort_inputs)
        density_map = ort_outs[0]
        density_map = np.squeeze(density_map, axis=(0, 1))
        nb_person = density_map.sum()

        # Time in millisecondes
        prediction_time = round((time.time() - start_time) * 1000)

        return int(nb_person), density_map, prediction_time

    def close_session(self):
        """
        Close the session used to infer
        :return: None
        """

        del self.model_session
        self.model_session = None
