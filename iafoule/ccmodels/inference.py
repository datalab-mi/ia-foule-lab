import importlib
import os

from PIL import Image


class CCModelInference:
    """Class for Crown Counting Model Inference"""

    ccmodels = ['mcnn', 'dsnet', 'mobilecount']

    def __init__(self, model_path=None):

        self.model_path = str(model_path)
        self.model_directory = os.path.dirname(self.model_path)
        self.model_filename = os.path.basename(self.model_path)
        tmp = os.path.splitext(self.model_filename)
        self.model_type = tmp[0].split('_')[0].lower()
        self.model_extension = os.path.splitext(self.model_filename)[-1][1:]

        module_name = self.model_type
        self.class_name = self.model_type.capitalize()

        self.model_session = None
        self.error_message = 'OK'

        if self.model_type not in CCModelInference.ccmodels:
            self.error_message = 'This model doesnot exists : {}'.format(self.model_type)
            return

        try:
            module = importlib.import_module("ccmodels." + module_name)
            class_ = getattr(module, self.class_name)
            self.ccmodel = class_(model_path=model_path)

            self.model_session = self.ccmodel.model_session
            self.error_message = self.ccmodel.error_message
        except Exception as e:
            self.error_message = 'Impossible to instanciate the model : {}'.format("ccmodels." + module_name)

    def __str__(self):
        res = "Class " + self.class_name + "\n"
        res += "model_path:{} \n".format(self.model_path)
        res += "model_directory:{} \n".format(self.model_directory)
        res += "model_filename:{} \n".format(self.model_filename)
        res += "model_extension:{} \n".format(self.model_extension)
        res += "model_type:{} \n".format(self.model_type)
        res += "model_session:{} \n".format(self.model_session)
        res += "error_message:{} \n".format(self.error_message)
        return res

    def predict(self, img: Image):
        """Crowd Couting Inference/Prediction
        , give nb of person and density map for a given image

        :param img: PIL Image
        :return (tuple) : - int(nb_person) : nb of person predicted on the image
                          - density_map : density map associated to the prediction
                          - prediction_time : Inference time in milliseconds
        """
        return self.ccmodel.predict(img)

    def close_session(self):
        """
        Close the session used to infer
        :return: None
        """
        return self.ccmodel.close_session()
