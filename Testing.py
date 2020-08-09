from fastai.vision import *

def predict(model_path, image_tensor):
    defaults.device = torch.device('cpu')
    learn = load_learner(model_path)
    pred_class, pred_idx, outputs = learn.predict(Image(image_tensor))
    return pred_class.obj, outputs[pred_idx]

if __name__ == "__predict__":
    predict()