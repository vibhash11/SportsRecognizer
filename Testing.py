from fastai.vision import *

def predict(model_path, filePath):
    defaults.device = torch.device('cpu')
    img = open_image(Path(filePath))
    learn = load_learner(model_path)
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class.obj, outputs[pred_idx]

if __name__ == "__predict__":
    predict()