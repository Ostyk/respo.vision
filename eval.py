import numpy as np
import seaborn as sns
from cf_matrix import make_confusion_matrix
from sklearn.metrics import confusion_matrix
sns.set_context('talk')
from tqdm import tqdm
from models import *
from dataset import *

class Evaluater(object):
    """
    Class to evaluate netowrk and plots graphs
    """
    
    def __init__(self, net, data_loader, categories, model_name, device):
        """
        :param: net -- network
        :param: data_loader -- pytorch data loader
        :param: categories -- class name string with dtype: list
        :param: model_name 
        :param: device
        """
        self.net = net
        self.data_loader = data_loader
        self.categories = categories
        self.model_name = model_name
        self.device = device
        
    def eval_trained_model(self):
        """
        Function that loads and evaluates trained models
        
        :return: y_pred, y_true
        """

        self.net.eval()
        y_pred = []
        y_true = []

        for inputs, labels in tqdm(self.data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                batch_pred = self.net(inputs)
                _, preds = torch.max(batch_pred, 1)

                y_pred.extend(preds.cpu().numpy().tolist())
                y_true.extend(labels.cpu().numpy().tolist())
                
        self.y_pred = y_pred
        self.y_true = y_true
    
    
    def confustion_matrix(self):
        
        cf_matrix = confusion_matrix(self.y_true , self.y_pred)
        labels = ['True Neg','False Pos','False Neg','True Pos']
        make_confusion_matrix(cf_matrix, group_names=labels,
                              categories=self.categories, cmap='binary',
                              title=f'Confusion Matrix of test set on {self.model_name} model')


def eval_model_test_set(model, test_path, batch_size, device):
    """
    Function that test a given model on a secret test set (no labels given) and retuns a csv with img_name, prediction
    :param model: pytorch model object
    :param test_path:
    :param batch_size:
    :param device:
    :return dataframe: || filename | probability of an instance being a match frame ||
    """
    
    model.eval()


    test_batch = np.zeros((batch_size, 3, 299, 299))
    filenames = []
    labels = []
    counter = 0
    for ind, img_name in tqdm(enumerate(os.listdir(test_path))):    
        if img_name.endswith('.jpg'):
            filenames.append(img_name)
            img_path = os.path.join(test_path, img_name)

            img = Image.open(img_path)
            img = img.resize((299,299), Image.ANTIALIAS)
            img = np.array(img)
            img = img * (1. / 255)
            img = np.moveaxis(img, -1, 0).astype(np.float32)

            test_batch[counter] = img
            counter += 1


            if counter==(batch_size):
                test_batch = torch.from_numpy(test_batch).float()
                test_batch = test_batch.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    batch_pred = model(test_batch)
                    _, preds = torch.max(batch_pred, 1)
                    m = nn.Softmax(dim=0)
                    softmax_output = m(batch_pred)
                    probs, preds = torch.max(softmax_output, 1)
                    #print(ind, preds)

                test_batch = np.zeros((batch_size, 3, 299, 299))
                counter = 0

                labels.extend(probs.cpu().numpy().tolist())
                
    s = pd.DataFrame([],columns = ['filename', 'probability'])
    s['filename'] = filenames
    s['probability'] = labels
    return s
