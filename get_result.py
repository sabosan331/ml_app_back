import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time, os
import copy
import random as rd
from torch.autograd import Variable
from PIL import Image

seed = 11
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(seed)
rd.seed(seed)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def resnet_test(dataset_id=0, model_id=0):
    
    # print(device)
    batch_size = 64

    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop((280,400) ),
            transforms.Resize( (224,224) ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        'test': transforms.Compose([
            transforms.CenterCrop((280,400) ),
            transforms.Resize( (224,224) ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    data_paths = ["/home/user/data/bearing/2class/tokushima1_train_test/"]

    import glob
    train_dir = data_paths[dataset_id] + "/train/"
    train_ok = glob.glob( train_dir + "/0_ok/*.jpg" )
    train_ng = glob.glob( train_dir + "/1_ng/*.jpg" )
    test_dir = data_paths[dataset_id] + "test/"
    test_ok = glob.glob( test_dir + "/0_ok/*.jpg" )
    test_ng = glob.glob( test_dir + "/1_ng/*.jpg" )

    exp_data = { "train_ok": len(train_ok),
                "train_ng": len(train_ng),
                "test_ok" : len(test_ok),
                "test_ng" : len(test_ng) }

    model_path = "./models/"
    model_name = ["resnet_with_crop_ver1.0.model"]
    model = torch.load(model_path + model_name[model_id])
    model.eval()
    print(model)

  

    test_dir = "/home/user/data/bearing/2class/tokushima/"
    test_datasets = ImageFolderWithPaths( root= test_dir , transform = data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1 ,shuffle=False, num_workers=1)

    model.eval() 
    correct = 0
    test_loss = 0.0; test_acc = 0.0

    criterion = nn.BCELoss()
    t = Variable(torch.Tensor([0.4]))
    t = t.to(device)

    y_test = []
    y_pred = []

    out_ng = []
    out_ok = []
    preds = []

    cnt = 0
    for data, target, paths in test_loader:
        data = data.to(device); target = target.to(device) #data, target = Variable(data), Variable(target)  # 微分可能に変換

        output = model(data)  # 入力dataをinputし、出力を求める
        output = torch.nn.Sigmoid()(output)
        
        
        preds = (output > t).float() * 1    
        loss = criterion(output,target.float())
    #     print(output.item())
        
        if target.item() == 0:
            out_ok.append(output.cpu().detach().numpy() )
        elif target.item() == 1:
            out_ng.append(output.cpu().detach().numpy() )
            
        y_pred.append(output.item())
        y_test.append(target.item())
        
        test_loss += loss.item()
        test_acc += preds.eq(target.float().data.view_as(preds)).sum()  # 正解と一緒だったらカウントアップ
        
        img = Image.open(paths[0])
        prob = np.round(output.item(), decimals=3)
        print(prob)
        # plt.imsave( "./res/sort_by_prob/" + str(prob) + ".png",img)

        cnt = cnt + 1
        # if cnt > 500: break

    test_loss = test_loss / len(test_loader.dataset)
    test_acc  = 100. *  test_acc / len(test_loader.dataset)

    print(test_loss, test_acc)


    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    from sklearn.metrics import roc_curve


    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    f1_scores = []
    for thresh in thresholds:
        f1_scores.append(f1_score(y_test, [1 if m > thresh else 0 for m in y_pred]))

    f1_scores = np.array(f1_scores)
    max_f1 = f1_scores.max() 
    max_f1_threshold =  thresholds[f1_scores.argmax()]

    print(max_f1,max_f1_threshold)

    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


    # th = np.min(out_ok)
    # th = 0.1
    th = max_f1_threshold

    bin_pred = [1 if p >= th else 0 for p in y_pred]
    m = confusion_matrix(y_test, bin_pred,labels=[1,0])

    acc = 100 * round( accuracy_score(y_test, bin_pred) ,4)
    rec = 100 * round( recall_score(y_test, bin_pred),4)
    prec = 100 * round( precision_score(y_test, bin_pred),4)
    f1 = 100 * round( f1_score(y_test, bin_pred),4)


    print('Confution Matrix:\n{}'.format(m))

    print("threshold:{:.3f}".format(th))
    print("--------------------------------------------------------------------------")
    print('Accuracy:{:.3f}'.format(acc))
    print('Recall:{:.3f}'.format(rec))
    print('Precision:{:.3f}'.format(prec))
    print('F1-measure:{:.3f}'.format(f1_score(y_test, bin_pred)))
    print("--------------------------------------------------------------------------")
    print("未検出率(FNR):{:.3f} (must be 0.0)".format( m[0,1]/(m[0,1] + m[0,0]) ) )
    print("過検出率(FPR):{:.3f}".format(m[1,0]/(m[1,0] + m[1,1])))
    print("stupit未検出率:{:.3f} (になるようにThreshold決定)".format( m[0,1]/( np.sum(m) ) ) )
    print("stupit過検出率:{:.3f}".format(m[1,0]/(np.sum(m))))
    print("--------------------------------------------------------------------------")

    return exp_data,m, acc, rec, prec, f1
    # return exp_data
