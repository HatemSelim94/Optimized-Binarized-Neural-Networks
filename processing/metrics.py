import numpy as np

from .helpers import labels, category2labels
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

import timeit

class Timer:
    def __init__(self) -> None:
        self.start_time = 0
        self.time = 0
        self.iter = 0
    def start(self):
        self.start_time = timeit.default_timer()
    def end(self):
        self.end_time = timeit.default_timer()
        self.time += self.end_time - self.start_time
        self.iter += 1
    def reset(self):
        self.time = 0
        self.iter = 0
    def mean(self):
        if self.iter:
            return self.time/self.iter
        else:
            return None
    def fps(self):
        return 1/self.mean()

    def __repr__(self):
        return str(f'{self.time/60:.2f} min')


class RoadMetrics():
    def __init__(self, n_cls=2):
        self.n_classes
        


#https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/metrics/stream_metrics.py
class SegMetrics():
    """
    Metrics for Semantic Segmentation 
    """
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, true_labels, predicted_labels):
        # form a confusion matrix from a batch
        for lt, lp in zip(true_labels, predicted_labels):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %.2f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        """construct confusion matrix

        Args:
            label_true (numpy array): True label
            label_pred (numpy array): Predicted label

        Returns:
            [numpy array]: Confusion matrix of size n*n. n is the number of classes
        """
        mask = (label_true >= 0) & (label_true < self.n_classes) # filter ignored labels if there are any 
        #print(f'mask: {mask} end')
        #print(self.n_classes * label_true[mask].astype(int)+ label_pred[mask])
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes) # count the number of occurences of of each value. 
        return hist
    

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IoU
            - fwavacc
            - dice coefficient
        """
        np.seterr(divide='ignore', invalid='ignore')
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_per_cls = np.diag(hist) / hist.sum(axis=1)
        mean_cls_acc = np.nanmean(acc_per_cls) 
        # using nanmean to filter NAN values that are result of dividing by zero
        # (classes that do not exist in both true and predicted images)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum() #frequency weighted average
        cls_iou = dict(zip(range(self.n_classes), iou))
        #avg_dice = 2*np.diag(hist)/(hist.sum(axis=1)+hist.sum(axis=0)+np.finfo(float).eps)
        #print(avg_dice)
        avg_dice = 2*mean_iou/(1+mean_iou) # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        np.seterr(divide='warn', invalid='warn')
        return {
                "Overall Pixel Acc": acc,
                "Mean Acc": mean_cls_acc,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iou,
                "Class IoU": cls_iou,
                "Dice Coefficient": avg_dice
            }
    def get_road_metrics(self):
        classes= {'Background':0, 'Road':1}
        hist = self.confusion_matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            pre = np.diag(hist)/hist.sum(axis=0)
            pre = pre[classes['Road']] 
            rec = np.diag(hist)/hist.sum(axis=1)
            rec = rec[classes['Road']]
            f1_score = 2*(pre*rec)/(pre+rec)
            iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            iou = iou[classes['Road']]
        return {'Pre':pre, 'Rec':rec, 'F1_score':f1_score, 'iou':iou}
    
    def get_labels_iou(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            hist = self.confusion_matrix
            iou = np.diag(hist)/(hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            mean_iou = np.nanmean(iou)
            cls_iou = dict(zip(range(self.n_classes), iou))
        return cls_iou, mean_iou
    
    def get_category_iou(self):
        catids = {'flat':[0,1],'construction':[2,3,4],'object':[5,6,7],'nature':[8,9],'sky':[10],'human':[11,12],'vehicle':[13,14,15,16,17,18]}
        cat_iou = {}
        cat_miou = []
        with np.errstate(divide='ignore', invalid='ignore'):
            for cat in catids.keys():
                lblids = catids[cat]
                hist = self.confusion_matrix
                tp = np.longlong(hist[lblids,:][:,lblids].sum())
                fn = np.longlong(hist[lblids,:].sum()) - tp
                notIgnoredAndNotInCategory = list(range(self.n_classes))
                for id in lblids:
                    notIgnoredAndNotInCategory.remove(id)
                fp = np.longlong(hist[notIgnoredAndNotInCategory,:][:,lblids].sum())
                iou = tp/(tp+fp+fn)
                cat_iou[cat] = iou
                cat_miou.append(iou)
        return cat_iou, np.nanmean(cat_miou)

    
    def get_iou(self):
        '''
            returns:
                mean_iou (float): mean intersection over union 
                cls_iou (dict): intersection over union per class
        '''
        np.seterr(divide='ignore', invalid='ignore')
        hist = self.confusion_matrix
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        cls_iou = dict(zip(range(self.n_classes), iou))
        np.seterr(divide='warn', invalid='warn')
        return mean_iou, cls_iou

    def plot_confusion_matrix(self):
        names = self.get_names()
        assert len(names) == self.n_classes
        confusion_matrix_df = pd.DataFrame(data=self.confusion_matrix, 
        index = names, columns=names)
        plt.figure(figsize=(len(names), len(names)))
        sns.heatmap(confusion_matrix_df, cmap=sns.cm.rocket) # sbs.cm.rocket
        plt.show()
    
    def get_names(self):
        if self.n_classes == 3:
            names = ['Background', 'Road', 'Vehicles']
        elif self.n_classes == 8:
            names = ['Void', 'Flat', 'Construction', 'Object','Nature','Sky','Human','Vehicles']
        else:
            raise NotImplementedError
        return names
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

# ROAD

def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou
    
if __name__ == '__main__':
    import torch
    classes_num = 8
    metric = SegMetrics(classes_num)
    true_labels = torch.randint(low = 0, high=classes_num, size=[1,1,20,20]).cpu().numpy()
    predicted_labels = torch.randint(low = 0, high=classes_num, size=[1,1,20,20]).cpu().numpy()
    metric.update(true_labels, predicted_labels)
    mean_iou,cls_iou = metric.get_iou()
    print(cls_iou)
    print(mean_iou, cls_iou[classes_num-1])
    metric.plot_confusion_matrix()