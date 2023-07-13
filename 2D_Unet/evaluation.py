import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt as edt

def dice_coefficient(pred_mask, mask,smooth=1e-10, n_classes=2):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(F.softmax(pred_mask, dim=1), dim=1) #shape=(20,244,244)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        dice_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas
            #print('class',true_class)
            #print('label',true_label)   
            intersect=torch.logical_and(true_class,true_label).sum().float().item()
            sum=true_class.sum().item()+true_label.sum().item()
            dice_co=2*intersect/sum
            dice_per_class.append(dice_co)
    return np.nanmean(dice_per_class)


class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])
            #pass

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.max(distances[indexes]))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
        ).float()

        #return torch.max(right_hd, left_hd)
        return (right_hd+left_hd)/2