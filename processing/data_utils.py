import matplotlib.pyplot as plt
import torch
from .helpers import labels

modified_labels = labels.copy()
modified_labels.reverse()

def show_sample(images, smnts):
    """show images
    Args:
        images (Any): Images 
        smnts  (Any): Targets 
    """
    batch: bool = (len(images.shape) == 4)
    if batch:
        n = len(images)
        fig, axs = plt.subplots(2, n, sharex='all', sharey='all')
        if torch.is_tensor(images[0]):
            for i in range(n):
                axs[0, i].imshow(images[i].permute(1, 2, 0))
                axs[1, i].imshow(smnts[i].permute(1, 2, 0)
                                 if smnts[i].shape[0] == 3 else smnts[i])
        else:
            for i in range(n):
                axs[0, i].imshow(images[i])
                axs[1, i].imshow(smnts[i])
    else:
        fig, axs = plt.subplots(2, 1, sharex='all', sharey='all')
        axs[0].imshow(images.permute(1, 2, 0)
                      if torch.is_tensor(images) else images)
        axs[1].imshow(smnts.permute(1, 2, 0) if (
            torch.is_tensor(smnts) and smnts.shape[0] == 3) else smnts)

    plt.show()


# https://www.programmersought.com/article/23915498478/
def decode_segmap(image, no_of_classes=34):
    modes = {'default':34, 'train':20, 'category':8, 'custom':3} # for readability
    assert no_of_classes == 3 or no_of_classes == 8 or no_of_classes == 20 or no_of_classes == 34
    r = torch.zeros_like(image, dtype=torch.uint8)
    g = torch.zeros_like(image, dtype=torch.uint8)
    b = torch.zeros_like(image, dtype=torch.uint8)
    if modes['default'] == no_of_classes:
        trainId2label = {label.id: label for label in labels}
    elif modes['train'] == no_of_classes:
        trainId2label = {label.trainId: label for label in labels}
    elif modes['category']== no_of_classes:
        trainId2label = {label.categoryId: label for label in labels}
    elif modes['custom']== no_of_classes:
        trainId2label = {label.customId: label for label in modified_labels}
    if no_of_classes==20:
        no_of_classes=19
    for class_id in range(no_of_classes):
        idx = (image == class_id)
        r[idx] = trainId2label[class_id].color[0]
        g[idx] = trainId2label[class_id].color[1]
        b[idx] = trainId2label[class_id].color[2]
    rgb = torch.stack([r, g, b], axis=2)
    return rgb


def show_batch_sample(images, targets, no_of_classes):
    if len(images.shape) == 4:  # batch
        fig, axs = plt.subplots(2, images.shape[0], sharex='all', sharey='all')
        for i in range(len(images)):
            axs[0, i].imshow(images[i].permute(1, 2, 0))
            axs[1, i].imshow(decode_segmap(targets[i], no_of_classes))
    else:
        fig, axs = plt.subplots(2, 1, sharex='all', sharey='all')
        axs[0].imshow(images.permute(1, 2, 0))
        axs[1].imshow(decode_segmap(targets, no_of_classes))
    plt.show()


def show_validation_sample(image, target, prediction, SegMetrics, no_of_classes):
    metric = SegMetrics(no_of_classes)
    metric.update(target.cpu().numpy(), prediction.cpu().numpy())
    results = metric.get_results()
    metrics_report = SegMetrics.to_str(results)
    fig, axs = plt.subplots(1, 3, sharex='all', sharey='all')
    axs[0].imshow(image.cpu().squeeze().permute(1, 2, 0))
    axs[0].set_title('Image')
    axs[1].imshow(decode_segmap(target.cpu().squeeze(), no_of_classes))
    axs[1].set_title('Target')
    axs[2].imshow(decode_segmap(prediction.cpu().squeeze(), no_of_classes))
    axs[2].set_title('Prediction')
    axs[2].text(0.5, 0.1, metrics_report, horizontalalignment='center',
    verticalalignment='center',size=8, ha='center',
    transform=axs[2].transAxes, bbox=dict(facecolor='white', alpha=0.3))
    plt.show()

def save_results(labels, predictions, file_name, no_of_classes=3):
    for id, (label, prediction) in enumerate(zip(labels, predictions)):
        plt.imsave(f'results/{file_name}_label_{id}.png', 
                    decode_segmap(label.cpu().squeeze(),no_of_classes).numpy())
        plt.imsave(f'results/{file_name}_prediction_{id}.png',
                    decode_segmap(prediction.cpu().squeeze(),no_of_classes).numpy())