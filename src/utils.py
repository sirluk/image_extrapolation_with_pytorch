import os
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torchvision


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets, and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(2, 2)
    ax[1, 1].remove()
    
    for i in range(len(inputs)):
        ax[0, 0].clear()
        ax[0, 0].set_title('input')
        ax[0, 0].imshow(inputs[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[0, 0].set_axis_off()
        ax[0, 1].clear()
        ax[0, 1].set_title('targets')
        ax[0, 1].imshow(targets[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[0, 1].set_axis_off()
        ax[1, 0].clear()
        ax[1, 0].set_title('predictions')
        ax[1, 0].imshow(predictions[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[1, 0].set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=1000)
    del fig

    
def my_collate(batch):
    # batch contains a list of tuples of structure (input_array, known_array, target_array)
    x, y, mask, idx = zip(*batch)
    return torch.stack(x), pad_sequence(y, batch_first=True), torch.cat(mask), idx


def random_rotation_flip(x):
    """only works for square tensors"""
    img_dims = list(range(len(x.shape)))[-2:]
    tf_list = [
        lambda x: x,
        lambda x: torch.rot90(x, 1, dims=img_dims),
        lambda x: torch.rot90(x, 2, dims=img_dims),
        lambda x: torch.rot90(x, 3, dims=img_dims),
        lambda x: torch.flip(x, dims=(img_dims[0],)),
        lambda x: torch.flip(x, dims=(img_dims[1],))
    ]
    idx = torch.randint(0, len(tf_list), size=(1,)).item()
    return tf_list[idx](x)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, denorm_stats=(0,1), resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.denorm_mean, self.denorm_std = denorm_stats
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction, target, feature_layers=[0, 1, 2, 3]):
        prediction = prediction * self.denorm_std + self.denorm_mean
        target = target * self.denorm_std + self.denorm_mean
        if prediction.shape[1] != 3:
            prediction = prediction.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        prediction = (prediction-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            prediction = self.transform(prediction, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        for i, block in enumerate(self.blocks):
            prediction = block(prediction)
            target = block(target)
            if i in feature_layers:
                loss += nn.functional.l1_loss(prediction, target)
        return loss
    
    
def get_stats(filepath):
    ar = np.asarray(Image.open(filepath), dtype=np.float32)
    ar /= 255
    return np.array([ar.sum(), np.square(ar).sum(), ar.size])    