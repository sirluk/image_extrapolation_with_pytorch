import os
from matplotlib import pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence


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


def random_rotation_flip(x, img_dims=(1,2)):
    """only works for square tensors"""
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