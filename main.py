import os
import numpy as np
import torch
import torch.utils.data
from src.dataset import ImageDS, BorderPredictionDS, denorm
from src.utils import plot, my_collate, random_rotation_flip
from src.architecture import CNNBase, BorderPredictionNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import dill as pickle


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, ds_stats: tuple):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # get dataset mean and std
    mean, std = ds_stats
    img_min, img_max = - mean / std, (1 - mean) / std 
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets, mask, file_names = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            
            # Get outputs for network
            outputs = model(inputs, mask)
            
            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance
            outputs = torch.clamp(outputs, img_min, img_max)

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            loss += torch.stack([mse(output, target) for output, target in zip(outputs, targets)]).sum()
        loss /= len(dataloader.dataset)

    return loss


def main(results_path, network_config: dict, learningrate: int = 1e-3, batch_size: int = 16, weight_decay: float = 1e-5,
         n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Prepare a path to plot to
    plotpath = os.path.join(results_path, 'plots')
    os.makedirs(plotpath, exist_ok=True)
    
    # Load dataset
    p = "data/data_train"
    ds = ImageDS(p)
    
    # load mean and std
    with open("data/image_stats.pkl", "rb") as f:
        mean, std = pickle.load(f)

    # Split dataset into training, validation, and test set randomly
    trainingset = torch.utils.data.Subset(ds, indices=np.arange(int(len(ds)*(4/5))))
    validationset = torch.utils.data.Subset(ds, indices=np.arange(int(len(ds)*(4/5)),len(ds)))

    # Create datasets and dataloaders with rotated targets without augmentation (for evaluation)
    tf_eval = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((90,90))
            ])
    trainingset_eval = BorderPredictionDS(dataset=trainingset, ds_stats = (mean, std), transform_chain=tf_eval, border_mode="fix")
    validationset = BorderPredictionDS(dataset=validationset, ds_stats = (mean, std), transform_chain=tf_eval, border_mode="fix")
    trainloader = DataLoader(trainingset_eval, batch_size=1, shuffle=False, num_workers=0, collate_fn=my_collate)
    valloader = DataLoader(validationset, batch_size=1, shuffle=False, num_workers=0, collate_fn=my_collate)
    
    # Create datasets and dataloaders with rotated targets with augmentation (for training)
    tf_aug = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop((90,90)),
                transforms.Lambda(lambda x: random_rotation_flip(x))
            ])
    trainingset_augmented = BorderPredictionDS(dataset=trainingset, ds_stats = (mean, std), transform_chain=tf_aug, border_mode="rand")
    trainloader_augmented = DataLoader(trainingset_augmented, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=my_collate)
    
    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))
    
    # Create Network   
    cnn = CNNBase(**network_config)
    #cnn = CNNBaseMulti(**network_config)
    net = BorderPredictionNet(cnn)
    net.to(device)
    
    # Get mse loss function
    mse = torch.nn.MSELoss()
    
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)
    
    print_stats_at = 1e2  # print status to tensorboard every x updates
    plot_at = 1e4  # plot every x updates
    validate_at = 5e3  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))
    
    # Train until n_updates update have been reached
    while update < n_updates:
        for data in trainloader_augmented:
            # Get next samples in `trainloader_augmented`
            inputs, targets, mask, ids = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            
            # Reset gradients
            optimizer.zero_grad()

            # Get outputs for network
            outputs = net(inputs, mask)
            
            # Calculate loss, do backward pass, and update weights
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)
            
            # Plot output
#             if update % plot_at == 0:
#                 plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
#                      plotpath, update)
            
            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=valloader, device=device, ds_stats=(mean, std))
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))
                    with open(os.path.join(results_path, "best_validation_loss.txt"), "w") as f:
                        f.write("Best validation loss: %s" % best_validation_loss)
            
            update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progess_bar.update()
            
            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break

    update_progess_bar.close()
    print('Finished Training!')
    
    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    val_loss = evaluate_model(net, dataloader=valloader, device=device, ds_stats=(mean, std))
    train_loss = evaluate_model(net, dataloader=trainloader, device=device, ds_stats=(mean, std))
    
    print(f"Scores:")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")
    
    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        print(f"training loss: {train_loss}", file=fh)


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file
    
    with open(config_file, 'r') as fh:
        config = json.load(fh)
    main(**config)
