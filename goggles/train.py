try:
    from itertools import ifilter
except ImportError:
    ifilter = filter
import sys

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from constants import *
sys.path.append(BASE_DIR)
from goggles.data.cub.dataset import CUBDataset
from goggles.loss import CustomLoss2
from goggles.models.semantic_ae import SemanticAutoencoder
from goggles.utils.vis import \
    get_image_from_tensor, save_prototype_patch_visualization


_make_cuda = lambda x: x.cuda() if torch.cuda.is_available() else x

ex = Experiment('goggles-experiment')
ex.observers.append(FileStorageObserver.create(os.path.join(ALL_RUNS_DIR)))


def _provision_run_dir(run_dir):
    new_dirs = [LOGS_DIR_NAME, IMAGES_DIR_NAME, PROTOTYPES_DIR_NAME]
    new_dirs = list(map(lambda d: os.path.join(run_dir, d), new_dirs))
    for new_dir in new_dirs:
        os.makedirs(new_dir)
    return new_dirs


@ex.config
def default_config():
    seed = 42                # RNG seed for the experiment
    filter_class_ids = None  # Class IDs used for training, uses all classes if None
    input_image_size = 128   # All images are resized to this value
    patch_size = 1           # Size of the patch that is flattened from the encoded output
    batch_size = 64          # Batch size used for training
    num_epochs = 25000       # Number of epochs to run the training for
    loss_lambda = 0.01       # Value of lambda used in the custom loss for prototype similarity


@ex.main
def main(_run, _log,
         seed,
         filter_class_ids,
         input_image_size,
         patch_size,
         batch_size,
         num_epochs,
         loss_lambda):

    # Set the RNG seed for torch
    torch.manual_seed(seed)

    # Check input parameters are in expected format
    assert filter_class_ids is None or type(filter_class_ids) is list
    if type(filter_class_ids) is list:
        assert all(type(class_id) is int for class_id in filter_class_ids)
    else:
        _log.warning('Training on all classes!!')
        confirm = input('Continue? [y/n] ')
        if confirm.lower() != 'y':
            return None

    # Provision the `sacred` run directory for this experiment
    RUN_DIR = _run.observers[0].dir
    LOGS_DIR, IMAGES_DIR, PROTOTYPES_DIR = _provision_run_dir(RUN_DIR)

    # Initialize log writer for tensorboard
    writer = SummaryWriter(LOGS_DIR)

    # Load datasets for training and testing
    train_dataset, train_dataset_with_non_random_transformation, \
        test_dataset = CUBDataset.load_dataset_splits(
        CUB_DATA_DIR, input_image_size, filter_class_ids)

    # Initialize the data loader
    train_dataloader = DataLoader(
        train_dataset, collate_fn=CUBDataset.custom_collate_fn,
        batch_size=batch_size, shuffle=True)

    # Define variables for attributes
    num_attributes = train_dataset.num_attributes
    all_attribute_labels = range(1, num_attributes + 1)
    attribute_names = [train_dataset.get_attribute(al).name
                       for al in all_attribute_labels]

    # Initialize the model
    model = _make_cuda(SemanticAutoencoder(
        input_image_size, patch_size, num_attributes))

    # Initialize the loss function and optimizer
    epoch_loss = None
    criterion = _make_cuda(CustomLoss2(lambda_val=loss_lambda))
    optimizer = optim.Adam(ifilter(lambda p: p.requires_grad,
                                   model.parameters()))

    # Initiate training
    pbar, steps = tqdm(range(1, num_epochs + 1)), 0
    for epoch in pbar:
        epoch_loss = 0.

        model.train()  # Setting the model in training mode for training
        for image, label, attribute_labels, padding_idx in train_dataloader:
            steps += 1  # Incrementing the global step
            model.zero_grad()  # Clearing the gradients for each mini-batch

            # Create the input variable and get the output from the model
            x = _make_cuda(torch.autograd.Variable(image))
            z, z_patches, reconstructed_x = model(x)

            # Get the associated prototypes for each image in the batch
            prototype_labels = _make_cuda(attribute_labels)
            positive_prototypes = model.prototypes(prototype_labels)

            # Get the *non-associated* prototypes for each image in the batch
            negative_prototypes = list()
            for img_al in attribute_labels:
                negative_al = _make_cuda(torch.LongTensor(list(filter(
                    lambda al: al not in img_al,
                    all_attribute_labels))))
                negative_prototypes.append(model.prototypes(negative_al))

            # Compute the loss
            loss = criterion(reconstructed_x, z_patches,
                             positive_prototypes, padding_idx, x,
                             negative_prototypes=negative_prototypes)

            # Do backprop and update the weights
            loss.backward()
            optimizer.step()

            # Update the epoch loss and add the step loss to tensorboard
            epoch_loss += loss.item()
            writer.add_scalar('loss/step_loss', loss, steps)

        # Add the epoch loss to tensorboard and update the progressbar
        writer.add_scalar('loss/epoch_loss', epoch_loss, steps)
        pbar.set_postfix(epoch_loss=epoch_loss)

        model.eval()  # Setting the model in evaluation mode for testing
        if (epoch % 5 == 0) or (epoch == num_epochs):
            # Compute the nearest patch for each prototype
            nearest_patches_for_prototypes = \
                model.get_nearest_patches_for_prototypes(
                    train_dataset_with_non_random_transformation)

            # Update each prototype to be equal to the nearest patch
            model.reproject_prototypes(nearest_patches_for_prototypes)

            if (epoch % 1000 == 0) or (epoch == num_epochs):
                # Save the prototype visualization
                save_prototype_patch_visualization(
                    model, train_dataset_with_non_random_transformation,
                    nearest_patches_for_prototypes, PROTOTYPES_DIR)

                # Save the reconstructed images for the test dataset
                # for every 1000 epochs
                for i_, (image, image_label, attribute_labels, _) \
                        in enumerate(test_dataset):
                    x = image.view((1,) + image.size())
                    x = _make_cuda(torch.autograd.Variable(x))
                    z, z_patches, reconstructed_x = model(x)

                    reconstructed_image = \
                        get_image_from_tensor(reconstructed_x)
                    reconstructed_image.save(
                        os.path.join(IMAGES_DIR, '%d-%d.png' % (epoch, i_)))

                # Save the intermediate model
                model.save_weights(os.path.join(RUN_DIR, MODEL_FILE_NAME))

        # Add the prototype embeddings to tensorboard at the end
        if epoch == num_epochs:
            writer.add_embedding(
                model.prototypes.weight[1:],
                metadata=attribute_names,
                global_step=steps)

    # Save the final model and commit the tensorboard logs
    model.save_weights(os.path.join(RUN_DIR, MODEL_FILE_NAME))
    writer.close()

    return epoch_loss


if __name__ == '__main__':
    ex.run_commandline()
