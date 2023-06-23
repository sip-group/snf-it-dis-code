# Standard imports
import os
import datetime

# Torch imports
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project imports
from losses.losses import get_classification_loss
from data.cdp_dataset import get_split
from models.models import ModelsManager
from utilities.utilities import tln_forward, get_device, set_reproducibility

from configuration.configuration_reader import ConfigurationReader, get_conf_path
from configuration.configuration_reader import \
    TEMPLATE_DIR, ORIGINAL_DIR, FAKE_DIR, ADDITIONAL_ORIGINALS, ADDITIONAL_FAKES, ADDITIONAL_TEST_ORIGINALS, \
    ADDITIONAL_TEST_FAKES, BATCH_SIZE, BAD_INDICES, CLASSIFICATION, EPOCHS, LATENT_DIMENSION, MLP, \
    MODEL, MULTI_GPU, PATIENCE, PERCENT_TRAIN, PERCENT_VAL, RUNS, SEED, STACK, STORE_DIR, TEMPLATE_LR, VERIFICATION_LR


def train_loop(models_manager, train_loader, val_loader, template_lr, verification_lr, max_epochs, criterion,
               train_patience, classification):
    """
    Trains Template and Verification networks.
    Logs of the training are available under the './tensorboard_logs/' directory.

    :param models_manager: Manager of the models that is used to storing them.
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param template_lr: Learning rate for the template network
    :param verification_lr: Learning rate for the verification network
    :param max_epochs: Maximum number of epochs that are going to take place
    :param criterion: Loss function that takes latent vectors for templates, originals and fakes as argument
    :param train_patience: Patience (in epochs) before early stopping stops the training procedure
    :param classification: Boolean that specifies whether we shall not use the template network
    """
    # Getting models
    t_net, v_net = models_manager.get_models()

    # Setting models in training mode
    t_net, v_net = t_net.train(), v_net.train()

    # Getting device
    device = models_manager.device

    # Defining training optimizers
    template_optim = Adam(t_net.parameters(), lr=template_lr)
    verification_optim = Adam(v_net.parameters(), lr=verification_lr)

    # Tensorboard logging
    dt = datetime.datetime.now()
    tb_writer = SummaryWriter(f"./tensorboard_logs/{dt.hour}{dt.minute}_{dt.day}{dt.month}{dt.year}")

    # Training the template and verification network
    patience_steps = 0
    best_ever_loss = float('inf')
    for epoch in range(max_epochs):
        train_loss = 0.0
        for train_batch in train_loader:
            # Computing forward pass and loss
            loss, _ = tln_forward(train_batch, t_net, v_net, criterion, device, no_template=classification)
            train_loss += loss.item() / len(train_loader)

            # Applying gradient descent
            verification_optim.zero_grad()
            template_optim.zero_grad()
            loss.backward()
            verification_optim.step()
            template_optim.step()

        # Validation
        val_loss = 0.0
        for val_batch in val_loader:
            loss, _ = tln_forward(val_batch, t_net, v_net, criterion, device, no_template=classification)
            val_loss += loss.item() / len(val_loader)

        # Storing best model / Stopping training
        if val_loss >= best_ever_loss:
            print(f"Epoch {epoch + 1}: Train loss {train_loss:.2f}\t Val loss {val_loss:.2f}")
            patience_steps += 1
            if patience_steps >= train_patience:
                break
        else:
            print(
                f"Epoch {epoch + 1}: Train loss {train_loss:.2f}\t Val loss {val_loss:.2f} --> Storing best model ever")
            # Resetting patience steps
            patience_steps = 0
            best_ever_loss = val_loss

            # Storing best model ever
            models_manager.store_networks()

        tb_writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, global_step=epoch)
    tb_writer.close()


def test_loop(models_manager,
              test_loader,
              criterion,
              classification=False,
              lv_file_post_name='(train)'
              ):
    """
        Tests Template and Verification networks.
        Stores the latent representations of this test set under the '/lv/' folder.

        :param models_manager: Manager of the models that is used to storing them.
        :param test_loader: Test data loader
        :param criterion: Loss function that takes latent vectors for templates, originals and fakes as argument
        :param x_dirs:
        :param f_dirs:
        :param classification: Whether we're only doing classification with the Verification Net
    """
    # Loading best models
    t_net, v_net = models_manager.get_models()

    # (Optional) - Setting models in evaluation mode (and keeping train batch-norm mean and std)
    """
    t_net, v_net = t_net.eval(), v_net.eval()

    def train_bn(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()

    t_net.apply(train_bn)
    v_net.apply(train_bn)
    """

    # Getting device and store directory
    device = models_manager.device
    store_dir = models_manager.store_dir

    # Computing test loss and obtaining latent vectors
    test_loss = 0.0
    all_t_lvs, all_x_lvs, all_f_lvs = None, [], []
    for test_batch in test_loader:
        loss, l_vectors = tln_forward(test_batch, t_net, v_net, criterion, device, no_template=classification)
        test_loss += loss.item() / len(test_loader)

        # Moving latent vectors to CPU
        t_lv, x_lvs, f_lvs = l_vectors

        if t_lv is not None:
            t_lv = t_lv.detach().cpu()
        x_lvs = [x_lv.detach().cpu() for x_lv in x_lvs]
        f_lvs = [f_lv.detach().cpu() for f_lv in f_lvs]

        if len(all_x_lvs) == 0:
            all_t_lvs = t_lv.reshape(-1, models_manager.latent_d) if t_lv is not None else None
            all_x_lvs = [x_lv.reshape(-1, models_manager.latent_d) for x_lv in x_lvs]
            all_f_lvs = [f_lv.reshape(-1, models_manager.latent_d) for f_lv in f_lvs]
        else:
            if all_t_lvs is not None:
                if len(t_lv.shape) < 2:
                    t_lv = t_lv.reshape(-1, models_manager.latent_d)
                all_t_lvs = torch.vstack((all_t_lvs, t_lv))

            for i in range(len(all_x_lvs)):
                if len(x_lvs[i].shape) < 2:
                    x_lvs[i] = x_lvs[i].reshape(-1, models_manager.latent_d)
                all_x_lvs[i] = torch.vstack((all_x_lvs[i], x_lvs[i]))

            for i in range(len(all_f_lvs)):
                if len(f_lvs[i].shape) < 2:
                    f_lvs[i] = f_lvs[i].reshape(-1, models_manager.latent_d)
                all_f_lvs[i] = torch.vstack((all_f_lvs[i], f_lvs[i]))

    print(f"Test loss is: {test_loss:.2f}")

    # Storing latent vectors of test templates, originals and fakes
    lv_store_dir = os.path.join(store_dir, 'lv')
    if not os.path.isdir(lv_store_dir):
        os.mkdir(lv_store_dir)

    if all_t_lvs is not None:
        torch.save(all_t_lvs, os.path.join(lv_store_dir, f"templates{lv_file_post_name}.pt"))

    # Getting x_dirs and f_dirs names
    x_dir_names = [dn.split('/')[-1] for dn in test_loader.dataset.x_dirs]
    f_dir_names = [dn.split('/')[-1] for dn in test_loader.dataset.f_dirs]

    for i in range(len(all_x_lvs)):
        file_name = f"o_{x_dir_names[i]}{lv_file_post_name}.pt"
        torch.save(all_x_lvs[i], os.path.join(lv_store_dir, file_name))

    for i in range(len(all_f_lvs)):
        file_name = f"f_{f_dir_names[i]}{lv_file_post_name}.pt"
        torch.save(all_f_lvs[i], os.path.join(lv_store_dir, file_name))


def main():
    """
    Main function that loads data, trains both template and verification network, stores the best models and outputs
    test performances.
    """
    # Getting program arguments
    conf = ConfigurationReader(get_conf_path())
    testing = conf.testing
    t_dir, x_dir, f_dir = conf[TEMPLATE_DIR], conf[ORIGINAL_DIR], conf[FAKE_DIR]
    # train_percent, val_percent = args[PERCENT_TRAIN] if not testing else 0, args[PERCENT_VAL] if not testing else 0
    train_percent, val_percent = conf[PERCENT_TRAIN], conf[PERCENT_VAL]
    add_original, add_fakes = conf[ADDITIONAL_ORIGINALS], conf[ADDITIONAL_FAKES]
    add_test_original, add_test_fakes = conf[ADDITIONAL_TEST_ORIGINALS], conf[ADDITIONAL_TEST_FAKES]
    bad_indices = conf[BAD_INDICES]
    batch_size = conf[BATCH_SIZE]
    classification = conf[CLASSIFICATION]
    ld = conf[LATENT_DIMENSION] if not classification else 1
    mlp_head = conf[MLP]
    model_name = conf[MODEL]
    multi_gpu = conf[MULTI_GPU]
    max_epochs = conf[EPOCHS]
    template_lr, verification_lr = conf[TEMPLATE_LR], conf[VERIFICATION_LR]
    train_patience = conf[PATIENCE]
    n_runs = conf[RUNS]
    do_stack = conf[STACK]
    store_dir = conf[STORE_DIR]
    seed = conf[SEED]

    for run in range(n_runs):
        # Setting reproducibility
        set_reproducibility(seed + run)

        # Getting store_dir for this run
        run_store_dir = os.path.join(store_dir, f'run_{seed + run}')
        if not os.path.isdir(run_store_dir):
            testing = False
            os.mkdir(run_store_dir)
        else:
            testing = True

        # Getting CDP directories
        x_dirs = [x_dir, *add_original]
        f_dirs = [f_dir, *add_fakes]
        assert len(x_dirs) > 0 and len(f_dirs) > 0

        # Getting data split
        train_data, val_data, test_data = get_split(t_dir,
                                                    x_dirs,
                                                    f_dirs,
                                                    train_percent,
                                                    val_percent,
                                                    bad_indexes=BAD_INDICES if bad_indices else None,
                                                    return_diff=classification and not do_stack,
                                                    return_stack=do_stack,
                                                    multi_gpu=multi_gpu
                                                    )

        if train_percent > 0:
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        if val_percent > 0:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Device
        device = get_device()

        # Loss criterion
        criterion = get_classification_loss(device)

        # Models manager
        t_in_size = train_data[0]['template'].shape
        v_in_size = train_data[0]['originals'][0].shape
        models_manager = ModelsManager(t_in_size, v_in_size, ld, model_name, multi_gpu, device, run_store_dir, mlp_head)

        # Training new models
        if not testing:
            # Updating args to keep the split information
            conf.args['n_train_data'] = len(train_data)
            conf.args['n_val_data'] = len(val_data)
            conf.args['n_test_data'] = len(test_data)

            # Logging test metrics and program arguments
            conf.log_conf()

            # Training models
            train_loop(models_manager, train_loader, val_loader, template_lr, verification_lr, max_epochs, criterion,
                       train_patience, classification)
        else:
            models_manager.load_networks()

        # Testing detection of fakes
        print("Testing model on the test set...")
        test_loop(models_manager, test_loader, criterion, classification, lv_file_post_name='(train)')

        # Testing on unseen originals and fakes
        if len(add_test_original) > 0 and len(add_test_fakes) > 0:
            # Getting final test directories and adjusting batch size depending on number of additional folders
            prev_n_folders = 1 + len(x_dirs) + len(f_dirs)
            x_dirs, f_dirs = add_test_original, add_test_fakes
            curr_n_folder = 1 + len(x_dirs) + len(f_dirs)
            batch_size = (batch_size * prev_n_folders) // curr_n_folder

            # Making batch size divisible by the number of CPUs
            if multi_gpu and torch.cuda.device_count() > 0:
                batch_size -= batch_size % torch.cuda.device_count()
                if batch_size == 0:
                    batch_size = torch.cuda.device_count()

            _, _, test_data = get_split(t_dir,
                                        x_dirs,
                                        f_dirs,
                                        train_percent=0,
                                        val_percent=0,
                                        bad_indexes=BAD_INDICES if bad_indices else None,
                                        return_diff=classification and not do_stack,
                                        return_stack=do_stack,
                                        multi_gpu=multi_gpu,
                                        train_post_transform=None
                                        )
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            print("Testing model on unseen test data")
            test_loop(models_manager, test_loader, criterion, classification, lv_file_post_name='(test)')


if __name__ == '__main__':
    main()
