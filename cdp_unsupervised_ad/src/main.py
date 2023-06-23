from torch.optim import Adam

from models.models import get_models
from models.utils import forward, store_models, load_models
from data.utils import load_cdp_data, BAD_IDX_MOBILE
from utils.anomaly_functions import *
from utils.utils import *


def train(mode, train_loader, val_loader, lr, device, epochs, result_dir="./", checkpoint_dir=None):
    """Training loop which trains models according to the mode, train and validation loaders, learning rate, device and
     epochs. The model/s is/are stored in the provided path."""
    # Creating / Loading the model
    models = get_models(mode, device=device) if not checkpoint_dir else load_models(mode, checkpoint_dir, device)
    optims = [Adam(model.parameters(), lr=lr) for model in models]

    # Training loop
    for model in models:
        model.train()

    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss, val_loss = 0.0, 0.0
        for batch in train_loader:
            t = batch["template"].to(device)
            x = batch["originals"][0].to(device)
            batch_loss = forward(mode, models, t, x)[0]

            for optim in optims:
                optim.zero_grad()

            batch_loss.backward()

            for optim in optims:
                optim.step()

            epoch_loss += batch_loss.item() / len(train_loader)

        for batch in val_loader:
            t = batch["template"].to(device)
            x = batch["originals"][0].to(device)

            batch_loss = forward(mode, models, t, x)[0]

            val_loss += batch_loss.item() / len(val_loader)

        epoch_str = f"Epoch {epoch + 1}/{epochs}\tTrain loss: {epoch_loss:.3f}\tVal loss: {val_loss:.3f}"
        if val_loss < best_loss:
            best_loss = val_loss
            store_models(mode, models, result_dir)
            epoch_str += " --> Stored best model(s)"
        print(epoch_str)


def test(mode, test_loader, device, title=None, result_dir="./", o_names=None, f_names=None):
    """Testing loop which """
    models = load_models(mode, result_dir, device)

    for model in models:
        model.eval()

    o_scores = [[] for _ in range(len(test_loader.dataset.x_dirs))]
    f_scores = [[] for _ in range(len(test_loader.dataset.f_dirs))]

    with torch.no_grad():
        for batch in test_loader:
            t = batch["template"].to(device)
            x = batch["originals"][0].to(device)

            for idx, x in enumerate(batch["originals"]):
                x = x.to(device)
                o_scores[idx].extend(get_anomaly_score(mode, models, t, x))

            for idx, f in enumerate(batch["fakes"]):
                f = f.to(device)
                f_scores[idx].extend(get_anomaly_score(mode, models, t, f))

        store_scores(o_scores, f_scores, result_dir, o_names, f_names)
        store_hist_picture(o_scores, f_scores, result_dir, title, o_names, f_names)


def main():
    # Parameters
    args = parse_args()
    mode = args[MODE]
    n_epochs = args[EPOCHS]
    result_dir = args[RESULT_DIR]
    lr = args[LR]
    bs = args[BS]
    tp = args[TP]
    vp = args[VP]
    no_train = args[NO_TRAIN]
    seed = args[SEED]
    o_names = args[ORIG_NAMES] 
    f_names = args[FAKE_NAMES]
    print(args)

    # Setting reproducibility
    set_reproducibility(seed)

    # Creating result directory
    create_dir(result_dir)

    # Getting program device
    device = get_device()

    # Loading data
    train_loader, val_loader, test_loader, _ = load_cdp_data(args, tp, vp, bs, bad_indexes=BAD_IDX_MOBILE)
    store_split(result_dir, train_loader, val_loader, test_loader)

    # Training new model(s) is result directory does not exist
    if not no_train:
        # Training loop
        print(f"Training new models.")
        checkpoint_dir = args[CHECKPOINT_DIR] if CHECKPOINT_DIR in args.keys() else None
        train(mode, train_loader, val_loader, lr, device, n_epochs, result_dir, checkpoint_dir)

    # Testing loop
    print(f"\n\nTesting trained model(s)")
    test(mode, test_loader, device, f"Results with mode ({mode})", result_dir, o_names, f_names)

    # Notifying program has finished
    print(f"\nProgram completed successfully. Results are available at {result_dir}")


if __name__ == '__main__':
    main()
