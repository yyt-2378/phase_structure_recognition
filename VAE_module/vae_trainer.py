import torch
import argparse
import yaml
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from STEMDataset import StemDataset
from vae_models.vanilla_vae import VanillaVAE


def train_vae(model, train_loader, num_epochs, learning_rate, log_dir, checkpoint_interval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            result = model(data, labels)
            loss_dict = model.loss_function(*result, M_N=0.005)

            loss = loss_dict['loss']
            loss.backward()
            total_loss += loss.item()

            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} Average Loss: {avg_loss:.4f}")

        # Logging to TensorBoard
        writer.add_scalar("Loss/Train", avg_loss, epoch + 1)

        # Save model checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}: {checkpoint_path}")

    writer.close()
    torch.save(model.state_dict(), r'SAVE/VAE.pt')
    print("Training finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic runner for VAE vae_models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae_train_batchsize_64.yaml')

    # config params
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # model params
    input_channel = config['model_params']['in_channels']
    latent_dim = config['model_params']['latent_dim']
    data_path = config['data_params']['data_path']
    train_batch_size = config['data_params']['train_batch_size']
    val_batch_size = config['data_params']['val_batch_size']
    patch_size = config['data_params']['patch_size']
    num_workers = config['data_params']['num_workers']

    # train params
    lr = config['exp_params']['LR']
    weight_decay = config['exp_params']['weight_decay']
    epoch = config['trainer_params']['max_epochs']

    train_dataset = StemDataset(mode='train', dir=data_path)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)
    print('Dataset loaded! length of train set is {0}'.format(len(train_dataset)))
    # Example usage
    # Assuming you have a dataset and train_loader ready
    model = VanillaVAE(in_channels=input_channel, latent_dim=latent_dim)
    train_vae(model, train_loader, num_epochs=epoch, learning_rate=lr,
              log_dir="logs", checkpoint_interval=5)
