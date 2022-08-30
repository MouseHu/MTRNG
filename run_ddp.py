import os
from network.fc import Cracker
from network.mtrng import MTCracker
from dataset.dataset import MTDataset
from torch_util import *
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from configs.config_prng_ddp import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as Ddp
from torch.utils.data.distributed import DistributedSampler
from dataset.dataloader import MultiEpochsDataLoader
import random


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare(rank, world_size, split, batch_size=32, pin_memory=True, num_workers=0):
    dataset = dataset_prototype(mt_data_dir, register_data_dir, split=split, seqlen=seqlen, nbits=input_bits)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    dataloader = MultiEpochsDataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                                       shuffle=False, num_workers=num_workers, sampler=sampler)

    return dataloader


# Testing Loop
def test(model, epoch, test_loader, writer, total_steps):
    model.eval()
    test_loss = 0.0
    test_correct = 0.0
    num_batches = 0
    cum_info = {}
    for i, data in tqdm(enumerate(test_loader)):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data
        inputs, labels = inputs.to(0), labels.to(0)

        # forward + backward + optimize
        inputs = 2 * inputs - 1.
        outputs = model.module(inputs)
        loss, correct, info = mtrng_loss(model, outputs, labels, l1_coef=weight_decay, predict_list=predict_list)
        for k, v in info.items():
            try:
                v = v.item()
            except AttributeError:
                pass
            except ValueError:
                pass
            if k in cum_info:
                cum_info[k] += v
            else:
                cum_info[k] = v
        # print statistics
        test_loss += loss.item()
        test_correct += correct.item()
        num_batches += 1

    print(
        f'[Testing] [{epoch}, {num_batches:3d}] loss: {test_loss / num_batches:.4f}, correct: {test_correct / num_batches :.4f}')
    writer.add_scalar("test/loss", test_loss / num_batches, total_steps)
    writer.add_scalar("test/correct", test_correct / num_batches, total_steps)
    for key, value in cum_info.items():
        try:
            writer.add_scalar("test/" + key, value / num_batches, total_steps)
        except ValueError:
            pass
        except AttributeError:
            pass
        except AssertionError:
            pass
    writer.flush()


# Training Loop
def run(rank, world_size):
    writer, test_loader = None, None
    running_loss = 0.0
    running_correct = 0.0

    total_steps = 0
    cum_steps = 0
    print(f"Running DDP on node {rank}.")
    setup(rank, world_size)

    if rank == 0:
        writer = SummaryWriter(log_dir)

    model = net_prototype(seqlen=seqlen, input_bits=input_bits, output_bits=output_bits).to(rank)
    model = Ddp(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = prepare(rank, world_size, (0, 0.7), batch_size, num_workers=num_workers)
    if rank == 0:
        test_loader = prepare(0, 1, (0.7, 1), batch_size, num_workers=num_workers)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        # print(f"rank:{rank} epoch:{epoch}")
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        # iterator = train_loader
        for data in pbar:
            total_steps += 1
            cum_steps += 1
            inputs, labels = data
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()

            inputs = 2 * inputs - 1.
            outputs = model(inputs)

            loss, correct, info = mtrng_loss(model, outputs, labels, l1_coef=weight_decay, predict_list=predict_list)

            loss.backward()
            optimizer.step()
            # if rank == 0:
            running_loss += loss.item()
            running_correct += correct.item()
            if rank == 0:
                pbar.set_description(
                    f'[{epoch + 1}, {total_steps}] [loss: {running_loss / cum_steps:.4f}, correct: {running_correct / cum_steps :.4f}]')
            if total_steps % print_freq == print_freq - 1 and rank == 0:  # print every 20 mini-batches
                writer.add_scalar("train/loss", running_loss / cum_steps, total_steps)
                writer.add_scalar("train/correct", running_correct / cum_steps, total_steps)
                writer.add_scalar("train/input_mean", inputs.mean().detach().cpu().numpy(), total_steps)
                writer.add_scalar("train/input_var", inputs.var().detach().cpu().numpy(), total_steps)
                # print(scheduler.get_lr())
                # writer.add_scalar("train/lr", scheduler.get_last_lr()[0], total_steps)
                for key, value in info.items():
                    try:
                        writer.add_scalar("train/" + key, value, total_steps)
                    except ValueError:
                        pass
                    except AttributeError:
                        pass
                    except AssertionError:
                        pass
                running_loss = 0.0
                running_correct = 0.0
                cum_steps = 0

        if epoch % test_epoch == test_epoch - 1 and rank == 0:
            assert rank == 0
            test(model, epoch, test_loader, writer, total_steps)
            torch.save(model.state_dict(), save_path)
    if rank == 0:
        torch.save(model.state_dict(), save_path)
        print('Finished Training')
    cleanup()


if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['NODE_RANK'])
    run(rank, world_size)
