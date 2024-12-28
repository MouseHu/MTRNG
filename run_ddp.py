import os
from torch_util import *
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from configs.config_mt_ddp import *
# from configs.config_mt_gpt_ddp2 import *
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
    # dataset = dataset_prototype(mt_data_dir, register_data_dir, split=split, seqlen=seqlen, nbits=input_bits)
    # dataset = dataset_prototype(mt_data_dir, register_data_dir, split=split, seqlen=seqlen, nbits=input_bits)
    dataset = dataset_prototype(mt_data_dir, split=split, seqlen=seqlen)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    dataloader = MultiEpochsDataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                                       shuffle=False, num_workers=num_workers, sampler=sampler)

    return dataloader


def configure_optimizers(model, train_config):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
    return optimizer


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
        if double_precision:
            inputs = inputs.double()
        else:
            # inputs = inputs.float()
            pass
        # forward + backward + optimize
        # inputs = 2 * inputs - 1.
        # inputs = inputs - 0.5
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

    # model = net_prototype(seqlen=seqlen, input_bits=input_bits, output_bits=output_bits).to(rank)
    model = net_prototype(seqlen=seqlen, input_bits=input_bits, output_bits=output_bits).to(rank)
    if double_precision:
        model = model.double()
    model = Ddp(model, device_ids=[rank])
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    if custom_optimizer:
        optimizer = configure_optimizers(model, trainer_config)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = prepare(rank, world_size, (0, train_split), batch_size, num_workers=num_workers)
    if rank == 0:
        test_batch_size = batch_size // 10
        test_loader = prepare(0, 1, (train_split, total_split), test_batch_size, num_workers=num_workers)

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
            if double_precision:
                inputs = inputs.double()
            else:
                # inputs = inputs.float()
                pass
            optimizer.zero_grad()

            # inputs = inputs - 0.5
            # inputs = 2 * inputs - 1.
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
                # writer.add_scalar("train/input_mean", inputs.mean().detach().cpu().numpy(), total_steps)
                # writer.add_scalar("train/input_var", inputs.var().detach().cpu().numpy(), total_steps)
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
