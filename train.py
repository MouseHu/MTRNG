import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from configs.mt import *
# from configs.config_adder import *
from configs.config_lehmer import *
from torch_util import *
from dataset.dataloader import MultiEpochsDataLoader

# print("train type ", train_type)
# writer = None
# Define network, dataset and optimizer

model = net_prototype().cuda()
if double_precision:
    model = model.double()
train_dataset = dataset_prototype(mt_data_dir, register_data_dir, split=(0, train_split))
test_dataset = dataset_prototype(mt_data_dir, register_data_dir, split=(train_split, total_split))
# if train_type == 0:
#     model = Temper().cuda()
#     train_dataset = TemperDataset(mt_data_dir, register_data_dir, split=(0, 0.7))
#     test_dataset = TemperDataset(mt_data_dir, register_data_dir, split=(0.7, 1))
#     # model.load_state_dict(torch.load(inverse_temper_dir))
# elif train_type == 1:
#     model = ResCNNTemper().cuda()
#     train_dataset = TemperDataset(register_data_dir, mt_data_dir, split=(0, 0.7))
#     test_dataset = TemperDataset(register_data_dir, mt_data_dir, split=(0.7, 1))
#     model.load_state_dict(torch.load(temper_dir))
#
# elif train_type == 2:
#     model = CNNTwister().cuda()
#     train_dataset = MTDataset(register_data_dir, split=(0, 0.7))
#     test_dataset = MTDataset(register_data_dir, split=(0.7, 1))
#     model.load_state_dict(torch.load(twister_dir))
# elif train_type == 3:
#
#     train_dataset = MTDataset(mt_data_dir, split=(0, 0.9), seqlen=seqlen)
#     test_dataset = MTDataset(mt_data_dir, split=(0.9, 1), seqlen=seqlen)
#     model = Cracker(seqlen=seqlen).cuda()
#
# elif train_type == 4:
#     # forward
#     # train_dataset = LehmerDataset(mt_data_dir, register_data_dir, split=(0, 0.7))
#     # test_dataset = LehmerDataset(mt_data_dir, register_data_dir, split=(0.7, 1))
#     train_dataset = LehmerForwardDataset(mt_data_dir, register_data_dir, split=(0, 0.7))
#     test_dataset = LehmerForwardDataset(mt_data_dir, register_data_dir, split=(0.7, 1))
#     # model = Adder(input_bits=input_bits, output_bits=output_bits).cuda()
#     # model = Adder(seqlen=1).double().cuda()
#     # model = ResFC(input_bits=input_bits, output_bits=output_bits,seqlen=1).cuda()
#     # model = Multiplier(input_bits=input_bits, output_bits=output_bits).cuda()
#     # model = MultiplierCracker(operand_num=64).cuda()
#     # model = LehmerCracker().cuda()
#     # model = LehmerSimpleCracker().cuda()
#     # model = NeuralMultiplier(operand_dim=64, final_result=True).cuda()
#     model = NeuralMultiplier(operand_dim=128, final_result=True).cuda()
#     # model = AttentionTwister(seqlen=input_bits, input_bits=1, output_bits=output_bits).cuda()
#
# elif train_type == 5:
#     # forward
#     train_dataset = LehmerForwardDataset(mt_data_dir, register_data_dir, split=(0, 0.7))
#     test_dataset = LehmerForwardDataset(mt_data_dir, register_data_dir, split=(0.7, 1))
#     # test_dataset = LehmerDataset(mt_data_dir, split=(0.7, 1))
#     # model = Adder(input_bits=input_bits, output_bits=output_bits).cuda()
#     model = Adder(seqlen=1).double().cuda()
#
# elif train_type == 6:
#     train_dataset = AdderDataset(mt_data_dir, split=(0, 0.7))
#     test_dataset = AdderDataset(mt_data_dir, split=(0.7, 1))
#     # test_dataset = LehmerDataset(mt_data_dir, split=(0.7, 1))
#     # model = Adder(input_bits=input_bits, output_bits=output_bits).cuda()
#     model = NeuralAdder().cuda()
# else:
#     assert 0, "train_type not supported"

# if custom_optimizer:
#     optimizer = model.get_optimizer(trainer_config)
# else:
#     optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim2.RAdam(model.parameters(), lr=lr)


if write:
    writer = SummaryWriter(log_dir)


# Testing Loop
def test(epoch, test_loader, total_steps):
    # model.eval()
    test_loss = 0.0
    test_correct = 0.0
    num_batches = 0
    cum_info = {}
    for i, data in tqdm(enumerate(test_loader)):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data

        # forward + backward + optimize
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        if double_precision:
            inputs = inputs.double()
        else:
            inputs = inputs.float()
        # inputs = 1. * inputs
        # inputs = 2.* inputs - 1
        # if train_type <= 4:
        #     inputs, labels = data
        #     inputs, labels = inputs.cuda(), labels.cuda()
        #     # inputs = 2 * inputs - 1.
        #     inputs = 1. * inputs
        # elif train_type == 6:
        #     inputs, labels = data
        #     inputs, labels = inputs.cuda(), labels.cuda()
        #     inputs = 1. * inputs
        # else:
        #     assert train_type == 5
        #     inputs, labels, binary_labels = data
        #     inputs, labels, binary_labels = inputs.cuda(), labels.cuda(), binary_labels.cuda()

        outputs = model(inputs)
        loss, correct, info = mtrng_loss(model, outputs, labels, l1_coef=weight_decay, predict_list=predict_list)
        # if train_type <= 4 or train_type > 5:
        #     loss, correct, info = mtrng_loss(model, outputs, labels, l1_coef=weight_decay, predict_list=predict_list)
        # # elif train_type == 4:
        # #     loss, correct, info = mse_loss(outputs, labels)
        # else:
        #     assert train_type == 5
        #     loss, correct, info = mse_loss(outputs, labels, binary_labels)

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
        try:
            correct = correct.item()
        except AttributeError:
            pass
        test_loss += loss.item()
        test_correct += correct
        num_batches += 1

    print(
        f'[Testing] [{epoch}, {num_batches:3d}] loss: {test_loss / num_batches:.4f}, correct: {test_correct / num_batches :.4f}')
    if write:
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


def train():
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=batch_size, pin_memory=False,
                                         shuffle=True, num_workers=num_workers)
    test_loader = MultiEpochsDataLoader(test_dataset, batch_size=batch_size, pin_memory=False,
                                        shuffle=False, num_workers=num_workers)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
    #                                            shuffle=True, num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
    #                                           shuffle=False, num_workers=num_workers)
    # Training Loop
    running_loss = 0.0
    running_correct = 0.0
    running_auto_correct = 0.0
    running_auto_loss = 0.0
    total_steps = 0
    cum_steps = 0
    epoch = 0
    # if not write:
    test(epoch + 1, test_loader, total_steps)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()

        for data in tqdm(train_loader):
            total_steps += 1
            cum_steps += 1
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            if double_precision:
                inputs = inputs.double()
            else:
                inputs = inputs.float()
            # inputs = inputs - 0.5
            # if train_type <= 4:
            #     inputs, labels = data
            #     inputs, labels = inputs.cuda(), labels.cuda()
            #     # inputs = 2 * inputs - 1.
            #     inputs = 1. * inputs
            # elif train_type == 6:
            #     inputs, labels = data
            #     inputs, labels = inputs.cuda(), labels.cuda()
            #     inputs = 1. * inputs
            # else:
            #     assert train_type == 5
            #     inputs, labels, binary_labels = data
            #     inputs, labels, binary_labels = inputs.cuda(), labels.cuda(), binary_labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, correct, info = mtrng_loss(model, outputs, labels, l1_coef=weight_decay,
                                             predict_list=predict_list)
            # forward + backward + optimize
            # if train_type == 3 and use_autoencoder:
            #     labels_for_input = 2 * labels - 1.
            #
            #     autoencoder = model.autoencoder(labels_for_input)
            #     loss, correct, info = mtrng_loss(model, outputs, labels, l1_coef=weight_decay,
            #                                      predict_list=predict_list)
            #     autoencoder_loss, autoencoder_correct, _ = mtrng_loss(model, autoencoder, labels, l1_coef=weight_decay,
            #                                                           predict_list=predict_list)
            #     loss += autoencoder_loss
            #     running_auto_correct += autoencoder_correct.item()
            #     running_auto_loss += autoencoder_loss.item()
            # elif train_type <= 4 or train_type > 5:
            #     loss, correct, info = mtrng_loss(model, outputs, labels, l1_coef=weight_decay,
            #                                      predict_list=predict_list)
            # # elif train_type == 4:
            # #     loss, correct, info = mse_loss(outputs, labels)
            # else:
            #     assert train_type == 5
            #     loss, correct, info = mse_loss(outputs, labels, binary_labels)

            loss.backward()
            optimizer.step()
            try:
                correct = correct.item()
            except AttributeError:
                pass
            # print statistics
            running_loss += loss.item()
            running_correct += correct

            if total_steps % print_freq == print_freq - 1:  # print every 20 mini-batches
                print(
                    f'[{epoch + 1}, {total_steps :5d}] loss: {running_loss / cum_steps:.4f}, correct: {running_correct / cum_steps :.4f}')
                if write:
                    # if train_type == 3 and use_autoencoder:
                    #     writer.add_scalar("train/autoencoder_loss", running_auto_loss / cum_steps, total_steps)
                    #     writer.add_scalar("train/autoencoder_correct", running_auto_correct / cum_steps, total_steps)
                    writer.add_scalar("train/loss", running_loss / cum_steps, total_steps)
                    writer.add_scalar("train/correct", running_correct / cum_steps, total_steps)
                    writer.add_scalar("train/input_mean", inputs.mean().detach().cpu().numpy(), total_steps)
                    writer.add_scalar("train/input_var", inputs.var().detach().cpu().numpy(), total_steps)

                    for key, value in info.items():
                        try:
                            writer.add_scalar("train/" + key, value, total_steps)
                        except ValueError:
                            pass
                        except AttributeError:
                            pass
                running_loss = 0.0
                running_correct = 0.0
                # running_auto_correct = 0.0
                # running_auto_loss = 0.0
                cum_steps = 0
        if epoch % test_epoch == test_epoch - 1:
            test(epoch + 1, test_loader, total_steps)
            torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), save_path)
    print('Finished Training')


if __name__ == "__main__":
    train()
