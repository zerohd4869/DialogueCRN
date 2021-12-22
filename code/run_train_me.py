import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from dataloader import MELDDataset
from loss import FocalLoss
from model import DialogueCRN
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def seed_everything(seed=2021):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(path, n_classes, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset(path=path, n_classes=n_classes)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(path=path, n_classes=n_classes, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text', target_names=None,
                        tensorboard=False):
    assert not train_flag or optimizer != None
    losses, preds, labels, masks = [], [], [], []

    if train_flag:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train_flag: optimizer.zero_grad()

        textf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]

        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if feature_type == "multi":
            dataf = torch.cat((textf, acouf), dim=-1)
        else:
            dataf = textf if feature_type == 'text' else acouf

        # seq_len, batch, n_classes
        log_prob = model(dataf, qmask, seq_lengths)
        label = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])
        umask = torch.cat([umask[j][:seq_lengths[j]] for j in range(len(umask))])
        loss = loss_f(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item())

        if train_flag:
            loss.backward()
            if tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(
        metrics.classification_report(labels, preds, target_names=target_names if target_names else None, sample_weight=masks, digits=4))
    all_matrix.append(["ACC"])
    for i in range(len(target_names)):
        all_matrix[-1].append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, []


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--status', type=str, default='train', help='optional status: train/test')

    parser.add_argument('--feature_type', type=str, default='text', help='feature type, multi/text/audio')

    parser.add_argument('--data_dir', type=str, default='../data/meld//MELD_features_raw.pkl', help='dataset dir')

    parser.add_argument('--output_dir', type=str, default='../outputs/meld/dialoguecrn_v1', help='saved model dir')

    parser.add_argument('--load_model_state_dir', type=str, default='./outputs/meld/base_model/dialoguecrn_22.pkl', help='load model state dir')

    parser.add_argument('--base_model', default='LSTM', help='base model, LSTM/GRU/Linear')

    parser.add_argument('--base_layer', type=int, default=1, help='the number of base model layers, 1/2')

    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')

    parser.add_argument('--patience', type=int, default=20, help='early stop')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.0, metavar='valid_rate', help='valid rate')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.0005, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')

    parser.add_argument('--step_s', type=int, default=4, help='the number of reason turns at situation-level')

    parser.add_argument('--step_p', type=int, default=0, help='the number of reason turns at speaker-level')

    parser.add_argument('--gamma', type=float, default=1.0, help='gamma 0/0.5/1/2')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--cls_type', type=str, default='emotion', help='choose between sentiment or emotion')

    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    args = parser.parse_args()
    print(args)

    epochs, batch_size, status, output_path, data_path, load_model_state_dir, base_model, base_layer, feature_type = \
        args.epochs, args.batch_size, args.status, args.output_dir, args.data_dir, args.load_model_state_dir, args.base_model, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    reason_steps = [args.step_s, args.step_p]

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    # MELD dataset
    n_speakers, hidden_size, input_size = 9, 100, None
    if args.cls_type.strip().lower() == 'emotion':
        n_classes = 7
        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
        class_weights = torch.FloatTensor(
            [1 / 0.469506857, 1 / 0.119346367, 1 / 0.026116137, 1 / 0.073096002, 1 / 0.168368836, 1 / 0.026334987, 1 / 0.117230814])
    else:
        n_classes = 3
        target_names = ['0', '1', '2']
        class_weights = torch.FloatTensor([1.0, 1.0, 1.0])
    if feature_type == 'multi':
        input_size = 900
    elif feature_type == 'text':
        input_size = 600
    elif feature_type == 'audio':
        input_size = 300
    else:
        print('Error: feature_type not set.')
        exit(0)

    seed_everything(seed=args.seed)
    model = DialogueCRN(base_model=base_model,
                        base_layer=base_layer,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_speakers=n_speakers,
                        n_classes=n_classes,
                        dropout=args.dropout,
                        cuda_flag=cuda_flag,
                        reason_steps=reason_steps)

    if cuda_flag:
        print('Running on GPU')
        # torch.cuda.set_device(1)
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    name = 'DialogueCRN'
    print('{} with {} as base model.'.format(name, base_model))
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(feature_type))

    loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader, test_loader = get_MELD_loaders(data_path,
                                                               n_classes,
                                                               valid=args.valid_rate,
                                                               batch_size=batch_size,
                                                               num_workers=0)

    if status == 'train':
        all_test_fscore, all_test_acc = [], []
        best_epoch, patience, best_eval_fscore, best_eval_loss = -1, 0, 0, None
        for e in range(epochs):
            start_time = time.time()
            train_loss, train_acc, train_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=train_loader, epoch=e, train_flag=True,
                                                                            optimizer=optimizer, cuda_flag=cuda_flag, feature_type=feature_type,
                                                                            target_names=target_names)
            valid_loss, valid_acc, valid_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=valid_loader, epoch=e,
                                                                            cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names)
            test_loss, test_acc, test_fscore, test_metrics, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader, epoch=e,
                                                                                    cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names)
            all_test_fscore.append(test_fscore)
            all_test_acc.append(test_acc)

            if args.valid_rate > 0:
                eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
            else:
                eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
            if e == 0 or best_eval_fscore < eval_fscore:
                best_epoch, best_eval_fscore = e, eval_fscore
                if not os.path.exists(output_path): os.makedirs(output_path)
                save_model_dir = os.path.join(output_path, '{}_{}.pkl'.format(name, e).lower())
                torch.save(model.state_dict(), save_model_dir)
            if best_eval_loss is None:
                best_eval_loss = eval_loss
            else:
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience = 0
                else:
                    patience += 1

            if args.tensorboard:
                writer.add_scalar('train: accuracy/f1/loss', train_acc / train_fscore / train_loss, e)
                writer.add_scalar('valid: accuracy/f1/loss', valid_acc / valid_fscore / valid_loss, e)
                writer.add_scalar('test: accuracy/f1/loss', test_acc / test_fscore / test_loss, e)
                writer.close()

            print(
                'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                    format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                           round(time.time() - start_time, 2)))
            print(test_metrics[0])
            print(test_metrics[1])

            if patience >= args.patience:
                print('Early stoping...', patience)
                break

        print('Final Test performance..')
        print('Epoch: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch,
                                                             all_test_acc[best_epoch] if best_epoch >= 0 else 0,
                                                             all_test_fscore[best_epoch] if best_epoch >= 0 else 0))

    elif status == 'test':
        start_time = time.time()
        model.load_state_dict(torch.load(args.load_model_state_dir))
        test_loss, test_acc, test_fscore, test_metrics, test_outputs = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader,
                                                                                           cuda_flag=cuda_flag, feature_type=feature_type,
                                                                                           target_names=target_names)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, 0)
            writer.close()
        print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        print(test_metrics[0])
        print(test_metrics[1])
    else:
        print('the status must be one of train/test')
        exit(0)
