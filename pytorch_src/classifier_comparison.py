import os
import argparse
import numpy as np
np.random.seed(0)
import torch
from torchvision import datasets, transforms
from networks import *
from dataset import *

def get_data(datadir):
    train_ds = datasets.MNIST(root=datadir, download=True)
    trainX = train_ds.train_data
    trainY = train_ds.train_labels.numpy()

    test_ds = datasets.MNIST(root=datadir, train=False)
    testX = test_ds.test_data
    testY = test_ds.test_labels.numpy()

    return trainX, trainY, testX, testY

def eval(args, model, device, data_loader):
    model.eval()
    f_X, Y = [], []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            z_param = model.encode(data)
            f_X.append(z_param[:, :args.zdim].cpu().numpy())
            Y.append(label.numpy())
    return np.concatenate(f_X, axis=0), np.concatenate(Y, axis=0)

def run_classifier(f_trainX, trainY, f_testX, testY):

    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "Linear SVM"
         # , "RBF SVM"
         ]

    classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        SVC(kernel="linear")
        # ,SVC(gamma=2, C=1)
        ]

    scores = []
    for name, clf in zip(names, classifiers):
        clf.fit(f_trainX, trainY)
        score = clf.score(f_testX, testY)
        print(name, score, flush=True)
        scores.append(score)
    print(scores)
    return scores

def main():

    parser = argparse.ArgumentParser(description='MNIST classifier testing')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='name of dataset')
    parser.add_argument('--logdir', type=str, default='./logs',
                        help='log directory')
    parser.add_argument('--datadir', type=str, default='../datasets')
    parser.add_argument('--dump', dest='dump', action='store_true', default=False,
                        help='dump')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, 
                        help='input batch size for testing (default: 1000)')

    args = parser.parse_args()
    args.datadir = os.path.join(args.datadir, args.dataset)
    args.logdir = os.path.join(args.logdir, args.dataset)

    if args.dump:
        dump_file = os.path.join(resdir, 'log.txt')
        sys.stdout = open(dump_file, 'w')

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = get_train_test_loaders(dataset=args.dataset,
                                            datadir=args.datadir,
                                            batch_size=args.batch_size,
                                            test_batch_size=args.test_batch_size,
                                            kwargs=kwargs)

    # trainX, trainY, testX, testY = get_data(args.datadir)

    all_results_dict = {}
    with open(os.path.join(args.logdir, 'best_configs.txt')) as infile:
        for line in infile:
            ckptdir = os.path.join(line[:-1], 'ckpts')
            savefile = os.path.join(ckptdir, 'best.pt')

            tokens = line[:-1].split('/')
            loss_type = tokens[-2]
            zdim = int(tokens[-3])
            is_linear_flag = True if tokens[-4] == 'linear' else False

            args.loss_type = loss_type
            args.zdim = zdim
            args.linear = is_linear_flag
            nin = 784 if args.dataset == 'mnist' else -1

            model = EncoderDecoder(nin=nin, args=args).to(device)
            state_dict = torch.load(savefile)
            model.load_state_dict(state_dict)

            f_trainX, trainY = eval(args, model, device, train_loader)
            f_testX, testY = eval(args, model, device, test_loader)
            
            print(f_trainX.shape, trainY.shape, f_testX.shape, testY.shape)

            scores = run_classifier(f_trainX, trainY, f_testX, testY)
            all_results_dict[(loss_type, zdim, tokens[-4])] = scores

    print(all_results_dict)
    import pickle
    with open(os.path.join(args.logdir, 'classifier.pkl'), 'wb') as outfile:
        pickle.dump(all_results_dict, outfile)

if __name__ == '__main__':
    main()