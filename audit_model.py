"""Auditing DP-SGD in black-box setting"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import argparse
from opacus.accountants.utils import get_noise_multiplier
import copy
from torch.utils.data import TensorDataset, DataLoader
import time
import dill

import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from models import Models
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t
from utils.clipbkd import craft_clipbkd, choose_worstcase_label

import gc


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def xavier_init_model(model):
    """Initialize model using Xavier initialization"""
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    model.apply(init_weights)


def kaiming_init_model(model):
    """Initialize model using Kaiming initialization (He init) for ReLU"""
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
    model.apply(init_weights)


def train_model(model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm, n_epochs, lr, spectral_signature_args, device='cpu', init_model=None, block_size=1024, out_dim=10, use_defense=False, store_canary_rank=False):
    """Train model w/ DP-SGD (no sub-sampling + gradients are summed instead of averaged)"""
    
    # initialize model, loss function, and optimizer
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim, model_name=model_name).to(device)
        if model_name == 'cnn':
            xavier_init_model(model)
        else:
            kaiming_init_model(model)
    else:
        model = copy.deepcopy(init_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # set noise level
    if epsilon is not None:
        # no subsampling, i.e., sample rate = 1
        noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, sample_rate=1.0,
            epochs=n_epochs, accountant='rdp')
    else:
        noise_multiplier = 0

    # TODO: only use when defense
    drop_mask = torch.zeros_like(y)
    
    # train model for n_epochs
    for epoch in tqdm(range(n_epochs), leave=False):
        print('Epoch:', epoch)
        optimizer.zero_grad()

        # Prob of predicting canary correctly
        model.eval()
        with torch.no_grad():
            output = model(X_target)
            probs = torch.nn.functional.softmax(output, dim=1)
            print(probs[0][4])
        model.train()

        curr_accumulated_gradients, drop_mask = clip_and_accum_grads(model, 
                                                X, 
                                                y, 
                                                optimizer, 
                                                criterion, 
                                                max_grad_norm, 
                                                block_size=block_size,
                                                drop_mask=drop_mask,
                                                device=device)
        
        with torch.no_grad():   
            for name, param in model.named_parameters():
                curr_param_gradient = curr_accumulated_gradients[name]

                if noise_multiplier > 0 and max_grad_norm is not None:
                        # add noise
                        curr_param_gradient = curr_param_gradient + noise_multiplier * max_grad_norm * torch.randn_like(curr_param_gradient)

                # update gradient of parameter
                param.grad = curr_param_gradient
                
        # update parameter
        optimizer.step()

    return model

def test_model(model, X, y, batch_size=128):
    """Test trained model on test set"""
    test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    model.eval()
    acc = 0
    with torch.no_grad():
        for curr_X, curr_y in test_loader:
            curr_y_hat = torch.argmax(model(curr_X), dim=1)
            acc += torch.sum(curr_y_hat == curr_y).cpu().item()
    model.train()
    
    return acc / len(y)

def save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only):
    """Save checkpoint"""
    # create folder if not exists
    os.makedirs(out_folder, exist_ok=True)

    # save random state
    random_state = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state()
    }
    dill.dump(random_state, open(f'{out_folder}/random_state.dill', 'wb'))

    # save intermediate values
    if fit_world_only:
        np.save(f'{out_folder}/outputs_{fit_world_only}.npy', outputs[fit_world_only])
        np.save(f'{out_folder}/losses_{fit_world_only}.npy', losses[fit_world_only])
        np.save(f'{out_folder}/all_losses_{fit_world_only}.npy', all_losses[fit_world_only])

        if fit_world_only == 'out':
            np.save(f'{out_folder}/train_set_accs.npy', train_set_accs)
            np.save(f'{out_folder}/test_set_accs.npy', test_set_accs)
    else:
        np.save(f'{out_folder}/outputs_in.npy', outputs['in'])
        np.save(f'{out_folder}/outputs_out.npy', outputs['out'])
        np.save(f'{out_folder}/train_set_accs.npy', train_set_accs)
        np.save(f'{out_folder}/test_set_accs.npy', test_set_accs)
        np.save(f'{out_folder}/losses_in.npy', losses['in'])
        np.save(f'{out_folder}/losses_out.npy', losses['out'])
        np.save(f'{out_folder}/all_losses_in.npy', all_losses['in'])
        np.save(f'{out_folder}/all_losses_out.npy', all_losses['out'])

def resume_checkpoint(out_folder, fit_world_only, resume):
    """Load checkpoint if resume is set to True and previous checkpoint exists, else create new empty checkpoint"""
    outputs = {'out': [], 'in': []}
    losses = {'out': [], 'in': []}
    all_losses = {'in': [], 'out': []}
    train_set_accs = []
    test_set_accs = []

    if os.path.exists(out_folder) and resume:
        # if folder exists and resume is set to true load previous values
        random_state = dill.load(open(f'{out_folder}/random_state.dill', 'rb'))
        np.random.set_state(random_state['np'])
        torch.random.set_rng_state(random_state['torch'])

        if fit_world_only:
            outputs[fit_world_only] = np.load(f'{out_folder}/outputs_{fit_world_only}.npy').tolist()
            losses[fit_world_only] = np.load(f'{out_folder}/losses_{fit_world_only}.npy').tolist()
            all_losses[fit_world_only] = np.load(f'{out_folder}/all_losses_{fit_world_only}.npy').tolist()

            if fit_world_only == 'out':
                train_set_accs = np.load(f'{out_folder}/train_set_accs.npy').tolist()
                test_set_accs = np.load(f'{out_folder}/test_set_accs.npy').tolist()
        else:
            outputs['in'] = np.load(f'{out_folder}/outputs_in.npy').tolist()
            outputs['out'] = np.load(f'{out_folder}/outputs_out.npy').tolist()
            train_set_accs = np.load(f'{out_folder}/train_set_accs.npy').tolist()
            test_set_accs = np.load(f'{out_folder}/test_set_accs.npy').tolist()
            losses['in'] = np.load(f'{out_folder}/losses_in.npy').tolist()
            losses['out'] = np.load(f'{out_folder}/losses_out.npy').tolist()
            all_losses['in'] = np.load(f'{out_folder}/all_losses_in.npy').tolist()
            all_losses['out'] = np.load(f'{out_folder}/all_losses_out.npy').tolist()
    else:
        # create folder and dump initial values in
        os.makedirs(out_folder, exist_ok=True)
        save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only)
    
    return outputs, losses, all_losses, train_set_accs, test_set_accs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use (mnist, cifar10, cifar100)')
    parser.add_argument('--model_name', type=str, default='lr', choices=list(Models.keys()), help='model to audit')
    parser.add_argument('--n_reps', type=int, default=200, help='number of models')
    parser.add_argument('--n_df', type=int, default=0, help='|D| (0 => use full dataset)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='gradient clipping norm')
    parser.add_argument('--epsilon', type=float, default=None, help='privacy parameter, epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='privacy parameter, delta')
    parser.add_argument('--target_type', type=str, default='blank', help='sample to use as target (blank, clipbkd, badnets, or path to target sample)')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--out', type=str, default='exp_data/', help='folder to write results to')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device to use (cpu, cuda:X)')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='', help='initialize all models to the same weights (if path provided, weights loaded from path (worst-case), else fix to some randomly chosen weights)')
    parser.add_argument('--block_size', type=int, default=1000, help='process samples within a batch in blocks to conserve GPU space')
    parser.add_argument('--resume', action='store_true', help='skip experiment if results are present')
    parser.add_argument('--fit_world_only', type=str, default=None, choices=['in', 'out'], help='just fit models in world and calculate losses')
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level for empirical eps estimation')
    parser.add_argument('--badnets_label', type=int, default=-1, help='assign badnets poison this label')

    # Options for Debugging
    parser.add_argument('--view_badnets', action='store_true')
    parser.add_argument('--store_canary_rank', action='store_true')
    parser.add_argument('--holdout_audit', action='store_true')

    # Options for Spectral Signature Search
    parser.add_argument('--find_outliers', action='store_true')
    parser.add_argument('--search_space', type=str, default='gradient')
    parser.add_argument('--scoring_fn', type=str, default='norm')

    # Options for Forgetting Canary Candidates
    parser.add_argument('--defense', type=str, default='', help='use filtering defense during audit')

    args = parser.parse_args()

    if args.max_grad_norm == -1: args.max_grad_norm = None

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(f'{out_folder}/models', exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    spectral_signature_args = None
    if args.find_outliers:
        spectral_signature_args = {'search_space': args.search_space, 
                                'scoring_fn': args.scoring_fn,
                                'store_canary_rank': [] if args.store_canary_rank else None, 
                                'out': out_folder,
                                'n_epochs': args.n_epochs,
                                'canary_dropped': False
                            }

    # load data (define D-)
    if args.n_df == 1:
        # load single data point
        X_out, y_out, out_dim = load_data(args.data_name, 1, device=device)
    else:
        # since n_df is 0 by default, loads full dataset
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1, device=device)

    init_model = None
    if args.fixed_init is not None:
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim, model_name=args.model_name).to(device)

        if args.fixed_init == '':
            # initialize model (average-case)
            if args.model_name == 'cnn':
                xavier_init_model(init_model)
            else:
                kaiming_init_model(init_model)
        else:
            # load weights from path (worst-case)
            init_model.load_state_dict(torch.load(args.fixed_init))
            # don't train on the first half of the dataset
            X_out, y_out = X_out[len(X_out) // 2:], y_out[len(y_out) // 2:]

            gc.collect()
    
    # craft target data point (x_T, y_T)
    if args.target_type == 'blank':
        # blank sample
        target_X = torch.zeros_like(X_out[[0]])
        target_y = torch.from_numpy(np.array([9])).to(device)
    elif args.target_type == 'badnets':
        target_X = X_out[-1]
        print('Original Label:', y_out[-1])
        target_y = torch.tensor(args.badnets_label).to(device)
        target_X[:, -4:, -4:] = torch.max(target_X)

        target_X = target_X.unsqueeze(0)
        target_y = target_y.unsqueeze(0)

        if args.view_badnets:
            plt.imshow(target_X.squeeze().numpy(), cmap='gray')
            plt.savefig(f'badnets_{args.badnets_label}.png')

    elif args.target_type == 'sanity_check':
        target_X = X_out[-1].unsqueeze(0)
        target_y = y_out[-1].unsqueeze(0)

    elif args.target_type == 'clipbkd':
        # ClipBKD sample
        target_X, target_y = craft_clipbkd(X_out, init_model, device)
    elif os.path.exists(args.target_type):
        # pre-crafted target sample
        target_X = torch.from_numpy(np.load(args.target_type)).to(device)
        if init_model is not None:
            target_y =  choose_worstcase_label(init_model, target_X)
        else:
            target_y = torch.from_numpy(np.array([9])).to(device)
    else:
        raise Exception(f'Target {args.target_type} not found')

    # define D = D- U {(x_T, y_T)}
    X_in, y_in = torch.vstack((X_out[:-1], target_X)), torch.cat((y_out[:-1], target_y))
    X_test, y_test, _ = load_data(args.data_name, None, split='test', device=device)
    
    # train M on D and D-
    # resume from checkpoint
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    models = {'in': [], 'out': []}
    outputs, losses, all_losses, train_set_accs, test_set_accs = resume_checkpoint(out_folder, args.fit_world_only, args.resume)

    for world in worlds:
        # set dataset according to "world"
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)

        # check how many reps initially completed
        reps_completed = len(losses[world])

        for rep in tqdm(range(reps_completed, args.n_reps // 2), initial=reps_completed, total=args.n_reps // 2):
            # train model
            model = train_model(args.model_name, 
                                            curr_X, 
                                            curr_y, 
                                            target_X, 
                                            target_y, 
                                            args.epsilon, 
                                            args.delta,
                                            args.max_grad_norm, 
                                            args.n_epochs, 
                                            args.lr, 
                                            spectral_signature_args, 
                                            device=device, 
                                            init_model=init_model,
                                            block_size=args.block_size, 
                                            out_dim=out_dim, 
                                            use_defense=args.defense, 
                                            store_canary_rank=args.store_canary_rank)
            
                        
            # get loss of model on target sample
            model.eval()
            with torch.no_grad():
                output = model(target_X)
                outputs[world].append(output[0].cpu().numpy())
                losses[world].append(-nn.CrossEntropyLoss()(output, target_y).cpu().item())

                
            # get test set accuracy from first 5 reps
            if rep < 5 and world == 'in':
                if len(X_out) > 0:
                    train_set_accs.append(test_model(model, X_in, y_in))
                    print('Train set acc:', train_set_accs[-1])
                test_set_accs.append(test_model(model, X_test, y_test))
                print('Test set acc:', test_set_accs[-1])

            if rep < 1 and world == 'in':
                if spectral_signature_args and spectral_signature_args['store_canary_rank'] is not None:
                    canary_ranks = np.array(spectral_signature_args['store_canary_rank'])
                    np.save(f'{out_folder}/canary_ranks.npy', canary_ranks)

                    plt.plot(np.arange(len(canary_ranks))[canary_ranks > -1], canary_ranks[canary_ranks > -1])
                    plt.xlabel('Epoch #')
                    plt.ylabel('Canary Outlier Score Rank')
                    plt.gca().invert_yaxis()
                    plt.grid(True)
                    plt.savefig(f'{out_folder}/canary_ranks.png')

            torch.cuda.empty_cache()

            # save checkpoint
            save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only)
        outputs[world] = np.array(outputs[world])

    if not args.fit_world_only:
        def audit_canary(losses, args):        
            k = len(losses['in'])
            t_losses = {'in': None, 'out': None}
            holdout_losses = {'in': None, 'out': None}

            if args.holdout_audit:
                k = len(losses['in']) // 2
            
            t_losses['in'] = losses['in'][:k]
            t_losses['out'] = losses['out'][:k]
            holdout_losses['in'] = losses['in'][k:]
            holdout_losses['out'] = losses['out'][k:]

            # calculate empirical epsilon using GDP
            mia_scores = np.concatenate([t_losses['in'], t_losses['out']])
            mia_labels = np.concatenate([np.ones_like(t_losses['in']), np.zeros_like(t_losses['out'])])

            max_t, emp_eps_loss = compute_eps_lower_from_mia(mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1)

            if args.holdout_audit:
                emp_eps_loss = compute_eps_lower_from_mia_given_t(np.concatenate(
                    [holdout_losses['in'], holdout_losses['out']]), 
                    np.concatenate([np.ones_like(holdout_losses['in']), np.zeros_like(holdout_losses['out'])]), 
                    args.alpha, 
                    args.delta, 
                    max_t, 
                    'GDP')
            
            return emp_eps_loss, mia_scores, mia_labels
        
        emp_eps_loss, mia_scores, mia_labels = audit_canary(losses, args)

        np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps_loss])
        np.save(f'{out_folder}/mia_scores.npy', mia_scores)
        np.save(f'{out_folder}/mia_labels.npy', mia_labels)
    
        print(f'Theoretical eps: {args.epsilon}')
        print(f'Empirical eps: {emp_eps_loss}')

    print(f'Train set accuracy: {np.mean(train_set_accs) * 100:.3f}%')
    print(f'Test set accuracy: {np.mean(test_set_accs) * 100:.3f}%')