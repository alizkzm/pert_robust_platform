#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from scipy import linalg
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA
import os
import sklearn.decomposition
import scipy.stats
import warnings
from utils import iterative_A


def _cov(X, shrinkage=-1):
    emp_cov = np.cov(np.asarray(X).T, bias=1)
    if shrinkage < 0:
        return emp_cov
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov


def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    means ： array-like of shape (n_classes, n_features)
        Outer classes means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]

    means_ = np.zeros(shape=(len(classes), X.shape[1]))
    for i in range(len(classes)):
        means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)    
    return means, means_


def split_data(data: np.ndarray, percent_train: float):
    split = data.shape[0] - int(percent_train * data.shape[0])
    return data[:split], data[split:]


def feature_reduce(features: np.ndarray, f: int=None):
    """
        Use PCA to reduce the dimensionality of the features.
        If f is none, return the original features.
        If f < features.shape[0], default f to be the shape.
	"""
    if f is None:
        return features
    if f > features.shape[0]:
        f = features.shape[0]
    
    return sklearn.decomposition.PCA(
        n_components=f,
        svd_solver='randomized',
        random_state=1919,
        iterated_power=1).fit_transform(features)


class TransferabilityMethod:	
    def __call__(self, 
        features: np.ndarray, y: np.ndarray,
                ) -> float:
        self.features = features		
        self.y = y
        return self.forward()

    def forward(self) -> float:
        raise NotImplementedError


class PARC(TransferabilityMethod):
	
    def __init__(self, n_dims: int=None, fmt: str=''):
        self.n_dims = n_dims
        self.fmt = fmt

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)
        
        num_classes = len(np.unique(self.y, return_inverse=True)[0])
        labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y

        return self.get_parc_correlation(self.features, labels)

    def get_parc_correlation(self, feats1, labels2):
        scaler = sklearn.preprocessing.StandardScaler()

        feats1  = scaler.fit_transform(feats1)

        rdm1 = 1 - np.corrcoef(feats1)
        rdm2 = 1 - np.corrcoef(labels2)
        
        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)
        
        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100

    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]


class SFDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        
    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(_cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = _cov(X, shrinkage=self.shrinkage) 

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N

        '''
        self.classes_ = np.unique(y)
        #n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage,)

        return self
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return softmax(scores)


def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        # k = min(N, D)
        N, D = f.shape  

        # direct SVD may be expensive
        if N > D: 
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k, s.shape = k, vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            
            # x has shape [k, 1], but actually x should have shape [N, 1]
            x = u.T @ y_  
            x2 = x ** 2
            # if k < N, we compute sum of xi for 0 singular values directly
            res_x2 = (y_ ** 2).sum() - x2.sum()  

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point
    #_fit = _fit_icml

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels

        :return: LogME score (how well f can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  
        # return np.argmax(logits, axis=-1)
        return prob


def LEEP(X, y, model_name='resnet50'):

    n = len(y)
    num_classes = len(np.unique(y))

    # read classifier
    # Group1: model_name, fc_name, model_ckpt
    ckpt_models = {
        'densenet121': ['classifier.weight', './models/group1/checkpoints/densenet121-a639ec97.pth'],
        'densenet169': ['classifier.weight', './models/group1/checkpoints/densenet169-b2777c0a.pth'],
        'densenet201': ['classifier.weight', './models/group1/checkpoints/densenet201-c1103571.pth'],
        'resnet34': ['fc.weight', './models/group1/checkpoints/resnet34-333f7ec4.pth'],
        'resnet50': ['fc.weight', './models/group1/checkpoints/resnet50-19c8e357.pth'],
        'resnet101': ['fc.weight', './models/group1/checkpoints/resnet101-5d3b4d8f.pth'],
        'resnet152': ['fc.weight', './models/group1/checkpoints/resnet152-b121ed2d.pth'],
        'mnasnet1_0': ['classifier.1.weight', './models/group1/checkpoints/mnasnet1.0_top1_73.512-f206786ef8.pth'],
        'mobilenet_v2': ['classifier.1.weight', './models/group1/checkpoints/mobilenet_v2-b0353104.pth'],
        'googlenet': ['fc.weight', './models/group1/checkpoints/googlenet-1378be20.pth'],
        'inception_v3': ['fc.weight', './models/group1/checkpoints/inception_v3_google-1a9a5a14.pth'],
    }
    ckpt_loc = ckpt_models[model_name][1]
    fc_weight = ckpt_models[model_name][0]
    fc_bias = fc_weight.replace('weight', 'bias')
    ckpt = torch.load(ckpt_loc, map_location='cpu')
    fc_weight = ckpt[fc_weight].detach().numpy()
    fc_bias = ckpt[fc_bias].detach().numpy()

    # p(z|x), z is source label
    prob = np.dot(X, fc_weight.T) + fc_bias
    prob = softmax(prob)   # p(z|x), N x C(source)

    pyz = np.zeros((num_classes, 1000))  # C(source) = 1000
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n
    
    pz = np.sum(pyz, axis=0)     # marginal probability
    py_z = pyz / pz              # conditional probability, C x C(source)
    py_x = np.dot(prob, py_z.T)  # N x C

    # leep = E[p(y|x)]
    leep_score = np.sum(py_x[np.arange(n), y]) / n
    return leep_score


def NLEEP(X, y, component_ratio=5):

    n = len(y)
    num_classes = len(np.unique(y))
    # PCA: keep 80% energy
    pca_80 = PCA(n_components=0.8)
    pca_80.fit(X)
    X_pca_80 = pca_80.transform(X)

    # GMM: n_components = component_ratio * class number
    n_components_num = component_ratio * num_classes
    gmm = GaussianMixture(n_components= n_components_num).fit(X_pca_80)
    prob = gmm.predict_proba(X_pca_80)  # p(z|x)
    
    # NLEEP
    pyz = np.zeros((num_classes, n_components_num))
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n   
    pz = np.sum(pyz, axis=0)    
    py_z = pyz / pz             
    py_x = np.dot(prob, py_z.T) 

    # nleep_score
    nleep_score = np.sum(py_x[np.arange(n), y]) / n
    return nleep_score


def LogME_Score(X, y):

    logme = LogME(regression=False)
    score = logme.fit(X, y)
    return score


def SFDA_Score(X, y):

    n = len(y)
    num_classes = len(np.unique(y))
    
    SFDA_first = SFDA()
    prob = SFDA_first.fit(X, y).predict_proba(X)  # p(y|x)
    
    # soften the probability using softmax for meaningful confidential mixture
    prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True) 
    means, means_ = _class_means(X, y)  # class means, outer classes means
    
    # ConfMix
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        y_prob = np.take(prob, indices, axis=0)
        y_prob = y_prob[:, y_]  # probability of correctly classifying x with label y        
        X[indices] = y_prob.reshape(len(y_prob), 1) * X[indices] + \
                            (1 - y_prob.reshape(len(y_prob), 1)) * means_[y_]
    
    SFDA_second = SFDA(shrinkage=SFDA_first.shrinkage)
    prob = SFDA_second.fit(X, y).predict_proba(X)   # n * num_cls

    # leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
    sfda_score = np.sum(prob[np.arange(n), y]) / n
    return sfda_score


def PARC_Score(X, y, ratio=2):
    
    num_sample, feature_dim = X.shape
    ndims = 32 if ratio > 1 else int(feature_dim * ratio)  # feature reduction dimension

    if num_sample > 15000:
        from utils_cr import initLabeled
        p = 15000.0 / num_sample
        labeled_index = initLabeled(y, p=p)
        features = X[labeled_index]
        targets = X[labeled_index]
        print("data are sampled to {}".format(features.shape))

    method = PARC(n_dims = ndims)
    parc_score = method(features=X, y=y)

    return parc_score


def extract_model_weights(model_name, source, target=None, is_pseudo_fine_tuned=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_pseudo_fine_tuned:
        path = f'models/fine-tuned-pseudo-label/{source}_{target}/{model_name}.pth'
    else:
        if source != 'imagenet':
            path = f'models/group1/checkpoints/{source}/{model_name}.pth'
        else:
            path = None
    
    MODEL_DICT = {
        'resnet34': torch.models.resnet34,
        'resnet50': torch.models.resnet50,
        'resnet101': torch.models.resnet101,
        'resnet152': torch.models.resnet152,
        'densenet121': torch.models.densenet121,
        'densenet169': torch.models.densenet169,
        'densenet201': torch.models.densenet201,
        'mobilenet_v2': torch.models.mobilenet_v2,
        'mnasnet1_0': torch.models.mnasnet1_0,
        'inception_v3': torch.models.inception_v3,
    }
    
    DATA_DICT = {
        'imagenet': 1000,
        'cifar10': 10,
        'cifar100': 100,
        'caltech101': 101,
        'dtd': 47,
        'aircraft': 100,
        'cars': 196,
        'flowers': 102,
        'food': 101,
        'pets': 37,
        'sun397': 397,
        'voc2007': 20
    }
    
    if model_name == 'googlenet':
        model = torch.models.googlenet(pretrained=(source == 'imagenet' and not is_pseudo_fine_tuned), 
                                 aux_logits=False)
    elif model_name == 'inception_v3':
        model = torch.models.inception_v3(pretrained=(source == 'imagenet' and not is_pseudo_fine_tuned), 
                                    aux_logits=True)
    else:
        model = MODEL_DICT[model_name](pretrained=(source == 'imagenet' and not is_pseudo_fine_tuned))
    
    target_dataset = target if is_pseudo_fine_tuned else source
    if target_dataset != 'imagenet' or is_pseudo_fine_tuned:
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Linear(model.fc.in_features, DATA_DICT[target_dataset])
        if model_name == 'inception_v3' and hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, DATA_DICT[target_dataset])
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, torch.nn.Sequential):
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, DATA_DICT[target_dataset])
            else:
                model.classifier = torch.nn.Linear(model.classifier.in_features, DATA_DICT[target_dataset])
    
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
    
    model.eval()
    
    all_weights = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'bn' not in name and 'batch' not in name:
                all_weights[name] = param.data.cpu().numpy().flatten()
    
    return all_weights


def Wasserstein_Score(X, y, model_name='resnet50', source='imagenet', target=None):
    source_weights = extract_model_weights(model_name, source)
    fine_tuned_weights = extract_model_weights(model_name, source, target=target, is_pseudo_fine_tuned=True)
    
    common_layers = set(source_weights.keys()).intersection(set(fine_tuned_weights.keys()))
    
    if not common_layers:
        raise ValueError("No common weight layers found between models")
    
    distances = []
    
    for layer_name in common_layers:
        source_layer = source_weights[layer_name]
        fine_tuned_layer = fine_tuned_weights[layer_name]
        
        layer_distance = wasserstein_distance(source_layer, fine_tuned_layer)
        distances.append(layer_distance)
    
    average_distance = np.mean(distances)
    transferability_score = 1.0 / (1.0 + average_distance)
    
    return transferability_score