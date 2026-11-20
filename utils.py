"""Réseau de neurones à 3 couches (784-30-20-10) codé "à la main" avec PyTorch.

Conventions (celles de PyTorch, reprises du sujet) :
- un neurone = une colonne, une image = une ligne ;
- une matrice de poids w[l] a pour forme (n_{l-1}, n_l) ;
- un minibatch de m images est un tenseur de forme (m, 784).
"""
import math

import torch


# ---------------------------------------------------------------------------
# Fonction d'activation
# ---------------------------------------------------------------------------

def logistic(x):
    """Fonction logistique L(x) = 1 / (1 + exp(-x)), appliquée composante par composante."""
    return 1 / (1 + torch.exp(-x))


def logistic_d(x):
    """Dérivée de la logistique : L'(x) = L(x) * (1 - L(x))."""
    s = logistic(x)
    return s * (1 - s)


# ---------------------------------------------------------------------------
# Réseau
# ---------------------------------------------------------------------------

class net():
    """Perceptron multicouche avec activation logistique sur toutes les couches.

    Attributs :
        sizes   : tuple des tailles de couches (entrée, cachée 1, cachée 2, sortie).
        weights : liste [w1, w2, w3], w_l de forme (n_{l-1}, n_l).
        bias    : liste [b1, b2, b3], b_l de forme (n_l,).
    """

    def __init__(self, c1_size, c2_size, c3_size, output_size, xavier=True):
        """Crée le réseau.

        Les biais sont initialisés à 0. Les poids suivent une loi normale
        de moyenne 0 et d'écart-type :
            - sqrt(2 / (n_{l-1} + n_l)) si xavier=True (initialisation Xavier/Glorot) ;
            - 1 si xavier=False (initialisation de base du sujet).
        """
        self.sizes = (c1_size, c2_size, c3_size, output_size)
        self.bias = [
            torch.zeros(c2_size),
            torch.zeros(c3_size),
            torch.zeros(output_size),
        ]
        self.weights = []
        for l in range(1, 4):
            n_in, n_out = self.sizes[l - 1], self.sizes[l]
            std = math.sqrt(2 / (n_in + n_out)) if xavier else 1.0
            self.weights.append(torch.normal(0, std, size=(n_in, n_out)))

    def forward(self, a0):
        """Propagation en avant.

        Entrée : a0, tenseur (m, 784) (m images, m peut valoir 1 ou tout le jeu de données).
        Sortie : (Lz, La)
            Lz = [z1, z2, z3]      préactivations, z_l = a_{l-1} w_l + b_l
            La = [a0, a1, a2, a3]  activations,    a_l = L(z_l)
        Le biais (n_l,) est ajouté à chaque ligne par broadcasting.
        """
        a = a0
        La = [a0]
        Lz = []
        for l in range(1, 4):
            z = a @ self.weights[l - 1] + self.bias[l - 1]
            Lz.append(z)
            a = logistic(z)
            La.append(a)
        return Lz, La

    def backward(self, La, Lz, label):
        """Rétropropagation sur un minibatch.

        Entrées : La, Lz renvoyés par forward ; label, tenseur one-hot (m, 10).
        Sortie : (gradients_w, gradients_b), listes indexées de 0 à 2 (couches 1 à 3).

        Les gradients renvoyés sont des SOMMES sur le minibatch (pas des moyennes) :
        c'est SGD.step qui divise par m. Ainsi poids et biais suivent la même convention.

        Formules (cross-entropy + logistique en sortie) :
            Delta_3     = a_3 - y
            Delta_{l-1} = (Delta_l w_l^T) o L'(z_{l-1})
            dLoss/dw_l  = a_{l-1}^T Delta_l   (somme sur les m images)
            dLoss/db_l  = somme des lignes de Delta_l
        """
        # Ldelta[k] contient Delta_{3-k} : on part de la sortie et on remonte.
        Ldelta = [La[-1] - label]
        for l in (3, 2):
            # Lz[l-2] est z_{l-1} (Lz est indexée à partir de la couche 1)
            Ldelta.append((Ldelta[-1] @ self.weights[l - 1].T) * logistic_d(Lz[l - 2]))
        Ldelta.reverse()  # maintenant Ldelta[l-1] = Delta_l

        gradients_w = [La[l - 1].T @ Ldelta[l - 1] for l in range(1, 4)]
        gradients_b = [Ldelta[l - 1].sum(0) for l in range(1, 4)]
        return gradients_w, gradients_b


# ---------------------------------------------------------------------------
# Optimiseur
# ---------------------------------------------------------------------------

class SGD():
    """Descente de gradient stochastique par minibatch, avec momentum optionnel.

    Mise à jour (m = taille du minibatch, g = gradient sommé sur le minibatch) :
        u_t = gamma * u_{t-1} + (alpha / m) * g_t
        w_t = w_{t-1} - u_t
    Avec gamma = 0, on retrouve la descente de gradient simple.
    """

    def __init__(self, val_alpha, network, val_gamma=0.4):
        """val_alpha : pas d'apprentissage ; network : réseau à optimiser ;
        val_gamma : coefficient d'inertie (0 = pas de momentum)."""
        self.alpha = val_alpha
        self.net = network
        self.gamma = val_gamma
        # Termes d'inertie, initialement nuls, de même forme que les paramètres
        self.uw = [torch.zeros_like(w) for w in network.weights]
        self.ub = [torch.zeros_like(b) for b in network.bias]

    def step(self, a0, label):
        """Un pas de descente sur le minibatch (a0, label).

        La taille m du minibatch est déduite de a0, ce qui gère automatiquement
        le dernier minibatch (plus petit) d'une epoch.
        """
        m = a0.shape[0]
        Lz, La = self.net.forward(a0)
        grad_w, grad_b = self.net.backward(La, Lz, label)
        for l in range(3):
            self.uw[l] = self.gamma * self.uw[l] + (self.alpha / m) * grad_w[l]
            self.ub[l] = self.gamma * self.ub[l] + (self.alpha / m) * grad_b[l]
            self.net.weights[l] = self.net.weights[l] - self.uw[l]
            self.net.bias[l] = self.net.bias[l] - self.ub[l]


# ---------------------------------------------------------------------------
# Évaluation et entraînement
# ---------------------------------------------------------------------------

def accuracy(net, images, labels):
    """Précision du réseau : proportion d'images correctement classées.

    images : (n, 784) ; labels : (n, 10) en one-hot.
    La classe prédite est l'indice du neurone de sortie le plus activé.
    Tout le jeu est passé en un seul forward (pas de boucle sur les images).
    """
    pred = net.forward(images)[1][-1].argmax(dim=1)
    truth = labels.argmax(dim=1)
    ntrue = (pred == truth).sum().item()
    return ntrue / images.size()[0]


def epoch(images, labels, opt, m=12):
    """Une epoch : parcourt tout le jeu d'entraînement par minibatchs de m images.

    Les images sont mélangées (même permutation pour images et étiquettes),
    puis prises m par m. Si la taille du jeu n'est pas multiple de m,
    le dernier minibatch contient les images restantes.
    """
    size = images.size()[0]
    perm = torch.randperm(size)
    imagesp = images[perm]
    labelsp = labels[perm]
    for c in range(0, size, m):
        # le slicing s'arrête de lui-même à la fin : dernier batch éventuellement plus petit
        opt.step(imagesp[c:c + m], labelsp[c:c + m])


def train(net, images_train, labels_train, images_test, labels_test,
          alpha, n, m=12, gamma=0.4):
    """Entraîne le réseau pendant n epochs.

    Paramètres : alpha = pas d'apprentissage, m = taille de minibatch,
    gamma = coefficient de momentum (0 pour le désactiver).
    Renvoie la liste des couples (précision train, précision test), de longueur n+1
    (le premier couple correspond au réseau avant entraînement).
    """
    opt = SGD(alpha, net, gamma)
    Lp = [(accuracy(net, images_train, labels_train),
           accuracy(net, images_test, labels_test))]
    for i in range(n):
        epoch(images_train, labels_train, opt, m)
        Lp.append((accuracy(net, images_train, labels_train),
                   accuracy(net, images_test, labels_test)))
        print(f"epoch {i + 1} terminée : train {Lp[-1][0]:.4f}, test {Lp[-1][1]:.4f}")
    return Lp
