"""TP1 Deep Learning : reconnaissance de chiffres MNIST avec un réseau codé à la main.

Le script charge MNIST, prépare les données, puis compare trois variantes
d'entraînement (Q13 à Q15) : base, Xavier, Xavier + momentum.
"""
import time

import matplotlib.pyplot as plt
import torch
import torchvision

import utils

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
c1_size = 784      # couche d'entrée : 28 x 28 pixels
c2_size = 30       # première couche cachée
c3_size = 20       # deuxième couche cachée
output_size = 10   # une sortie par chiffre
alpha = 0.05       # pas d'apprentissage
m = 12             # taille des minibatchs
n_epochs = 10      # nombre d'epochs par entraînement

# ---------------------------------------------------------------------------
# Chargement des données (téléchargées la première fois)
# ---------------------------------------------------------------------------
train_set = torchvision.datasets.MNIST(root='.', train=True, download=True)
images_train = train_set.data
labels_train = train_set.targets

test_set = torchvision.datasets.MNIST(root='.', train=False, download=True)
images_test = test_set.data
labels_test = test_set.targets

print("Tailles des images (apprentissage, test) :")
print(images_train.size(), images_test.size())
print("Tailles des étiquettes (apprentissage, test) :")
print(labels_train.size(), labels_test.size())

# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------
# Images : (N, 28, 28) entiers 0..255 -> (N, 784) réels entre 0 et 1
images_train = images_train.view(60000, 784) / 255
images_test = images_test.view(10000, 784) / 255

# Étiquettes : entiers 0..9 -> vecteurs one-hot à 10 colonnes
labels_train = torch.nn.functional.one_hot(labels_train, num_classes=10)
labels_test = torch.nn.functional.one_hot(labels_test, num_classes=10)

print("Après prétraitement :")
print(images_train.size(), images_test.size(), labels_train.size(), labels_test.size())

# ---------------------------------------------------------------------------
# Réseau non entraîné : on attend environ 10 % de précision
# ---------------------------------------------------------------------------
untrained = utils.net(c1_size, c2_size, c3_size, output_size)
print("Précision avant entraînement :",
      utils.accuracy(untrained, images_train, labels_train))

# ---------------------------------------------------------------------------
# Comparaison des variantes : (nom, initialisation Xavier, gamma du momentum)
# ---------------------------------------------------------------------------
variantes = [
    ("base (N(0,1), sans momentum)", False, 0.0),
    ("Xavier, sans momentum", True, 0.0),
    ("Xavier + momentum (gamma=0.4)", True, 0.4),
]

resultats = {}
for nom, xavier, gamma in variantes:
    print(f"\n=== {nom} ===")
    reseau = utils.net(c1_size, c2_size, c3_size, output_size, xavier=xavier)
    t0 = time.time()
    Lp = utils.train(reseau, images_train, labels_train,
                     images_test, labels_test, alpha, n_epochs, m=m, gamma=gamma)
    print(f"temps : {time.time() - t0:.1f} s")
    resultats[nom] = Lp

# ---------------------------------------------------------------------------
# Courbes : taux d'erreur sur le jeu de test en fonction de l'epoch
# ---------------------------------------------------------------------------
for nom, Lp in resultats.items():
    erreur_test = [1 - test for (_, test) in Lp]
    plt.plot(erreur_test, label=nom)
plt.xlabel("epoch")
plt.ylabel("taux d'erreur (test)")
plt.yscale("log")
plt.legend()
plt.show()
