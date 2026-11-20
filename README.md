# TP1 Deep Learning : réseau de neurones from scratch sur MNIST

Réseau de neurones codé à la main avec des tenseurs PyTorch (sans `nn.Module` ni autograd) pour reconnaître des chiffres manuscrits.

## Principe

- **Données** : MNIST, images 28×28 aplaties en vecteurs de 784 valeurs entre 0 et 1, étiquettes en one-hot.
- **Réseau** : 784 → 30 → 20 → 10, activation logistique sur toutes les couches. La classe prédite est le neurone de sortie le plus activé.
- **Entraînement** : rétropropagation et descente de gradient par minibatchs (m = 12, α = 0.05), avec perte cross-entropy.
- **Variantes comparées** : initialisation normale simple, initialisation Xavier, Xavier + momentum (γ = 0.4).

## Fichiers

- `utils.py` : classe `net` (forward, backward), classe `SGD`, fonctions `accuracy`, `epoch`, `train`.
- `tp_deep_mnist.py` : charge les données, entraîne les trois variantes et trace le taux d'erreur test par epoch.

## Utilisation

```bash
pip install torch torchvision matplotlib
python tp_deep_mnist.py
```

MNIST est téléchargé automatiquement dans le dossier courant à la première exécution.
