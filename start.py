from VectorizedDataset import VectorizedDataset
from SpectrumKernel import SpectrumKernel
from SVMClassifier import SVMClassifier
import kernels
import utils

path = 'data/'

Kernel = SpectrumKernel

n_grams_list = [4,3,2]

VD = VectorizedDataset(Kernel, path, n_grams_list)

SVMC = {}
Cs = [0.9, 1.4, 1.2]
for i in range(3):
    SVMC[i] = SVMClassifier(kernel=kernels.rbf, C=Cs[i])
    SVMC[i].fit(VD.X[i]['train'], VD.Y[i]['train'])
    print("fit done for training {}".format(i))
    print("Training accuracy for classifier {} : ".format(i) + str(utils.compute_val_accuracy(SVMC[i], VD.X[i]['train'], VD.Y[i]['train'])))
    print("Validation accuracy for classifier {} : ".format(i) +  str(utils.compute_val_accuracy(SVMC[i], VD.X[i]['val'], VD.Y[i]['val'])))

utils.generate_submission_file(SVMC, path, n_grams_list, submission_filename = 'Yte.csv')
