<<<<<<< HEAD
from eigenfaces import Eigenfaces
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ks = []
    thes = []
    for i in range(5, 100, 5):
        thes.append(i/100)
        eig = Eigenfaces("./orl_faces", i/100)
        ks.append(eig.k)

    plt.figure()
    plt.plot(thes, ks, '-m')
    plt.xlabel("threshold")
    plt.ylabel("value of k")
    plt.suptitle('Dependency between the number of principal '
                 'components and threshold')
    plt.savefig('graph.jpg')
=======
from eigenfaces import Eigenfaces
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ks = []
    thetas = []

    for i in range(5, 100, 5):
        thetas.append(i/100)
        eig = Eigenfaces("./orl_faces", i/100)
        ks.append(eig.k)

    plt.figure()
    plt.plot(thetas, ks, '-m')
    plt.xlabel("threshold")
    plt.ylabel("value of k")
    plt.suptitle('Dependency between the number of principal '
                 'components and threshold')
    plt.savefig('graph.jpg')
>>>>>>> 1f7c8d25cb31c51171a318e89d368313513ed870
