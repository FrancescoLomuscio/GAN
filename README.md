# GAN

Varie versioni di GAN che ho realizzato per imparare questa architettura.

##### GAN_SIN
La prima versione realizzata è quella per generare valore della funzione *sin(x)*.
Con lr = 0.001, 300 epoche e Adam come ottimizzatore per discriminator e generator si ottiene il seguente risultato:

![Risultati GAN_SIN](https://github.com/FrancescoLomuscio/GAN/blob/master/GAN_SIN/output_adam_001_300.png)

##### GAN_MNIST
La seconda versione realizzata è quella per generare valore della funzione numeri interi scritti a mano compresi tra 0 e 9 usando il dataset MNIST.
Con lr = 0.0001, 50 epoche e Adam come ottimizzatore per discriminator e generator si ottiene il seguente risultato:

![Risultati GAN_MNIST](https://github.com/FrancescoLomuscio/GAN/blob/master/GAN_MNIST/resultGANMNIST.png)
