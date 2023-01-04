Sito web di questo repository: https://ccaudek.github.io/ds4psy_2023/


# Creazione dell'ambiente virtuale

In linea di principio, l'ambiente virtuale può essere installato utilizzando

```{bash}
conda env create -f environment.yml
```

sulla riga di comando. Questo richiede molto, molto tempo per l'installazione. La costruzione dell'ambiente virtuale è lenta perché `ds4psy_2023` dipende da MOLTI pacchetti, che a loro volta hanno molte dipendenze.