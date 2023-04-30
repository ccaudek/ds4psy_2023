(sec-regr-intro)=
# Introduzione

In psicologia, i ricercatori vogliono trovare connessioni tra variabili e confrontare le condizioni sperimentali. La correlazione di Pearson aiuta a descrivere le relazioni tra variabili, ma non è sufficiente perché il ricercatore vuole descrivere la relazione tra le variabili nella popolazione e non solo nel campione. Il modello di regressione lineare viene utilizzato per questo scopo e utilizza la funzione matematica più semplice, la funzione lineare, per descrivere la relazione tra le variabili. Il modello di regressione lineare permette al ricercatore di fare inferenze sulla relazione tra le variabili e di comprendere le proprietà geometriche della funzione lineare. In questo e nei successivi capitoli vedremo come costruire un modello statistico basato sull'approccio bayesiano utilizzando il modello di regressione lineare.

## La funzione lineare

Iniziamo con un ripasso sulla funzione di lineare. Si chiama *funzione lineare* una funzione del tipo

$$
f(x) = a + b x,
$$

dove $a$ e $b$ sono delle costanti. Il grafico di tale funzione è una retta di cui il parametro $b$ è detto *coefficiente angolare* e il parametro $a$ è detto *intercetta* con l'asse delle $y$ \[infatti, la retta interseca l'asse $y$ nel punto $(0,a)$, se $b \neq 0$\].

Per assegnare un'interpretazione geometrica alle costanti $a$ e $b$ si consideri la funzione

$$
y = b x.
$$

Tale funzione rappresenta un caso particolare, ovvero quello della *proporzionalità diretta* tra $x$ e $y$. Il caso generale della linearità

$$
y = a + b x
$$

non fa altro che sommare una costante $a$ a ciascuno dei valori $y = b x$. Nella funzione lineare $y = a + b x$, se $b$ è positivo allora $y$ aumenta al crescere di $x$; se $b$ è negativo $y$ diminuisce al crescere di $x$; se $b=0$ la retta è orizzontale, ovvero $y$ non muta al variare di $x$.

Consideriamo ora più in dettaglio il coefficiente $b$. Si consideri un punto $x_0$ e un incremento arbitrario $\varepsilon$, come indicato nella @fig-linearfunction. Le differenze $\Delta x = (x_0 + \varepsilon) - x_0$ e $\Delta y = f(x_0 + \varepsilon) - f(x_0)$ sono detti *incrementi* di $x$ e $y$. Il coefficiente angolare $b$ è uguale al rapporto

$$
b = \frac{\Delta y}{\Delta x} = \frac{f(x_0 + \varepsilon) - f(x_0)}{(x_0 + \varepsilon) - x_0},
$$

indipendentemente dalla grandezza degli incrementi $\Delta x$ e $\Delta y$. Il modo più semplice per assegnare un'interpretazione geometrica al coefficiente angolare (o pendenza) della retta è quello di porre $\Delta x = 1$. In tali circostanze, $b = \Delta y$.

```{figure} images/linear_function.png
---
height: 300px
name: linear_function-fig
---
La funzione lineare $y = a + bx$.
```

Possiamo dunque dire che la pendenza $b$ di un retta è uguale all'incremento $\Delta y$ associato ad un incremento unitario nella $x$.


## Una media per ciascuna osservazione

In precedenza abbiamo visto come stimare i parametri di un modello bayesiano nel quale le osservazioni sono indipendenti e identicamente distribuite secondo una densità gaussiana,

$$
Y_i \stackrel{i.i.d.}{\sim} \mathcal{N}(\mu, \sigma), \quad i = 1, \dots, n.
$$ (eq-normalsamplingmodel)

Il modello dell'eq. {eq}`eq-normalsamplingmodel` assume che ogni $Y_i$ sia la realizzazione di una v.c. distribuita come $\mathcal{N}(\mu, \sigma^2)$. Da un punto di vista bayesiano, questo modello può essere implementato imponendo delle distribuzioni a priori ai parametri $\mu$ e $\sigma$ e generando la verosimiglianza in base ai dati osservati. Per esempio, possiamo usare le seguenti distribuzioni a priori.

$$
\begin{align}
Y_i \mid \mu, \sigma & \stackrel{iid}{\sim} \mathcal{N}(\mu, \sigma^2)\notag\\
\mu & \sim \mathcal{N}(\mu_0, \tau^2) \notag\\
\sigma & \sim \mbox{Cauchy}(x_0, \gamma) \notag
\end{align}
$$

Con queste informazioni, possono poi essere trovate le distribuzioni a posteriori dei parametri {cite:p}`gelman2020regression`. Vediamo ora come sia possibile estendere questo modello bayesiano in modo che possa descrivere la *relazione lineare* tra due variabili.

## Relazione lineare tra la media $y \mid x$ e il predittore

Spesso il ricercatore si trova a osservare altre variabili che sono associate ad ogni risposta $y_i$. Una di queste variabili viene chiamata $x$ e, nel contesto del modello di regressione, viene definita come *predittore* o *variabile indipendente*. L'obiettivo del ricercatore è di predire il valore di $y_i$ a partire dal valore di $x_i$. Quindi, come possiamo estendere il modello dell'equazione {eq}`eq-normalsamplingmodel` per studiare la relazione tra $y_i$ e $x_i$?

L'equazione {eq}`eq-normalsamplingmodel` rappresenta un modello statistico che assume che tutte le osservazioni $Y_i$ abbiano una media comune $\mu$. Tuttavia, se vogliamo considerare anche una nuova variabile $x_i$ che assume valori diversi per ogni $Y_i$, dobbiamo modificare il modello. In particolare, invece della media comune $\mu$, introduciamo una media $\mu_i$ specifica per ciascuna osservazione $Y_i$:

$$
Y_i \mid \mu_i, \sigma \stackrel{ind}{\sim} \mathcal{N}(\mu_i, \sigma), \quad i = 1, \dots, n.
$$ (eq-normalsamplinglinearmodel)

Questa nuova equazione, chiamata *modello di regressione lineare*, ci permette di studiare la relazione tra $Y_i$ e $x_i$ considerando $\mu_i$ come una funzione lineare di $x_i$.

L'eq. {eq}`eq-normalsamplinglinearmodel` afferma che ciascuna osservazione $Y_i$ è estratta casualmente dalla distribuzione $\mathcal{N}(\mu_i, \sigma)$, dove $\mu_i$ è la media associata alla $i$-esima osservazione e $\sigma$ è la deviazione standard comune a tutte le osservazioni. Per modellare la relazione tra il predittore $x_i$ e la risposta $Y_i$, il modello di regressione assume che la media della distribuzione da cui abbiamo estratto $Y_i$, ovvero $\mu_i$, sia una funzione lineare del predittore $x_i$, ovvero $\mu_i = \beta_0 + \beta_ 1 x_i$, dove $\beta_0$ e $\beta_ 1$ sono i parametri incogniti rappresentanti l'intercetta e la pendenza della retta di regressione, rispettivamente. 

In altre parole, il modello di regressione suppone che esista una relazione lineare tra $x_i$ e $Y_i$ e che ogni valore di $Y_i$ sia una realizzazione di una variabile casuale con media $\mu_i$ e deviazione standard $\sigma$. 

Nell'eq. {eq}`eq-regmodel`, $x_i$ è considerata una costante nota e $\beta_0$ e $\beta_ 1$ sono variabili casuali, che vengono stimate tramite l'inferenza bayesiana. Questo procedimento consiste nell'assegnare una distribuzione a priori a $\beta_0$ e a $\beta_1$, trovare la verosimiglianza dei dati e calcolare la distribuzione a posteriori dei parametri $\beta_0$ e $\beta_1$.

Il modello dell'eq. {eq}`eq-regmodel` postula che il valore atteso di ciascuna osservazione $Y_i$ sia una funzione lineare del corrispondente predittore $x_i$. La costante $\beta_0$ rappresenta il valore atteso di $Y_i$ quando $x_i=0$, mentre la costante $\beta_1$ rappresenta l'incremento atteso di $Y_i$ quando $x_i$ aumenta di un'unità.

Va sottolineato che il modello fornisce una stima del valore atteso $\mu_i$ e non del valore effettivo di ciascuna osservazione $Y_i$. In altre parole, la relazione lineare tra $\mu_i$ e $x_i$ non può essere utilizzata per prevedere il valore esatto di $Y_i$ per un dato valore di $x_i$, ma solo per fornire una stima del valore atteso di $Y_i$ per quel dato valore di $x_i$.

## Il modello lineare

Sostituendo la relazione lineare dell'eq. {eq}`eq-regmodel` nell'eq. {eq}`eq-normalsamplinglinearmodel`, otteniamo il modello di regressione lineare:

$$
Y_i \mid \beta_0, \beta_ 1, \sigma \stackrel{ind}{\sim} \mathcal{N}(\beta_0 + \beta_ 1 x_i, \sigma), \quad i = 1, \dots, n.
$$ (eq-samplinglinearmodel)

In questo modello, ogni osservazione $Y_i$ è estratta a caso da una distribuzione Normale con media $\beta_0 + \beta_ 1 x_i$, dove $\beta_0$ è l'intercetta e $\beta_1$ è la pendenza della retta di regressione. La deviazione standard $\sigma$ rappresenta la varianza comune a tutte le osservazioni. Questo modello è chiamato "lineare" perché la relazione tra $Y$ e $x$ è lineare e "bivariato" perché coinvolge solo due variabili, $Y$ e $x$.

### Assunzioni

Il modello di regressione lineare bivariato assume che la variabile $x$ sia fissa e costante tra i diversi campioni. Per ogni valore di $x$, il modello ipotizza che la variabile $y$ segua una distribuzione Normale di media $\mu_i = \alpha + \beta x_i$ e deviazione standard $\sigma$, dove $\alpha$ e $\beta$ sono i parametri del modello. Questa è l'assunzione di normalità e linearità del modello. Il modello assume anche che le distribuzioni condizionate $p(y \mid x_i)$ siano omoschedastiche, cioè che abbiano la stessa deviazione standard $\sigma$ per tutti i valori di $x_i$. Infine, il modello assume che i dati campionati siano indipendenti e che gli errori $\varepsilon_i$ siano distribuiti in maniera Normale con media zero e deviazione standard $\sigma$. In altre parole, il modello ipotizza che ogni osservazione $y_i$ sia una realizzazione della variabile casuale $Y = y_i \mid X = x_i$.

## Commenti e considerazioni finali 

Il modello di regressione lineare bivariato viene utilizzato per studiare la connessione lineare tra due variabili $x$ e $Y$, e per determinare la direzione e l'entità di tale legame. Inoltre, questo modello statistico consente di fare previsioni sul valore della variabile dipendente $Y$, sulla base del valore assunto dalla variabile indipendente $x$.
