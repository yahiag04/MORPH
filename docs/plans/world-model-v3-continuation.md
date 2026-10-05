# MORPH World Model V3 — piano di continuazione

**Aggiornato:** 5 ottobre 2026
**Ultimo commit pubblicato:** `b1d675c` — `docs: record world model v3 training results`

Questo file registra lo stato verificato del progetto, l'ordine delle attività rimanenti e i criteri per dichiararlo pronto. Per i contratti tecnici completi consultare [`world-model-v3.md`](world-model-v3.md); questo documento serve a riprendere il lavoro senza rifare attività già concluse.

## Obiettivo e limiti da dichiarare

Costruire e valutare un world model action-conditioned per la manipolazione Panda–scatola–vassoio in MuJoCo: stato strutturato e comando applicato in ingresso, traiettorie future e contatti in uscita, poi selezione e controllo a breve orizzonte.

Il risultato riguarda la simulazione e usa lo stato strutturato del simulatore. Non dimostra trasferimento su un robot fisico, previsione video dai pixel o capacità VLA. Il video umano già nel README è una dimostrazione separata, non una validazione del controllo reale.

## Stato verificato

### Completato e pubblicato

- Task 0–3: inventario dati e ambienti, loader/audit, valutatore, modello V3 e confronti iniziali 4D/8D con seed 17, 29 e 43.
- Task 4: raccolta appaiata di 100 famiglie di sviluppo, 8 varianti ciascuna; 800 episodi, 772.095 transizioni, 640 train e 160 validation. Nessun errore di raccolta; replay verificato su tutti gli 800 episodi con errore massimo assoluto 0. Il test finale è ancora escluso.
- Task 5: implementazione e test pubblicati in `84d4314`. Sono disponibili sampler centrato sugli eventi, loss per gruppi, loss multistep autoregressiva, confronti uniform/event e curriculum 25→50→100.

I test V3 sono passati (41 test) e quelli del collector controfattuale sono passati (5 test). Sono inoltre passati CLI help, compilazione Python e `git diff --check`. Non aggiungere pesi/checkpoint, dataset grezzi, video originali o percorsi personali a Git.

### M1 e M2 event completo; ablation M2 uniforme in corso

M1, configurazione `configs/world_model_v3/m1-expanded.json`, seed 17, 3 membri, patience 8, CPU, è completo nel run locale `world-model-v3/M1-expanded-seed17-20261004`. Usa lo split train/validation combinato degli archivi corretti; il test non è stato caricato per scegliere il modello.

Epoche e checkpoint selezionati:

| Membro | Stato | Epoche completate | Best validation loss | Epoca del best |
| --- | --- | ---: | ---: | ---: |
| 0 | completato | 26 | 0,09915 | 18 |
| 1 | completato | 27 | 0,09881 | 19 |
| 2 | completato | 40 | 0,09419 | 34 |

M1 ha prodotto `validation.json` e `validation_events.json`. RMSE posizione scatola, validation uniforme: 1,31/4,20/7,23/13,15 mm a 0,1/0,5/1/2 s; persistenza: 3,04/9,36/15,39/24,80 mm. Finestre centrate sugli eventi: 3,32/13,69/13,53/14,10 mm contro 13,09/35,13/40,44/49,17 mm per la persistenza. F1 degli eventi di contatto a un passo: 0,637 pinza e 0,654 supporto. Report completo e intervalli bootstrap sono negli artefatti locali; il validation ha 29 famiglie, quindi l'evidenza resta limitata. L'orientamento durante gli eventi resta un limite.

M2 con oversampling eventi è stato avviato il 5 ottobre 2026, seed 17, sullo stesso split, configurazione `configs/world_model_v3/event.json`, output locale `world-model-v3/M2-event-seed17-20261005`. Controllare prima lo stato effettivo e non lanciare duplicati. Le epoche precedenti impiegavano circa 57–60 secondi.

M2 event è completo: membri terminati dopo `[22,28,23]` epoche; best validation loss `[0.12042,0.12806,0.12272]`. Errore posizione scatola uniforme a 0,1/0,5/1/2 s: `1,34/4,79/7,64/12,40 mm`; centrato sugli eventi: `2,61/10,46/11,66/13,13 mm`. Rispetto a M1, migliora le finestre evento e la previsione a 2 s, con lieve regressione sulle finestre uniformi a 0,5–1 s. F1 eventi a un passo: pinza `0,667`, supporto `0,651`. Sono 29 famiglie di validation: non è ancora una conferma multi-seed.

L'ablation con campionamento uniforme è stata avviata il 5 ottobre, seed 17, configurazione `configs/world_model_v3/event_no_oversampling.json`, output `world-model-v3/M2-uniform-seed17-20261005`. Prima di riprendere, controllare il processo e `status.json`.

## Ripresa del run attivo

Dalla root del worktree, individuare prima gli interpreti e i dati già presenti. I dati restano fuori dal repository:

```bash
export MORPH_DATA_ROOT="$HOME/Documents/MORPH-local-data"
export MORPH_V3_RUNS="$MORPH_DATA_ROOT/world-model-v3"
export MORPH_TRAIN_PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
export MORPH_CONTACT_DATA="$MORPH_DATA_ROOT/contact-expansion-corrected-20261004"
export MORPH_COUNTERFACTUAL_DATA="$MORPH_V3_RUNS/counterfactual-development-compact-20261004"

ps -Ao pid,etime,command | rg 'train_world_model_v3.py'
cat "$MORPH_V3_RUNS/M2-uniform-seed17-20261005/status.json"
```

Se l'ablation uniforme è ancora attiva, **non avviarne una copia**. Se è terminata senza `status: complete`, controllare il log e riprendere solo se gli hash del manifest corrispondono:

```bash
PYTHONPATH=src "$MORPH_TRAIN_PY" scripts/train_world_model_v3.py \
  --dataset-dir "$MORPH_CONTACT_DATA" \
  --dataset-dir "$MORPH_COUNTERFACTUAL_DATA" \
  --config configs/world_model_v3/event_no_oversampling.json \
  --seed 17 \
  --output-dir "$MORPH_V3_RUNS/M2-uniform-seed17-20261005" \
  --resume
```

Il training completo genera `validation.json` (finestre uniformi) e `validation_events.json` (finestre centrate sugli eventi). Salvare fuori da Git i report completi; nei risultati pubblici esportare solo gli aggregati necessari.

## Ordine delle attività rimanenti

### 1. Chiudere Task 5: selezione del modello di dinamica

1. Attendere la fine di M1 e registrare durata reale, epoca scelta e report uniform/event.
2. Allenare `configs/world_model_v3/event.json` (M2 con oversampling eventi), poi `configs/world_model_v3/event_no_oversampling.json` (ablation uniforme), sullo stesso split combinato e seed 17. Usare directory distinte e lo stesso massimo, patience, batch e architettura di M1.
3. Confrontare almeno posizione/velocità oggetto, orientamento, giunti, F1 di ciascun contatto, F1 degli eventi, coverage e finestre in movimento; separare metriche uniformi da quelle centrate sugli eventi. Non scegliere solo dalla loss complessiva.
4. M3 curriculum usa gli orizzonti 25/50/100 per 15/15/10 epoche al massimo. Il pilot già eseguito serviva solo a misurare il costo: 59/112/220 secondi per epoca; non è una prova di convergenza. Avviarlo solo dopo aver stimato il tempo totale. Non superare H=50 nei task successivi se H=100 diverge o non supera le soglie.
5. Selezionare i due candidati migliori solo su validation. Confermarli con seed 29 e 43, mantenendo lo split; seed 17 è la prima replica. Per isolare l'effetto dei dati, confrontare con M1 allenato sugli stessi dati espansi. Non usare il test storico per scegliere.
6. Se due esperimenti sulla stessa ipotesi non migliorano le metriche, chiudere la diramazione con esito negativo e seguire la diagnosi nel piano principale (allineamento/unità, baseline dinamiche, parziale osservabilità, copertura eventi, coerenza fisica, modi di contatto).

Comandi base per M2; cambiare config e nome cartella per l'ablation, poi seed per le conferme:

```bash
PYTHONPATH=src "$MORPH_TRAIN_PY" scripts/train_world_model_v3.py \
  --dataset-dir "$MORPH_CONTACT_DATA" \
  --dataset-dir "$MORPH_COUNTERFACTUAL_DATA" \
  --config configs/world_model_v3/event.json \
  --seed 17 \
  --output-dir "$MORPH_V3_RUNS/M2-event-seed17-20261004"
```

### 2. Task 6: incertezza e ranking entro scena

- Conservare traiettorie separate per membro durante tutto il rollout; misurare costi e incertezza fisica senza fondere metri, radianti e probabilità senza pesi espliciti.
- Valutare le varianti dell'archivio appaiate dallo stesso stato iniziale; confrontare ranking e regret con punteggio V2, casuale e sempre nominale.
- Riportare pareggi, copertura, famiglie informative e intervalli bootstrap. Calibrare eventuali soglie solo su validation. Meno di 30 famiglie informative significa evidenza insufficiente per una conclusione forte.
- Se il ranking lungo fallisce, non dichiarare risolta la selezione globale. Un controller a orizzonte breve è ammissibile solo se supera separatamente le soglie di dinamica e scelta locale.

### 3. Task 7: adapter di controllo

- Aggiungere l'hook opzionale dopo il comando nominale/IK. Senza hook o con comando identico al nominale, stati, tempi, sequenza, successo e comportamento devono restare invariati.
- Validare dimensione `(8,)`, finitezza e `ctrlrange`; applicare la correzione per un solo intervallo di 100 ms e ricalcolare il nominale dallo stato osservato.
- Registrare comando nominale, proposto ed effettivo, fallback e fase. Testare identity, NaN, fuori limite, hold-time e regressioni nell'ambiente MuJoCo.

### 4. Task 8: MPC appreso a breve orizzonte

- Implementare random shooting batched con 64 candidati; includere sempre il candidato zero-correzione, limiti attuatori, fallback e misura di latenza. Massimo orizzonte iniziale 1 s, o 0,5 s se solo quello supera la validazione.
- Il pianificatore usa solo stato, comando nominale e contesto disponibili adesso, più le proprie previsioni. Non usare stati futuri del dataset, esiti, rollout MuJoCo delle alternative o lookup.
- Applicare solo il primo comando, osservare, poi pianificare di nuovo; nessun passo MuJoCo dentro il valutatore candidati.
- Confrontare sugli stessi stati iniziali nominale, ricerca casuale, MPC appreso e ablation `beta=0`; conservare fallimenti e timeout. Riportare successo appaiato, errore, latenza mediana/p95 e fallback.

### 5. Task 9: test finale, report e showcase

- Congelare modello/configurazione/costo/limiti/orizzonte e hash del protocollo **prima** del test.
- Raccogliere 100 famiglie finali nuove, seed `20261042`, senza usarle per training, calibrazione o selezione. Test dinamico e confronto online devono usare le stesse condizioni iniziali nominali e i tre checkpoint congelati.
- Riportare intervalli appaiati per famiglia e tutti i risultati negativi. Se ci sono meno di 30 famiglie informative, dichiarare insufficiente l'evidenza; non sostituire il test dopo aver visto i risultati.
- Valutare i checkpoint congelati anche sui replay dei 16 video come diagnostica storica separata, senza riaddestramento e senza chiamarla validazione reale.
- Suite completa di regressione alla fine; dichiarare test passati e test saltati per dipendenze. Creare metriche e figura research-style; mantenere il confronto video umano/Panda già pubblicato e aggiungere un confronto nominale/MPC solo se i risultati lo giustificano.
- Aggiornare README con problema, dati, metodo, protocollo, baseline/ablation, intervalli, errori, latenza, riproducibilità e limiti. Chiamare rendering MuJoCo “simulazione”, e output del modello “predizione”; non presentarli come video generati dal modello.
- Fare revisione di file staged, `git diff --check`, dimensioni e contenuti privati; committare e pushare per milestone, senza forzare la storia. Verificare che il remoto contenga l'ultimo commit.

## Criteri quantitativi già fissati

- Dinamica: RMSE oggetto a 0,5 s e 1 s sotto la migliore persistenza/velocità costante; F1 degli eventi sopra persistenza. Controllare anche moto e contatti.
- Selezione entro scena: ranking ≥0,70, intervallo bootstrap 95% con limite inferiore >0,50 e almeno 30 famiglie informative.
- Controllo: miglioramento appaiato di almeno 10 punti percentuali, intervallo 95% della differenza sopra zero; riportare fallback e breakdown per seed.
- Contare sempre rollout invalidi e riparazioni quaternion; non rimuoverli silenziosamente.

Sono obiettivi fissati prima della conferma, non risultati già raggiunti né garanzie. Il progetto deve riportare fedelmente se una soglia viene mancata.

## Cosa serve da Yahia

Per proseguire non servono altre riprese, un robot fisico, Colab o spesa cloud: dati e codice necessari alle attività correnti sono già presenti. **L'unica cosa pratica adesso è lasciare il Mac acceso e impedire che vada in stop finché il training M1 in corso non termina.** Se il training viene interrotto, i checkpoint permettono di controllare manifest e riprendere; non serve rifarlo da zero.

L'eventuale uso di risorse cloud o una nuova raccolta di video/dati richiede prima una stima di costo e un motivo sperimentale concreto. Una futura validazione fisica sarà un'estensione distinta e richiederà hardware e misure spaziali calibrate.
