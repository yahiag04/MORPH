# MORPH — World Model V3: specifica e piano operativo

**Stato:** piano da eseguire; nessuna delle modifiche V3 descritte qui è già implementata.
**Base verificata:** `main`, commit `7ed4052`, 4 ottobre 2026.
**Obiettivo:** ottenere un world model action-conditioned che predica dinamica dell'oggetto e contatti e dimostri un miglioramento misurabile delle decisioni di manipolazione in MuJoCo.
**Architettura:** ensemble di dinamiche apprese su stato strutturato, dati con azioni alternative dalla stessa situazione, valutazione per scenario e controllo predittivo a orizzonte breve. Il controller esistente fornisce una proposta nominale; il modello ne valuta possibili correzioni.
**Tecnologie:** Python, NumPy, PyTorch opzionale, MuJoCo, unittest, Matplotlib, FFmpeg per i video.
**Specifica:** le sezioni 1–7 di questo documento definiscono le decisioni da rispettare; le sezioni successive definiscono l'esecuzione. Riferimenti storici: `docs/designs/world-model-v2.md` e `docs/plans/contact-data-expansion.md`.
**Metodo di lavoro:** esecuzione sequenziale, una milestone alla volta; spuntare le checklist solo dopo aver registrato l'evidenza. Conservare un diario locale degli esperimenti per riprendere dopo un'interruzione.

## 1. Risultato richiesto e confini

Il modello deve imparare le conseguenze delle azioni: `stato attuale + comando -> distribuzione di stati futuri`. Deve continuare a prevedere posizioni, velocità e contatti; un classificatore successo/fallimento da solo non soddisfa questo obiettivo.

Il risultato atteso è un modello solido **nel dominio dichiarato: Panda, scatola, vassoio e fisica MuJoCo**. La stima dello stato sfrutta inizialmente lo stato strutturato del simulatore. I video umani forniscono dimostrazioni e traiettorie, ma non misurano lo stato completo di un robot reale. Un risultato positivo in simulazione non prova trasferimento sul robot fisico, né previsione video dai pixel, né capacità VLA.

L'esecuzione si divide in tre consegne, ognuna utilizzabile e documentabile:

1. **A — Diagnosi e dinamica:** dati corretti, confronto azioni cartesiane/attuatori, valutazione dei contatti e modello migliore su validation.
2. **B — Decisioni offline:** confronto tra sequenze alternative dalla stessa condizione iniziale, con evidenza che il modello distingue conseguenze utili e dannose.
3. **C — Controllo:** uso del modello durante l'esecuzione, con aumento del successo su scenari nuovi.

Completare A prima di investire in C. Se una soglia sperimentale non è raggiunta, salvare il risultato negativo e seguire il percorso diagnostico indicato; non chiamare il progetto migliorato solo perché il software esegue il training.

## 2. Baseline da preservare

Fonti: `results/metrics/world_model_fidelity.json`, `results/metrics/synthetic_world_model.json`, `results/metrics/contact_dataset.json`, `README.md`.

| Esperimento | Dati | Evidenza attuale |
| --- | --- | --- |
| Replay delle dimostrazioni | 16 video, 32 episodi, 25.920 transizioni | 7/16 successi per ciascun metodo; esiti identici tra direct e confidence-aware per ogni video |
| Dinamica da dimostrazioni, 0,5 s | 25 passi da 20 ms | RMSE oggetto: modello 0,00394 m, persistenza 0,00097 m |
| Dinamica da dimostrazioni, 11,2 s | prefisso comune di 560 passi | RMSE oggetto: 0,10639/0,10933 m; contact accuracy/F1 0,5/0,5 |
| Selezione tra video | 16 candidati sorgente | ranking 0,5; regret globale 1,0; confronto esplorativo tra scene diverse |
| Dataset sintetico | 240 episodi, 60 famiglie, 207.670 transizioni | 71 successi; posizioni, yaw e comandi variabili; massa, attrito e geometria fissi |
| Modello sintetico, 0,5 s | 36 episodi test, 9 famiglie | RMSE oggetto 0,01104/0,01575 m; contact F1 0,969 contro persistenza 0,951 |

Precisazioni da mantenere:

- I 32 replay non sono 32 dimostrazioni umane indipendenti.
- Non ci sono coppie direct/confidence-aware con esito discordante: la selezione entro video non è dimostrata. Un regret pari a zero su queste coppie non prova che il modello sappia scegliere.
- Il punteggio attuale è la frazione di sei criteri fisici soddisfatti, non una probabilità di successo calibrata.
- Il dataset sintetico e quello derivato dai video devono avere report distinti.
- I risultati sintetici attuali usano un modello allenato su 31 famiglie, con 11 famiglie interne per early stopping e 9 ulteriori famiglie per calibrazione. Un nuovo protocollo 42 train/9 validation va confrontato con una baseline riaddestrata con lo stesso protocollo.
- I test storici sono già stati esaminati. Restano confronti di regressione; la conferma finale V3 richiede nuovi scenari riservati.

## 3. Diagnosi: fatti e ipotesi

| Osservazione verificata | Implicazione da verificare |
| --- | --- |
| La rete usa 37 valori di stato e 4 comandi; gli archivi conservano anche 8 comandi effettivi agli attuatori | I comandi effettivi potrebbero ridurre l'ambiguità introdotta dall'IK e dal mantenimento del comando |
| Il controllo aggiorna ogni 100 ms, le osservazioni arrivano ogni 20 ms | Cinque transizioni condividono il comando; timing e azione applicata devono coincidere |
| Il training usa finestre di 25 passi | La previsione di episodi di molti secondi può accumulare errori fuori dall'orizzonte addestrato |
| Le finestre di training sono campionate senza priorità agli eventi di contatto | Eventi brevi ma decisivi possono essere poco rappresentati |
| La calibrazione V2 ammette guadagni residui pari a zero | Una buona media può nascondere gruppi di stato quasi congelati; riportare sempre i guadagni |
| Il rollout V2 fa avanzare lo stato medio dell'ensemble | Può nascondere traiettorie alternative e sottostimare l'incertezza accumulata |

Queste sono motivazioni per esperimenti controllati. Non attribuire il fallimento a una singola causa senza un confronto misurato.

## 4. Vincoli globali

- Preservare API, checkpoint e report V2. Implementare V3 in moduli separati e produrre nuovi report con nomi V3.
- Mantenere gli originali dei video, dati per episodio, snapshot, checkpoint e log dettagliati fuori da Git. Pubblicare codice, configurazioni riproducibili e aggregati privi di identificativi locali.
- Ogni famiglia di scenari, con tutte le sue varianti e tutti i suoi snapshot, appartiene a una sola partizione.
- Normalizzazione, pesi di classe, campionamento e iperparametri dipendono solo da train/validation. Nessuna scelta dipende dal test finale.
- Nessuna etichetta di successo o fase futura negli input della dinamica. Le fasi registrate possono servire per campionare e valutare; al planner si può dare solo la fase disponibile nel controller al momento della decisione.
- Usare gli stessi sensori, controller nominale, condizioni iniziali, timeout e criterio di successo nei confronti di controllo.
- Durante la valutazione delle alternative, il planner appreso può usare solo il proprio modello. I rollout MuJoCo delle alternative servono alla raccolta dati e all'eventuale baseline oracle, etichettata separatamente.
- Il modello deve prevedere con l'azione che verrà realmente applicata. Se un fallback applica un'altra azione, registrare quella effettiva.
- Non cambiare la definizione di successo per alzare i numeri. Non selezionare solo seed, scene o video favorevoli.
- Nessuna spesa cloud o trasferimento dei dati verso servizi esterni senza una scelta esplicita dell'utente.
- Commit con l'identità Git già configurata e messaggi tecnici. Il push periodico è autorizzato; rispettare le protezioni del repository. Non forzare push, non sovrascrivere modifiche altrui, non aggiungere firme o riferimenti agli strumenti usati per scrivere il progetto.
- Lavorare in un nuovo branch/worktree isolato dopo aver controllato `git worktree list`. Lasciare intatti gli altri worktree.

## 5. Ambienti e dati locali

Questi percorsi sono convenzioni locali da verificare, non percorsi da inserire nei report pubblici:

```bash
export MORPH_DATA_ROOT="${HOME}/Documents/MORPH-local-data"
export MORPH_SYNTH_DATA="${MORPH_DATA_ROOT}/contact-expansion-corrected-20261004"
export MORPH_VIDEO_DATA="${MORPH_DATA_ROOT}/demonstration-feedback-20261003/object-relative-v2"
export MORPH_V3_RUNS="${MORPH_DATA_ROOT}/world-model-v3"
export MORPH_SIM_PY="${HOME}/Documents/.venv/bin/python"
export MORPH_TRAIN_PY="$(command -v python3)"
```

Al momento della stesura:

- `MORPH_SYNTH_DATA` contiene l'archivio corretto: 240 episodi, 207.670 transizioni. Esiste anche `contact-expansion-20261003`, storico: non sceglierlo solo perché appare prima nella lista.
- `contact-world-model-corrected-20261004` contiene il checkpoint e i risultati sintetici corretti.
- `MORPH_VIDEO_DATA` contiene `direct_transitions.npz`, `confidence-aware_transitions.npz`, `per_clip/` e `world-model-fidelity-cv/`.
- L'ambiente `MORPH_SIM_PY` ha MuJoCo e le dipendenze della pipeline. `python3` ha PyTorch 2.14.0 ma non MuJoCo. Python è 3.14; ricontrollare versioni e compatibilità prima di intervenire.

Raccolta e training offline possono usare processi diversi comunicando tramite NPZ/JSON. Per il controllo online serve un ambiente dedicato con entrambe le dipendenze: crearlo solo alla milestone C, senza modificare gli ambienti funzionanti. Verificare prima se i pacchetti sono già disponibili; non reinstallare tutto a ogni sessione. Registrare le versioni effettive usate.

## 6. Contratti dati, modello e artefatti

### 6.1 Stato e azioni

Lo stato pubblico della prima V3 resta di 37 valori:

| Slice Python | Significato |
| --- | --- |
| `0:3` | posizione end-effector, m |
| `3:10`, `10:17` | posizioni e velocità dei 7 giunti, rad e rad/s |
| `17:20`, `20:24` | posizione oggetto e quaternione `wxyz` |
| `24:27`, `27:30` | velocità lineare e angolare oggetto |
| `30`, `31` | apertura e velocità pinza |
| `32`, `33` | contatto pinza/oggetto e supporto/oggetto |
| `34:37` | centro del vassoio, costante nell'episodio |

Due modalità di input per l'ablation:

- `cartesian4`: array `actions` esistente, target XYZ e comando apertura; dimensione 4.
- `actuator8`: array `actuator_controls` esistente; dimensione 8. Sono **comandi di controllo MuJoCo**, non necessariamente coppie o aperture in metri. Salvare ordine, nomi e limiti degli attuatori; non assumere che il comando della pinza sia espresso nelle stesse unità di `actions[:, 3]`.

La prima implementazione online usa `actuator8`, perché può applicare esattamente la sequenza valutata. `cartesian4` rimane il controllo sperimentale. Se `actuator8` fallisce le soglie di affidabilità, risolvere quel problema prima del planner; non promuovere automaticamente la modalità cartesiana a un planner con semantica diversa.

Input opzionale successivo: concatenare `package_xyz - ee_xyz` e `tray_xyz - package_xyz`, cioè 6 feature relative deterministiche. Si calcolano sullo stato corrente, incluso quello predetto durante il rollout. Non modificano il formato di stato archiviato.

Archivi: conservare entrambe le rappresentazioni delle azioni e i campi `episode_ids`, `group_ids`, `splits`, `source_kinds`, tempi di inizio/fine e intervallo. Il loader V3 rifiuta comandi mancanti, NaN, dimensioni errate, episodi interleaved, interruzioni temporali e sovrapposizioni tra partizioni. Usare `allow_pickle=False`.

### 6.2 Intervalli e finestra temporale

Per questi archivi: `physics_dt=0.002 s`, `observation_dt=0.02 s`, `control_dt=0.1 s`. Validare i valori dal manifest; niente inferenze silenziose dal numero di passi.

| Secondi | Passi di previsione a 20 ms |
| --- | --- |
| 0,1 | 5 |
| 0,5 | 25 |
| 1,0 | 50 |
| 2,0 | 100 |

Le finestre non attraversano episodi, famiglie o gap. Escludere esplicitamente gli episodi troppo corti e riportarne il numero. La valutazione breve usa finestre anche attorno agli eventi; quella lunga include rollout dall'inizio dell'episodio. Non confondere un endpoint a 0,5 s dall'inizio con la qualità in tutte le fasi del task.

### 6.3 Interfacce da implementare

Creare `src/world_model/v3/` con `__init__.py` e questi contratti; le classi dati risiedono in `contracts.py`:

```python
@dataclass(frozen=True)
class DynamicsConfig:
    action_mode: str  # cartesian4 oppure actuator8
    relative_features: bool = False
    ensemble_size: int = 3
    hidden_dim: int = 128
    observation_dt: float = 0.02

@dataclass
class EpisodeBatch:
    states: np.ndarray       # [N, 37]
    actions: np.ndarray      # [N, 4] oppure [N, 8]
    next_states: np.ndarray  # [N, 37]
    episode_ids: np.ndarray  # [N]
    group_ids: np.ndarray    # [N]
    phases: np.ndarray       # [N], metadati; non input della dinamica
    start_times: np.ndarray  # [N]
    end_times: np.ndarray    # [N]
    metadata: dict

@dataclass
class RolloutBatch:
    states: np.ndarray      # [E, K, H+1, 37], una traiettoria per membro/candidato
    contact_probs: np.ndarray  # [E, K, H, 2]
    valid: np.ndarray       # [E, K], False se un rollout diverge

@dataclass
class PlannerDecision:
    control: np.ndarray    # [8], comando effettivo per il prossimo intervallo
    fallback: bool
    diagnostics: dict      # costo, incertezza, candidati, tempo di calcolo
```

`model.py` definisce `DynamicsV3` con:

- `predict_member(member_index: int, states, actions) -> (next_states, contact_logits)` su tensori PyTorch con batch.
- `rollout(initial_states: np.ndarray, actions: np.ndarray) -> RolloutBatch`, input `[K,37]` e `[K,H,A]`.
- `save(destination: Path, manifest: dict) -> None` e `load(source: Path, expected: dict | None = None) -> DynamicsV3`.

Ogni membro evolve il proprio stato. Un valore medio è ammesso per visualizzare, non deve rientrare come stato comune dei membri. Le probabilità di contatto entrano nel rollout in `[0,1]`; le soglie binarie servono per le metriche, con la stessa convenzione documentata in train e inferenza. I target osservati restano binari.

Mantenere fisso il contesto del vassoio. Normalizzare i quaternioni e gestire il caso di norma quasi nulla usando l'ultimo quaternione valido, contando questi interventi nei diagnostici. Per le metriche di orientamento considerare `q` e `-q` equivalenti.

### 6.4 Manifest e ripresa

Ogni run ha una directory distinta con `config.json`, `manifest.json`, `metrics.json`, checkpoint, log e `status.json`. Il manifest contiene almeno:

- versione schema, commit e hash dei sorgenti che influenzano il training;
- hash dei manifest/dati di input e degli assegnamenti alle partizioni;
- nomi/ordine delle feature, modalità azioni, unità, intervallo temporale;
- seed, versione Python/PyTorch/NumPy/MuJoCo dove applicabile, dispositivo;
- configurazione loss/sampler, normalizzazione solo train, epoca migliore;
- criteri usati per scegliere il checkpoint, costo temporale e motivo dell'arresto.

Non riusare checkpoint solo perché il nome del file esiste. `load(..., expected=...)` deve rifiutare differenze in schema, feature, azioni, intervallo, dataset o partizioni. Salvare stato ottimizzatore e RNG se si promette una ripresa esatta del training. Separare `last` da `best`; scrivere file temporanei e rinominarli dopo il salvataggio completo.

### 6.5 CLI da rendere disponibili

Questi comandi sono il contratto delle **nuove** CLI, da implementare nei task indicati; non esistono ancora. `--resume` deve essere esplicito e verificare il manifest. Una directory non vuota senza `--resume` produce errore. `--max-epochs` è un override registrato nel manifest; `--seed` prevale sul valore della configurazione.

```bash
# Task 1: audit, senza training.
PYTHONPATH=src "$MORPH_SIM_PY" scripts/audit_world_model_data.py \
  --dataset-dir "$MORPH_SYNTH_DATA" --output "$MORPH_V3_RUNS/audit.json"

# Task 3: primo confronto M1 dopo dry run e pilot temporale.
PYTHONPATH=src "$MORPH_TRAIN_PY" scripts/train_world_model_v3.py \
  --dataset-dir "$MORPH_SYNTH_DATA" \
  --config configs/world_model_v3/actuator.json \
  --seed 17 --max-epochs 40 --output-dir "$MORPH_V3_RUNS/M1-seed17"

# Task 2/3: valutazione sulla validation del dataset fornito.
PYTHONPATH=src "$MORPH_TRAIN_PY" scripts/evaluate_world_model_v3.py \
  --dataset-dir "$MORPH_SYNTH_DATA" --partition validation \
  --checkpoint "$MORPH_V3_RUNS/M1-seed17/best.pt" \
  --output "$MORPH_V3_RUNS/M1-seed17/validation.json"

# Task 4: smoke del nuovo collector.
PYTHONPATH=src "$MORPH_SIM_PY" scripts/collect_counterfactual_dataset.py \
  --families 4 --seed 20261041 --split development \
  --output-dir "$MORPH_V3_RUNS/counterfactual-smoke"
```

Per allenare sui dati esistenti più quelli nuovi, `--dataset-dir` deve poter essere ripetuto; il loader unisce solo partizioni corrispondenti, verifica namespace/hash delle famiglie e rifiuta duplicati. La modalità test finale del valutatore richiede anche `--frozen-protocol` con il manifest del protocollo congelato. L'output pubblico deve essere un'esportazione aggregata del report locale, non una copia del suo manifest con i percorsi personali.

## 7. Protocollo sperimentale e costi

### 7.1 Confronti obbligatori

| ID | Modello | Scopo |
| --- | --- | --- |
| P0 | persistenza dello stato/contatti | baseline minima |
| P1 | velocità costante per posizione oggetto e velocità zero come baseline separata | controllare se il modello apprende più del moto elementare |
| M0 | MLP di riferimento con stato 37D e azioni 4D, riaddestrata con il protocollo V3 | confronto equo di dati e split |
| M1 | stesso modello/protocollo, azioni 8D | isolare l'effetto della rappresentazione delle azioni |
| M2 | M1 + feature relative + obiettivo per gruppi e campionamento eventi | verificare la previsione di oggetto/contatti |
| M3 | M2 + curriculum dell'orizzonte | verificare la riduzione della deriva |

M2 contiene tre interventi correlati: se migliora, fare almeno una rimozione mirata del campionamento eventi; evitare di attribuire tutto il guadagno a una singola modifica. Il trattamento dei quaternioni e il rollout dei singoli membri devono essere identici in M0/M1 per non confondere il confronto azioni. Nella prima serie M0–M3 i guadagni residui sono tutti 1: la calibrazione V2 resta nel confronto storico, non viene attivata silenziosamente. M0 è una baseline controllata V3, non una riproduzione numericamente identica della V2 storica.

### 7.2 Budget iniziale

- Prima un dry run di 2 epoche su 4 famiglie train e 2 validation, solo per verificare pipeline, memoria e salvataggi. Non usarne le metriche come risultato scientifico.
- Poi un pilot di 5 epoche per misurare tempo/epoca e stimare il costo di ogni configurazione. Il dispositivo di default è CPU; confrontare MPS solo se disponibile e se gli operatori sono supportati.
- Pilot comparativo: seed `17`, 3 membri, 128 unità, batch di 64 finestre, AdamW `lr=5e-4`, weight decay `1e-5`, gradient clipping norma `1.0`, massimo 40 epoche, patience 8. Applicare lo stesso budget alle configurazioni confrontate.
- Per il curriculum: massimo 15 epoche a 25 passi, 15 a 50, 10 a 100. Se il batch non entra in memoria, ridurre a 32 per tutte le configurazioni confrontate e registrare la modifica. Early stopping sul criterio validation indicato nel task 3.
- Confermare solo i due candidati migliori con seed `29` e `43`, riutilizzando il seed `17` già completato. Nessuna griglia automatica di decine di combinazioni.
- Prima espansione dati: 100 famiglie × 8 varianti = 800 episodi di sviluppo. Non avviare migliaia di episodi finché il collector non supera uno smoke test di 4 famiglie.
- Test finale: 100 nuove famiglie, seed separato e configurazione congelata; con 8 varianti sono 800 episodi offline. Eseguire il confronto online almeno sulle stesse 100 condizioni iniziali nominali, per ciascuno dei tre checkpoint seed.
- Registrare durata dei pilot ed ETA prima delle esecuzioni lunghe. Una sessione interrotta riparte dagli artefatti compatibili già completati.

Questi numeri sono budget iniziali e obiettivi di verifica, non garanzie sulle prestazioni. Se il budget locale è eccessivo, completare la consegna A e documentare il costo della B/C prima di proporre hardware aggiuntivo.

### 7.3 Metriche e soglie

Riportare separatamente: RMSE oggetto in metri, orientamento in radianti, velocità lineare e angolare, giunti, contatti pinza e supporto. Per orientamento usare errore angolare con `abs(dot(q_pred, q_true))`, non RMSE dei quattro coefficienti.

Per i contatti riportare precision, recall, F1 e prevalenza per tipo; anche F1 degli eventi di acquisizione/perdita del contatto con tolleranza di ±2 osservazioni (±40 ms), matching uno-a-uno entro episodio. Non fondere i due contatti in un'unica accuracy dominata dal supporto quasi sempre presente.

Per la selezione: ranking solo tra candidati dello stesso stato/scena iniziale; dare metà punto ai pareggi; escludere dal denominatore le coppie con identico esito e contarle. Riportare successo del candidato selezionato e regret rispetto al migliore disponibile, includendo le scene senza alcun candidato riuscito.

Convenzioni per i casi vuoti: se `2*TP+FP+FN=0`, F1 è `null` e si riporta supporto zero; se non ci sono coppie informative, ranking è `null`. I gruppi senza supporto non diventano risultati perfetti e non vengono nascosti nei conteggi.

Per incertezza: associazione con errore per famiglia e capacità di rilevare i rollout errati. L'accordo dei membri non prova correttezza. Non chiamare calibrata una correlazione positiva.

Soglie di promozione proposte, fissate prima del test finale:

- Dinamica: RMSE oggetto a 0,5 s e 1 s inferiore alla migliore baseline P0/P1; F1 degli eventi superiore alla persistenza. Valutare anche finestre in movimento e vicino ai contatti.
- Selezione: ranking entro scena almeno `0.70`, con limite inferiore dell'intervallo bootstrap al 95% sopra `0.50`; almeno 30 famiglie informative. Il valore è un obiettivo da verificare.
- Controllo: incremento di successo di almeno **10 punti percentuali**, con intervallo al 95% della differenza appaiata sopra zero; riportare anche il risultato per seed e il tasso di fallback. Sotto tale soglia, descrivere il risultato misurato senza dichiarare raggiunto l'obiettivo.
- Nessuna divergenza nascosta: riportare quota rollout invalidi e correzioni del quaternione, senza rimuoverli silenziosamente dalle medie.

Bootstrap: 2.000 ricampionamenti per famiglia, seed fisso `20261004`; tutte le varianti e i risultati dei seed di una famiglia rimangono insieme. I frame non sono osservazioni indipendenti. Per le differenze tra controller usare ricampionamento appaiato delle stesse famiglie.

## 8. Mappa del codice

| File esistente | Ruolo e cautela |
| --- | --- |
| `src/world_model/state_dynamics.py` | baseline V2; loss, guadagni residui, normalizzazione e checkpoint |
| `src/world_model/fidelity.py` | rollout V2, punteggio fisico, confronto esiti; riusare solo utilità realmente compatibili |
| `src/world_model/transition_timing.py` | controlli sugli intervalli temporali |
| `src/evaluation/contact_demo_evaluator.py` | osservazione coerente a 37 valori, azioni e archivi |
| `src/evaluation/synthetic_contact_dataset.py` | generazione delle 60 famiglie, raccolta e audit |
| `src/simulation/demonstration_manipulation.py` | controller a fasi; attenzione agli stati interni e ai comandi mantenuti |
| `src/simulation/contact_manipulation_env.py` | scena e criteri fisici |
| `scripts/train_synthetic_world_model.py` | loader e training sintetico storico |
| `scripts/evaluate_world_model_cv.py` | valutazione per video; preservare l'esperimento storico |

Nuovi file previsti, da creare al task indicato:

- `src/world_model/v3/{__init__,contracts,data,model,training,evaluation,planning}.py`
- `src/evaluation/counterfactual_dataset.py`
- `src/simulation/contact_control_adapter.py`, `src/simulation/contact_state.py`
- `scripts/audit_world_model_data.py`, `scripts/train_world_model_v3.py`, `scripts/evaluate_world_model_v3.py`
- `scripts/collect_counterfactual_dataset.py`, `scripts/run_world_model_mpc.py`
- `configs/world_model_v3/{cartesian,actuator,event,curriculum,planner}.json`
- `tests/test_world_model_v3_{data,model,training,evaluation,planning}.py`
- `tests/test_counterfactual_dataset.py`, `tests/test_contact_control_adapter.py`

Creare i moduli solo quando necessari. Evitare refactoring generali della pipeline video o una nuova infrastruttura distribuita.

## 9. Task 0 — Riprendere dal repository e identificare gli input

**File letti:** README, report baseline, i file della mappa e questo piano.
**Output:** inventario locale `MORPH_V3_RUNS/inventory.json` e `progress.md`.

- [ ] Controllare branch, diff, remoto e worktree con `git status -sb`, `git log -5 --oneline`, `git remote -v`, `git worktree list`. Se la base è cambiata, annotare le differenze prima di procedere.
- [ ] Creare il worktree di lavoro senza eliminare quelli esistenti. Usare il prefisso di branch richiesto dall'ambiente, se presente.
- [ ] Verificare i percorsi della sezione 5 e gli import nei rispettivi interpreti. Leggere i manifest; confermare conteggi, versione schema e partizioni del dataset corretto.
- [ ] Identificare nomi, ordine e `ctrlrange` degli attuatori dal modello compilato corrispondente. Non dedurli dalle sole posizioni nell'array.
- [ ] Registrare hash dei dati, baseline numeriche e presenza dei checkpoint. Nessun training in questo task.

**Completamento:** dati corretti individuati, ambienti funzionanti e prossimo task annotato. Se un file manca, cercarlo nella radice dati nota e segnalare il percorso mancante preciso.

## 10. Task 1 — Loader V3 e audit delle azioni

**Creare:** `v3/contracts.py`, `v3/data.py`, `scripts/audit_world_model_data.py`, `tests/test_world_model_v3_data.py`.
**Interfacce:** `load_partitions(dataset_dir: Path, *, action_mode: str) -> dict[str, EpisodeBatch]`; `audit_actions(partitions: dict[str, EpisodeBatch]) -> dict`.

Per la diagnostica storica dei video definire inoltre `load_human_replays(results_dir: Path, *, action_mode: str) -> EpisodeBatch`: legge i due archivi direct/confidence-aware, distingue gli episodi e assegna entrambi al gruppo del video sorgente. Nessuna partizione sintetica viene inventata per questi dati; `metadata` li marca come `human_replay` e `historical_diagnostic`.

- [ ] Scrivere test con episodi piccoli generati in memoria: selezione esatta degli 8 controlli salvati, nessuno shift di una riga, rifiuto controlli mancanti, train/validation disgiunti, gap temporali e quaternion non validi. Cambiare solo gli ID stringa non deve cambiare gli input numerici.
- [ ] Eseguire `PYTHONPATH=src "$MORPH_TRAIN_PY" -m unittest discover -s tests -p 'test_world_model_v3_data.py' -v`; verificare il fallimento per il contratto ancora assente.
- [ ] Implementare il loader senza importare MuJoCo nel runtime di training. Riutilizzare la validazione dei tempi; non passare azioni 8D al validatore V2 che richiede 4D.
- [ ] Implementare l'audit per famiglia, fase e tipo contatto: eventi, durata, movimento oggetto, comandi costanti per cinque osservazioni e intervalli fuori distribuzione. Verificare il mantenimento dei controlli usando i timestamp, non assumendo che ogni transizione cambi comando.
- [ ] Eseguire i test mirati e l'audit sull'archivio corretto. Salvare report locale e un aggregato senza percorsi/ID quando utile.
- [ ] Commit: `feat: audit dynamics action representations`.

**Completamento:** un archivio può essere letto in entrambe le modalità mantenendo stessi stati, partizioni e tempi. Un problema di allineamento va risolto prima del training.

## 11. Task 2 — Valutatore per dinamica, eventi e selezione

**Creare:** `v3/evaluation.py`, `scripts/evaluate_world_model_v3.py`, `tests/test_world_model_v3_evaluation.py`.
**Interfacce:** `evaluate_dynamics(model, episodes: EpisodeBatch, *, horizons: tuple[int, ...]) -> dict`; `evaluate_candidates(predictions: list[dict], outcomes: list[dict]) -> dict`; `bootstrap_families(rows: list[dict], *, seed: int = 20261004, samples: int = 2000) -> dict`.

- [x] Scrivere test in cui un modello perfetto ha errore zero, un modello costante fallisce quando l'oggetto si muove, `q` e `-q` hanno errore zero, un contatto sempre attivo non ottiene buon F1 sugli eventi, e due eventi predetti non possono abbinarsi alla stessa transizione vera.
- [x] Aggiungere casi selezione con pareggi, esiti tutti uguali, scene senza successi e candidati mancanti. Le coppie tra scene diverse non devono entrare nel ranking entro scena.
- [x] Implementare P0/P1: persistenza; `p(t+h)=p(t)+v(t)*h*dt` per posizione; velocità zero come confronto separato. Le baseline non leggono stati futuri per produrre predizioni.
- [x] Implementare endpoint a 5/25/50/100 passi, errore lungo la traiettoria e metriche per fase/evento. Riportare numero di finestre e famiglie per ogni metrica.
- [x] Per le finestre evento, calcolare anche le stesse metriche su un campionamento uniforme: l'oversampling serve al training, non a gonfiare la metrica pubblica. Le metriche dinamiche usano finestre uniformi non sovrapposte; gli eventi sono contati su tutta la timeline uniforme.
- [x] Implementare gli intervalli bootstrap per famiglia e verifica della mancata sovrapposizione con train/validation del checkpoint. La modalità finale richiede un manifest di test congelato.
- [x] Eseguire `PYTHONPATH=src "$MORPH_TRAIN_PY" -m unittest discover -s tests -p 'test_world_model_v3_evaluation.py' -v` e salvare i risultati delle baseline sui soli dati di sviluppo.
- [x] Commit: `feat: evaluate contact events and scenario decisions`.

**Evidenza:** 14 test V3 per il valutatore; suite completa 163 test OK, 13 skip per PyTorch opzionale nell'ambiente simulator. CLI `--help`, compilazione e `git diff --check` passano. Baseline salvate fuori Git in `MORPH-local-data/world-model-v3/baseline-development-20261004.json`: 204 episodi e 51 famiglie train+validation; il test è escluso. Sul relativo split, a 0,5 s, la baseline persistenza ha RMSE posizione oggetto di 0,01032 m; CV 0,01310 m; F1 eventi pinza/supporto per tutte le baseline semplici è 0,0. Questi numeri sono riferimento di sviluppo, non prova di miglioramento del modello.

**Completamento:** il valutatore distingue una buona metrica media da un modello che non predice i contatti decisivi. In questa fase un modello sintetico deterministico basta per validare le metriche.

## 12. Task 3 — Modello minimo e confronto controllato 4D/8D

**Creare:** `v3/model.py`, `v3/training.py`, `scripts/train_world_model_v3.py`, configurazioni cartesian/actuator e test modello/training.
**Interfaccia training:** `fit_dynamics(train: EpisodeBatch, validation: EpisodeBatch, *, config: dict, output_dir: Path) -> DynamicsV3`.

- [x] Scrivere test per dimensioni 4D/8D, dipendenza effettiva dalle azioni su un sistema lineare noto, conservazione del vassoio, quaternioni validi, riproducibilità con seed, rifiuto checkpoint incompatibili e normalizzazione indipendente dai dati validation/test.
- [x] Implementare tre MLP con tre strati nascosti da 128 e SiLU, head per 32 delta continui e due logits contatto. Bootstrap per famiglia, conservando le sequenze. Non aumentare capacità in questa ablation.
- [x] Per M0/M1 usare la stessa loss tipo V2 e rollout di 25 passi, con guadagni residui tutti 1 e calibrazione disattivata. Conservare il report V2 calibrato come riferimento storico distinto. Un'eventuale ablation della calibrazione è successiva, dichiarata e usa solo validation.
- [x] Validare ogni membro a intervalli fissi di un'epoca, sulla stessa lista deterministica di finestre validation. Criterio checkpoint: media delle loss validation per famiglia; per gli esperimenti successivi riportare sempre anche RMSE oggetto e F1 eventi per scegliere la configurazione finale.
- [x] Implementare manifest, resume verificato, `best`/`last`, log JSONL e stop per valori non finiti. Un run interrotto non deve apparire completato.
- [x] Eseguire test modello e training con `-p 'test_world_model_v3_model.py'` e `-p 'test_world_model_v3_training.py'` nell'interprete PyTorch. Usare dati sintetici piccoli nei test; i training reali sono esperimenti separati.
- [x] Eseguire dry run, pilot temporale e M0/M1 con lo stesso split 42/9, seed e budget. Non leggere le 9 famiglie test storiche per la selezione della configurazione.
- [x] Salvare tabella comparativa con durata, errori, F1 eventi, guadagni residui, finestre e famiglie. Se cambia una dipendenza, invalidare il checkpoint attraverso il manifest.
- [x] Commit: `feat: train actuator-conditioned dynamics`.

**Evidenza:** suite completa simulator 171 test OK, 21 skip nell'interprete senza PyTorch; i 30 test V3 per data/evaluation/model/training sono stati eseguiti nei runtime previsti. Dry run 4/2 completato; pilot 5 epoche su 42/9 richiede circa 58 s di training per modello. M0/M1 seed 17 sono stati allenati fino all'early stopping: M0 `(26,16,20)` epoche, M1 `(27,16,19)`, circa 239 s di training ciascuno, calibrazione esclusa e guadagni unitari. Report aggregato fuori Git: `MORPH-local-data/world-model-v3/M0-M1-development-comparison-20261004.json`.

| Orizzonte | M0 4D RMSE oggetto | M1 8D RMSE oggetto | Persistenza | M0/M1: differenza appaiata 8D−4D (IC 95%) |
| --- | ---: | ---: | ---: | ---: |
| 0,1 s | 1,56 mm | 1,45 mm | 3,07 mm | −0,104 mm [−0,225; +0,012] |
| 0,5 s | 5,35 mm | 5,50 mm | 10,05 mm | +0,154 mm [−0,020; +0,319] |
| 1,0 s | 8,00 mm | 8,26 mm | 16,66 mm | +0,215 mm [−0,232; +0,715] |
| 2,0 s | 13,59 mm | 14,40 mm | 27,77 mm | +0,663 mm [−0,420; +1,832] |

Entrambi i modelli migliorano la baseline di persistenza e la baseline a velocità costante; sugli eventi, M0 ottiene F1 0,378/0,424 (pinza/supporto) e M1 0,339/0,444. Tutte le finestre sono valide. Le 9 famiglie validation sono poche per una conclusione generale; nessuna metrica del test storico è stata usata per la selezione.

**Conferma multi-seed:** i seed `17,29,43` mantengono le stesse famiglie di split. Mediando seed e famiglie, RMSE posizione oggetto a 0,5/2,0 s: M0 `5,43/15,32 mm`, M1 `5,44/16,35 mm`, persistenza `9,98/27,57 mm`, velocità costante `14,62/58,39 mm`. IC bootstrap per famiglia al 95%: M0 `4,52–6,40 mm` e `14,02–16,52 mm`; M1 `4,53–6,39 mm` e `14,58–17,78 mm`. Differenza appaiata M1−M0: a 0,5 s `+0,009 mm` (IC `−0,093–+0,118`), a 2 s `+1,031 mm` (IC `+0,165–+1,947`). F1 medio eventi pinza/supporto: M0 `0,349/0,406`, M1 `0,347/0,424`. Report locale completo, con breakdown per seed e confronto bootstrap: `MORPH-local-data/world-model-v3/M0-M1-three-seed-validation-20261004.json`.

Per l'integrazione si mantiene `actuator8`, che rappresenta il comando realmente applicato. Il suo svantaggio misurato a 2 s limita l'orizzonte operativo: il planner iniziale non supera 1 s e Task 5 deve verificare il curriculum fino a 2 s prima di qualunque uso più lungo. L'effetto 4D/8D resta piccolo rispetto ai guadagni contro le baseline semplici.

**Decisione:** continuare con `actuator8` per le fasi successive. Se le prestazioni peggiorano nettamente, verificare prima unità, ordine, alignment, gripper e stato omesso; il semplice aumento di epoche non è la prima correzione.

## 13. Task 4 — Dati controfattuali dalla stessa scena

**Creare:** `src/evaluation/counterfactual_dataset.py`, `scripts/collect_counterfactual_dataset.py`, `tests/test_counterfactual_dataset.py`.
**Interfacce:** `make_counterfactual_families(*, families: int, seed: int, split: str) -> list[dict]`; `collect_counterfactual_dataset(output_dir: Path, scenarios: list[dict], *, model_path: Path) -> dict`.

Qui `split` accetta `development` (assegna 80% famiglie a train e 20% a validation) oppure `test` (tutte test). Per lo smoke di 4 famiglie usare 3 train/1 validation. Le etichette scritte negli archivi restano `train`, `validation`, `test`; gli ID di famiglia includono seed e indice per evitare collisioni tra raccolte.

- [x] Usare come punto di partenza il collector esistente. Fissare per famiglia pickup, tray, yaw, condizioni iniziali, path di base, parametri fisici e seed. Ogni variante cambia solo l'intervento dichiarato.
- [x] Definire 8 varianti: nominale; velocità ×1,5; velocità ×0,7; offset presa +15 mm sull'asse X; offset presa −15 mm; offset +30 mm; rilascio anticipato; rilascio ritardato. Nei due rilasci cambiare solo `release_open_fraction` (rispettivamente 0,2 e 0,9), mantenendo l'altezza nominale. Nessuna variante riceve in anticipo un'etichetta di successo.
- [x] Gli archivi esistenti hanno varianti più grossolane; mantenerli come sviluppo e aggiungere i nuovi esempi. Non spostare le vecchie famiglie test in train automaticamente.
- [x] Le 100 nuove famiglie di sviluppo si dividono 80 train/20 validation prima della simulazione; tutte le varianti restano insieme. Usare seed `20261040`, registrandolo. Lo smoke test usa un seed distinto, `20261041`, e non entra nel test finale.
- [x] Salvare il vero stato iniziale necessario al replay, oltre allo stato osservabile 37D. Per duplicare una scena usare una copia completa `MjData` o `mjSTATE_INTEGRATION`; conservare tempo, attivazioni, warmstart e controlli. Registrare anche fingerprint del modello compilato e RNG. Se si introduce un checkpoint a metà episodio, servono anche gli stati interni del controller; la prima raccolta parte dall'inizio e non richiede quel refactoring.
- [x] Registrare separatamente comandi realmente applicati e osservazioni. Verificare che riprodurre il nastro di controlli da quello stesso snapshot riproduca la traiettoria e l'esito entro tolleranza numerica dichiarata.
- [x] Scrivere test: varianti con identico snapshot iniziale, una variazione dichiarata per variante, split disgiunti, nessuna sovrascrittura, errore di raccolta presente nel manifest, ripetizione deterministica e nessuna modifica allo stato live durante una simulazione di controllo su una copia.
- [x] Eseguire `PYTHONPATH=src "$MORPH_SIM_PY" -m unittest discover -s tests -p 'test_counterfactual_dataset.py' -v`; poi smoke di 4 famiglie. Solo dopo audit positivo raccogliere le 100 famiglie.
- [x] Contare famiglie con esiti discordanti e distribuzione eventi. Se sono meno di 30, aumentare la granularità degli offset/rilasci sul solo sviluppo, registrando una nuova versione del dataset. Conservare anche scene tutte riuscite e tutte fallite.
- [ ] Commit: `feat: collect paired manipulation interventions`.

**Risultati sviluppo seed `20261040`:** 100 famiglie, 800 episodi, 772.095 transizioni; 640 train e 160 validation; zero errori di raccolta. Replay di tutti gli 800 nastri verificato, errore massimo assoluto osservato `0`. Esiti discordanti in 76 famiglie (24 tutte fallite, nessuna tutta riuscita). Successi per intervento: nominale 61/100; velocità ×1,5 45/100; velocità ×0,7 50/100; pickup +15 mm 8/100; pickup −15 mm 37/100; pickup +30 mm 0/100; rilascio anticipato 62/100; rilascio ritardato 62/100. Copertura nei dati train/validation: cambi di contatto pinza 2.205/576, cambi di supporto 1.284/306, transizioni con oggetto in movimento 180.025/48.386. Gli snapshot numerici occupano 32 KB per 4 famiglie smoke: il modello compilato viene ricostruito dai parametri salvati e verificato con fingerprint, evitando di duplicare circa 38 MB di mesh per famiglia.

**Completamento:** esistono vere alternative per lo stesso stato iniziale e un archivio riproducibile delle loro conseguenze. Non basta assegnare lo stesso `group_id` a scene iniziali diverse.

## 14. Task 5 — Training orientato a contatti e orizzonti utili

**Modificare:** moduli V3 data/model/training; aggiungere configurazioni event/curriculum e test mirati.

- [x] Implementare le 6 feature relative della sezione 6.1 e un test che le ricalcoli dai rollout predetti, senza usare coordinate future osservate.
- [x] Il sampler sceglie prima uniformemente una famiglia, poi un episodio. Il 50% delle finestre proviene dal campionamento uniforme, il 50% da finestre che includono un cambio di contatto; se non ci sono eventi, usare il sampler uniforme e registrarlo. La finestra evento inizia in modo che includa fino a 5 passi prima del cambio, senza attraversare i bordi dell'episodio.
- [x] Loss M2: media degli errori per gruppo normalizzati con statistiche train; pesi iniziali `package_position=3`, `package_orientation=1`, `package_linear_velocity=1`, `package_angular_velocity=1`, `ee_position=1`, `arm_position=1`, `arm_velocity=0.5`, `gripper_aperture=1`, `gripper_velocity=0.5`. Dividere la somma per la somma dei pesi. Orientamento: `1 - dot(q_pred,q_true)^2` dopo normalizzazione; mai penalizzare il solo cambio di segno del quaternione.
- [x] Aggiungere BCE dei due contatti con peso complessivo 1. Calcolare `pos_weight=negativi/positivi` sui soli dati train e limitarlo a `[0.5,10]`; se manca una classe usare peso 1 e riportare l'assenza. Conservare metriche separate per le due head.
- [x] Loss autoregressiva: media ai passi 1, 5 e H della finestra; bilanciamento iniziale 0,5 loss sulle transizioni vere e 0,5 loss autoregressiva. Nel termine sulle transizioni vere usare realmente `s_t` osservato; nella parte autoregressiva reinserire le predizioni. Salvare i due termini separatamente.
- [x] M2 mantiene H=25. M3 introduce il curriculum 25→50→100 della sezione 7.2. Non cambiare contemporaneamente capacità, dati e budget senza una baseline equivalente.
- [x] Scrivere test su frequenze del sampler, assenza di leakage, gradienti finiti a ogni orizzonte, equivalenza quaternion e durata fisica corretta. Eseguire i test V3 pertinenti.
- [ ] Valutare M2/M3 su validation uniforme e su finestre evento. Confrontare anche M2 senza oversampling eventi. Scegliere sulla base congiunta di oggetto/contatti, non solo loss totale.
- [ ] Confermare i due candidati migliori con i tre seed previsti. Training sul dataset espanso va confrontato con M1 riaddestrato sugli stessi dati; tenere distinto il guadagno dovuto al dataset.
- [ ] Commit: `feat: train contact-aware multistep dynamics`.

**Preliminary pilots, seed 17 (five epochs/member except the three-stage curriculum smoke):** M1 retrained and both M2 variants use the same combined train/validation family split. M1 uniform object RMSE is 4.90 mm at 0.5 s and 12.80 mm at 2 s; M2 uniform sampler is 4.60/12.58 mm; M2 event sampler is 6.10/18.39 mm. On event-centered windows at 0.5 s, M2 with oversampling reaches 9.78 mm and contact endpoint F1 0.958, versus 15.75 mm and 0.885 without oversampling. These are pilots, not selected final checkpoints. M3 one-epoch-per-stage smoke ran at 25/50/100 steps in 59/112/220 s; object RMSE was 14.07 mm at 2 s after one epoch at H=100, insufficient to judge convergence. Full runs remain pending.

**Decisione:** se le soglie dinamica falliscono, procedere alla diagnosi della sezione 19 prima del planner. Nessun obbligo di raggiungere H=100 se a H=50 il modello diverge: riportare il problema e correggerlo.

## 15. Task 6 — Incertezza e ranking entro scena

**Modificare:** model/evaluation V3, CLI valutazione e test relativi.

- [ ] Testare che membri con dinamiche diverse conservino traiettorie diverse per tutto il rollout; il rollout non deve collassare sulla media a ogni passo. Aggiungere casi di un membro divergente e candidati tutti invalidi.
- [ ] Calcolare il costo per membro e la distribuzione dei costi tra membri. Usare spread nello spazio fisico dell'oggetto e distribuzione dei contatti; non mescolare arbitrariamente metri, radianti e delta normalizzati in un unico numero senza documentarne i pesi.
- [ ] Conservare il punteggio fisico V2 come baseline del ranking. Valutare le sequenze complete di comandi delle varianti, inizializzate dallo stesso stato, con esiti MuJoCo allineati per famiglia/variante.
- [ ] Non confondere questa valutazione offline con il controllo online: le sequenze complete sono già disponibili nell'archivio; il planner online non può recuperare il futuro del controller dalle traiettorie test.
- [ ] Se il ranking lungo rimane debole ma la dinamica breve migliora, riportare entrambi i risultati. Valutare anche la scelta su finestre brevi vicino alla presa/rilascio rispetto a progressi fisici espliciti, senza sostituire quelle etichette al successo dell'intero task.
- [ ] Calibrare su validation solo le eventuali soglie di incertezza. Per una soglia iniziale usare il 95° percentile dello spread nelle finestre validation con errore posizione oggetto ≤2 cm a 0,5 s; se ci sono meno di 30 finestre utili, dichiarare la calibrazione insufficiente. La soglia è operativa, non una garanzia di probabilità d'errore.
- [ ] Produrre aggregati con ranking, pareggi, copertura, regret, intervalli e confronto con punteggio V2/random/sempre nominale. Scegliere i pesi usando solo validation.
- [ ] Commit: `feat: rank manipulation alternatives with ensemble rollouts`.

**Completamento B:** il risultato entro scena è misurato e riproducibile. Se il ranking lungo fallisce, il planner breve può essere studiato solo se ha superato le soglie di dinamica e scelta locale; non dichiarare risolta la selezione globale.

## 16. Task 7 — Adapter del controllo senza cambiare la baseline

**Creare:** `src/simulation/contact_control_adapter.py`, `tests/test_contact_control_adapter.py`.
**Modificare in modo mirato:** `src/simulation/demonstration_manipulation.py` e il recorder, preservando i default.
**Interfaccia:** hook opzionale `control_override(state: np.ndarray, nominal_control: np.ndarray, context: dict) -> np.ndarray`, invocato dopo il calcolo IK del comando nominale e prima dei passi di simulazione.

- [ ] Spostare il calcolo dell'osservazione coerente in `src/simulation/contact_state.py` e reimportare `contact_task_state` da lì in `contact_demo_evaluator.py`, preservandone nome/API. L'adapter usa quel modulo: evitare un import circolare tra simulazione ed evaluator. Riutilizzare il buffer di osservazione e non eseguire `mj_forward` sullo stato live per leggere la fotografia corrente.
- [ ] `context` contiene solo informazione disponibile ora: fase corrente, ultimo comando applicato, tempi, geometria nota del task e obiettivi correnti. Non passare `env`, stato futuro, etichette finali o l'intero episodio registrato al planner.
- [ ] Se l'hook è assente o restituisce esattamente il comando nominale, sequenze di comandi, tempi, stati e risultato devono coincidere con la versione precedente. In particolare mantenere 50 physics steps per controllo e 10 per osservazione.
- [ ] Validare forma `(8,)`, finitezza e limiti degli attuatori prima dell'applicazione. Registrare comando nominale, proposto ed effettivo, motivi di fallback e fase corrente.
- [ ] Applicare la proposta per un solo intervallo di 100 ms, poi ricalcolare il nominale dallo stato osservato. Le cinque osservazioni intermedie devono registrare proprio quel comando mantenuto.
- [ ] Testare hook identità, comando fuori limite, NaN, hold-time e settling. Una proposta non valida produce fallback documentato sul nominale, non un'etichetta di successo né il silenzioso scarto dell'episodio.
- [ ] Eseguire test adapter e regressioni `test_demonstration_manipulation.py`, `test_contact_demo_evaluator.py` nell'ambiente MuJoCo.
- [ ] Commit: `feat: expose bounded contact-control corrections`.

**Completamento:** è possibile eseguire una correzione senza alterare il comportamento originale quando il modello è disattivato.

## 17. Task 8 — MPC breve guidato dal modello

**Creare:** `v3/planning.py`, `scripts/run_world_model_mpc.py`, configurazione planner e `tests/test_world_model_v3_planning.py`.
**Interfaccia:** `choose_control(model: DynamicsV3, state: np.ndarray, nominal_control: np.ndarray, context: dict, *, config: dict, rng: np.random.Generator) -> PlannerDecision`.

- [ ] Creare un ambiente locale dedicato con PyTorch e MuJoCo. Verificare import e compatibilità prima di avviare prove lunghe. Conservare i due ambienti offline esistenti.
- [ ] Prima implementazione: random shooting batched, 64 candidati, orizzonte 1 s (10 comandi da 100 ms, ognuno ripetuto per 5 passi della dinamica). Se il modello è affidabile solo a 0,5 s usare 5 comandi e dichiararlo; non superare l'orizzonte validato.
- [ ] Includere sempre un candidato a correzione zero. La proposta nominale iniziale è il comando corrente mantenuto nell'orizzonte, non una futura traiettoria ricostruita con osservazioni test. Generare residui articolari con deviazione standard 0,01 rad, limitati a ±0,03 rad, su tre nodi temporali interpolati linearmente. Imporre i limiti effettivi `ctrlrange`; i limiti di variazione si applicano ai residui, senza falsificare il candidato nominale.
- [ ] In fase approach mantenere la pinza nominale. In grasp/release aggiungere candidati con cambio open/close immediato, dopo 0,2 s o dopo 0,4 s, usando i veri valori di comando del modello. In carry mantenere la pinza chiusa. Queste restrizioni definiscono un controller ibrido con fasi fornite dal nominale.
- [ ] Usare un costo di avanzamento per fase, normalizzato in metri: approach/grasp distanza pinza-oggetto; carry distanza oggetto-vassoio e deficit di sollevamento; release distanza dal centro utile del vassoio, velocità residua e supporto. La quota di sollevamento di riferimento resta 0,05 m. Il costo deve premiare progresso entro l'orizzonte, non richiedere il successo completo già nel prossimo secondo.
- [ ] Specificare i pesi in `planner.json` prima dei confronti: distance weight 1; lift deficit weight 2 durante carry; speed penalty 0,1 s sulla velocità lineare durante release; perdita di grasp durante carry penalty 0,05 m; terminal support mancante durante release penalty 0,02 m. I costi di posizione/velocità sono mediati sui passi; quelli terminali si applicano all'ultimo passo. Per grasp aggiungere penalty terminale `0,02 m * (1 - p_gripper_contact)`.
- [ ] Costo candidato: media dei costi dei membri + `beta * std(costi)`, beta iniziale 1. La penalità di deviazione dal nominale è `0,005 m * mean((residui_articolari / 0,03 rad)^2)`. Per beta confrontare solo `{0,1,2}` su validation se necessario. Documentare esplicitamente che i costi sono euristiche di controllo, non probabilità.
- [ ] Scartare rollout non finiti e rifiutare proposte oltre la soglia di incertezza validation. Se nessun candidato è utilizzabile, restituire il nominale e contare il fallback. Una traiettoria impossibile non diventa accettabile facendo clip delle posizioni predette.
- [ ] Applicare solo il primo comando della sequenza selezionata. Osservare lo stato nuovo e ripianificare. Non eseguire `mj_step` per valutare candidati dentro `choose_control`.
- [ ] Testare orizzonte/hold-time, candidato nominale, limiti, azione applicata uguale a quella valutata, fallback e miglior scelta su una dinamica giocattolo nota. Una fixture che lancia errore se viene invocata la fisica dei candidati deve continuare a passare.
- [ ] Misurare latenza mediana e p95, candidati/secondo e fallback. La simulazione può attendere il calcolo; non chiamare il controller real-time se il budget di 100 ms viene superato.
- [ ] Provare su sviluppo: nominale, stessa ricerca con scelta casuale, MPC appreso e ablation beta=0. Stessi stati iniziali e criterio di successo. Conservare tutti gli episodi, inclusi timeout e fallimenti.
- [ ] Solo se l'MPC mostra beneficio, valutare CEM con 2 iterazioni e 8 élite come ottimizzazione successiva; non introdurlo prima per moltiplicare il costo.
- [ ] Commit: `feat: control manipulation with short-horizon dynamics`.

**Completamento C preliminare:** confronto di controllo su sviluppo con successo, errore, latenza e fallback. Un'alta percentuale di fallback richiede distinguere il contributo del nominale da quello del modello.

## 18. Task 9 — Conferma finale, report e showcase

**Creare:** `results/metrics/world_model_v3.json`, `results/figures/world_model_v3.png` e documentazione dei comandi V3.
**Modificare:** README solo dopo aver misurato il risultato finale.

- [ ] Congelare configurazione modello, checkpoint dei tre seed, planner, costo, geometria, orizzonte e criteri. Salvare hash del protocollo finale.
- [ ] Generare 100 famiglie finali indipendenti con seed `20261042`, nella stessa distribuzione dichiarata, senza selezionarle in base agli esiti. Non usarle per training, soglie o calibrazione. Un eventuale test OOD deve avere un report separato.
- [ ] Eseguire valutazione dinamica/selezione e controllo appaiato con nominale e MPC. Riportare anche i risultati negativi e gli intervalli della sezione 7.3. Se ci sono meno di 30 famiglie informative, dichiarare insufficiente l'evidenza del ranking senza sostituire a posteriori il test.
- [ ] Valutare inoltre i checkpoint sintetici congelati sui replay dei 16 video, usando i comandi salvati e senza riaddestramento su quei video. Pubblicare questo trasferimento come diagnostica separata: è un dataset storico già osservato, non il nuovo test cieco. Non sommarlo alle famiglie sintetiche né chiamarlo validazione sul robot reale.
- [ ] Verificare che l'MPC appreso non abbia usato simulazioni future, condizioni reali dei candidati o lookup di esiti. Un eventuale oracle MuJoCo ha una colonna separata e non è il risultato del modello.
- [ ] Eseguire i test nuovi pertinenti e una suite di regressione completa al termine. Non rilanciare suite complete dopo ogni correzione del testo. Dichiarare separatamente test passati e test saltati per dipendenze mancanti.
- [ ] Mantenere il confronto video umano/Panda già pubblicato. Aggiungere, se i risultati lo giustificano, un confronto nominale/MPC dalla stessa scena con criterio di selezione dichiarato, per esempio prima famiglia del test in ordine prefissato. Un esempio di recupero può essere aggiunto ma va etichettato come tale.
- [ ] Se si visualizzano stati predetti dal world model, etichettarli come predizioni; il replay fisico MuJoCo e il video umano restano distinti. Non presentare un rendering del simulatore come generazione video del modello.
- [ ] README in stile research: problema, dati, architettura, protocollo, baselines/ablation, intervalli, casi falliti, latenza, riproducibilità, limiti. Il titolo della conclusione deve riflettere i risultati raggiunti.
- [ ] Controllare `git diff --check`, file staged, dimensioni e assenza di originali/checkpoint/percorsi locali/identificativi privati. Commit per milestone e push sotto l'identità già configurata. Integrare secondo lo stato effettivo del repository senza forzare la storia.
- [ ] Registrare commit pubblicato e corrispondenza col remoto. Aggiornare stato del piano con link ai nuovi report, lasciando tracciabili le baseline storiche.

**Completamento:** codice e report sono riproducibili, i risultati finali sono misurati e la documentazione distingue obiettivi raggiunti, parziali e falliti.

## 19. Diagnosi se le metriche non migliorano

Seguire quest'ordine, usando solo train/validation e un nuovo run per ogni ipotesi:

1. **Allineamento e unità.** Controllare se l'azione archiviata produce davvero quella transizione, inclusi gripper, IK e controllo mantenuto. Verificare i comandi degli attuatori e l'intervallo; non correggere un problema di schema con più training.
2. **Baseline statiche.** Separare finestre con oggetto fermo, in movimento e con cambi di contatto. Se la vittoria dipende solo dalle finestre statiche, il limite principale resta aperto.
3. **Stato parzialmente osservabile.** I 37 valori non sono l'intero `MjData`; apertura totale della pinza comprime due dita. Se stati osservati simili e azioni uguali producono esiti diversi, confrontare un'aggiunta esplicita di stato rilevante, oppure 5 osservazioni passate (100 ms) e azioni passate. Usare solo storia disponibile anche online; cambiare la versione schema e i test.
4. **Eventi non coperti.** Aggiungere train/validation attorno a piccoli offset o tempi di presa/rilascio che cambiano l'esito. Conservare i fallimenti. Il test finale rimane escluso.
5. **Incoerenza fisica.** Misurare posizione/velocità, giunti/end-effector e orientamento. Se serve, introdurre un integratore esplicito o derivare l'end-effector dalla cinematica dei giunti; confrontare con una baseline equivalente e dichiarare la parte analitica.
6. **Modi di contatto.** Solo dopo i passi precedenti, provare una head o mixture per appoggio/presa/moto libero. I modi al futuro devono essere predetti; niente etichette vere future negli input.
7. **Varietà fisica.** Massa/attrito/dimensioni variabili sono una seconda estensione. Includere i parametri noti nel contesto, oppure usare una storia per inferirli e dichiarare la parziale osservabilità. Randomizzarli lasciandoli invisibili a una rete senza memoria aggiunge ambiguità.
8. **Metodo alternativo.** PETS probabilistico completo o TD-MPC2 sono confronti successivi, se dati e protocollo sono già solidi. Portare un nuovo metodo non garantisce che il modello risolva il task con gli stessi dati.

Se due esperimenti consecutivi sulla stessa ipotesi non aiutano, chiudere quella diramazione con un risultato negativo e passare alla diagnosi successiva. Non ripetere indefinitamente training quasi identici. Dopo aver usato un test finale per scegliere modifiche, rinominarlo come sviluppo e riservare un nuovo test per qualsiasi nuova conferma.

## 20. Checklist di revisione trasversale

I cinque rischi prioritari e i task che devono provarne il comportamento:

1. **Azioni 4D/8D o intervalli confusi:** rifiuto checkpoint/schema errati e prova hold-time — task 1, 3, 7.
2. **Leakage tra varianti, snapshot o test già osservati:** split per famiglia, input senza futuro, test finale nuovo — task 1, 2, 4, 9.
3. **Modello statico premiato dalle medie:** baseline zero/persistenza, eventi e finestre in movimento — task 2, 5.
4. **Rollout implausibile selezionato con alta fiducia:** membri indipendenti, invalidi contati, fallback e limiti — task 6, 8.
5. **Planner aiutato dal simulatore o baseline alterata:** hook identità, stesso stato iniziale, stessa informazione, nessun futuro nella ricerca — task 4, 7, 8, 9.

## 21. Diario e passaggio tra sessioni

Mantenere fuori da Git, in `MORPH_V3_RUNS/progress.md`:

```text
Commit/branch/worktree:
Task completato e relativa evidenza:
Task in corso e prossimo comando:
Interprete e dipendenze:
Dataset/versione/hash/split:
Run completati e posizione di metrics/checkpoint:
Run attivo: PID, comando, log e ultimo avanzamento:
Decisione sperimentale presa e criterio:
Problemi aperti:
Budget temporale misurato e prossimo run previsto:
```

All'avvio leggere questo piano e quel diario, controllare processi/log prima di avviare un duplicato, poi riprendere il primo task incompleto. Non usare la sola esistenza di un checkpoint come prova del completamento. Prima di terminare una sessione indicare se esiste un processo davvero attivo; non promettere un avviso automatico se non è stata configurata una funzione che lo fornisce.

## 22. Cosa serve all'utente

- **Per iniziare A/B:** nessun nuovo video, nessun robot fisico e nessun abbonamento cloud. Dati corretti e runtime locali sono già presenti; chi esegue il piano controlla i percorsi e gestisce il codice.
- **Per il training locale:** lasciare il computer acceso e preferibilmente collegato all'alimentazione durante i run effettivamente avviati. Misurare prima la durata del pilot.
- **Se si valuta una GPU esterna:** presentare prima tempo stimato, dati da trasferire e costo/limiti. L'accesso all'account e l'autorizzazione alla spesa restano decisioni dell'utente.
- **Per una futura validazione reale:** serviranno robot/sensori o nuove riprese calibrate con oggetto, altezza e riferimento spaziale osservabili. Non è un prerequisito della V3 in simulazione.
- **Per il showcase finale:** mantenere il video umano già scelto; ulteriori riprese sono utili solo quando è definito quale informazione manca.

## 23. Riferimenti tecnici

- [PETS: Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114): ensemble, incertezza e propagazione delle traiettorie; riferimento per le estensioni, non una descrizione dell'implementazione corrente.
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828): modello e pianificazione orientati al controllo; possibile confronto successivo.
- [MuJoCo: simulation and state](https://mujoco.readthedocs.io/en/stable/programming/simulation.html): stato completo, copie e riproducibilità dei replay. Conservare lo stato del solver quando richiesto dalla riproducibilità; la sola osservazione 37D non basta a ricostruire il simulatore.
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339): riferimento già discusso per esecuzione asincrona di policy; non risolve da solo gli errori della dinamica oggetto/contatti di questo progetto.
