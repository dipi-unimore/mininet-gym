# ATTACKS_HO - Piano tuning rapido

Data: 2026-04-16

## Stato dal brief di oggi

### Risolti

- Supporto risultati test in TEST e TEST_<algo>
- Fix estrazione score test (incluso formato metrics.score)
- Gestione incompleti con reason nel pannello
- Zoom immagini con navigazione prev/next + caption
- Refactor backend in service + route modulari
- Export PDF migliorato e stabile
- Fix semantica round-step vs microstep in attacks_ho
- Fix fairness su incoming attack gia mitigato
- KPI mitigazione (artifact, grafico, pannello web, PDF)
- Semaforo mitigazione allineato web + PDF

### Aperti

- Tuning convergenza/apprendimento (non bloccante)
- Eventuale affinamento soglie semaforo (80/50 attuali)

## Mini tabella esperimenti prioritari

| ID | Priorita | Obiettivo | Modifiche proposte (config/default.yaml) | KPI da confrontare | Criterio di stop | Esito |
|---|---|---|---|---|---|---|
| E1 | Alta | Aumentare segnale di attacco utile | attacks.likely: 0.35 -> 0.45; attacks.no_attack_timeout: 5 -> 3 | test mean score %, mitigation ratio %, false positive trend | 1 run completo, poi confronto con baseline | TODO |
| E2 | Alta | Ridurre unblock prematuro | attacks.unblock_min_hold_rounds: 2 -> 3; attacks.unblock_required_normal_streak: 2 -> 3 | mitigation ratio %, under_attack total, score stability | 1 run completo; tenere se ratio sale senza crollo score | TODO |
| E3 | Media | Migliorare stabilita apprendimento tabellari | Q-learning_quick.exploration_decay: 0.995 -> 0.997; Sarsa_slow.exploration_decay: 0.9995 -> 0.999 | min/mean/max score %, accuracy trend, varianza tra agenti | 1 run completo; tenere se migliora media e riduce oscillazioni | TODO |

## Varianti pronte da lanciare

- Baseline: config/experiments/attacks_ho_baseline_20260416.yaml
- E1: config/experiments/attacks_ho_e1_20260416.yaml
- E2: config/experiments/attacks_ho_e2_20260416.yaml
- E3: config/experiments/attacks_ho_e3_20260416.yaml

## Comandi rapidi di lancio

1. Backup del config attuale:
	cp config/default.yaml config/experiments/config_backup_before_tuning_20260416.yaml
2. Seleziona variante (esempio E1):
	cp config/experiments/attacks_ho_e1_20260416.yaml config/default.yaml
3. Avvia run:
	source .venv/bin/activate && python main.py
4. Ripristino baseline dopo test:
	cp config/experiments/attacks_ho_baseline_20260416.yaml config/default.yaml

## Procedura sintetica per ogni run

1. Duplica config di baseline e applica solo le modifiche dell'esperimento.
2. Esegui training + test completi.
3. Registra i KPI dal pannello risultati (Training Summary, Test Summary, mitigation chart).
4. Aggiorna la colonna Esito con PASS/FAIL + nota breve.

## Template risultato rapido

- Baseline usata: ...
- Esperimento: E1/E2/E3
- Mean score %: ...
- Mitigation ratio %: ...
- Note: ...
- Decisione: KEEP/REVERT
