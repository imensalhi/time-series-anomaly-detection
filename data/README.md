# Dossier `data/`

Ce dossier contient l'ensemble des données utilisées dans le projet de **détection d'anomalies dans les séries temporelles**.

## Structure du dossier

```
data/
├── raw/            # Données brutes, telles qu'elles ont été collectées (ne pas modifier)
├── processed/      # Données prétraitées et nettoyées, prêtes pour l'entraînement
├── train/          # Jeux de données d'entraînement
├── test/           # Jeux de données de test / évaluation
└── README.md       # Ce fichier
```

## Description des sous-dossiers

### `raw/`
Contient les données **brutes** issues des sources originales (capteurs, APIs, fichiers CSV, etc.).  
Ces fichiers ne doivent **pas** être modifiés manuellement afin de garantir la reproductibilité.

### `processed/`
Contient les données après les étapes de **prétraitement** :
- Normalisation / standardisation des valeurs
- Gestion des valeurs manquantes
- Ré-échantillonnage des séries temporelles
- Encodage des labels d'anomalies

### `train/`
Contient les jeux de données utilisés pour **entraîner** les modèles de détection d'anomalies.

### `test/`
Contient les jeux de données utilisés pour **évaluer** et **valider** les performances des modèles.

## Format des données

Les fichiers de données sont généralement au format **CSV** avec la structure suivante :

| Colonne      | Type      | Description                                      |
|--------------|-----------|--------------------------------------------------|
| `timestamp`  | datetime  | Horodatage de la mesure                          |
| `value`      | float     | Valeur de la série temporelle                    |
| `label`      | int (0/1) | Indicateur d'anomalie : `0` = normal, `1` = anomalie |

## Sources des données

Les jeux de données utilisés dans ce projet peuvent provenir de :
- [Yahoo Anomaly Detection Dataset (S5)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)
- [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB)
- [KPI Anomaly Detection Dataset](https://github.com/NetManAIOps/KPI-Anomaly-Detection)

## Notes

- Ne pas committer de fichiers de données volumineux directement dans le dépôt Git.  
  Utiliser **Git LFS** ou un stockage externe (S3, GDrive, etc.) pour les fichiers dépassant quelques Mo.
- Un fichier `.gitkeep` est présent dans chaque sous-dossier vide pour préserver la structure du dépôt.
