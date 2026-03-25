# Simulatore Mix Energetico

Applicazione web realizzata in **Streamlit** per simulare diversi scenari di mix energetico e valutarne l'impatto in termini di produzione, costi ed emissioni.

## Utilizza il tool

Accedi direttamente al simulatore qui:

**https://simulatoremix-dgbxrbhlgppuoesfxewcpw.streamlit.app/**

## Cosa fa

Il tool permette di:

- simulare diverse combinazioni di fonti energetiche;
- analizzare il contributo di fotovoltaico, eolico, batterie, nucleare e gas;
- valutare scenari di decarbonizzazione;
- confrontare costi ed emissioni;
- esplorare un compromesso tra sostenibilità ed equilibrio economico.

## Tecnologie utilizzate

- Streamlit
- Pandas
- NumPy
- Plotly
- Numba
- OpenPyXL

## File principali del progetto

- `app.py` → applicazione principale
- `requirements.txt` → dipendenze Python
- `gme.xlsx` → dati di input
- `dataset_fotovoltaico_produzione.csv` → profili fotovoltaici
- `dataset_eolico_produzione.csv` → profili eolici


## Avvio in locale

1. Crea e attiva un ambiente virtuale Python.
2. Installa le dipendenze:

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Avvia l'app:

```bash
streamlit run app.py
```

## Note

Questa versione è pensata come strumento semplice di simulazione ed esplorazione scenari.
