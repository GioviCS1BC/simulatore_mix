import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import njit

# ==========================================
# CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(page_title="Simulatore Mix Energetico PRO - 30 Anni", layout="wide")

# ==========================================
# PESI GEOGRAFICI CURVE MEDIE
# ==========================================
PV_WEIGHTS_NORD = {
    'Lombardia orientale, area Brescia_NORD': 0.2956,
    'Veneto centrale, area Padova_NORD': 0.2313,
    'Emilia-Romagna orientale, area Ferrara,pianura_NORD': 0.2213,
    'Piemonte meridionale, area Cuneo_NORD': 0.1874,
    'Friuli-Venezia Giulia, area Udine_NORD': 0.0644,
}
PV_WEIGHTS_SUD = {
    'Puglia, area Lecce_SUD': 0.3241,
    'Sicilia interna, area Caltanissetta,Enna_SUD': 0.2117,
    'Lazio meridionale, area Latina_SUD': 0.1982,
    'Sardegna, area Oristano,Campidano_SUD': 0.1330,
    'Campania interna, area Benevento_SUD': 0.1330,
}
WIND_WEIGHTS_NORD = {
    'Crinale savonese entroterra ligure_NORD': 0.6020,
    'Appennino emiliano, area Monte Cimone_NORD': 0.2239,
    'Piemonte sud-occidentale , Cuneese_NORD': 0.0945,
    'Veneto orientale , Delta del Po_NORD': 0.0647,
    'Valle d’Aosta , area alpina_NORD': 0.0149,
}
WIND_WEIGHTS_SUD = {
    'Puglia, area Foggia,Daunia_SUD': 0.3093,
    'Sicilia occidentale, area Trapani_SUD': 0.2267,
    'Campania, area Benevento,Avellino_SUD': 0.1950,
    'Basilicata, area Melfi,Potenza_SUD': 0.1489,
    'Calabria, area Crotone,Catanzaro_SUD': 0.1201,
}

DEFAULT_PV_NORD_SHARE = 0.4800
DEFAULT_WIND_NORD_SHARE = 0.0163

# ==========================================
# FUNZIONI DI SUPPORTO DATI
# ==========================================
def _serie_pesata(df, pesi_colonne, scala=1.0, clip_upper=1.0):
    colonne_mancanti = [col for col in pesi_colonne if col not in df.columns]
    if colonne_mancanti:
        raise KeyError("Nel dataset mancano le colonne: " + ", ".join(colonne_mancanti))
    serie = sum(pd.to_numeric(df[col], errors='coerce').fillna(0.0) * peso for col, peso in pesi_colonne.items())
    serie = (serie / scala).clip(lower=0.0)
    if clip_upper is not None:
        serie = serie.clip(upper=clip_upper)
    return serie.astype(float)

def _mappa_profilo_annuale_su_indice(profilo_orario, indice_target):
    profilo = profilo_orario.copy()
    profilo.index = pd.to_datetime(profilo.index)
    chiavi_sorgente = list(zip(profilo.index.month, profilo.index.day, profilo.index.hour))
    mappa = {chiave: valore for chiave, valore in zip(chiavi_sorgente, profilo.values)}
    valori = []
    for ts in indice_target:
        chiave = (ts.month, ts.day, ts.hour)
        if chiave in mappa:
            valori.append(mappa[chiave])
        elif ts.month == 2 and ts.day == 29:
            valori.append(mappa.get((2, 28, ts.hour), mappa.get((3, 1, ts.hour), 0.0)))
        else:
            valori.append(0.0)
    return pd.Series(valori, index=indice_target, dtype=float)

@st.cache_data
def leggi_gme(file_gme):
    df_gme = pd.read_excel(file_gme, engine='openpyxl')
    colonna_volumi = df_gme.columns[2]
    if df_gme[colonna_volumi].dtype == 'object':
        df_gme[colonna_volumi] = df_gme[colonna_volumi].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df_gme[colonna_volumi] = pd.to_numeric(df_gme[colonna_volumi], errors='coerce')
    df_gme['Ora'] = pd.to_numeric(df_gme['Ora'], errors='coerce')
    data_convertita = pd.to_datetime(df_gme['Data'], dayfirst=True, errors='coerce')
    ore_aggiuntive = pd.to_timedelta(df_gme['Ora'] - 1, unit='h')
    df_gme['Datetime'] = data_convertita + ore_aggiuntive
    df_gme = df_gme.dropna(subset=['Datetime', colonna_volumi]).copy()
    df_gme.set_index('Datetime', inplace=True)
    df_gme.rename(columns={colonna_volumi: 'Fabbisogno_MW'}, inplace=True)
    return df_gme[['Fabbisogno_MW']]

@st.cache_data
def carica_profili_rinnovabili(file_fotovoltaico, file_eolico):
    df_pv = pd.read_csv(file_fotovoltaico)
    df_pv['time'] = pd.to_datetime(df_pv['time'], errors='coerce')
    df_pv.set_index('time', inplace=True)
    df_wind = pd.read_csv(file_eolico)
    df_wind['time'] = pd.to_datetime(df_wind['time'], errors='coerce')
    df_wind.set_index('time', inplace=True)
    profili = {
        'pv_nord': pd.Series(_serie_pesata(df_pv, PV_WEIGHTS_NORD, scala=1000.0).values, index=df_pv.index),
        'pv_sud': pd.Series(_serie_pesata(df_pv, PV_WEIGHTS_SUD, scala=1000.0).values, index=df_pv.index),
        'wind_nord': pd.Series(_serie_pesata(df_wind, WIND_WEIGHTS_NORD, scala=1.0).values, index=df_wind.index),
        'wind_sud': pd.Series(_serie_pesata(df_wind, WIND_WEIGHTS_SUD, scala=1.0).values, index=df_wind.index),
    }
    return profili

@st.cache_data
def carica_dati(file_fotovoltaico, file_gme, file_eolico, quota_pv_nord, quota_eolico_nord):
    df_gme = leggi_gme(file_gme)
    profili = carica_profili_rinnovabili(file_fotovoltaico, file_eolico)
    profilo_pv = (profili['pv_nord'] * quota_pv_nord) + (profili['pv_sud'] * (1.0 - quota_pv_nord))
    profilo_wind = (profili['wind_nord'] * quota_eolico_nord) + (profili['wind_sud'] * (1.0 - quota_eolico_nord))
    df_completo = df_gme.copy()
    df_completo['Fattore_Capacita_PV'] = _mappa_profilo_annuale_su_indice(profilo_pv, df_completo.index)
    df_completo['Fattore_Capacita_Wind'] = _mappa_profilo_annuale_su_indice(profilo_wind, df_completo.index)
    return df_completo.ffill()

# ==========================================
# 2. SIMULAZIONE FISICA (Numba) - OTTIMIZZATA PER 30 ANNI
# ==========================================
@njit
def simula_rete_light_fast(produzione_pv, produzione_wind, fabbisogno,
                           pv_mw, wind_mw, nucleare_mw, bess_mwh, bess_mw, gas_mw,
                           hydro_fluente_mw, hydro_bacino_mw, hydro_bacino_max_mwh, hydro_inflow_mw,
                           efficienza_bess=0.9):
    # Simulazione oraria per un singolo anno (uguale a prima)
    ore = len(fabbisogno)
    soc_corrente = bess_mwh * 0.5
    soc_hydro = hydro_bacino_max_mwh * 0.5
    prod_pv_array = produzione_pv * pv_mw
    prod_wind_array = produzione_wind * wind_mw
    potenza_nucleare_costante = nucleare_mw * 0.90
    gas_usato_totale, deficit_totale, overgen_totale = 0.0, 0.0, 0.0
    hydro_dispatched_totale, bess_scarica_totale = 0.0, 0.0
    sqrt_eff = np.sqrt(efficienza_bess)

    for t in range(ore):
        soc_hydro += hydro_inflow_mw
        if soc_hydro > hydro_bacino_max_mwh: soc_hydro = hydro_bacino_max_mwh
        generazione_base = prod_pv_array[t] + prod_wind_array[t] + hydro_fluente_mw + potenza_nucleare_costante
        bilancio_netto = generazione_base - fabbisogno[t]

        if bilancio_netto > 0:
            spazio_libero_batteria = bess_mwh - soc_corrente
            potenza_assorbibile_max = spazio_libero_batteria / sqrt_eff
            potenza_carica_effettiva = min(bilancio_netto, bess_mw, potenza_assorbibile_max)
            soc_corrente += potenza_carica_effettiva * sqrt_eff
            overgen_totale += (bilancio_netto - potenza_carica_effettiva)
        else:
            energia_richiesta = abs(bilancio_netto)
            potenza_scarica_bess = min(energia_richiesta, bess_mw)
            energia_out_bess = potenza_scarica_bess / sqrt_eff
            if soc_corrente >= energia_out_bess:
                soc_corrente -= energia_out_bess
                energia_richiesta -= potenza_scarica_bess
                bess_scarica_totale += potenza_scarica_bess
            else:
                energia_disp_bess = soc_corrente * sqrt_eff
                soc_corrente = 0.0
                energia_richiesta -= energia_disp_bess
                bess_scarica_totale += energia_disp_bess

            if energia_richiesta > 0:
                potenza_scarica_hydro = min(energia_richiesta, hydro_bacino_mw)
                if soc_hydro >= potenza_scarica_hydro:
                    soc_hydro -= potenza_scarica_hydro
                    energia_richiesta -= potenza_scarica_hydro
                    hydro_dispatched_totale += potenza_scarica_hydro
                else:
                    hydro_dispatched_totale += soc_hydro
                    energia_richiesta -= soc_hydro
                    soc_hydro = 0.0

            if energia_richiesta > 0:
                uso_gas = min(energia_richiesta, gas_mw)
                gas_usato_totale += uso_gas
                deficit_totale += (energia_richiesta - uso_gas)

    return gas_usato_totale, deficit_totale, overgen_totale, hydro_dispatched_totale, bess_scarica_totale

@njit
def calcola_capacita_anno(anno, start_yr, end_yr, val_start, val_target, step_wise=False):
    if end_yr <= start_yr: end_yr = start_yr + 1 
    if anno <= start_yr: return val_start
    if anno >= end_yr: return val_target
    quota = (anno - start_yr) / (end_yr - start_yr)
    valore = val_start + (val_target - val_start) * quota
    if step_wise: return np.floor(valore)
    return valore

@njit
def simula_scenario_30_anni(prod_pv, prod_wind, fabbisogno, 
                            pv_target, wind_target, nuc_target, bess_target,
                            pv_sq, wind_sq, nuc_sq, bess_sq,
                            t_start, t_end, anni_transizione=30):
    # Restituisce i totali CUMULATI per i 30 anni di transizione
    gas_tot, def_tot, over_tot, hydro_disp_tot, bess_out_tot = 0.0, 0.0, 0.0, 0.0, 0.0
    pv_gen_tot, wind_gen_tot, nuc_gen_tot = 0.0, 0.0, 0.0
    bess_installed_tot_mwh_years = 0.0 # Per calcolare l'OPEX cumulato
    vre_gen_tot = 0.0 # Per calcolare i costi di integrazione
    
    ore_anno = len(fabbisogno)
    
    for anno in range(anni_transizione + 1):
        # Capacità in GW per questo specifico anno
        pv_gw = calcola_capacita_anno(anno, t_start['pv'], t_end['pv'], pv_sq, pv_target)
        wind_gw = calcola_capacita_anno(anno, t_start['wind'], t_end['wind'], wind_sq, wind_target)
        nuc_gw = calcola_capacita_anno(anno, t_start['nuc'], t_end['nuc'], nuc_sq, nuc_target, step_wise=True)
        bess_gwh = calcola_capacita_anno(anno, t_start['bess'], t_end['bess'], bess_sq, bess_target)
        
        gas, dfc, ovr, hyd, bss = simula_rete_light_fast(
            prod_pv, prod_wind, fabbisogno,
            pv_gw * 1000.0, wind_gw * 1000.0, nuc_gw * 1000.0, bess_gwh * 1000.0, 
            50000.0, 50000.0, 2500.0, 12000.0, 5000000.0, 2850.0
        )
        
        gas_tot += gas
        def_tot += dfc
        over_tot += ovr
        hydro_disp_tot += hyd
        bess_out_tot += bss
        bess_installed_tot_mwh_years += (bess_gwh * 1000.0)
        
        pv_gen_anno = np.sum(prod_pv * (pv_gw * 1000.0))
        wind_gen_anno = np.sum(prod_wind * (wind_gw * 1000.0))
        pv_gen_tot += pv_gen_anno
        wind_gen_tot += wind_gen_anno
        nuc_gen_tot += (nuc_gw * 1000.0 * 0.90 * ore_anno)
        vre_gen_tot += (pv_gen_anno + wind_gen_anno)
        
    return (gas_tot, def_tot, over_tot, hydro_disp_tot, bess_out_tot, 
            pv_gen_tot, wind_gen_tot, nuc_gen_tot, bess_installed_tot_mwh_years, vre_gen_tot)

@st.cache_data
def simula_motore_30_anni(array_pv, array_wind, array_fabbisogno, t_start, t_end, anni_transizione=30):
    # Griglia scenari (leggermente ridotta per garantire UI real-time, totale 256 scenari)
    scenari_pv_gw = [40, 70, 100, 150]
    scenari_wind_gw = [10, 30, 60, 90]
    scenari_bess_gwh = [10, 50, 150, 300]
    scenari_nuc_gw = [0, 5, 10, 20]

    # Status Quo di partenza (anno 0)
    pv_sq, wind_sq, nuc_sq, bess_sq = 40.0, 10.0, 0.0, 10.0
    risultati_30y = []

    for pv in scenari_pv_gw:
        for wind in scenari_wind_gw:
            for bess in scenari_bess_gwh:
                for nuc in scenari_nuc_gw:
                    (gas_tot, def_tot, over_tot, hydro_disp_tot, bess_out_tot, 
                     pv_gen_tot, wind_gen_tot, nuc_gen_tot, bess_inst_years, vre_tot) = simula_scenario_30_anni(
                        array_pv, array_wind, array_fabbisogno,
                        float(pv), float(wind), float(nuc), float(bess),
                        pv_sq, wind_sq, nuc_sq, bess_sq,
                        t_start, t_end, anni_transizione
                    )
                    
                    risultati_30y.append({
                        'Target_PV': pv, 'Target_Wind': wind, 'Target_BESS': bess, 'Target_Nuc': nuc,
                        'gas_mwh': gas_tot, 'deficit_mwh': def_tot, 'overgen_mwh': over_tot,
                        'hydro_disp_mwh': hydro_disp_tot, 'bess_scarica_mwh': bess_out_tot,
                        'pv_gen_mwh': pv_gen_tot, 'wind_gen_mwh': wind_gen_tot, 'nuc_gen_mwh': nuc_gen_tot,
                        'bess_inst_years': bess_inst_years, 'vre_gen_tot': vre_tot
                    })

    return risultati_30y

def applica_economia_cumulata(risultati_30y, fabbisogno_annuo_mwh, mercato, anni_transizione=30):
    # Fabbisogno totale sui 30 anni
    fabbisogno_cumulato = fabbisogno_annuo_mwh * (anni_transizione + 1)
    hydro_fluente_cumulato = 2500.0 * 8760 * (anni_transizione + 1)
    
    LCA_EMISSIONI = {'pv': 45.0, 'wind': 11.0, 'hydro': 24.0, 'nuc': 12.0, 'bess': 50.0, 'gas': 550.0}
    
    # Costo annualizzato BESS (CAPEX spalmato + OPEX annuo)
    wacc = mercato.get('wacc_bess', 0.05)
    vita = mercato.get('bess_vita', 15)
    opex_f_rate = mercato.get('bess_opex_fix', 0.015)
    crf = (wacc * (1 + wacc) ** vita) / ((1 + wacc) ** vita - 1) if wacc > 0 else 1 / vita
    costo_bess_annuo_per_mwh = (mercato['bess_capex'] * crf) + (mercato['bess_capex'] * opex_f_rate)

    storia = []
    for r in risultati_30y:
        # Costi cumulati (somma dei 30 anni)
        costo_pv_tot = r['pv_gen_mwh'] * mercato['cfd_pv']
        costo_wind_tot = r['wind_gen_mwh'] * mercato['cfd_wind']
        costo_nuc_tot = r['nuc_gen_mwh'] * mercato['cfd_nuc']
        costo_hydro_tot = (hydro_fluente_cumulato + r['hydro_disp_mwh']) * mercato['gas_eur_mwh']
        costo_gas_tot = r['gas_mwh'] * mercato['gas_eur_mwh']
        costo_bess_tot = r['bess_inst_years'] * costo_bess_annuo_per_mwh
        costo_blackout_tot = r['deficit_mwh'] * mercato['voll']
        
        # Integrazione di rete (semplificata sui totali)
        quota_vre_media = r['vre_gen_tot'] / fabbisogno_cumulato
        costo_base_integr = mercato['costo_base_integrazione'] * (quota_vre_media ** 2)
        costo_sistema_totale = r['vre_gen_tot'] * costo_base_integr # Semplificazione: no sconto bess qui
        
        costo_totale_30y = (costo_pv_tot + costo_wind_tot + costo_nuc_tot + costo_hydro_tot + 
                            costo_gas_tot + costo_bess_tot + costo_blackout_tot + costo_sistema_totale)
        
        costo_medio_bolletta = costo_totale_30y / fabbisogno_cumulato
        percentuale_gas = (r['gas_mwh'] / fabbisogno_cumulato) * 100
        
        # Emissioni cumulate LCA
        emi_tot = (r['pv_gen_mwh'] * LCA_EMISSIONI['pv'] +
                   r['wind_gen_mwh'] * LCA_EMISSIONI['wind'] +
                   (hydro_fluente_cumulato + r['hydro_disp_mwh']) * LCA_EMISSIONI['hydro'] +
                   r['nuc_gen_mwh'] * LCA_EMISSIONI['nuc'] +
                   r['bess_scarica_mwh'] * LCA_EMISSIONI['bess'] +
                   r['gas_mwh'] * LCA_EMISSIONI['gas'])
        
        carbon_intensity_media = emi_tot / fabbisogno_cumulato
        
        storia.append({
            'Configurazione': f"{r['Target_PV']}P|{r['Target_Wind']}W|{r['Target_BESS']}B|{r['Target_Nuc']}N",
            'Target_PV': r['Target_PV'], 'Target_Wind': r['Target_Wind'], 
            'Target_BESS': r['Target_BESS'], 'Target_Nuc': r['Target_Nuc'],
            'Costo_Medio_30y': costo_medio_bolletta,
            'Carbon_Intensity_30y': carbon_intensity_media,
            'Gas_%_30y': percentuale_gas,
            'Overgen_TWh_30y': r['overgen_mwh'] / 1e6,
            'Gas_Mld_30y': costo_gas_tot / 1e9
        })

    df_risultati = pd.DataFrame(storia)
    # Ottimo: minor emissione entro il +5% del costo minimo assoluto sui 30 anni
    min_costo = df_risultati['Costo_Medio_30y'].min()
    scenari_ok = df_risultati[df_risultati['Costo_Medio_30y'] <= min_costo * 1.05]
    miglior_config = scenari_ok.sort_values(by='Carbon_Intensity_30y').iloc[0].to_dict()
    
    return miglior_config, df_risultati

# ==========================================
# 4. INTERFACCIA UTENTE (STREAMLIT)
# ==========================================
st.title("⚡ Simulatore Mix Energetico PRO - Pareto Cumulato su 30 Anni")
st.markdown("Valuta le emissioni e i costi sull'**intero arco della transizione**. I ritardi burocratici si traducono in maggior gas bruciato prima che gli impianti entrino a regime.")

# --- SIDEBAR: TIMING & MERCATO ---
st.sidebar.header("⏱️ Tempi di Costruzione (Anni)")
st.sidebar.caption("0 = Oggi. Definisce l'anno di inizio installazione e l'anno in cui si raggiunge il Target di piano.")
anni_transizione = st.sidebar.slider("Orizzonte di transizione (Anni)", 10, 40, 30)

col_t1, col_t2 = st.sidebar.columns(2)
t_start = {
    'pv': col_t1.number_input("Inizio PV", 0, 40, 1),
    'wind': col_t1.number_input("Inizio Eol", 0, 40, 3),
    'bess': col_t1.number_input("Inizio BESS", 0, 40, 1),
    'nuc': col_t1.number_input("Inizio Nuc", 0, 40, 12, help="Lungo permitting")
}
t_end = {
    'pv': col_t2.number_input("Fine PV", 1, 40, 15),
    'wind': col_t2.number_input("Fine Eol", 1, 40, 18),
    'bess': col_t2.number_input("Fine BESS", 1, 40, 15),
    'nuc': col_t2.number_input("Fine Nuc", 1, 50, 22)
}

st.sidebar.header("⚙️ Mercato & LCA")
mercato = {
    'cfd_pv': st.sidebar.slider("CfD PV (€/MWh)", 20.0, 150.0, 60.0, step=5.0),
    'cfd_wind': st.sidebar.slider("CfD Wind (€/MWh)", 30.0, 150.0, 80.0, step=5.0),
    'cfd_nuc': st.sidebar.slider("CfD Nuc (€/MWh)", 50.0, 200.0, 120.0, step=5.0),
    'bess_capex': st.sidebar.slider("CAPEX BESS (€/MWh)", 50000.0, 300000.0, 100000.0, step=10000.0),
    'wacc_bess': 0.05, 'bess_opex_fix': 0.015, 'bess_vita': 15,
    'gas_eur_mwh': st.sidebar.slider("Gas (€/MWh)", 30.0, 300.0, 130.0, step=10.0),
    'costo_base_integrazione': 10.0, 'voll': 3000.0
}

try:
    cartella_script = os.path.dirname(os.path.abspath(__file__))
    df_completo = carica_dati(
        os.path.join(cartella_script, "dataset_fotovoltaico_produzione.csv"),
        os.path.join(cartella_script, "gme.xlsx"),
        os.path.join(cartella_script, "dataset_eolico_produzione.csv"),
        DEFAULT_PV_NORD_SHARE, DEFAULT_WIND_NORD_SHARE
    )
    
    # Numba type casting arrays
    array_pv = df_completo['Fattore_Capacita_PV'].to_numpy(dtype=np.float64)
    array_wind = df_completo['Fattore_Capacita_Wind'].to_numpy(dtype=np.float64)
    array_fabbisogno = df_completo['Fabbisogno_MW'].to_numpy(dtype=np.float64)
    
    # Crea una dict per Numba (necessita di un formato compatibile o tipi base, usiamo Numba Typed Dict o passiamo i parametri espansi)
    # Per semplicità in Streamlit st.cache espandiamo il dizionario nei valori base:
    t_start_numba = np.array([t_start['pv'], t_start['wind'], t_start['nuc'], t_start['bess']], dtype=np.float64)
    
    with st.spinner("Calcolo di 30 anni per tutti gli scenari... (Veloce grazie a Numba)"):
        # Chiamata al motore (usiamo i dict standard perché Numba riesce a gestire type deduction se wrappato bene, ma passiamo come Numba-safe array/dict)
        from numba.core import types
        from numba.typed import Dict
        d_start, d_end = Dict.empty(key_type=types.unicode_type, value_type=types.float64), Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        for k,v in t_start.items(): d_start[k] = float(v)
        for k,v in t_end.items(): d_end[k] = float(v)

        risultati_30y = simula_motore_30_anni(array_pv, array_wind, array_fabbisogno, d_start, d_end, anni_transizione)

    fabbisogno_annuo_mwh = df_completo['Fabbisogno_MW'].sum()
    miglior_config, df_plot = applica_economia_cumulata(risultati_30y, fabbisogno_annuo_mwh, mercato, anni_transizione)

    # --- RISULTATI ---
    st.subheader("🏆 Il Miglior Target a Regime (Vincitore della Maratona 30 Anni)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Costo Medio (30 Anni)", f"{miglior_config['Costo_Medio_30y']:.1f} €/MWh")
    col2.metric("Carbon Intensity Media", f"{miglior_config['Carbon_Intensity_30y']:.1f} gCO₂/kWh")
    col3.metric("Spesa Gas Cumulata", f"{miglior_config['Gas_Mld_30y']:.1f} Mld €")
    col4.metric("Spreco Rete Cumulato", f"{miglior_config['Overgen_TWh_30y']:.1f} TWh")

    st.markdown(
        f"**Target da raggiungere nell'anno prestabilito:** {miglior_config['Target_PV']} GW Solare | "
        f"{miglior_config['Target_Wind']} GW Eolico | {miglior_config['Target_BESS']} GWh Batterie | "
        f"{miglior_config['Target_Nuc']} GW Nucleare"
    )

    st.subheader("📊 Frontiera di Pareto: Costi vs Emissioni (CUMULATI su 30 Anni)")
    fig = px.scatter(
        df_plot, x='Carbon_Intensity_30y', y='Costo_Medio_30y', color='Target_Nuc',
        color_continuous_scale='Plasma', hover_data=['Target_PV', 'Target_Wind', 'Target_BESS'],
        labels={
            'Carbon_Intensity_30y': "Carbon Intensity Media 30 Anni (gCO₂/kWh)",
            'Costo_Medio_30y': "Costo Medio Bolletta 30 Anni (€/MWh)",
            'Target_Nuc': "Nucleare (GW)"
        }
    )
    fig.add_trace(go.Scatter(
        x=[miglior_config['Carbon_Intensity_30y']], y=[miglior_config['Costo_Medio_30y']],
        mode='markers', marker=dict(color='lime', size=15, line=dict(color='black', width=2)),
        name='Ottimo Scelto'
    ))
    fig.update_layout(xaxis_autorange="reversed", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- TRAIETTORIA DELL'OTTIMO ---
    st.markdown("---")
    st.subheader("🛤️ Traiettoria dell'Ottimo e Consumo di Gas")
    
    # Rigeneriamo i dati annuali dell'ottimo per plottare l'area chart
    storia_t = []
    pv_sq, wind_sq, nuc_sq, bess_sq = 40.0, 10.0, 0.0, 10.0
    
    # Helper python per la UI (evitiamo Numba qui che è solo UI)
    def calc_cap_py(anno, s, e, start_v, tgt_v, step=False):
        if e <= s: e = s + 1
        if anno <= s: return start_v
        if anno >= e: return tgt_v
        v = start_v + (tgt_v - start_v) * ((anno - s)/(e - s))
        return np.floor(v) if step else v

    for anno in range(anni_transizione + 1):
        pv_gw = calc_cap_py(anno, t_start['pv'], t_end['pv'], pv_sq, miglior_config['Target_PV'])
        wind_gw = calc_cap_py(anno, t_start['wind'], t_end['wind'], wind_sq, miglior_config['Target_Wind'])
        nuc_gw = calc_cap_py(anno, t_start['nuc'], t_end['nuc'], nuc_sq, miglior_config['Target_Nuc'], True)
        bess_gwh = calc_cap_py(anno, t_start['bess'], t_end['bess'], bess_sq, miglior_config['Target_BESS'])
        
        # Simulazione veloce Python solo per l'ottimo, per plottare il grafico
        gas, _, _, _, _ = simula_rete_light_fast(
            array_pv, array_wind, array_fabbisogno, pv_gw*1000, wind_gw*1000, nuc_gw*1000, bess_gwh*1000,
            50000, 50000, 2500, 12000, 5000000, 2850
        )
        storia_t.append({'Anno': anno, 'PV_GW': pv_gw, 'Wind_GW': wind_gw, 'Nuc_GW': nuc_gw, 'Gas_TWh': gas/1e6})

    df_t = pd.DataFrame(storia_t)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['PV_GW'], mode='lines', stackgroup='one', name='PV (GW)', fillcolor='gold'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Wind_GW'], mode='lines', stackgroup='one', name='Wind (GW)', fillcolor='lightskyblue'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Nuc_GW'], mode='lines', stackgroup='one', name='Nuc (GW)', fillcolor='mediumpurple'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Gas_TWh'], mode='lines+markers', name='Gas (TWh/anno)', line=dict(color='red', width=3)), secondary_y=True)
    
    fig2.update_layout(hovermode="x unified", height=450)
    fig2.update_yaxes(title_text="Capacità Installata (GW)", secondary_y=False)
    fig2.update_yaxes(title_text="Gas Bruciato (TWh)", secondary_y=True, range=[0, df_t['Gas_TWh'].max()*1.2])
    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Errore: {e}")
