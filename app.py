import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ==========================================
# CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(page_title="Simulatore Mix Energetico PRO", layout="wide")

# ==========================================
# 1. FUNZIONI DI CARICAMENTO DATI
# ==========================================
@st.cache_data
def carica_dati(file_pvgis, file_gme, file_wind):
    # Caricamento PVGIS
    skip_lines = 0
    with open(file_pvgis, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                skip_lines = i
                break
    df_pv = pd.read_csv(file_pvgis, skiprows=skip_lines, usecols=[0, 1], names=['time', 'PV_data'], header=0)
    df_pv['time'] = pd.to_datetime(df_pv['time'], format='%Y%m%d:%H%M', errors='coerce')
    df_pv.dropna(subset=['time'], inplace=True) 
    df_pv['PV_data'] = pd.to_numeric(df_pv['PV_data'], errors='coerce').fillna(0)
    df_pv['time'] = df_pv['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Rome').dt.tz_localize(None)
    df_pv['Fattore_Capacita_PV'] = (df_pv['PV_data'] / 1000.0).clip(upper=1.0)
    df_pv.set_index('time', inplace=True)
    df_pv = df_pv[['Fattore_Capacita_PV']]

    # Caricamento GME
    df_gme = pd.read_excel(file_gme, engine='openpyxl')
    colonna_volumi = df_gme.columns[2] 
    if df_gme[colonna_volumi].dtype == 'object':
        df_gme[colonna_volumi] = df_gme[colonna_volumi].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df_gme[colonna_volumi] = pd.to_numeric(df_gme[colonna_volumi], errors='coerce')
    data_convertita = pd.to_datetime(df_gme['Data'], dayfirst=True, errors='coerce')
    ore_aggiuntive = pd.to_timedelta(df_gme['Ora'] - 1, unit='h')
    df_gme['Datetime'] = data_convertita + ore_aggiuntive
    df_gme.set_index('Datetime', inplace=True)
    df_gme.rename(columns={colonna_volumi: 'Fabbisogno_MW'}, inplace=True)
    df_gme = df_gme[['Fabbisogno_MW']].dropna()

    # Caricamento Vento
    skip_lines = 0
    with open(file_wind, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                skip_lines = i
                break
    df_wind = pd.read_csv(file_wind, skiprows=skip_lines) 
    df_wind['time'] = pd.to_datetime(df_wind['time'])
    df_wind['time'] = df_wind['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Rome').dt.tz_localize(None)
    df_wind.set_index('time', inplace=True)
    df_wind.rename(columns={'electricity': 'Fattore_Capacita_Wind'}, inplace=True)
    df_wind = df_wind[['Fattore_Capacita_Wind']]

    # Sincronizzazione Anni
    anno_gme = df_gme.index[0].year
    df_pv.index = df_pv.index + pd.DateOffset(years=(anno_gme - df_pv.index[0].year))
    df_pv.index = df_pv.index.round('H')
    df_wind.index = df_wind.index + pd.DateOffset(years=(anno_gme - df_wind.index[0].year))
    df_wind.index = df_wind.index.round('H')
    
    df_completo = df_gme.join(df_pv, how='inner').join(df_wind, how='inner')
    df_completo.ffill(inplace=True)
    
    return df_completo

# ==========================================
# 2. SIMULAZIONE FISICA AVANZATA
# ==========================================
@njit
def simula_rete_light_fast(produzione_pv, produzione_wind, fabbisogno, 
                           pv_mw, wind_mw, nucleare_mw, bess_mwh, bess_mw, gas_mw,
                           hydro_fluente_mw, hydro_bacino_mw, hydro_bacino_max_mwh, hydro_inflow_mw,
                           efficienza_bess=0.9):
    ore = len(fabbisogno)
    
    soc_corrente = bess_mwh * 0.5 
    soc_hydro = hydro_bacino_max_mwh * 0.5 
    
    prod_pv_array = produzione_pv * pv_mw
    prod_wind_array = produzione_wind * wind_mw
    potenza_nucleare_costante = nucleare_mw * 0.90 
    
    gas_usato_totale = 0.0
    deficit_totale = 0.0
    overgen_totale = 0.0
    hydro_dispatched_totale = 0.0
    bess_scarica_totale = 0.0 # NUOVO: Tiene traccia del lavoro delle batterie
    
    sqrt_eff = np.sqrt(efficienza_bess)
    
    for t in range(ore):
        soc_hydro += hydro_inflow_mw
        if soc_hydro > hydro_bacino_max_mwh:
            soc_hydro = hydro_bacino_max_mwh
            
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
            
            # A. Le batterie si scaricano
            potenza_scarica_bess = min(energia_richiesta, bess_mw)
            energia_out_bess = potenza_scarica_bess / sqrt_eff
            if soc_corrente >= energia_out_bess:
                soc_corrente -= energia_out_bess
                energia_richiesta -= potenza_scarica_bess
                bess_scarica_totale += potenza_scarica_bess # Registriamo i MWh scaricati
            else:
                energia_disp_bess = soc_corrente * sqrt_eff
                soc_corrente = 0.0
                energia_richiesta -= energia_disp_bess
                bess_scarica_totale += energia_disp_bess # Registriamo i MWh scaricati
                
            # B. Dighe Idroelettriche
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
                    
            # C. Gas Naturale
            if energia_richiesta > 0:
                uso_gas = min(energia_richiesta, gas_mw)
                gas_usato_totale += uso_gas
                deficit_totale += (energia_richiesta - uso_gas)
                
    return gas_usato_totale, deficit_totale, overgen_totale, hydro_dispatched_totale, bess_scarica_totale

# ==========================================
# 3. MOTORE DI OTTIMIZZAZIONE
# ==========================================
def calcola_costo_bolletta(pv_mw, wind_mw, nucleare_mw, bess_mwh, gas_mwh_annuo, deficit_mwh_annuo, 
                           ore_eq_pv, ore_eq_wind, hydro_fluente_tot_mwh, hydro_dispatched_tot_mwh, fabbisogno_tot_mwh, mercato):
    
    costo_annuo_pv = (pv_mw * ore_eq_pv) * mercato['cfd_pv']
    costo_annuo_wind = (wind_mw * ore_eq_wind) * mercato['cfd_wind']
    costo_annuo_hydro = (hydro_fluente_tot_mwh + hydro_dispatched_tot_mwh) * mercato['gas_eur_mwh'] 
    costo_annuo_nuc = (nucleare_mw * 0.90 * 8760) * mercato['cfd_nuc']
    costo_annuo_bess = (bess_mwh * mercato['bess_capex']) / mercato['bess_vita']
    costo_annuo_gas = gas_mwh_annuo * mercato['gas_eur_mwh']
    costo_annuo_blackout = deficit_mwh_annuo * mercato['voll']
    
    totale = costo_annuo_pv + costo_annuo_wind + costo_annuo_hydro + costo_annuo_nuc + costo_annuo_bess + costo_annuo_gas + costo_annuo_blackout
    return totale / fabbisogno_tot_mwh

def ottimizza_sistema(df_completo, mercato):
    scenari_pv_gw = [40, 50, 60, 100, 150]
    scenari_wind_gw = [10, 20, 30, 60, 90]
    scenari_bess_gwh = [0, 20, 50, 150, 300,500]
    scenari_nuc_gw = [0,5,10, 20, 40, 60]
    
    GAS_CAPACITA_FISSA_MW = 50000  
    BESS_POTENZA_FISSA_MW = 50000  
    HYDRO_FLUENTE_MW = 2500.0      
    HYDRO_BACINO_MW = 12000.0      
    HYDRO_BACINO_MAX_MWH = 5000000.0 
    HYDRO_INFLOW_MW = 2850.0       
    
    fabbisogno_tot_mwh = df_completo['Fabbisogno_MW'].sum()
    ore_totali = len(df_completo)
    ore_eq_pv = df_completo['Fattore_Capacita_PV'].sum()
    ore_eq_wind = df_completo['Fattore_Capacita_Wind'].sum()
    
    hydro_fluente_tot_mwh = HYDRO_FLUENTE_MW * ore_totali
    
    array_pv = df_completo['Fattore_Capacita_PV'].values
    array_wind = df_completo['Fattore_Capacita_Wind'].values
    array_fabbisogno = df_completo['Fabbisogno_MW'].values
    
    # Valori di emissione LCA (IPCC + Utente) in gCO2/kWh (che equivale a kgCO2/MWh)
    LCA_EMISSIONI = {
        'pv': 45.0,
        'wind': 11.0,
        'hydro': 24.0,
        'nuc': 12.0,
        'bess': 50.0,
        'gas': 550.0
    }
    
    miglior_costo = float('inf')
    miglior_config = None
    storia = []
    
    progress_bar = st.progress(0)
    totale_iter = len(scenari_pv_gw) * len(scenari_wind_gw) * len(scenari_bess_gwh) * len(scenari_nuc_gw)
    iter_corrente = 0
    
    for pv in scenari_pv_gw:
        for wind in scenari_wind_gw:
            for bess in scenari_bess_gwh:
                for nuc in scenari_nuc_gw:
                    
                    pv_mw = pv * 1000.0
                    wind_mw = wind * 1000.0
                    nuc_mw = nuc * 1000.0
                    bess_mwh = bess * 1000.0
                    
                    gas_mwh, deficit_mwh, overgen_mwh, hydro_disp_mwh, bess_scarica_mwh = simula_rete_light_fast(
                        array_pv, array_wind, array_fabbisogno, 
                        pv_mw, wind_mw, nuc_mw, bess_mwh, BESS_POTENZA_FISSA_MW, GAS_CAPACITA_FISSA_MW,
                        HYDRO_FLUENTE_MW, HYDRO_BACINO_MW, HYDRO_BACINO_MAX_MWH, HYDRO_INFLOW_MW
                    )
                    
                    costo_bolletta = calcola_costo_bolletta(
                        pv_mw, wind_mw, nuc_mw, bess_mwh, gas_mwh, deficit_mwh, 
                        ore_eq_pv, ore_eq_wind, hydro_fluente_tot_mwh, hydro_disp_mwh, fabbisogno_tot_mwh, mercato
                    )
                    
                    # --- NUOVO CALCOLO EMISSIONI LCA (IPCC) ---
                    # Moltiplichiamo l'energia totale prodotta/erogata da ogni fonte per il suo fattore emissivo
                    emissioni_pv = (pv_mw * ore_eq_pv) * LCA_EMISSIONI['pv']
                    emissioni_wind = (wind_mw * ore_eq_wind) * LCA_EMISSIONI['wind']
                    emissioni_hydro = (hydro_fluente_tot_mwh + hydro_disp_mwh) * LCA_EMISSIONI['hydro']
                    emissioni_nuc = (nuc_mw * 0.90 * 8760) * LCA_EMISSIONI['nuc']
                    emissioni_bess = bess_scarica_mwh * LCA_EMISSIONI['bess']
                    emissioni_gas = gas_mwh * LCA_EMISSIONI['gas']
                    
                    emissioni_totali_kg = emissioni_pv + emissioni_wind + emissioni_hydro + emissioni_nuc + emissioni_bess + emissioni_gas
                    
                    # gCO2/kWh = kgCO2 / MWh totali
                    carbon_intensity = emissioni_totali_kg / fabbisogno_tot_mwh
                    
                    storia.append({
                        'Configurazione': f"{pv}PV|{wind}W|{bess}B|{nuc}N",
                        'PV_GW': pv, 'Wind_GW': wind, 'BESS_GWh': bess, 'Nuc_GW': nuc,
                        'Costo_Bolletta': costo_bolletta,
                        'Carbon_Intensity': carbon_intensity,
                        'Overgen_TWh': overgen_mwh / 1e6
                    })
                    
                    if costo_bolletta < miglior_costo:
                        miglior_costo = costo_bolletta
                        miglior_config = storia[-1]
                        
                    iter_corrente += 1
                    progress_bar.progress(iter_corrente / totale_iter)
                    
    progress_bar.empty() 
    return miglior_config, pd.DataFrame(storia)

# ==========================================
# 4. INTERFACCIA UTENTE (STREAMLIT)
# ==========================================
st.title("⚡ Ottimizzatore Mix Energetico e Decarbonizzazione (LCA)")
st.markdown("Scopri l'equilibrio tra Rinnovabili, Batterie e Nucleare valutando le emissioni dell'intero ciclo di vita.")

@st.dialog("📖 Come funziona questo simulatore?")
def mostra_spiegazione():
    st.markdown("""
    **Benvenuto nel Simulatore di Mix Energetico 1.0!**

    *Si tratta di una Beta vibecodata, se vuoi darmi una mano a svilupparla scrivi a giovanni at unbelclima punto it*
    
    Questo tool calcola l'Ottimo di Pareto tra costi in bolletta ed emissioni di CO₂, 
    simula la rete elettrica ora per ora (8760 ore annue).
    Si basa su dati reali di produzione fotovoltaica ed eolica.
    
    ### 🌿 Modello LCA (Life Cycle Assessment)
    Le emissioni sono ora calcolate sull'intero ciclo di vita delle tecnologie (costruzione, estrazione mineraria, smaltimento), usando i dati mediani **IPCC**:
    - **Fotovoltaico:** 45 gCO₂/kWh
    - **Eolico:** 11 gCO₂/kWh
    - **Idroelettrico:** 24 gCO₂/kWh
    - **Nucleare:** 12 gCO₂/kWh
    - **Batterie:** 50 gCO₂/kWh (per energia erogata)
    - **Gas Naturale:** 550 gCO₂/kWh
    """)

col_vuota, col_bottone = st.columns([4, 1])
with col_bottone:
    if st.button("ℹ️ Info / Istruzioni / Fonti"):
        mostra_spiegazione()

st.sidebar.header("⚙️ Parametri di Mercato")
mercato = {
    'cfd_pv': st.sidebar.slider("CfD Fotovoltaico (€/MWh)", 20.0, 150.0, 50.0, step=5.0),
    'cfd_wind': st.sidebar.slider("CfD Eolico (€/MWh)", 30.0, 150.0, 70.0, step=5.0),
    'cfd_nuc': st.sidebar.slider("CfD Nucleare (€/MWh)", 50.0, 200.0, 120.0, step=5.0),
    'bess_capex': st.sidebar.slider("CAPEX Batterie (€/MWh installato)", 50000.0, 300000.0, 100000.0, step=10000.0),
    'gas_eur_mwh': st.sidebar.slider("Prezzo Gas / Fossili (€/MWh)", 30.0, 300.0, 130.0, step=10.0),
    'bess_vita': 15,     
    'voll': 3000.0
}

try:
    cartella_script = os.path.dirname(os.path.abspath(__file__))
    file_pvgis = os.path.join(cartella_script, "pvgis.csv")
    file_gme = os.path.join(cartella_script, "gme.xlsx")
    file_wind = os.path.join(cartella_script, "wind.csv")
    
    df_completo = carica_dati(file_pvgis, file_gme, file_wind)
    
    miglior_config, df_plot = ottimizza_sistema(df_completo, mercato)
    
    st.subheader("🏆 Il Miglior Compromesso (Ottimo Economico)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Costo Bolletta", f"{miglior_config['Costo_Bolletta']:.1f} €/MWh")
    col2.metric("Carbon Intensity (LCA)", f"{miglior_config['Carbon_Intensity']:.1f} gCO₂/kWh")
    col3.metric("Nucleare Richiesto", f"{miglior_config['Nuc_GW']} GW")
    col4.metric("Batterie Richieste", f"{miglior_config['BESS_GWh']} GWh")
    
    st.markdown(f"**Mix Impianti:** {miglior_config['PV_GW']} GW Solare | {miglior_config['Wind_GW']} GW Eolico | **Spreco Rete:** {miglior_config['Overgen_TWh']:.1f} TWh/anno")
    
    st.subheader("📊 Frontiera di Pareto: Costi vs Emissioni (Intero Ciclo di Vita)")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(df_plot['Carbon_Intensity'], df_plot['Costo_Bolletta'], 
                         c=df_plot['Nuc_GW'], cmap='plasma', s=100, alpha=0.8, edgecolors='black')
    
    ax.scatter(miglior_config['Carbon_Intensity'], miglior_config['Costo_Bolletta'], 
               facecolors='none', edgecolors='lime', s=300, linewidth=3, label="Miglior Mix Economico")

    ax.set_xlabel("Carbon Intensity Media LCA (gCO₂ / kWh)")
    ax.set_ylabel("Costo Medio in Bolletta (€ / MWh)")
    ax.invert_xaxis() 
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Capacità Nucleare Installata (GW)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

except FileNotFoundError:
    st.error("⚠️ File dati non trovati! Assicurati che i file `pvgis.csv`, `gme.xlsx` e `wind.csv` siano nella stessa cartella di questo script.")
