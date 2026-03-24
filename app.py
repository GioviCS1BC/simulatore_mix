import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

    anno_gme = df_gme.index[0].year
    df_pv.index = df_pv.index + pd.DateOffset(years=(anno_gme - df_pv.index[0].year))
    df_pv.index = df_pv.index.round('H')
    df_wind.index = df_wind.index + pd.DateOffset(years=(anno_gme - df_wind.index[0].year))
    df_wind.index = df_wind.index.round('H')
    
    df_completo = df_gme.join(df_pv, how='inner').join(df_wind, how='inner')
    df_completo.ffill(inplace=True)
    return df_completo

# ==========================================
# 2. SIMULAZIONE FISICA (Numba)
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
    bess_scarica_totale = 0.0
    
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

# ==========================================
# 3. MOTORE DI CALCOLO SEPARATO (Cache ottimizzata)
# ==========================================
# L'underscore '_' davanti a _df_completo dice a Streamlit di NON consumare RAM per analizzarlo!
@st.cache_data
def simula_tutti_scenari_fisici(_df_completo):
    scenari_pv_gw = [40, 50, 80, 100, 150]
    scenari_wind_gw = [10, 20, 30, 60, 90]
    scenari_bess_gwh = [10, 30, 50, 150, 300]
    scenari_nuc_gw = [0, 2, 5, 10, 15, 20, 25, 30]
    
    GAS_CAPACITA_FISSA_MW = 50000  
    BESS_POTENZA_FISSA_MW = 50000  
    HYDRO_FLUENTE_MW = 2500.0      
    HYDRO_BACINO_MW = 12000.0      
    HYDRO_BACINO_MAX_MWH = 5000000.0 
    HYDRO_INFLOW_MW = 2850.0 
    
    array_pv = _df_completo['Fattore_Capacita_PV'].values
    array_wind = _df_completo['Fattore_Capacita_Wind'].values
    array_fabbisogno = _df_completo['Fabbisogno_MW'].values
    
    risultati_fisici = []
    
    for pv in scenari_pv_gw:
        for wind in scenari_wind_gw:
            for bess in scenari_bess_gwh:
                for nuc in scenari_nuc_gw:
                    gas_mwh, def_mwh, over_mwh, hydro_disp_mwh, bess_out_mwh = simula_rete_light_fast(
                        array_pv, array_wind, array_fabbisogno, 
                        pv*1000.0, wind*1000.0, nuc*1000.0, bess*1000.0, BESS_POTENZA_FISSA_MW, GAS_CAPACITA_FISSA_MW,
                        HYDRO_FLUENTE_MW, HYDRO_BACINO_MW, HYDRO_BACINO_MAX_MWH, HYDRO_INFLOW_MW
                    )
                    
                    risultati_fisici.append({
                        'PV_GW': pv, 'Wind_GW': wind, 'BESS_GWh': bess, 'Nuc_GW': nuc,
                        'gas_mwh': gas_mwh, 'deficit_mwh': def_mwh, 'overgen_mwh': over_mwh,
                        'hydro_disp_mwh': hydro_disp_mwh, 'bess_scarica_mwh': bess_out_mwh
                    })
                    
    return risultati_fisici

def applica_economia_e_trova_ottimo(risultati_fisici, df_completo, mercato):
    fabbisogno_tot_mwh = df_completo['Fabbisogno_MW'].sum()
    ore_eq_pv = df_completo['Fattore_Capacita_PV'].sum()
    ore_eq_wind = df_completo['Fattore_Capacita_Wind'].sum()
    hydro_fluente_tot_mwh = 2500.0 * len(df_completo)
    
    LCA_EMISSIONI = {'pv': 45.0, 'wind': 11.0, 'hydro': 24.0, 'nuc': 12.0, 'bess': 50.0, 'gas': 550.0}
    
    # --- PARAMETRI FINANZIARI BESS ---
    wacc = mercato.get('wacc_bess', 0.05)      # Default 5% se non presente
    vita = mercato.get('bess_vita', 15)
    opex_f_rate = mercato.get('bess_opex_fix', 0.015) # Default 1.5% del CAPEX/anno
    
    # Calcolo del Fattore di Recupero del Capitale (Rata di ammortamento)
    if wacc > 0:
        crf = (wacc * (1 + wacc)**vita) / ((1 + wacc)**vita - 1)
    else:
        crf = 1 / vita

    miglior_costo = float('inf')
    miglior_config = None
    storia = []
    
    for r in risultati_fisici:
        pv_mw = r['PV_GW'] * 1000.0
        wind_mw = r['Wind_GW'] * 1000.0
        nuc_mw = r['Nuc_GW'] * 1000.0
        bess_mwh = r['BESS_GWh'] * 1000.0
        
        # --- COSTI FONTI ---
        costo_pv = (pv_mw * ore_eq_pv) * mercato['cfd_pv']
        costo_wind = (wind_mw * ore_eq_wind) * mercato['cfd_wind']
        costo_hydro = (hydro_fluente_tot_mwh + r['hydro_disp_mwh']) * mercato['gas_eur_mwh'] 
        costo_nuc = (nuc_mw * 1 * 8760) * mercato['cfd_nuc']
        
        # --- NUOVO CALCOLO COSTO BESS (Finanziario + Fisso) ---
        capex_investimento = bess_mwh * mercato['bess_capex']
        quota_ammortamento = capex_investimento * crf
        quota_opex_fissa = capex_investimento * opex_f_rate
        costo_bess = quota_ammortamento + quota_opex_fissa
        
        costo_gas = r['gas_mwh'] * mercato['gas_eur_mwh']
        costo_blackout = r['deficit_mwh'] * mercato['voll']
        # --- CALCOLO PENETRAZIONE RINNOVABILI (VRE) ---
        energia_vre_totale = (pv_mw * ore_eq_pv) + (wind_mw * ore_eq_wind)
        quota_vre = energia_vre_totale / fabbisogno_tot_mwh
        
        # --- COSTI DI SISTEMA (Rete, Bilanciamento, Inerzia) ---
        # Più aumenta la quota VRE, più Terna deve spendere per tenere la rete stabile.
        # Usiamo una funzione quadratica: 0€ al 0%, ~25€/MWh al 100% di penetrazione.
        costo_unitario_integrazione = 25.0 * (quota_vre ** 2) 
        costo_sistema_totale = energia_vre_totale * costo_unitario_integrazione
        
        # --- CALCOLO FINALE BOLLETTA ---
        # Sommiamo il nuovo costo di sistema al numeratore
        costo_bolletta = (
            costo_pv + costo_wind + costo_hydro + costo_nuc + 
            costo_bess + costo_gas + costo_blackout + 
            costo_sistema_totale  # <--- AGGIUNTO QUI
        ) / fabbisogno_tot_mwh
        # --- EMISSIONI LCA ---
        emi_pv = (pv_mw * ore_eq_pv) * LCA_EMISSIONI['pv']
        emi_wind = (wind_mw * ore_eq_wind) * LCA_EMISSIONI['wind']
        emi_hydro = (hydro_fluente_tot_mwh + r['hydro_disp_mwh']) * LCA_EMISSIONI['hydro']
        emi_nuc = (nuc_mw * 1 * 8760) * LCA_EMISSIONI['nuc']
        emi_bess = r['bess_scarica_mwh'] * LCA_EMISSIONI['bess']
        emi_gas = r['gas_mwh'] * LCA_EMISSIONI['gas']
        
        carbon_intensity = (emi_pv + emi_wind + emi_hydro + emi_nuc + emi_bess + emi_gas) / fabbisogno_tot_mwh
        
        config = {
            'Configurazione': f"{r['PV_GW']}PV|{r['Wind_GW']}W|{r['BESS_GWh']}B|{r['Nuc_GW']}N",
            'PV_GW': r['PV_GW'], 'Wind_GW': r['Wind_GW'], 'BESS_GWh': r['BESS_GWh'], 'Nuc_GW': r['Nuc_GW'],
            'Costo_Bolletta': costo_bolletta,
            'Carbon_Intensity': carbon_intensity,
            'Overgen_TWh': r['overgen_mwh'] / 1e6
        }
        storia.append(config)
        
        if costo_bolletta < miglior_costo:
            miglior_costo = costo_bolletta
            miglior_config = config
            
    return miglior_config, pd.DataFrame(storia)

# ==========================================
# 4. INTERFACCIA UTENTE (STREAMLIT)
# ==========================================
st.title("⚡ Ottimizzatore Mix Energetico e Decarbonizzazione (BETA)")
st.markdown("Scopri l'equilibrio tra Rinnovabili, Batterie e Nucleare valutando le emissioni dell'intero ciclo di vita.")

@st.dialog("📖 Come funziona questo simulatore?")
def mostra_spiegazione():
    st.markdown("""
    **Benvenuto nel Simulatore di Mix Energetico 1.0 di CS1BC!**
    per smanettare coi parametri clicca le freccette in alto a sinistra e aggiorni il risultato della funzione obiettivo
    Cos'è CS1BC? è un collettivo strafigo! unbelclima.it

    
    ATTENZIONE!:
    i dataset di produzione rinnovabile sono reali ora per ora ma per ora puntuali (presi in centro Italia), 
        nelle prossime versioni simuleremo una generazione distribuita
    
    ### 🌿 Modello LCA (Life Cycle Assessment)
    Le emissioni sono calcolate sull'intero ciclo di vita (dati IPCC):
    - **Fotovoltaico:** 45 gCO₂/kWh
    - **Eolico:** 11 gCO₂/kWh
    - **Idroelettrico:** 24 gCO₂/kWh
    - **Nucleare:** 12 gCO₂/kWh
    - **Batterie:** 50 gCO₂/kWh (per energia erogata)
    - **Gas Naturale:** 550 gCO₂/kWh
    *Si tratta di una Beta vibecodata, se vuoi darmi una mano a svilupparla scrivi a giovanni at unbelclima punto it*
    *guarda il modello su: https://github.com/GioviCS1BC/simulatore_mix/ *
    """)

col_vuota, col_bottone = st.columns([4, 1])
with col_bottone:
    if st.button("ℹ️ Info / Istruzioni / Fonti"):
        mostra_spiegazione()

st.sidebar.header("⚙️ Parametri di Mercato")
mercato = {
    'cfd_pv': st.sidebar.slider("CfD Fotovoltaico (€/MWh)", 20.0, 150.0, 60.0, step=5.0),
    'cfd_wind': st.sidebar.slider("CfD Eolico (€/MWh)", 30.0, 150.0, 80.0, step=5.0),
    'cfd_nuc': st.sidebar.slider("CfD Nucleare (€/MWh)", 50.0, 200.0, 120.0, step=5.0),
    'bess_capex': st.sidebar.slider("CAPEX Batterie (€/MWh installato)", 50000.0, 300000.0, 100000.0, step=10000.0),
    'wacc_bess': st.sidebar.slider("WACC Batterie (%)", 0.0, 15.0, 5.0, step=0.5) / 100,
    'bess_opex_fix': st.sidebar.slider("Manutenzione Annua BESS (% del CAPEX)", 0.0, 5.0, 1.5, step=0.1) / 100,
    'bess_vita': 15,
    'gas_eur_mwh': st.sidebar.slider("Prezzo Gas / Fossili (€/MWh)", 30.0, 300.0, 130.0, step=10.0),    
    'voll': 3000.0
}

try:
    cartella_script = os.path.dirname(os.path.abspath(__file__))
    file_pvgis = os.path.join(cartella_script, "pvgis.csv")
    file_gme = os.path.join(cartella_script, "gme.xlsx")
    file_wind = os.path.join(cartella_script, "wind.csv")
    
    df_completo = carica_dati(file_pvgis, file_gme, file_wind)
    
    with st.spinner("Calcolo della rete elettrica... (Solo al primo avvio)"):
        risultati_fisici = simula_tutti_scenari_fisici(df_completo)
    
    miglior_config, df_plot = applica_economia_e_trova_ottimo(risultati_fisici, df_completo, mercato)
    
    st.subheader("🏆 Il Miglior Compromesso (Ottimo Economico)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Costo Bolletta", f"{miglior_config['Costo_Bolletta']:.1f} €/MWh")
    col2.metric("Carbon Intensity (LCA)", f"{miglior_config['Carbon_Intensity']:.1f} gCO₂/kWh")
    col3.metric("Nucleare Richiesto", f"{miglior_config['Nuc_GW']} GW")
    col4.metric("Batterie Richieste", f"{miglior_config['BESS_GWh']} GWh")
    
    st.markdown(f"**Mix Impianti:** {miglior_config['PV_GW']} GW Solare | {miglior_config['Wind_GW']} GW Eolico | **Spreco Rete:** {miglior_config['Overgen_TWh']:.1f} TWh/anno")
    
    st.subheader("📊 Frontiera di Pareto: Costi vs Emissioni (Interattivo!)")
    
    # --- IL NUOVO GRAFICO PLOTLY (Zero RAM, 100% Interattivo) ---
    fig = px.scatter(
        df_plot, 
        x='Carbon_Intensity', 
        y='Costo_Bolletta', 
        color='Nuc_GW',
        color_continuous_scale='Plasma',
        hover_data=['PV_GW', 'Wind_GW', 'BESS_GWh'], # Mostra questi dati se passi il mouse!
        labels={
            'Carbon_Intensity': "Carbon Intensity Media LCA (gCO₂/kWh)",
            'Costo_Bolletta': "Costo Medio in Bolletta (€/MWh)",
            'Nuc_GW': "Nucleare (GW)"
        }
    )
    
    # Aggiungiamo il cerchio verde per l'ottimo economico
    fig.add_trace(go.Scatter(
        x=[miglior_config['Carbon_Intensity']],
        y=[miglior_config['Costo_Bolletta']],
        mode='markers',
        marker=dict(color='lime', size=15, line=dict(color='black', width=2)),
        name='Mix + Economico',
        hoverinfo='skip'
    ))

    fig.update_layout(xaxis_autorange="reversed", height=600)
    
    # Renderizza il grafico in Streamlit
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("⚠️ File dati non trovati! Assicurati che i file `pvgis.csv`, `gme.xlsx` e `wind.csv` siano nel cloud.")
