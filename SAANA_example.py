# -*- coding: utf-8 -*-
"""
SAANA_example.py

Scrip to evaluate and visualize nutrient loads:
    
    - VEMALA-data:
        - valuma-alueet
        - maa-alueet
        - vesimuodostumat
    
    - Parameters:
        - N
        - P
        - TOC
        
    - Ramboll-data:
        - vesiensuojelurakenteet (VIRHEELLINEN) -> saatu uusiksi Sara Vaskiolta (Raaseporinjoki-hanke)
        
    - Data (Accessed 30.5.2024):
        - WSFS-Vemala kuormitustiedot (Syke: https://ckan.ymparisto.fi/dataset/%7B7E6EB982-A3CA-4DE3-87C0-1B61462442DF%7D)
        - Corine maanpeite 2018 (Syke: https://ckan.ymparisto.fi/dataset/%7B0B4B2FAC-ADF1-43A1-A829-70F02BF0C0E5%7D)
        - Valuma-aluejako 1990 (Syke: https://ckan.ymparisto.fi/dataset/%7B44394B13-85D7-4998-BD06-8ADC77C7455C%7D)
        - Vesiensuojelurakenteet (Sara Vaskio, Raaseporinjoki-hanke)
        - Maanparannustoimenpiteet ja peltoalueet (Sara Vaskio, Raaseporinjoki-hanke)
        - Hallinnolliset aluejaot, 1:10k (MML: https://asiointi.maanmittauslaitos.fi/karttapaikka/tiedostopalvelu/hallinnolliset_aluejaot_vektori)
        - Maastokartta, 1:100k (MML: https://asiointi.maanmittauslaitos.fi/karttapaikka/tiedostopalvelu/maastokartta_vektori)
        - Admin 0 - Countries, 1:10m (Natural Earth: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/)

Requirements:
    - os
    - matplotlib
    - geopandas
    - pandas
    - numpy
    - shapely

References:    
    - Matplotlib 3.8.4 documentation: https://matplotlib.org/stable/
    - Matplotlib: List of named colors: https://matplotlib.org/stable/gallery/color/named_colors.html
    - Matplotlib: Colormap reference: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    - Matplotlib: Hatch style reference: https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html
    - OS: https://docs.python.org/3/library/os.html
    - Geopandas: https://geopandas.org/en/stable/
    - Pandas: https://pandas.pydata.org/
    - Numpy: https://numpy.org/
    - Shapely: https://shapely.readthedocs.io/en/stable/manual.html

Created on Mon Apr 29 15:04:38 2024

@author: Ville Lindgren
"""

############################################################################################################################################################

# MAHDOLLISESTI KIINNOSTAVIA SUUREITA:
    
    # VESIMUODOSTUMAT:
    #     - virtaama_m3s                  Uoman virtaaman tai järven lähtövirtaaman keskiarvo Vemalassa (m3/s).
    #     - p_tulevayht_kgv_v1_nyky       Vesimuodostumaan tuleva keskimääräinen kokonaisfosforin kuormitus (kg/v).
    #     - p_lahtevayht_kgv_v1_nyky      Vesimuodostumasta lähtevä keskimääräinen kokonaisfosforin kuormitus (kg/v).
    #     - p_pitoisuus_ugl_v1_nyky       Simuloitu keskimääräinen kokonaisfosforin pitoisuus (ug/l).
    #     - p_pitoisuus_kesa_ugl_v1_nyky  Simuloitu keskimääräinen kokonaisfosforin pitoisuus kesäkuukausina (kesä-syyskuu) (ug/l).
    #     - n_tulevayht_kgv_v1_nyky       Vesimuodostumaan tuleva keskimääräinen kokonaistypen kuormitus (kg/v).
    #     - n_lahtevayht_kgv_v1_nyky      Vesimuodostumasta lähtevä keskimääräinen kokonaistypen kuormitus (kg/v).
    #     - n_pitoisuus_ugl_v1_nyky       Simuloitu keskimääräinen kokonaistypen pitoisuus (ug/l).
    #     - n_pitoisuus_kesa_ugl_v1_nyky  Simuloitu keskimääräinen kokonaistypen pitoisuus kesäkuukausina (kesä-syyskuu) (ug/l).
        
    # MAA-ALUEET:
    #     - p_yht_kgv_alue_v1_nyky        Maa-alueelta syntyvä kokonaisfosforin kuormitus yhteensä (kg/v). Sisältää luonnonhuuhtouman, peltoviljelyn, metsätalouden, vakituisen haja-asutuksen, loma-asunnot, huleveden, laskeuman ja pistekuormituksen.
    #     - p_yht_kgkm2v_alue_v1_nyky     Maa-alueelta syntyvä kokonaisfosforin kuormitus yhteensä (kg/km2/v). Sisältää luonnonhuuhtouman, peltoviljelyn, metsätalouden, vakituisen haja-asutuksen, loma-asunnot, huleveden, laskeuman ja pistekuormituksen.
    #     - n_yht_kgv_alue_v1_nyky        Maa-alueelta syntyvä kokonaistypen kuormitus yhteensä (kg/v). Sisältää luonnonhuuhtouman, peltoviljelyn, metsätalouden, vakituisen haja-asutuksen, loma-asunnot, huleveden, laskeuman ja pistekuormituksen.
    #     - n_yht_kgkm2v_alue_v1_nyky     Maa-alueelta syntyvä kokonaistypen kuormitus yhteensä (1000 kg/km2/v*). Sisältää luonnonhuuhtouman, peltoviljelyn, metsätalouden, vakituisen haja-asutuksen, loma-asunnot, huleveden, laskeuman ja pistekuormituksen.

############################################################################################################################################################

# IMPORT PACKAGES
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd
import pandas as pd
import numpy as np

############################################################################################################################################################

# INPUT DIRECTORIES
gis_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\Ohjaus-dippa\aineistoja" #tähän polku kansioon mistä alla olevat vemala-datat löytyy

# INPUT FILEPATHS: Vemala
fp_maa = os.path.join(gis_dir, "Raasepori_maa_alueet.gpkg")
fp_vesi = os.path.join(gis_dir, "Raasepori_vesimuodostumat.gpkg")
fp_valuma = os.path.join(gis_dir, "Raasepori_valuma_alueet.gpkg")

# IMPORT INPUT DATA
raasepori_maa = gpd.read_file(fp_maa)
raasepori_vesi = gpd.read_file(fp_vesi)
raasepori_valuma = gpd.read_file(fp_valuma)

############################################################################################################################################################

# OUTPUTS
# Tässä määritellään paikka mihin tulokset tallennetaan tulosten, jos ylipäätään halutaan tallentaa mitään

save_figs = False #Boolean condition (True/False): kertoo tallennetaanko tuloksia vai ei

if save_figs:
    out_dir = os.path.join(gis_dir, "Outputs") #Määritetään tulosten tallennuskansio
    if not os.path.exists(out_dir): #Tarkistaa löytyykö halutun nimistä kansiota määritellystä tiedostopolusta ja luo sellaisen jos ei löytynyt
        os.makedirs(out_dir)

############################################################################################################################################################

##### PLOT 1
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_maa.plot(ax = ax, column="p_pelto_kgkm2v_alue_v1_nyky", legend=True)
fig.suptitle("Maa-alueelta syntyvä kokonaisfosforin kuormitus peltoviljelystä (kg/km2/v)", fontsize=12)
if save_figs: #Jos skriptin alussa on määritetty, että save_figs=True, niin alla olevat tallennusrivit ajetaan.
    plt.savefig(os.path.join(out_dir, "test.png")) #Tallentaa yllä plotatun kuvan .png-kuvatiedostona
    plt.savefig(os.path.join(out_dir, "test.pdf"), format="pdf", bbox_inches="tight") #Tallentaa yllä plotatun kuvan .pdf-tiedostona

############################################################################################################################################################

##### PLOT 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_maa.plot(ax = ax, color="green")
raasepori_vesi.plot(ax = ax, color="blue")
raasepori_valuma.plot(ax = ax, facecolor="none", edgecolor="red")

############################################################################################################################################################

##### PLOT 3
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10), constrained_layout = True)
# subplot 1
raasepori_valuma.plot(ax = ax1)
ax1.title.set_text("valuma-alueet")
# subplot 2
raasepori_maa.plot(ax = ax2)
ax2.title.set_text("maa-alueet")
# subplot 3
raasepori_vesi.plot(ax = ax3)
ax3.title.set_text("vesimuodostumat")
# subplot 4
raasepori_maa.plot(ax = ax4, color="green")
raasepori_vesi.plot(ax = ax4, color="blue")
raasepori_valuma.plot(ax = ax4, facecolor="none", edgecolor="red")

############################################################################################################################################################

##### PLOT 4
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_vesi.plot(ax = ax, column="p_pelto_kgv_v1_nyky", legend=True)
fig.suptitle("Vesimuodostumaan tuleva kokonaisfosforin kuormitus peltoviljelystä (kg/v)", fontsize=12)

############################################################################################################################################################

#TODO: Selvitä miksi pitoisuuksissa on näin iso ero!!!

# Vuotuisella virtaamalla suhteutettu kuormitus vesistöön (kuormitus vesistöön / virtaama)
# Yksikkömuunnos -> mikrogramma/litra 
raasepori_vesi["testi_p_pelto"] = (raasepori_vesi["p_pelto_kgv_v1_nyky"]*1000000) / (raasepori_vesi["virtaama_m3s"]*60*60*24*365*1000)

##### PLOT 5
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_vesi.plot(ax = ax, column="testi_p_pelto", legend=True)
fig.suptitle(u"Pitoisuus: Vesimuodostumaan tuleva virtaamalla suhteutettu kokonaisfosforin kuormitus peltoviljelystä (\u03bcg/l)", fontsize=12)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_vesi.plot(ax = ax, column="p_pitoisuus_ugl_v1_nyky", legend=True)

raasepori_vesi["testi_p_kok"] = (raasepori_vesi["p_tulevayht_kgv_v1_nyky"]*1000000) / (raasepori_vesi["virtaama_m3s"]*60*60*24*365*1000)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_vesi.plot(ax = ax, column="testi_p_kok", legend=True)

############################################################################################################################################################

# Verrataan kahta eri tapaa pitoisuuden laskemiseen: virtaamalla suhteutettu ja VEMALAn simuloima
raasepori_vesi["testi_p_tulevayht"] = (raasepori_vesi["p_tulevayht_kgv_v1_nyky"]*1000000) / (raasepori_vesi["virtaama_m3s"]*60*60*24*365*1000)
raasepori_vesi["testi_p_pitoisuus_sim"] = raasepori_vesi["p_pitoisuus_ugl_v1_nyky"]

##### PLOT 6
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10), constrained_layout = True)
raasepori_vesi.plot(ax = ax1, column="testi_p_pelto", legend=True)
ax1.title.set_text(u"Pitoisuus: Vesimuodostumaan tuleva virtaamalla suhteutettu kokonaisfosforin kuormitus (\u03bcg/l)")
raasepori_vesi.plot(ax = ax2, column="p_pitoisuus_ugl_v1_nyky", legend=True)
ax2.title.set_text(u"VEMALAn nimuloitu keskimääräinen kokonaisfosforin pitoisuus (\u03bcg/l)")

############################################################################################################################################################

# Pitoisuus-statistiikkoja
kokp_vesi_pelto = np.array(raasepori_vesi["testi_p_pelto"])
kokp_vesi_pelto_stats = pd.DataFrame({"count" : [len(kokp_vesi_pelto)]})
kokp_vesi_pelto_stats["min"] = np.min(kokp_vesi_pelto)
kokp_vesi_pelto_stats["max"] = np.max(kokp_vesi_pelto)
kokp_vesi_pelto_stats["mean"] = np.mean(kokp_vesi_pelto)
kokp_vesi_pelto_stats["median"] = np.median(kokp_vesi_pelto)
kokp_vesi_pelto_stats["std"] = np.std(kokp_vesi_pelto)
kokp_vesi_pelto_stats["cv"] = np.var(kokp_vesi_pelto) / np.mean(kokp_vesi_pelto)
print(kokp_vesi_pelto_stats)

##### PLOT 7
plt.figure()
plt.plot(kokp_vesi_pelto)

############################################################################################################################################################

# Itse piirretty joen pätkä
joki_81_073U0033_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\Ohjaus-dippa\aineistoja" #tähän polku kansioon mistä alla olevat itse katkaistu jokiosuus löytyy
joki_81_073U0033_fp = os.path.join(joki_81_073U0033_dir, "81_073U0033.shp")
joki_81_073U0033 = gpd.read_file(joki_81_073U0033_fp)

# Asetetaan oikea koordinaattijärjestelmä
joki_81_073U0033.crs = "EPSG:3067"

##### PLOT 8
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_vesi.plot(ax = ax, color="lightblue")
fig.suptitle("Kaksitasouoma", fontsize=12)
# Kaisa
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax, color="red")
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax, color="red")
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax, color="red")
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax, color="red")
# Ramboll
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax, color="orange")
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax, color="orange")
joki_81_073U0033.plot(ax = ax, color="yellow")

############################################################################################################################################################

# Vesiensuojelutoimet
toimenpiteet_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\Ohjaus-dippa\aineistoja\vesiensuojelurakenteet_shape" #tähän polku kansioon mistä alla olevat Vesiensuojelutoimet-datat löytyy
toimenpiteet_fp = os.path.join(toimenpiteet_dir, "vesiensuojelurakenteet.shp")
toimenpiteet = gpd.read_file(toimenpiteet_fp)

# Asetetaan oikea koordinaattijärjestelmä
toimenpiteet.crs = "EPSG:3067"

# Jaotellaan toimenpiteet uusiin muuttujiin
# pohjapadot ja -kynnykset:
pohjapadot = toimenpiteet[toimenpiteet["Tyyppi"] == "Pohjapato"]
pohjapadot = pohjapadot.append(toimenpiteet[toimenpiteet["Tyyppi"] == "Pohjapato/kynnys"])
# putkipato
putkipato = toimenpiteet[toimenpiteet["Tyyppi"] == "Putkipato"]
# kosteikot ja laskeutusaltaat
kosteikot = toimenpiteet[toimenpiteet["Tyyppi"] == "Kosteikko"]
kosteikot = kosteikot.append(toimenpiteet[toimenpiteet["Tyyppi"] == "Kosteikkoalue"])
laskeutusaltaat = toimenpiteet[toimenpiteet["Tyyppi"] == "Laskeutusallas"]
# eroosiosuoja
eroosiosuojat = toimenpiteet[toimenpiteet["Tyyppi"] == "Eroosiosuoja"]
# virtaamanhallinta
virtaamanhallinta = toimenpiteet[toimenpiteet["Tyyppi"] == "Virtaamanhallinta"]
# vesiensuojelurakenne
vesiensuojelurakenne = toimenpiteet[toimenpiteet["Tyyppi"] == "Vesiensuojelurakenne"]

##### PLOT 9
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_vesi.plot(ax = ax, color="lightblue")
# kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
joki_81_073U0033.plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
# Seuraava rivi plotataan vain, jotta kaksitasouomat saadaan legendiin
toimenpiteet.plot(ax = ax, color="red", label="kaksitasouoma")
kosteikot.plot(ax = ax, color="orange", label="kosteikko")
# toimenpiteet[toimenpiteet["Tyyppi"] == "Kosteikko"].plot(ax = ax, color="orange")
laskeutusaltaat.plot(ax = ax, color="yellow", label="laskeutusallas")
pohjapadot.plot(ax = ax, color="green", label="pohjapato")
eroosiosuojat.plot(ax = ax, color="purple", label="eroosiosuoja")
virtaamanhallinta.plot(ax = ax, color="cyan", label="virtaamanhallinta")
vesiensuojelurakenne.plot(ax = ax, color="magenta", label="vesiensuojelurakenne")
putkipato.plot(ax = ax, color="blue", label="putkipato")
ax.legend(loc="lower right")
fig.suptitle("Vesiensuojelutoimet", fontsize=12)

############################################################################################################################################################

# Maanpeite ja Valuma-alue
corine2018_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\Ohjaus-dippa\aineistoja\corine2018" #tähän polku kansioon mistä alla olevat Corine maanpeite-datat löytyy
corine2018_fp = os.path.join(corine2018_dir, "clc2018eu25ha.shp")
corine2018 = gpd.read_file(corine2018_fp)
raasepori_jako3_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\Ohjaus-dippa\aineistoja" #tähän polku kansioon mistä alla oleva Raaseporinjoen valuma-alue (3.jakovaihe) löytyy
raasepori_jako3_fp = os.path.join(raasepori_jako3_dir, "raasepori_jako3.shp")
raasepori_jako3 = gpd.read_file(raasepori_jako3_fp)

# Clipataan corine maanpeite-datasta raasepori valuma-alue
corine2018_clip = gpd.clip(corine2018, raasepori_jako3)

# Kategoriat (corine2018_clip["Level3Suo"]):
    # Harvapuustoiset alueet
    # Havumetsät
    # Järvet
    # Lehtimetsät
    # Pellot
    # Pienipiirteinen maatalousmosaiikki
    # Sekametsät
    # Teollisuuden ja palveluiden alueet
    # Väljästi rakennetut asuinalueet

syke_jako_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot" #tähän polku kansioon mistä alla olevat Syken päävaluma-aluejako ja 3.jakovaiheen alueet löytyy
syke_jako_fp = os.path.join(syke_jako_dir, "syke-valuma", "Paajako.shp")
syke_jako = gpd.read_file(syke_jako_fp)
syke_jako3_fp = os.path.join(syke_jako_dir, "syke-valuma", "Jako3.shp")
syke_jako3 = gpd.read_file(syke_jako3_fp)

##### PLOT 10
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
# Tausta
ax.set_xlim(308500, 322000)
ax.set_ylim(6651000, 6666000)
syke_jako.plot(ax = ax, color="gainsboro", edgecolor='gainsboro')
syke_jako3.plot(ax = ax, facecolor="none", edgecolor="black")
# Maanpeite
corine2018_clip.plot(ax = ax, color="blue")
corine2018_clip[corine2018_clip["Level3Suo"] == "Järvet"].plot(ax = ax, color="aqua")
corine2018_clip[corine2018_clip["Level3Suo"] == "Väljästi rakennetut asuinalueet"].plot(ax = ax, color="red")
corine2018_clip[corine2018_clip["Level3Suo"] == "Teollisuuden ja palveluiden alueet"].plot(ax = ax, color="darkviolet")
corine2018_clip[corine2018_clip["Level3Suo"] == "Pellot"].plot(ax = ax, color="yellow")
corine2018_clip[corine2018_clip["Level3Suo"] == "Pienipiirteinen maatalousmosaiikki"].plot(ax = ax, color="gold")
corine2018_clip[corine2018_clip["Level3Suo"] == "Harvapuustoiset alueet"].plot(ax = ax, color="yellowgreen")
corine2018_clip[corine2018_clip["Level3Suo"] == "Havumetsät"].plot(ax = ax, color="green")
corine2018_clip[corine2018_clip["Level3Suo"] == "Lehtimetsät"].plot(ax = ax, color="lightgreen")
corine2018_clip[corine2018_clip["Level3Suo"] == "Sekametsät"].plot(ax = ax, color="limegreen")
# Legenda
patch_aqua = mpatches.Patch(color='aqua', label='Järvet')
patch_red = mpatches.Patch(color='red', label='Väljästi rakennetut asuinalueet')
patch_darkviolet = mpatches.Patch(color='darkviolet', label='Teollisuuden ja palveluiden alueet')
patch_yellow = mpatches.Patch(color='yellow', label='Pellot')
patch_gold = mpatches.Patch(color='gold', label='Pienipiirteinen maatalousmosaiikki')
patch_yellowgreen = mpatches.Patch(color='yellowgreen', label='Harvapuustoiset alueet')
patch_green = mpatches.Patch(color='green', label='Havumetsät')
patch_lightgreen = mpatches.Patch(color='lightgreen', label='Lehtimetsät')
patch_limegreen = mpatches.Patch(color='limegreen', label='Sekametsät')
ax.legend(loc="lower right", handles=[patch_aqua, patch_red, patch_darkviolet, patch_yellow, patch_gold, patch_yellowgreen, patch_green, patch_lightgreen, patch_limegreen])
fig.suptitle("Corine maanpeite 2018 (Level 3)", fontsize=12)
# Lisätään plotattavia elementtejä yllä olevaan karttaan
raasepori_vesi.plot(ax = ax, facecolor="aqua", edgecolor="aqua", linewidth=1) #vesimuodostumat
raasepori_valuma.plot(ax = ax, facecolor="none", edgecolor="black", linewidth=0.5) #vemala valuma-alueet
syke_jako3.plot(ax = ax, facecolor="none", edgecolor="black") #syke 3.jakovaiheen valuma-alueet 2010
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
joki_81_073U0033.plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
toimenpiteet.plot(ax = ax, facecolor="magenta", edgecolor="black") #vesiensuojelutoimet

##### PLOT 10.2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
# Tausta
ax.set_xlim(308500, 322000)
ax.set_ylim(6651000, 6666000)
syke_jako.plot(ax = ax, color="gainsboro", edgecolor='gainsboro')
syke_jako3.plot(ax = ax, facecolor="none", edgecolor="black")
# Maanpeite
corine2018_clip.plot(ax = ax, color="blue")
corine2018_clip[corine2018_clip["Level2Suo"] == "Sisävedet"].plot(ax = ax, color="aqua")
corine2018_clip[corine2018_clip["Level2Suo"] == "Asuinalueet"].plot(ax = ax, color="red")
corine2018_clip[corine2018_clip["Level2Suo"] == "Teollisuuden, palveluiden ja liikenteen alueet"].plot(ax = ax, color="darkviolet")
corine2018_clip[corine2018_clip["Level2Suo"] == "Peltomaat"].plot(ax = ax, color="yellow")
corine2018_clip[corine2018_clip["Level2Suo"] == "Heterogeeniset maatalousvaltaiset alueet"].plot(ax = ax, color="gold")
corine2018_clip[corine2018_clip["Level2Suo"] == "Harvapuustoiset metsät ja pensastot"].plot(ax = ax, color="yellowgreen")
corine2018_clip[corine2018_clip["Level2Suo"] == "Sulkeutuneet metsät"].plot(ax = ax, color="green")
# Legenda
patch_aqua = mpatches.Patch(color='aqua', label='Sisävedet')
patch_red = mpatches.Patch(color='red', label='Asuinalueet')
patch_darkviolet = mpatches.Patch(color='darkviolet', label='Teollisuuden, palveluiden ja liikenteen alueet')
patch_yellow = mpatches.Patch(color='yellow', label='Peltomaat')
patch_gold = mpatches.Patch(color='gold', label='Heterogeeniset maatalousvaltaiset alueet')
patch_yellowgreen = mpatches.Patch(color='yellowgreen', label='Harvapuustoiset metsät ja pensastot')
patch_green = mpatches.Patch(color='green', label='Sulkeutuneet metsät')
ax.legend(loc="lower right", handles=[patch_aqua, patch_red, patch_darkviolet, patch_yellow, patch_gold, patch_yellowgreen, patch_green])
fig.suptitle("Corine maanpeite 2018 (Level 2)", fontsize=12)
# Lisätään plotattavia elementtejä yllä olevaan karttaan
raasepori_vesi.plot(ax = ax, facecolor="aqua", edgecolor="aqua", linewidth=1) #vesimuodostumat
raasepori_valuma.plot(ax = ax, facecolor="none", edgecolor="black", linewidth=0.5) #vemala valuma-alueet
syke_jako3.plot(ax = ax, facecolor="none", edgecolor="black") #syke 3.jakovaiheen valuma-alueet 2010
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
joki_81_073U0033.plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
toimenpiteet.plot(ax = ax, facecolor="magenta", edgecolor="black") #vesiensuojelutoimet

##### PLOT 10.3
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
# Tausta
ax.set_xlim(308500, 322000)
ax.set_ylim(6651000, 6666000)
syke_jako.plot(ax = ax, color="gainsboro", edgecolor='gainsboro')
syke_jako3.plot(ax = ax, facecolor="none", edgecolor="black")
# Maanpeite
corine2018_clip.plot(ax = ax, color="blue")
corine2018_clip[corine2018_clip["Level1Suo"] == "Vesialueet"].plot(ax = ax, color="aqua")
corine2018_clip[corine2018_clip["Level1Suo"] == "Rakennetut alueet"].plot(ax = ax, color="red")
corine2018_clip[corine2018_clip["Level1Suo"] == "Maatalousalueet"].plot(ax = ax, color="yellow")
corine2018_clip[corine2018_clip["Level1Suo"] == "Metsät sekä avoimet kankaat ja kalliomaat"].plot(ax = ax, color="green")
# Legenda
patch_aqua = mpatches.Patch(color='aqua', label='Vesialueet')
patch_red = mpatches.Patch(color='red', label='Rakennetut alueet')
patch_yellow = mpatches.Patch(color='yellow', label='Maatalousalueet')
patch_green = mpatches.Patch(color='green', label='Metsät sekä avoimet kankaat ja kalliomaat')
ax.legend(loc="lower right", handles=[patch_aqua, patch_red, patch_yellow, patch_green])
fig.suptitle("Corine maanpeite 2018 (Level 1)", fontsize=12)
# Lisätään plotattavia elementtejä yllä olevaan karttaan
raasepori_vesi.plot(ax = ax, facecolor="aqua", edgecolor="aqua", linewidth=1) #vesimuodostumat
raasepori_valuma.plot(ax = ax, facecolor="none", edgecolor="black", linewidth=0.5) #vemala valuma-alueet
syke_jako3.plot(ax = ax, facecolor="none", edgecolor="black") #syke 3.jakovaiheen valuma-alueet 2010
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
joki_81_073U0033.plot(ax = ax, facecolor="none", edgecolor="magenta", linewidth=2) #kaksitasouoma
toimenpiteet.plot(ax = ax, facecolor="magenta", edgecolor="black") #vesiensuojelutoimet

############################################################################################################################################################

##### PLOT 11
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(10,10), constrained_layout = True)
# subplot 1
raasepori_maa.plot(ax = ax1, column="p_pelto_kgkm2v_alue_v1_nyky", legend=True)
# subplot 2
raasepori_maa.plot(ax = ax2, column="p_pelto_kgkm2v_alue_v1_nyky", legend=True, cmap="Paired")
# subplot 3
raasepori_maa.plot(ax = ax3, column="p_pelto_kgkm2v_alue_v1_nyky", legend=True, cmap="YlOrBr")
# subplot 4
raasepori_maa.plot(ax = ax4, column="p_pelto_kgkm2v_alue_v1_nyky", legend=True, cmap="RdYlBu")
# subplot 5
raasepori_maa.plot(ax = ax5, column="p_pelto_kgkm2v_alue_v1_nyky", legend=True, cmap="RdYlBu_r")
# subplot 6
raasepori_maa.plot(ax = ax6, column="p_pelto_kgkm2v_alue_v1_nyky", legend=True, cmap="RdYlGn_r")
# titles
ax1.title.set_text("viridis (default)")
ax2.title.set_text("Paired")
ax3.title.set_text("YlOrBr")
ax4.title.set_text("RdYlBu")
ax5.title.set_text("RdYlBu_r (reversed)")
ax6.title.set_text("RdYlGn_r (reversed)")
fig.suptitle("Maa-alueelta syntyvä kokonaisfosforin kuormitus peltoviljelystä (kg/km2/v)", fontsize=12)

############################################################################################################################################################

raasepori_maa["test"] = raasepori_maa["p_yht_kgv_alue_v1_nyky"] / raasepori_valuma["ala_km2"] #pinta-alalla suhteutettu fosforikuorma

##### PLOT 12
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10), constrained_layout = True)
# raasepori_maa.plot(ax = ax1, column="p_yht_kgv_alue_v1_nyky", legend=True)
# raasepori_vesi.plot(ax = ax2, column="p_tulevayht_kgv_v1_nyky", legend=True)
raasepori_maa.plot(ax = ax1, column="p_yht_kgv_alue_v1_nyky", legend=True, cmap="RdYlBu_r")
# raasepori_maa.plot(ax = ax1, column="test", legend=True, cmap="RdYlBu_r")
raasepori_vesi.plot(ax = ax2, column="p_tulevayht_kgv_v1_nyky", legend=True, cmap="RdYlBu_r")
ax1.title.set_text("Maa-alueelta")
ax2.title.set_text("Vesimuodostumaan")
fig.suptitle("Kokonaisfosforin kuormitus (kg/v)", fontsize=12)
# Lisätään maanpeitetieto
corine2018_clip[corine2018_clip["Level3Suo"] == "Väljästi rakennetut asuinalueet"].plot(ax = ax1, facecolor="none", edgecolor="black", linewidth=0.5, hatch="|||")
corine2018_clip[corine2018_clip["Level3Suo"] == "Teollisuuden ja palveluiden alueet"].plot(ax = ax1, facecolor="none", edgecolor="black", linewidth=0.5, hatch="---")
corine2018_clip[corine2018_clip["Level3Suo"] == "Pellot"].plot(ax = ax1, facecolor="none", edgecolor="black", linewidth=0.5, hatch="o")
corine2018_clip[corine2018_clip["Level3Suo"] == "Pienipiirteinen maatalousmosaiikki"].plot(ax = ax1, facecolor="none", edgecolor="black", linewidth=0.5, hatch="..")

############################################################################################################################################################

# Taustakarttoja varten:
maat_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot" #tähän polku kansioon mistä alla olevat datat maiden rajoista löytyy
maat_suomi = gpd.read_file(os.path.join(maat_dir, "mml-hallintorajat_10k", "2021", "SuomenValtakunta_2021_10k.shp"))
maat_suomi = maat_suomi.to_crs(raasepori_valuma.crs)
# http://www.naturalearthdata.com/downloads/10m-cultural-vectors/
countries = gpd.read_file(os.path.join(maat_dir, "ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp"))
countries = countries.to_crs(raasepori_valuma.crs)
maat_finland = countries[countries["SOVEREIGNT"]=="Finland"]
maat_sweden = countries[countries["SOVEREIGNT"]=="Sweden"]
maat_norway = countries[countries["SOVEREIGNT"]=="Norway"]
maat_estonia = countries[countries["SOVEREIGNT"]=="Estonia"]
maat_russia = countries[countries["SOVEREIGNT"]=="Russia"]
raasepori_index = ["81.073"]
raasepori_alue = syke_jako3[syke_jako3["Jako3Tunnu"].isin(raasepori_index)]

##### PLOT 13
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10)) #, gridspec_kw={'height_ratios': [1], "width_ratios": [1,2]}
# AX1
ax1.set_xlim(0, 800000)
ax1.set_ylim(6500000, 7850000)
maat_sweden.plot(ax = ax1, color="gainsboro")
maat_norway.plot(ax = ax1, color="gainsboro")
maat_estonia.plot(ax = ax1, color="gainsboro")
maat_russia.plot(ax = ax1, color="gainsboro")
syke_jako.plot(ax = ax1, color="darkgray", ec="black", linewidth=0.25)
maat_suomi.plot(ax = ax1, fc="none", ec="black", linewidth=2)
# maat_finland.plot(ax = ax1, fc="none", ec="black", linewidth=1)
raasepori_valuma.plot(ax = ax1, facecolor="red", edgecolor="red", linewidth=0.5) #vemala valuma-alueet
# pohjoisnuoli
x, y, arrow_length = 0.1, 0.95, 0.1
ax1.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black'),
            ha='center', va='center',
            xycoords=ax1.transAxes)
# mittakaava
scalebar1 = ScaleBar(1, "m", length_fraction=0.4, location="lower right", scale_loc="top")
ax1.add_artist(scalebar1)
# AX2
ax2.set_xlim(308500, 322000)
ax2.set_ylim(6651000, 6666000)
syke_jako.plot(ax = ax2, fc="gainsboro", ec="black", linewidth=0.5)
raasepori_valuma.plot(ax = ax2, fc="darkgray", ec="black", linewidth=0.5)
raasepori_vesi.plot(ax = ax2, facecolor="aqua", edgecolor="aqua", linewidth=1) #vesimuodostumat
raasepori_valuma.plot(ax = ax2, fc="none", ec="black", linewidth=0.5)
raasepori_alue.plot(ax = ax2, fc="none", ec="black", linewidth=2)
# pohjoisnuoli
x, y, arrow_length = 0.9, 0.90, 0.1 #width=5, headwidth=15)
ax2.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black'),
            ha='center', va='center',
            xycoords=ax2.transAxes)
# mittakaava
scalebar2 = ScaleBar(1, "m", length_fraction=0.4, location="lower right", scale_loc="top")
ax2.add_artist(scalebar2)
# fig.tight_layout()

############################################################################################################################################################

# Tehdään kopio vesimuodostumat sisältävästä muuttujasta ja manipuloidaan sen geometrisiä muotoja vastaamaan osavaluma-alueiden muotoja.
# Tämä tehdään, jotta kartasta saadaan selkeämpi.
# Vertaa seuraavan kuvan vasemman puoleista karttaa oikean puoleiseen karttaan.
raasepori_vesi_test = raasepori_vesi.copy()
raasepori_vesi_test["geometry"] = raasepori_valuma["geometry"]

##### PLOT 14
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10), constrained_layout = True)
fig.suptitle("Vesimuodostumaan tuleva kokonaisfosforin kuormitus peltoviljelystä (kg/v)", fontsize=12)
# AX1
ax1.title.set_text("geometry = vesimuodostumat")
# raasepori_vesi.plot(ax = ax1, column="p_pelto_kgv_v1_nyky", legend=True, cmap="RdYlGn_r")
raasepori_vesi.plot(ax = ax1, column="p_pelto_kgv_v1_nyky", legend=True, cmap="YlOrBr")
raasepori_valuma.plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=0.5)
# AX2
ax2.title.set_text("geometry = valuma-alueet")
# raasepori_vesi_test.plot(ax = ax2, column="p_pelto_kgv_v1_nyky", legend=True, cmap="RdYlGn_r")
raasepori_vesi_test.plot(ax = ax2, column="p_pelto_kgv_v1_nyky", legend=True, cmap="YlOrBr")
# raasepori_vesi_test[raasepori_vesi_test["p_pelto_kgv_v1_nyky"] > 0].plot(ax = ax2, column="p_pelto_kgv_v1_nyky", legend=True, cmap="YlOrBr")
# raasepori_vesi_test[raasepori_vesi_test["p_pelto_kgv_v1_nyky"] == 0].plot(ax = ax2, column="p_pelto_kgv_v1_nyky", legend=True, color="white") #Testi: jos kuormitus on 0, vaihdetaan väriksi valkoinen
raasepori_vesi.plot(ax = ax2, color="aqua", facecolor="none", edgecolor="aqua", linewidth=1)
raasepori_valuma.plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=0.5)

# Lisätään pellot
corine2018_clip[corine2018_clip["Level3Suo"] == "Pellot"].plot(ax = ax2, facecolor="none", edgecolor="black", linewidth=0.5, hatch="/")
corine2018_clip[corine2018_clip["Level3Suo"] == "Pienipiirteinen maatalousmosaiikki"].plot(ax = ax2, facecolor="none", edgecolor="black", linewidth=0.5, hatch=".")

############################################################################################################################################################

##### PLOT 15
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10), constrained_layout = True)
# AX1
ax1.title.set_text("Peltoviljelystä vesimuodostumaan tuleva kokonaisfosforin kuormitus (kg/v)")
raasepori_vesi_test.plot(ax = ax1, column="p_pelto_kgv_v1_nyky", legend=True, cmap="YlOrBr")
raasepori_vesi.plot(ax = ax1, color="aqua", facecolor="none", edgecolor="aqua", linewidth=1)
raasepori_valuma.plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=0.5)
corine2018_clip[corine2018_clip["Level3Suo"] == "Pellot"].plot(ax = ax1, facecolor="none", edgecolor="black", linewidth=0.5, hatch="/")
corine2018_clip[corine2018_clip["Level3Suo"] == "Pienipiirteinen maatalousmosaiikki"].plot(ax = ax1, facecolor="none", edgecolor="black", linewidth=0.5, hatch=".")
# AX2
ax2.title.set_text(u"Peltoviljelystä vesimuodostumaan tuleva kokonaisfosforin pitoisuus (\u03bcg/l)")
raasepori_vesi_test.plot(ax = ax2, column="testi_p_pelto", legend=True, cmap="YlOrBr")
raasepori_vesi.plot(ax = ax2, color="aqua", facecolor="none", edgecolor="aqua", linewidth=1)
raasepori_valuma.plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=0.5)
corine2018_clip[corine2018_clip["Level3Suo"] == "Pellot"].plot(ax = ax2, facecolor="none", edgecolor="black", linewidth=0.5, hatch="/")
corine2018_clip[corine2018_clip["Level3Suo"] == "Pienipiirteinen maatalousmosaiikki"].plot(ax = ax2, facecolor="none", edgecolor="black", linewidth=0.5, hatch=".")

# Lisätään toimenpiteet
# AX1
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
joki_81_073U0033.plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
toimenpiteet.plot(ax = ax1, facecolor="red", edgecolor="black", label="kaksitasouoma")
# AX2
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
joki_81_073U0033.plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
toimenpiteet.plot(ax = ax2, facecolor="red", edgecolor="black", label="kaksitasouoma")

############################################################################################################################################################

maanparannus_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\Ohjaus-dippa\aineistoja\Maanparannus"
maanparannus_fp = os.path.join(maanparannus_dir, "Maanparannus_Raaseporinjoki_2021.shp")
maanparannus = gpd.read_file(maanparannus_fp)

from shapely.ops import cascaded_union

# Tehdään lista, joka sisältää kaikki 500+ peltoja esittävää polygonia
polygons = []
for i in range(len(maanparannus)):  
    polygons.append(maanparannus.iloc[i]["geometry"])

# Testi plotteja
gpd.GeoSeries(polygons).boundary.plot(color="blue")
plt.show()
gpd.GeoSeries(polygons).boundary.plot(facecolor="blue", edgecolor="blue")
plt.show()

# Luodaan oma funktio, joka yhdistää kaksi polygonia yhdeksi, ja käyttää vain uloimpia ääriviivoja sen muotona
def combineBorders(*geoms):
    return cascaded_union([geom if geom.is_valid else geom.buffer(0) for geom in geoms])

# Käydään yllä luotu polygon-lista läpi ja yhdistetään kaikki sen sisältämät muodot yhdeksi polygoniksi
for i in range(len(polygons)-1):
    if i == 0:
        polygons_merged = combineBorders(polygons[i], polygons[i+1])
    else:
        polygons_merged = combineBorders(polygons_merged, polygons[i+1])

# Testi plotteja
gpd.GeoSeries(polygons_merged).boundary.plot(color="blue")
plt.show()
gpd.GeoSeries(polygons_merged).boundary.plot(facecolor="blue", edgecolor="blue")
plt.show()

# Luodaan uusi geodataframe-muuttuja (kopio raaseporin maa-alueista)
# Älä välitä muista sarakkeista kuin "geometry", kaikki muut viittaa raasepori_maa dataan
raasepori_pellot = raasepori_maa.copy()
raasepori_pellot = raasepori_pellot.drop(raasepori_pellot.index.to_list()[1:], axis=0)

# Testi plotti
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_pellot.plot(ax = ax, color="blue")

# Muutetaan "geometry"-soly vastaamaan yllä luotua peltoja vastaavaa polygon-yhdistelmää (= multipolygonia)
raasepori_pellot["geometry"] = polygons_merged

# Testi plotti
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_pellot.plot(ax = ax, color="blue")

# Clipataan raasepori valuma-alueella ylimääräiset peltoalueet datasta
raasepori_pellot = gpd.clip(raasepori_pellot, raasepori_jako3)

# Testi plotti
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
raasepori_pellot.plot(ax = ax, color="blue")

# Vertailu plotti
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10), constrained_layout = True)
# AX1
raasepori_maa.plot(ax = ax1, color="green")
raasepori_vesi.plot(ax = ax1, color="blue")
raasepori_pellot.plot(ax = ax1, color="yellow")
raasepori_valuma.plot(ax = ax1, facecolor="none", edgecolor="red")
# AX2
raasepori_maa.plot(ax = ax2, color="green")
raasepori_vesi.plot(ax = ax2, color="blue")
raasepori_pellot.plot(ax = ax2, facecolor="none", edgecolor="yellow", linewidth=1, hatch="..")
raasepori_valuma.plot(ax = ax2, facecolor="none", edgecolor="red")

############################################################################################################################################################

#https://asiointi.maanmittauslaitos.fi/karttapaikka/tiedostopalvelu
#Lataa paikkatietoaineistoja -> Avoimet vektoriaineistot -> Maastokartta (vektori) -> 1:100000 (ESRI shapefile)
pienemmat_uomat_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\Ohjaus-dippa\aineistoja\mml-vektori"
pienemmat_uomat_fp = os.path.join(pienemmat_uomat_dir, "K42_VesiViiva.shp")
pienemmat_uomat = gpd.read_file(pienemmat_uomat_fp)

vesialue_fp = os.path.join(pienemmat_uomat_dir, "K42_VesiAlue.shp")
vesialue = gpd.read_file(vesialue_fp)

# Asetetaan oikea koordinaattijärjestelmä
pienemmat_uomat.crs = "EPSG:3067"
vesialue.crs = "EPSG:3067"

# Clipataan pienemmat_uomat raasepori valuma-alueella
pienemmat_uomat_clip = gpd.clip(pienemmat_uomat, raasepori_jako3)
vesialue_clip = gpd.clip(vesialue, raasepori_jako3)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
# AX1
raasepori_maa.plot(ax = ax, color="green")
pienemmat_uomat_clip.plot(ax = ax, color="aqua")
raasepori_vesi.plot(ax = ax, color="blue")
raasepori_pellot.plot(ax = ax, color="yellow")
# raasepori_valuma.plot(ax = ax, facecolor="none", edgecolor="red")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20,10), constrained_layout = True)
# AX1
ax1.title.set_text("Vemalan vesimuodostumat")
raasepori_maa.plot(ax = ax1, color="green")
raasepori_vesi.plot(ax = ax1, color="blue", facecolor="none", edgecolor="blue", linewidth=1)
# raasepori_pellot.plot(ax = ax1, color="yellow")
# raasepori_valuma.plot(ax = ax1, facecolor="none", edgecolor="red")
# AX2
ax2.title.set_text("Uomat (K42_VesiViiva.shp) MML:n maastokartasta")
raasepori_maa.plot(ax = ax2, color="green")
pienemmat_uomat_clip.plot(ax = ax2, color="blue", facecolor="none", edgecolor="blue", linewidth=1)
# raasepori_pellot.plot(ax = ax2, color="yellow")
# raasepori_valuma.plot(ax = ax2, facecolor="none", edgecolor="red")
# AX2
ax3.title.set_text("Molemmat + pellot")
raasepori_maa.plot(ax = ax3, color="green")
raasepori_vesi.plot(ax = ax3, color="blue", facecolor="none", edgecolor="blue", linewidth=1)
pienemmat_uomat_clip.plot(ax = ax3, color="blue", facecolor="none", edgecolor="blue", linewidth=1)
raasepori_pellot.plot(ax = ax3, color="yellow")
# raasepori_valuma.plot(ax = ax3, facecolor="none", edgecolor="red")

############################################################################################################################################################

##### TEST 1
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10), constrained_layout = True)
# AX1
ax1.title.set_text("Peltoviljelystä vesimuodostumaan tuleva kokonaisfosforin kuormitus (kg/v)")
raasepori_vesi_test.plot(ax = ax1, column="p_pelto_kgv_v1_nyky", legend=True, cmap="YlOrBr")
raasepori_valuma.plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=0.5)
corine2018_clip[corine2018_clip["Level3Suo"] == "Pellot"].plot(ax = ax1, facecolor="none", edgecolor="black", linewidth=0.5, hatch="..")
corine2018_clip[corine2018_clip["Level3Suo"] == "Pienipiirteinen maatalousmosaiikki"].plot(ax = ax1, facecolor="none", edgecolor="black", linewidth=0.5, hatch="..")
raasepori_vesi.plot(ax = ax1, facecolor="aqua", edgecolor="aqua", linewidth=1)
pienemmat_uomat_clip.plot(ax = ax1, facecolor="none", edgecolor="aqua", linewidth=1)
# AX2
ax2.title.set_text("Peltoviljelystä vesimuodostumaan tuleva kokonaisfosforin kuormitus (kg/v)")
raasepori_vesi_test.plot(ax = ax2, column="p_pelto_kgv_v1_nyky", legend=True, cmap="YlOrBr")
# raasepori_valuma.plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=0.5)
raasepori_pellot.plot(ax = ax2, facecolor="none", edgecolor="black", linewidth=0.5, hatch="..")
raasepori_vesi.plot(ax = ax2, color="aqua", facecolor="none", edgecolor="aqua", linewidth=1)
pienemmat_uomat_clip.plot(ax = ax2, color="aqua", facecolor="none", edgecolor="aqua", linewidth=1)
# # Lisätään toimenpiteet
# # AX1
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax1, color="red", facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
# joki_81_073U0033.plot(ax = ax1, facecolor="none", edgecolor="red", linewidth=2)
# toimenpiteet.plot(ax = ax1, facecolor="red", edgecolor="black", label="kaksitasouoma")
# # AX2
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
# raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
# joki_81_073U0033.plot(ax = ax2, facecolor="none", edgecolor="red", linewidth=2)
# toimenpiteet.plot(ax = ax2, facecolor="red", edgecolor="black", label="kaksitasouoma")

##### TEST 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), constrained_layout = True)
# Kartta
ax.title.set_text("Peltoviljelystä vesimuodostumaan tuleva kokonaisfosforin kuormitus (kg/v)")
raasepori_vesi_test.plot(ax = ax, column="p_pelto_kgv_v1_nyky", legend=True, cmap="YlOrBr")
raasepori_valuma.plot(ax = ax, facecolor="none", edgecolor="red", linewidth=0.5)
raasepori_pellot.plot(ax = ax, facecolor="none", edgecolor="black", linewidth=0.5, hatch="..")
raasepori_vesi.plot(ax = ax, facecolor="aqua", edgecolor="aqua", linewidth=1)
# vesialue_clip.plot(ax = ax, facecolor="aqua", edgecolor="aqua", linewidth=1)# Lisätään toimenpiteet
pienemmat_uomat_clip.plot(ax = ax, facecolor="none", edgecolor="aqua", linewidth=1)# Lisätään toimenpiteet
# Lisätään toimenpiteet
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0008"].plot(ax = ax, color="red", facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0013"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0014"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0015"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0018"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
raasepori_vesi[raasepori_vesi["vemalatunnus"] == "81.073U0019"].plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
joki_81_073U0033.plot(ax = ax, facecolor="none", edgecolor="red", linewidth=2)
toimenpiteet.plot(ax = ax, facecolor="red", edgecolor="black", label="kaksitasouoma")

#TODO: Miksi markerit menee visuaalisesti uoman alle plotattaessa?
