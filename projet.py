import re
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def initialisation():
    #Parcours du répertoire de fichier base de donnee
    path_to_file = glob.glob("base de donnee/" + '*csv')
    file_list = [i.split("/")[-1] for i in tqdm(path_to_file)]
    print("Base de donnée à travailler")
    [print("Index:", i, "Base de donnée:", str(value).replace("base de donnee", "").replace(".csv", "")) for i, value in enumerate(file_list)]
    # mettre les fichiers csv sous forme de DataFrame
    liste_data = [pd.DataFrame(pd.read_csv(file)) for file in tqdm(file_list)]
    return liste_data,file_list

def checkvaleursmanquantespourcentage(datas, filename):
    html = []
    k = 0
    for i, file in enumerate(datas):
        missing_values = pd.concat([file.isna().sum(), (((file.isna().sum() / file.shape[0]) * 100)).round(3),(file.notna().sum()+file.isna().sum())], keys=['Values missing', 'Frequence','NB ligne'], axis=1)
        html.append(missing_values.to_html(na_rep="", index=True, classes="table table-dark"))
    html_string = '''
    <html>
        <head><title>Données manquantes</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        </head><body >
        {table}
        </body>
    </html>.
    '''
    text_file = open("donneesmanquantes.html", "w")
    for  values in html:
        while k < len(filename):
            text_file.write(html_string.format(table=f'Nom:{str(filename[k]).replace("base de donnee","").replace(chr(92),"")}{(values)}'))
            break
        k += 1

    text_file.close()
    print("Opération réussie")


def Dropcolonne(datas):
    tab = []
    index = int(input("Nombre d'éléments que vous voulez supprimer: "))
    for i in range(index):
        enter = input("Nom de colonne à supprimer:")
        tab.append(enter)
    datas = datas.drop(columns=tab)
    return datas

def Qualitativebinarizer(datas):
    emptylist = []
    number = int(input("Rentrer le nombre pour choisir la base de donnée:"))
    index = int(input("Nombre d'éléments: "))
    for i in range(index):
        enter = input("Donner la colonne 'qualitative' pour la changer en binaire:")
        emptylist.append(enter)
    data = datas[number]
    enc = OneHotEncoder(handle_unknown="ignore",sparse=True)
    transformed = enc.fit_transform(data[emptylist])
    new = pd.DataFrame.sparse.from_spmatrix(transformed,columns=enc.get_feature_names_out())
    data = pd.concat([data, new], axis=1).drop(columns=emptylist)
    print(data)
    return data


def Normalisation(datas):
    lis = []
    index = int(input("Nombre d'éléments qu'on va normaliser : "))
    if index <= 0:
        return datas
    else:
        for i in tqdm(range(index)):
            enter = input(" Nom de colonne à Normaliser:")
            lis.append(enter)
        scaler = MinMaxScaler()
        datas[lis] = scaler.fit_transform(datas[lis])
        return datas

def Imputation(datas):
    print("Imputation")
    col = datas.columns[(datas.isna().sum()*100)/datas.shape[0]>=90]
    datas = datas.drop(columns=col)
    print(datas)
    return datas

def replacecolumnsJours(datas):
    new = "Jour_en_fct_du_début_de_l'essai"
    old = input("Nom de l'Ancienne colonne: ")
    datas.rename(columns={old:new},inplace=True)
    return datas

def replacecolonne(datas):
    essai = int(input("nombre de colonne à changer:"))
    for i in range(essai):
        old = input("Nom de l'ancienne colonne:")
        new = input("nom de nouvelle colonne:")
        datas.rename(columns={old:new},inplace=True)
    print(datas)
    datas.to_csv("watch.csv")

#Problème dans la base de donnée Labs ou Test_Result possède variables quali et quanti
#Problème alsfr=solution pd.getdummies
def changealsfrs(datas):
    df = pd.get_dummies(columns=["Q1_Speech","Q2_Salivation","Q3_Swallowing","Q4_Handwriting","Q5a_Cutting_without_Gastrostomy","Q5b_Cutting_with_Gastrostomy","Q6_Dressing_and_Hygiene","Q7_Turning_in_Bed","Q8_Walking","Q9_Climbing_Stairs","Q10_Respiratory","R_1_Dyspnea","R_2_Orthopnea","R_3_Respiratory_Insufficiency"],data=datas,prefix_sep="_class")
    df.to_csv("alsfrs.csv")

#Problème ConMeds

def replacevaluesConMeds(datas):
    df = datas["Dose"]
    df.mask(df.str.contains("/") == True,np.nan)
    df.mask(df.str.contains("-") == True, np.nan)
    df = df.str.replace(",",".")
    df = df.replace(["u","200 or 400","nk","UU","1 TO 3","2 or 3","600 to 1200","60 TO 20", "UNK","Not known","24 to 4 mg to 21 pack", "unknown", "Unknown", "unk","UU/UU/uuu", "U", "hr", "PRN", "varies", "VARIES"], np.nan)
    df = df.replace(["HALF"], 0.5)
    df = df.replace(["0.083%","5%","1%","2%","0.5%","0.05%"],[0.083,5,1,2,0.5,0.05])
    df = df.replace(["2.5+0.625","20+12.5", "1750+250", "2625+375","150+12.5","25+6.25","25 + 6.25"], [3.125,32.5, 2000, 3000,165.5,31.25,31.25])
    df = df.replace(["50 mg","25mgs and 75mgs","250(totol dose)","200 mg"], [50,100,250,200])
    df = df.replace(["40 (diluted)"],40)
    df = df.replace(["ONE PATCH", "one", "1 spray","Spray","cream"], 1)
    df = df.replace(["2 sprays"], 2)
    df = df.replace(["12hr"], 12)
    df = df.replace('1000  [5000/dayly]', 1000)
    datas = datas.merge(df.rename("dose"), left_index=True, right_index=True)
    datas = datas.drop(columns="Dose")
    datas["dose"] = pd.to_numeric(datas["dose"],errors='coerce')
    datas.to_csv("ConMeds.csv")


#Probleme resolu 1H30
def changeLabs(datas):
    df = pd.get_dummies(datas,columns=["Test_Name"])
    for i in tqdm(range(4,df.shape[1])):
        values = df.iloc[:,i]
        for j in range(values.shape[0]):
            if df[values.name].loc[j] == 1:
                df[values.name].loc[j] = df["Test_Result"].loc[j]
    df.to_csv("Test.csv")

def replaceLabs(datas):
    df = datas["Test_Result"]
    df = df.str.replace("-","")
    df = df.str.replace(",", ".")
    for value in df:
        if value == np.nan:
            continue
        else:
            print(value)


#Filtrage approfondi
def colonnefilter(datas,file):
    df = datas
    tab = []
    tabhisto = []
    for i in tqdm(range(0,df.shape[1])):
        values = df.iloc[:,i]
        values = values.replace(0,np.nan)
        subcondition = (values.isna().sum() * 100) / values.shape[0]
        condition = subcondition >= 95
        col = [condition]
        if col == [True] and values.name != "Jour_en_fct_du_début_de_l'essai":
            tab.append(values.name)
            tabhisto.append(subcondition)
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Pourcentage de données manquantes sur la Database :{file}")
    ax.bar(tab, height=tabhisto,width=0.2)
    plt.xlabel("Nom de colonne")
    plt.ylabel("Pourcentage")
    ax.set_xticklabels(tab, rotation='vertical', fontsize=5)
    plt.show()


#Alléger la mémoire des databases

def downcast_dtypes(df):
    _start = df.memory_usage(deep=True).sum() / 1024 ** 2
    float_cols = [c for c in df if df[c].dtype in["float32","float64"]]
    int_cols = [c for c in df if df[c].dtype in ["int64","int32","int16"]]
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols] = df[int_cols].astype(np.int8)
    _end = df.memory_usage(deep=True).sum() / 1024 ** 2
    saved = (_start - _end) / _start *100
    print(f"Saved {saved:.2f}%")
    return df


def mergedatas(liste):
    newdataframe = []
    for i in range(0,len(liste)):
        df = downcast_dtypes(liste[i])
        newdataframe.append(df)
    [print(newdataframe[i].info()) for i in range(len(newdataframe))]
    merge = reduce(lambda left,right: pd.merge(left,right,how="outer",on=["subject_id","Jour_en_fct_du_début_de_l'essai"]),newdataframe)
    merge.drop_duplicates(subset=["subject_id","Jour_en_fct_du_début_de_l'essai"],keep='first',inplace=True,ignore_index=True)
    print(merge)
    merge.to_csv("One.csv")


if __name__=='__main__':

    c = int(input("Voulez vous normaliser et encoder (1), filtrer(2) ou merge(n'importe quelle chiffre)\n"))
    if c == 1:
        print("ENCODAGE ET NORMALISATION")
        liste_data,file_list = initialisation()
        df = Qualitativebinarizer(liste_data)
        df = Imputation(df)
        df = Normalisation(df)
        df = Dropcolonne(df)
        df = replacecolumnsJours(df)
        print(df)
        df.to_csv("new.csv",index=False)
    elif c==2:
        print("Filtrage Statistique")
        path_to_file = glob.glob("NouvellebaseJour/" + '*csv')
        file_list = [i.split("/")[-1] for i in tqdm(path_to_file)]
        liste_data = [pd.DataFrame(pd.read_csv(file)) for file in tqdm(file_list)]
        [colonnefilter(liste_data[i],file_list[i]) for i in range(len(liste_data))]

    else :
        path_to_file = glob.glob("NouvellebaseJour/" + '*csv')
        file_list = [i.split("/")[-1] for i in tqdm(path_to_file)]
        liste_data = [pd.DataFrame(data=pd.read_csv(file)) for file in tqdm(file_list)]
        replacecolonne(liste_data[7])
        #mergedatas(liste_data)
