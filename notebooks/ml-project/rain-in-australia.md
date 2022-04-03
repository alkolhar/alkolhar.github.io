# Machine Learning Projekt
<img src="img/ost-logo.png" align="right">
<b>Herbstsemester 2021</b><br>
<a href="mailto:noe.schreiber@ost.ch">Noé Schreiber</a> und <a href="mailto:alex.koller@ost.ch">Alex Koller</a>

## Inhaltsverzeichnis <a name="toc"></a>

- [1. Einleitung](#einleitung)
- [2. Zielsetzung und Vorgehensweise](#ziel)
- [3. Explorative Datenanalyse](#eda)
    - [3.1 Dimensionen des Datensatzes](#eda-dim)
    - [3.2 Vorschau der Daten](#eda-preview)
    - [3.3 Betrachtung der Datentypen](#eda-types)
    - [3.4 Statistische Merkmal der Daten](#eda-stat)
    - [3.5 Univariate Analyse des Targets](#eda-univ)
    - [3.6 Bivariate Analyse](#eda-biv)
        - [3.6.1 Betrachtung der kategorischen Variablen](#eda-bivK)
        - [3.6.2 Betrachtung der numerischen Variablen](#eda-bivN)
    - [3.7 Multivariate Analyse](#eda-mul)
    - [3.8 Data Preparation](#dataprep)
- [4. No Free Lunch](#nfl)
    - [4.1 k-Nearest-Neighbors](#knn)
    - [4.2 Logistische Regression](#logreg)
    - [4.3 Decision Tree](#dectree)
    - [4.4 Random Forests](#randfor)
    - [4.5 Boosting](#boosting)
- [5. Validierung](#val)
- [6. Entscheid](#entscheid)
- [7. Schlussfolgerung und Ausblick](#ausblick)
- [8. Referenzen](#ref)


***
***
## Einleitung <a name="einleitung"></a>

Das Team "Wetterwarte Waldau" beschäftigt sich mit der Wettervorhersage einzelner Ortschaften in Australien. Für Landwirtschaft und Ökosystem sind auf einem solch trockenen Kontinent die Regentage von besonderer Bedeutung. Das Augenmerk wird auf die Vorhersage genannter Regentage gesetzt.

***
***
## Zielsetzung und Vorgehensweise <a name="ziel"></a>

Es soll aus einem Datensatz mit Wetterdaten in Australien ein Learner implementiert werden, mit dem Wettervorhersagen gemacht werden können. Konkret geht es darum, Regentage zu bestimmen. Diese sollen aus den Merkmalen der Vortage, Luftdruck, relative Luftfeuchtigkeit, Windrichtung und -stärke, Temperatur, Verdunstung und Wolkenanteil am Himmel bestimmt werden.

***
***
## Explorative Datenanalyse <a name="eda"></a>

_Die explorative Datenanalyse wird auch als explorative Statistik bezeichnet und bildet ein Teilgebiet der Statistik. Es werden Daten analysiert, zu denen kaum oder sogar keine bekannten Zusammenhänge bestehen._


```python
# Standard imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Daten laden
df = pd.read_csv('./data/weatherAUS.csv', sep = ',');

# Einstellungen für plot
palette1 = ["wheat" for _ in range(3)]
palette1[0] = "gold"
```

***
### Dimensionen des Datensatzes <a name="eda-dim"></a>


```python
df.shape
```




    (145460, 23)



Wir haben 145'460 statistische Einheiten im Datensatz erfasst, welche jeweils über 23 Merkmale verfügen.
***
### Vorschau der Daten <a name="eda-preview"></a>


```python
df.head().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Date</th>
      <td>2008-12-01</td>
      <td>2008-12-02</td>
      <td>2008-12-03</td>
      <td>2008-12-04</td>
      <td>2008-12-05</td>
    </tr>
    <tr>
      <th>Location</th>
      <td>Albury</td>
      <td>Albury</td>
      <td>Albury</td>
      <td>Albury</td>
      <td>Albury</td>
    </tr>
    <tr>
      <th>MinTemp</th>
      <td>13.4</td>
      <td>7.4</td>
      <td>12.9</td>
      <td>9.2</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>MaxTemp</th>
      <td>22.9</td>
      <td>25.1</td>
      <td>25.7</td>
      <td>28.0</td>
      <td>32.3</td>
    </tr>
    <tr>
      <th>Rainfall</th>
      <td>0.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Evaporation</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Sunshine</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>WindGustDir</th>
      <td>W</td>
      <td>WNW</td>
      <td>WSW</td>
      <td>NE</td>
      <td>W</td>
    </tr>
    <tr>
      <th>WindGustSpeed</th>
      <td>44.0</td>
      <td>44.0</td>
      <td>46.0</td>
      <td>24.0</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>WindDir9am</th>
      <td>W</td>
      <td>NNW</td>
      <td>W</td>
      <td>SE</td>
      <td>ENE</td>
    </tr>
    <tr>
      <th>WindDir3pm</th>
      <td>WNW</td>
      <td>WSW</td>
      <td>WSW</td>
      <td>E</td>
      <td>NW</td>
    </tr>
    <tr>
      <th>WindSpeed9am</th>
      <td>20.0</td>
      <td>4.0</td>
      <td>19.0</td>
      <td>11.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>WindSpeed3pm</th>
      <td>24.0</td>
      <td>22.0</td>
      <td>26.0</td>
      <td>9.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Humidity9am</th>
      <td>71.0</td>
      <td>44.0</td>
      <td>38.0</td>
      <td>45.0</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>Humidity3pm</th>
      <td>22.0</td>
      <td>25.0</td>
      <td>30.0</td>
      <td>16.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>Pressure9am</th>
      <td>1007.7</td>
      <td>1010.6</td>
      <td>1007.6</td>
      <td>1017.6</td>
      <td>1010.8</td>
    </tr>
    <tr>
      <th>Pressure3pm</th>
      <td>1007.1</td>
      <td>1007.8</td>
      <td>1008.7</td>
      <td>1012.8</td>
      <td>1006.0</td>
    </tr>
    <tr>
      <th>Cloud9am</th>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Cloud3pm</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Temp9am</th>
      <td>16.9</td>
      <td>17.2</td>
      <td>21.0</td>
      <td>18.1</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>Temp3pm</th>
      <td>21.8</td>
      <td>24.3</td>
      <td>23.2</td>
      <td>26.5</td>
      <td>29.7</td>
    </tr>
    <tr>
      <th>RainToday</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>RainTomorrow</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



Die Merkmale haben folgende Eigenschaften:

| Variablenname | Typ | Einheit | Bemerkung |
| -------- | -------- | -------- | -------- | 
| `Date` |(Datum) | Der Tag, an dem gemessen wurde
| `Location` | (Qualitativ, nominal) | Die Stadt in der gemessen wurde | |
| `MinTemp` | (Quantitativ, stetig) | Die tiefste gemessene Temperatur | [°C] |
| `MaxTemp` | (Quantitativ, stetig) | Die höchste gemessene Temperatur | [°C] |
| `Rainfall` | (Quantitativ, stetig) | Die Niederschlagsmenge in den 24 Stunden vor 9 Uhr | [mm] |
| `Evaporation` | (Quantitativ, stetig) | [Pfannenverdunstung](https://de.wikibrief.org/wiki/Pan_evaporation) in den 24 Stunden vor 9 Uhr | [mm] |
| `Sunshine` | | Strahlender Sonnenschein in den 24 Stunden bis Mitternacht | [h] |
| `WindGustDir` | (Qualitativ, nominal) | Richtung der stärksten Böe in den 24 Stunden bis Mitternacht | [16 Himmelsrichtung] |
| `WindGustSpeed` | (Quantitativ, stetig) | Geschwindigkeit der stärksten Böe in den 24 Stunden bis Mitternacht | [km/h] |
| `WindDir9am` | (Qualitativ, nominal) | Windrichtung gemittelt über 10 Minuten vor 9 Uhr | [16 Himmelsrichtung] |
| `WindDir3pm` | (Qualitativ, nominal) | Windrichtung gemittelt über 10 Minuten vor 15 Uhr | [16 Himmelsrichtung] |
| `WindSpeed9am` | (Quantitativ, stetig) | Windgeschwindigkeit gemittelt über 10 Minuten vor 9 Uhr | [km/h] |
| `WindSpeed3pm` | (Quantitativ, stetig) | Windgeschwindigkeit gemittelt über 10 Minuten vor 15 Uhr | [km/h] |
| `Humidity9am` | (Quantitativ, stetig) | Relative Luftfeuchtigkeit gemittelt über 10 Minuten vor 9 Uhr | [%] |
| `Humidity3pm` | (Quantitativ, stetig) | Relative Luftfeuchtigkeit gemittelt über 10 Minuten vor 15 Uhr | [%] |
| `Pressure9am` | (Quantitativ, stetig) | Atmosphärischer Druck auf mittlere Meereshöhe gemittelt über 10 Minuten vor 9 Uhr | [hpa] |
| `Pressure3pm` | (Quantitativ, stetig) | Atmosphärischer Druck auf mittlere Meereshöhe gemittelt über 10 Minuten vor 15 Uhr | [hpa] |
| `Cloud9am` | (Quantitativ, nominal) | Anteil des Himmels, der um 9 Uhr von Wolken verdeckt ist | [Achtel] |
| `Cloud3pm` | (Quantitativ, nominal) | Anteil des Himmels, der um 15 Uhr von Wolken verdeckt ist | [Achtel] |
| `Temp9am` | (Quantitativ, stetig) | Temperatur um 9 Uhr | [°C] |
| `Temp3pm` | (Quantitativ, stetig) | Temperatur um 15 Uhr | [°C] |
| `RainToday` | (Qualitativ, nominal) | Indikatorvariable, ob es an diesem Tag geregnet hat | [Yes/No] |
| `RainTomorrow` | (Qualitativ, nominal) | Indikatorvariable, ob es am nächsten Tag geregnet hat | [Yes/No] |

***
### Betrachtung der Zusammenfassung <a name="eda-types"></a>


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 145460 entries, 0 to 145459
    Data columns (total 23 columns):
     #   Column         Non-Null Count   Dtype  
    ---  ------         --------------   -----  
     0   Date           145460 non-null  object 
     1   Location       145460 non-null  object 
     2   MinTemp        143975 non-null  float64
     3   MaxTemp        144199 non-null  float64
     4   Rainfall       142199 non-null  float64
     5   Evaporation    82670 non-null   float64
     6   Sunshine       75625 non-null   float64
     7   WindGustDir    135134 non-null  object 
     8   WindGustSpeed  135197 non-null  float64
     9   WindDir9am     134894 non-null  object 
     10  WindDir3pm     141232 non-null  object 
     11  WindSpeed9am   143693 non-null  float64
     12  WindSpeed3pm   142398 non-null  float64
     13  Humidity9am    142806 non-null  float64
     14  Humidity3pm    140953 non-null  float64
     15  Pressure9am    130395 non-null  float64
     16  Pressure3pm    130432 non-null  float64
     17  Cloud9am       89572 non-null   float64
     18  Cloud3pm       86102 non-null   float64
     19  Temp9am        143693 non-null  float64
     20  Temp3pm        141851 non-null  float64
     21  RainToday      142199 non-null  object 
     22  RainTomorrow   142193 non-null  object 
    dtypes: float64(16), object(7)
    memory usage: 25.5+ MB
    

Bei der Zusammenfassung fällt auf, dass wir mit einigen qualitativen Merkmalen arbeiten müssen. Ebenfalls scheinen wir einige `Null` Werte und fehlende Einträge zu haben. Um diese kümmern wir uns im Data Preparation.
***
### Statistische Merkmale der Daten <a name="eda-stat"></a>


```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MinTemp</th>
      <td>143975.0</td>
      <td>12.194034</td>
      <td>6.398495</td>
      <td>-8.5</td>
      <td>7.6</td>
      <td>12.0</td>
      <td>16.9</td>
      <td>33.9</td>
    </tr>
    <tr>
      <th>MaxTemp</th>
      <td>144199.0</td>
      <td>23.221348</td>
      <td>7.119049</td>
      <td>-4.8</td>
      <td>17.9</td>
      <td>22.6</td>
      <td>28.2</td>
      <td>48.1</td>
    </tr>
    <tr>
      <th>Rainfall</th>
      <td>142199.0</td>
      <td>2.360918</td>
      <td>8.478060</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.8</td>
      <td>371.0</td>
    </tr>
    <tr>
      <th>Evaporation</th>
      <td>82670.0</td>
      <td>5.468232</td>
      <td>4.193704</td>
      <td>0.0</td>
      <td>2.6</td>
      <td>4.8</td>
      <td>7.4</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>Sunshine</th>
      <td>75625.0</td>
      <td>7.611178</td>
      <td>3.785483</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.4</td>
      <td>10.6</td>
      <td>14.5</td>
    </tr>
    <tr>
      <th>WindGustSpeed</th>
      <td>135197.0</td>
      <td>40.035230</td>
      <td>13.607062</td>
      <td>6.0</td>
      <td>31.0</td>
      <td>39.0</td>
      <td>48.0</td>
      <td>135.0</td>
    </tr>
    <tr>
      <th>WindSpeed9am</th>
      <td>143693.0</td>
      <td>14.043426</td>
      <td>8.915375</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>19.0</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>WindSpeed3pm</th>
      <td>142398.0</td>
      <td>18.662657</td>
      <td>8.809800</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>19.0</td>
      <td>24.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>Humidity9am</th>
      <td>142806.0</td>
      <td>68.880831</td>
      <td>19.029164</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>70.0</td>
      <td>83.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Humidity3pm</th>
      <td>140953.0</td>
      <td>51.539116</td>
      <td>20.795902</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>52.0</td>
      <td>66.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Pressure9am</th>
      <td>130395.0</td>
      <td>1017.649940</td>
      <td>7.106530</td>
      <td>980.5</td>
      <td>1012.9</td>
      <td>1017.6</td>
      <td>1022.4</td>
      <td>1041.0</td>
    </tr>
    <tr>
      <th>Pressure3pm</th>
      <td>130432.0</td>
      <td>1015.255889</td>
      <td>7.037414</td>
      <td>977.1</td>
      <td>1010.4</td>
      <td>1015.2</td>
      <td>1020.0</td>
      <td>1039.6</td>
    </tr>
    <tr>
      <th>Cloud9am</th>
      <td>89572.0</td>
      <td>4.447461</td>
      <td>2.887159</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Cloud3pm</th>
      <td>86102.0</td>
      <td>4.509930</td>
      <td>2.720357</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Temp9am</th>
      <td>143693.0</td>
      <td>16.990631</td>
      <td>6.488753</td>
      <td>-7.2</td>
      <td>12.3</td>
      <td>16.7</td>
      <td>21.6</td>
      <td>40.2</td>
    </tr>
    <tr>
      <th>Temp3pm</th>
      <td>141851.0</td>
      <td>21.683390</td>
      <td>6.936650</td>
      <td>-5.4</td>
      <td>16.6</td>
      <td>21.1</td>
      <td>26.4</td>
      <td>46.7</td>
    </tr>
  </tbody>
</table>
</div>



Aus der obigen Tabelle geht hervor, dass es Aussreisser bei `Rainfall`, `Evaporation`, `WindSpeed9am` und `Windspeed3pm`geben kann. Dies erkennt man aus der Differenz zwischen dem max-Wert und den 75%-Quantil. Um dies zu überprüfen stellen wir die auffälligen Merkmale, Kapitel [Betrachtung der numerischen Variablen](#eda-bivN), als Histogramm dar.


```python
fig=plt.figure(figsize=(6,2),facecolor='white')

ax0=fig.add_subplot(1,1,1)
ax0.text(1.1,1,"Key figures",color='black',fontsize=28, fontweight='bold', fontfamily='monospace',ha='center')

ax0.text(0,0.4,"145k",color='gold',fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(0,0.001,"Number of items \nin the dataset",color='dimgrey',fontsize=17, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(0.75,0.4,"23",color='gold',fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(0.75,0.001,"Number of features \nin the dataset",color='dimgrey',fontsize=17, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(1.5,0.4,"31880",color='gold',fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(1.5,0.001,"Number of\nrainy days in",color='dimgrey',fontsize=17, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(2.25,0.4,"49",color='gold',fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(2.25,0.001,"Australian\nCities",color='dimgrey',fontsize=17, fontweight='light', fontfamily='monospace',ha='center')

ax0.set_yticklabels('')
ax0.tick_params(axis='y',length=0)
ax0.tick_params(axis='x',length=0)
ax0.set_xticklabels('')

for direction in ['top','right','left','bottom']:
    ax0.spines[direction].set_visible(False)
```


    
![png](output_16_0.png)
    


In 49 australischen Städten wurden 145'460 Messungen mit 23 unterschiedlichen Merkmalen aufgenommen. Im Datensatz befinden sich 31'880 Regentage.


***
### Univariate Analyse des Targets <a name="eda-univ"></a>


```python
dfTarget = df['RainTomorrow']
dfTarget.isnull().sum()
```




    3267




```python
dfTarget.unique()
dfTarget.value_counts()/len(dfTarget)
```




    No     0.758394
    Yes    0.219146
    Name: RainTomorrow, dtype: float64




```python
fig=plt.figure(figsize=(20,10),facecolor='white')

gs=fig.add_gridspec(1,1)

ax=[None for _ in range(2)]

ax[0]=fig.add_subplot(gs[0,0])

ax[0].text(-0.6,122000,"Verteilung der Regentage",fontsize=23,fontweight='bold', fontfamily='monospace')
ax[0].text(-0.6,117000,"In nur 21.9% der Messungen fällt Regen",fontsize=16,fontweight='light', fontfamily='monospace')

sns.countplot(data=df,x='RainTomorrow',ax=ax[0],order=['No','Yes'],palette=palette1,zorder=2)

for i in range(1):
    ax[i].grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(5,10))
    ax[i].set_ylabel('')
    ax[i].set_xlabel('')
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)


plt.show()
```


    
![png](output_20_0.png)
    


Obenstehendes Diagramm visualisiert alle 145'460 Einträge, von denen bei 31'880 Einträgen Regen festgestellt wurde.

***
### Bivariate Analyse  <a name="eda-biv"></a>

Im Folgenden soll das Zusammenspiel zweier Merkmale analysiert und grafisch dargestellt werden. Konkret wird Niederschlag und Luftfeichtigkeit, sowie Windgeschwindigkeit zu verschiedenen Zeiten verglichen.

#### Betrachtung der kategorischen Variablen <a name="eda-bivK"></a>


```python
categorical = [var for var in df.columns if df[var].dtype=='O']

print('Es sind {} quantitative Merkmale im Datensatz vorhanden\n'.format(len(categorical)))

print('Die quantitativen Merkmale sind:', categorical)
```

    Es sind 7 quantitative Merkmale im Datensatz vorhanden
    
    Die quantitativen Merkmale sind: ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    


```python
df[categorical].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>W</td>
      <td>W</td>
      <td>WNW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>WNW</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>WSW</td>
      <td>W</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>NE</td>
      <td>SE</td>
      <td>E</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>W</td>
      <td>ENE</td>
      <td>NW</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



Erkenntnisse der quantitativen Merkmale
- Der Zeitstempel ist vorhanden, jedoch nur als `String`
- Wir können das Modell an verschiedenen Ortschaften testen

#### Betrachtung der numerischen Variablen  <a name="eda-bivN"></a>


```python
fig=plt.figure(figsize=(20,10),facecolor='white')
gs=fig.add_gridspec(2,2)
ax=[None for i in range(4)]

# Rainfall
ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(-60, 140000,'Verteilung des Niederschlags',fontsize=23,fontweight='bold', fontfamily='monospace')
sns.histplot(x=df['Rainfall'], ax=ax[0], color='gold', alpha=1,zorder=2,linewidth=1,edgecolor='black',shrink=0.7, bins=50)
# Evaporation
ax[1]=fig.add_subplot(gs[0,1])
ax[1].text(-17, 28500,'Verteilung der Luftfeuchtigkeit',fontsize=23,fontweight='bold', fontfamily='monospace')
sns.histplot(x=df['Evaporation'], ax=ax[1], color='gold', alpha=1,zorder=2,linewidth=1,edgecolor='black',shrink=0.7, bins=50)
# Windspeed (9AM)
ax[2]=fig.add_subplot(gs[1,0])
ax[2].text(-20, 26000,'Verteilung der Windgeschwindigkeit 9 Uhr',fontsize=23,fontweight='bold', fontfamily='monospace')
sns.histplot(x=df['WindSpeed9am'], ax=ax[2], color='gold', alpha=1,zorder=2,linewidth=1,edgecolor='black',shrink=0.7, bins=50)
# Windspeed (3PM)
ax[3]=fig.add_subplot(gs[1,1])
ax[3].text(-10, 13700,'Verteilung der Windgeschwindigkeit 15 Uhr',fontsize=23,fontweight='bold', fontfamily='monospace')
sns.histplot(x=df['WindSpeed3pm'], ax=ax[3], color='gold', alpha=1,zorder=2,linewidth=1,edgecolor='black',shrink=0.7, bins=50)

for i in range(4):
    ax[i].set_ylabel('')
    ax[i].grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)
        
plt.tight_layout()
```


    
![png](output_25_0.png)
    


Alle vier Merkmale sind stark Rechtsschief. Mit einem Boxplot können wir nun erkennen, um wieviele Ausreisser es sich handelt.


```python
# Ausreisser in 'Rainfall' finden
# Python hätte mit Sicherheit eine Methode hierfür...
# |--------[===|===]--------|
# ^ loVal  ¦       ¦        ^ hiVal
#          ¦.......¦ -> IQR
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
loVal = df.Rainfall.quantile(0.25) - (IQR * 1.5)
loVal = loVal if (loVal > 0) else 0 # Regenmenge kann nicht negativ sein
hiVal = df.Rainfall.quantile(0.75) + (IQR * 1.5)

fig=plt.figure(figsize=(20,4),facecolor='white')
gs=fig.add_gridspec(1,1)

ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(-20,-0.7, "Boxplot Regenmenge",fontsize=23,fontweight='bold', fontfamily='monospace')
ax[0].text(-20,-0.6, "Ausreisser sind Werte < "+str(loVal)+" oder > " + str(hiVal),fontsize=16,fontweight='light', fontfamily='monospace')
sns.boxplot(x = 'Rainfall', data = df, palette = palette1);
```


    
![png](output_27_0.png)
    



```python
# Ausreisser in 'Evaporation' finden
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
loVal = df.Evaporation.quantile(0.25) - (IQR * 1.5)
loVal = loVal if (loVal > 0) else 0 # Evaporation kann nicht negativ sein
hiVal = df.Evaporation.quantile(0.75) + (IQR * 1.5)

fig=plt.figure(figsize=(20,4),facecolor='white')
gs=fig.add_gridspec(1,1)


ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(-8,-0.7, "Boxplot Verdunstung",fontsize=23,fontweight='bold', fontfamily='monospace')
ax[0].text(-8,-0.6, "Ausreisser sind Werte < "+str(loVal)+" oder > " + str(hiVal),fontsize=16,fontweight='light', fontfamily='monospace')
sns.boxplot(x = 'Evaporation', data = df, palette = palette1);
```


    
![png](output_28_0.png)
    


Sowie bei Regenmenge, als auch bei Verdunstung besteht der Boxplot vor allem aus Ausreissern. Das war zu erwarten, da es sich nur bei einem kleinen Teil der Tage um Regentage handelt. Bei der Verdunstung ist bereits ein aussagekräftigerer Boxplot zu sehen, da hier nicht nur Regenfall, sondern auch allgemeine Luftfeuchtigkeit (die als Morgentau wieder verdunsten kann) eine Rolle spielt.


```python
# Ausreisser in 'WindSpeed9am' finden
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
loVal = df.WindSpeed9am.quantile(0.25) - (IQR * 1.5)
loVal = loVal if (loVal > 0) else 0 # Windgeschwindigkeit kann nicht negativ sein (Betrag)
hiVal = df.WindSpeed9am.quantile(0.75) + (IQR * 1.5)

fig=plt.figure(figsize=(20,4),facecolor='white')
gs=fig.add_gridspec(1,1)

ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(-7,-0.7, "Boxplot Windgeschwindigkeit 9 Uhr",fontsize=23,fontweight='bold', fontfamily='monospace')
ax[0].text(-7,-0.6, "Ausreisser sind Werte < "+str(loVal)+" oder > " + str(hiVal),fontsize=16,fontweight='light', fontfamily='monospace')
sns.boxplot(x = 'WindSpeed9am', data = df, palette = palette1);
```


    
![png](output_30_0.png)
    



```python
# Ausreisser in 'WindSpeed3pm' finden
IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
loVal = df.WindSpeed3pm.quantile(0.25) - (IQR * 1.5)
loVal = loVal if (loVal > 0) else 0 # Windgeschwindigkeit kann nicht negativ sein (Betrag)
hiVal = df.WindSpeed3pm.quantile(0.75) + (IQR * 1.5)

fig=plt.figure(figsize=(20,4),facecolor='white')
gs=fig.add_gridspec(1,1)

ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(-4.7,-0.7, "Boxplot Windgeschwindigkeit 15 Uhr",fontsize=23,fontweight='bold', fontfamily='monospace')
ax[0].text(-4.7,-0.6, "Ausreisser sind Werte < "+str(loVal)+" oder > " + str(hiVal),fontsize=16,fontweight='light', fontfamily='monospace')
sns.boxplot(x = 'WindSpeed3pm', data = df, palette = palette1);
```


    
![png](output_31_0.png)
    


Die Anzahl der Ausreisser ist sehr hoch, darum werden wir uns im Data Preparation kümmern müssen.


```python
fig=plt.figure(figsize=(20,20),facecolor='white')
gs=fig.add_gridspec(2,1)
ax=[None for i in range(2)]

# MinTemp
ax[0]=fig.add_subplot(gs[0,0]);
ax[0].text(0, 38,'Minimale Temperatur nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='MinTemp', ax=ax[0], color='gold', linewidth=1)
g.set(xticklabels=[])
g.set(xlabel=None)
# MaxTemp
ax[0]=fig.add_subplot(gs[1,0]);
ax[0].text(0, 54,'Maximale Temperatur nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='MaxTemp', ax=ax[0], color='gold', linewidth=1)
g.set_xticklabels(g.get_xticklabels(), rotation=90);
```


    
![png](output_33_0.png)
    



```python
fig=plt.figure(figsize=(20,20),facecolor='white')
gs=fig.add_gridspec(2,1)
ax=[None for i in range(2)]

# MinTemp
ax[0]=fig.add_subplot(gs[0,0]);
ax[0].text(0, 400,'Niederschlag nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Rainfall', ax=ax[0], color='gold', linewidth=1)
g.set(xticklabels=[])
g.set(xlabel=None)
# MaxTemp
ax[0]=fig.add_subplot(gs[1,0]);
ax[0].text(0, 160,'Verdunstung nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Evaporation', ax=ax[0], color='gold', linewidth=1)
g.set_xticklabels(g.get_xticklabels(), rotation=90);
```


    
![png](output_34_0.png)
    



```python
fig=plt.figure(figsize=(20,20),facecolor='white')
gs=fig.add_gridspec(2,1)
ax=[None for i in range(2)]

# MinTemp
ax[0]=fig.add_subplot(gs[0,0]);
ax[0].text(0, 145,'Windgeschwindigkeit 9 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='WindSpeed9am', ax=ax[0], color='gold', linewidth=1)
g.set(xticklabels=[])
g.set(xlabel=None)
# MaxTemp
ax[0]=fig.add_subplot(gs[1,0]);
ax[0].text(0, 99,'Windgeschwindigkeit 15 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='WindSpeed3pm', ax=ax[0], color='gold', linewidth=1)
g.set_xticklabels(g.get_xticklabels(), rotation=90);
```


    
![png](output_35_0.png)
    



```python
fig=plt.figure(figsize=(20,20),facecolor='white')
gs=fig.add_gridspec(2,1)
ax=[None for i in range(2)]

# MinTemp
ax[0]=fig.add_subplot(gs[0,0]);
ax[0].text(0, 110,'Luftfeuchtigkeit 9 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Humidity9am', ax=ax[0], color='gold', linewidth=1)
g.set(xticklabels=[])
g.set(xlabel=None)
# MaxTemp
ax[0]=fig.add_subplot(gs[1,0]);
ax[0].text(0, 110,'Luftfeuchtigkeit 15 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Humidity3pm', ax=ax[0], color='gold', linewidth=1)
g.set_xticklabels(g.get_xticklabels(), rotation=90);
```


    
![png](output_36_0.png)
    



```python
fig=plt.figure(figsize=(20,20),facecolor='white')
gs=fig.add_gridspec(2,1)
ax=[None for i in range(2)]

# MinTemp
ax[0]=fig.add_subplot(gs[0,0]);
ax[0].text(0, 1045,'Luftdruck 9 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Pressure9am', ax=ax[0], color='gold', linewidth=1)
g.set(xticklabels=[])
g.set(xlabel=None)
# MaxTemp
ax[0]=fig.add_subplot(gs[1,0]);
ax[0].text(0, 1045,'Luftdruck 15 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Pressure3pm', ax=ax[0], color='gold', linewidth=1)
g.set_xticklabels(g.get_xticklabels(), rotation=90);
```


    
![png](output_37_0.png)
    



```python
fig=plt.figure(figsize=(20,20),facecolor='white')
gs=fig.add_gridspec(2,1)
ax=[None for i in range(2)]

# MinTemp
ax[0]=fig.add_subplot(gs[0,0]);
ax[0].text(0, 10,'Bewölkung 9 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Cloud9am', ax=ax[0], color='gold', linewidth=1)
g.set(xticklabels=[])
g.set(xlabel=None)
# MaxTemp
ax[0]=fig.add_subplot(gs[1,0]);
ax[0].text(0, 10,'Bewölkung 15 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Cloud3pm', ax=ax[0], color='gold', linewidth=1)
g.set_xticklabels(g.get_xticklabels(), rotation=90);
```


    
![png](output_38_0.png)
    



```python
fig=plt.figure(figsize=(20,20),facecolor='white')
gs=fig.add_gridspec(2,1)
ax=[None for i in range(2)]

# MinTemp
ax[0]=fig.add_subplot(gs[0,0]);
ax[0].text(0, 45,'Temperatur 9 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Temp9am', ax=ax[0], color='gold', linewidth=1)
g.set(xticklabels=[])
g.set(xlabel=None)
# MaxTemp
ax[0]=fig.add_subplot(gs[1,0]);
ax[0].text(0, 52,'Temperatur 15 Uhr nach Ort',fontsize=23,fontweight='bold', fontfamily='monospace')
g = sns.boxplot(data=df, x='Location', y='Temp3pm', ax=ax[0], color='gold', linewidth=1)
g.set_xticklabels(g.get_xticklabels(), rotation=90);
```


    
![png](output_39_0.png)
    


***
### Multivariate Analyse  <a name="eda-mul"></a>

Bei der multivariaten Analyse werden alle Merkmale und ihre Einflüsse aufeinander analysiert. Zuerst soll das anhand einer Datenmatrix ersichtlich gemacht werden, die als sogenannte "Heatmap" dargestellt wird.

<b> Heatmap</b>



```python
correlation = df.corr()
plt.figure(figsize=(16,12))

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linewidths=.5, cmap='YlOrRd')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)           
plt.show()
```


    
![png](output_41_0.png)
    



```python
fig=plt.figure(figsize=(20,20),facecolor='white')
gs=fig.add_gridspec(3,2)
ax=[None for i in range(6)]

# Rainfall
ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(0, -1,'Niederschlag - Stärkste Böe',fontsize=23,fontweight='bold', fontfamily='monospace')
table = pd.pivot_table(df, values='Rainfall', index='Location', columns='WindGustDir', aggfunc=np.median)
sns.heatmap(table, cmap='YlOrRd')
# Evaporation
ax[1]=fig.add_subplot(gs[0,1])
ax[1].text(0, -1,'Niederschlag - Luftdruck 9 Uhr',fontsize=23,fontweight='bold', fontfamily='monospace')
table = pd.pivot_table(df, values='Rainfall', index='Location', columns='Pressure9am', aggfunc=np.median)
sns.heatmap(table, cmap='YlOrRd')
# Windspeed (9AM)
ax[2]=fig.add_subplot(gs[1,0])
ax[2].text(0, -1,'Sonnenschein - Stärkste Böe',fontsize=23,fontweight='bold', fontfamily='monospace')
table = pd.pivot_table(df, values='Sunshine', index='Location', columns='WindGustDir', aggfunc=np.median)
sns.heatmap(table, cmap='YlOrRd')
# Windspeed (3PM)
ax[3]=fig.add_subplot(gs[1,1])
ax[3].text(0, -1,'Sonnenschein - Luftdruck 9 Uhr',fontsize=23,fontweight='bold', fontfamily='monospace')
table = pd.pivot_table(df, values='Sunshine', index='Location', columns='Pressure9am', aggfunc=np.median)
sns.heatmap(table, cmap='YlOrRd')
# Windspeed (9AM)
ax[4]=fig.add_subplot(gs[2,0])
ax[4].text(0, -1,'Bewölkung 9 Uhr - Stärkste Böe',fontsize=23,fontweight='bold', fontfamily='monospace')
table = pd.pivot_table(df, values='Cloud9am', index='Location', columns='WindGustDir', aggfunc=np.median)
sns.heatmap(table, cmap='YlOrRd')
# Windspeed (3PM)
ax[5]=fig.add_subplot(gs[2,1])
ax[5].text(0, -1,'Bewölkung 9 Uhr - Luftdruck 9 Uhr',fontsize=23,fontweight='bold', fontfamily='monospace')
table = pd.pivot_table(df, values='Cloud9am', index='Location', columns='Pressure9am', aggfunc=np.median)
sns.heatmap(table, cmap='YlOrRd')

for i in range(6):
    ax[i].set_ylabel('')
    ax[i].grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)
        
plt.tight_layout()
```


    
![png](output_42_0.png)
    


<b>Interpretation</b>

Je höher die Zahl (also je dunkler das Kästchen), desto stärker korrelieren die Merkmale zueinander. Ist die Zahl 0, gibt es keine Korrelation. Bei -1 herrscht umgekehrte Proportionalität (je mehr Wolken, desto weniger Sonnenschein). Dazu gehören:

- `MinTemp` $\leftrightarrow{}$ `MaxTemp`
- `Temp9am` $\leftrightarrow{}$ `Temp3pm`
- `WindGustSpeed` $\leftrightarrow{}$ `WindSpeed3pm`
- `Pressure9am` $\leftrightarrow{}$ `Pressure3pm`

<b>Pair Plot</b>

Als nächstes schauen wir uns die stark korrelierenden Merkmale im Pair Plot an.


```python
corVar = ['Location', 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']
sns.pairplot(df[corVar], kind='scatter', diag_kind='hist', corner=True, palette=palette1)
plt.show()
```


    
![png](output_44_0.png)
    


## Data Preparation <a name="dataprep"></a>
 Nun kümmern wir uns um die fehlenden Werte und teilen den Datensatz nach der `Location` auf, um eine bessere Übersicht zu bekommen


```python
# Fehlende Werte eliminieren
df_nona = df.dropna(); # von 145'460 auf 56'420 -> evt. kritisch
df_nona.shape
```




    (56420, 23)



Wenn wir alle fehlende Werte aus dem Datensatz streichen würden, hätten wir nur noch rund einen Drittel der Daten zur Verfügung.<br>
Wir werden deshalb die fehlenden Werte mit dem Median befüllen.


```python
df['MinTemp']=df['MinTemp'].fillna(df['MinTemp'].median())
df['MaxTemp']=df['MinTemp'].fillna(df['MaxTemp'].median())
df['Rainfall']=df['Rainfall'].fillna(df['Rainfall'].median())
df['Evaporation']=df['Evaporation'].fillna(df['Evaporation'].median())
df['Sunshine']=df['Sunshine'].fillna(df['Sunshine'].median())
df['WindGustSpeed']=df['WindGustSpeed'].fillna(df['WindGustSpeed'].median())
df['WindSpeed9am']=df['WindSpeed9am'].fillna(df['WindSpeed9am'].median())
df['WindSpeed3pm']=df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].median())
df['Humidity9am']=df['Humidity9am'].fillna(df['Humidity9am'].median())
df['Humidity3pm']=df['Humidity3pm'].fillna(df['Humidity3pm'].median())
df['Pressure9am']=df['Pressure9am'].fillna(df['Pressure9am'].median())
df['Pressure3pm']=df['Pressure3pm'].fillna(df['Pressure3pm'].median())
df['Cloud9am']=df['Cloud9am'].fillna(df['Cloud9am'].median())
df['Cloud3pm']=df['Cloud3pm'].fillna(df['Cloud3pm'].median())
df['Temp9am']=df['Temp9am'].fillna(df['Temp9am'].median())
df['Temp3pm']=df['Temp3pm'].fillna(df['Temp3pm'].median())
```

Für die kategorischen Variablen, machen wir dasselbe mit der `.mode()` Funktion.


```python
df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])
df['RainToday'] = df['RainToday'].fillna(df['RainToday'].mode()[0])
df['RainTomorrow'] = df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
df.isnull().sum()
```




    Date             0
    Location         0
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustDir      0
    WindGustSpeed    0
    WindDir9am       0
    WindDir3pm       0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    RainToday        0
    RainTomorrow     0
    dtype: int64



Da wir nun keine fehlende Werte mehr haben, können wir uns dem encoding der kategorischen Variablen zuwenden.


```python
from sklearn import preprocessing
df['WindDir9am'] = preprocessing.LabelEncoder().fit_transform(df['WindDir9am'])
df['WindDir3pm'] = preprocessing.LabelEncoder().fit_transform(df['WindDir3pm'])
df['WindGustDir'] = preprocessing.LabelEncoder().fit_transform(df['WindGustDir'])
```

Ebenfalls ersetzen wir `Yes / No` mit `1 / 0`


```python
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
```

und geben jeder Location ihre Koordinaten.


```python
df['lat'] = np.nan
df['lon'] = np.nan

# Behold, the almighty coordinates mapping procedure!
df['lat'] = np.where(df['Location']=='Albury', -36.073730, df['lat']) 
df['lon'] = np.where(df['Location']=='Albury', 146.913544, df['lon']) 

df['lat'] = np.where(df['Location']=='Adelaide', -34.928497, df['lat']) 
df['lon'] = np.where(df['Location']=='Adelaide', 138.600739, df['lon']) 

df['lat'] = np.where(df['Location']=='AliceSprings', -23.698042, df['lat']) 
df['lon'] = np.where(df['Location']=='AliceSprings', 133.880753, df['lon']) 

df['lat'] = np.where(df['Location']=='BadgerysCreek', -33.887421, df['lat']) 
df['lon'] = np.where(df['Location']=='BadgerysCreek', 150.740509, df['lon']) 

df['lat'] = np.where(df['Location']=='Ballarat', -37.5623025, df['lat']) 
df['lon'] = np.where(df['Location']=='Ballarat', 143.8605649, df['lon']) 

df['lat'] = np.where(df['Location']=='Bendigo', -36.7588767, df['lat']) 
df['lon'] = np.where(df['Location']=='Bendigo', 144.2825931, df['lon']) 

df['lat'] = np.where(df['Location']=='Brisbane', -27.4689682, df['lat']) 
df['lon'] = np.where(df['Location']=='Brisbane', 153.0234991, df['lon']) 

df['lat'] = np.where(df['Location']=='Cairns', -16.9206657, df['lat']) 
df['lon'] = np.where(df['Location']=='Cairns', 145.7721854, df['lon']) 

df['lat'] = np.where(df['Location']=='Cobar', -31.4983333, df['lat']) 
df['lon'] = np.where(df['Location']=='Cobar', 145.8344444, df['lon']) 

df['lat'] = np.where(df['Location']=='CoffsHarbour', -30.2962407, df['lat']) 
df['lon'] = np.where(df['Location']=='CoffsHarbour', 153.1135293, df['lon']) 

df['lat'] = np.where(df['Location']=='Moree', -29.4617202, df['lat']) 
df['lon'] = np.where(df['Location']=='Moree', 149.8407153, df['lon']) 

df['lat'] = np.where(df['Location']=='Newcastle', -32.9192953, df['lat']) 
df['lon'] = np.where(df['Location']=='Newcastle', 151.7795348, df['lon']) 

df['lat'] = np.where(df['Location']=='NorahHead', -33.2816667, df['lat']) 
df['lon'] = np.where(df['Location']=='NorahHead', 151.5677778, df['lon']) 

df['lat'] = np.where(df['Location']=='NorfolkIsland', -29.0289575, df['lat']) 
df['lon'] = np.where(df['Location']=='NorfolkIsland', 167.9587289, df['lon']) 

df['lat'] = np.where(df['Location']=='Penrith', -33.7510792, df['lat']) 
df['lon'] = np.where(df['Location']=='Penrith', 150.6941684, df['lon']) 

df['lat'] = np.where(df['Location']=='Richmond', -20.569655, df['lat']) 
df['lon'] = np.where(df['Location']=='Richmond', 142.9283602, df['lon']) 

df['lat'] = np.where(df['Location']=='Sydney', -33.8548157, df['lat']) 
df['lon'] = np.where(df['Location']=='Sydney', 151.2164539, df['lon']) 

df['lat'] = np.where(df['Location']=='SydneyAirport', -33.9498935, df['lat']) 
df['lon'] = np.where(df['Location']=='SydneyAirport', 151.1819682, df['lon']) 

df['lat'] = np.where(df['Location']=='WaggaWagga', -35.115, df['lat']) 
df['lon'] = np.where(df['Location']=='WaggaWagga', 147.3677778, df['lon']) 

df['lat'] = np.where(df['Location']=='Williamtown', -32.815, df['lat']) 
df['lon'] = np.where(df['Location']=='Williamtown', 151.8427778, df['lon']) 

df['lat'] = np.where(df['Location']=='Wollongong', -34.4243941, df['lat']) 
df['lon'] = np.where(df['Location']=='Wollongong', 150.89385, df['lon']) 

df['lat'] = np.where(df['Location']=='Canberra', -35.2975906, df['lat']) 
df['lon'] = np.where(df['Location']=='Canberra', 149.1012676, df['lon']) 

df['lat'] = np.where(df['Location']=='Tuggeranong', -35.4209771, df['lat']) 
df['lon'] = np.where(df['Location']=='Tuggeranong', 149.0921341, df['lon']) 

df['lat'] = np.where(df['Location']=='MountGinini', -35.5297196, df['lat']) 
df['lon'] = np.where(df['Location']=='MountGinini', 148.7726345, df['lon']) 

df['lat'] = np.where(df['Location']=='Sale', -38.1050358, df['lat']) 
df['lon'] = np.where(df['Location']=='Sale', 147.0647902, df['lon']) 

df['lat'] = np.where(df['Location']=='MelbourneAirport', -37.667111, df['lat']) 
df['lon'] = np.where(df['Location']=='MelbourneAirport', 144.8334808, df['lon']) 

df['lat'] = np.where(df['Location']=='Melbourne', -37.8142176, df['lat']) 
df['lon'] = np.where(df['Location']=='Melbourne', 144.9631608, df['lon']) 

df['lat'] = np.where(df['Location']=='Mildura', -34.1847265, df['lat']) 
df['lon'] = np.where(df['Location']=='Mildura', 142.1624972, df['lon']) 

df['lat'] = np.where(df['Location']=='Nhil', -35.4713087, df['lat']) 
df['lon'] = np.where(df['Location']=='Nhil', 141.3062355, df['lon']) 

df['lat'] = np.where(df['Location']=='Portland', -38.3456231, df['lat']) 
df['lon'] = np.where(df['Location']=='Portland', 141.6042304, df['lon']) 

df['lat'] = np.where(df['Location']=='Watsonia', -37.7110022, df['lat']) 
df['lon'] = np.where(df['Location']=='Watsonia', 145.083635, df['lon']) 

df['lat'] = np.where(df['Location']=='Dartmoor', -37.9225444, df['lat']) 
df['lon'] = np.where(df['Location']=='Dartmoor', 141.2766551, df['lon']) 

df['lat'] = np.where(df['Location']=='GoldCoast', -28.0023731, df['lat']) 
df['lon'] = np.where(df['Location']=='GoldCoast', 153.4145987, df['lon']) 

df['lat'] = np.where(df['Location']=='Townsville', -19.2569391, df['lat']) 
df['lon'] = np.where(df['Location']=='Townsville', 146.8239537, df['lon']) 

df['lat'] = np.where(df['Location']=='MountGambier', -37.8246698, df['lat']) 
df['lon'] = np.where(df['Location']=='MountGambier', 140.7820068, df['lon']) 

df['lat'] = np.where(df['Location']=='Nuriootpa', -34.4693354, df['lat']) 
df['lon'] = np.where(df['Location']=='Nuriootpa', 138.9939006, df['lon']) 

df['lat'] = np.where(df['Location']=='Woomera', -31.1999142, df['lat']) 
df['lon'] = np.where(df['Location']=='Woomera', 136.8253532, df['lon']) 

df['lat'] = np.where(df['Location']=='Albany', -35.0247822, df['lat']) 
df['lon'] = np.where(df['Location']=='Albany', 117.883608, df['lon']) 

df['lat'] = np.where(df['Location']=='Witchcliffe', -34.0263348, df['lat']) 
df['lon'] = np.where(df['Location']=='Witchcliffe', 115.1004768, df['lon']) 

df['lat'] = np.where(df['Location']=='PearceRAAF', -31.6739604, df['lat']) 
df['lon'] = np.where(df['Location']=='PearceRAAF', 116.0175435, df['lon']) 

df['lat'] = np.where(df['Location']=='PerthAirport', -31.9406095, df['lat']) 
df['lon'] = np.where(df['Location']=='PerthAirport', 115.9676077, df['lon']) 

df['lat'] = np.where(df['Location']=='Perth', -31.9527121, df['lat']) 
df['lon'] = np.where(df['Location']=='Perth', 115.8604796, df['lon']) 

df['lat'] = np.where(df['Location']=='SalmonGums', -32.9815347, df['lat']) 
df['lon'] = np.where(df['Location']=='SalmonGums', 121.6439417, df['lon']) 

df['lat'] = np.where(df['Location']=='Walpole', -34.9776796, df['lat']) 
df['lon'] = np.where(df['Location']=='Walpole', 116.7310063, df['lon']) 

df['lat'] = np.where(df['Location']=='Hobart', -42.8825088, df['lat']) 
df['lon'] = np.where(df['Location']=='Hobart', 147.3281233, df['lon']) 

df['lat'] = np.where(df['Location']=='Launceston', -41.4340813, df['lat']) 
df['lon'] = np.where(df['Location']=='Launceston', 147.1373496, df['lon']) 

df['lat'] = np.where(df['Location']=='Darwin', -12.46044, df['lat']) 
df['lon'] = np.where(df['Location']=='Darwin', 130.8410469, df['lon']) 

df['lat'] = np.where(df['Location']=='Katherine', -14.4646157, df['lat']) 
df['lon'] = np.where(df['Location']=='Katherine', 132.2635993, df['lon']) 

df['lat'] = np.where(df['Location']=='Uluru', -25.3455545, df['lat']) 
df['lon'] = np.where(df['Location']=='Uluru', 131.0369615, df['lon']) 

```


```python
# import the library and its Marker clusterization service
!pip install folium
import folium
from folium.plugins import MarkerCluster
# Create a map object and center it to the avarage coordinates to m
m = folium.Map(location=df[["lat", "lon"]].mean().to_list(), zoom_start=4)
# if the points are too close to each other, cluster them, create a cluster overlay with MarkerCluster, add to m
marker_cluster = MarkerCluster().add_to(m)
# draw only once per location
df_loc = df.drop_duplicates(subset = ["Location"])
# draw the markers and assign popup and hover texts
# add the markers the the cluster layers so that they are automatically clustered
for i,r in df_loc.iterrows():
    location = (r["lat"], r["lon"])
    folium.Marker(location=location,
                      popup = r['Location'],
                      tooltip=sum(df['Location'] == r['Location']))\
    .add_to(marker_cluster)
# display the map
m
```

    Requirement already satisfied: folium in c:\users\alexk\anaconda3\lib\site-packages (0.12.1.post1)
    Requirement already satisfied: branca>=0.3.0 in c:\users\alexk\anaconda3\lib\site-packages (from folium) (0.4.2)
    Requirement already satisfied: jinja2>=2.9 in c:\users\alexk\anaconda3\lib\site-packages (from folium) (2.11.3)
    Requirement already satisfied: numpy in c:\users\alexk\anaconda3\lib\site-packages (from folium) (1.20.3)
    Requirement already satisfied: requests in c:\users\alexk\anaconda3\lib\site-packages (from folium) (2.26.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\alexk\anaconda3\lib\site-packages (from jinja2>=2.9->folium) (1.1.1)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\alexk\anaconda3\lib\site-packages (from requests->folium) (2021.10.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\alexk\anaconda3\lib\site-packages (from requests->folium) (1.26.7)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\alexk\anaconda3\lib\site-packages (from requests->folium) (3.2)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\alexk\anaconda3\lib\site-packages (from requests->folium) (2.0.4)
    




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css%22/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_44bd7bd5c3c44678a92964bada2948a1%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/leaflet.markercluster.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.Default.css%22/%3E%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_44bd7bd5c3c44678a92964bada2948a1%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_44bd7bd5c3c44678a92964bada2948a1%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_44bd7bd5c3c44678a92964bada2948a1%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B-32.49528808291161%2C%20141.92955329449327%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%204%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_1ea5e2ffc1de4971b21a35578fdf2e1f%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_44bd7bd5c3c44678a92964bada2948a1%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%20%3D%20L.markerClusterGroup%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20map_44bd7bd5c3c44678a92964bada2948a1.addLayer%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_700c817a8cda4d7d83ff5a6a0da624a3%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-36.07373%2C%20146.913544%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b70301ce95674cdabe7e057a77a7fdb6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b570a4baa2a2466aaf3d3461f5a34137%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b570a4baa2a2466aaf3d3461f5a34137%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAlbury%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b70301ce95674cdabe7e057a77a7fdb6.setContent%28html_b570a4baa2a2466aaf3d3461f5a34137%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_700c817a8cda4d7d83ff5a6a0da624a3.bindPopup%28popup_b70301ce95674cdabe7e057a77a7fdb6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_700c817a8cda4d7d83ff5a6a0da624a3.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_58f59a0bf9bb40a7aa233a0ea466a520%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-33.887421%2C%20150.740509%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c4c4dbf91738495f866741ca8a8d3520%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c7631525af054599be3190f651b89b6d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c7631525af054599be3190f651b89b6d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBadgerysCreek%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c4c4dbf91738495f866741ca8a8d3520.setContent%28html_c7631525af054599be3190f651b89b6d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_58f59a0bf9bb40a7aa233a0ea466a520.bindPopup%28popup_c4c4dbf91738495f866741ca8a8d3520%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_58f59a0bf9bb40a7aa233a0ea466a520.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_b47ce6ab551840d6b1c60b59274af311%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-31.4983333%2C%20145.8344444%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_018f1217f45f4da3aa9a0a636a39fcf1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2dcdcd6722a5499684176e657a81483d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2dcdcd6722a5499684176e657a81483d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECobar%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_018f1217f45f4da3aa9a0a636a39fcf1.setContent%28html_2dcdcd6722a5499684176e657a81483d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_b47ce6ab551840d6b1c60b59274af311.bindPopup%28popup_018f1217f45f4da3aa9a0a636a39fcf1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_b47ce6ab551840d6b1c60b59274af311.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_da0a3162008f42a489ef0dda659360d8%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-30.2962407%2C%20153.1135293%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_19975e3c7fa447a58298563dbcbc2cb6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5f5c9049919f45859e81ad402b59f43b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5f5c9049919f45859e81ad402b59f43b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffsHarbour%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_19975e3c7fa447a58298563dbcbc2cb6.setContent%28html_5f5c9049919f45859e81ad402b59f43b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_da0a3162008f42a489ef0dda659360d8.bindPopup%28popup_19975e3c7fa447a58298563dbcbc2cb6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_da0a3162008f42a489ef0dda659360d8.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_4db544e97c0a4e248180d30d9b4875cf%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-29.4617202%2C%20149.8407153%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f85acb2934a0453b8b3fa32bf82a09e1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e1b3669ba94c4f2c8732e8c5f48c590f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e1b3669ba94c4f2c8732e8c5f48c590f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMoree%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f85acb2934a0453b8b3fa32bf82a09e1.setContent%28html_e1b3669ba94c4f2c8732e8c5f48c590f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_4db544e97c0a4e248180d30d9b4875cf.bindPopup%28popup_f85acb2934a0453b8b3fa32bf82a09e1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_4db544e97c0a4e248180d30d9b4875cf.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_bb5383cf0ac34d53b5ffbc09e4b10113%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-32.9192953%2C%20151.7795348%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_47af6d6b88b349b1ae9b17dcde2ab067%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d8c7faeb69cd48ce9cbd8f25a64d6d24%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d8c7faeb69cd48ce9cbd8f25a64d6d24%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENewcastle%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_47af6d6b88b349b1ae9b17dcde2ab067.setContent%28html_d8c7faeb69cd48ce9cbd8f25a64d6d24%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_bb5383cf0ac34d53b5ffbc09e4b10113.bindPopup%28popup_47af6d6b88b349b1ae9b17dcde2ab067%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_bb5383cf0ac34d53b5ffbc09e4b10113.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203039%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_7df5fc70155d4a449fdd1271def50e8a%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-33.2816667%2C%20151.5677778%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7fdc4f23a834493888a89c20bd15452f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4d765eeb74fd49b897576992e7b46c71%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4d765eeb74fd49b897576992e7b46c71%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorahHead%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7fdc4f23a834493888a89c20bd15452f.setContent%28html_4d765eeb74fd49b897576992e7b46c71%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_7df5fc70155d4a449fdd1271def50e8a.bindPopup%28popup_7fdc4f23a834493888a89c20bd15452f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_7df5fc70155d4a449fdd1271def50e8a.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203004%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_e6c0dfb0caef4f2095436f08fdcfd119%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-29.0289575%2C%20167.9587289%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_37c5221ae6f64cc49f6c5801f834e449%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c63492f86cd44041ac50d4dfa3c8d50f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c63492f86cd44041ac50d4dfa3c8d50f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorfolkIsland%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_37c5221ae6f64cc49f6c5801f834e449.setContent%28html_c63492f86cd44041ac50d4dfa3c8d50f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_e6c0dfb0caef4f2095436f08fdcfd119.bindPopup%28popup_37c5221ae6f64cc49f6c5801f834e449%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_e6c0dfb0caef4f2095436f08fdcfd119.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_35c00599f80647c8bb8d5cccab2527a4%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-33.7510792%2C%20150.6941684%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_039c3bc1857040d58a6cd44854ddf0a9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1cf11ad4ed9e4d318d713e3ae750e630%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1cf11ad4ed9e4d318d713e3ae750e630%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPenrith%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_039c3bc1857040d58a6cd44854ddf0a9.setContent%28html_1cf11ad4ed9e4d318d713e3ae750e630%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_35c00599f80647c8bb8d5cccab2527a4.bindPopup%28popup_039c3bc1857040d58a6cd44854ddf0a9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_35c00599f80647c8bb8d5cccab2527a4.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203039%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_b1b97cfae6ab436d8bf2592356546c03%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-20.569655%2C%20142.9283602%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_54cd217ab40c4ee991ce2169547f1b09%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7c1a298bf6224c2a8ab03f48d10361d6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7c1a298bf6224c2a8ab03f48d10361d6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERichmond%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_54cd217ab40c4ee991ce2169547f1b09.setContent%28html_7c1a298bf6224c2a8ab03f48d10361d6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_b1b97cfae6ab436d8bf2592356546c03.bindPopup%28popup_54cd217ab40c4ee991ce2169547f1b09%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_b1b97cfae6ab436d8bf2592356546c03.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_75070403363840828be25985491e71da%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-33.8548157%2C%20151.2164539%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_33f3e1b2f54f4f459ffd0ead34feca40%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_92e18fbd1c97491484f28d1794f0f751%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_92e18fbd1c97491484f28d1794f0f751%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESydney%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_33f3e1b2f54f4f459ffd0ead34feca40.setContent%28html_92e18fbd1c97491484f28d1794f0f751%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_75070403363840828be25985491e71da.bindPopup%28popup_33f3e1b2f54f4f459ffd0ead34feca40%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_75070403363840828be25985491e71da.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203344%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_71d38275cac44a81a35030ecb7a8e177%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-33.9498935%2C%20151.1819682%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_467c3219ffbb4cffbdb8f627ae731b5c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7ff3e676632749fda0c73dbfd2e70ef8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7ff3e676632749fda0c73dbfd2e70ef8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESydneyAirport%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_467c3219ffbb4cffbdb8f627ae731b5c.setContent%28html_7ff3e676632749fda0c73dbfd2e70ef8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_71d38275cac44a81a35030ecb7a8e177.bindPopup%28popup_467c3219ffbb4cffbdb8f627ae731b5c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_71d38275cac44a81a35030ecb7a8e177.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_861b2943309a421b8ad4ccf1ee308554%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-35.115%2C%20147.3677778%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_825dc7216d18478fa48f273dd1011fc1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d992ba1d81d5455db5e80f118dee1cc3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d992ba1d81d5455db5e80f118dee1cc3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWaggaWagga%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_825dc7216d18478fa48f273dd1011fc1.setContent%28html_d992ba1d81d5455db5e80f118dee1cc3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_861b2943309a421b8ad4ccf1ee308554.bindPopup%28popup_825dc7216d18478fa48f273dd1011fc1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_861b2943309a421b8ad4ccf1ee308554.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_f6748548790941b9a02f6f1e659e008f%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-32.815%2C%20151.8427778%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_77d34199d93f4a9f9c42c99956e2a3bc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0fa712478b594ed9be61b3655750bbed%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0fa712478b594ed9be61b3655750bbed%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWilliamtown%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_77d34199d93f4a9f9c42c99956e2a3bc.setContent%28html_0fa712478b594ed9be61b3655750bbed%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_f6748548790941b9a02f6f1e659e008f.bindPopup%28popup_77d34199d93f4a9f9c42c99956e2a3bc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_f6748548790941b9a02f6f1e659e008f.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_5faef8f3979f4e2cb266aaeaa5a3dbe4%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-34.4243941%2C%20150.89385%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e400ef09a4f543e0b29735cde27b81da%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_44864eafdcb747e786b9363e382543ba%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_44864eafdcb747e786b9363e382543ba%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWollongong%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e400ef09a4f543e0b29735cde27b81da.setContent%28html_44864eafdcb747e786b9363e382543ba%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_5faef8f3979f4e2cb266aaeaa5a3dbe4.bindPopup%28popup_e400ef09a4f543e0b29735cde27b81da%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_5faef8f3979f4e2cb266aaeaa5a3dbe4.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_e232d218c40f4011b18b1f71948f6b3d%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-35.2975906%2C%20149.1012676%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c5e6502688e64bb782ab04f631b66426%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4f6056964ca14929b7ef326f1dba5493%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4f6056964ca14929b7ef326f1dba5493%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECanberra%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c5e6502688e64bb782ab04f631b66426.setContent%28html_4f6056964ca14929b7ef326f1dba5493%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_e232d218c40f4011b18b1f71948f6b3d.bindPopup%28popup_c5e6502688e64bb782ab04f631b66426%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_e232d218c40f4011b18b1f71948f6b3d.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203436%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_94c7a2515cc14038acd56ecc8d5877c9%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-35.4209771%2C%20149.0921341%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_caa977568b3b4e8681054f4ac5fc3ccd%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0146d5eece53431a94e30898fdd772ab%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0146d5eece53431a94e30898fdd772ab%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ETuggeranong%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_caa977568b3b4e8681054f4ac5fc3ccd.setContent%28html_0146d5eece53431a94e30898fdd772ab%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_94c7a2515cc14038acd56ecc8d5877c9.bindPopup%28popup_caa977568b3b4e8681054f4ac5fc3ccd%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_94c7a2515cc14038acd56ecc8d5877c9.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203039%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_fb73e36b53ff4dce969750963d7875f4%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-35.5297196%2C%20148.7726345%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_05815c3e647447988ee8a80430acc768%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7eb8921871214472b4f01ed73ae44b19%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7eb8921871214472b4f01ed73ae44b19%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMountGinini%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_05815c3e647447988ee8a80430acc768.setContent%28html_7eb8921871214472b4f01ed73ae44b19%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_fb73e36b53ff4dce969750963d7875f4.bindPopup%28popup_05815c3e647447988ee8a80430acc768%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_fb73e36b53ff4dce969750963d7875f4.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_9fa60db0912a4a17b1ba4c0aa6fae176%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-37.5623025%2C%20143.8605649%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7575eb2a8dfb4ad599c74adfd10bd5d4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c12eef97ed354880a111629512358bff%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c12eef97ed354880a111629512358bff%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBallarat%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7575eb2a8dfb4ad599c74adfd10bd5d4.setContent%28html_c12eef97ed354880a111629512358bff%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_9fa60db0912a4a17b1ba4c0aa6fae176.bindPopup%28popup_7575eb2a8dfb4ad599c74adfd10bd5d4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_9fa60db0912a4a17b1ba4c0aa6fae176.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_27f5f3d9343147e5992570b81acd5642%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-36.7588767%2C%20144.2825931%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6ce1a80c562d4f5885da51e6ea74806e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3fdbdd7150e54ce89e23128243f21c89%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3fdbdd7150e54ce89e23128243f21c89%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBendigo%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6ce1a80c562d4f5885da51e6ea74806e.setContent%28html_3fdbdd7150e54ce89e23128243f21c89%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_27f5f3d9343147e5992570b81acd5642.bindPopup%28popup_6ce1a80c562d4f5885da51e6ea74806e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_27f5f3d9343147e5992570b81acd5642.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_441bf8f821784f9496227db165050d14%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-38.1050358%2C%20147.0647902%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b4847f042f674decae851935cfb34cac%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2bfb5de282a6434498bd7355a9b43f89%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2bfb5de282a6434498bd7355a9b43f89%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESale%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b4847f042f674decae851935cfb34cac.setContent%28html_2bfb5de282a6434498bd7355a9b43f89%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_441bf8f821784f9496227db165050d14.bindPopup%28popup_b4847f042f674decae851935cfb34cac%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_441bf8f821784f9496227db165050d14.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_3789b61769ac4f78b3200d23f940c9c1%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-37.667111%2C%20144.8334808%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1741d1cee1d3404b930369792c65d28b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_de755f25b3e54cb18e13abaf284e706c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_de755f25b3e54cb18e13abaf284e706c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMelbourneAirport%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1741d1cee1d3404b930369792c65d28b.setContent%28html_de755f25b3e54cb18e13abaf284e706c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_3789b61769ac4f78b3200d23f940c9c1.bindPopup%28popup_1741d1cee1d3404b930369792c65d28b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_3789b61769ac4f78b3200d23f940c9c1.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_e106453d1ebc4c6a869463ecdc9c3197%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-37.8142176%2C%20144.9631608%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_eeeef2dc69044166b482a5e75bffe91d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9f9c262ba1154a359d28d8b93e536dd3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9f9c262ba1154a359d28d8b93e536dd3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMelbourne%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_eeeef2dc69044166b482a5e75bffe91d.setContent%28html_9f9c262ba1154a359d28d8b93e536dd3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_e106453d1ebc4c6a869463ecdc9c3197.bindPopup%28popup_eeeef2dc69044166b482a5e75bffe91d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_e106453d1ebc4c6a869463ecdc9c3197.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203193%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_655ebb57cf5c499484331128684b03d2%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-34.1847265%2C%20142.1624972%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1d152d36300a458bac46e78d6b86afb0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f6c2954eaaf94329b10b922a7bb77687%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f6c2954eaaf94329b10b922a7bb77687%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMildura%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1d152d36300a458bac46e78d6b86afb0.setContent%28html_f6c2954eaaf94329b10b922a7bb77687%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_655ebb57cf5c499484331128684b03d2.bindPopup%28popup_1d152d36300a458bac46e78d6b86afb0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_655ebb57cf5c499484331128684b03d2.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_a842795275e940e2b93bc05578e61c20%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-35.4713087%2C%20141.3062355%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d1bef111a85f46de8ff977acbc0ba899%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f5aa5c34ed8c462da319cff7aa33d863%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f5aa5c34ed8c462da319cff7aa33d863%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENhil%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d1bef111a85f46de8ff977acbc0ba899.setContent%28html_f5aa5c34ed8c462da319cff7aa33d863%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_a842795275e940e2b93bc05578e61c20.bindPopup%28popup_d1bef111a85f46de8ff977acbc0ba899%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_a842795275e940e2b93bc05578e61c20.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%201578%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_2448305c325245c7ad89937f875ae3fc%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-38.3456231%2C%20141.6042304%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ff3cb6926ec3499cbbac83255d64898b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_12d8a156d8e54c2d96eefe7885636717%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_12d8a156d8e54c2d96eefe7885636717%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPortland%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ff3cb6926ec3499cbbac83255d64898b.setContent%28html_12d8a156d8e54c2d96eefe7885636717%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_2448305c325245c7ad89937f875ae3fc.bindPopup%28popup_ff3cb6926ec3499cbbac83255d64898b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_2448305c325245c7ad89937f875ae3fc.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_54c64b43e23f4449878e98e0bf614d6a%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-37.7110022%2C%20145.083635%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_90ef8c2bf14541259bb88ea5dd79c06a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_321ce795965748e28e9a3b9ddc3b8edb%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_321ce795965748e28e9a3b9ddc3b8edb%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWatsonia%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_90ef8c2bf14541259bb88ea5dd79c06a.setContent%28html_321ce795965748e28e9a3b9ddc3b8edb%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_54c64b43e23f4449878e98e0bf614d6a.bindPopup%28popup_90ef8c2bf14541259bb88ea5dd79c06a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_54c64b43e23f4449878e98e0bf614d6a.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_fec6f46a6c034dac9534c06eb5442a11%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-37.9225444%2C%20141.2766551%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dc84ac032bed40b28f19f79dd545eac6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_96e20f99ed1b4a748d2b9488f77907f6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_96e20f99ed1b4a748d2b9488f77907f6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDartmoor%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dc84ac032bed40b28f19f79dd545eac6.setContent%28html_96e20f99ed1b4a748d2b9488f77907f6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_fec6f46a6c034dac9534c06eb5442a11.bindPopup%28popup_dc84ac032bed40b28f19f79dd545eac6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_fec6f46a6c034dac9534c06eb5442a11.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_28dd3dc3d9c64eb78f95aa77717817fc%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-27.4689682%2C%20153.0234991%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8939ce3684cd4b98895cbfe428f6344a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5ca53d8be93e44c59fcb2271aae99fb0%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5ca53d8be93e44c59fcb2271aae99fb0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrisbane%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8939ce3684cd4b98895cbfe428f6344a.setContent%28html_5ca53d8be93e44c59fcb2271aae99fb0%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_28dd3dc3d9c64eb78f95aa77717817fc.bindPopup%28popup_8939ce3684cd4b98895cbfe428f6344a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_28dd3dc3d9c64eb78f95aa77717817fc.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203193%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_2201aa64fc474b35b224caf0ba1f12ae%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-16.9206657%2C%20145.7721854%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_69ba9fff097449dabcbd03e27410a250%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_faecf3a92281459f892652d74b558bef%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_faecf3a92281459f892652d74b558bef%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECairns%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_69ba9fff097449dabcbd03e27410a250.setContent%28html_faecf3a92281459f892652d74b558bef%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_2201aa64fc474b35b224caf0ba1f12ae.bindPopup%28popup_69ba9fff097449dabcbd03e27410a250%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_2201aa64fc474b35b224caf0ba1f12ae.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_9aa70f3a19c3494e80aa5c546e7b1951%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-28.0023731%2C%20153.4145987%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6889154fde8f4a69805fca4dbc9bb488%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0cf41f6592ee428f81cfe1a687ed96ee%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0cf41f6592ee428f81cfe1a687ed96ee%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGoldCoast%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6889154fde8f4a69805fca4dbc9bb488.setContent%28html_0cf41f6592ee428f81cfe1a687ed96ee%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_9aa70f3a19c3494e80aa5c546e7b1951.bindPopup%28popup_6889154fde8f4a69805fca4dbc9bb488%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_9aa70f3a19c3494e80aa5c546e7b1951.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_eedf0a88721246f093e817320b3d1f81%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-19.2569391%2C%20146.8239537%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ab3cd7f4485b4740a8bef747d3656965%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b6bed04800fe48a393294b688d271013%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b6bed04800fe48a393294b688d271013%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ETownsville%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ab3cd7f4485b4740a8bef747d3656965.setContent%28html_b6bed04800fe48a393294b688d271013%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_eedf0a88721246f093e817320b3d1f81.bindPopup%28popup_ab3cd7f4485b4740a8bef747d3656965%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_eedf0a88721246f093e817320b3d1f81.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_8dd57eee5f374815a203a6ea7d011664%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-34.928497%2C%20138.600739%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_edc6b95ae0594a62963fc9c830c75c3d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d9007e373ce440f6a4560230a75aa2ae%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d9007e373ce440f6a4560230a75aa2ae%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAdelaide%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_edc6b95ae0594a62963fc9c830c75c3d.setContent%28html_d9007e373ce440f6a4560230a75aa2ae%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_8dd57eee5f374815a203a6ea7d011664.bindPopup%28popup_edc6b95ae0594a62963fc9c830c75c3d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_8dd57eee5f374815a203a6ea7d011664.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203193%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_a7b0467542ff4ce29abf542a0681479f%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-37.8246698%2C%20140.7820068%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3dcfeff48e2343e1a2f523ef80bb4dc9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0fbaa74052bc4876898595459145b72f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0fbaa74052bc4876898595459145b72f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMountGambier%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3dcfeff48e2343e1a2f523ef80bb4dc9.setContent%28html_0fbaa74052bc4876898595459145b72f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_a7b0467542ff4ce29abf542a0681479f.bindPopup%28popup_3dcfeff48e2343e1a2f523ef80bb4dc9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_a7b0467542ff4ce29abf542a0681479f.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_247bada06f4b42a8946fa34db7cbe111%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-34.4693354%2C%20138.9939006%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_82d93cbd27174f9f8a46900414021a50%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_be38ad8016f548fab43fcbdf1f60ecd7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_be38ad8016f548fab43fcbdf1f60ecd7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENuriootpa%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_82d93cbd27174f9f8a46900414021a50.setContent%28html_be38ad8016f548fab43fcbdf1f60ecd7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_247bada06f4b42a8946fa34db7cbe111.bindPopup%28popup_82d93cbd27174f9f8a46900414021a50%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_247bada06f4b42a8946fa34db7cbe111.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_57fe413a2d1f4ccd80dfce386ee76b05%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-31.1999142%2C%20136.8253532%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3e7f8079594c4387a154c95b69106710%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_70dd6fe2b0b84c459c53e2becf24a389%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_70dd6fe2b0b84c459c53e2becf24a389%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWoomera%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3e7f8079594c4387a154c95b69106710.setContent%28html_70dd6fe2b0b84c459c53e2becf24a389%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_57fe413a2d1f4ccd80dfce386ee76b05.bindPopup%28popup_3e7f8079594c4387a154c95b69106710%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_57fe413a2d1f4ccd80dfce386ee76b05.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_3b5ae5ae07e448648f8bb806fe7cb483%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-35.0247822%2C%20117.883608%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_766f9d890785465b8d0748c470538b68%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_29afcf1015ea4bcbbd9f395f557aa3ea%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_29afcf1015ea4bcbbd9f395f557aa3ea%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAlbany%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_766f9d890785465b8d0748c470538b68.setContent%28html_29afcf1015ea4bcbbd9f395f557aa3ea%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_3b5ae5ae07e448648f8bb806fe7cb483.bindPopup%28popup_766f9d890785465b8d0748c470538b68%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_3b5ae5ae07e448648f8bb806fe7cb483.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_d47605809ff4495da472ffb28daea931%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-34.0263348%2C%20115.1004768%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ef219e8837f041b881b88efc90ee72de%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_31e07db310794528a8eaf5ad5a89b1ad%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_31e07db310794528a8eaf5ad5a89b1ad%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWitchcliffe%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ef219e8837f041b881b88efc90ee72de.setContent%28html_31e07db310794528a8eaf5ad5a89b1ad%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_d47605809ff4495da472ffb28daea931.bindPopup%28popup_ef219e8837f041b881b88efc90ee72de%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_d47605809ff4495da472ffb28daea931.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_2b225926eb3a47bc87e24903b794d949%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-31.6739604%2C%20116.0175435%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_33e7eecd2fae456db91657c09ee4d864%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c100357d95b24b13911bd528256429b8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c100357d95b24b13911bd528256429b8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPearceRAAF%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_33e7eecd2fae456db91657c09ee4d864.setContent%28html_c100357d95b24b13911bd528256429b8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_2b225926eb3a47bc87e24903b794d949.bindPopup%28popup_33e7eecd2fae456db91657c09ee4d864%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_2b225926eb3a47bc87e24903b794d949.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_90f8c32510414f4d850f0a1d80067318%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-31.9406095%2C%20115.9676077%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2197f7ad92284b08b6b9f7a99e3cb838%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a1f70615afa3423f82a2d867fb5a494b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a1f70615afa3423f82a2d867fb5a494b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPerthAirport%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2197f7ad92284b08b6b9f7a99e3cb838.setContent%28html_a1f70615afa3423f82a2d867fb5a494b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_90f8c32510414f4d850f0a1d80067318.bindPopup%28popup_2197f7ad92284b08b6b9f7a99e3cb838%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_90f8c32510414f4d850f0a1d80067318.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203009%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_633c35990a59442aa1cf8bb0a16b7f25%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-31.9527121%2C%20115.8604796%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a88ebff4076c4d78823bbea704a4d092%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_947e54cc486640f69650b0111079d645%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_947e54cc486640f69650b0111079d645%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPerth%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a88ebff4076c4d78823bbea704a4d092.setContent%28html_947e54cc486640f69650b0111079d645%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_633c35990a59442aa1cf8bb0a16b7f25.bindPopup%28popup_a88ebff4076c4d78823bbea704a4d092%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_633c35990a59442aa1cf8bb0a16b7f25.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203193%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_94bf55a7f6414b1db40fd56bd9b6d99f%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-32.9815347%2C%20121.6439417%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fc0f052493394b18adb937367f02b1bb%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e25242d4103c49688ffbc37d6a1b6092%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e25242d4103c49688ffbc37d6a1b6092%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESalmonGums%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fc0f052493394b18adb937367f02b1bb.setContent%28html_e25242d4103c49688ffbc37d6a1b6092%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_94bf55a7f6414b1db40fd56bd9b6d99f.bindPopup%28popup_fc0f052493394b18adb937367f02b1bb%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_94bf55a7f6414b1db40fd56bd9b6d99f.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203001%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_40d1276fd6634ce1ad50cd0d48c9b054%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-34.9776796%2C%20116.7310063%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3382b18faf314236b4aceb665e2ac784%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_00a3d4c81dd848c0a2db2b030af06f8d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_00a3d4c81dd848c0a2db2b030af06f8d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWalpole%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3382b18faf314236b4aceb665e2ac784.setContent%28html_00a3d4c81dd848c0a2db2b030af06f8d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_40d1276fd6634ce1ad50cd0d48c9b054.bindPopup%28popup_3382b18faf314236b4aceb665e2ac784%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_40d1276fd6634ce1ad50cd0d48c9b054.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203006%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_e2a930f94f4f4c1a9988580f65ceb31d%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-42.8825088%2C%20147.3281233%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_83cba397459542e183f8e97870ae5c83%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8329f66c9b48405b8f53d5d6e374e9f1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8329f66c9b48405b8f53d5d6e374e9f1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHobart%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_83cba397459542e183f8e97870ae5c83.setContent%28html_8329f66c9b48405b8f53d5d6e374e9f1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_e2a930f94f4f4c1a9988580f65ceb31d.bindPopup%28popup_83cba397459542e183f8e97870ae5c83%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_e2a930f94f4f4c1a9988580f65ceb31d.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203193%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_06ca284c0b6d4dc0a89d61e07683532d%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-41.4340813%2C%20147.1373496%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_80f0ca323f354ab4a8cb370337748efa%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_de75b68ee19446908c6ad5e31c14800f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_de75b68ee19446908c6ad5e31c14800f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELaunceston%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_80f0ca323f354ab4a8cb370337748efa.setContent%28html_de75b68ee19446908c6ad5e31c14800f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_06ca284c0b6d4dc0a89d61e07683532d.bindPopup%28popup_80f0ca323f354ab4a8cb370337748efa%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_06ca284c0b6d4dc0a89d61e07683532d.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_d512f57cea234d8b8f767c69a3db1105%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-23.698042%2C%20133.880753%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7ba5cf725dd145d39d2b22594a1020ac%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_76e3f82a6018439780b5fb48ae1e5dbd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_76e3f82a6018439780b5fb48ae1e5dbd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAliceSprings%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7ba5cf725dd145d39d2b22594a1020ac.setContent%28html_76e3f82a6018439780b5fb48ae1e5dbd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_d512f57cea234d8b8f767c69a3db1105.bindPopup%28popup_7ba5cf725dd145d39d2b22594a1020ac%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_d512f57cea234d8b8f767c69a3db1105.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203040%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_4fccbe05a1a04e17bc5c74181224bce8%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-12.46044%2C%20130.8410469%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_71f8ecbfa40e41f5aadf784e93dd337d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bfa97f95844d4e0c8ec29ba7670687f3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bfa97f95844d4e0c8ec29ba7670687f3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDarwin%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_71f8ecbfa40e41f5aadf784e93dd337d.setContent%28html_bfa97f95844d4e0c8ec29ba7670687f3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_4fccbe05a1a04e17bc5c74181224bce8.bindPopup%28popup_71f8ecbfa40e41f5aadf784e93dd337d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_4fccbe05a1a04e17bc5c74181224bce8.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%203193%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_ee4d640980d74007b2238ff7136d0c22%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-14.4646157%2C%20132.2635993%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ee425b4f00af449a9363bd77ee159b9f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3aaa6af4d91a4f3ca84d79804e03f00d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3aaa6af4d91a4f3ca84d79804e03f00d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKatherine%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ee425b4f00af449a9363bd77ee159b9f.setContent%28html_3aaa6af4d91a4f3ca84d79804e03f00d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_ee4d640980d74007b2238ff7136d0c22.bindPopup%28popup_ee425b4f00af449a9363bd77ee159b9f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_ee4d640980d74007b2238ff7136d0c22.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%201578%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20marker_757b0dc9d9d34167966a045327ac6f24%20%3D%20L.marker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B-25.3455545%2C%20131.0369615%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28marker_cluster_8d2246f8652b434ab5988ee14dddfd1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_79da05a51b534ab58a1578e5679e11e0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b1668799a8b64f9593a1698e38efd366%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b1668799a8b64f9593a1698e38efd366%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUluru%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_79da05a51b534ab58a1578e5679e11e0.setContent%28html_b1668799a8b64f9593a1698e38efd366%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20marker_757b0dc9d9d34167966a045327ac6f24.bindPopup%28popup_79da05a51b534ab58a1578e5679e11e0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20marker_757b0dc9d9d34167966a045327ac6f24.bindTooltip%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60%3Cdiv%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%201578%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%3C/div%3E%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22sticky%22%3A%20true%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%20%20%20%20%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
# Zeitstempel umformatieren
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace = True)

# Kontrolle
pd.pivot_table(df, index='Location', aggfunc=np.mean).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Location</th>
      <th>Adelaide</th>
      <th>Albany</th>
      <th>Albury</th>
      <th>AliceSprings</th>
      <th>BadgerysCreek</th>
      <th>Ballarat</th>
      <th>Bendigo</th>
      <th>Brisbane</th>
      <th>Cairns</th>
      <th>Canberra</th>
      <th>...</th>
      <th>Townsville</th>
      <th>Tuggeranong</th>
      <th>Uluru</th>
      <th>WaggaWagga</th>
      <th>Walpole</th>
      <th>Watsonia</th>
      <th>Williamtown</th>
      <th>Witchcliffe</th>
      <th>Wollongong</th>
      <th>Woomera</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cloud3pm</th>
      <td>5.000000</td>
      <td>4.852961</td>
      <td>5.190789</td>
      <td>3.458882</td>
      <td>5.000000</td>
      <td>5.842434</td>
      <td>4.892105</td>
      <td>3.880676</td>
      <td>4.487171</td>
      <td>4.863795</td>
      <td>...</td>
      <td>4.102632</td>
      <td>5.000000</td>
      <td>4.956274</td>
      <td>4.241276</td>
      <td>5.000000</td>
      <td>5.263210</td>
      <td>4.595547</td>
      <td>5.000000</td>
      <td>5.398684</td>
      <td>3.353274</td>
    </tr>
    <tr>
      <th>Cloud9am</th>
      <td>5.000000</td>
      <td>5.189145</td>
      <td>5.589803</td>
      <td>3.082566</td>
      <td>5.000000</td>
      <td>5.654605</td>
      <td>4.591118</td>
      <td>3.918885</td>
      <td>4.590132</td>
      <td>4.911234</td>
      <td>...</td>
      <td>4.340132</td>
      <td>5.000000</td>
      <td>5.119772</td>
      <td>3.939847</td>
      <td>5.000000</td>
      <td>5.246261</td>
      <td>4.760053</td>
      <td>5.000000</td>
      <td>5.540789</td>
      <td>2.826520</td>
    </tr>
    <tr>
      <th>Day</th>
      <td>15.719699</td>
      <td>15.715461</td>
      <td>15.715461</td>
      <td>15.715461</td>
      <td>15.712529</td>
      <td>15.715461</td>
      <td>15.715461</td>
      <td>15.719699</td>
      <td>15.715461</td>
      <td>15.717986</td>
      <td>...</td>
      <td>15.715461</td>
      <td>15.710431</td>
      <td>15.692649</td>
      <td>15.712529</td>
      <td>15.698270</td>
      <td>15.712529</td>
      <td>15.712529</td>
      <td>15.712529</td>
      <td>15.715461</td>
      <td>15.712529</td>
    </tr>
    <tr>
      <th>Evaporation</th>
      <td>5.317852</td>
      <td>4.269309</td>
      <td>4.800000</td>
      <td>8.727895</td>
      <td>4.800000</td>
      <td>4.800000</td>
      <td>4.430362</td>
      <td>5.404322</td>
      <td>6.079572</td>
      <td>4.586554</td>
      <td>...</td>
      <td>7.127138</td>
      <td>4.800000</td>
      <td>4.800000</td>
      <td>5.161449</td>
      <td>4.800000</td>
      <td>4.644533</td>
      <td>6.094350</td>
      <td>4.800000</td>
      <td>4.800000</td>
      <td>9.362147</td>
    </tr>
    <tr>
      <th>Humidity3pm</th>
      <td>44.839336</td>
      <td>63.778947</td>
      <td>47.880263</td>
      <td>24.177303</td>
      <td>51.393154</td>
      <td>59.014803</td>
      <td>46.477961</td>
      <td>53.903226</td>
      <td>61.751645</td>
      <td>45.846915</td>
      <td>...</td>
      <td>57.400987</td>
      <td>47.206318</td>
      <td>24.198352</td>
      <td>42.586574</td>
      <td>66.030273</td>
      <td>54.708873</td>
      <td>53.933200</td>
      <td>57.330675</td>
      <td>64.999013</td>
      <td>28.614158</td>
    </tr>
    <tr>
      <th>Humidity9am</th>
      <td>59.717194</td>
      <td>74.727961</td>
      <td>74.058553</td>
      <td>39.720066</td>
      <td>76.772017</td>
      <td>81.736184</td>
      <td>70.544408</td>
      <td>64.051049</td>
      <td>69.936842</td>
      <td>73.159488</td>
      <td>...</td>
      <td>63.954276</td>
      <td>72.808490</td>
      <td>42.923954</td>
      <td>67.859089</td>
      <td>75.991018</td>
      <td>78.344965</td>
      <td>71.769026</td>
      <td>71.255899</td>
      <td>67.684211</td>
      <td>53.200399</td>
    </tr>
    <tr>
      <th>MaxTemp</th>
      <td>12.579142</td>
      <td>12.928783</td>
      <td>9.539539</td>
      <td>13.141908</td>
      <td>11.142107</td>
      <td>7.369474</td>
      <td>8.595954</td>
      <td>16.411337</td>
      <td>21.217434</td>
      <td>6.830908</td>
      <td>...</td>
      <td>20.412336</td>
      <td>7.243699</td>
      <td>14.411977</td>
      <td>9.617182</td>
      <td>11.808882</td>
      <td>10.136225</td>
      <td>12.779661</td>
      <td>10.778730</td>
      <td>14.924967</td>
      <td>13.361914</td>
    </tr>
    <tr>
      <th>MinTemp</th>
      <td>12.579142</td>
      <td>12.928783</td>
      <td>9.539539</td>
      <td>13.141908</td>
      <td>11.142107</td>
      <td>7.369474</td>
      <td>8.595954</td>
      <td>16.411337</td>
      <td>21.217434</td>
      <td>6.830908</td>
      <td>...</td>
      <td>20.412336</td>
      <td>7.243699</td>
      <td>14.411977</td>
      <td>9.617182</td>
      <td>11.808882</td>
      <td>10.136225</td>
      <td>12.779661</td>
      <td>10.778730</td>
      <td>14.924967</td>
      <td>13.361914</td>
    </tr>
    <tr>
      <th>Month</th>
      <td>6.534294</td>
      <td>6.410855</td>
      <td>6.410855</td>
      <td>6.410855</td>
      <td>6.353274</td>
      <td>6.410855</td>
      <td>6.410855</td>
      <td>6.534294</td>
      <td>6.410855</td>
      <td>6.461874</td>
      <td>...</td>
      <td>6.410855</td>
      <td>6.412636</td>
      <td>6.367554</td>
      <td>6.353274</td>
      <td>6.352628</td>
      <td>6.353274</td>
      <td>6.353274</td>
      <td>6.353274</td>
      <td>6.410855</td>
      <td>6.353274</td>
    </tr>
    <tr>
      <th>Pressure3pm</th>
      <td>1016.799749</td>
      <td>1016.486612</td>
      <td>1015.759276</td>
      <td>1012.874211</td>
      <td>1015.610103</td>
      <td>1016.262039</td>
      <td>1015.870921</td>
      <td>1015.099530</td>
      <td>1011.099441</td>
      <td>1016.169645</td>
      <td>...</td>
      <td>1011.905691</td>
      <td>1015.654031</td>
      <td>1013.312548</td>
      <td>1015.859854</td>
      <td>1016.228077</td>
      <td>1016.085045</td>
      <td>1015.917115</td>
      <td>1016.664706</td>
      <td>1016.050461</td>
      <td>1015.966334</td>
    </tr>
    <tr>
      <th>Pressure9am</th>
      <td>1018.761165</td>
      <td>1018.270164</td>
      <td>1018.368355</td>
      <td>1016.685033</td>
      <td>1018.413493</td>
      <td>1017.843158</td>
      <td>1018.024178</td>
      <td>1018.232790</td>
      <td>1014.152467</td>
      <td>1018.895285</td>
      <td>...</td>
      <td>1015.165493</td>
      <td>1018.501152</td>
      <td>1017.104563</td>
      <td>1018.510037</td>
      <td>1017.876214</td>
      <td>1018.029944</td>
      <td>1018.433034</td>
      <td>1018.359721</td>
      <td>1018.140296</td>
      <td>1018.628880</td>
    </tr>
    <tr>
      <th>RainToday</th>
      <td>0.215785</td>
      <td>0.296711</td>
      <td>0.202961</td>
      <td>0.080263</td>
      <td>0.193752</td>
      <td>0.256908</td>
      <td>0.184868</td>
      <td>0.222048</td>
      <td>0.312500</td>
      <td>0.183062</td>
      <td>...</td>
      <td>0.171053</td>
      <td>0.186904</td>
      <td>0.073511</td>
      <td>0.178132</td>
      <td>0.315702</td>
      <td>0.245264</td>
      <td>0.232635</td>
      <td>0.292124</td>
      <td>0.234539</td>
      <td>0.067132</td>
    </tr>
    <tr>
      <th>RainTomorrow</th>
      <td>0.215471</td>
      <td>0.296711</td>
      <td>0.203289</td>
      <td>0.080263</td>
      <td>0.193752</td>
      <td>0.256908</td>
      <td>0.184868</td>
      <td>0.222048</td>
      <td>0.312500</td>
      <td>0.183062</td>
      <td>...</td>
      <td>0.170724</td>
      <td>0.186904</td>
      <td>0.073511</td>
      <td>0.178132</td>
      <td>0.315702</td>
      <td>0.245264</td>
      <td>0.232635</td>
      <td>0.292124</td>
      <td>0.234539</td>
      <td>0.067132</td>
    </tr>
    <tr>
      <th>Rainfall</th>
      <td>1.516317</td>
      <td>2.245987</td>
      <td>1.895855</td>
      <td>0.880526</td>
      <td>2.134064</td>
      <td>1.733158</td>
      <td>1.616184</td>
      <td>3.113373</td>
      <td>5.643816</td>
      <td>1.732596</td>
      <td>...</td>
      <td>3.477566</td>
      <td>2.134847</td>
      <td>0.756527</td>
      <td>1.691193</td>
      <td>2.726015</td>
      <td>1.854636</td>
      <td>3.046893</td>
      <td>2.840811</td>
      <td>3.526316</td>
      <td>0.487471</td>
    </tr>
    <tr>
      <th>Sunshine</th>
      <td>8.038772</td>
      <td>6.955526</td>
      <td>8.400000</td>
      <td>9.379770</td>
      <td>8.400000</td>
      <td>8.400000</td>
      <td>8.400000</td>
      <td>8.086063</td>
      <td>7.701184</td>
      <td>7.958236</td>
      <td>...</td>
      <td>8.449770</td>
      <td>8.400000</td>
      <td>8.400000</td>
      <td>8.227019</td>
      <td>8.400000</td>
      <td>6.381954</td>
      <td>7.838485</td>
      <td>8.400000</td>
      <td>8.400000</td>
      <td>8.832170</td>
    </tr>
    <tr>
      <th>Temp3pm</th>
      <td>21.556874</td>
      <td>19.041217</td>
      <td>21.373059</td>
      <td>28.006217</td>
      <td>22.501894</td>
      <td>16.796875</td>
      <td>20.250296</td>
      <td>24.736298</td>
      <td>27.909112</td>
      <td>19.476193</td>
      <td>...</td>
      <td>27.769934</td>
      <td>19.429648</td>
      <td>29.034347</td>
      <td>21.749950</td>
      <td>18.582102</td>
      <td>19.395181</td>
      <td>22.498504</td>
      <td>19.811233</td>
      <td>19.945987</td>
      <td>25.197807</td>
    </tr>
    <tr>
      <th>Temp9am</th>
      <td>16.928688</td>
      <td>16.242401</td>
      <td>14.368059</td>
      <td>21.335658</td>
      <td>16.573446</td>
      <td>11.692303</td>
      <td>13.848092</td>
      <td>21.834732</td>
      <td>25.817105</td>
      <td>12.698370</td>
      <td>...</td>
      <td>25.634967</td>
      <td>12.601053</td>
      <td>20.971356</td>
      <td>14.925457</td>
      <td>15.933367</td>
      <td>13.779661</td>
      <td>18.077401</td>
      <td>16.543104</td>
      <td>18.175987</td>
      <td>17.924294</td>
    </tr>
    <tr>
      <th>WindDir3pm</th>
      <td>10.290949</td>
      <td>8.266447</td>
      <td>8.758882</td>
      <td>6.361184</td>
      <td>6.558990</td>
      <td>8.441776</td>
      <td>8.409539</td>
      <td>5.125900</td>
      <td>5.342105</td>
      <td>7.931607</td>
      <td>...</td>
      <td>2.675987</td>
      <td>7.755512</td>
      <td>5.748416</td>
      <td>8.832170</td>
      <td>9.357951</td>
      <td>8.535394</td>
      <td>7.502493</td>
      <td>9.535394</td>
      <td>6.907895</td>
      <td>8.556663</td>
    </tr>
    <tr>
      <th>WindDir9am</th>
      <td>6.513937</td>
      <td>7.139474</td>
      <td>6.508553</td>
      <td>4.864474</td>
      <td>8.200399</td>
      <td>7.227961</td>
      <td>7.487500</td>
      <td>9.157845</td>
      <td>8.608882</td>
      <td>6.588475</td>
      <td>...</td>
      <td>6.392434</td>
      <td>6.990457</td>
      <td>4.005703</td>
      <td>4.232968</td>
      <td>6.636394</td>
      <td>6.517115</td>
      <td>9.095048</td>
      <td>6.724826</td>
      <td>8.368421</td>
      <td>6.815221</td>
    </tr>
    <tr>
      <th>WindGustDir</th>
      <td>9.407767</td>
      <td>13.000000</td>
      <td>8.930592</td>
      <td>5.479276</td>
      <td>7.495846</td>
      <td>8.235855</td>
      <td>8.972697</td>
      <td>5.940182</td>
      <td>6.462829</td>
      <td>7.273574</td>
      <td>...</td>
      <td>3.218750</td>
      <td>7.555117</td>
      <td>5.471483</td>
      <td>7.935527</td>
      <td>8.851630</td>
      <td>8.604187</td>
      <td>8.739448</td>
      <td>9.206381</td>
      <td>8.268092</td>
      <td>8.488202</td>
    </tr>
    <tr>
      <th>WindGustSpeed</th>
      <td>36.519574</td>
      <td>39.000000</td>
      <td>33.040789</td>
      <td>40.505592</td>
      <td>33.736790</td>
      <td>44.921711</td>
      <td>38.851974</td>
      <td>28.351394</td>
      <td>38.037171</td>
      <td>39.959837</td>
      <td>...</td>
      <td>38.841776</td>
      <td>35.268180</td>
      <td>41.190748</td>
      <td>36.669990</td>
      <td>39.795409</td>
      <td>38.070788</td>
      <td>41.557660</td>
      <td>40.073114</td>
      <td>45.585197</td>
      <td>44.089731</td>
    </tr>
    <tr>
      <th>WindSpeed3pm</th>
      <td>15.476981</td>
      <td>18.995066</td>
      <td>14.399671</td>
      <td>18.101645</td>
      <td>14.085078</td>
      <td>22.761842</td>
      <td>17.161513</td>
      <td>11.018791</td>
      <td>21.910855</td>
      <td>19.119616</td>
      <td>...</td>
      <td>24.250000</td>
      <td>14.921685</td>
      <td>17.065906</td>
      <td>16.219010</td>
      <td>17.244511</td>
      <td>15.185111</td>
      <td>22.229312</td>
      <td>19.164506</td>
      <td>21.898355</td>
      <td>20.602526</td>
    </tr>
    <tr>
      <th>WindSpeed9am</th>
      <td>9.954901</td>
      <td>12.513158</td>
      <td>8.225329</td>
      <td>14.726316</td>
      <td>8.131938</td>
      <td>19.958553</td>
      <td>12.936184</td>
      <td>6.946132</td>
      <td>15.901974</td>
      <td>10.580326</td>
      <td>...</td>
      <td>15.225987</td>
      <td>7.693649</td>
      <td>17.511407</td>
      <td>12.855766</td>
      <td>13.905855</td>
      <td>9.911266</td>
      <td>16.454968</td>
      <td>13.964440</td>
      <td>16.580263</td>
      <td>19.993353</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>2012.504228</td>
      <td>2012.730921</td>
      <td>2012.730921</td>
      <td>2012.730921</td>
      <td>2012.779661</td>
      <td>2012.730921</td>
      <td>2012.730921</td>
      <td>2012.504228</td>
      <td>2012.730921</td>
      <td>2012.167928</td>
      <td>...</td>
      <td>2012.730921</td>
      <td>2012.731162</td>
      <td>2014.835868</td>
      <td>2012.779661</td>
      <td>2012.781437</td>
      <td>2012.779661</td>
      <td>2012.779661</td>
      <td>2012.779661</td>
      <td>2012.730921</td>
      <td>2012.779661</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>-34.928497</td>
      <td>-35.024782</td>
      <td>-36.073730</td>
      <td>-23.698042</td>
      <td>-33.887421</td>
      <td>-37.562303</td>
      <td>-36.758877</td>
      <td>-27.468968</td>
      <td>-16.920666</td>
      <td>-35.297591</td>
      <td>...</td>
      <td>-19.256939</td>
      <td>-35.420977</td>
      <td>-25.345554</td>
      <td>-35.115000</td>
      <td>-34.977680</td>
      <td>-37.711002</td>
      <td>-32.815000</td>
      <td>-34.026335</td>
      <td>-34.424394</td>
      <td>-31.199914</td>
    </tr>
    <tr>
      <th>lon</th>
      <td>138.600739</td>
      <td>117.883608</td>
      <td>146.913544</td>
      <td>133.880753</td>
      <td>150.740509</td>
      <td>143.860565</td>
      <td>144.282593</td>
      <td>153.023499</td>
      <td>145.772185</td>
      <td>149.101268</td>
      <td>...</td>
      <td>146.823954</td>
      <td>149.092134</td>
      <td>131.036961</td>
      <td>147.367778</td>
      <td>116.731006</td>
      <td>145.083635</td>
      <td>151.842778</td>
      <td>115.100477</td>
      <td>150.893850</td>
      <td>136.825353</td>
    </tr>
  </tbody>
</table>
<p>26 rows × 49 columns</p>
</div>



### Deklarieren des Targets



```python
X = df.drop(['RainTomorrow', 'Location'], axis=1) # Alle features
y = df['RainTomorrow'] # Target
```

"RainTomorrow" ist unser Target. Für alle Features werden die Werte von "RainTomorrow" und "Location" eliminiert, da sie weiter nicht benötigt werden.

### Datensatz in training und test aufteilen


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Trainingsdaten vs Testdaten
X_train.shape, X_test.shape
```




    ((116368, 25), (29092, 25))



Die in der EDA auffällig Schiefen Kategorien werden nun korrigiert. Dazu werden alle Ausreisser mit dem 75%-Quantil ersetzt.


```python
#2.0, 14.6, 37.0, 40.5
def max_value(df1, variable, top):
    return np.where(df1[variable]>top, top, df1[variable]);

for df in [X_train, X_test]:
    df['Rainfall'] = max_value(df, 'Rainfall', 2.0)
    df['Evaporation'] = max_value(df, 'Evaporation', 14.6)
    df['WindSpeed9am'] = max_value(df, 'WindSpeed9am', 37.0)
    df['WindSpeed3pm'] = max_value(df, 'WindSpeed3pm', 40.5)
```

    C:\Users\alexk\AppData\Local\Temp/ipykernel_9460/1222557316.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Rainfall'] = max_value(df, 'Rainfall', 2.0)
    C:\Users\alexk\AppData\Local\Temp/ipykernel_9460/1222557316.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Evaporation'] = max_value(df, 'Evaporation', 14.6)
    C:\Users\alexk\AppData\Local\Temp/ipykernel_9460/1222557316.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['WindSpeed9am'] = max_value(df, 'WindSpeed9am', 37.0)
    C:\Users\alexk\AppData\Local\Temp/ipykernel_9460/1222557316.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['WindSpeed3pm'] = max_value(df, 'WindSpeed3pm', 40.5)
    


```python
fig=plt.figure(figsize=(20,10),facecolor='white')
gs=fig.add_gridspec(4,1)

# Ausreisser in 'Rainfall' finden
ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(0,-0.55, "Boxplot Regenmenge",fontsize=23,fontweight='bold', fontfamily='monospace')
sns.boxplot(x = 'Rainfall', data = df, palette = palette1);

# Ausreisser in 'Evaporation' finden
ax[0]=fig.add_subplot(gs[1,0])
ax[0].text(0,-0.55, "Boxplot Verdunstung",fontsize=23,fontweight='bold', fontfamily='monospace')
sns.boxplot(x = 'Evaporation', data = df, palette = palette1);

# Ausreisser in 'WindSpeed9am' finden
ax[0]=fig.add_subplot(gs[2,0])
ax[0].text(0,-0.55, "Boxplot Windgeschwindigkeit 9 Uhr",fontsize=23,fontweight='bold', fontfamily='monospace')
sns.boxplot(x = 'WindSpeed9am', data = df, palette = palette1);

# Ausreisser in 'WindSpeed3pm' finden
ax[0]=fig.add_subplot(gs[3,0])
ax[0].text(0,-0.55, "Boxplot Windgeschwindigkeit 15 Uhr",fontsize=23,fontweight='bold', fontfamily='monospace')
sns.boxplot(x = 'WindSpeed3pm', data = df, palette = palette1);

plt.tight_layout();
```


    
![png](output_66_0.png)
    


Die Boxplots sind jetzt angenehmer zu interpretieren.

***
***
## No Free Lunch <a name="nfl"></a>
In den folgenden Abschnitten werden verschiedene Learner getestet und miteinander verglichen. Es soll der beste Learner für die Auswertung unserer Daten gefunden werden.


### k-Nearest-Neighbors <a name="knn"></a>
Damit unser kNN-Algorithmus optimal funktioniert, werden alle Daten mit dem Standardscaler standardisiert.


```python
# Zuerst Daten skalieren
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Ein ideales k soll gefunden werden. Dafür wird k von 1 bis 40 iteriert. Die Ergebnisse des idealen k werden in Diagrammen dargestellt.


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

accuracyList = []; precisionList=[]; recallList=[]; foneList=[];
# Calculating error for K values between 1 and 40

best_accuracy = 0;
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred_i = knn.predict(X_test)
    accuracyList.append(accuracy_score(y_test, y_pred_i))
    foneList.append(f1_score(y_test, y_pred_i, average='weighted'))
    precisionList.append(precision_score(y_test, y_pred_i, average='weighted'))
    recallList.append(recall_score(y_test, y_pred_i, average='weighted'))
    if accuracyList[i-1] > best_accuracy:
        y_pred = y_pred_i
        best_accuracy = accuracyList[i-1]
```

Der kNN funktionert am besten im niedrigdimensionalen Raum. Mit unseren 25 scheint der Algortihmus ein passender zu sein, jedoch ist unsere Datenmenge hoch, was zu einem hohen Rechenaufwand führen wird.


```python
k_values = range(1, 40)

fig=plt.figure(figsize=(20,10),facecolor='white')
gs=fig.add_gridspec(2,2)
ax=[None for i in range(4)]

# Accuracy
max_val = max(accuracyList)
max_idx = accuracyList.index(max_val)
ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(0, max_val*1.005,'Accuracy ' + str(max_val) + ' bei k = ' + str(max_idx) , fontsize=23, fontweight='bold', fontfamily='monospace')
sns.lineplot(x=k_values, y=accuracyList, ax=ax[0], palette=palette1, color="gold", linewidth=1)
# F1-Score
max_val = max(foneList)
max_idx = foneList.index(max_val)
ax[1]=fig.add_subplot(gs[0,1])
ax[1].text(0, max_val*1.005,'F1-Score ' + str(max_val) + ' bei k = ' + str(max_idx), fontsize=23, fontweight='bold', fontfamily='monospace')
sns.lineplot(x=k_values, y=foneList, ax=ax[1], palette=palette1, color="gold", linewidth=1)
# Recall
max_val = max(recallList)
max_idx = recallList.index(max_val)
ax[2]=fig.add_subplot(gs[1,0])
ax[2].text(0, max_val*1.005,'Recall ' + str(max_val) + ' bei k = ' + str(max_idx), fontsize=23, fontweight='bold', fontfamily='monospace')
sns.lineplot(x=k_values, y=recallList, ax=ax[2], palette=palette1, color="gold", linewidth=1)
# Precision
max_val = max(precisionList)
max_idx = precisionList.index(max_val)
ax[3]=fig.add_subplot(gs[1,1])
ax[3].text(0, max_val*1.005,'Precision ' + str(max_val) + ' bei k = ' + str(max_idx), fontsize=23, fontweight='bold', fontfamily='monospace')
sns.lineplot(x=k_values, y=precisionList, ax=ax[3], palette=palette1, color="gold", linewidth=1)

for i in range(4):
    ax[i].set_ylabel('')
    ax[i].grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)
        
plt.tight_layout()
plt.savefig('plt/knn_scores.png')
```


    
![png](output_73_0.png)
    


Die oben Dargestellten Diagramme visualisieren die Qualtitätsmasse (Accuracy, F1-Score, Recall, Precision) für die Klassifizierung bei verschiedenen Werten für k. Es ergibt sich ein optimales k von 26. Um den kNN später mit anderen Learnern vergleichen zu können, erstellen wir eine Konfusionsmatrix zu den Ergebnissen unserer Test-Daten.

Konfusionsmatrix:
   * oben links: right negative - richtige Vorhersage, kein Regen
   * unten rechts: right positive - richtige Vorhersage, Regen
   * oben rechts: false positive - falsche Vorhersage, Regen vorhergesagt
   * unten links: false negative - falsche Vorhersage, kein Regen vorhergesagt


```python
from sklearn.metrics import classification_report, confusion_matrix

plt.figure(figsize=(16,12))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap='YlOrRd', fmt='5.0f');
plt.savefig('plt/knn_confusion-matrix.png')
# [True Negative   False Positive]
# [False Negative  True Positive]
```


    
![png](output_75_0.png)
    



```python
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline


pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=26))

train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_knn, X=X_train, y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, n_jobs=-1)
```


```python
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

if max(train_mean) > max(test_mean):
    max_val = max(train_mean)
    max_idx = np.argmax(train_mean, axis=0)
else:
    max_val = max(test_mean)
    max_idx = np.argmax(test_mean, axis=0)


max_idx = train_sizes[max_idx]
fig=plt.figure(figsize=(20,10),facecolor='white')
gs=fig.add_gridspec(1,1)
ax=[None for i in range(1)]

# Learning curve bei train:test split
ax[0]=fig.add_subplot(gs[0,0])
ax[0].text(0, max_val*1.005,'Learningcurve ' + str(max_val) + ' bei split ' + str(max_idx) + ':'+str(len(X_train)-max_idx), fontsize=23, fontweight='bold', fontfamily='monospace')
plt.plot(train_sizes, train_mean,
    color='gold', marker='o',
    markersize=5,
    label='Korrektklassifizierungsrate Training')
plt.fill_between(train_sizes,
    train_mean + train_std,
    train_mean - train_std,
    alpha=0.15, color='gold')
plt.plot(train_sizes, test_mean,
    color='wheat', linestyle='--',
    marker='s', markersize=5,
    label='Korrektklassifizierungsrate Validierung')
plt.fill_between(train_sizes,
    test_mean + test_std,
    test_mean - test_std,
    alpha=0.15, color='wheat')

for i in range(1):
    ax[i].set_ylabel('')
    ax[i].grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)

plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plt/knn_learncurve.png')
```


    
![png](output_77_0.png)
    


Die Learningkurven (oben) gehen weit auseinander, was auf eine hohe Varianz schliessen lässt. Zur Behebung könnten neue Trainingsdaten beschafft oder eine Regularisierung eingeführt werden. Zuerst wollen wir aber die Performance anderer Learner begutachten.


```python
def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='gold', label='ROC')
    plt.plot([0, 1], [0, 1], color='wheat', linestyle='--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()
```


```python
import time
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, plot_confusion_matrix, roc_curve, classification_report
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    
    if verbose == False:
        model.fit(X_train, y_train, verbose = 0)
    else:
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    fone = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("F1-Score = {}".format(fone))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plot_roc_curve(fper, tper)
    
    plot_confusion_matrix(model, X_test, y_test, cmap='YlOrRd')
    
    return model, accuracy, roc_auc, fone, precision, recall, time_taken
```

### Optimierter kNN Algorithmus <a name="opt_knn"></a>


```python
from sklearn.neighbors import KNeighborsClassifier

params_lr = {'n_neighbors': 26, 'n_jobs':-1}

model_knn = KNeighborsClassifier(**params_lr)

model_knn, accuracy_knn, roc_auc_knn, fone_knn, precision_knn, recall_knn, time_knn = run_model(model_knn, X_train, y_train, X_test, y_test)
```

    Accuracy = 0.8471401072459783
    ROC Area under Curve = 0.6928097422433414
    F1-Score = 0.8289086105913285
    Precision = 0.8391139792437127
    Recall = 0.8471401072459783
    Time taken = 83.7783465385437
                  precision    recall  f1-score   support
    
               0    0.85656   0.96638   0.90816     22752
               1    0.77651   0.41924   0.54450      6340
    
        accuracy                        0.84714     29092
       macro avg    0.81654   0.69281   0.72633     29092
    weighted avg    0.83911   0.84714   0.82891     29092
    
    


    
![png](output_82_1.png)
    



    
![png](output_82_2.png)
    


### Logistische Regression <a name="logreg"></a>
Als zweiter Learner implementieren wir die logistische Regression. Genau wie beim kNN werten wir die Scores (Qualitätsmasse) aus und visualisieren die Ergebnisse TP, FP, TN, FN in einer Konfusionsmatrix.


```python
from sklearn.linear_model import LogisticRegression

params_lr = {'penalty': 'l2', 'solver':'lbfgs'}

model_lr = LogisticRegression(**params_lr)
model_lr, accuracy_lr, roc_auc_lr, fone_lr, precision_lr, recall_lr, time_lr = run_model(model_lr, X_train, y_train, X_test, y_test)
```

    Accuracy = 0.8481025711535818
    ROC Area under Curve = 0.7193661208277318
    F1-Score = 0.8368485575362024
    Precision = 0.8378417231291334
    Recall = 0.8481025711535818
    Time taken = 0.31665921211242676
                  precision    recall  f1-score   support
    
               0    0.86984   0.94757   0.90704     22752
               1    0.72301   0.49117   0.58495      6340
    
        accuracy                        0.84810     29092
       macro avg    0.79642   0.71937   0.74600     29092
    weighted avg    0.83784   0.84810   0.83685     29092
    
    


    
![png](output_84_1.png)
    



    
![png](output_84_2.png)
    


Die Werte weichen nur leicht von denen des kNN ab, die Rechenzeit ist hier aber deutlich tiefer. Die ROC-Kurve verläuft nahezu identisch mit der des kNN, was eine Modellbewertung zwischen diesen beiden Learnern schwierig macht.

### Decision Trees <a name="dectree"></a>


```python
from sklearn.tree import DecisionTreeClassifier

params_dt = {'max_depth': 16,
             'max_features': "sqrt"}

model_dt = DecisionTreeClassifier(**params_dt)
model_dt, accuracy_dt, roc_auc_dt, fone_dt, precision_dt, recall_dt, time_dt = run_model(model_dt, X_train, y_train, X_test, y_test)
```

    Accuracy = 0.8195723910353362
    ROC Area under Curve = 0.7011828543793565
    F1-Score = 0.8124535072358451
    Precision = 0.8088713755010292
    Recall = 0.8195723910353362
    Time taken = 0.29729151725769043
                  precision    recall  f1-score   support
    
               0    0.86536   0.91104   0.88761     22752
               1    0.60615   0.49132   0.54273      6340
    
        accuracy                        0.81957     29092
       macro avg    0.73576   0.70118   0.71517     29092
    weighted avg    0.80887   0.81957   0.81245     29092
    
    


    
![png](output_87_1.png)
    



    
![png](output_87_2.png)
    


Bei den Decision Trees ist die Genauigkeit leicht tiefer als bei den ersten beiden Learnern. Auch hier zeigt sich anhand der Rechenzeit den deutlich geringeren Berechnungsaufwand als beim kNN.

### Random Forest <a name="randfor"></a>


```python
from sklearn.ensemble import RandomForestClassifier

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf, fone_rf, precision_rf, recall_rf, time_rf = run_model(model_rf, X_train, y_train, X_test, y_test)
```

    Accuracy = 0.8610958339062286
    ROC Area under Curve = 0.7346134509754334
    F1-Score = 0.8499370827980688
    Precision = 0.8536489043537054
    Recall = 0.8610958339062286
    Time taken = 18.639240026474
                  precision    recall  f1-score   support
    
               0    0.87544   0.95882   0.91523     22752
               1    0.77546   0.51041   0.61562      6340
    
        accuracy                        0.86110     29092
       macro avg    0.82545   0.73461   0.76543     29092
    weighted avg    0.85365   0.86110   0.84994     29092
    
    


    
![png](output_90_1.png)
    



    
![png](output_90_2.png)
    


Was Genauigkeit betrifft, ist der Random Forest der beste der verwendeten Learner. Wie zu erwarten war, ist die Rechenzeit höher als bei einfachen Decision Trees oder logistischer Regression, aber immernoch deutlich tiefer, als beim kNN. Aufgrund der Genauigkeit werden wir versuchen, diesen Learner zu boosten.

### Dimensionsreduktion am optimierten kNN Algorithmus<a name="dimred"></a>


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Dimensionsreduktion mit PCA
pca = PCA()
pca.fit(X_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
explainedVariance=0.95
d = np.argmax(cumsum >= explainedVariance) + 1

print("Dimension des Unterraumes: %i" % d)
```

    Dimension des Unterraumes: 17
    


```python
pca = PCA(n_components=explainedVariance)
X_train_PCA = pca.fit_transform(X_train)
```


```python
pca.n_components_
```




    17




```python
np.sum(pca.explained_variance_ratio_)
```




    0.9521831740899325




```python
pca = PCA(n_components = 17)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.fit_transform(X_test)

params_lr = {'n_neighbors': 26, 'n_jobs':-1}

model_knn_pca = KNeighborsClassifier(**params_lr)

model_knn_pca, accuracy_knn_pca, roc_auc_knn_pca, fone_knn_pca, precision_knn_pca, recall_knn_pca, time_knn_pca = run_model(model_knn_pca, X_train_PCA, y_train, X_test_PCA, y_test)
```

    Accuracy = 0.8220472982263165
    ROC Area under Curve = 0.6490057240435341
    F1-Score = 0.7982585875838469
    Precision = 0.8050803646119159
    Recall = 0.8220472982263165
    Time taken = 84.07404732704163
                  precision    recall  f1-score   support
    
               0    0.83909   0.95574   0.89362     22752
               1    0.68303   0.34227   0.45603      6340
    
        accuracy                        0.82205     29092
       macro avg    0.76106   0.64901   0.67483     29092
    weighted avg    0.80508   0.82205   0.79826     29092
    
    


    
![png](output_97_1.png)
    



    
![png](output_97_2.png)
    


Mit einer Dimensionsreduktion erhofften wir uns geringere Rechenzeiten beim kNN. Die damit verbundene Verringerung der Genauigkeit wird in Kauf genommen. Das Ergebnis zeigt aber, dass eine Optimierung in diese Richtung für unser Problem nicht infrage kommt.

***
***
## Validierung <a name="val"></a>
Kreuzvalidierte Performance dieser Lerner und Hyperparameter-Tuning

### Vergleich der Modelle


```python
accuracy_scores = [accuracy_knn, accuracy_knn_pca, accuracy_lr, accuracy_dt, accuracy_rf]
roc_auc_scores = [roc_auc_knn, roc_auc_knn_pca, roc_auc_lr, roc_auc_dt, roc_auc_rf]
fone_scores = [fone_knn, fone_knn_pca, fone_lr, fone_dt, fone_rf]
tt = [time_knn, time_knn_pca, time_lr, time_dt, time_rf]

model_data = {'Model': ['kNN', 'kNN_PCA', 'Logistic Regression', 'Decision Tree', 'Random Forest'],
              'Accuracy': accuracy_scores,
              'ROC_AUC': roc_auc_scores,
              'F1 Score': fone_scores,
              'Time taken': tt}
data = pd.DataFrame(model_data)

fig, ax1 = plt.subplots(figsize=(12,10))
ax1.set_title('Model Comparison: Accuracy and Time taken for execution', fontsize=13)
ax1.set_xlabel('Model', fontsize=13)
ax1.set_ylabel('Time taken', fontsize=13)
ax2 = sns.barplot(x='Model', y='Time taken', data = data, palette=palette1)
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', fontsize=13, color='tab:red')
ax2 = sns.lineplot(x='Model', y='Accuracy', data = data, sort=False, color='tab:red')
ax2.tick_params(axis='y', color='tab:red')
ax2.set(ylim=(0.8, 0.9));
```


    
![png](output_101_0.png)
    


***
***
### Boosting von Random Forest <a name="boosting"></a>


```python
# Boosting des Random Forest Classifier
from sklearn.ensemble import AdaBoostClassifier

params_ada = {'base_estimator': RandomForestClassifier(**params_rf),
              'n_estimators': 500,
              'algorithm': 'SAMME.R',
              'learning_rate': 1    
             }

model_ada = AdaBoostClassifier(**params_ada)
model_ada, accuracy_ada, roc_auc_ada, fone_ada, precision_ada, recall_ada, time_ada = run_model(model_ada, X_train, y_train, X_test, y_test)
```

    Accuracy = 0.8663550116870617
    ROC Area under Curve = 0.7404788763327077
    F1-Score = 0.8552041925148892
    Precision = 0.8602878607646162
    Recall = 0.8663550116870617
    Time taken = 621.21639752388
                  precision    recall  f1-score   support
    
               0    0.87752   0.96361   0.91855     22752
               1    0.79844   0.51735   0.62787      6340
    
        accuracy                        0.86636     29092
       macro avg    0.83798   0.74048   0.77321     29092
    weighted avg    0.86029   0.86636   0.85520     29092
    
    


    
![png](output_103_1.png)
    



    
![png](output_103_2.png)
    



```python
!pip install catboost
!pip install xgboost
```

    Requirement already satisfied: catboost in c:\users\alexk\anaconda3\lib\site-packages (1.0.3)
    Requirement already satisfied: scipy in c:\users\alexk\anaconda3\lib\site-packages (from catboost) (1.7.1)
    Requirement already satisfied: numpy>=1.16.0 in c:\users\alexk\anaconda3\lib\site-packages (from catboost) (1.20.3)
    Requirement already satisfied: pandas>=0.24.0 in c:\users\alexk\anaconda3\lib\site-packages (from catboost) (1.3.4)
    Requirement already satisfied: matplotlib in c:\users\alexk\anaconda3\lib\site-packages (from catboost) (3.4.3)
    Requirement already satisfied: plotly in c:\users\alexk\anaconda3\lib\site-packages (from catboost) (5.5.0)
    Requirement already satisfied: graphviz in c:\users\alexk\anaconda3\lib\site-packages (from catboost) (0.19.1)
    Requirement already satisfied: six in c:\users\alexk\anaconda3\lib\site-packages (from catboost) (1.16.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\users\alexk\anaconda3\lib\site-packages (from pandas>=0.24.0->catboost) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in c:\users\alexk\anaconda3\lib\site-packages (from pandas>=0.24.0->catboost) (2021.3)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\alexk\anaconda3\lib\site-packages (from matplotlib->catboost) (3.0.4)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\alexk\anaconda3\lib\site-packages (from matplotlib->catboost) (1.3.1)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\alexk\anaconda3\lib\site-packages (from matplotlib->catboost) (8.4.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\alexk\anaconda3\lib\site-packages (from matplotlib->catboost) (0.10.0)
    Requirement already satisfied: tenacity>=6.2.0 in c:\users\alexk\anaconda3\lib\site-packages (from plotly->catboost) (8.0.1)
    Requirement already satisfied: xgboost in c:\users\alexk\anaconda3\lib\site-packages (1.5.1)
    Requirement already satisfied: numpy in c:\users\alexk\anaconda3\lib\site-packages (from xgboost) (1.20.3)
    Requirement already satisfied: scipy in c:\users\alexk\anaconda3\lib\site-packages (from xgboost) (1.7.1)
    


```python
import catboost as cb
params_cb ={'iterations': 50,
            'max_depth': 16}

model_cb = cb.CatBoostClassifier(**params_cb);
model_cb, accuracy_cb, roc_auc_cb, fone_cb, precision_cb, recall_cb, time_cb = run_model(model_cb, X_train, y_train, X_test, y_test, verbose=False);
```

    Accuracy = 0.8493743984600578
    ROC Area under Curve = 0.7377577095174079
    F1-Score = 0.841963845371988
    Precision = 0.8404305978093283
    Recall = 0.8493743984600578
    Time taken = 153.00309205055237
                  precision    recall  f1-score   support
    
               0    0.87948   0.93561   0.90668     22752
               1    0.70029   0.53991   0.60973      6340
    
        accuracy                        0.84937     29092
       macro avg    0.78988   0.73776   0.75820     29092
    weighted avg    0.84043   0.84937   0.84196     29092
    
    


    
![png](output_105_1.png)
    



    
![png](output_105_2.png)
    



```python
import xgboost as xgb
params_xgb ={'n_estimators': 500,
             'max_depth': 16,
             'use_label_encoder': False}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, fone_xgb, precision_xgb, recall_xgb, time_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)
```

    [20:57:49] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.1/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    Accuracy = 0.8681768183693112
    ROC Area under Curve = 0.7675846571674497
    F1-Score = 0.8621037384858815
    Precision = 0.8615743226466247
    Recall = 0.8681768183693112
    Time taken = 73.17169570922852
                  precision    recall  f1-score   support
    
               0    0.89206   0.94589   0.91819     22752
               1    0.75216   0.58927   0.66083      6340
    
        accuracy                        0.86818     29092
       macro avg    0.82211   0.76758   0.78951     29092
    weighted avg    0.86157   0.86818   0.86210     29092
    
    


    
![png](output_106_1.png)
    



    
![png](output_106_2.png)
    


### Vergleich der Modelle


```python
accuracy_scores = [accuracy_ada, accuracy_cb, accuracy_xgb, accuracy_rf]
roc_auc_scores = [roc_auc_ada, roc_auc_cb, roc_auc_xgb, roc_auc_rf]
fone_scores = [fone_ada, fone_cb, fone_xgb, fone_rf]
tt = [time_ada, time_cb, time_xgb, time_rf]

model_data = {'Model': ['AdaBoost', 'CatBoost', 'XGBoost', 'Random Forest'],
              'Accuracy': accuracy_scores,
              'ROC_AUC': roc_auc_scores,
              'F1 Score': fone_scores,
              'Time taken': tt}
data = pd.DataFrame(model_data)

fig, ax1 = plt.subplots(figsize=(12,10))
ax1.set_title('Model Comparison: Accuracy and Time taken for execution', fontsize=13)
ax1.set_xlabel('Model', fontsize=13)
ax1.set_ylabel('Time taken', fontsize=13)
ax2 = sns.barplot(x='Model', y='Time taken', data = data, palette=palette1)
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', fontsize=13, color='tab:red')
ax2 = sns.lineplot(x='Model', y='Accuracy', data = data, sort=False, color='tab:red')
ax2.tick_params(axis='y', color='tab:red')
ax2.set(ylim=(0.8, 0.9));
```


    
![png](output_108_0.png)
    



```python
fig, ax3 = plt.subplots(figsize=(12,10))
ax3.set_title('Model Comparison: Area under ROC and F1 Score', fontsize=13)
ax3.set_xlabel('Model', fontsize=13)
ax3.set_ylabel('ROC area under curve', fontsize=13, color='tab:blue')
ax4 = sns.barplot(x='Model', y='ROC_AUC', data = data, palette=palette1)
ax3.tick_params(axis='y')
ax4 = ax3.twinx()
ax4.set_ylabel('F1 Score', fontsize=13, color='tab:red')
ax4 = sns.lineplot(x='Model', y='F1 Score', data = data, sort=False, color='tab:red')
ax4.tick_params(axis='y', color='tab:red')
ax4.set(ylim=(0.8, 0.9));
plt.show()
```


    
![png](output_109_0.png)
    


***
***
## Entscheid <a name="entscheid"></a>
Von allen Learnern hat der Random Forest am besten abgeschlossen. Wir haben diesen Learner mit drei verschiedenen Boostern optimiert. Sowie bei Rechenzeit und Genauigkeit, als auch bei F1-Score und ROC-AUC hat sich der XGBoost als der beste herausgestellt.

***
***
## Schlussfolgerung und Ausblick <a name="ausblick"></a>
Spannend wäre zu beobachten, wie sich die Ergebnisse verändern, wenn die Data Preparation anders gemacht wird. Beispielsweise könnten fehlende Daten nicht einfach durch den Median ersetzt werden, sondern könnten aus anderen Datensätzen Werte entnommen werden, die für unseren Datensatz zutreffen. Es müsste dann auch untersucht werden, ob der Random Forest sich immernoch als der ideale Learner beweisen kann. Auf genauere Data Preparation könnten auch genauere Ergebnisse folgen. Bestehende Learner können alle noch optimiert werden. Sind die Grenzen des maschinellen Lernens erreicht, so muss auf neuronale Netzwerke zurückgegriffen werden.

***
***
## Referenzen <a name="ref"></a>
['Rain in Australia' Datensatz](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)<br>
[Erkärung der Merkmale](http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml)

***
***
[Zum Inhaltsverzeichnis](#toc)
