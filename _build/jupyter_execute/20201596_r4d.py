#!/usr/bin/env python
# coding: utf-8

# # 4. Modelo IS LM

# In[1]:


# María Fernanda Carrillo (20201596)


# ## Parte I: Reporte

# La pregunta de investigación que el autor se realiza es el análisis de la política monetaria aplicada en el año 2008-2009 para hacer frente a las consecuencias que había dejado la crisis internacional en la economía estadounidense; la cual curiosamente tenía componentes no convencionales, que expandían el modelo tradicional de demanda y oferta agregada. A cargo del Sistema de Reserva Federal, es que se menciona que se toman medidas poco habituales dentro del modelo de IS-LM, pues lo que efectuaron fue una administración de una tasa interés de corto plazo, así como la implementación de una participación directa en el mercado de bonos de largo plazo a los mercados existentes. 
# 
# Una de las debilidades de este modelo que el autor plantea para el análisis de este modelo es que si bien ha resultado efectivo en este caso es que para demostrar aún más la efectividad de estas decisiones y del modelo keynesiano que se ha aplicado para poder hacer frente a las crisis sería adecuado el extrapolarlo a otros ejemplos. Más aún para poder afirmar con mucha más base el hecho de que en efecto, los modelos viejos aún tienen prevalencia por sobre los nuevos incluso. 
# No obstante, tiene muchísimas fortalezas, en tanto presta atención a muchas variables que antaño no eran consideradas, así enriqueciendo a los modelos antiguos y haciéndolos mucho más útiles y realistas al momento de analizar fenómenos actuales. Por ejemplo, presta atención a la expectativa que tiene la población y, cuando las políticas económicas son anticipadas es que pierden efectividad. 
# 
# Asimismo, demuestra en el análisis que, como se mencionó, el modelo keynesiano aún está vigente (lo revaloriza), sin obviar las acotaciones y variables que el autor agrega, es que contribuye al conocimiento económico en tanto demuestra que aún son utilizables. Y que, a comparación de teorizaciones contemporáneas es que aun puede prevalecer lo antiguo, que en ocasiones la intervención estatal puede ser la solución a graves declives económicos y que medidas puntuales significan la reactivación de una economía que ha pasado por lo peor.  Pues la intervención en los mercados de deuda de largo plazo, había teorizado años atrás, es una necesidad cuando la economía ingresa a una situación de una tasa de interés tan baja que se prefiere el efectivo a la adquisición de deuda con baja rentabilidad, lo que llamó una ‘trampa de liquidez’. Aún así quedan cosas que mejorar en estos antiguos modelos, a los cuales se les pueden acotar algunos preceptos o modificar levemente: como la distinción entre la tasa de interés de corto y largo plazo, pues inicialmente solo se concebía una sola tasa. 
# 
# Y, el efecto que tienen estas es sumamente diferentes entre sí. Pues, por ejemplo, es que a la compra de bonos de largo plazo es que solo reduce la tasa de interés de largo plazo, mientras que aún se tiene incertidumbre sobre el corto plazo.  No obstante, prevalece la idea de que no importa el tiempo, es que aún Keynes y sus postulados son vigentes al día de hoy.  
# 
# Del mismo modo es que la idea de la importancia y prevalencia que el modelo keynesiano tiene aún actualmente es avalado por el autor Torrero, en el artículo ¿Tenemos algo que aprender de Keynes ahora? En el cual recalca la teoría keynesiana, y cuan relevante es hasta el día de hoy, fundamentalmente mirándolo por el lado de la incertidumbre que existe y la necesidad que existe en cuanto las instituciones han de adecuarse a las circunstancias del mercado, tal y como se presenta en el texto de Mendoza. 
# 
# Fuente: https://ebuah.uah.es/dspace/bitstream/handle/10017/27557/tenemos_torrero_IAESDT_2016_N07.pdf?sequence=1&isAllowed=y

# In[2]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
#from causalgraphicalmodels import CausalGraphicalModel


# ## Parte II: Código

# Trabajo conjunto con Angela Rodriguez

# ### A partir del siguiente sistema de ecuaciones que representan el modelo IS-LM

# #### Ecuaciones de Ingreso (Ye) y la tasa de interés (re) de equilibrio

# La ecuación de la Curva IS es: 
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1-(b-m(1-t)}{h}(Y)$$

# De manera más simplifica sería:
# 
# $$r = \frac{B_0}{h} - \frac{B_1}{h}(Y)$$
# 
# - Donde $ B_0 = C_o + I_o + G_o + X_o $ y $ B_1 = 1 - (b - m)(1 - t) $

# La ecuación de la Curva LM es:
# 
# $$ r = - \frac{1}{j}\frac{M^s_0}{P_0} + \frac{k}{j}Y $$ 

# Al igualar ambas Curvas logramos obtener el nivel de ingresos en equilibrio (Ye) y la tasa de interés de equilibrio (re). Se crea así el modelo IS-LM:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}(Y)= - \frac{1}{j}\frac{M^s_0}{Pj} + \frac{k}{j}Y $$

# - Ingreso de equilibrio: 
# 
# $$ Y^e = \frac{jB_0}{kh + jB_1} + (\frac{h}{kh + jB_1})\frac{Ms_0}{P_0} $$

# - Tasa de interés de equilibrio:
# 
# $$ r^e = \frac{kB_0}{kh + jB_1} + (\frac{B_1}{kh + jB_1})\frac{Ms_0}{P_0} $$

# #### Grafico del equilibrio simultáneo en los mercados de bienes y de dinero

# In[3]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo)/h - ( ( 1-(b-m)*(1-t) ) / h)*Y  
    return r_IS

r_is = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def r_LM(k, j, Ms, P, Y):
    r_LM = - (1/j)*(Ms/P) + (k/j)*Y
    return r_LM

r_lm = r_LM( k, j, Ms, P, Y)


# In[4]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(r_lm)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
# Curva IS
ax.plot(Y, r_is, label = "IS", color = "#554BDE") #IS
# Curva LM
ax.plot(Y, r_lm, label="LM", color = "#F060D3")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
# Graficar la linea horizontal - r
plt.axvline(x=52.5,  ymin= 0, ymax= 0.53, linestyle = ":", color = "black")
# Grafica la linea vertical - Y
plt.axhline(y=94, xmin= 0, xmax= 0.53, linestyle = ":", color = "black")

# Plotear los textos 
plt.text(49,100, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-10, '$Y_0$', fontsize = 12, color = 'black')


# Título, ejes y leyenda
ax.set(title="Equilibrio simultáneo en los mercados de bienes y de dinero", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# In[ ]:


# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
Y_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# ### 2. Estática Comparativa

# #### Efectos sobre Y y r en una disminución del gasto fiscal    ( ∆Go < 0)

# #### Intuición: Revisar

# ##### Mercado de bienes (IS):

# $$ Go↓ → DA↓ → DA < Y → Y↓ $$

# ##### Mercado de dinero (Lm): 

# $$ Y↓ → Md↓ → Md < Ms → r↓ $$

# #### Matemáticamente: 

# In[7]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[8]:


df_Y_eq_Go = diff(Y_eq, Go)
print("El Diferencial del producto con respecto al diferencial del gasto autonomo = ", df_Y_eq_Go)  # este diferencial es positivo


# ¿$∆Y$ sabiendo que $∆G_0 < 0$?
# 
# $$ \frac{∆Y}{∆G_0} = \frac{j}{(h*k + j*(-(1 - t)*(b - m) + 1))} $$
# 
# $$ \frac{∆Y}{(-)} = (+) $$
# 
# $$ ∆Y < 0 $$

# In[9]:


df_r_eq_Go = diff(r_eq, Go)
print("El Diferencial de la tasa de interes con respecto al diferencial del gasto autonomo = ", df_r_eq_Go)  # este diferencial es positivo


# ¿$∆r$ sabiendo que $∆G_0 < 0$?
# 
# $$ \frac{∆r}{∆G_0} = \frac{k}{(h*k + j*(-(1 - t)*(b - m) + 1))} $$
# 
# $$ \frac{∆r}{(-)} = (+) $$
# 
# $$ ∆r < 0 $$

# #### Gráficos:

# In[6]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[7]:


#--------------------------------------------------
    # NUEVA curva IS: reducción Gasto de Gobienro (Go)
    
# Definir SOLO el parámetro cambiado
Go = 25

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[12]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "#E33D87") #IS_orginal
ax.plot(Y, r_G, label = "IS_(G_1)", color = "#E33D87", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "#66E5F3")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=52,  ymin= 0, ymax= 0.53, linestyle = ":", color = "grey")
plt.axhline(y=94, xmin= 0, xmax= 0.53, linestyle = ":", color = "grey")

plt.axvline(x=43,  ymin= 0, ymax= 0.44, linestyle = ":", color = "grey")
plt.axhline(y=75, xmin= 0, xmax= 0.44, linestyle = ":", color = "grey")

plt.text(38,60, '$E_1$', fontsize = 14, color = 'black')
plt.text(50,100, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(-1,80, '$r_1$', fontsize = 12, color = 'black')
plt.text(47,-10, '$Y_0$', fontsize = 12, color = 'black')
plt.text(38,-10, '$Y_1$', fontsize = 12, color = 'black')

plt.text(69, 52, '←', fontsize=15, color='grey')
plt.text(46, 15, '←', fontsize=15, color='grey')
plt.text(10, 82, '↓', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Disminución en el Gasto de Gobierno", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# ### Efectos sobre Y y r en una disminución de la masa monetaria (Ms < 0)

# ### Intuición:

# #### Mercado de Bienes:

# $$ r↑ → I↓ → DA < Y → Y↓ $$

# #### Mercado de Dinero

# $$ M_s↓ → M^o↓ → M^o < M^d → r↑ $$

# ### Matemáticamente: Revisar 

# In[13]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[14]:


df_Y_eq_Ms = diff(Y_eq, Ms)
print("El Diferencial del Producto con respecto al diferencial de la masa monetaria = ", df_Y_eq_Ms)  # este diferencial es positivo


# ¿$∆Y$ sabiendo que $∆M^s < 0$?
# 
# $$ \frac{∆Y}{∆M^s} = \frac{h}{(P*(h*k + j*(-(1 - t)*(b - m) + 1)))} $$
# 
# $$ \frac{∆Y}{(-)} = (+) $$
# 
# $$ ∆Y < 0 $$

# In[15]:


df_r_eq_Ms = diff(r_eq, Ms)
print("El Diferencial de la tasa de interes con respecto al diferencial de la masa monetaria = ", df_r_eq_Ms)  # este diferencial es positivo


# ¿$∆r$ sabiendo que $∆M^s < 0$?
# 
# $$ \frac{∆r}{∆M^s} = \frac{-(-(1 - t)*(b - m) + 1)}{(P*(h*k + j*(-(1 - t)*(b - m) + 1)))} $$
# 
# $$ \frac{∆r}{(-)} = (-) $$
# 
# $$ ∆r > 0 $$

# ### Gráfico:

# In[14]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 700             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[15]:


# Definir SOLO el parámetro cambiado
Ms = 200

# Generar nueva curva LM con la variacion del Ms
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[17]:


# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "#ED0000") #IS_orginal
ax.plot(Y, i, label="LM_(MS_0)", color = "#FF8F20")  #LM_original

ax.plot(Y, i_Ms, label="LM_(MS_1)", color = "#FF8F20", linestyle = 'dashed')  #LM_modificada

# Lineas de equilibrio_0 
plt.axvline(x=60,  ymin= 0, ymax= 0.53, linestyle = ":", color = "grey")
plt.axhline(y=85, xmin= 0, xmax= 0.6, linestyle = ":", color = "grey")

# Lineas de equilibrio_1 
plt.axvline(x=52,  ymin= 0, ymax= 0.57, linestyle = ":", color = "grey")
plt.axhline(y=94, xmin= 0, xmax= 0.53, linestyle = ":", color = "grey")

# Textos ploteados
plt.text(58,92, '$E_0$', fontsize = 14, color = 'black')
plt.text(50,101, '$E_1$', fontsize = 14, color = 'black')
plt.text(-1,75, '$r_0$', fontsize = 12, color = 'black')
plt.text(62,-40, '$Y_0$', fontsize = 12, color = 'black')
plt.text(-1,100, '$r_1$', fontsize = 12, color = '#3D59AB')
plt.text(53,-40, '$Y_1$', fontsize = 12, color = '#3D59AB')

plt.text(69, 115, '←', fontsize=15, color='grey')
plt.text(55, 15, '←', fontsize=15, color='grey')
plt.text(10, 87, '↑', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Efecto de una disminución de la masa monetaria", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# ### Efectos sobre Y y r en un incremento de la tasa de impuestos (t > 0)

# ### Intuición:

# #### Mercado de Bienes:

# $$ t↑ → DA↓ → DA < Y → Y↓$$

# #### Mercado de Dinero:

# $$ Y↓ → Md↓ → Md < Ms → r↓ $$

# ### Matemáticamente: REVISAR!

# In[19]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[20]:


df_Y_eq_t = diff(Y_eq, t)
print("El Diferencial del producto con respecto al diferencial de la tasa impositiva = ", df_Y_eq_t)  # este diferencial es positivo


# ¿$∆Y$ sabiendo que $∆t > 0$?
# 
# $$ \frac{∆Y}{∆t} = (-) $$
# 
# $$ \frac{∆Y}{(+)} = (-) $$
# 
# $$ ∆Y < 0 $$

# In[21]:


df_r_eq_t = diff(r_eq, t)
print("El Diferencial de la tasa de interes con respecto al diferencial a la tasa impositiva= ", df_r_eq_t)  # este diferencial es positivo


# ¿$∆r$ sabiendo que $∆t > 0$?
# 
# $$ \frac{∆r}{∆t} = (-) $$
# 
# $$ \frac{∆r}{(+)} = (-) $$
# 
# $$ ∆r < 0 $$

# ### Gráfico: 

# In[20]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.1

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[21]:


#--------------------------------------------------
    # NUEVA curva IS: 
    
# Definir SOLO el parámetro cambiado
t = 3

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[25]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "#C784FC") #IS_orginal
ax.plot(Y, r_t, label = "IS_(G_1)", color = "#C784FC", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "#82C7B1")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=48,  ymin= 0, ymax= 0.48, linestyle = ":", color = "grey")
plt.axhline(y=86, xmin= 0, xmax= 0.48, linestyle = ":", color = "grey")

plt.axvline(x=54,  ymin= 0, ymax= 0.54, linestyle = ":", color = "grey")
plt.axhline(y=98, xmin= 0, xmax= 0.54, linestyle = ":", color = "grey")

plt.text(43,70, '$E_1$', fontsize = 14, color = 'black')
plt.text(52,103, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,101, '$r_0$', fontsize = 12, color = 'black')
plt.text(-1,79, '$r_1$', fontsize = 12, color = 'black')
plt.text(55,-10, '$Y_0$', fontsize = 12, color = 'black')
plt.text(44,-10, '$Y_1$', fontsize = 12, color = 'black')

plt.text(69, 61, '←', fontsize=15, color='grey')
plt.text(49, 15, '←', fontsize=15, color='grey')
plt.text(10, 89, '↓', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Aumento en la Tasa Impositiva", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# ## Puntos extra

# ### 1. A partir del siguiente sistema de ecuaciones que representan el modelo IS-LM
# #### 1.1 Ecuaciones de Ingreso $(Y^e)$ y tasa de interes $(r^e)$ de equilibrio (escriba paso a paso la derivacion de estas ecuaciones).

# - Curva IS:
# 
# A partir de la nueva identidad Ingreso-Gasto: $ Y = C + I + G $
# 
# $$ Y = C_0 + bY^d + I_0 - hr + G_0$$
# 
# $$ Y = C_0 + I_0 + G_0 - hr + b(1-t)Y $$
# 
# $$ hr = C_0 + I_0 + G_0 + b(1-t)Y - Y $$
# 
# $$ hr = C_0 + I_0 + G_0 - Y(1- b(1-t)) $$
# 
# La ecuación de la curva IS es:
# 
# $$ r = \frac{C_0 + I_0 + G_0}{h} - \frac{1- b(1-t)}{h}Y $$
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $B_0 = C_0 + I_0 + G_0 $ y $ B_1 = 1- b(1-t) $

# - Curva LM:
# 
# $$ \frac{M^s_0}{P_0} = kY - j(r + π^e) $$
# 
# $$ j(r + π^e) = kY - \frac{M^s_0}{P_0} $$
# 
# $$ r + π^e = - \frac{M^s_0}{jP_0} + \frac{kY}{j} $$
# 
# La ecuación de la curva LM es:
# 
# $$ r = - \frac{M^s_0}{jP_0} + \frac{k}{j}Y - π^e $$

# - Equilibrio modelo IS-LM:
# 
# Para hallar $Y^e$:
# 
# $$ \frac{B_0}{h} - \frac{B_1}{h}Y = - \frac{M^s_0}{jP_0} + \frac{k}{j}Y - π^e $$
# 
# $$ \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e = \frac{k}{j}Y + \frac{B_1}{h}Y $$
# 
# $$ Y(\frac{k}{j} + \frac{B_1}{h}) = \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e $$
# 
# $$ Y(\frac{hk + jB_1}{jh}) = \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e $$
# 
# $$ Y^e = \frac{jB_0}{kh + jB_1} + \frac{M_0^s}{P_0} \frac{h}{kh + jB_1} + \frac{jh}{kh + jB_1} π^e $$
# 
# Para hallar $r^e$:
# 
# $$ r^e = - \frac{Ms_o}{P_o} (\frac{B_1}{kh + jB_1}) + \frac{kB_o}{kh + jB_1} - \frac{B_1}{kh + jB_1} π^e $$

# #### 1.2 Grafique el equilibrio simultáneo en los mercados de bienes y de dinero.

# In[54]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20
π = 4

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


# In[55]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "#CE66FF") #IS
ax.plot(Y, i_2, label="LM", color = "#FF69B6")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=55,  ymin= 0, ymax= 0.54, linestyle = ":", color = "black")
plt.axhline(y=94, xmin= 0, xmax= 0.55, linestyle = ":", color = "black")
plt.text(53,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-15, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Equilibrio modelo IS-LM", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# In[56]:


# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# ### 2. Estática comparativa:
# #### 2.1. Analice los efectos sobre las variables endógenas Y, r de una disminución de los Precios $(∆P_0 < 0)$. El análisis debe ser intuitivo, matemático y gráfico.

# - Intuición:
# 
# $$ P↓ → M^s↑ → M^s > M^d → r↓ $$
# 
# $$ r↓ → I↑ → DA↑ → DA > Y → Y↑ $$
# 
# - Matemática:

# In[57]:


# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# In[58]:


df_Y_eq_P = diff(Y_eq, P)
print("El Diferencial del Producto con respecto al diferencial del nivel de precios = ", df_Y_eq_P)


# ¿$∆Y$ sabiendo que $∆P < 0$?
# 
# $$ \frac{∆Y}{∆P} = (-) $$
# 
# $$ \frac{∆Y}{(-)} = (-) $$
# 
# $$ ∆Y > 0 $$

# In[59]:


df_r_eq_P = diff(r_eq, P)
print("El Diferencial de la tasa de interés con respecto al diferencial del nivel de precios = ", df_r_eq_P)


# ¿$∆r$ sabiendo que $∆P < 0$?
# 
# $$ \frac{∆r}{∆P} = (+) $$
# 
# $$ \frac{∆r}{(-)} = (+) $$
# 
# $$ ∆r < 0 $$

# In[60]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20
π = 4

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


#--------------------------------------------------
    # Nueva curva LM 
    
P = 5

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2_P = i_LM_2( k, j, Ms, P, Y, π)


# In[61]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "#1594F6") #IS
ax.plot(Y, i_2, label="LM", color = "#3AA63F")  #LM
ax.plot(Y, i_2_P, label="LM", color = "#3AA63F", linestyle ='dashed')  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=55,  ymin= 0, ymax= 0.6, linestyle = ":", color = "black")
plt.axhline(y=94, xmin= 0, xmax= 0.55, linestyle = ":", color = "black")
plt.text(53,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-45, '$Y_0$', fontsize = 12, color = 'black')

plt.axvline(x=64.5,  ymin= 0, ymax= 0.56, linestyle = ":", color = "black")
plt.axhline(y=85, xmin= 0, xmax= 0.64, linestyle = ":", color = "black")
plt.text(62,90, '$E_1$', fontsize = 14, color = 'black')
plt.text(0,75, '$r_1$', fontsize = 12, color = 'black')
plt.text(66,-45, '$Y_1$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Disminución del Precio", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# #### 2.2 Analice los efectos sobre las variables endógenas Y, r de una disminución de la inflación esperada $(∆π < 0)$. El análisis debe ser intuitivo, matemático y gráfico.

# - Intuición:
# 
# $$ π↓ → r↑ $$
# 
# $$ r↑ → I↓ → DA↓ → DA < Y → Y↓ $$
# 
# - Matemática:

# In[62]:


# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# In[63]:


df_Y_eq_π = diff(Y_eq, π)
print("El Diferencial del Producto con respecto al diferencial del nivel de inflación = ", df_Y_eq_π)


# ¿$∆Y$ sabiendo que $∆π < 0$?
# 
# $$ \frac{∆Y}{∆π} = (+) $$
# 
# $$ \frac{∆Y}{(-)} = (+) $$
# 
# $$ ∆Y < 0 $$

# In[64]:


df_r_eq_π = diff(r_eq, π)
print("El Diferencial de la tasa de interés con respecto al diferencial del nivel de inflación = ", df_r_eq_π)


# ¿$∆r$ sabiendo que $∆π < 0$?
# 
# $$ \frac{∆r}{∆π} = (-) $$
# 
# $$ \frac{∆r}{(-)} = (-) $$
# 
# $$ ∆r > 0 $$

# - Gráfico:

# In[65]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20
π = 20

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


#--------------------------------------------------
    # Nueva curva LM 
    
π = 2

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2_π = i_LM_2( k, j, Ms, P, Y, π)


# In[67]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "#F10080") #IS
ax.plot(Y, i_2, label="LM", color = "#5E2390")  #LM
ax.plot(Y, i_2_π, label="LM", color = "#5E2390", linestyle ='dashed')  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=54,  ymin= 0, ymax= 0.57, linestyle = ":", color = "black")
plt.axhline(y=95, xmin= 0, xmax= 0.54, linestyle = ":", color = "black")
plt.text(52,103, '$E_1$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_1$', fontsize = 12, color = 'black')
plt.text(50,-35, '$Y_1$', fontsize = 12, color = 'black')

plt.axvline(x=60,  ymin= 0, ymax= 0.55, linestyle = ":", color = "black")
plt.axhline(y=89, xmin= 0, xmax= 0.6, linestyle = ":", color = "black")
plt.text(58,95, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,80, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-35, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Disminución de la inflación esperada", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()

