#!/usr/bin/env python
# coding: utf-8

# # Segundo reporte

# In[1]:


# María Fernanda Carrillo 


# # Parte I: Reporte

# En este artículo la pregunta fundamental de investigación es que busca explicar cuáles fueron los métodos de restablecimiento de la política monetaria (tras explicar brevemente las causas) tras los regulares procesos de inflación e hiperinflación desde los años 1970 hasta los 90’s; así como el detallar por qué todos esos procesos y/o intentos por estabilizar la economía resultaron en fracasos y porqué fue que el proyecto de Fujimori finalmente funcionó. 
# 
# El análisis que realiza es sumamente destacable, ya que lo que hace no solo es revisar cada uno de las propuestas dadas a lo largo de esos años, sino que señala cuáles fueron los errores, no solo económicos, sino dificultades coyunturales (en algunas ocasiones) que explican el porqué estos no funcionaron. Ya que comprende el papel insustituible del Estado y el impacto de las decisiones que estos jugaban; no obstante, presta atención al rol de otros jugadores, como las élites o la población, quienes con sus opiniones y manifestaciones a favor o en contra de las decisiones estatales lograban cambios importantes en su enfoque o iniciativas, como la desconfianza general a la economía de mercado inicial o el posterior rechazo al intervencionismo estatal. Sin embargo, algo que se puede argumentar en contra del texto es que utiliza un desarrollo mucho más académico, lo cual, si bien es sumamente útil para expertos, inhibe que la población en general pueda disfrutar de su contenido. Ya que el autor no se detiene a explicar algunos conceptos clave como, por ejemplo, la ‘hiperestanflación’ o ‘señoreaje’ y da por sentado que el lector sabe de lo que está refiriéndose; asimismo al centrarse en un enfoque monetario también deja de lado muchos aspectos socio-culturales que influyeron en una medida incluso mayor a la que menciona. 
# 
# Este documento de manera fascinante es que, tal vez sin apuntar a ello inicialmente, lo que hace es explicar acertadamente el porqué es que los peruanos, en una perspectiva muy generalizadora, le tienen un rechazo internalizado al intervencionismo estatal; ello principalmente pues es que adjudicamos muchas de nuestras crisis y debacles económicos a estos (claro está, como principal factor). De la misma manera, ofrece una explicación sumamente detallada y certera acerca de las causas y consecuencias históricas de los procesos altamente impactantes en el Perú; ello puede que con el fin de advertir a todo aquello que lea este ‘paper’ de algunas consecuencias o situaciones particulares que se han de evitar para no volver a pasar por situaciones como estas. Particularmente tomando un enfoque que se puede decir no es keynesiano, al menos no en tanto no aboga por el intervencionismo estatal con el fin de promover la economía, sino que más bien se sitúa en una postura más alejada de este y que aboga por una determinación dada por el libre mercado. 
# 
# Y la postura que sostienen los autores de este artículo se respalda en lo que Rodríguez et alt. (2022) mencionan en el artículo “Instituciones del régimen económico constitucional de 1979 y gran depresión de la economía peruana: 1988-1990”, ya que concluyen finalmente que fue el control excesivo que ejerció Belaunde sobre la inversión privada, así como una ineficiente actividad empresarial estatal lo que llevó a niveles de inflación enormes. “[…] el estatismo e intervencionismo que resultó ser nefasto para el país.”, adjudicándolo directamente como la causa de la crisis económica más grande del país en los últimos años. La restricción que ejercieron sobre la economía fue lo que sucesivamente ocasionaron en su conjunto la recesión económica y lo que finalmente terminan Martinelli y Vega explicando en el artículo: fue la liberalización económica lo que permitió la recuperación del país y el crecimiento sostenido que ha venido llevando desde su implementación. 
# 
# Bibliografía: 
# 
#     RODRIGUEZ, Vladimir et alt. 
#     2022	“Instituciones del régimen económico constitucional de 1979 y gran depresión de la economía peruana: 1988-1990”. En Notas históricas y geográficas, pp. 18-47. Consulta: 11 de setiembre de 2022. 
# 
# 

# # Parte II: Código

# In[2]:


# Trabajado con Angela Rodriguez (20200748)


# In[3]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel


# ## a. Modelo Ingreso-Gasto: la Curva IS

# ## Derive paso a paso la curva IS matemáticamente (Y=DA)

# La curva IS se deriva de la igualdad entre el ingreso (Y)  y la demanda agregada (DA):
# $$ Y = C + I + G + X - M$$
# 
# Considerando que: 
# 
# $$ C = C_0 + bY^d $$
# $$ I = I_0 - hr $$
# $$ G = G_0 $$
# $$ T = tY $$
# $$ X = X_0 $$
# $$ M = mY^d $$
# 

# Para llegar al equilibrio Ahorro-Inversión, debemos crear un artificio. Por ende pasamos a restar la tributación (T) de ambos miembros de la igualdad.
# 
# $$ Y - T = C + I - T + G + X - M$$
# $$ Y^d = C + I - T + G + X - M$$

# Esta igualdad se puede reescribir de la siguiente forma:
# $$ (Y^d - C) + (T-G) + (M-X) = I = S$$

# Las tres partes de la derecha constituyen los tres componentes del ahorro total (S) : ahorro privado/interno (Sp), ahorro del gobierno (Sg)  y ahorro externo (Se):
# $$ S = Sp + Sg + Se$$

# Entonces, el ahorro total es igual a la inversión:
# $$ Sp + Sg + Se = I$$
# $$ S(Y) = I(r)$$

# Hacemos reemplazos tomando en cuenta las ecuaciones básicas, se obtiene que:
# 
# 
# $$ S_p + S_g + S_e = I_0 - hr $$
# $$ (Y^d - C_0 - bY^d) + (T - G_0) + (mY^d - X_0) = I_0 - hr $$
# 
# Considerando las observaciones anteriores sobre los componentes de la condición de equilibrio $(Y)$:
# 
# $$ [1 - (b - m)(1 - t)]Y - (C_0 + G_0 + X_0) = I_0 - hr $$

# Continuamos derivando:
# $$ hr = Io -[1-(b-m)(1-t)Y - (C_0 + G_0 + X_0)]$$
# 
# $$ hr = (C_0 + G_0 + I_0 + X_0) - (1-(b-m)(1-t))Y$$

# La curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso:
# 
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1-(b-m(1-t)}{h}(Y)$$

# Y puede simplificarse en:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $ B_0 = C_0 + G_0 + I_0 + X_0  $ es el intercepto y $  B_1 = 1 - (b - m)(1 - t) $ es la pendiente.

# Es en esta condición de igual en el cual el mercado de bienes hay una relación negativa entre el producto y el ingreso

# ## Encuentre ∆r/∆Y

# In[4]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, Y = symbols('Co Io Go Xo h r b m t Y')

# determinar ecuación
f = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h

# función diferencial
df_r = diff(f, Y) # diff(función, variable_analizar
df_r #∆Y/∆Go


# La pendiente es representada por el diferencial de $ ∆r/∆Y $: $- \frac{1 + (b - m)(1 - t)}{h} $ y es negativa.

# ## Explique cómo se derive la Curva IS (Y=DA)

# Partimos del hecho de que el ingreso está determinado por el consumo, inversión, el gasto y exportaciones netas. Ahora, lo que queremos es llegar a un equilibrio llamado ‘ahorro inversión’ y para ello debemos crear un artificio (Y - T  (Yd)= C + I + G + X - M - T).  Luego vamos a agrupar por sectores: interno, gobierno y externo (Yd - C) + (T-G) + (M-X). Con estas tres formas de gobierno obtenemos la inversión (S = I). Luego continuamos derivando y remplazando las ecuaciones hasta llegar a la ecuación que que expresa la tasa de interés en función del ingreso. Entonces a partir de la ecuación de la Demanda Agregada podemos llegar a saber que el equilibrio en ahorro es igual a la inversión y que en una condición de igual en el mercado de bienes hay una relación negativa entre el producto y el ingreso que se puede graficar a partir del equilibrio DA=Y en una recta de 45°.

# Recordemos la ecuación del ingreso de equilibrio a corto plazo que fue obtenida a partir del equilibrio $(Y = DA)$:
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# 
# Esta ecuación, después de algunas operaciones, puede expresarse en función de la tasa de interés $(r)$:
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# 
# Entonces, la curva IS puede ser simplificada de la siguiente manera:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $ B_0 = C_0 + G_0 + I_0 + X_0  $ y $  B_1 = 1 - (b - m)(1 - t) $

# - Demanda Agregada

# In[8]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)

#--------------------------------------------------
# Recta de 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)

#--------------------------------------------------
# Segunda curva de ingreso de equilibrio

    # Definir cualquier parámetro autónomo
Go = 35

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# - Curva IS

# In[9]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[14]:


# Gráfico de la derivación de la curva IS a partir de la igualdad (DA = Y)

    # Dos gráficos en un solo cuadro (ax1 para el primero y ax2 para el segundo)
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: ingreso de Equilibrio
ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.plot(Y, DA_IS_K, label = "DA_0", color = "#E0BBE4") 
ax1.plot(Y, DA_G, label = "DA_1", color = "#957DAD") 
ax1.plot(Y, L_45, color = "#404040") 

ax1.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
ax1.axvline(x = 54,  ymin= 0, ymax = 0.54, linestyle = ":", color = "grey")

ax1.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
ax1.text(2.5, -3, '$◝$', fontsize = 30, color = 'black')
ax1.text(72, 0, '$Y_0$', fontsize = 12, color = 'black')
ax1.text(56, 0, '$Y_1$', fontsize = 12, color = 'black')
ax1.text(67, 185, '$E_0$', fontsize = 12, color = 'black')
ax1.text(50, 142, '$E_1$', fontsize = 12, color = 'black')

ax1.set(title = "Derivación de la curva IS a partir del equilibrio $Y=DA$", xlabel = r'Y', ylabel = r'DA')
ax1.legend()

#---------------------------------
    # Gráfico 2: Curva IS

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.plot(Y, r, label = "IS", color = "#F3A5BC") 

ax2.axvline(x = 70.5, ymin= 0, ymax = 1, linestyle = ":", color = "grey")
ax2.axvline(x = 54,  ymin= 0, ymax = 1, linestyle = ":", color = "grey")
plt.axhline(y = 151.5, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
plt.axhline(y = 165, xmin= 0, xmax = 0.55, linestyle = ":", color = "grey")

ax2.text(72, 128, '$Y_0$', fontsize = 12, color = 'black')
ax2.text(56, 128, '$Y_1$', fontsize = 12, color = 'black')
ax2.text(1, 153, '$r_0$', fontsize = 12, color = 'black')
ax2.text(1, 167, '$r_1$', fontsize = 12, color = 'black')
ax2.text(72, 152, '$E_0$', fontsize = 12, color = 'black')
ax2.text(55, 166, '$E_1$', fontsize = 12, color = 'black')

ax2.legend()

plt.show()


# ## b. La Curva IS o el equilibrio Ahorro- Inversión

# Entonces, el ahorro total es igual a la inversión:
# $$ (Y^d - C) + (T-G) + (M-X) = S = I$$
# $$ Sp + Sg + Se = I$$
# $$ S(Y) = I(r)$$

# Hacemos reemplazos tomando en cuenta las ecuaciones básicas se obtiene que:
# 
# $$ (Y^d - C_0 - bY^d) + (T - G_0) + (mY^d - X_0) = I_0 -hr$$
# 
# $$ Y^d -(C_0 + bY^d) +[tY - G_0] + [mY^d - X_0] = Io - hr$$
# 
# $$ [1-(b-m)(1-t)]Y-(C_0 + G_0 + X_0) = Io - hr$$
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1-(b-m(1-t)}{h}(Y)$$

# In[15]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

IS_IS = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[16]:


# Gráfico de la curva IS

# Dimensiones del gráfico
y_max = np.max(IS_IS)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, IS_IS, label = "IS_IS", color = "#E68AA9") #Demanda agregada
ax.text(98, 132, '$IS$', fontsize = 14, color = 'black')
# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

custom_xlim = (-8, 115)
custom_ylim = (110,230)
plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)

# Título, ejes y leyenda
ax.set(title = "Curva IS de Equilibrio en el Mercado de Bienes", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# ## c. Desequilibrios en el mercado de bienes

# In[17]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r1 = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[19]:


#Dimensiones:
y_max = np.max(r1)
fig, ax = plt.subplots(figsize=(10, 8))
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())
ax.plot(Y, r1, label = "IS", color = "#658C72") 
#Lineas punteadas:
ax.axvline(x = 70.5, ymin= 0, ymax = 0.45, linestyle = ":", color = "grey")
ax.axvline(x = 54,  ymin= 0, ymax = 0.45, linestyle = ":", color = "grey")
ax.axvline(x = 37,  ymin= 0, ymax = 0.45, linestyle = ":", color = "grey")
plt.axhline(y = 165, xmin= 0, xmax = 0.67, linestyle = ":", color = "grey")
#Leyenda:
ax.text(72, 122, '$Y_B$', fontsize = 12, color = 'black')
ax.text(55, 122, '$Y_A$', fontsize = 12, color = 'black')
ax.text(38, 122, '$Y_C$', fontsize = 12, color = 'black')
ax.text(-8, 167, '$r_A$', fontsize = 12, color = 'black')
ax.text(70, 167, '$B$', fontsize = 14, color = 'black')
ax.text(53, 167, '$A$', fontsize = 14, color = 'black')
ax.text(36, 167, '$C$', fontsize = 14, color = 'black')

#Coordenadas de exceso:

ax.text(70,180, 'Exceso\n    de\n Oferta', fontsize = 15, color = 'black')
ax.text(38,145, '  Exceso\n      de\nDemanda', fontsize = 15, color = 'black')

ax.set(title = "Equilibrio y desequilibrio en el Mercado de Bienes-Exceso de Demanda y Oferta", xlabel = 'Y', ylabel= 'r')
ax.legend()

custom_xlim = (-10, 110)
custom_ylim = (120,220)
plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)

plt.show()


# Si la Curva IS representa los puntos donde hay un equilibrio en el mercado de bienes, los puntos fuera de la curva señalan un desequilibrio en el mercado. Al lado derecho se encuentra el exceso de demanda (Inversión (I) < Ahorro (S)), mientras que al lado izquierdo se encuentra el desequilibrio por execeso de oferta. 

# ## d. Movimientos de la curva IS

# ## Política Fiscal Contractica con caída del Gasto del Gobierno

# ### Intuición: (∆G < 0)

# $$ Go↓ → G↓ → DA↓ → DA < Y → Y↓$$

# ### Gráfico

# In[20]:


#--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
Go = 60

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[21]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "black") #IS orginal
ax.plot(Y, r_G, label = "IS_G", color = "#A8A6DB", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(47, 162, '∆Go', fontsize=12, color='black')
plt.text(49, 159, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Incremento en el Gasto de Gobierno $(G_0)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# ## Política Fiscal Expansiva con caída de la Tasa de Impuesto 

# ### Intuición: (∆t < 0)

# $$ ↓t → DA↑ → DA > Y → Y↑ $$

# ### Grafico:

# In[23]:


#--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo)/h - (Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
t = 0.1

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[24]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "black") #IS orginal
ax.plot(Y, r_t, label = "IS_t", color = "#7A86CC", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(47, 174.5, '∆t', fontsize=12, color='black')
plt.text(47, 172, '→', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Caída en la Tasa impositiva $(t)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# ## Caída de la Propensión Marginal a Consumir

# ### Intuición: (∆b < 0)

# $$ b↓ → C_o↓  → DA↓  → DA < Y → Y↓  $$

# ### Gráfico:

# In[157]:


#--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.5

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

b1 = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
b = 0.4

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h,Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

IS_b = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[158]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(b1)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, b1, label = "b", color = "black") #IS orginal
ax.plot(Y, IS_b, label = "IS_b", color = "#F69F84", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(58, 143, '∆b', fontsize=12, color='black')
plt.text(58, 139, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Caída de la Propensión Marginal a Consumir $(t)$", xlabel= 'Y', ylabel= 't')
ax.legend()

plt.show()

