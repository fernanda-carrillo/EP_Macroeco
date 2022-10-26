#!/usr/bin/env python
# coding: utf-8

# In[1]:


#20201596 María Fernanda Carrillo


# # I. REPORTE

# La pregunta de investigación sobre la cual Dancourt centra la investigación es la siguiente: cuáles fueron las causas y las consecuencias e impacto inflacionario en la economía peruana del proceso de ‘vacas flacas’ (y lo recesivo que resultó) que el Perú pasó entre los años 2013—15’; asimismo discute las medidas tomadas por el Estado y brinda asesoramiento sobre cuáles han de ser los próximos movimientos del futuro gobierno peruano con el fin de responder efectivamente a este fenómeno y las desastrosas consecuencias que tiene en el país. 
# 
# Una de las principales fortalezas que en el artículo se encuentran es que, a pesar de los variados conceptos, logra ser conciso en su explicación. Así a pesar de que el documento no sea tan extenso, resulta sencillo de comprender pese a que hay terminología e ideas propias de economía, hasta alguien que no posee tantos conocimientos previos en esa rama pueda comprender. 
# El enfoque keynesiano con el que el autor aborda el tema, presta la debida atención al estado y al papel fundamental que este tiene para el poder revertir la balanza en cuanto a la estabilidad que se pierde con este proceso de ‘vacas flacas’ en el Perú. Ya que logra identificar acciones negligentes o deficientes del Estado, como lo sería la excesiva venta de los dólares estatales, la cual menciona el autor compromete la política monetaria futura; siendo más aún que se trata de un sistema bancario dolarizado, al cual le chocaría con más fuerza un choque externo adverso y derribaría rápidamente su estabilidad. Asimismo, es realista en tanto, identifica adecuadamente que uno de los mayores problemas con el cual nuestra economía lidia es la gran dependencia que tenemos en las exportaciones, ya que no solo son capaces de condicionar las épocas de equilibrio y relativa bonanza en nuestro país con cambios de tipo externo en el precio de los metales (lógica keynesiana del producto determinado por el mercado monetario externo). 
# 
# Asimismo, es realista en su desarrollo, pues uno de los problemas que reconoce que hace que los inversionistas se retengan de invertir (parte fundamental en el proceso de estimulación económica) son las metas irrealistas de inflación que el estado peruano ha fijado, los cuales al estar afuera del rango la mayoría del tiempo, no brinda seguridad a los posibles inversores. Algo que fácilmente podría ser resuelto tan solo brindando un nuevo margen más sensato y acorde a la realidad
# Aporta al enfoque keynesiano en tanto, al detallar lo particularmente importante que resulta el Estado para poder dinamizar la economía interna es que ha de considerársele un actor central. Más aún, siendo que en la sección final dedicada a la planificación de su intervención no solo la presume a un corto plazo como lo postula la teoría keynesiana. Lo que hace Dancourt en este reporte y el consiguiente aviso que da lo hace basándose en la idea de una intervención a largo plazo en la política peruana, ya que el proceso de diversificación que propone es uno que llevaría años, si no décadas (considerando lo lento que opera el Estado en nuestro país), para poder ser llevada a cabo satisfactoriamente. Así contribuyendo al enfoque de la teoría en general y a la mirada que se le da al papel del Estado en la economía, ya que luego de todos los argumentos y dinámicas que presenta a lo largo del artículo es que no podemos obviar su papel, ya que no solo mueve a la economía mediante la inserción de dinero, sino que son sus acciones referentes a las tasas de interés y las expectativas que tenga lo que atraen a los inversionistas y mueven aún más a la economía. 
# 
# Mas aún, su contribución, como muchas otras, no se limita tan solo al análisis del fenómeno y a una crítica hacia el aparato estatal. Sino que el aporte que realiza, un consejo de corte profesional, con un plan de acción y medidas específicas y perfectamente viables a aplicar en nuestro país muestra un genuino interés y preocupación por la situación. No solo instándoles a mejorar medidas económicas concretas, a corto plazo, sino que, como se mencionó antes, extrapola ese ámbito y busca ser de utilidad para un futuro incluso lejano. Además de, obviamente, el traer a la luz un tema que seguramente a muchos se les pasaba por alto, y es gracias al lenguaje sencillo y comprensible desarrollo que tiene el texto es que permite que este fenómeno se haga de conocimiento público, para que este lo tome en cuenta para opinar o tomar decisiones futuras, como el votar, respecto a lo que los dirigentes realizan en respuesta a estos, sin que se les pase nada por alto. Así como el advertir, debido a lo cíclico que es, de un posible retorno y urgencia de las acciones que se han de tomar para combatir este. 
# Asimismo, lo estipulado respecto a las expectativas de la inflación impactan negativamente, y significativamente, es ratificado por el ‘paper’ del BCRP. Donde Rossini et alt. mencionan que, la formación de las expectativas de inflación se ven aún más contaminadas por una economía dolarizada, debido a la depreciación del tipo de cambio y a que son influidos por la evolución del tipo de cambio que manejan. Por ello es necesario, e incluso urgente, el estabilizar los precios para poder recuperar la confianza en la moneda peruana, lo que llevaría finalmente a una desdolarización que tanto nos hace falta. “Una adecuada postura de política monetaria puede permitir que las expectativas de inflación se mantengan ancladas en niveles cercanos a la meta de inflación” (Rossini et alt 2016: 12)
# 
# (Fuente: https://www.bcrp.gob.pe/docs/Publicaciones/Revista-Estudios-Economicos/31/ree-31-rossini-vega-quispe-perez.pdf)
# 

# # II. CÓDIGO

# In[2]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel


# ## 1. Función de Demanda de Consumo

# La función de Consumo es: $$ C = C_0 + bYd => C = C_0 + b(1-t)Y$$

# En esta función se representa como parte del ingreso se utiliza en los gastos de consumo. El consumo fijo representa al consumo autónomo, aquel que es fijo y no cambia, utilizado para servicios como la renta o la comida y la b representa a la propesión marginal a consumir, es decir aquella que es variable.

# In[9]:


# Parámetros

Y_size = 100 

Co = 35
b = 0.8
t = 0.3

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def C(Co, b, t, Y):
    C = Co + b*(1-t)*Y
    return C

C = C(Co, b, t, Y)


# In[13]:


# Gráfico

# Dimensiones del gráfico
fig, ax = plt.subplots(figsize=(10, 8))

custom_xlim = (0, 130)
custom_ylim = (20, 130)

plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)


# Curvas a graficar
ax.plot(Y, C, label = "Consumo", color = "C1") #Demanda agregada

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto agregado
    # punto de equilibrio
plt.text(80, 95, r'$PMgC=\frac{\Delta C}{\Delta Y^d}=b=0.5$', fontsize = 13, color = 'black')


# Título y leyenda
ax.set(title="Función de demanda de Consumo", xlabel= '$Y^d$', ylabel= 'C')
ax.legend() #mostrar leyenda

plt.show()


# ## 2. Función de Demanda de Inversión

# La función es: $$ I = I_0 - hr$$

# En esta función se encuentra a la inversión autónoma (I0) como valor fijo, y el h que que representa la sensibilidad (que va en un rango de 0,1) que muestran los inversionistas ante la subida o bajada de la tasa de interés (r). En este sentido, la función nos permite entender como los gastos de inversión aumenta el stock de la economía, provenientes de los empresarios, y como las expectativas y especulaciones afectan el desarrollo y producción de la economía. Dicha inversión además se ve estimulada al ser menor la tasa de interés.

# In[15]:


# Parámetros

r_size = 90
r = np.arange(r_size)
r 

h = 0.3
Io = 30



# Ecuación -> I = Io - hr

def F_DI(Io, h, r):
    F_DI = Io - (h*r)
    return F_DI

F_DI = F_DI(Io, h, r)

F_DI
 


# In[16]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(F_DI)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(r, F_DI, label = "Inversion", color = 'C2') #Demanda agregada
ax.set(title="Función de Demanda de Inversion", xlabel= r'r', ylabel= r'I')
ax.legend() #mostrar leyenda

plt.text(90,5,r'$I=I_0-hr$', fontsize=15, color = 'black')

xline = (0, 110)
yline = (-10, 40)
plt.setp(ax, xlim=xline, ylim=yline)

ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())   


# ## 3. ¿Cuáles son los supuestos del modelo Ingreso-Gasto Keynesiano?

# Los supuestos del modelo de ingreso-gasto keynesiano son 4: el nivel de precios debe ser fijo, el nivel producción se adapta a los cambios dados en la demanda agregada (es decir la oferta se adapta a la demanda), la tasa de interés está determinada exógenamente y además que este se trata de un modelo de corto plazo únicamente (Jimenez 2020: pp.52).

# ## 4. Encuentra el nivel de Ingreso de Equilibrio

# La ecuación de equilibrio para el Ingreso Agregado deriva de la condición de equilibrio donde el ingreso es igual a la demanda agregada: $DA = Y$. Sabemos que:
# 
# $$ DA = C + I + G + X - M $$
# 
# Donde:
# 
# $$ C = C_0 + bY^d $$
# $$ I = I_0 - hr $$
# $$ G = G_0 $$
# $$ X = X_0 $$
# $$ M = mY^d $$
# 
# $$ Y^d = 1 - t $$

# Entonces: 
# 
# $$ DA = C_0 + I_0 + G_0 + X_0 - hr + Y(b - m)(1 - t) $$
# 
# Así que:
# 
# $$ DA = α_0 + α_1Y $$
# 
# Donde $ α_0 = (C_0 + I_0 + G_0 + X_0 -hr)$ es el intercepto y $ α_1 = (b - m)(1 - t) $ es la pendiente de la función
# 

# Ahora, considerando la condición de equilibrio $Y = DA$, la ecuación del ingreso de equilibrio es:
# 
# $$ Y = C_0 + bY^d + I_0 -hr + G_0 + X_0 - mY^d $$
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# Donde $\frac{1}{1 - (b - m)(1 - t)}$ es el multiplicador keynesiano $(k)$.

# - Gráfico con descripciones

# In[19]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[20]:


# Recta de 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)


# In[22]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(DA_IS_K, label = "DA", color = "#957DAD") #Demanda agregada
ax.plot(L_45, color = "#404040") #Línea de 45º

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Líneas punteadas punto de equilibrio
plt.axvline(x=70.5,  ymin= 0, ymax= 0.69, linestyle = ":", color = "grey")
plt.axhline(y=176, xmin= 0, xmax= 0.7, linestyle = ":", color = "grey")

plt.axvline(x=70.5,  ymin= 0, ymax= 0.69, linestyle = ":", color = "grey")
plt.axhline(y=145, xmin= 0, xmax= 1, linestyle = ":", color = "grey")

# Texto agregado
    # punto de equilibrio
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(0, 152, '$α_o$', fontsize = 15, color = 'black')
    # línea 45º
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
    # ecuaciones
plt.text(87, 203, '$DA = α_0 + α_1 Y$', fontsize = 11.5, color = '#957DAD')
plt.text(80, 165, '$α_1 = (b-m)(1-t)$', fontsize = 11.5, color = 'black')
plt.text(73, 125, '$α_0 = C_o + I_o + G_o + X_o - hr$', fontsize = 11.5, color = 'black')

plt.text(92, 192, '$↓$', fontsize = 13, color = 'black')
plt.text(85, 135, '$↑$', fontsize = 13, color = 'black')

# Título y leyenda
ax.set(title="El ingreso de Equilibrio a Corto Plazo", xlabel= r'Y', ylabel= r'DA')
ax.legend() #mostrar leyenda

plt.show()


# ## 5. Estática Comparativa en el modelo de Ingreso-Gasto Keynesino

# ### 5.1 Política Fiscal Expansiva con aumento del Gasto del Gobierno

# ### Intuicion: 

# $$ Go↑ → DA↑ → DA > Y → Y↑ $$

# Se entiende de que si hay una aumento del gasto público esto supondrá un aumento de la Demanda Agregada (DA), por la regla keynesiana. Pero entonces la DA será mayor al nivel de equilibrio. Para ser equivalente entonces el equilibrio el nivel de producción tiene que aumentar también. Por lo cual podemos afirmar que el aumento del gasto público genera inevitablemente un crecimiento en el nivel de producción

# ### Matemáticamente: $∆Go > 0  →  ¿∆Y?$

# Sabiendo que: 
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# o, considerando el multiplicador keynesiano, $ k > 0 $:
# 
# $$ Y = k (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# 
# $$ ∆Y = k (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$

# Pero, si no ha habido cambios en $C_0$, $I_0$, $X_0$, $h$ ni $r$, entonces: 
# 
# $$∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$$
# 
# $$ ∆Y = k (∆G_0) $$

# Sabiendo que $∆G_0 > 0 $ y que $k > 0$, la multiplicación de un número psitivo con un positivo dará otro positivo:
# 
# $$ ∆Y = (+)(+) $$
# $$ ∆Y > 0 $$

# ### Gráfico

# In[23]:


#--------------------------------------------------
# Curva de ingreso de equilibrio ORIGINAL

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

    # Ecuación 
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
Go = 100

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[24]:


# líneas punteadas autómaticas

    # definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

    # coordenadas de las curvas (x,y)
A = [DA_IS_K[0], Y[0]] # DA, coordenada inicio
B = [DA_IS_K[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# In[25]:


# coordenadas de las curvas (x,y)
A = [DA_G[0], Y[0]] # DA, coordenada inicio
B = [DA_G[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

# creación de intersección

intersec_G = line_intersection((A, B), (C, D))
intersec_G # (y,x)


# In[29]:


# Gráfico
y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA^e", color = "#7B60BD") #curva ORIGINAL
ax.plot(Y, DA_G, label = "DA_G", color = "#CB5FA6", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axhline(y=intersec[0], xmin= 0, xmax= 0.69, linestyle = ":", color = "grey")
plt.axvline(x=intersec[1], ymin= 0, ymax= 0.69, linestyle = ":", color = "grey")

plt.axhline(y=intersec_G[0], xmin= 0, xmax= 0.82, linestyle = ":", color = "grey")
plt.axvline(x=intersec_G[1], ymin= 0, ymax= 0.82, linestyle = ":", color = "grey")


# Texto agregado
plt.text(0, 135, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 182, '$DA_G$', fontsize = 11.5, color = 'black')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(87, 0, '$Y_G$', fontsize = 12, color = 'black')
plt.text(76, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(20, 165, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Incremento del Gasto del Gobierno $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# ### 5.2 Política Fiscal Expansiva con una reducción de la Tasa de Tributación

# ### Intuición:

# $$ t↓ → Co↑ → DA↑ → DA > Y → Y↑ $$
# $$ t↓ → M↑ → DA↓ → DA < Y → Y↓ $$

# Cuando la tasa de interés disminuye, el consumo fijo aumenta, así produce que la demanda agregada también lo haga el nivel de producción (pues a mayor cantidad de dinero que haya, la gente se motiva a consumir más y por tanto hay mayor oferta y por tanto mayor nivel de producción. 
# No obstante hay otro proceso que ocurre al mismo tiempo, cuando la tasa disminuye, pues hace que las importaciones aumenten, lo que a su vez baja la demanda agregada. Ello hace que sea menor el nivel de producción y baje irremediablemente, lo que a su vez, impacta mucho más en la economía del país en cuestión.

# ### Matemáticamente: $∆t < 0  →  ¿∆Y?$

# In[36]:


Co, Io, Go, Xo, h, r, b, m, t = symbols('Co Io Go Xo h r b m t')

f = (Co + Io + Go + Xo - h*r)/(1-(b-m)*(1-t))


df_t = diff(f, t)
df_t #∆Y/∆t


# Considernado el diferencial de $∆t$:
# 
# $$ \frac{∆Y}{∆t} = \frac{(m-b)(Co + Go + Io + Xo - hr)}{(1-(1-t)(b-m)+1)^2} $$
# 
# - Sabiendo que b > m, entonces $(m-b) < 0$
# - Los componentes del intercepto no cambian: $∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$
# - Cualquier número elevado al cuadrado será positivo: $ (1-(1-t)(b-m)+1)^2 > 0 $
# 
# Entonces:
# 
# $$ \frac{∆Y}{∆t} = \frac{(-)}{(+)} $$
# 
# Dado que $∆t < 0$, la división de dos positivos da otro positivo:
# 
# $$ \frac{∆Y}{(-)} = \frac{(-)}{(+)} $$
# 
# $$ ∆Y = \frac{(-)(-)}{(+)} $$
# 
# $$ ∆Y > 0 $$
# 

# ### Gráfico:

# In[31]:


##--------------------------------------------------
# Curva de ingreso de equilibrio ORIGINAL

    # Parámetros
Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3 #tasa de tributación
r = 0.9

Y = np.arange(Y_size)

    # Ecuación 
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
t = 0.01

# Generar la ecuación con el nuevo parámetros
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_t = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[39]:


# Gráfico
y_max = np.max(DA_IS_K)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_IS_K, label = "DA", color = "#D669AA") #curva ORIGINAL
ax.plot(Y, DA_t, label = "DA_t", color = "#F0A3DA", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
plt.axhline(y = 176, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
plt.axvline(x = 77,  ymin= 0, ymax = 0.75, linestyle = ":", color = "grey")
plt.axhline(y = 192, xmin= 0, xmax = 0.75, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 200, '$DA_t$', fontsize = 11.5, color = 'black')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(80, 0, '$Y_t$', fontsize = 12, color = 'black')
plt.text(72, 45, '$→$', fontsize = 18, color = 'black')
plt.text(20, 180, '$↑$', fontsize = 18, color = 'black')

# Título y leyenda
ax.set(title = "Reducción de la Tasa de Tributación", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# ## 6 Estática Comparativa en el modelo de Ingreso-Gasto Keynesiano con Regla Contracíclica

# ### 6.1 Función DA y Recta de 45°

# Debido a que el parámetro 'g' afecta a la función de demanda agregada, tanto su pendiente como el valor multiplicador, van a cambiar inevitablemente en el modelo de Regla Contracíclica: 
# 
# De: $$ DA = C + I + G + X - M$$
# 

# A:
#    $$DA = (Co + Io + Go + Xo - h*r) + [((b - m)*(1 - t)-g)]*Y$$

# In[41]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
g = 0.2 #solo valores entre 0-0.4
Xo = 2
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_Cont = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[44]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(DA_Cont)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(DA_Cont, label = "DA", color = "#D40000") #Demanda agregada
ax.plot(L_45, color = "#404040") #Línea de 45º

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Líneas punteadas punto de equilibrio
plt.axvline(x=64,  ymin= 0, ymax= 0.63, linestyle = ":", color = "grey")
plt.axhline(y=161, xmin= 0, xmax= 0.63, linestyle = ":", color = "grey")
plt.axhline(y=145, xmin= 0, xmax= 1, linestyle = ":", color = "grey")

# Texto agregado
    # punto de equilibrio
plt.text(0, 165, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(65, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(0, 135, '$α_o$', fontsize = 15, color = 'black')
    # línea 45º
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
    # ecuaciones
plt.text(82, 185, '$DA = α_0 + α_1 Y$', fontsize = 11.5, color = 'black')
plt.text(75, 151, '$α_1 = [(b-m)(1-t)-g]$', fontsize = 11.5, color = 'black')
plt.text(73, 125, '$α_0 = C_o + I_o + G_o + X_o - hr$', fontsize = 11.5, color = 'black')

plt.text(87, 175, '$↓$', fontsize = 13, color = 'black')
plt.text(85, 135, '$↑$', fontsize = 13, color = 'black')

# Título y leyenda
ax.set(title="El ingreso de Equilibrio a Corto Plazo con regla contracíclica", xlabel= r'Y', ylabel= r'DA')
ax.legend() #mostrar leyenda

plt.show()


# ### 6.2 Nivel de Ingreso de Equilibrio

# La ecuación de equilibrio para el Ingreso se deriva de la condición de equilibrio donde el nivel de producción es igual a la Demanda Agregada: $DA = Y$:
# 
# $$ DA = C + I + G + X - M $$
# 
# Donde:
# 
# $$ C = C_0 + bY^d $$
# $$ I = I_0 - hr $$
# $$ G = G_0 - gY $$
# $$ X = X_0 $$
# $$ M = mY^d $$
# 
# $$ Y^d = 1 - t $$
# Entonces sabiendo que la demanda agregada bajo una política fiscal con regla contracíclia es: 
# 
# $$ DA = C_0 + I_0 + G_0 + X_0 - hr + ((b - m)(1 - t)-g)Y $$
# 

# Considerando la condición de equilibrio $Y = DA$, la ecuación del ingreso de equilibrio es:
# 
# $$ Y = C_0 + bY^d + I_0 -hr + G_0 + X_0 - mY^d $$
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)+g} (C_0 + I_0 + G_0 + X_0 - hr) $$

# ### 6.3 Política Fiscal Expansiva con Aumento del Gasto del Gobierno

# ### Intuicion:

# $$ G↑ → (Go - gy) → Go↑ → DA↑ → DA > Y → Y↑ $$

# El 
# 

# ### Matemáticamente: $∆Go > 0  →  ¿∆Y?$

# $$ Y = \frac{1}{1 - (b - m)(1 - t)+g} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# o, considerando el multiplicador keynesiano, $ k > 0 $:
# 
# $$ Y = k (C_0 + I_0 + G_0+ X_0 - hr) $$
# 
# 
# $$ ∆Y = k (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$

# En un contexto de cambios nulos en $C_0$, $I_0$, $X_0$, $h$ ni $r$, entonces: 
# 
# $$∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$$
# 
# $$ ∆Y = k (∆G_0) $$

# Sabiendo que $∆G_0 > 0 $ y que $k > 0$, la multiplicación de un número positivo con un positivo dará otro positivo:
# 
# $$ ∆Y = (+)(+) $$
# $$ ∆Y > 0 $$

# ### Gráficos

# In[46]:


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
g = 0.2

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_Cont = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# Nueva curva

Go = 100

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_C_G = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[47]:


# líneas punteadas autómaticas

    # definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

    # coordenadas de las curvas (x,y)
A = [DA_Cont[0], Y[0]] # DA, coordenada inicio
B = [DA_Cont[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)

    # coordenadas de las curvas (x,y)
A = [DA_C_G[0], Y[0]] # DA, coordenada inicio
B = [DA_C_G[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec_G = line_intersection((A, B), (C, D))
intersec_G # (y,x)


# In[51]:


# Gráfico
y_max = np.max(DA_Cont)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(DA_Cont, label = "DA", color = "#FF2E6D") #curva ORIGINAL
ax.plot(DA_C_G, label = "DA_G", color = "#FFD21F", linestyle = 'dashed') #NUEVA curva
ax.plot(L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axhline(y=intersec[0], xmin= 0, xmax= 0.64, linestyle = ":", color = "grey")
plt.axvline(x=intersec[1], ymin= 0, ymax= 0.64, linestyle = ":", color = "grey")

plt.axhline(y=intersec_G[0], xmin= 0, xmax= 0.76, linestyle = ":", color = "grey")
plt.axvline(x=intersec_G[1], ymin= 0, ymax= 0.76, linestyle = ":", color = "grey")


# Texto agregado
plt.text(0, 135, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 182, '$DA_G$', fontsize = 11.5, color = 'black')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(60, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(72, 0, '$Y_G$', fontsize = 12, color = 'black')
plt.text(70, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(20, 165, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Incremento del Gasto del Gobierno $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# ### 6.4 ¿Cuál es el papel que juega el parámetro 'g' en el multiplicador keynesiano?

# En las ecuaciones mostradas anteriormente podemos ver que aparece el parámetro 'g'. En el primer caso es que hace que la magnitud que la pendiente maneja se reduzca de la función que la demanda agregada ejerce; en contraposición con lo que ocurre en el segundo caso, que es donde funge como un acrecentador, resultando en que finalmente se acorte. 
# 

# ### 6.5 ¿El tamaño del efecto de la política fiscal en el 6 es el mismo que en el apartado 5?

# Lo único que aumenta es el gasto autónomo en ambos casos, lo que provoca que tenga consecuencias diferentes en los apartados: en el primer caso lo que ocurre es que, como se puede apreciar es que las líneas suben, lo que significa que a mayor gasto ocurre una mayor producción. En el segundo caso las líneas bajan, lo que se traduce en una política expansiva anticíclica debido a que hay un menor gasto y por tanto una menor producción según la lógica keynesiana. Cambia esta en tanto cambia la función de la demanda, ya que esta política expansiva fiscal lo que hace es que promueve la producción y en consecuencia el empleo. 

# ### 6.6 Reducción de exportaciones en una crisis mundial

# ### Deducción intuitiva

# $$ Xo↓ → X → DA↓ → DA < Y → Y↓$$

# Como se trata de una ecuación, es que ambas partes de la deducción se perjudican cuando una de ellas disminuye, explicado con estas flechas es que vemos que la cantidad fija de las exportaciones se reduce, lo que a su vez ocasiona que las exportaciones totales del país se reduzcan. Ello lleva a que la semanda agregada disminuya y así se vuelve menor al nivel de producción que el país tiene. Lo que produce que finalmente, la cadena de eventos termine por afectar negativamente también al nivel de producción (Y). 

# ### Matemático: $$ΔXo < 0 → ¿ΔY?$$

# In[53]:


# Diferenciales

    # nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, g = symbols('Co Io Go Xo h r b m t g')

    # determinar ecuación
f = (Co + Io + Go + Xo - h*r)/(1 - (b - m)*(1 - t) + g)

    # función diferencial
df_Xo = diff(f, Xo) # diff(función, variable_analizar
df_Xo #∆Y/∆Go



# Considerdo el diferencial obtenido:
# 
# $$ \frac{∆Y}{∆X_0} = \frac{1}{1 - (b - m)(1 - t) + g} $$
# 
# Sabiendo que al multiplicador keynesiano $(k > 0)$ se le adiciona el parámetro $g$, el denominador continuará siendo positivo (aunque más grande y con una pendiente de curva reducida).
# 
# Y considerando que $∆X_0 < 0 $, la multiplicación de un número positivo con un negativo dará un negativo:
# 
# $$ \frac{∆Y}{(-)} = (+) $$
# $$ ∆Y = (-)(+) $$
# $$ ∆Y < 0 $$

# ### Gráfico:

# In[56]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
g = 0.2 #solo valores entre 0-0.4
Xo = 15
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_Cont = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# Nueva curva

Xo = 1

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_C = (Co  + Io + Go + Xo - h*r) + [(b-m)*(1-t)-g]*Y
    return DA_C

DA_C_X = DA_C(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[57]:


# líneas punteadas autómaticas

    # definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

    # coordenadas de las curvas (x,y)
A = [DA_Cont[0], Y[0]] # DA, coordenada inicio
B = [DA_Cont[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)

    # coordenadas de las curvas (x,y)
A = [DA_C_X[0], Y[0]] # DA, coordenada inicio
B = [DA_C_X[-1], Y[-1]] # DA, coordenada fin

C = [L_45[0], Y[0]] # L_45, coordenada inicio
D = [L_45[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección
intersec_X = line_intersection((A, B), (C, D))
intersec_X # (y,x)


# In[61]:


# Gráfico
y_max = np.max(DA_Cont)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(DA_Cont, label = "DA", color = "#AEA0DB") #curva ORIGINAL
ax.plot(DA_C_X, label = "DA_X", color = "#EE989A", linestyle = 'dashed') #NUEVA curva
ax.plot(L_45, color = "#404040") #línea de 45º

# Lineas punteadas
plt.axhline(y=intersec[0], xmin= 0, xmax= 0.68, linestyle = ":", color = "grey")
plt.axvline(x=intersec[1], ymin= 0, ymax= 0.68, linestyle = ":", color = "grey")

plt.axhline(y=intersec_X[0], xmin= 0, xmax= 0.63, linestyle = ":", color = "grey")
plt.axvline(x=intersec_X[1], ymin= 0, ymax= 0.63, linestyle = ":", color = "grey")


# Texto agregado
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 135, '$DA_X$', fontsize = 11.5, color = 'black')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(71, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(60, 0, '$Y_X$', fontsize = 12, color = 'black')
plt.text(65.5, 45, '$←$', fontsize = 15, color = 'grey')
plt.text(20, 165, '$↓$', fontsize = 15, color = 'grey')

# Título y leyenda
ax.set(title = "Reducción de las Exportaciones", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()

