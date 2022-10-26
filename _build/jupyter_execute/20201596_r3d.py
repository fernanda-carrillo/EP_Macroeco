#!/usr/bin/env python
# coding: utf-8

# In[1]:


# María Fernanda Carrillo (20201596)


# ## Parte I: Reporte

# Lo que fundamentalmente busca este texto es indagar sobre el impacto que tuvo la cuarentena en la economía, el nivel de producción y de precios, basándose en una reproducción que dividía a la economía en dos sectores: el primero, directamente afectado por el cese de actividades y el segundo que siguió operando, mas recibió los efectos adversos de la demanda disminuida del otro sector. Asimismo, estudia el progreso que tiene la economía para restituirse nuevamente, tras este proceso transitorio que fue la cuarentena, evaluándolo en trimestres. 
# 
# Una de las fortalezas de este texto es que se presenta un modelo económico bien construido y justificado, ya que explica enteramente el proceso de creación de los modelos utilizados, así como la incorporación o modificación de ciertas variables dentro de las ecuaciones utilizadas. Ya que su justificación no solo es matemática, sino que también es intuitiva al ser explicada cómo unos factores afectan sucesivamente a otros en el texto. Asimismo, los gráficos son bastante ilustrativos de este proceso, ya que crean una imagen mental que, debido a la alta cantidad de variables, sería imposible de imaginar por sí misma, o como mínimo confundiría. 
# 
# No obstante, es que debido al uso continuo de estas fórmulas es que el artículo se vuelve bastante complejo, ya que, si bien los argumentos que esgrime son bastante convincentes y entendibles, cuando se introducen las variables por sí mismas y las fórmulas es que, para un lector inexperto o uno que lea no tan en profundidad, se pierde dentro de estas. Y debido a que esto es recurrente, es que aquel lector comprenderá el texto más o menos en un 50%. 
# 
# Asimismo, es que el autor menciona que dentro de su modelo hay alternativas de construcción. Ya que, por un lado, es que cabe destacar la respuesta de la política estatal en el campo también económico, no solo sanitario, de una manera mucho más realista; tomando a las medidas de alivio fiscal como una deducción de la tasa impositiva. De la misma manera es que se pueden conceptualizar a los agentes como racionales, así generando una dinámica diferente en la transición al estado estacionario. Por último es que se ha ignorado las consecuencias que tiene la cuarentena sobre el producto potencial, debido a que el modelo propuesto por el autor se centraba más en la demanda que en otros aspectos.  
# 
# La contribución que dio al campo en general es que da la importancia respectiva a la política crediticia a la que adjudica el éxito de la reactivación de la economía y la recuperación del PBI potencial. Así trayendo atención sobre este tema al cual las autoridades respectivas habrían de tomar en cuenta una mayor atención. Sin embargo, al reconocer también las falencias propias de su argumentación, explicados en el párrafo anterior, es que deja una ventana entreabierta a que tanto otros autores como ‘policymakers’ indaguen sobre estos y puedan estudiar su respectivo impacto. 
# 
# De la misma manera es que en el artículo de “COVID-19 y shock externo” de Jaramillo y Ñopo, se conviene en el papel que tiene la política crediticia, la cual ayuda a evitar un quiebre en la cadena de pagos, así como en el despido masivo a sus trabajadores a fin de abaratar costos. Ya que lo que esa decisión provocaría sería eliminar completamente el ingreso de estos trabajadores, así haciendo imperfecto el modelo, y más realistamente, cortando completamente la demanda sobre los bienes de ambos sectores expresados en el artículo leído previamente, ocasionando un mayor déficit y por tanto una caída mayor de la economía y un mayor tiempo de recuperación, o al menos uno más prolongado; decisión que definitivamente el estado opta por evitar lo mayor posible. 
# 
# Bibliografía: 
# JARAMILLO, Miguel y ÑOPO, Hugo 
# 2020	COVID-19 y shock externo: impactos económicos y opciones de política en el
# Perú. Lima: GRADE, 2020. Consulta: 18 de setiembre de 2022. 
# http://repositorio.grade.org.pe/bitstream/handle/20.500.12820/579/GRADEdi107-1.pdf?sequence=1&isAllowed=y

# ## Parte II: Código

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


# ## Equilibrio en el mercado de Dinero: Explique y Derive

# ### 1. Instrumentos de política monetaria que puede utilizar el Banco Central

# Entendiendo que el Banco Central es la autoridad competente de controlar y gestionar la oferta monetaria, tiene la posibilidad de hacer uso de políticas monetarias expansivas (con el aumento de oferta se reduce la tasa de interés) o contractivas (reducción de la demanda reduce el nivel de producción y empleo)  mediante el usos de instrumentos de política monetaria.

# En primer lugar, existe la oferta monetaria como instrumento de política. Mediante este tipo de políticas el Banco central realiza operaciones mediante la compra o venta de activos financieron o bonos a los bancos comericales (Jiménez 2020, pp. 80). Cuando compran bonos del mercado para inyectarlos en la economía se le llama una política monetaria expansivas. Mediante esta política aumenta la oferta monetaria gracias al incremento de sus activos. Cuando venden bonos al mercado y retiran dinero de la economía se le llama una política monetaria contractiva. Mediante esta política se reduce la oferta monetaria debido a la disminución de la base monetaria. 

# En segundo lugar, el coeficiente legal de encaje es también un instrumento de política. A través de una política monetaria expansiva, se disminuye la tasa de encaje al incrementar el dinero disponibles de los bancos para realizar préstamos. Es decir, "disminución de la tasa de encaje aumenta la 
# posibilidad de creación de dinero bancario porque aumenta el multiplicador de dinero bancario; y, este aumento del dinero bancario implica un aumento de la oferta monetaria" (Jiménez 2020, pp. 82). Lo contrario sucede con una política monetaria contractiva en el cuál aumenta el coeficiente de encaje y disminuye la oferta monetaria ya que los bancos aumenta su proporción de depósitos reduciendo así el multiplicador bancario.

# Y finalmente, está la tasa de interés (r) como instrumento de política. En este caso, el r se convierte en instrumento y la oferta monetaria se convierte en una variable endógena, es decir, la r se vuelve la variable referencial. Con una política monetaria expansiva se reduce la tasa de interés y por ende aumentado la oferta monetaria. Lo contrario sucede con una política monetaria contractiva ya que aumenta la tasa de interés y por ende la oferta monetaria. 

# ### 2. Derive la oferta real de dinero: Ms = Mo/P

# La oferta de dinero (M^s) representa cuando dinero hay de verdad en la economía. Se la considera una variable exógena e instrumento de la política monetaria. Asimismo, vamosa a ver a la oferta dividida entre el nivel general de precios de la economía (P)

# $$ \frac{M^s}{P} = \frac{M^s_0}{P} $$

# ### 3. Demanda de Dinero: L1 + L2 - Md = kY - ji

# Para poder formular y entender la demanda de dinero se deben analizar los motivos por el cuál se demanda dinero: motivos de transacción, precaución y el método especulativo.

# En el primer bloque se encuenta el motivo de transacción y de precaución. En el primer motivo se demanda dinero para poder realizar transacciones y se entiende que esta transacción está en una relación directa con el producto de la economía. Consecuentemente, a partir de este motivo, la demanda de dinero depende positivmente del Ingreso (Y). Con el segundo motivo, se pide dinero como manera de precaución para pagar las deudas. En este sentido, la demanda de dinero dependerá positivamente del ingreso. Bajo estos dos supuestos se entiende la elasticidad del total de los ingresos de la economía (k): mientras más elastacidad, mayor cantidad de dinero que se demanda en cuanto se incrementa el PBI. 

# $$ L_1 = kY $$

# En el siguiente bloque, se entiende a partir del motivo de especulación que la demanda de dinero depende inversamente de las tasas de interés de los bonos. Las personas tienen dos opciones respecto a como manejar su dinero: tenerlo ellos mismos o en el banco. Los individuos preferirán tener su propio dinero cuando la tasa de interés (i) disminuye (= cantidad de dinero sube), pero cuando la tasa de interés aumenta, preferieren depositarlo en el banco (=cantidad de dinero baja). En este sentido existe una elesticidad de sustitución del dinero en el bolsillo propio de los individuos versus depositado en el banco (j)

# $$ L_2 = -ji $$

# Si asumismos que ambas demandas están en términos reales, la función de la demanda de dinero sería:
# 
# $$ M^d = L_1 + L_2 $$ 
# 
# $$ M^d = kY -ji $$

# Recapitulando, los parametros "k" y "j" indican respectivamente la sensibilidad de la demanda de dinero ante variaciones del ingreso “Y”, e indica cuán sensible es la demanda de  dinero ante las variaciones de la tasa de interés nominal de los bonos, “i”.

# ### 4. Ecuación de equilibrio en el mercado de dinero

# Se deriva apartir de la eucación de la Oferta de Dinero (M^s) y Demanda de Dinero (M^d) en equilibrio:
# 
# $$ M^s = M^d $$
# 
# $$ \frac{M^s}{P} = kY - ji $$

# ¿Por qué la i? Suponiendo que la inflación esperada es cero, no habrá mucha diferencia entre la tasa de interés nominal (i) y la real (r). Por estar zazón, se puede reemplazar de la siguiente manera: 
# 
# $$ \frac{M^s}{P} = kY - jr $$
# 
# $$ M_0 = P(kY -jr) $$

# ### 5. Grafique Equilibrio en el mercado

# In[3]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[7]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(-3, 8, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## Estática Comparativa en el mercado de Dinero

# ### 1.Explique y grafique si ∆Y < 0

# De acuerdo a Jiménez, cuanto el ingreso de la producción aumenta, la demanda monetaria también aumenta (2020, pp. 86). Entonces en el caso contrario, cuando el ingreso de la producción disminuye, la demanda monetaria también disminuye y por ende la tasa de interés también tiene que disminuir para regresar el equilibrio. 

# In[5]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 36
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[6]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y_1 = 20
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[8]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Reducción de la producción en el mercado monetario", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')
#ax1.plot(MD_1, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=0, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.5, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50, -5, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 8, "$E_0$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=8, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 0, "$r_1$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$E_1$", fontsize = 12, color = 'black')


ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()


# ### 2.Explique y grafique si ∆k <0

# Cuando la sensibilidad/elasticidad del ingreso disminuye se requiere de una disminución de la tasa de interés para que el mercado se equilibre. Mientras menos elastacidad, menor cantidad de dinero que se demanda en cuanto disminuye el PBI. 

# In[9]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_00 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS1 = MS_0 / P
MS1


# In[10]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.2
j = 0.2                
P  = 10 
Y_1 = 20
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_01 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS1 = MS_0 / P
MS1


# In[11]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_00, label= '$L_0$', color = '#CD5C5C')
#ax1.plot(MD_01, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=0, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 8, "$r_0$", fontsize = 12, color = 'black')
ax1.text(57, -10, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(51, 8, "$E_0$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 0, "$r_1$", fontsize = 12, color = 'black')
ax1.text(51, -6, "$E_1$", fontsize = 12, color = 'black')


ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()


# ### 3.Explique y grafique si ∆Ms < 0

# Una disminución en la oferta hace que en el mercado se tenga que reducir la tasa de interés y así aumentar la demanda y restablecer el equilibrio en el mercado. Esto va generar a que la recta de la oferta de dinero se deplace a la izquierda. 

# In[12]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[13]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P_1  = 20 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P_1, r, Y)
# Necesitamos crear la oferta de dinero.
MS_1 = MS_0 / P_1
MS


# In[22]:


# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=8, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.5, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 7.5, "$E_0$", fontsize = 12, color = 'black')


# Nuevas curvas a partir del cambio en el nivel del producto
#ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS_1,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=13, xmin= 0, xmax= 0.28, linestyle = ":", color = "black")
ax1.text(0, 12.5, "$r_1$", fontsize = 12, color = 'black')
ax1.text(25, 0, "$(Ms/P)_1$", fontsize = 12, color = 'black')
ax1.text(25, 12.5, "$E_1$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## Curva LM

# ### 1.Paso a paso LM matemáticamente (a partir del equilibrio en el Mercado Monetario) y grafique

# Siendo la ecuación de equilibrio en el Mercado de Dinero:

# $$  \frac{M^s_0}{P} = {kY - jr} $$

# $$  \frac{M^s_0}{P} = M^s = M^d$$

# $$ \frac{M^s_0}{p}=jr $$
# 
# $$ kY -\frac{M^s_0}{P} = jr $$
# 
# $$ \frac{kY}{j} - \frac{M^s_0}{Pj} = r $$
# 
# $$ r = - \frac{M^s_0}{Pj} + \frac{kY}{j} $$ 
# 
# $$ r = - \frac{1}{j}\frac{M^s_0}{Pj} + \frac{k}{j}Y $$ 
# 

# In[23]:


#1----------------------Equilibrio mercado monetario

    # Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35

r = np.arange(r_size)


    # Ecuación
def Ms_MD(k, j, P, r, Y):
    Ms_MD = P*(k*Y - j*r)
    return Ms_MD

Ms_MD = Ms_MD(k, j, P, r, Y)


    # Nuevos valores de Y
Y1 = 45

def Ms_MD_Y1(k, j, P, r, Y1):
    Ms_MD = P*(k*Y1 - j*r)
    return Ms_MD

Ms_Y1 = Ms_MD_Y1(k, j, P, r, Y1)


Y2 = 25

def Ms_MD_Y2(k, j, P, r, Y2):
    Ms_MD = P*(k*Y2 - j*r)
    return Ms_MD

Ms_Y2 = Ms_MD_Y2(k, j, P, r, Y2)

#2----------------------Curva LM

    # Parameters
Y_size = 100

k = 0.5
j = 0.2                
P  = 10               
Ms = 30            

Y = np.arange(Y_size)


# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[24]:


# Gráfico de la derivación de la curva LM a partir del equilibrio en el mercado monetario

    # Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8)) 


#---------------------------------
    # Gráfico 1: Equilibrio en el mercado de dinero
    
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')
ax1.plot(Y, Ms_MD, label= '$L_0$', color = '#CD5C5C')
ax1.plot(Y, Ms_Y1, label= '$L_1$', color = '#CD5C5C')
ax1.plot(Y, Ms_Y2, label= '$L_2$', color = '#CD5C5C')
ax1.axvline(x = 45,  ymin= 0, ymax= 1, color = "grey")

ax1.axhline(y=35, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=135, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=85, xmin= 0, xmax= 1, linestyle = ":", color = "black")

ax1.text(47, 139, "C", fontsize = 12, color = 'black')
ax1.text(47, 89, "B", fontsize = 12, color = 'black')
ax1.text(47, 39, "A", fontsize = 12, color = 'black')

ax1.text(0, 139, "$r_2$", fontsize = 12, color = 'black')
ax1.text(0, 89, "$r_1$", fontsize = 12, color = 'black')
ax1.text(0, 39, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()
 

#---------------------------------
    # Gráfico 2: Curva LM
    
ax2.set(title="LM SCHEDULE", xlabel=r'Y', ylabel=r'r')
ax2.plot(Y, i, label="LM", color = '#3D59AB')

ax2.axhline(y=160, xmin= 0, xmax= 0.69, linestyle = ":", color = "black")
ax2.axhline(y=118, xmin= 0, xmax= 0.53, linestyle = ":", color = "black")
ax2.axhline(y=76, xmin= 0, xmax= 0.38, linestyle = ":", color = "black")

ax2.text(67, 164, "C", fontsize = 12, color = 'black')
ax2.text(51, 122, "B", fontsize = 12, color = 'black')
ax2.text(35, 80, "A", fontsize = 12, color = 'black')

ax2.text(0, 164, "$r_2$", fontsize = 12, color = 'black')
ax2.text(0, 122, "$r_1$", fontsize = 12, color = 'black')
ax2.text(0, 80, "$r_0$", fontsize = 12, color = 'black')

ax2.text(72.5, -14, "$Y_2$", fontsize = 12, color = 'black')
ax2.text(56, -14, "$Y_1$", fontsize = 12, color = 'black')
ax2.text(39, -14, "$Y_0$", fontsize = 12, color = 'black')

ax2.axvline(x=70,  ymin= 0, ymax= 0.69, linestyle = ":", color = "black")
ax2.axvline(x=53,  ymin= 0, ymax= 0.53, linestyle = ":", color = "black")
ax2.axvline(x=36,  ymin= 0, ymax= 0.38, linestyle = ":", color = "black")

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.legend()

plt.show()


# ### 2.¿Cuál es el efecto de una disminución en la Masa Monetaria ∆Ms < 0? 

# $$ M^s_0↓ → \frac{M^s_0}{P_0}↓ → r↓ → M^d > M^s → r↑ $$
# 

# In[25]:


#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 500             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
Ms = 150

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[26]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = 'black')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3D59AB', linestyle = 'dashed')

# Texto agregado
plt.text(45, 76, '∆$M^s$', fontsize=12, color='black')
plt.text(45, 70, '←', fontsize=15, color='grey')

# Título y leyenda
ax.set(title = "Disminución en la Masa Monetaria $(M^s)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()


# ### 3.¿Cuál es el efecto de un aumento  ∆k>0?

# $$ k↑ → \frac{k}{j}↑ → M^d > M^s →  r↑ $$

# In[27]:


#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 500             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
k = 6

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[37]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = 'black')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3D59AB', linestyle = 'dashed')

# Texto agregado
plt.text(45, 130, '∆$M^s$', fontsize=12, color='black')
plt.text(43, 130, '↑', fontsize=15, color='grey')

# Título y leyenda
ax.set(title = "Aumento en la sensibilidad del ingreso $(k)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()

