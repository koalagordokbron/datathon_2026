Voltia

Componentes del equipo
- Pablo Morcillo Cuenca
- Adrían Moreno López
- Raúl Moreno Serrano

Todo el código está disponible en el siguiente repositorio de GitHub: https://github.com/koalagordokbron/datathon_2026

Índice
1. Contexto del mercado eléctrico
2. Motivación del proyecto
3. Estructura del proyecto
4. Datos utilizados
5. Resutlados obtenidos
6. Mejoras

------------------------------------------------------------------------
1. CONTEXTO DEL MERCADO ELÉCTRICO
------------------------------------------------------------------------
1.1. ¿QUÉ ES EL POOL ENERGÉTICO (MERCADO MAYORISTA)?
El "Pool" es el mercado diario de electricidad gestionado por el Operador 
del Mercado Ibérico de Energía (OMIE). Funciona bajo un sistema de subasta 
marginalista:
- Oferta y Demanda: Los productores de energía (nucleares, renovables, gas, etc.) 
  ofertan su energía para cada hora del día siguiente, y las comercializadoras 
  ofertan su compra.
- Casación: Se ordenan las ofertas de venta de menor a mayor precio y las de 
  compra de mayor a menor. El punto donde se cruzan fija el precio para todas 
  las transacciones de esa hora.
- Importancia: Este precio marginal es la base sobre la que se calculan las 
  facturas de millones de consumidores y la rentabilidad de las plantas generadoras.

1.2. ¿QUÉ ES EL PVPC?
El PVPC (Precio Voluntario para el Pequeño Consumidor) es la tarifa regulada 
de electricidad en España, disponible para potencias contratadas menores a 10 kW.
- Cálculo: Históricamente estaba indexada 100% al precio horario del Pool. 
  Desde 2024, incluye un factor de corrección basado en mercados de futuros 
  para reducir la volatilidad, aunque el precio del Pool sigue siendo su 
  componente principal.
- Variabilidad: A diferencia de las tarifas de mercado libre (precio fijo), 
  el PVPC cambia cada hora y cada día.

------------------------------------------------------------------------
2. MOTIVACIÓN DEL PROYECTO
------------------------------------------------------------------------
El mercado eléctrico se ha vuelto extremadamente volátil debido a factores 
geopolíticos, el coste del gas y los derechos de emisión de CO2. Los motivos 
principales para desarrollar este modelo son:

A. Optimización Económica: Permitir a los usuarios y empresas anticipar los 
   picos y valles de precios para desplazar el consumo a horas más baratas.
B. Gestión Inteligente de Energía: Integración en sistemas Smart Grid o 
   baterías domésticas para decidir cuándo cargar/descargar automáticamente.
C. Reto Técnico: Aplicar Deep Learning (Redes Neuronales Recurrentes) a 
   series temporales complejas con alta estacionalidad y dependencia de 
   variables exógenas.

------------------------------------------------------------------------
3. ESTRUCTURA DEL PROYECTO
------------------------------------------------------------------------
Vamos a explicar la estructura del proyecto:

A. data: en esta carpeta guardamos tanto los datos crudos (raw), como los
   los que hemos procesado ya (processed).
B. src: guardamos tanto el procesamiento de datos (LimpiezaDeDatosPVPCPrediction.ipynb)
   y el modelo LSTM que hemos creado.
C. artifacts: aquí se encuentran el modelo y los escaladores que utilizamos
   para procesar los datos.
D. El notebook principal (ModeloLSTMPCPCPytorch.ipynb): en este notebook
   creamos y entrenamos una instancia del modelo, además de mostrar los
   resultados que hemos obtenido.

------------------------------------------------------------------------
4. DATOS UTILIZADOS
------------------------------------------------------------------------
Utilizamos los datos divididos por horas de todo 2024 en la Península Íberica,
obtenidos de fuentes como ESIOS.

Utilizamos datos como:
- Velocidad del viento a 100 metros de altitud.
- Radiación.
- Temperatura.
- Sensación térmica.
- Humedad.

------------------------------------------------------------------------
5. RESULTADOS OBTENIDOS
------------------------------------------------------------------------
Como podéis observar en el notebook principal, hacemos una predicción del PVPC
a 8 días (200 horas) con un error de 10.98 €/MWh. Esto quiere decir que solo nos desviamos
1.09 céntimos de euro del precio del KWh.

Actualmente las herramientas públicas solo publican el precio oficial para el siguiente día,
o solo te dicen si la semana será "cara" o "barata".

------------------------------------------------------------------------
6. MEJORAS POTENCIALES PARA EL MODELO LSTM
------------------------------------------------------------------------
Para aumentar la precisión del modelo y reducir el error, se proponen las 
siguientes mejoras:

6.1. INCLUSIÓN DE VARIABLES EXÓGENAS (MULTIVARIANTE)
   - Materias Primas: Incluir el precio del gas (TTF) y derechos de CO2 (ETS), 
     que suelen marcar el precio en las horas más caras.
   - Interconexiones: Flujo de energía con Francia y Portugal.

6.2. MEJORAS EN LA ARQUITECTURA DE LA RED
   - Mecanismos de Atención: Implementar capas de "Attention" para que el modelo 
     sepa ponderar qué instantes pasados son más relevantes para la predicción actual.

6.3. OPTIMIZACIÓN DE HIPERPARÁMETROS
   - Búsqueda sistemática (Grid Search o Optuna) para encontrar el tamaño óptimo 
     de ventana (window size), número de neuronas, learning rate y batch size.

4.4. ENFOQUE HÍBRIDO
   - Combinar el modelo LSTM con CNN (Redes Convolucionales) para extraer 
     patrones locales antes de procesar la secuencia temporal.




Sabemos que a este proyecto le queda mucho por hacer, ya sea mejorar la calidad de los
datos o mejorar la estructura del modelo, pero creemos que podemos crear una herramienta
muy potente, al alcance de todos y que puede ayudar a mucha gente.

VOLTIA