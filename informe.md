# Optimización Logística para Distribución de Helados

Código de proyecto: HELA2

---

## Índice
- [Introducción](#introducción)
- [Marco Teórico](#marco-teórico)
  - [Problema del Viajero con Ventanas de Tiempo (TSPTW)](#tsptw)
  - [Ant Colony Optimization (ACO)](#aco)
  - [Sistema de Penalizaciones](#penalizaciones)
  - [Tecnologías Utilizadas](#tech)
- [Análisis del Problema](#analisis)
  - [Modelado y Ponderación de la Red Vial](#modelado)
  - [Restricciones del Dominio](#restricciones)
- [Diseño Experimental e Implementación](#diseño)
  - [Arquitectura del Sistema](#arquitectura)
  - [Baselines](#baselines)
  - [Implementación del Algoritmo](#implementacion)
- [Resultados](#resultados)
- [Conclusiones](#conclusiones)

### Introducción

El presente informe describe una solución computacional a un problema de optimización logística para la distribución de helados de una empresa ficticia en una ciudad cualquiera. 

El problema consiste en encontrar una ruta óptima para un vehículo que parte desde un depósito, visita un conjunto de heladerías y regresa al depósito, minimizando el tiempo total del viaje y cumpliendo con las restricciones del dominio, como las ventanas de tiempo de las heladerías o el límite de carga del camión.

La solución implementa el algoritmo de optimización por colonia de hormigas (ACO) para resolver el Problema del Viajero con Ventanas de Tiempo (TSPTW).


### Marco Teórico

#### Problema del Viajero con Ventanas de Tiempo (TSPTW)

El Problema del Viajero (TSP, por sus siglas en inglés: *Traveling Salesperson Problem*) es uno de los problemas de optimización combinatoria y teoría de grafos más estudiados en las ciencias de la computación. Su objetivo consiste en encontrar la ruta más corta posible que visite un conjunto de ubicaciones exactamente una vez y regrese al punto de origen.

Desde la perspectiva de la teoría de la complejidad, el TSP clásico pertenece a la clase **NP-Hard**. Esto significa que es al menos tan difícil como los problemas más complejos de la clase NP y que, por lo tanto, no se conoce un algoritmo capaz de resolverlo en tiempo polinomial en el peor de los casos. A medida que el número de ubicaciones ($n$) crece, el espacio de soluciones posibles aumenta de forma factorial.

En este proyecto, el TSP convencional se ve modificado por la incorporación de restricciones temporales asociadas a cada nodo, lo que transforma el modelo en el **Problema del Viajero con Ventanas de Tiempo (TSPTW)**. Para modelar formalmente este escenario sobre la red vial real, se calcula la matriz de caminos mínimos entre todos los puntos clave. Esto permite representar el problema mediante un grafo dirigido completo virtual (ya que el grafo inicial no es un grafo completo) definido como $G = (V, A)$, donde:

$V = {0, 1, 2, \dots, n}$ es el conjunto de vértices o nodos. El nodo $0$ representa el depósito central (punto de partida y de retorno de la ruta), mientras que los nodos $1$ hasta $n$ representan las distintas heladerías a visitar.
$A = {(i, j) \mid i, j \in V, i \neq j}$ es el conjunto de aristas que representan las trayectorias óptimas de tránsito entre cualquier par de nodos de interés.
Cada arista $(i, j) \in A$ tiene asociado un costo de tránsito o distancia $d_{ij}$ (calculado mediante el algoritmo de Dijkstra sobre la red real) y un tiempo de viaje estimado $t_{ij}$. Para representar la secuencia de la ruta, se introduce la variable de decisión binaria $x_{ij}$, la cual toma el valor de $1$ si la ruta pasa directamente del nodo $i$ al nodo $j$, y $0$ en caso contrario.

A diferencia del TSPTW clásico, que descarta soluciones que violen los horarios, este proyecto adopta un enfoque de ventanas de tiempo blandas. En este esquema, cada ubicación $i \in V$ está caracterizada por un intervalo temporal preferente de atención $[e_i, l_i]$ y un tiempo de servicio o descarga constante $s$ (equivalente a unload_time en la implementación):

- $e_i$ (Earliest arrival time): Es el tiempo más temprano recomendado para iniciar el servicio en el nodo $i$. Si el vehículo arriba en un tiempo $a_i < e_i$, debe esperar hasta la apertura de la ventana (acumulando tiempo de espera), y se le asigna una penalización proporcional al adelanto.
- $l_i$ (Latest arrival time): Es el tiempo máximo recomendado para el arribo al nodo $i$. Si $a_i > l_i$, el servicio se realiza pero se penaliza la demora.
- $s$ (Service time): El tiempo fijo requerido para descargar el producto en cada heladería antes de continuar el viaje (con $s_0 = 0$ para el depósito).

Para medir la viabilidad temporal del recorrido, se introduce la variable continua $w_i$, que define el inicio del servicio en el nodo $i$. Cuando se transita de $i$ a $j$ ($x_{ij} = 1$), la consistencia del flujo temporal se rige por:

$$w_j \ge \max(a_j, e_j) \quad \text{donde} \quad a_j = w_i + s + t_{ij}$$

Al flexibilizar las restricciones temporales mediante penalizaciones, la función objetivo no busca únicamente minimizar la distancia recorrida, sino equilibrar el trayecto con el cumplimiento de las ventanas, la capacidad del vehículo y la jornada máxima de trabajo. Así, la función objetivo de minimización se formaliza como:

$$\min \left( \lambda \sum_{i \in V} \sum_{j \in V, j \neq i} d_{ij} x_{ij} + \mu \cdot P_{\text{total}} \right)$$

Donde $\lambda$ y $\mu$ son coeficientes de ponderación, y $P_{\text{total}}$ es el valor de penalización acumulado por las desviaciones temporales, excesos de capacidad de carga y superación del tiempo total de operación permitido.

Esta formulación suavizada del TSPTW mantiene el carácter NP-Hard del problema original, pero expande el espacio de búsqueda permitiendo al algoritmo explorar de forma heurística soluciones subóptimas en términos de ventanas a cambio de una reducción sustancial en la distancia física recorrida.


#### Ant Colony Optimization (ACO)

El algoritmo de optimización por colonia de hormigas (ACO) es una metaheurística estocástica inspirada en el comportamiento de las hormigas reales que buscan el camino más corto entre su nido y una fuente de alimento. Este enfoque se enmarca dentro de la categoría de métodos de optimización basados en enjambres (swarm intelligence), diseñados para resolver problemas de optimización combinatoria complejos.

En este desarrollo, el algoritmo se aplica modelando a un conjunto de agentes artificiales (hormigas) que recorren la red vial de forma iterativa para construir soluciones candidatas. La toma de decisiones de cada hormiga en un nodo $i$ para elegir el siguiente destino $j$ se realiza de manera probabilística, balanceando tres componentes esenciales a través de una regla de transición: la concentración del rastro de feromona ($\tau_{ij}$), que representa el aprendizaje histórico del enjambre; la información heurística local ($\eta_{ij} = 1/d_{ij}$), que prioriza los destinos geográficamente más cercanos; y un **factor de urgencia temporal** ($\psi_{ij}$), una adaptación propia orientada a favorecer aquellas heladerías cuyas ventanas de tiempo $[e_j, l_j]$ estén más próximas a expirar. Al finalizar cada iteración, la calidad global de los recorridos construidos determina el depósito de feromona en los caminos más eficientes, guiando la convergencia del algoritmo en las iteraciones subsecuentes.

#### Sistema de Penalizaciones

Para posibilitar la resolución de un problema acotado por múltiples restricciones operativas sin comprometer la exploración de soluciones por parte de la colonia, se implementa un esquema de penalizaciones que modela las restricciones del dominio como condiciones blandas.

La formulación del sistema de penalizaciones responde a la necesidad de transformar un problema de optimización combinatoria multiobjetivo en uno monoobjetivo, permitiendo que la colonia optimice una medida agregada de calidad del recorrido. Para cada hormiga $k$ y cada heladería visitada $j \in V$, se definen tres componentes de penalización: 

- Penalización temporal ($P_{k,j}^T$): Se calcula como la suma del tiempo de espera (cuando el vehículo arriba antes de la ventana, $e_j$) y el exceso sobre la ventana de cierre (cuando el arribo excede $l_j$), ponderados por factores de ineficiencia temporal. Matemáticamente:

$$ P_{k,j}^T = \alpha_1 \cdot \max\left(0, e_j - w_{k,j}^T\right) + \alpha_2 \cdot \max\left(0, w_{k,j}^T - l_j\right) $$

Donde $w_{k,j}^T$ representa el inicio del servicio en el nodo $j$ según la ruta propuesta por la hormiga $k$, y $\alpha_1$ y $\alpha_2$ son coeficientes que escalan la importancia relativa de la espera y la demora, respectivamente.

- Penalización de capacidad ($P_{k,j}^C$): Se aplica cuando la carga solicitada en la heladería $j$ excede la capacidad remanente del vehículo en ese punto del recorrido. Si $C$ es la capacidad total del camión y $q_j$ la demanda de la heladería, la penalización se formula como:

$$ P_{k,j}^C = \beta \cdot \max\left(0, \text{carga actual}_{k,j} + q_j - C\right) $$

Donde $\beta$ es el factor de ponderación por unidad de capacidad excedida.

Debe notarse que, si la carga del camion llega a 0 (es decir, se descargó todo el producto) en un nodo intermedio, la ruta se considera incompleta y se detiene en ese punto. No obstante, la penalizacion se agrega igualmente para castigar el hecho de no haber completado la ruta y demostrar numericamente la inviabilidad de realizar dicho viaje.


- Penalización de jornada laboral ($P_{k,j}^J$): Se genera si el tiempo acumulado en la ruta, incluyendo servicio y desplazamientos, sobrepasa la jornada máxima permitida $J_{\text{max}}$. La penalización es proporcional al exceso, dado por:

$$ P_{k,j}^J = \gamma \cdot \max\left(0, \text{tiempo total}_{k,j} - J_{\text{max}}\right) $$

Con $\gamma$ como el factor de costo por unidad de tiempo de jornada excedida.

El costo total asociado al recorrido de cada hormiga $k$ se obtiene sumando estas contribuciones a lo largo de toda su trayectoria. Así, para una ruta completa $R_k$ de una hormiga, el costo consolidado es:

$$ \text{Costo}(R_k) = \sum_{j \in V} \left( P_{k,j}^T + P_{k,j}^C + P_{k,j}^J \right) $$

Finalmente, el algoritmo busca minimizar este costo agregado en lugar de la distancia pura. Al incorporar las penalizaciones en la función objetivo, el sistema de ACO puede explorar trayectorias que, aunque físicamente más largas, resulten en un menor costo operativo al minimizar las demoras, respetar las ventanas de tiempo y mantener la carga dentro de los límites del vehículo.


#### Tecnologías Utilizadas

Las herramientas y tecnologias utilizadas en el desarrollo de este proyecto son:

- Python 3.14
- FastAPI (interfaz web y API)
- NetworkX (libreria para trabajar con grafos)
- OSMNX (interfaz con OpenStreetMap para obtener grafos viales de ciudades)
- SQLModel (modelado de la base de datos)
- SQLite (base de datos utilizada para cachear ciertos datos)
- Pandas y Matplotlib (analisis y visualizacion de datos)


### Análisis del Problema

#### Modelado y Ponderación de la Red Vial

Habiendo ya el modelo matematico utilizado para diseñar el problema, se puede proceder a describir algunos detalles de la implementacion.

En primer lugar, se tiene que, al ser un grafo ponderado, las aristas tienen un peso asociado que representa la distancia entre dos nodos. Este puede ser expresado de dos formas, elegibles al momento de ejecutar el algoritmo:

- `length`: distancia en metros
- `street_time`: tiempo estimado en segundos

Podemos que decir que `length` es el peso *default* del algoritmo ya que es el obtenido directamente desde OpenStreetMap, sin embargo, en caso de que el usuario desee, se puede utilizar `street_time` como peso del grafo. Este ultimo fue considerado como opcion teniendo en cuenta la velocidad promedio de un camion y la maxima de las calles, intentando simular un escenario lo mas realista posible.

`street_time` se calcula de la siguiente forma:

$$t_{\text{street}} = \frac{l}{\frac{v_{\text{max}}}{3.6}}$$

Donde:

* **$l$** es la longitud física del segmento de calle en metros (el atributo `length` proveniente de OpenStreetMap).

* **$v_{\text{max}}$** es la velocidad máxima de la calle permitida en km/h. Se divide por $3.6$ para convertir el valor a metros por segundo ($m/s$), logrando que el tiempo resultante ($t_{\text{street}}$) se exprese en segundos.

Dado que la red vial real extraída de OpenStreetMap no siempre cuenta con información explícita de límites de velocidad para cada tramo, se implemento la función [`_ensure_street_time`](src/core/graph.py#L22) que, en una primera instancia, intenta obtener el valor del atributo `maxspeed`. Si no existe, se infiere a partir de [`SPEED_DEFAULTS`](src/core/graph.py#L9), que asigna valores arbitrarios en base al tipo de calle.


#### Restricciones del Dominio

