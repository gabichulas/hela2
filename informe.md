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
  - [Descripción de la Red de Distribución](#descripcion)
  - [Restricciones del Dominio](#restricciones)
- [Diseño Experimental e Implementación](#diseño)
  - [Arquitectura del Sistema](#arquitectura)
  - [Modelado de Datos](#datos)
  - [Implementación del Algoritmo](#implementacion)
- [Pruebas y Resultados](#pruebas-y-resultados)
- [Conclusiones](#conclusiones)
- [Bibliografía](#bibliografía)


### Introducción

El presente informe describe una solución computacional a un problema de optimización logística para la distribución de helados en una ciudad. 

El problema consiste en encontrar una ruta óptima para un vehículo que parte desde un depósito, visita un conjunto de heladerías y regresa al depósito, cumpliendo con ventanas de tiempo en cada heladería y minimizando el tiempo total de viaje. 

La solución implementa el algoritmo de optimización por colonia de hormigas (ACO) para resolver el Problema del Viajero con Ventanas de Tiempo (TSPTW).


