from graph_utils import get_graph_by_city, plot_graph

g = get_graph_by_city("Plaza Independencia, Mendoza, Argentina", 3000)

plot_graph(g, "img/plot.jpg")