import networkx as nx
import matplotlib.pyplot as plt
G2 = nx.read_pajek('data/football.net')
print(nx.info(G2))
print('Is directed graph:',nx.is_directed(G2))
print('path lengtn:',nx.average_shortest_path_length(G2))
nx.draw_networkx(G2)
plt.show()