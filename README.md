# Parallelized-Dijkstra-s-in-C
Parallel Implementation of Dijkstra's in C

Dijkstra’s algorithm solves “the single-source shortest path problem” in a weighted, di- rected graph. 
The input consists of a weighted, directed graph and a specified vertex in the graph. A directed graph or digraph 
is a collection of vertices and directed edges, which join one vertex to another. A weighted digraph has numerical 
weights assigned to the vertices and/or the edges.

The input graph for Dijkstra’s algorithm has nonnegative weights assigned to the edges, and the algorithm finds a shortest 
path from the specified vertex, the source vertex, to every other vertex in the graph. The algorithm uses four main data 
structures:

1. The adjacency matrix mat of the graph: mat[v][w] is the cost or weight of the edge from vertex v to vertex w. If there is no edge joining the vertices, the cost or weight is ∞.
2. A list dist: dist[v] is the current estimate of the cost of the least cost path from the source to v.
3. A list pred: pred[v] is the “predecessor” of vertex v on the current path 0->v.
4. A set known consisting of those vertices for which the shortest path from the source to
the vertex is known.

Sample Input:

0 1 2 3

4 0 5 6

7 8 0 9

8 7 6 0

Sample Output:
The distance from 0 to each vertex is:

 v    dist 0 -> v
***   ***********

 1        1

 2        2

 3        3
 
The shortest path from 0 to each vertex is:

 v   Path 0 -> v
***  ************

  1     0  1

  2     0  2

  3     0  3
