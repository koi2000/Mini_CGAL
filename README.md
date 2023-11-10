# Mini-CGAL

由于CGAL的性能问题，需要重写部分CGAL中的数据结构。实现CGAL的子集并尝试迁移到GPU使用。

## 总览

其中重写的方法很多。重载了很多运算符。

## 数据结构

Point

Vector：squared_length

Vertex

facet

halfedge

Polyhedron_3：遍历所有vertex，遍历所有half_edge

## 算法

boundingbox

