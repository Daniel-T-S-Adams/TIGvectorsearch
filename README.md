# Vector Search Challenge

This project provides a framework for implementing and testing different vector search algorithms. The challenge is to find, for each query vector, a corresponding database vector such that the average Euclidean distance between queries and their matches is below a given threshold.

## Problem Definition

Given:
- A set of database vectors (n_database × dimension)
- A set of query vectors (n_queries × dimension)
- All vector components are drawn from uniform[0,1]
- A distance threshold

Task:
Find an assignment of database vectors to query vectors such that the Euclidean distance between each query and its assigned database vector is below the threshold, or determine that no such assignment exists.
