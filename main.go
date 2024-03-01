package main

import (
	"golang_ml_algorithms/decision_trees"
	"golang_ml_algorithms/random_forest"
)

func main() {
	path := "./data/"
	decision_trees.Trainer(path)
	random_forest.Trainer(path)

}
