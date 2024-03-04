package clustering

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

type KNN struct {
	// A K nearest neighbours classifier
	K      int
	XTrain [][]float64
	YTrain []float64
}

func (model *KNN) Fit(X [][]float64, y []float64) {
	model.XTrain = X
	model.YTrain = y
}

func (model *KNN) predict(x []float64) float64 {
	distances := make([]float64, len(model.XTrain))
	for i, xTrain := range model.XTrain {
		distances[i] = model.euclideanDistance(x, xTrain)
	}
	indexes := make([]int, len(distances))
	floats.Argsort(distances, indexes)

	topK := indexes[:model.K]
	yPred := make([]float64, len(topK))

	for i, index := range topK {
		yPred[i] = model.YTrain[index]
	}
	maxLabel, _ := stat.Mode(yPred, nil)
	return maxLabel
}

func (model *KNN) euclideanDistance(x []float64, xTrain []float64) float64 {
	p := mat.NewDense(1, len(x), x)
	q := mat.NewDense(1, len(xTrain), xTrain)
	result := mat.NewDense(1, len(x), make([]float64, len(x)))
	result.Sub(p, q)
	result.MulElem(result, result)
	resultFloat := math.Sqrt(mat.Sum(result))
	return resultFloat
}

func (model *KNN) Predict(X [][]float64) []float64 {
	// For each prediction x get the k best label given the k nearest points
	results := make([]float64, len(X))
	for i, x := range X {
		results[i] = model.predict(x)
	}
	return results
}
