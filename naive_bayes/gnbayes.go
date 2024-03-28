package naive_bayes

import (
	"golang_ml_algorithms/utils"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

type GNaiveBayes struct {
	meanVars map[float64][][]float64
	classes  map[float64]int
	priors   map[float64]float64
}

func (model *GNaiveBayes) getMeanVar(X mat.Matrix) [][]float64 {
	_, lenFeatures := X.Dims()
	var results [][]float64
	for i := 0; i < lenFeatures; i++ {
		var col []float64
		col = mat.Col(col, i, X)

		mean, variance := stat.MeanStdDev(col, nil)
		results = append(results, []float64{mean, variance})

	}
	return results

}

func (model *GNaiveBayes) getPosteriorProb(x []float64, meanVar [][]float64) []float64 {
	lenFeatures := len(x)
	var results []float64
	for i := 0; i < lenFeatures; i++ {
		colVal := x[i]

		mean := meanVar[i][0]
		variance := meanVar[i][1]

		PDF := (1 / math.Sqrt(2*math.Pi*variance)) * math.Exp(
			-math.Pow(colVal-mean, 2)/(2*math.Pow(variance, 2)))
		results = append(results, PDF)
	}
	return results
}
func (model *GNaiveBayes) Fit(X [][]float64, y []float64) {
	model.classes = make(map[float64]int)
	model.priors = make(map[float64]float64)
	model.meanVars = make(map[float64][][]float64)

	var Xclasses = make(map[float64][][]float64)
	lenX := len(X)
	for i := 0; i < lenX; i++ {
		index, ok := model.classes[y[i]]
		// If the key exists
		if !ok {
			index = len(model.classes)
			model.classes[y[i]] = index
		}
		Xclasses[y[i]] = append(Xclasses[y[i]], X[i])
	}
	matXClasses := make(map[float64]*mat.Dense)
	for class := range model.classes {
		matXClasses[class] = utils.ConvertToMat(Xclasses[class])
		_, lenClass := matXClasses[class].Dims()
		model.priors[class] = float64(lenClass) / float64(lenX)
		model.meanVars[class] = model.getMeanVar(matXClasses[class])
	}
}

func (model *GNaiveBayes) predict(x []float64) float64 {
	posteriors := make(map[float64][]float64)
	cums := make(map[float64]float64)
	var predClass float64
	maxClassCum := 0.0
	for class := range model.classes {
		posteriors[class] = model.getPosteriorProb(x, model.meanVars[class])
		cums[class] = model.priors[class]
		for i := 0; i < len(x); i++ {
			cums[class] = cums[class] * posteriors[class][i]
		}
		if cums[class] > maxClassCum {
			predClass = class
			maxClassCum = cums[class]
		}
	}
	return predClass
}

func (model *GNaiveBayes) Predict(X [][]float64) []float64 {
	// For each prediction x get the k best label given the k nearest points
	results := make([]float64, len(X))
	for i, x := range X {
		results[i] = model.predict(x)
	}
	return results
}
