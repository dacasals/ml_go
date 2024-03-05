package linear_regression

import (
	"golang_ml_algorithms/utils"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type LinearRegression struct {
	LR          float64
	nIterations int
	weights     *mat.Dense
	bias        float64
}

func (model *LinearRegression) Fit(X [][]float64, y []float64) {
	nSamples := float64(len(X))
	nFeatures := len(X[0])
	XDense := utils.ConvertToMat(X)

	weights := mat.NewDense(nFeatures, 1, make([]float64, nFeatures))
	bias := 0.0

	for i := 0; i < model.nIterations; i++ {
		var dot mat.Dense
		dot.Mul(XDense, weights)

		yPred := utils.AddConstToMat(dot, bias)
		var diff mat.Dense

		diff.Sub(yPred.ColView(0), utils.FromArrayToVec(y))
		var dot2 mat.Dense
		dot2.Mul(XDense.T(), &diff)

		dw := utils.MulMatWithConst(dot2, 1.0/nSamples)
		db := (1.0 / nSamples) * floats.Sum(diff.RawMatrix().Data)

		dwLr := utils.MulMatWithConst(*dw, model.LR)

		weights.Sub(weights, dwLr)
		bias -= db * model.LR
	}
	model.weights = weights
	model.bias = bias
}

func (model *LinearRegression) Predict(X [][]float64) []float64 {

	XMat := utils.ConvertToMat(X)
	var yPred mat.Dense
	yPred.Mul(XMat, model.weights)
	yPredScalar := utils.AddConstToMat(yPred, model.bias)
	return yPredScalar.RawMatrix().Data
}
