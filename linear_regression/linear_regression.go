package linear_regression

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type LinearRegression struct {
	LR          float64
	nIterations int
	weights     *mat.Dense
	bias        float64
}

func AddConstToMat(a mat.Dense, scalar float64) *mat.Dense {

	aRaw := a.RawMatrix()
	floats.AddConst(scalar, aRaw.Data)

	result := mat.NewDense(aRaw.Rows, aRaw.Cols, aRaw.Data)
	return result
}
func DivConst(alpha float64, x []float64) {
	for i := range x {
		x[i] *= alpha
	}
}
func MulMatWithConst(a mat.Dense, scalar float64) *mat.Dense {

	aRaw := a.RawMatrix()
	DivConst(scalar, aRaw.Data)

	result := mat.NewDense(aRaw.Rows, aRaw.Cols, aRaw.Data)
	return result
}
func FromArrayToVec(array []float64) *mat.VecDense {
	arrayLen := len(array)
	return mat.NewVecDense(arrayLen, array)

}

func (model *LinearRegression) Fit(X [][]float64, y []float64) {
	nSamples := float64(len(X))
	nFeatures := len(X[0])
	XDense := model.convertToMat(X)

	weights := mat.NewDense(nFeatures, 1, make([]float64, nFeatures))
	bias := 0.0

	for i := 0; i < model.nIterations; i++ {
		var dot mat.Dense
		dot.Mul(XDense, weights)

		yPred := AddConstToMat(dot, bias)
		var diff mat.Dense

		diff.Sub(yPred.ColView(0), FromArrayToVec(y))
		var dot2 mat.Dense
		dot2.Mul(XDense.T(), &diff)

		dw := MulMatWithConst(dot2, 1.0/nSamples)
		db := (1.0 / nSamples) * floats.Sum(diff.RawMatrix().Data)

		dwLr := MulMatWithConst(*dw, model.LR)

		weights.Sub(weights, dwLr)
		bias -= db * model.LR
	}
	model.weights = weights
	model.bias = bias
}

func (model *LinearRegression) convertToMat(X [][]float64) *mat.Dense {
	XSq := make([]float64, 0)
	for i := 0; i < len(X); i++ {
		XSq = append(XSq, X[i]...)
	}
	XDense := mat.NewDense(len(X), len(X[0]), XSq)
	return XDense
}

func (model *LinearRegression) Predict(X [][]float64) []float64 {

	XMat := model.convertToMat(X)
	var yPred mat.Dense
	yPred.Mul(XMat, model.weights)
	yPredScalar := AddConstToMat(yPred, model.bias)
	return yPredScalar.RawMatrix().Data
}
