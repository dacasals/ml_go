package linear_regression

import (
	"fmt"
	"golang_ml_algorithms/utils"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func Trainer(path string) {

	XTrainDF := utils.ReadCSV(path + "X_train.csv")
	yTrainDF := utils.ReadCSV(path + "y_train.csv")
	XTestDF := utils.ReadCSV(path + "X_test.csv")
	yTestDF := utils.ReadCSV(path + "y_test.csv")

	XTrain := utils.Matrix{XTrainDF}.Mat2Float()
	yTrain := utils.Matrix{yTrainDF}.Col("0").Float()
	XTest := utils.Matrix{XTestDF}.Mat2Float()
	yTest := utils.Matrix{yTestDF}.Col("0").Float()

	model := LinearRegression{
		LR: 0.003, nIterations: 100000,
	}
	model.Fit(XTrain, yTrain)
	predictions := model.Predict(XTest)
	mse := MSE(yTest, predictions)
	fmt.Printf("LinearRegression MSE = %g \n", mse)

}

func MSE(yTest, yPred []float64) float64 {
	var diff mat.Dense
	lenY := float64(len(yPred))
	diff.Sub(FromArrayToVec(yTest), FromArrayToVec(yPred))
	diff.MulElem(&diff, &diff)
	f := floats.Sum(diff.RawMatrix().Data) / lenY
	return f
}
