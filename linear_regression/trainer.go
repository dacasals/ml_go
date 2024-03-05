package linear_regression

import (
	"fmt"
	"golang_ml_algorithms/utils"
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
	mse := utils.MSE(yTest, predictions)
	fmt.Printf("LinearRegression MSE = %g \n", mse)
}
