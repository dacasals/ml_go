package clustering

import (
	"fmt"
	"golang_ml_algorithms/utils"
)

func KNNTrainer(path string) {

	XTrainDF := utils.ReadCSV(path + "iris_X_train.csv")
	yTrainDF := utils.ReadCSV(path + "iris_y_train.csv")
	XTestDF := utils.ReadCSV(path + "iris_X_test.csv")
	yTestDF := utils.ReadCSV(path + "iris_y_test.csv")

	XTrain := utils.Matrix{XTrainDF}.Mat2Float()
	yTrain := utils.Matrix{yTrainDF}.Col("0").Float()
	XTest := utils.Matrix{XTestDF}.Mat2Float()
	yTest := utils.Matrix{yTestDF}.Col("0").Float()

	model := KNN{
		K: 5,
	}
	model.Fit(XTrain, yTrain)
	predictions := model.Predict(XTest)
	acc := utils.Accuracy(yTest, predictions)

	fmt.Printf(" KNN Accuracy = %g \n", acc)
}
