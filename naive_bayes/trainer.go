package naive_bayes

import (
	"fmt"
	"golang_ml_algorithms/utils"
)

func Trainer(path string) {

	XTrainDF := utils.ReadCSV(path + "wine_X_train.csv")
	yTrainDF := utils.ReadCSV(path + "wine_y_train.csv")
	XTestDF := utils.ReadCSV(path + "wine_X_test.csv")
	yTestDF := utils.ReadCSV(path + "wine_y_test.csv")

	XTrain := utils.Matrix{XTrainDF}.Mat2Float()
	yTrain := utils.Matrix{yTrainDF}.Col("0").Float()
	XTest := utils.Matrix{XTestDF}.Mat2Float()
	yTest := utils.Matrix{yTestDF}.Col("0").Float()

	model := GNaiveBayes{}
	model.Fit(XTrain, yTrain)
	predictions := model.Predict(XTest)
	acc := utils.Accuracy(yTest, predictions)

	fmt.Printf(" Naive Bayes Accuracy = %g \n", acc)
}
