package random_forest

import (
	"fmt"
	"golang_ml_algorithms/utils"
)

func Trainer() {
	path := "/home/yoyi/source/python/ml_from_scratch/"
	XTrainDF := utils.ReadCSV(path + "x_train.csv")
	yTrainDF := utils.ReadCSV(path + "y_train.csv")
	XTestDF := utils.ReadCSV(path + "x_test.csv")
	yTestDF := utils.ReadCSV(path + "y_test.csv")

	XTrain := utils.Matrix{XTrainDF}.Mat2Float()
	yTrain := utils.Matrix{yTrainDF}.Col("0").Float()
	XTest := utils.Matrix{XTestDF}.Mat2Float()
	yTest := utils.Matrix{yTestDF}.Col("0").Float()

	model := RandomForest{
		numTrees:        10,
		MinSamplesSplit: 2,
		MaxDepth:        100,
		NumFeatures:     len(XTrain[0]),
	}
	model.Fit(XTrain, yTrain)
	predictions := model.Predict(XTest)
	acc := utils.Accuracy(yTest, predictions)
	fmt.Printf(" RandomForest Accuracy = %g \n", acc)
}
