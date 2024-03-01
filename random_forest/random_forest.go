package random_forest

import (
	"golang_ml_algorithms/decision_trees"
	"golang_ml_algorithms/utils"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

type RandomForest struct {
	numTrees        int
	MinSamplesSplit int
	MaxDepth        int
	NumFeatures     int
	trees           []decision_trees.DecisionTree
}

func BootstrapData(X [][]float64, y []float64) ([][]float64, []float64) {
	var XShuffled [][]float64
	var YShuffled []float64
	seqIndexes := utils.Range(0, len(X), 1)
	repeatedLen := int(float32(len(seqIndexes)) * 0.2)

	rand.Shuffle(len(seqIndexes), func(i, j int) { seqIndexes[i], seqIndexes[j] = seqIndexes[j], seqIndexes[i] })
	var seqIndexesCopy = make([]int, repeatedLen)
	_ = copy(seqIndexesCopy, seqIndexes)

	allIndex := append(seqIndexes[:len(X)-repeatedLen], seqIndexesCopy...)

	for _, index := range allIndex {
		XShuffled = append(XShuffled, X[index])
		YShuffled = append(YShuffled, y[index])
	}
	return XShuffled, YShuffled
}
func (model *RandomForest) Fit(X [][]float64, y []float64) {

	for i := 0; i < model.numTrees; i++ {
		tree := decision_trees.DecisionTree{
			MinSamplesSplit: model.MinSamplesSplit,
			MaxDepth:        model.MaxDepth,
			NumFeatures:     model.NumFeatures,
		}

		XShuffled, yShuffled := BootstrapData(X, y)

		tree.Fit(XShuffled, yShuffled)

		model.trees = append(model.trees, tree)
	}

}

func (model *RandomForest) Predict(X [][]float64) []float64 {
	var resultsByTree []float64
	for _, tree := range model.trees {
		yPred := tree.Predict(X)

		resultsByTree = append(resultsByTree, yPred...)
	}
	matt := mat.NewDense(model.numTrees, len(X), resultsByTree)
	var results []float64

	for i := 0; i < len(X); i++ {
		var firstModel []float64
		votingTrees := mat.Col(firstModel, i, matt)
		mCLabel := utils.MostCommonLabel(votingTrees)
		results = append(results, mCLabel)
	}

	return results
}
