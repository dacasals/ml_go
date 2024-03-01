package decision_trees

import (
	"golang_ml_algorithms/utils"
	"math"
	"math/rand"
)

type TreeNode struct {
	Threshold float64
	Value     float64
	Feature   int
	Left      *TreeNode
	Right     *TreeNode
}

func (node TreeNode) IsLeaf() bool {
	return node.Left == nil && node.Right == nil
}

type DecisionTree struct {
	MinSamplesSplit int
	MaxDepth        int
	NumFeatures     int
	Root            TreeNode
}

func NewDecisionTree() DecisionTree {

	return DecisionTree{
		MinSamplesSplit: 2,
		MaxDepth:        100,
		NumFeatures:     -1,
	}
}

func (model *DecisionTree) Fit(X [][]float64, y []float64) {
	//TODO implement me
	numRows := len(X)
	if numRows < 2 {
		panic("X data should have at least 2 rows")
	}
	model.NumFeatures = len(X[0])
	model.Root = model.expandTree(X, y, 0)
}

func (model DecisionTree) Predict(X [][]float64) []float64 {
	var yPred []float64
	for _, x := range X {
		pred := model.traverseTree(x, &model.Root)
		yPred = append(yPred, pred)
	}
	return yPred
}

func (model *DecisionTree) expandTree(X [][]float64, y []float64, depth int) TreeNode {

	n_samples := len(X)
	n_features := 0
	if n_samples > 0 {
		n_features = len(X[0])
	}
	labels, _ := Unique(y)
	n_labels := len(labels)
	// If stop conditions reached return most_common_label
	if depth > model.MaxDepth || n_samples < model.MinSamplesSplit || n_labels == 1 {
		mcl := model.MostCommonLabel(y)
		node := TreeNode{Value: mcl}
		return node
	}

	seqIndexes := utils.Range(0, n_features, 1)
	rand.Shuffle(len(seqIndexes), func(i, j int) { seqIndexes[i], seqIndexes[j] = seqIndexes[j], seqIndexes[i] })

	// Find best feature
	bestFeatureIndex, bestFeatureThreshold := model.bestFeatureIndex(X, y, seqIndexes)

	colValuesBestFeature := getColValues(X, bestFeatureIndex)
	leftIndexes, _, rightIndexes, _ := splitChildren(colValuesBestFeature, bestFeatureThreshold)

	left := model.expandTree(GetRowsByIndexes(X, leftIndexes), getItemsByIndex(y, leftIndexes), depth+1)
	right := model.expandTree(GetRowsByIndexes(X, rightIndexes), getItemsByIndex(y, rightIndexes), depth+1)
	node := TreeNode{Feature: bestFeatureIndex, Left: &left, Right: &right, Threshold: bestFeatureThreshold}

	return node
}

func GetRowsByIndexes(X [][]float64, indexRows []int) [][]float64 {
	var result [][]float64
	for _, index := range indexRows {
		result = append(result, X[index])
	}
	return result
}

func (model *DecisionTree) MostCommonLabel(y []float64) float64 {
	counter := make(map[float64]int)
	max_y := y[0]
	max_y_count := 1
	for i := 1; i < len(y); i++ {
		value := y[i]
		if _, ok := counter[value]; !ok {
			counter[value] = 1
			continue
		}
		counter[value] += 1
		if counter[value] > max_y_count {
			max_y = value
			max_y_count = counter[value]
		}

	}
	return max_y
}

func Unique(list []float64) ([]float64, map[float64]int) {
	uniqueMap := map[float64]int{}
	var unique []float64
	for _, v := range list {
		if _, exist := uniqueMap[v]; !exist {
			uniqueMap[v] = 0
			unique = append(unique, v)
		}
		uniqueMap[v] += 1
	}

	return unique, uniqueMap
}
func getColValues(df [][]float64, col int) []float64 {
	var colValues []float64
	for i := 0; i < len(df); i++ {
		colValues = append(colValues, df[i][col])
	}
	return colValues

}
func (model *DecisionTree) bestFeatureIndex(X [][]float64, y []float64, featureIndexes []int) (int, float64) {
	// Find feature with Max IG
	bestGain := -1.0
	var bestSplitIndex int
	var bestSplitThreshold float64

	for i := 0; i < len(featureIndexes); i++ {
		featureIndex := featureIndexes[i]
		XColumn := getColValues(X, featureIndex)
		thresholds, _ := Unique(XColumn)
		for j := 0; j < len(thresholds); j++ {

			threshold := thresholds[j]
			gain := model.informationGain(y, XColumn, threshold)
			if gain > bestGain {
				bestGain = gain
				bestSplitIndex = featureIndex
				bestSplitThreshold = threshold
			}
		}
	}
	return bestSplitIndex, bestSplitThreshold
}

func splitChildren(values []float64, threshold float64) ([]int, []float64, []int, []float64) {

	var left_indexes []int
	var right_indexes []int
	var left_values []float64
	var right_values []float64

	for i := 0; i < len(values); i++ {
		if values[i] <= threshold {
			left_indexes = append(left_indexes, i)
			left_values = append(left_values, values[i])
			continue
		}
		right_indexes = append(right_indexes, i)
		right_values = append(right_values, values[i])

	}
	return left_indexes, left_values, right_indexes, right_values
}

func getItemsByIndex(list []float64, indexes []int) []float64 {
	var result []float64
	for _, index := range indexes {
		result = append(result, list[index])
	}
	return result
}
func (model *DecisionTree) informationGain(y []float64, colValues []float64, threshold float64) float64 {
	//IG = E[parent] - weighted_avg * E(children)

	parentEntropy := model.entropy(y)
	lenY := float64(len(y))
	leftIndexes, _, rightIndexes, _ := splitChildren(colValues, threshold)

	wLeft := float64(len(leftIndexes)) / lenY
	wRight := float64(len(rightIndexes)) / lenY
	leftYValues := getItemsByIndex(y, leftIndexes)
	rightYValues := getItemsByIndex(y, rightIndexes)

	leftChildEntropy := model.entropy(leftYValues)
	rightChildEntropy := model.entropy(rightYValues)

	childrenEntropy := wLeft*leftChildEntropy + wRight*rightChildEntropy
	IG := parentEntropy - childrenEntropy
	return IG
}

func (model *DecisionTree) entropy(y []float64) float64 {
	//E[x] = - np.sum(p(X_i) * np.log_2(P(x_i))) given that P(X) = #x/n
	n := float64(len(y))
	E_x := 0.0
	unique, counterFreq := Unique(y)
	for i := 0; i < len(unique); i++ {
		freq := counterFreq[unique[i]]
		p_x := float64(freq) / n
		log_p_x := math.Log2(p_x)
		E_x += p_x * log_p_x
	}
	return -E_x
}

func (model *DecisionTree) traverseTree(x []float64, node *TreeNode) float64 {
	if node.IsLeaf() {
		return node.Value
	}

	if x[node.Feature] <= node.Threshold {
		return model.traverseTree(x, node.Left)
	}
	return model.traverseTree(x, node.Right)
}
