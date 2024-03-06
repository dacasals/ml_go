package var_metrics

import (
	"fmt"
	"golang_ml_algorithms/utils"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
)

type MutualInformation struct {
}

func (metric *MutualInformation) findUnique(uniqueA map[string]int, A []string) {
	index := 0
	for _, val := range A {
		_, err := uniqueA[val]
		if !err {
			uniqueA[val] = index
			index += 1
		}
	}
}
func (metric *MutualInformation) miDiscrete(A, B []string) float64 {
	uniqueA := make(map[string]int)
	uniqueB := make(map[string]int)

	metric.findUnique(uniqueA, A)
	metric.findUnique(uniqueB, B)
	if len(uniqueA) == 1 || len(uniqueB) == 1 {
		return 0.0
	}
	lenRows := len(A)
	countersMatrix := mat.NewDense(len(uniqueA), len(uniqueB), nil)
	fmt.Printf("%g", countersMatrix.RawMatrix().Data)
	for i := 0; i < lenRows; i++ {
		valA := A[i]
		valB := B[i]
		indexA := uniqueA[valA]
		indexB := uniqueB[valB]
		countersMatrix.Set(indexA, indexB, countersMatrix.At(indexA, indexB)+1)
	}
	joinProb := mat.NewDense(len(uniqueA), len(uniqueB), nil)
	for _, index := range uniqueA {
		joinProb.Set(0, index, floats.Sum(countersMatrix.RawRowView(index)))
	}
	for _, col := range uniqueB {
		colsVal := make([]float64, countersMatrix.RawMatrix().Cols)
		squeezedMat := countersMatrix.RawMatrix().Data
		// For each col index sum all row values
		for i := col; i < len(squeezedMat); i += len(uniqueB) {
			colsVal = append(colsVal, squeezedMat[i])
		}
		joinProb.Set(1, col, floats.Sum(colsVal))
	}
	countersMatrix = utils.MulMatWithConst(*countersMatrix, 1.0/float64(lenRows))
	joinProb = utils.MulMatWithConst(*joinProb, 1.0/float64(lenRows))
	accum := 0.0
	for _, indexA := range uniqueA {
		for _, indexB := range uniqueB {
			pX := joinProb.At(0, indexA)
			pY := joinProb.At(1, indexB)
			pXY := countersMatrix.At(indexA, indexB)
			if pXY != 0.0 {
				accum += pXY * math.Log(pXY/(pX*pY))
			}
		}
	}
	return accum
}
