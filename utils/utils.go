package utils

import (
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
)

func Range(start int, max int, step int) []int {
	count := (max - start) / step
	nums := make([]int, count)
	for i := range nums {
		nums[i] = start + i*step
	}
	return nums
}

func ReadCSV(filePath string) dataframe.DataFrame {
	file, err := os.Open(filePath)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}
	df := dataframe.ReadCSV(file)
	return df
}

type Matrix struct {
	dataframe.DataFrame
}

func (m Matrix) At(i, j int) float64 {
	return m.Elem(i, j).Float()
}

func (m Matrix) T() mat.Matrix {
	return mat.Transpose{m}
}

func (m Matrix) Mat2Float() [][]float64 {
	var rows = m.Nrow()
	var floatMat = make([][]float64, rows, rows)
	for i := 0; i < rows; i++ {
		floatMat[i] = mat.Row(nil, i, m)
	}
	return floatMat
}

func Accuracy(y_test []float64, y_pred []float64) float64 {
	n := len(y_test)
	truePreds := 0
	for i := 0; i < n; i++ {
		if y_test[i] == y_pred[i] {
			truePreds += 1
		}
	}
	return float64(truePreds) / float64(n)
}

func MostCommonLabel(list []float64) float64 {
	counter := make(map[float64]int)
	max_y := list[0]
	max_y_count := 1
	for i := 1; i < len(list); i++ {
		value := list[i]
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

func FromArrayToVec(array []float64) *mat.VecDense {
	arrayLen := len(array)
	return mat.NewVecDense(arrayLen, array)

}

func ConvertToMat(X [][]float64) *mat.Dense {
	XSq := make([]float64, 0)
	for i := 0; i < len(X); i++ {
		XSq = append(XSq, X[i]...)
	}
	XDense := mat.NewDense(len(X), len(X[0]), XSq)
	return XDense
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

func MSE(yTest, yPred []float64) float64 {
	var diff mat.Dense
	lenY := float64(len(yPred))
	diff.Sub(FromArrayToVec(yTest), FromArrayToVec(yPred))
	diff.MulElem(&diff, &diff)
	f := floats.Sum(diff.RawMatrix().Data) / lenY
	return f
}
