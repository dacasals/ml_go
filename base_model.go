package main

type BaseModel interface {
	Fit(X [][]float64, y []float64)

	Predict(X [][]float64) []float64
}
