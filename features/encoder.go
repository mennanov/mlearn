package features

type PartialMatrix struct {
	Matrix  []float64
	Columns []string
}

type Encoder interface {
	Encode(data [][]string) (PartialMatrix, error)
}
