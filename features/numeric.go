package features

import "strconv"

type NumericEncoder struct {
	ColumnName string
}

func (e *NumericEncoder) Encode(data [][]string, column int) (PartialMatrix, error) {
	c := len(data)
	p := PartialMatrix{Matrix: make([]float64, c), Columns: []string{e.ColumnName}}
	for i, row := range data {
		v, err := strconv.ParseFloat(row[column], 64)
		if err != nil {
			return PartialMatrix{}, err
		}
		p.Matrix[i] = v
	}
	return p, nil
}
