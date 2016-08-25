package features

import "strconv"

type NumericEncoder struct {
	Column     int
	ColumnName string
}

func (e *NumericEncoder) Encode(data [][]string) (PartialMatrix, error) {
	c := len(data)
	p := PartialMatrix{Matrix: make([]float64, c), Columns: []string{e.ColumnName}}
	for i, row := range data {
		v, err := strconv.ParseFloat(row[e.Column], 64)
		if err != nil {
			return PartialMatrix{}, err
		}
		p.Matrix[i] = v
	}
	return p, nil
}

type NumericMultiplicationEncoder struct {
	Columns    []int
	ColumnName string
}

func (e *NumericMultiplicationEncoder) Encode(data [][]string) (PartialMatrix, error) {
	c := len(data)
	p := PartialMatrix{Matrix: make([]float64, c), Columns: []string{e.ColumnName}}
	for i, row := range data {
		r := float64(1)
		for _, c := range e.Columns {
			v, err := strconv.ParseFloat(row[c], 64)
			if err != nil {
				return PartialMatrix{}, err
			}
			r *= v
		}
		p.Matrix[i] = r
	}
	return p, nil
}

type NumericSumEncoder struct {
	Columns    []int
	ColumnName string
}

func (e *NumericSumEncoder) Encode(data [][]string) (PartialMatrix, error) {
	c := len(data)
	p := PartialMatrix{Matrix: make([]float64, c), Columns: []string{e.ColumnName}}
	for i, row := range data {
		r := float64(0)
		for _, c := range e.Columns {
			v, err := strconv.ParseFloat(row[c], 64)
			if err != nil {
				return PartialMatrix{}, err
			}
			r += v
		}
		p.Matrix[i] = r
	}
	return p, nil
}
