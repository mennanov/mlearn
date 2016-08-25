package features

import (
	"sort"
)

type CategoricalEncoder struct {
	Column int
	Sort   bool
	Map    func(s string) string
}

func (e *CategoricalEncoder) Encode(data [][]string) (PartialMatrix, error) {
	uniqueValues := make(map[string]bool)
	// Collect a set of unique values for the given column.
	for _, row := range data {
		uniqueValues[e.Map(row[e.Column])] = true
	}
	c := len(uniqueValues)
	// Create a list of columns.
	columns := make([]string, c)
	i := 0
	for k := range uniqueValues {
		columns[i] = k
		i++
	}
	if e.Sort {
		sort.Strings(columns)
	}
	p := PartialMatrix{Matrix: make([]float64, len(data)*c), Columns: columns}
	// Iterate over the data again and fill in the partial matrix.
	for i, row := range data {
		cell := e.Map(row[e.Column])
		for j, v := range columns {
			index := i*c + j
			if v == cell {
				p.Matrix[index] = 1
			} else {
				p.Matrix[index] = 0
			}
		}
	}
	return p, nil
}
