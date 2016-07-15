package features

type CategoricalEncoder struct{}

func (e *CategoricalEncoder) Encode(data [][]string, column int) (PartialMatrix, error) {
	unique_values := make(map[string]bool)
	// Collect a set of unique values for the given column.
	for _, row := range data {
		unique_values[row[column]] = true
	}
	c := len(unique_values)
	// Create a list of columns.
	columns := make([]string, c)
	i := 0
	for k := range unique_values {
		columns[i] = k
		i++
	}
	p := PartialMatrix{Matrix: make([]float64, len(data)*c), Columns: columns}
	// Iterate over the data again and fill in the partial matrix.
	for i, row := range data {
		cell := row[column]
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
