package mlearn

import (
	"errors"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"github.com/mennanov/mlearn/features"
	"io"
	"strconv"
	"strings"
	"text/tabwriter"
)

type Matrix struct {
	*mat64.Dense
	ColumnNames []string
}

func (m *Matrix) Print(writer io.Writer) {
	w := new(tabwriter.Writer)
	w.Init(writer, 0, 8, 1, '\t', 0)
	fmt.Fprintln(w, strings.Join(m.ColumnNames, "\t"))
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		row := make([]string, c)
		for j := 0; j < c; j++ {
			row[j] = strconv.FormatFloat(m.At(i, j), 'g', 4, 64)
		}
		fmt.Fprintln(w, strings.Join(row, "\t"))
	}
	w.Flush()
}

func NewMatrix(data [][]string, encoders ...features.Encoder) (*Matrix, error) {
	n := len(data)
	m := len(data[0])
	if m != len(encoders) {
		return new(Matrix), errors.New("The number of provided encoders does not match the number of columns")
	}
	partial_matrices := make([]features.PartialMatrix, n)
	// Calculate the total number of columns.
	r := 0
	// Build a list of column names.
	columns := make([]string, r)
	for i, encoder := range encoders {
		partial_matrix, err := encoder.Encode(data, i)
		if err != nil {
			return new(Matrix), err
		}
		partial_matrices[i] = partial_matrix
		c := len(partial_matrix.Columns)
		r += c
		columns = append(columns, partial_matrix.Columns...)
	}
	// Prepare the data for mat64.Dense.
	mat := make([]float64, n*r)
	for i := 0; i < n; i++ {
		j := 0
		for _, pm := range partial_matrices {
			cc := len(pm.Columns)
			for k := 0; k < cc; k++ {
				mat[i*r+j+k] = pm.Matrix[i*cc+k]
			}
			j += cc
		}
	}
	return &Matrix{mat64.NewDense(n, r, mat), columns}, nil
}
