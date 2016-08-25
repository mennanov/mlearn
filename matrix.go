package mlearn

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"github.com/mennanov/mlearn/features"
	"io"
	"strconv"
	"strings"
	"text/tabwriter"
)

// PrintMatrixTabular produces a nicely formatted table-like matrix representation.
func PrintMatrixTabular(m *mat64.Dense, columns []string, writer io.Writer) {
	w := new(tabwriter.Writer)
	w.Init(writer, 0, 8, 1, '\t', 0)
	fmt.Fprintln(w, strings.Join(columns, "\t"))
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

func NewMatrixFromData(data [][]string, encoders ...features.Encoder) (*mat64.Dense, []string, error) {
	n := len(data)
	m := len(encoders)
	partialMatrices := make([]features.PartialMatrix, m)
	// Calculate the total number of columns.
	r := 1
	// Build a list of column names. Intercept is included by default.
	columns := []string{"intercept"}
	for i, encoder := range encoders {
		partialMatrix, err := encoder.Encode(data)
		if err != nil {
			return new(mat64.Dense), []string{}, err
		}
		partialMatrices[i] = partialMatrix
		r += len(partialMatrix.Columns)
		columns = append(columns, partialMatrix.Columns...)
	}
	// Prepare the data for mat64.Dense.
	mat := make([]float64, n*r)
	for i := 0; i < n; i++ {
		// Fill in the intercept value.
		mat[i*r] = 0
		j := 0
		for _, pm := range partialMatrices {
			cc := len(pm.Columns)
			for k := 0; k < cc; k++ {
				mat[i*r+j+k+1] = pm.Matrix[i*cc+k]
			}
			j += cc
		}
	}
	return mat64.NewDense(n, r, mat), columns, nil
}

// NewVectorFromStringData creates a mat64.Vector from an array of strings.
func NewVectorFromStringData(data []string) (*mat64.Vector, error) {
	float64Data := make([]float64, len(data))
	for i, v := range data {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return &mat64.Vector{}, err
		}
		float64Data[i] = f
	}
	return mat64.NewVector(len(data), float64Data), nil
}
