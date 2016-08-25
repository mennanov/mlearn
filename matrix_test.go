package mlearn

import (
	"bytes"
	"github.com/gonum/matrix/mat64"
	"github.com/mennanov/mlearn/features"
	"reflect"
	"testing"
)

func TestNewMatrixFromData(t *testing.T) {
	data := [][]string{{"cat", "0.5", "1"}, {"dog", "1.5", "2"}, {"horse", "125", "3"}}
	m, columns, err := NewMatrixFromData(data,
		&features.CategoricalEncoder{Sort: true, Map: func(s string) string {
			return s
		}},
		&features.NumericEncoder{ColumnName: "weight"},
		&features.NumericEncoder{ColumnName: "age"})
	if err != nil {
		t.Error(err)
	}
	// Column names set check.
	columnsExpected := []string{"intercept", "cat", "dog", "horse", "weight", "age"}
	if !reflect.DeepEqual(columns, columnsExpected) {
		t.Errorf("Unexpected set of columns in Matrix.ColumnNames: %v, must be %v", columns, columnsExpected)
	}
	// Underlying matrix values set check.
	dataExpected := []float64{1, 1, 0, 0, 0.5, 1, 1, 0, 1, 0, 1.5, 2, 1, 0, 0, 1, 125, 3}
	rawData := m.RawMatrix().Data
	if !reflect.DeepEqual(rawData, dataExpected) {
		t.Errorf("Unexpected set of values in Matrix: %v, must be %v", rawData, dataExpected)
	}
	// Underlying matrix dimensions check.
	if len(rawData) != 18 {
		t.Errorf("Unexpected len of the underlying Dense matrix: %s, expected %s", len(rawData), 18)
	}
}

func TestPrintMatrixTabular(t *testing.T) {
	columns := []string{"x1", "x2"}
	mat := []float64{0, 1, 2, 3}
	matrix := mat64.NewDense(2, 2, mat)
	buf := new(bytes.Buffer)
	PrintMatrixTabular(matrix, columns, buf)
	printExpected := "x1\tx2\n0\t1\n2\t3\n"
	printActual := buf.String()
	if printExpected != printActual {
		t.Errorf("Unexpected print output: %t, expected: %t", printActual, printExpected)
	}
}

func TestNewVectorFromStringData(t *testing.T) {
	expected := mat64.NewVector(3, []float64{1, 2, 3})
	actual, err := NewVectorFromStringData([]string{"1", "2", "3"})
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected mat64.NewVector: %v, expected %v", actual, expected)
	}
}
