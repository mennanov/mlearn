package mlearn

import (
	"bytes"
	"github.com/mennanov/mlearn/features"
	"reflect"
	"testing"
)

func TestNewMatrixFromData(t *testing.T) {
	data := [][]string{{"cat", "0.5", "1"}, {"dog", "1.5", "2"}, {"horse", "125", "3"}}
	m, err := NewMatrixFromData(
		data, &features.CategoricalEncoder{}, &features.NumericEncoder{ColumnName: "weight"},
		&features.NumericEncoder{ColumnName: "age"})
	if err != nil {
		t.Error(err)
	}
	// Column names set check.
	columnsActual := make(map[string]int)
	for _, c := range m.ColumnNames {
		columnsActual[c] += 1
	}
	columnsExpected := map[string]int{"cat": 1, "dog": 1, "horse": 1, "weight": 1, "age": 1}
	if !reflect.DeepEqual(columnsActual, columnsExpected) {
		t.Errorf("Unexpected set of columns in Matrix.ColumnNames: %v, must be %v", columnsActual,
			columnsExpected)
	}
	// Underlying matrix values set check.
	dataExpected := map[float64]int{1: 4, 0: 6, 0.5: 1, 1.5: 1, 2: 1, 125: 1, 3: 1}
	dataActual := map[float64]int{}
	rawData := m.RawMatrix().Data
	for _, v := range rawData {
		dataActual[v] += 1
	}
	if !reflect.DeepEqual(dataActual, dataExpected) {
		t.Errorf("Unexpected set of values in Matrix: %v, must be %v", dataActual, dataExpected)
	}
	// Underlying matrix dimensions check.
	if len(rawData) != 15 {
		t.Errorf("Unexpected len of the underlying Dense matrix: %s, expected %s", len(rawData), 15)
	}
}

func TestMatrix_Print(t *testing.T) {
	columns := []string{"x1", "x2"}
	mat := []float64{0, 1, 2, 3}
	matrix := NewMatrix(2, 2, mat, columns)
	buf := new(bytes.Buffer)
	matrix.Print(buf)
	printExpected := "x1\tx2\n0\t1\n2\t3\n"
	printActual := buf.String()
	if printExpected != printActual {
		t.Errorf("Unexpected print output: %t, expected: %t", printActual, printExpected)
	}
}
