package mlearn

import (
	"github.com/mennanov/mlearn/features"
	"reflect"
	"testing"
)

func TestNewMatrix(t *testing.T) {
	data := [][]string{{"cat", "0.5", "1"}, {"dog", "1.5", "2"}, {"horse", "125", "3"}}
	m, err := NewMatrix(
		data, &features.CategoricalEncoder{}, &features.NumericEncoder{ColumnName: "weight"},
		&features.NumericEncoder{ColumnName: "age"})
	if err != nil {
		t.Error(err)
	}
	// Column names set check.
	columns_actual := make(map[string]int)
	for _, c := range m.ColumnNames {
		columns_actual[c] += 1
	}
	columns_expected := map[string]int{"cat": 1, "dog": 1, "horse": 1, "weight": 1, "age": 1}
	if !reflect.DeepEqual(columns_actual, columns_expected) {
		t.Errorf("Unexpected set of columns in Matrix.ColumnNames: %v, must be %v", columns_actual,
			columns_expected)
	}
	// Underlying matrix values set check.
	data_expected := map[float64]int{1: 4, 0: 6, 0.5: 1, 1.5: 1, 2: 1, 125: 1, 3: 1}
	data_actual := map[float64]int{}
	for _, v := range m.RawMatrix().Data {
		data_actual[v] += 1
	}
	if !reflect.DeepEqual(data_actual, data_expected) {
		t.Errorf("Unexpected set of values in Matrix: %v, must be %v", data_actual, data_expected)
	}

}
