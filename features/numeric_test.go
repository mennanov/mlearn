package features

import (
	"reflect"
	"testing"
)

func TestNumericEncoder_Encode(t *testing.T) {
	encoder := NumericEncoder{Column: 2, ColumnName: "x1"}
	testData := [][]string{{"-1", "0", "1.5"}, {"3", "-4", "5e4"}}
	actual, err := encoder.Encode(testData)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{Matrix: []float64{1.5, 5e4}, Columns: []string{"x1"}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v", actual, expected)
	}
}

func TestNumericMultiplicationEncoder_Encode(t *testing.T) {
	encoder := NumericMultiplicationEncoder{Columns: []int{0, 1}, ColumnName: "x1"}
	testData := [][]string{{"-1", "1", "1.5"}, {"3", "-4", "5e4"}}
	actual, err := encoder.Encode(testData)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{Matrix: []float64{-1, -12}, Columns: []string{"x1"}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v", actual, expected)
	}
}

func TestNumericSumEncoder_Encode(t *testing.T) {
	encoder := NumericSumEncoder{Columns: []int{0, 1}, ColumnName: "x1"}
	testData := [][]string{{"-1", "1", "1.5"}, {"3", "-4", "5e4"}}
	actual, err := encoder.Encode(testData)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{Matrix: []float64{0, -1}, Columns: []string{"x1"}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v", actual, expected)
	}
}
