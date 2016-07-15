package features

import (
	"reflect"
	"testing"
)

func TestNumericEncoder_Encode(t *testing.T) {
	encoder := NumericEncoder{ColumnName: "x1"}
	testData := [][]string{{"-1", "0", "1.5"}, {"3", "-4", "5e4"}}
	actual, err := encoder.Encode(testData, 2)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{Matrix: []float64{1.5, 5e4}, Columns: []string{"x1"}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v", actual, expected)
	}
}
