package features

import (
	"reflect"
	"testing"
)

func TestCategoricalEncoder_Encode(t *testing.T) {
	encoder := CategoricalEncoder{}
	testData := [][]string{{"cat", "1"}, {"mouse", "2"}}
	actual, err := encoder.Encode(testData, 0)
	if err != nil {
		t.Error(err)
	}
	// Order can not be guaranteed, so we need to test both cases.
	expected1 := PartialMatrix{Matrix: []float64{1, 0, 0, 1}, Columns: []string{"cat", "mouse"}}
	expected2 := PartialMatrix{Matrix: []float64{0, 1, 1, 0}, Columns: []string{"mouse", "cat"}}
	if !reflect.DeepEqual(expected1, actual) && !reflect.DeepEqual(expected2, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v or %v", actual, expected1, expected2)
	}
}
