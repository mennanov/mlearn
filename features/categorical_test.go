package features

import (
	"reflect"
	"strings"
	"testing"
)

func TestCategoricalEncoder_Encode(t *testing.T) {
	encoder := CategoricalEncoder{Sort: true, Map: func(s string) string { return s }}
	testData := [][]string{{"cat", "1"}, {"mouse", "2"}}
	actual, err := encoder.Encode(testData, 0)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{Matrix: []float64{1, 0, 0, 1}, Columns: []string{"cat", "mouse"}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v or %v", actual, expected)
	}
}

func TestCategoricalEncoder_EncodeNonUniqueValues(t *testing.T) {
	encoder := CategoricalEncoder{Sort: true, Map: func(s string) string { return s }}
	testData := [][]string{{"cat", "1"}, {"cat", "2"}, {"cat", "3"}, {"dog", "5"}}
	actual, err := encoder.Encode(testData, 0)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{Matrix: []float64{1, 0, 1, 0, 1, 0, 0, 1}, Columns: []string{"cat", "dog"}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v or %v", actual, expected)
	}
}

func TestCategoricalEncoder_EncodeWithMapToUpper(t *testing.T) {
	encoder := CategoricalEncoder{Sort: true, Map: func(s string) string { return strings.ToUpper(s) }}
	testData := [][]string{{"cat", "1"}, {"mouse", "2"}, {"dog", "3"}}
	actual, err := encoder.Encode(testData, 0)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{
		Matrix: []float64{1, 0, 0, 0, 0, 1, 0, 1, 0},
		Columns: []string{"CAT", "DOG", "MOUSE"}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v or %v", actual, expected)
	}
}
