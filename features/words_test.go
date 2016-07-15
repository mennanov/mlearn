package features

import (
	"reflect"
	"strings"
	"testing"
)

func TestWordsCounterEncoder_Encode(t *testing.T) {
	encoder := WordsCounterEncoder{Separator: func(s string) []string { return strings.Split(s, " ") }, Sort: true}
	testData := [][]string{{"marry has a dog and a cat", "1"}, {"susan has a dog", "2"}}
	actual, err := encoder.Encode(testData, 0)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{
		Matrix:  []float64{2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1},
		Columns: []string{"a", "and", "cat", "dog", "has", "marry", "susan"}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v", actual, expected)
	}
}
