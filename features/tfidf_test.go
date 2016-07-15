package features

import (
	"reflect"
	"strings"
	"testing"
)

func TestTFIDFEncoder_Encode(t *testing.T) {
	encoder := TFIDFEncoder{Separator: func(s string) []string { return strings.Split(s, " ") }, Sort: true}
	testData := [][]string{{"at the office", "1"}, {"at the house", "2"}}
	actual, err := encoder.Encode(testData, 0)
	if err != nil {
		t.Error(err)
	}
	expected := PartialMatrix{
		Matrix:  []float64{2. / 3, 0, 1, 2. / 3, 2. / 3, 1, 0, 2. / 3},
		Columns: []string{"at", "house", "office", "the"}}

	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v", actual, expected)
	}
}
