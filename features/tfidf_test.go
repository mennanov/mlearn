package features

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestTFIDFEncoder_Encode(t *testing.T) {
	encoder := TFIDFEncoder{Column: 0, Separator: func(s string) []string {
		return strings.Split(s, " ")
	}, Sort: true}
	testData := [][]string{{"at the office", "1"}, {"at the house", "2"}}
	actual, err := encoder.Encode(testData)
	if err != nil {
		t.Error(err)
	}
	expected := &PartialMatrix{
		Matrix:  []float64{math.Log(2. / 3), 0, 0, math.Log(2. / 3), math.Log(2. / 3), 0, 0, math.Log(2. / 3)},
		Columns: []string{"at", "house", "office", "the"}}

	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Unexpected PartialMatrix: %v, expected %v", actual, expected)
	}
}
