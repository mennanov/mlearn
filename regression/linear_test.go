package regression

import (
	"github.com/gonum/matrix/mat64"
	"math"
	"reflect"
	"testing"
)

func TestRSS(t *testing.T) {
	for _, tc := range []struct {
		Y           *mat64.Vector
		X           *mat64.Dense
		W           *mat64.Vector
		ExpectedRSS float64
	}{
		{
			Y:           mat64.NewVector(3, []float64{4, 8, 12}),
			X:           mat64.NewDense(3, 2, []float64{1, 1, 2, 2, 3, 3}),
			W:           mat64.NewVector(2, []float64{2, 2}),
			ExpectedRSS: 0,
		},
		{
			Y:           mat64.NewVector(3, []float64{6, 10, 14}),
			X:           mat64.NewDense(3, 2, []float64{1, 1, 2, 2, 3, 3}),
			W:           mat64.NewVector(2, []float64{2, 2}),
			ExpectedRSS: 12,
		},
	} {
		rss := RSS(tc.Y, tc.X, tc.W)
		if rss != tc.ExpectedRSS {
			t.Errorf("Expected RSS: %v, got %v", tc.ExpectedRSS, rss)
		}
	}
}

func TestRMSE(t *testing.T) {
	rmse := RMSE(8, 2)
	if rmse != 2 {
		t.Errorf("Expected RMSE: 2, got %v", rmse)
	}
}

// roundVector rounds the values of the vector to the nearest integers.
func roundVector(v *mat64.Vector) *mat64.Vector {
	for i := 0; i < v.Len(); i++ {
		v.SetVec(i, math.Floor(v.At(i, 0)+.5))
	}
	return v
}

func TestClosedForm(t *testing.T) {
	X := mat64.NewDense(2, 2, []float64{2, 3, 2, 2})
	Y := mat64.NewVector(2, []float64{4, 6})
	expected := mat64.NewVector(2, []float64{5, -2})
	actual := ClosedForm(X, Y)
	if !reflect.DeepEqual(expected.RawVector().Data, roundVector(actual).RawVector().Data) {
		t.Errorf("Expected coefficients: %v, got %v", expected, actual)
	}
}
