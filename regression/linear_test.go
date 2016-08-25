package regression_test

import (
	"github.com/gonum/matrix/mat64"
	"github.com/mennanov/mlearn/regression"
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
		rss := regression.RSS(tc.Y, tc.X, tc.W)
		if rss != tc.ExpectedRSS {
			t.Errorf("Expected RSS: %v, got %v", tc.ExpectedRSS, rss)
		}
	}
}

func TestRMSE(t *testing.T) {
	rmse := regression.RMSE(8, 2)
	if rmse != 2 {
		t.Errorf("Expected RMSE: 2, got %v", rmse)
	}
}

// RoundVector rounds the values of the vector to the nearest integers.
func RoundVector(v *mat64.Vector) *mat64.Vector {
	for i := 0; i < v.Len(); i++ {
		v.SetVec(i, math.Floor(v.At(i, 0)+.5))
	}
	return v
}

func TestClosedForm(t *testing.T) {
	X := mat64.NewDense(2, 2, []float64{2, 3, 2, 2})
	Y := mat64.NewVector(2, []float64{4, 6})
	expected := mat64.NewVector(2, []float64{5, -2})
	actual := regression.ClosedForm(X, Y)
	if !reflect.DeepEqual(expected.RawVector().Data, RoundVector(actual).RawVector().Data) {
		t.Errorf("Expected coefficients: %v, got %v", expected, actual)
	}
}

func TestRSSGradient(t *testing.T) {
	X := mat64.NewDense(2, 3, []float64{2, 3, 2, 2, 2, 2})
	Y := mat64.NewVector(2, []float64{4, 6})
	W := mat64.NewVector(3, []float64{1, 1, 1})
	expected := mat64.NewVector(3, []float64{6, 9, 6})
	actual := regression.RSSGradient(Y, X, W, 0.5)
	if !reflect.DeepEqual(expected.RawVector().Data, RoundVector(actual).RawVector().Data) {
		t.Errorf("Expected gradient: %v, got %v", expected, actual)
	}
}

func TestRSSGradient_SquareMatrix(t *testing.T) {
	X := mat64.NewDense(2, 2, []float64{2, 3, 2, 2})
	Y := mat64.NewVector(2, []float64{4, 6})
	W := mat64.NewVector(2, []float64{1, 1})
	expected := mat64.NewVector(2, []float64{-2, -1})
	actual := regression.RSSGradient(Y, X, W, 0.5)
	if !reflect.DeepEqual(expected.RawVector().Data, RoundVector(actual).RawVector().Data) {
		t.Errorf("Expected gradient: %v, got %v", expected, actual)
	}
}

func TestGradientDescent(t *testing.T) {
	X := mat64.NewDense(2, 2, []float64{2, 3, 2, 2})
	Y := mat64.NewVector(2, []float64{4, 6})
	actual, _ := regression.GradientDescent(X, Y, 4.5e-2, 2e-4, 1000)
	expected := mat64.NewVector(2, []float64{5, -2})
	if !reflect.DeepEqual(expected.RawVector().Data, RoundVector(actual).RawVector().Data) {
		t.Errorf("Expected coefficients: %v, got %v", expected, actual)
	}
}
