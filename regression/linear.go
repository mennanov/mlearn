package regression

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

// RSS stands for Residual sum of squares.
func RSS(Y *mat64.Vector, X *mat64.Dense, W *mat64.Vector) float64 {
	n, _ := X.Dims()
	prediction := mat64.NewVector(n, nil)
	prediction.MulVec(X, W)
	loss := mat64.NewVector(n, nil)
	loss.SubVec(Y, prediction)
	rss := mat64.NewVector(1, nil)
	rss.MulVec(loss.T(), loss)
	return rss.At(0, 0)
}

// RMSE is a Root-Mean-Square error or a standard deviation in the case of RSS.
func RMSE(rss float64, n int) float64 {
	return math.Sqrt(rss / float64(n))
}

// ClosedForm computes coefficients using a closed-form solution defined by the equation W=inverse(X'*X)*X'*Y.
// Important: this solution is very inefficient as it requires to compute an inverse of the matrix X'*X which is O(N^3).
// Also, for the inverse of the matrix to exist the number of ROWS must be greater than the number of FEATURES.
func ClosedForm(X *mat64.Dense, Y *mat64.Vector) *mat64.Vector {
	r, c := X.Dims()
	xtx := mat64.NewDense(c, c, nil)
	xtx.Mul(X.T(), X)
	finv := mat64.NewDense(c, c, nil)
	finv.Inverse(xtx)
	f := mat64.NewDense(c, r, nil)
	f.Mul(finv, X.T())
	weights := mat64.NewVector(c, nil)
	weights.MulVec(f, Y)
	return weights
}
