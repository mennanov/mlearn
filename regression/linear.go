package regression

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

// Predict the value using given weights vector W.
// Returns the vector of predictions.
func Predict(X *mat64.Dense, W *mat64.Vector) *mat64.Vector {
	n, _ := X.Dims()
	prediction := mat64.NewVector(n, nil)
	prediction.MulVec(X, W)
	return prediction
}

// RSS stands for Residual sum of squares.
func RSS(Y *mat64.Vector, X *mat64.Dense, W *mat64.Vector) float64 {
	prediction := Predict(X, W)
	loss := predictionLoss(Y, prediction)
	rss := mat64.NewVector(1, nil)
	// Compute a square of loss. It will be a scalar number (or a vector of 1x1).
	rss.MulVec(loss.T(), loss)
	return rss.At(0, 0)
}

// predictionLoss computes the difference between predicted and actual values.
// Returns a vector of differences.
func predictionLoss(Y, prediction *mat64.Vector) *mat64.Vector {
	n, _ := Y.Dims()
	loss := mat64.NewVector(n, nil)
	loss.SubVec(Y, prediction)
	return loss
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

// GradientDescent algorithm for the linear regression computes the parameters of the function by taking a gradient
// (partial derivatives) of the RSS and going down the slope iteratively.
// Iteration stops when the maxIterations is exceeded or when the change in the gradient is less than epsilon.
func GradientDescent(X *mat64.Dense, Y *mat64.Vector, step, tolerance float64, maxIterations int) (*mat64.Vector, int) {
	_, c := X.Dims()
	W := mat64.NewVector(c, nil)
	t := 0
	for ; t <= maxIterations; t++ {
		gradient := RSSGradient(Y, X, W, step)
		// Update the coefficients by subtracting the gradient.
		W.SubVec(W, gradient)
		// If gradient has changed just a little - stop.
		if magnitude(gradient) < tolerance {
			break
		}
	}
	return W, t
}

// magnitude computes the magnitude of the vector v as an SQRT(v' * v).
func magnitude(v *mat64.Vector) float64 {
	m := mat64.NewVector(1, nil)
	m.MulVec(v.T(), v)
	return math.Sqrt(m.At(0, 0))
}

// RSSGradient computes a gradient of the RSS function.
// Gradient is just a vector of partial derivatives with respect to j-th feature W.
func RSSGradient(Y *mat64.Vector, X *mat64.Dense, W *mat64.Vector, step float64) *mat64.Vector {
	_, c := X.Dims()
	gradient := mat64.NewVector(c, nil)
	prediction := Predict(X, W)
	loss := predictionLoss(Y, prediction)
	// Compute partial derivatives with respect to j-th feature.
	// To do that we treat all other non j-th features as constants.
	for j := 0; j < c; j++ {
		// j-th column of the matrix X.
		jX := X.ColView(j)
		// partial is a 1x1 matrix to hold the result of jX'*loss.
		partial := mat64.NewDense(1, 1, nil)
		partial.Mul(jX.T(), loss)
		// The final formula for partial looks like this: -2 * (jX' * loss).
		// So we multiply the partial by -2 and a step coefficient to simplify calculations in GradientDescent.
		p := -2 * step * partial.At(0, 0)
		gradient.SetVec(j, p)
	}
	return gradient
}
