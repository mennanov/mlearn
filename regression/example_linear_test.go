package regression_test

import (
	"encoding/csv"
	"fmt"
	"github.com/mennanov/mlearn"
	"github.com/mennanov/mlearn/features"
	"github.com/mennanov/mlearn/regression"
	"io"
	"os"
)

// loadCSVFile reads the csv file and extracts its features and target vectors as slices of strings.
func loadCSVFile(file string, targetColumn int, featureColumns []int) ([][]string, []string) {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	featuresMatrix := [][]string{}
	targetsVector := []string{}
	for i := 0; ; i++ {
		row, err := csvReader.Read()
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			break
		}
		// Skip the headers row.
		if i == 0 {
			continue
		}
		// Add a target value for the current row.
		targetsVector = append(targetsVector, row[targetColumn])
		// Add a features vector (a slice of strings) for the current row.
		row_features := make([]string, len(featureColumns))
		for j, c := range featureColumns {
			row_features[j] = row[c]
		}
		featuresMatrix = append(featuresMatrix, row_features)
	}
	return featuresMatrix, targetsVector
}

func ExampleGradientDescent() {
	columnIdx := []int{3, 4, 5, 6, 7, 14, 15, 17, 18, 16}
	targetColumnIdx := 2
	encoders := []features.Encoder{
		&features.NumericMultiplicationEncoder{Columns: []int{0, 0}, ColumnName: "bedrooms_square"},
		&features.NumericMultiplicationEncoder{Columns: []int{0, 1}, ColumnName: "bedrooms_bathrooms"},
		&features.NumericEncoder{Column: 1, ColumnName: "bathrooms"},
		&features.NumericEncoder{Column: 2, ColumnName: "sqft_living"},
		&features.NumericMultiplicationEncoder{Columns: []int{4, 4}, ColumnName: "floors_square"},
		&features.NumericEncoder{Column: 5, ColumnName: "yr_built"},
		&features.NumericEncoder{Column: 6, ColumnName: "yr_renovated"},
		&features.NumericSumEncoder{Columns: []int{7, 8}, ColumnName: "lat_lng"},
	}
	featuresTrainStr, targetsTrainStr := loadCSVFile("../data/kc_house_train_data.csv", targetColumnIdx,
		columnIdx)
	featuresTrain, columns, err := mlearn.NewMatrixFromData(featuresTrainStr, encoders...)
	if err != nil {
		panic(err)
	}
	targetsTrain, err := mlearn.NewVectorFromStringData(targetsTrainStr)
	if err != nil {
		panic(err)
	}
	fmt.Println(columns)
	r, _ := featuresTrain.Dims()
	weights, iterations := regression.GradientDescent(featuresTrain, targetsTrain, 1.1e-12, 5e-2, 1000)
	fmt.Println("Gradient Descend converged after iterations: ", iterations)
	rssTrain := regression.RSS(targetsTrain, featuresTrain, weights)
	rmseTrain := regression.RMSE(rssTrain, r)
	fmt.Println("Train RSS:", rssTrain, "Train RMSE:", rmseTrain)
	// Load the test data set.
	featuresTestStr, targetsTestStr := loadCSVFile("../data/kc_house_test_data.csv", targetColumnIdx,
		columnIdx)
	featuresTest, _, err := mlearn.NewMatrixFromData(featuresTestStr, encoders...)
	if err != nil {
		panic(err)
	}
	targetsTest, err := mlearn.NewVectorFromStringData(targetsTestStr)
	if err != nil {
		panic(err)
	}
	r, _ = featuresTest.Dims()
	rssTest := regression.RSS(targetsTest, featuresTest, weights)
	rmseTest := regression.RMSE(rssTest, r)
	fmt.Println("Test RSS:", rssTest, "Test RMSE:", rmseTest)
	// Output:
	// [intercept bedrooms_square bedrooms_bathrooms bathrooms sqft_living floors_square yr_built yr_renovated lat_lng]
	// Gradient Descend converged after iterations:  364
	// Train RSS: 1.1830968451864368e+15 Train RMSE: 260876.69959623617
	// Test RSS: 2.703172938108974e+14 Test RMSE: 252823.86889048142
}
