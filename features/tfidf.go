package features

import (
	"math"
	"sort"
)

type TFIDFEncoder struct {
	Separator func(string) []string
	Sort      bool
}

func (e *TFIDFEncoder) Encode(data [][]string, column int) (PartialMatrix, error) {
	wordFreq := make(map[string]int)
	n := len(data)
	wordCountsByRow := make([]map[string]int, n)
	// Collect a map of word frequencies in all documents and word counts for each row.
	for i, row := range data {
		wordCountsByRow[i] = make(map[string]int)
		for _, word := range e.Separator(row[column]) {
			wordFreq[word] += 1
			// Increment word counter.
			wordCountsByRow[i][word] += 1
		}
	}
	c := len(wordFreq)
	// Create a list of columns.
	columns := make([]string, c)
	i := 0
	for k := range wordFreq {
		columns[i] = k
		i++
	}
	if e.Sort {
		sort.Strings(columns)
	}
	p := PartialMatrix{Matrix: make([]float64, len(data)*c), Columns: columns}
	// Fill in the partial matrix with word counts.
	for i, wordCounts := range wordCountsByRow {
		for j, w := range columns {
			// Get a count value for each word in a row.
			if v, ok := wordCounts[w]; ok {
				// Compute TF-IDF
				p.Matrix[i*c+j] = float64(v) * math.Log(float64(n)/float64(1+wordFreq[w]))
			} else {
				p.Matrix[i*c+j] = 0
			}
		}
	}
	return p, nil
}
