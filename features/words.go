package features

import (
	"sort"
)

type WordsCounterEncoder struct {
	Column    int
	Separator func(string) []string
	Sort      bool
}

func (e *WordsCounterEncoder) Encode(data [][]string) (*PartialMatrix, error) {
	uniqueWords := make(map[string]bool)
	n := len(data)
	wordCountsByRow := make([]map[string]int, n)
	// Collect a set of unique words and count words for each row.
	for i, row := range data {
		wordCountsByRow[i] = make(map[string]int)
		for _, word := range e.Separator(row[e.Column]) {
			uniqueWords[word] = true
			// Increment word counter.
			wordCountsByRow[i][word] += 1
		}
	}
	c := len(uniqueWords)
	// Create a list of columns.
	columns := make([]string, c)
	i := 0
	for k := range uniqueWords {
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
				p.Matrix[i*c+j] = float64(v)
			} else {
				p.Matrix[i*c+j] = 0
			}
		}
	}
	return &p, nil
}
