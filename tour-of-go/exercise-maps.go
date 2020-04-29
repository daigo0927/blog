package main

import (
	"golang.org/x/tour/wc"
	"strings"
)

func WordCount(s string) map[string]int {
	m := make(map[string]int)
	for _, w := range strings.Fields(s) {
		m[w] += 1
	}
	return m
}

func main() {
	wc.Test(WordCount)
}
