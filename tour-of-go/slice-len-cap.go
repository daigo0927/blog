package main

import "fmt"

func main() {
	s := []int{2, 3, 5, 7, 11, 13}
	printSlice(s)

	// Slice the slice to give it zero length.
	s = s[:3]
	printSlice(s)

	// Extend its length
	s = s[:6]
	printSlice(s)

	// Drop its first two values
	s = s[2:]
	printSlice(s)

	// s = s[:6] // Error
	// printSlice(s)
}

func printSlice(s []int) {
	fmt.Printf("len=%d cap=%d %v\n", len(s), cap(s), s)
}
